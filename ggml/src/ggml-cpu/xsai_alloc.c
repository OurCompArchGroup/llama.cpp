/*
 * xsai_alloc.c - Custom memory allocator for llama.cpp on XSAI RISC-V.
 *
 * Uses a simple free-list with first-fit allocation and block coalescing.
 * All allocations are padded to 64-byte alignment (suitable for AME / tensor
 * operations).
 *
 * Thread-safe: a pthread_mutex_t serialises all alloc/free operations.
 *
 * Build-time defines:
 *   RESERVED_MEMORY_SIZE      - total pool size in bytes (default 1 GiB)
 *   RESERVED_PHYS_BASE_ADDR   - if defined, mmap /dev/mem at this physical
 *                               address instead of anonymous mmap
 */

#include "xsai_alloc.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#ifndef RESERVED_MEMORY_SIZE
#define RESERVED_MEMORY_SIZE (1024L * 1024L * 1024L) /* 1 GiB */
#warning "RESERVED_MEMORY_SIZE not defined, using 1 GiB"
#endif

/* ------------------------------------------------------------------ */
/* Internal data structures                                             */
/* ------------------------------------------------------------------ */

#define ALIGN 64

/* Each free block has an intrusive header at its start. */
typedef struct free_block {
    size_t size;               /* usable bytes following this header */
    struct free_block *next;
    char _pad[ALIGN - sizeof(size_t) - sizeof(void *)];
} free_block_t;

/* Each allocated block has a small header just before the returned pointer. */
typedef struct alloc_header {
    size_t size;               /* usable bytes following this header */
    char _pad[ALIGN - sizeof(size_t)];
} alloc_header_t;

#define ALLOC_HDR_SIZE  ((size_t)sizeof(alloc_header_t))  /* == ALIGN */
#define FREE_HDR_SIZE   ((size_t)sizeof(free_block_t))    /* == ALIGN */

static void          *pool_base  = NULL;
static size_t         pool_size  = 0;
static free_block_t  *free_list  = NULL;   /* sorted by address */
/* 1 when the pool is backed by physically-contiguous memory (huge pages or
 * RESERVED_PHYS_BASE_ADDR).  The AMU may then use "single-TLB-base + PA
 * offset" addressing: PA = TLB(pool_base) + (ptr - pool_base). */
static int            pool_phys_contiguous = 0;

static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;

/* Diagnostics: number of active allocations, total bytes allocated. */
static size_t xsai_alloc_count       = 0;  /* currently live allocations */
static size_t xsai_alloc_bytes       = 0;  /* currently live bytes (incl. headers) */
static size_t xsai_alloc_total_count = 0;  /* total alloc calls ever */
static size_t xsai_alloc_peak_bytes  = 0;  /* high-water mark of live bytes */

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

/* Must be called with pool_lock held. */
static int pool_init(void) {
    pool_size = (size_t)RESERVED_MEMORY_SIZE;

#ifdef RESERVED_PHYS_BASE_ADDR
    /* Map a pre-reserved physically-contiguous region via /dev/mem.
     * The entire pool is one contiguous PA range, so pool_phys_contiguous=1.
     * The AMU can do: PA = RESERVED_PHYS_BASE_ADDR + (ptr - pool_base). */
    fprintf(stderr, "[xsai_alloc] ACTIVE mode=phys  phys_base=0x%lx  size=%zu MiB\n",
            (unsigned long)(RESERVED_PHYS_BASE_ADDR), pool_size >> 20);
    int fd = open("/dev/mem", O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "[xsai_alloc] open /dev/mem failed (need root?)\n");
        return -1;
    }
    pool_base = mmap(NULL, pool_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, (off_t)(RESERVED_PHYS_BASE_ADDR));
    close(fd);
    if (pool_base != MAP_FAILED)
        pool_phys_contiguous = 1;
#else
    /* Try to obtain a physically-contiguous pool via huge pages.
     *
     * Huge-page PTEs cover the whole page in a single TLB entry, so every
     * allocation within a single huge page is also physically contiguous by
     * definition.  This is the software contract required for the AMU's
     * "single-TLB-base + PA-offset" addressing mode:
     *   PA_row = TLB(base_VA) + row * stride
     *
     * Prerequisite (RISC-V Linux):
     *   echo N > /proc/sys/vm/nr_hugepages   (N >= pool_size / 2MiB)
     * or reserve at boot with hugepages=N.
     */
    pool_base = MAP_FAILED;

#if defined(__linux__) && defined(MAP_HUGETLB)
    /* Try default huge-page size (typically 2 MiB on RISC-V Sv39). */
    pool_base = mmap(NULL, pool_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
                     -1, 0);
    if (pool_base != MAP_FAILED) {
        pool_phys_contiguous = 1;
        fprintf(stderr, "[xsai_alloc] ACTIVE mode=anon-hugepage  size=%zu MiB"
                        "  pool_base=%p  (phys-contiguous)\n",
                pool_size >> 20, pool_base);
    }
#endif /* __linux__ && MAP_HUGETLB */

    if (pool_base == MAP_FAILED) {
        /* Hugepages unavailable: fall back to regular anonymous pages.
         * Physical contiguity is NOT guaranteed across 4 KiB boundaries.
         * The AMU must NOT use single-TLB-base mode in this configuration. */
        fprintf(stderr, "[xsai_alloc] WARNING: hugepages unavailable -- "
                        "physical contiguity NOT guaranteed.\n"
                        "  AMU single-TLB-base mode will corrupt data when a "
                        "tile crosses a physical page boundary.\n"
                        "  Set vm.nr_hugepages >= %zu or define "
                        "RESERVED_PHYS_BASE_ADDR.\n",
                pool_size >> 21 /* pool_size / 2MiB */);
        pool_base = mmap(NULL, pool_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (pool_base != MAP_FAILED) {
            mlock(pool_base, pool_size); /* pin pages; NOT a contiguity guarantee */
        }
        pool_phys_contiguous = 0;
    }
    fprintf(stderr, "[xsai_alloc] pool_base = %p\n", pool_base);
#endif /* RESERVED_PHYS_BASE_ADDR */

    if (pool_base == MAP_FAILED) {
        fprintf(stderr, "[xsai_alloc] mmap failed\n");
        pool_base = NULL;
        return -1;
    }

    /* Single free block covering the whole pool. */
    free_list        = (free_block_t *)pool_base;
    free_list->size  = pool_size - FREE_HDR_SIZE;
    free_list->next  = NULL;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

void *xsai_malloc(size_t size) {
    if (size == 0) return NULL;

    pthread_mutex_lock(&pool_lock);

    if (pool_base == NULL) {
        if (pool_init() != 0) {
            pthread_mutex_unlock(&pool_lock);
            return NULL;
        }
    }

    /* Reserve space for the alloc header and align. */
    size_t need = align_up(size + ALLOC_HDR_SIZE, ALIGN);

    free_block_t *prev = NULL;
    free_block_t *cur  = free_list;
    while (cur) {
        size_t avail = cur->size + FREE_HDR_SIZE;
        if (avail >= need) {
            size_t leftover = avail - need;
            if (leftover >= FREE_HDR_SIZE + ALIGN) {
                /* Split: create a new free block for the leftover. */
                free_block_t *split = (free_block_t *)((char *)cur + need);
                split->size = leftover - FREE_HDR_SIZE;
                split->next = cur->next;
                if (prev) prev->next = split;
                else       free_list = split;
            } else {
                /* Use the whole block (leftover too small to split). */
                if (prev) prev->next = cur->next;
                else       free_list = cur->next;
            }

            alloc_header_t *hdr = (alloc_header_t *)cur;
            hdr->size = need - ALLOC_HDR_SIZE;

            xsai_alloc_count++;
            xsai_alloc_total_count++;
            xsai_alloc_bytes += need;
            if (xsai_alloc_bytes > xsai_alloc_peak_bytes)
                xsai_alloc_peak_bytes = xsai_alloc_bytes;

            pthread_mutex_unlock(&pool_lock);
            return (char *)hdr + ALLOC_HDR_SIZE;
        }
        prev = cur;
        cur  = cur->next;
    }

    fprintf(stderr, "[xsai_alloc] out of memory (requested %zu bytes, active=%zu, used=%zu MiB)\n",
            size, xsai_alloc_count, xsai_alloc_bytes >> 20);
    pthread_mutex_unlock(&pool_lock);
    return NULL;
}

void xsai_free(void *ptr) {
    if (!ptr) return;

    pthread_mutex_lock(&pool_lock);

    if (!pool_base ||
        (uintptr_t)ptr < (uintptr_t)pool_base ||
        (uintptr_t)ptr >= (uintptr_t)pool_base + pool_size) {
        fprintf(stderr, "[xsai_alloc] xsai_free(%p): pointer NOT in pool "
                "[%p, %p) -- called free() instead\n",
                ptr, pool_base, (char *)pool_base + pool_size);
        pthread_mutex_unlock(&pool_lock);
        free(ptr); /* fall back to libc for non-pool pointers */
        return;
    }

    alloc_header_t *hdr = (alloc_header_t *)((char *)ptr - ALLOC_HDR_SIZE);
    free_block_t   *blk = (free_block_t *)hdr;
    blk->size = hdr->size + ALLOC_HDR_SIZE - FREE_HDR_SIZE;

    xsai_alloc_count--;
    xsai_alloc_bytes -= (blk->size + FREE_HDR_SIZE);

    /* Insert into free list sorted by address. */
    free_block_t *prev = NULL;
    free_block_t *cur  = free_list;
    while (cur && (char *)cur < (char *)blk) {
        prev = cur;
        cur  = cur->next;
    }
    blk->next = cur;
    if (prev) prev->next = blk;
    else       free_list = blk;

    /* Coalesce with next block. */
    if (blk->next &&
        (char *)blk + FREE_HDR_SIZE + blk->size == (char *)blk->next) {
        blk->size += FREE_HDR_SIZE + blk->next->size;
        blk->next  = blk->next->next;
    }

    /* Coalesce with previous block. */
    if (prev &&
        (char *)prev + FREE_HDR_SIZE + prev->size == (char *)blk) {
        prev->size += FREE_HDR_SIZE + blk->size;
        prev->next  = blk->next;
    }

    pthread_mutex_unlock(&pool_lock);
}

int xsai_in_pool(const void *ptr) {
    return pool_base != NULL
        && (uintptr_t)ptr >= (uintptr_t)pool_base
        && (uintptr_t)ptr <  (uintptr_t)pool_base + pool_size;
}

/* Returns 1 if the pool is backed by physically-contiguous memory.
 *
 * When this returns 1 the AMU may use "single-TLB-base + PA-offset" mode:
 *   PA(ptr) = TLB(pool_base) + (ptr - pool_base)
 * i.e. one TLB lookup for pool_base, then all intra-pool addresses are
 * resolved by pure PA arithmetic without further page-table walks.
 *
 * This is valid because every VA offset within the pool maps to the
 * same PA offset from the pool's physical base — guaranteed by the
 * huge-page or /dev/mem mapping that backs the pool.
 *
 * Returns 0 if the pool uses ordinary 4 KiB pages: physical contiguity
 * is not guaranteed, and the AMU must perform a TLB lookup whenever it
 * crosses a virtual page boundary. */
int xsai_pool_phys_contiguous(void) {
    return pool_phys_contiguous;
}

void xsai_alloc_print_stats(void) {
    pthread_mutex_lock(&pool_lock);
    if (pool_base == NULL) {
        pthread_mutex_unlock(&pool_lock);
        return; /* allocator was never activated */
    }

    size_t used_bytes = xsai_alloc_bytes;
    size_t live_count = xsai_alloc_count;
    size_t total      = xsai_alloc_total_count;
    size_t peak       = xsai_alloc_peak_bytes;
    size_t cap        = pool_size;

    /* Walk free list to compute fragmentation / free space. */
    size_t free_bytes  = 0;
    size_t free_blocks = 0;
    free_block_t *cur  = free_list;
    while (cur) {
        free_bytes  += cur->size + FREE_HDR_SIZE;
        free_blocks++;
        cur = cur->next;
    }

    pthread_mutex_unlock(&pool_lock);

    fprintf(stderr, "\n");
    fprintf(stderr, "======= xsai_alloc statistics =================================\n");
    fprintf(stderr, "  pool capacity     : %zu MiB  (%zu bytes)\n",
            cap >> 20, cap);
    fprintf(stderr, "  total alloc calls : %zu\n", total);
    fprintf(stderr, "  live allocations  : %zu\n", live_count);
    fprintf(stderr, "  live bytes        : %.3f MiB  (%zu bytes)\n",
            (double)used_bytes / (1024.0 * 1024.0), used_bytes);
    fprintf(stderr, "  peak usage        : %.3f MiB  (%zu bytes)\n",
            (double)peak / (1024.0 * 1024.0), peak);
    fprintf(stderr, "  peak / capacity   : %.1f%%\n",
            cap > 0 ? (double)peak / (double)cap * 100.0 : 0.0);
    fprintf(stderr, "  free blocks       : %zu  (%.3f MiB free)\n",
            free_blocks, (double)free_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "================================================================\n");
    fprintf(stderr, "\n");
}
