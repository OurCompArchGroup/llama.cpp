#ifndef XSAI_ALLOC_H
#define XSAI_ALLOC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Custom allocator backed by a reserved physical/anonymous memory region.
 * Enabled at build time via -DGGML_XSAI_ALLOC.
 *
 * Physical address and pool size are controlled by:
 *   RESERVED_PHYS_BASE_ADDR   (optional, maps /dev/mem if defined)
 *   RESERVED_MEMORY_SIZE      (pool size in bytes, default 1 GiB) */
void *xsai_malloc(size_t size);
void  xsai_free(void *ptr);

/* Returns 1 if ptr is inside the xsai pool, 0 otherwise.
 * Useful for diagnostics / verify the allocator is active. */
int   xsai_in_pool(const void *ptr);

/* Print a human-readable statistics summary to stderr.
 * Reports: pool capacity, total alloc calls, live allocations,
 * live bytes, peak usage, peak/capacity ratio, free blocks. */
void  xsai_alloc_print_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* XSAI_ALLOC_H */
