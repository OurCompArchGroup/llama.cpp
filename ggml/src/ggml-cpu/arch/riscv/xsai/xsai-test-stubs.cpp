// Link stubs for the standalone xsai adapter tests (GGML_XSAI_BACKEND_TESTS).
//
// The adapter sources use GGML_ABORT from ggml-impl.h, and the tests link them
// without ggml-base. -Wl,--gc-sections drops every unreferenced function, so
// only the symbols reached by the tests need a definition; ggml_abort is the
// one such symbol that ggml-base would otherwise provide.
//
// This file is intentionally not part of any ggml-cpu source list: it is added
// per test executable in ggml/src/CMakeLists.txt.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) __attribute__((noreturn));

extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "ggml_abort (%s:%d): ", file, line);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}
