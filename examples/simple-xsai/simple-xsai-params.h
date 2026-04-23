#pragma once

#include "llama.h"

// XSAI model-load knobs intentionally exposed by this example.
// CPU-only policy is enforced in the conversion helper rather than by callers.
struct llama_model_xsai_parms {
    // Optional model loading progress hook.
    llama_progress_callback progress_callback;
    void * progress_callback_user_data;

    // Optional GGUF metadata overrides.
    const llama_model_kv_override * kv_overrides;

    // Load-time feature switches relevant to CPU inference.
    bool vocab_only;
    bool use_mlock;
    bool check_tensors;
    bool no_alloc;
};

inline llama_model_xsai_parms llama_model_xsai_default_parms() {
    const llama_model_params defaults = llama_model_default_params();
    return {
        defaults.progress_callback,
        defaults.progress_callback_user_data,
        defaults.kv_overrides,
        defaults.vocab_only,
        true,
        defaults.check_tensors,
        defaults.no_alloc,
    };
}

inline llama_model_params to_llama_model_params(const llama_model_xsai_parms & xsai_params) {
    llama_model_params model_params = llama_model_default_params();

    // XSAI runs CPU inference only: disable all GPU/device offload paths.
    model_params.n_gpu_layers                = 0;
    model_params.progress_callback           = xsai_params.progress_callback;
    model_params.progress_callback_user_data = xsai_params.progress_callback_user_data;
    model_params.kv_overrides                = xsai_params.kv_overrides;
    model_params.vocab_only                  = xsai_params.vocab_only;

    // Keep model data resident in normal memory instead of demand-paged mmap/direct I/O.
    model_params.use_mmap                    = false;
    model_params.use_direct_io               = false;
    model_params.use_mlock                   = xsai_params.use_mlock;
    model_params.check_tensors               = xsai_params.check_tensors;
    model_params.no_alloc                    = xsai_params.no_alloc;
    return model_params;
}

// XSAI context knobs intentionally exposed by this example.
// Threading is kept explicit because simulation typically wants deterministic CPU usage.
struct llama_context_xsai_parms {
    uint32_t n_ctx;
    uint32_t n_batch;
    int32_t n_threads;
    int32_t n_threads_batch;
    bool no_perf;
};

inline llama_context_xsai_parms llama_context_xsai_default_parms() {
    const llama_context_params defaults = llama_context_default_params();
    return {
        defaults.n_ctx,
        defaults.n_batch,
        1,
        1,
        1,
    };
}

inline llama_context_params to_llama_context_params(const llama_context_xsai_parms & xsai_params) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = xsai_params.n_ctx;
    ctx_params.n_batch         = xsai_params.n_batch;
    ctx_params.n_threads       = xsai_params.n_threads;
    ctx_params.n_threads_batch = xsai_params.n_threads_batch;
    ctx_params.no_perf         = xsai_params.no_perf;

    // XSAI uses CPU execution only, so keep context-side offload disabled as well.
    ctx_params.offload_kqv     = false;
    ctx_params.op_offload      = false;
    return ctx_params;
}
