# CI

This CI implements heavy-duty workflows that run on self-hosted runners. Typically the purpose of these workflows is to
cover hardware configurations that are not available from Github-hosted runners and/or require more computational
resource than normally available.

It is a good practice, before publishing changes to execute the full CI locally on your machine. For example:

```bash
mkdir tmp

# CPU-only build
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with CUDA support
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with SYCL support
source /opt/intel/oneapi/setvars.sh
GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with MUSA support
GG_BUILD_MUSA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with RISC-V AME support
GG_BUILD_RV_AME=1 \
GG_RV_AME_LLVM_HOME=/path/to/llvm \
GG_RV_AME_SYSROOT=/path/to/sysroot \
GG_RV_AME_QEMU_BIN=/path/to/qemu-riscv64 \
GG_RV_AME_MODEL_BF16=/path/to/model-bf16.gguf \
GG_RV_AME_MODEL_F16=/path/to/model-f16.gguf \
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# etc.
```

# Adding self-hosted runners

- Add a self-hosted `ggml-ci` workflow to [[.github/workflows/build.yml]] with an appropriate label
- Request a runner token from `ggml-org` (for example, via a comment in the PR or email)
- Set-up a machine using the received token ([docs](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners))
- Optionally update [ci/run.sh](https://github.com/ggml-org/llama.cpp/blob/master/ci/run.sh) to build and run on the target platform by gating the implementation with a `GG_BUILD_...` env

## RISC-V AME local integration mode

When `GG_BUILD_RV_AME=1` is set, `ci/run.sh` switches to an AME-specific integration flow instead of the default native CI suite. The AME mode:

- cross-builds llama.cpp with `GGML_RV_AME=ON`
- runs `test-backend-ops` support and correctness checks for representative `RISCV_AME` `MUL_MAT` cases
- runs `llama-perplexity` on `wikitext-2`
- checks the BF16/F16 perplexity delta when both model paths are provided

Required environment variables:

- `GG_RV_AME_LLVM_HOME`
- `GG_RV_AME_SYSROOT`
- `GG_RV_AME_QEMU_BIN`
- `GG_RV_AME_MODEL_BF16`

Optional environment variables:

- `GG_RV_AME_MODEL_F16`
- `GG_RV_AME_QEMU_CPU`
- `GG_RV_AME_PPL_MAX_DELTA`
