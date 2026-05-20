#!/usr/bin/env bash
#
# sample usage:
#
# mkdir tmp
#
# # CPU-only build
# bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with CUDA support
# GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with SYCL support
# GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with VULKAN support
# GG_BUILD_VULKAN=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with WebGPU support
# GG_BUILD_WEBGPU=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with MUSA support
# GG_BUILD_MUSA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with KLEIDIAI support
# GG_BUILD_KLEIDIAI=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with RISC-V AME support
# GG_BUILD_RV_AME=1 \
# GG_RV_AME_LLVM_HOME=/path/to/llvm \
# GG_RV_AME_SYSROOT=/path/to/sysroot \
# GG_RV_AME_QEMU_BIN=/path/to/qemu-riscv64 \
# GG_RV_AME_MODEL_BF16=/path/to/model-bf16.gguf \
# GG_RV_AME_MODEL_BASELINE=/path/to/model-baseline.gguf \
# bash ./ci/run.sh ./tmp/results ./tmp/mnt
#

if [ -z "$2" ]; then
    echo "usage: $0 <output-dir> <mnt-dir>"
    exit 1
fi

mkdir -p "$1"
mkdir -p "$2"

OUT=$(realpath "$1")
MNT=$(realpath "$2")

rm -f $OUT/*.log
rm -f $OUT/*.exit
rm -f $OUT/*.md
rm -f $OUT/*.csv
rm -f $OUT/*.jsonl

sd=`dirname $0`
cd $sd/../
SRC=`pwd`

CMAKE_EXTRA="-DLLAMA_FATAL_WARNINGS=${LLAMA_FATAL_WARNINGS:-ON} -DLLAMA_OPENSSL=OFF -DGGML_SCHED_NO_REALLOC=ON"

if [ ! -z ${GG_BUILD_METAL} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_METAL=ON"
fi

if [ ! -z ${GG_BUILD_CUDA} ]; then
    # TODO: Remove GGML_CUDA_CUB_3DOT2 flag once CCCL 3.2 is bundled within CTK and that CTK version is used in this project
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_CUDA=ON -DGGML_CUDA_CUB_3DOT2=ON"

    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')
        if [[ -n "$CUDA_ARCH" && "$CUDA_ARCH" =~ ^[0-9]+$ ]]; then
            CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
        else
            echo "Warning: Using fallback CUDA architectures"
            CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89"
        fi
    else
        echo "Error: nvidia-smi not found, cannot build with CUDA"
        exit 1
    fi
fi

if [ ! -z ${GG_BUILD_ROCM} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_HIP=ON"
    if [ -z ${GG_BUILD_AMDGPU_TARGETS} ]; then
        echo "Missing GG_BUILD_AMDGPU_TARGETS, please set it to your GPU architecture (e.g. gfx90a, gfx1100, etc.)"
        exit 1
    fi

    CMAKE_EXTRA="${CMAKE_EXTRA} -DGPU_TARGETS=${GG_BUILD_AMDGPU_TARGETS}"
fi

if [ ! -z ${GG_BUILD_SYCL} ]; then
    if [ -z ${ONEAPI_ROOT} ]; then
        echo "Not detected ONEAPI_ROOT, please install oneAPI base toolkit and enable it by:"
        echo "source /opt/intel/oneapi/setvars.sh"
        exit 1
    fi
    # Use only main GPU
    export ONEAPI_DEVICE_SELECTOR="level_zero:0"
    # Enable sysman for correct memory reporting
    export ZES_ENABLE_SYSMAN=1
    # to circumvent precision issues on CPY operations
    export SYCL_PROGRAM_COMPILE_OPTIONS="-cl-fp32-correctly-rounded-divide-sqrt"
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_SYCL=1 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON"
fi

if [ ! -z ${GG_BUILD_VULKAN} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_VULKAN=1"

    # if on Mac, disable METAL
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_METAL=OFF -DGGML_BLAS=OFF"
    fi

fi

if [ ! -z ${GG_BUILD_WEBGPU} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_WEBGPU=1 -DGGML_METAL=OFF -DGGML_BLAS=OFF"

    if [ ! -z "${GG_BUILD_WEBGPU_DAWN_PREFIX}" ]; then
        if [ -z "${CMAKE_PREFIX_PATH}" ]; then
            export CMAKE_PREFIX_PATH="${GG_BUILD_WEBGPU_DAWN_PREFIX}"
        else
            export CMAKE_PREFIX_PATH="${GG_BUILD_WEBGPU_DAWN_PREFIX}:${CMAKE_PREFIX_PATH}"
        fi
    fi

    # For some systems, Dawn_DIR needs to be set explicitly, e.g., the lib64 path
    if [ ! -z "${GG_BUILD_WEBGPU_DAWN_DIR}" ]; then
        CMAKE_EXTRA="${CMAKE_EXTRA} -DDawn_DIR=${GG_BUILD_WEBGPU_DAWN_DIR}"
    fi
fi

if [ ! -z ${GG_BUILD_MUSA} ]; then
    # Use qy1 by default (MTT S80)
    MUSA_ARCH=${MUSA_ARCH:-21}
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_MUSA=ON -DMUSA_ARCHITECTURES=${MUSA_ARCH}"
fi

if [ ! -z ${GG_BUILD_NO_SVE} ]; then
    # arm 9 and newer enables sve by default, adjust these flags depending on the cpu used
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.5-a+fp16+i8mm"
fi

if [ -n "${GG_BUILD_KLEIDIAI}" ]; then
    echo ">>===== Enabling KleidiAI support"

    CANDIDATES=(
        "armv9-a+dotprod+i8mm+sve2"
        "armv9-a+dotprod+i8mm"
        "armv8.6-a+dotprod+i8mm"
        "armv8.2-a+dotprod"
    )
    CPU=""

    for cpu in "${CANDIDATES[@]}"; do
        if echo 'int main(){}' | ${CXX:-c++} -march="$cpu" -x c++ - -c -o /dev/null >/dev/null 2>&1; then
            CPU="$cpu"
            break
        fi
    done

    if [ -z "$CPU" ]; then
        echo "ERROR: None of the required ARM baselines (armv9/armv8.6/armv8.2 + dotprod) are supported by this compiler."
        exit 1
    fi

    echo ">>===== Using ARM baseline: ${CPU}"

    CMAKE_EXTRA="${CMAKE_EXTRA:+$CMAKE_EXTRA } \
        -DGGML_NATIVE=OFF \
        -DGGML_CPU_KLEIDIAI=ON \
        -DGGML_CPU_AARCH64=ON \
        -DGGML_CPU_ARM_ARCH=${CPU} \
        -DBUILD_SHARED_LIBS=OFF"
fi

## helpers

# download a file if it does not exist or if it is outdated
function gg_wget {
    local out=$1
    local url=$2

    local cwd=`pwd`

    mkdir -p $out
    cd $out

    # should not re-download if file is the same
    wget -nv -c -N $url

    cd $cwd
}

function gg_printf {
    printf -- "$@" >> $OUT/README.md
}

function gg_run {
    ci=$1

    set -o pipefail
    set -x

    gg_run_$ci | tee $OUT/$ci.log
    cur=$?
    echo "$cur" > $OUT/$ci.exit

    set +x
    set +o pipefail

    gg_sum_$ci

    ret=$((ret | cur))
}

## ci

# ctest_debug

function gg_run_ctest_debug {
    cd ${SRC}

    rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug

    set -e

    # Check cmake, make and ctest are installed
    gg_check_build_requirements

    (time cmake -DCMAKE_BUILD_TYPE=Debug ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j$(nproc)                                  ) 2>&1 | tee -a $OUT/${ci}-make.log

    (time ctest --output-on-failure -L main -E "test-opt|test-backend-ops" ) 2>&1 | tee -a $OUT/${ci}-ctest.log

    set +e
}

function gg_sum_ctest_debug {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in debug mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
    gg_printf '\n'
}

# ctest_release

function gg_run_ctest_release {
    cd ${SRC}

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    # Check cmake, make and ctest are installed
    gg_check_build_requirements

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j$(nproc)                                    ) 2>&1 | tee -a $OUT/${ci}-make.log

    if [ -z ${GG_BUILD_LOW_PERF} ]; then
        (time ctest --output-on-failure -L 'main|python' ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    else
        (time ctest --output-on-failure -L main -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    fi

    set +e
}

function gg_sum_ctest_release {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in release mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

# test_scripts

function gg_run_test_scripts {
    cd ${SRC}

    set -e

    (cd ./tools/gguf-split && time bash tests.sh "$SRC/build-ci-release/bin" "$MNT/models") 2>&1 | tee -a $OUT/${ci}-scripts.log
    (cd ./tools/quantize   && time bash tests.sh "$SRC/build-ci-release/bin" "$MNT/models") 2>&1 | tee -a $OUT/${ci}-scripts.log

    set +e
}

function gg_sum_test_scripts {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs test scripts\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-scripts.log)"
    gg_printf '```\n'
    gg_printf '\n'
}

function gg_get_model {
    #local gguf_0="$MNT/models/qwen3/0.6B/ggml-model-f16.gguf"
    local gguf_0="$MNT/models/qwen3/0.6B/ggml-model-q4_0.gguf"
    if [[ -s $gguf_0 ]]; then
        echo -n "$gguf_0"
    else
        echo >&2 "No model found. Can't run gg_run_ctest_with_model."
        exit 1
    fi
}

function gg_run_ctest_with_model_debug {
    cd ${SRC}

    local model; model=$(gg_get_model)
    cd build-ci-debug
    set -e

    (LLAMACPP_TEST_MODELFILE="$model" time ctest --output-on-failure -L model) 2>&1 | tee -a $OUT/${ci}-ctest.log

    set +e
    cd ..
}

function gg_run_ctest_with_model_release {
    cd ${SRC}

    local model; model=$(gg_get_model)
    cd build-ci-release
    set -e

    (LLAMACPP_TEST_MODELFILE="$model" time ctest --output-on-failure -L model) 2>&1 | tee -a $OUT/${ci}-ctest.log

    # test memory leaks
    #if [[ ! -z ${GG_BUILD_METAL} ]]; then
    #    # TODO: this hangs for some reason ...
    #    (time leaks -quiet -atExit -- ./bin/test-thread-safety -m $model --parallel 2 -t 2 -p "hello") 2>&1 | tee -a $OUT/${ci}-leaks.log
    #fi

    set +e
    cd ..
}

function gg_sum_ctest_with_model_debug {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest with model files in debug mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

function gg_sum_ctest_with_model_release {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest with model files in release mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

# qwen3_0_6b

function gg_run_qwen3_0_6b {
    cd ${SRC}

    gg_wget models-mnt/qwen3/0.6B/ https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json
    gg_wget models-mnt/qwen3/0.6B/ https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/tokenizer.json
    gg_wget models-mnt/qwen3/0.6B/ https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/tokenizer_config.json
   #gg_wget models-mnt/qwen3/0.6B/ https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/special_tokens_map.json
    gg_wget models-mnt/qwen3/0.6B/ https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/model.safetensors


    gg_wget models-mnt/wikitext/ https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    unzip -o models-mnt/wikitext/wikitext-2-raw-v1.zip -d models-mnt/wikitext/

    path_models="../models-mnt/qwen3/0.6B"
    path_wiki="../models-mnt/wikitext/wikitext-2-raw"

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j$(nproc)                                    ) 2>&1 | tee -a $OUT/${ci}-make.log

    python3 ../convert_hf_to_gguf.py ${path_models} --outfile ${path_models}/ggml-model-f16.gguf  --outtype f16
    python3 ../convert_hf_to_gguf.py ${path_models} --outfile ${path_models}/ggml-model-bf16.gguf --outtype bf16

    model_f16="${path_models}/ggml-model-f16.gguf"
    model_bf16="${path_models}/ggml-model-bf16.gguf"
    model_q8_0="${path_models}/ggml-model-q8_0.gguf"
    model_q4_0="${path_models}/ggml-model-q4_0.gguf"
    model_q4_1="${path_models}/ggml-model-q4_1.gguf"
    model_q5_0="${path_models}/ggml-model-q5_0.gguf"
    model_q5_1="${path_models}/ggml-model-q5_1.gguf"
    model_q2_k="${path_models}/ggml-model-q2_k.gguf"
    model_q3_k="${path_models}/ggml-model-q3_k.gguf"
    model_q4_k="${path_models}/ggml-model-q4_k.gguf"
    model_q5_k="${path_models}/ggml-model-q5_k.gguf"
    model_q6_k="${path_models}/ggml-model-q6_k.gguf"

    wiki_test="${path_wiki}/wiki.test.raw"

    ./bin/llama-quantize ${model_bf16} ${model_q8_0} q8_0 $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q4_0} q4_0 $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q4_1} q4_1 $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q5_0} q5_0 $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q5_1} q5_1 $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q2_k} q2_k $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q3_k} q3_k $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q4_k} q4_k $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q5_k} q5_k $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q6_k} q6_k $(nproc)

    (time ./bin/llama-fit-params --model ${model_f16} 2>&1 | tee -a $OUT/${ci}-fp-f16.log)

    (time ./bin/llama-completion -no-cnv --model ${model_f16}  -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    (time ./bin/llama-completion -no-cnv --model ${model_bf16} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-bf16.log
    (time ./bin/llama-completion -no-cnv --model ${model_q8_0} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log
    (time ./bin/llama-completion -no-cnv --model ${model_q4_0} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_0.log
    (time ./bin/llama-completion -no-cnv --model ${model_q4_1} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_1.log
    (time ./bin/llama-completion -no-cnv --model ${model_q5_0} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_0.log
    (time ./bin/llama-completion -no-cnv --model ${model_q5_1} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_1.log
    (time ./bin/llama-completion -no-cnv --model ${model_q2_k} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q2_k.log
    (time ./bin/llama-completion -no-cnv --model ${model_q3_k} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q3_k.log
    (time ./bin/llama-completion -no-cnv --model ${model_q4_k} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_k.log
    (time ./bin/llama-completion -no-cnv --model ${model_q5_k} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_k.log
    (time ./bin/llama-completion -no-cnv --model ${model_q6_k} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q6_k.log

    (time ./bin/llama-perplexity --model ${model_f16}  -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    if [ -z ${GG_BUILD_NO_BF16} ]; then
        (time ./bin/llama-perplexity --model ${model_bf16} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-bf16.log
    fi
    (time ./bin/llama-perplexity --model ${model_q8_0} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log
    (time ./bin/llama-perplexity --model ${model_q4_0} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_0.log
    (time ./bin/llama-perplexity --model ${model_q4_1} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_1.log
    (time ./bin/llama-perplexity --model ${model_q5_0} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_0.log
    (time ./bin/llama-perplexity --model ${model_q5_1} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_1.log
    (time ./bin/llama-perplexity --model ${model_q2_k} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q2_k.log
    (time ./bin/llama-perplexity --model ${model_q3_k} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q3_k.log
    (time ./bin/llama-perplexity --model ${model_q4_k} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_k.log
    (time ./bin/llama-perplexity --model ${model_q5_k} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_k.log
    (time ./bin/llama-perplexity --model ${model_q6_k} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q6_k.log

    (time ./bin/llama-imatrix --model ${model_f16} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-imatrix.log

    (time ./bin/llama-save-load-state --model ${model_q4_0} -ngl 10 -c 1024 -fa off --no-op-offload) 2>&1 | tee -a $OUT/${ci}-save-load-state.log
    (time ./bin/llama-save-load-state --model ${model_q4_0} -ngl 10 -c 1024 -fa on  --no-op-offload) 2>&1 | tee -a $OUT/${ci}-save-load-state.log
    (time ./bin/llama-save-load-state --model ${model_q4_0} -ngl 99 -c 1024 -fa off                ) 2>&1 | tee -a $OUT/${ci}-save-load-state.log
    (time ./bin/llama-save-load-state --model ${model_q4_0} -ngl 99 -c 1024 -fa on                 ) 2>&1 | tee -a $OUT/${ci}-save-load-state.log

    function check_ppl {
        qnt="$1"
        ppl=$(echo "$2" | grep -oE "[0-9]+\.[0-9]+" | tail -n 1)

        if [ $(echo "$ppl > 20.0" | bc) -eq 1 ]; then
            printf '  - %s @ %s (FAIL: ppl > 20.0)\n' "$qnt" "$ppl"
            return 20
        fi

        printf '  - %s @ %s OK\n' "$qnt" "$ppl"
        return 0
    }

    check_ppl "f16"  "$(cat $OUT/${ci}-tg-f16.log  | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    if [ -z ${GG_BUILD_NO_BF16} ]; then
        check_ppl "bf16" "$(cat $OUT/${ci}-tg-bf16.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    fi
    check_ppl "q8_0" "$(cat $OUT/${ci}-tg-q8_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_0" "$(cat $OUT/${ci}-tg-q4_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_1" "$(cat $OUT/${ci}-tg-q4_1.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_0" "$(cat $OUT/${ci}-tg-q5_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_1" "$(cat $OUT/${ci}-tg-q5_1.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
   #check_ppl "q2_k" "$(cat $OUT/${ci}-tg-q2_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log # note: ppl > 20.0 for this quant and model
    check_ppl "q3_k" "$(cat $OUT/${ci}-tg-q3_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_k" "$(cat $OUT/${ci}-tg-q4_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_k" "$(cat $OUT/${ci}-tg-q5_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q6_k" "$(cat $OUT/${ci}-tg-q6_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log

    cat $OUT/${ci}-imatrix.log | grep "Final" >> $OUT/${ci}-imatrix-sum.log

    set +e
}

function gg_sum_qwen3_0_6b {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Qwen3 0.6B:\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '- perplexity:\n%s\n' "$(cat $OUT/${ci}-ppl.log)"
    gg_printf '- imatrix:\n```\n%s\n```\n' "$(cat $OUT/${ci}-imatrix-sum.log)"
    gg_printf '- f16:\n```\n%s\n```\n'  "$(cat $OUT/${ci}-tg-f16.log)"
    if [ -z ${GG_BUILD_NO_BF16} ]; then
        gg_printf '- bf16:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-bf16.log)"
    fi
    gg_printf '- q8_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q8_0.log)"
    gg_printf '- q4_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_0.log)"
    gg_printf '- q4_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_1.log)"
    gg_printf '- q5_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_0.log)"
    gg_printf '- q5_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_1.log)"
    gg_printf '- q2_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q2_k.log)"
    gg_printf '- q3_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q3_k.log)"
    gg_printf '- q4_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_k.log)"
    gg_printf '- q5_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_k.log)"
    gg_printf '- q6_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q6_k.log)"
    gg_printf '- save-load-state: \n```\n%s\n```\n' "$(cat $OUT/${ci}-save-load-state.log)"
}

# bge-small

function gg_run_embd_bge_small {
    cd ${SRC}

    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/config.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/tokenizer.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/tokenizer_config.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/special_tokens_map.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/pytorch_model.bin
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/sentence_bert_config.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/vocab.txt
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/modules.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/config.json

    gg_wget models-mnt/bge-small/1_Pooling https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/1_Pooling/config.json

    path_models="../models-mnt/bge-small"

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j$(nproc)                                    ) 2>&1 | tee -a $OUT/${ci}-make.log

    python3 ../convert_hf_to_gguf.py ${path_models} --outfile ${path_models}/ggml-model-f16.gguf

    model_f16="${path_models}/ggml-model-f16.gguf"
    model_q8_0="${path_models}/ggml-model-q8_0.gguf"

    ./bin/llama-quantize ${model_f16} ${model_q8_0} q8_0

    (time ./bin/llama-fit-params --model ${model_f16} 2>&1 | tee -a $OUT/${ci}-fp-f16.log)

    (time ./bin/llama-embedding --model ${model_f16}  -p "I believe the meaning of life is" -ngl 99 -c 0 --no-op-offload) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    (time ./bin/llama-embedding --model ${model_q8_0} -p "I believe the meaning of life is" -ngl 99 -c 0 --no-op-offload) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log

    set +e
}

function gg_sum_embd_bge_small {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'BGE Small (BERT):\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '- f16: \n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-f16.log)"
    gg_printf '- q8_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q8_0.log)"
}

# rerank_tiny

function gg_run_rerank_tiny {
    cd ${SRC}

    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/raw/main/config.json
    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/raw/main/tokenizer.json
    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/raw/main/tokenizer_config.json
    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/raw/main/special_tokens_map.json
    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/resolve/main/pytorch_model.bin
    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/raw/main/vocab.json

    path_models="../models-mnt/rerank-tiny"

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j$(nproc)                                    ) 2>&1 | tee -a $OUT/${ci}-make.log

    python3 ../convert_hf_to_gguf.py ${path_models} --outfile ${path_models}/ggml-model-f16.gguf

    model_f16="${path_models}/ggml-model-f16.gguf"

    (time ./bin/llama-fit-params --model ${model_f16} 2>&1 | tee -a $OUT/${ci}-fp-f16.log)

    # for this model, the SEP token is "</s>"
    (time ./bin/llama-embedding --model ${model_f16} -p "what is panda?\thi\nwhat is panda?\tit's a bear\nwhat is panda?\tThe giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China." -ngl 99 -c 0 --pooling rank --embd-normalize -1 --no-op-offload --verbose-prompt) 2>&1 | tee -a $OUT/${ci}-rk-f16.log

    # sample output
    # rerank score 0:    0.029
    # rerank score 1:    0.029
    # rerank score 2:    0.135

    # check that the score is in the range [$3, $4]
    function check_score {
        qnt="$1"
        score=$(echo "$2" | grep -oE "[0-9]+\.[0-9]+" | tail -n 1)

        if [ $(echo "$score < $3" | bc) -eq 1 ] || [ $(echo "$score > $4" | bc) -eq 1 ]; then
            printf '  - %s @ %s (FAIL: score not in range [%s, %s])\n' "$qnt" "$score" "$3" "$4"
            return 20
        fi

        printf '  - %s @ %s OK\n' "$qnt" "$score"
        return 0
    }

    check_score "rerank score 0" "$(cat $OUT/${ci}-rk-f16.log | grep "rerank score 0")" "0.00" "0.05" | tee -a $OUT/${ci}-rk-f16.log
    check_score "rerank score 1" "$(cat $OUT/${ci}-rk-f16.log | grep "rerank score 1")" "0.00" "0.05" | tee -a $OUT/${ci}-rk-f16.log
    check_score "rerank score 2" "$(cat $OUT/${ci}-rk-f16.log | grep "rerank score 2")" "0.10" "0.30" | tee -a $OUT/${ci}-rk-f16.log

    set +e
}

function gg_sum_rerank_tiny {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Rerank Tiny (Jina):\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '- f16: \n```\n%s\n```\n' "$(cat $OUT/${ci}-rk-f16.log)"
}

# riscv_ame

function gg_require_rv_ame_env {
    if [ -z "${GG_RV_AME_LLVM_HOME}" ]; then
        echo >&2 "Missing GG_RV_AME_LLVM_HOME"
        exit 1
    fi

    if [ -z "${GG_RV_AME_SYSROOT}" ]; then
        echo >&2 "Missing GG_RV_AME_SYSROOT"
        exit 1
    fi

    if [ -z "${GG_RV_AME_QEMU_BIN}" ]; then
        echo >&2 "Missing GG_RV_AME_QEMU_BIN"
        exit 1
    fi

    if [ -z "${GG_RV_AME_MODEL_BF16}" ]; then
        echo >&2 "Missing GG_RV_AME_MODEL_BF16"
        exit 1
    fi

    if [ -z "${GG_RV_AME_MODEL_BASELINE}" ] && [ ! -z "${GG_RV_AME_MODEL_F16}" ]; then
        GG_RV_AME_MODEL_BASELINE="${GG_RV_AME_MODEL_F16}"
    fi

    if [ -z "${GG_RV_AME_MODEL_BASELINE}" ]; then
        echo >&2 "Missing GG_RV_AME_MODEL_BASELINE"
        exit 1
    fi

    if [ ! -d "${GG_RV_AME_LLVM_HOME}" ]; then
        echo >&2 "LLVM dir not found: ${GG_RV_AME_LLVM_HOME}"
        exit 1
    fi

    if [ ! -d "${GG_RV_AME_SYSROOT}" ]; then
        echo >&2 "Sysroot dir not found: ${GG_RV_AME_SYSROOT}"
        exit 1
    fi

    if [ ! -x "${GG_RV_AME_QEMU_BIN}" ]; then
        echo >&2 "QEMU binary not executable: ${GG_RV_AME_QEMU_BIN}"
        exit 1
    fi

    if [ ! -f "${GG_RV_AME_MODEL_BF16}" ]; then
        echo >&2 "BF16 model not found: ${GG_RV_AME_MODEL_BF16}"
        exit 1
    fi

    if [ ! -f "${GG_RV_AME_MODEL_BASELINE}" ]; then
        echo >&2 "Baseline model not found: ${GG_RV_AME_MODEL_BASELINE}"
        exit 1
    fi

    GG_RV_AME_LLVM_HOME=$(realpath "${GG_RV_AME_LLVM_HOME}")
    GG_RV_AME_SYSROOT=$(realpath "${GG_RV_AME_SYSROOT}")
    GG_RV_AME_QEMU_BIN=$(realpath "${GG_RV_AME_QEMU_BIN}")
    GG_RV_AME_MODEL_BF16=$(realpath "${GG_RV_AME_MODEL_BF16}")
    GG_RV_AME_MODEL_BASELINE=$(realpath "${GG_RV_AME_MODEL_BASELINE}")
}

function gg_extract_rv_ame_ppl {
    local log_file="$1"
    grep 'Final estimate: PPL =' "${log_file}" | tail -n 1 | sed -E 's/.*PPL = ([0-9.]+).*/\1/'
}

function gg_extract_rv_ame_bench_ts {
    local log_file="$1"
    grep -m 1 '"avg_ts"' "${log_file}" | sed -E 's/.*"avg_ts": ([0-9.]+).*/\1/'
}

function gg_run_riscv_ame {
    cd ${SRC}

    gg_require_rv_ame_env

    command -v wget >/dev/null 2>&1
    command -v unzip >/dev/null 2>&1

    local qemu_cpu="${GG_RV_AME_QEMU_CPU:-rv64,v=true,vlen=128,h=true,zvfh=true,zvfhmin=true,zvfbfwma=true,zvfbfmin=true,zfbfmin=true,x-matrix=true,rlen=512,mlen=65536,melen=32}"
    local qemu_run="${GG_RV_AME_QEMU_BIN} -cpu ${qemu_cpu} -L ${GG_RV_AME_SYSROOT}"
    local ppl_ctx="${GG_RV_AME_PPL_CTX:-64}"
    local ppl_batch="${GG_RV_AME_PPL_BATCH:-64}"
    local ppl_chunks="${GG_RV_AME_PPL_CHUNKS:-8}"
    local ppl_delta_max="${GG_RV_AME_PPL_MAX_DELTA:-20.0}"
    local ppl_rel_delta_max="${GG_RV_AME_PPL_MAX_REL_DELTA:-0.01}"
    local bench_prompt="${GG_RV_AME_BENCH_PROMPT:-64}"
    local bench_batch="${GG_RV_AME_BENCH_BATCH:-64}"
    local bench_ubatch="${GG_RV_AME_BENCH_UBATCH:-64}"
    local bench_repetitions="${GG_RV_AME_BENCH_REPETITIONS:-3}"
    local bench_min_ratio="${GG_RV_AME_BENCH_MIN_RATIO:-0.50}"
    local wiki_dir="${MNT}/wikitext"
    local wiki_zip="${wiki_dir}/wikitext-2-raw-v1.zip"
    local wiki_test="${wiki_dir}/wikitext-2-raw/wiki.test.raw"
    local rv_ame_status=0

    mkdir -p "${wiki_dir}"
    gg_wget "${wiki_dir}" https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    unzip -o "${wiki_zip}" -d "${wiki_dir}" 2>&1 | tee -a $OUT/${ci}-data.log

    rm -rf build-ci-rv-ame && mkdir build-ci-rv-ame && cd build-ci-rv-ame

    set -e

    (time cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=../cmake/riscv64-toolchain.cmake \
        -DRISCV_ROOT_PATH="${GG_RV_AME_LLVM_HOME}" \
        -DRISCV_SYSROOT="${GG_RV_AME_SYSROOT}" \
        -DRISCV_TRIPLE="riscv64-unknown-linux-gnu" \
        -DRISCV_MARCH="rv64gc_zba_zicbop" \
        -DRISCV_MABI="lp64d" \
        -DRISCV_USE_LLVM=ON \
        -DGGML_CCACHE=OFF \
        -DLLAMA_CURL=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_RV_AME=ON \
        -DGGML_RV_ZFH=ON \
        -DGGML_RV_ZVFH=ON \
        -DGGML_RVV=ON \
        -DGGML_OPENMP=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_TESTS=ON) 2>&1 | tee -a $OUT/${ci}-cmake.log

    (time cmake --build . --config Release -j$(nproc)) 2>&1 | tee -a $OUT/${ci}-make.log

    local ame_backend_params='type_a=(q8_0|bf16),type_b=(f32|bf16),m=(288|768),n=128,k=(288|768),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1'

    (time bash -lc "${qemu_run} ./bin/test-backend-ops support -b CPU --buft RISCV_AME -o MUL_MAT -p '${ame_backend_params}' --output csv") \
        2>&1 | tee -a $OUT/${ci}-support.csv

    grep -E 'type_a=(q8_0|bf16)' $OUT/${ci}-support.csv | tee -a $OUT/${ci}-support-focus.csv

    grep -q '"support","1","yes"' $OUT/${ci}-support-focus.csv
    awk -F, '
        /"support","1","yes"/ { n++ }
        END {
            if (n < 12) {
                printf("Expected at least 12 AME MUL_MAT support cases, got %d\n", n) > "/dev/stderr";
                exit 1;
            }
        }
    ' $OUT/${ci}-support-focus.csv

    (time bash -lc "${qemu_run} ./bin/test-backend-ops test -b CPU --buft RISCV_AME -o MUL_MAT -p '${ame_backend_params}' --output console") \
        2>&1 | tee -a $OUT/${ci}-backend-ops.log

    (time bash -lc "${qemu_run} ./bin/llama-perplexity --model \"${GG_RV_AME_MODEL_BF16}\" -f \"${wiki_test}\" -c ${ppl_ctx} -b ${ppl_batch} --chunks ${ppl_chunks}") \
        2>&1 | tee -a $OUT/${ci}-ppl-bf16.log

    local ppl_bf16
    ppl_bf16=$(gg_extract_rv_ame_ppl $OUT/${ci}-ppl-bf16.log)
    if [ -z "${ppl_bf16}" ]; then
        echo >&2 "Failed to parse BF16 perplexity"
        exit 1
    fi

    (time bash -lc "${qemu_run} ./bin/llama-perplexity --model \"${GG_RV_AME_MODEL_BASELINE}\" -f \"${wiki_test}\" -c ${ppl_ctx} -b ${ppl_batch} --chunks ${ppl_chunks}") \
        2>&1 | tee -a $OUT/${ci}-ppl-baseline.log

    local ppl_baseline
    ppl_baseline=$(gg_extract_rv_ame_ppl $OUT/${ci}-ppl-baseline.log)
    if [ -z "${ppl_baseline}" ]; then
        echo >&2 "Failed to parse baseline perplexity"
        exit 1
    fi

    printf 'PPL_CTX=%s\n' "${ppl_ctx}" | tee -a $OUT/${ci}-ppl-summary.log
    printf 'PPL_BATCH=%s\n' "${ppl_batch}" | tee -a $OUT/${ci}-ppl-summary.log
    printf 'PPL_CHUNKS=%s\n' "${ppl_chunks}" | tee -a $OUT/${ci}-ppl-summary.log
    printf 'PPL_BASELINE_MODEL=%s\n' "${GG_RV_AME_MODEL_BASELINE}" | tee -a $OUT/${ci}-ppl-summary.log
    printf 'PPL_BF16=%s\n' "${ppl_bf16}" | tee -a $OUT/${ci}-ppl-summary.log
    printf 'PPL_BASELINE=%s\n' "${ppl_baseline}" | tee -a $OUT/${ci}-ppl-summary.log

    set +e
    awk -v a="${ppl_bf16}" -v b="${ppl_baseline}" -v t_abs="${ppl_delta_max}" -v t_rel="${ppl_rel_delta_max}" '
        BEGIN {
            d = a - b;
            if (d < 0) d = -d;
            rel = b == 0 ? 0 : d / b;
            fail = (b == 0 && d > t_abs) || (b != 0 && d > t_abs && rel > t_rel);
            printf("PPL_DELTA=%.6f\n", d);
            printf("PPL_REL_DELTA=%.6f\n", rel);
            printf("PPL_DELTA_MAX=%.6f\n", t_abs);
            printf("PPL_REL_DELTA_MAX=%.6f\n", t_rel);
            printf("PPL_STATUS=%s\n", fail ? "FAIL" : "OK");
            if (fail) {
                printf("PPL_CHECK=FAIL: PPL drift exceeds the configured absolute and relative thresholds\n");
                printf("PPL delta %.6f and relative delta %.6f exceed thresholds %.6f / %.6f\n", d, rel, t_abs, t_rel) > "/dev/stderr";
                exit 1;
            }
            printf("PPL_CHECK=OK: PPL drift is within the configured thresholds\n");
        }
    ' | tee -a $OUT/${ci}-ppl-summary.log
    local ppl_status=${PIPESTATUS[0]}
    set -e
    if [ "${ppl_status}" -ne 0 ]; then
        rv_ame_status=1
    fi

    (time bash -lc "${qemu_run} ./bin/llama-bench --model \"${GG_RV_AME_MODEL_BASELINE}\" -p ${bench_prompt} -n 0 -b ${bench_batch} -ub ${bench_ubatch} -t 1 -r ${bench_repetitions} --no-warmup -o jsonl") \
        2>&1 | tee -a $OUT/${ci}-bench-baseline.jsonl
    (time bash -lc "${qemu_run} ./bin/llama-bench --model \"${GG_RV_AME_MODEL_BF16}\" -p ${bench_prompt} -n 0 -b ${bench_batch} -ub ${bench_ubatch} -t 1 -r ${bench_repetitions} --no-warmup -o jsonl") \
        2>&1 | tee -a $OUT/${ci}-bench-bf16.jsonl

    local bench_baseline_ts
    local bench_bf16_ts
    bench_baseline_ts=$(gg_extract_rv_ame_bench_ts $OUT/${ci}-bench-baseline.jsonl)
    bench_bf16_ts=$(gg_extract_rv_ame_bench_ts $OUT/${ci}-bench-bf16.jsonl)
    if [ -z "${bench_baseline_ts}" ] || [ -z "${bench_bf16_ts}" ]; then
        echo >&2 "Failed to parse llama-bench throughput"
        exit 1
    fi

    printf 'BENCH_PROMPT=%s\n' "${bench_prompt}" | tee -a $OUT/${ci}-bench-summary.log
    printf 'BENCH_BATCH=%s\n' "${bench_batch}" | tee -a $OUT/${ci}-bench-summary.log
    printf 'BENCH_UBATCH=%s\n' "${bench_ubatch}" | tee -a $OUT/${ci}-bench-summary.log
    printf 'BENCH_REPETITIONS=%s\n' "${bench_repetitions}" | tee -a $OUT/${ci}-bench-summary.log
    printf 'BENCH_BASELINE_AVG_TS=%s\n' "${bench_baseline_ts}" | tee -a $OUT/${ci}-bench-summary.log
    printf 'BENCH_BF16_AVG_TS=%s\n' "${bench_bf16_ts}" | tee -a $OUT/${ci}-bench-summary.log
    set +e
    awk -v a="${bench_bf16_ts}" -v b="${bench_baseline_ts}" -v t="${bench_min_ratio}" '
        BEGIN {
            ratio = b == 0 ? 0 : a / b;
            fail = ratio < t;
            printf("BENCH_BF16_TO_BASELINE_RATIO=%.6f\n", ratio);
            printf("BENCH_MIN_RATIO=%.6f\n", t);
            printf("BENCH_STATUS=%s\n", fail ? "FAIL" : "OK");
            if (fail) {
                printf("BENCH_CHECK=FAIL: BF16 throughput ratio is below the configured threshold\n");
                printf("BF16 bench ratio %.6f is below threshold %.6f\n", ratio, t) > "/dev/stderr";
                exit 1;
            }
            printf("BENCH_CHECK=OK: BF16 throughput ratio is within the configured threshold\n");
        }
    ' | tee -a $OUT/${ci}-bench-summary.log
    local bench_status=${PIPESTATUS[0]}
    set -e
    if [ "${bench_status}" -ne 0 ]; then
        rv_ame_status=1
    fi

    {
        if [ "${rv_ame_status}" -eq 0 ]; then
            printf 'AME_CI_STATUS=OK\n'
            printf 'AME_CI_CHECK=OK: backend ops, PPL, and bench checks all passed\n'
        else
            printf 'AME_CI_STATUS=FAIL\n'
            printf 'AME_CI_CHECK=FAIL: one or more AME CI checks failed\n'
        fi
        printf '\n'
        printf '[backend-ops]\n'
        printf 'BACKEND_OPS_STATUS=OK\n'
        printf 'BACKEND_OPS_CHECK=OK: AME MUL_MAT support and correctness checks passed\n'
        printf '\n'
        printf '[ppl]\n'
        cat $OUT/${ci}-ppl-summary.log
        printf '\n'
        printf '[bench]\n'
        cat $OUT/${ci}-bench-summary.log
    } > $OUT/${ci}-checks.log

    if [ "${rv_ame_status}" -ne 0 ]; then
        exit "${rv_ame_status}"
    fi

    set +e
}

function gg_sum_riscv_ame {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'RISC-V AME local integration:\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '- support focus:\n```\n%s\n```\n' "$(cat $OUT/${ci}-support-focus.csv)"
    gg_printf '- ppl summary:\n```\n%s\n```\n' "$(cat $OUT/${ci}-ppl-summary.log)"
    gg_printf '- bench summary:\n```\n%s\n```\n' "$(cat $OUT/${ci}-bench-summary.log 2>/dev/null || true)"
    gg_printf '- combined checks:\n```\n%s\n```\n' "$(cat $OUT/${ci}-checks.log 2>/dev/null || true)"
    gg_printf '- backend ops:\n```\n%s\n```\n' "$(tail -n 40 $OUT/${ci}-backend-ops.log)"
}

function gg_check_build_requirements {
    if ! command -v cmake &> /dev/null; then
        gg_printf 'cmake not found, please install'
    fi

    if ! command -v make &> /dev/null; then
        gg_printf 'make not found, please install'
    fi

    if ! command -v ctest &> /dev/null; then
        gg_printf 'ctest not found, please install'
    fi
}

## main

export LLAMA_LOG_PREFIX=1
export LLAMA_LOG_TIMESTAMPS=1

if [ -z "${GG_BUILD_LOW_PERF}" ] && [ -z "${GG_BUILD_RV_AME}" ]; then
    # Create symlink: ./llama.cpp/models-mnt -> $MNT/models
    rm -rf ${SRC}/models-mnt
    mnt_models=${MNT}/models
    mkdir -p ${mnt_models}
    ln -sfn ${mnt_models} ${SRC}/models-mnt

    # Create a fresh python3 venv and enter it
    if ! python3 -m venv "$MNT/venv"; then
        echo "Error: Failed to create Python virtual environment at $MNT/venv."
        exit 1
    fi
    source "$MNT/venv/bin/activate"

    pip install -r ${SRC}/requirements.txt --disable-pip-version-check
    pip install --editable gguf-py --disable-pip-version-check
fi

ret=0

if [ ! -z "${GG_BUILD_RV_AME}" ]; then
    test $ret -eq 0 && gg_run riscv_ame
else
    test $ret -eq 0 && gg_run ctest_debug
    test $ret -eq 0 && gg_run ctest_release

    if [ -z ${GG_BUILD_LOW_PERF} ]; then
        test $ret -eq 0 && gg_run embd_bge_small
        test $ret -eq 0 && gg_run rerank_tiny

        if [ -z ${GG_BUILD_CLOUD} ] || [ ${GG_BUILD_EXTRA_TESTS_0} ]; then
            test $ret -eq 0 && gg_run test_scripts
        fi

        test $ret -eq 0 && gg_run qwen3_0_6b

        test $ret -eq 0 && gg_run ctest_with_model_debug
        test $ret -eq 0 && gg_run ctest_with_model_release
    fi
fi

cat $OUT/README.md

exit $ret
