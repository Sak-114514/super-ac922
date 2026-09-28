#!/bin/sh
set -eu

# Run on AC922 after installing a verified 1Cat-vLLM wheel into the selected
# RAM virtualenv. Modes: baseline, mtp1, mtp4, ngram, ngram_gpu, dflash.
# Defaults to TP2 on GPUs 0,1; for the cross-group comparison set
# AC922_TP_SIZE=4 and CUDA_VISIBLE_DEVICES=0,1,2,3.
mode=${1:?usage: serve_qwen38_ac922.sh MODE [PORT]}
port=${2:-8000}
root=${AC922_BUILD_ROOT:-/home/debian/ac922-vllm/ram-build/fs}
model=${AC922_MODEL_PATH:-/home/debian/models/qwen3.8-27b-awq-nicosuter}
branch=${AC922_BRANCH:-main}
tp=${AC922_TP_SIZE:-2}

case "$branch" in
    main)
        venv=$root/venv
        unset PYTHONPATH
        export LD_LIBRARY_PATH=/usr/local/nccl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
        ;;
    alternate)
        venv=$root/venv-third-party
        export PYTHONPATH=$venv/lib/python3.11/site-packages:$root/venv/lib/python3.11/site-packages
        export LD_LIBRARY_PATH=$root/third-party-test/extra-libs:/usr/local/nccl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
        ;;
    *) echo "Unknown branch: $branch" >&2; exit 2 ;;
esac

test -d "$model"
test -x "$venv/bin/vllm"
export PATH=$venv/bin:/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export CUDA_HOME=/usr/local/cuda
export NCCL_P2P_LEVEL=NVL NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0
export VLLM_SM70_QUANT_BACKEND=turbomind
export VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=0
export VLLM_1CAT_ENABLE_QWEN35_MTP_DEFAULTS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

spec=
case "$mode" in
    baseline) ;;
    mtp1|mtp4)
        export VLLM_DEBUG_MTP_LOAD=1 VLLM_DEBUG_MTP_LOAD_VERBOSE=1
        if [ "$mode" = mtp1 ]; then tokens=1; else tokens=4; fi
        spec="{\"method\":\"mtp\",\"num_speculative_tokens\":$tokens,\"use_local_argmax_reduction\":true}"
        ;;
    ngram|ngram_gpu)
        spec="{\"method\":\"$mode\",\"num_speculative_tokens\":4,\"prompt_lookup_min\":2,\"prompt_lookup_max\":5}"
        ;;
    dflash)
        draft=${AC922_DFLASH_MODEL:?Set AC922_DFLASH_MODEL to a verified DFlash2 checkpoint}
        test -d "$draft"
        spec="{\"method\":\"dflash\",\"model\":\"$draft\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}"
        ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

set -- "$venv/bin/vllm" serve "$model" \
    --tensor-parallel-size "$tp" \
    --quantization compressed-tensors \
    --dtype float16 \
    --max-model-len "${AC922_MAX_MODEL_LEN:-2048}" \
    --max-num-seqs "${AC922_MAX_NUM_SEQS:-8}" \
    --gpu-memory-utilization "${AC922_GPU_MEMORY_UTILIZATION:-0.90}" \
    --port "$port"
if [ -n "$spec" ]; then set -- "$@" --speculative-config "$spec"; fi
if [ "${AC922_ENFORCE_EAGER:-0}" = 1 ]; then set -- "$@" --enforce-eager; fi
exec "$@"
