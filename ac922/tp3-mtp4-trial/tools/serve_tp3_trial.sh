#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -eu

mode=${1:-baseline}
case "$mode" in
    baseline) default_len=200000; default_util=0.90 ;;
    dflash) default_len=65536; default_util=0.94 ;;
    mtp) default_len=65536; default_util=0.90 ;;
    *) echo "usage: $0 baseline|dflash|mtp" >&2; exit 2 ;;
esac
model=${AC922_MODEL_PATH:-/home/debian/models/qwen3.8-27b-awq-nicosuter}
draft=${AC922_DFLASH_MODEL:-/home/debian/models/qwen3.8-27b-dflash2}
runner=${AC922_RUNNER:-/home/debian/ac922-vllm/run_1cat_525.sh}
site=${AC922_VLLM_SITE:-/dev/shm/ac922-tp3/site-packages}

test -d "$model"
test -x "$runner"
test -f "$site/vllm/model_executor/models/qwen3_5_tp3_heads.py"
export AC922_RUN_ROOT=${AC922_RUN_ROOT:-/dev/shm/ac922-tp3}
export AC922_RUNTIME_ROOT=${AC922_RUNTIME_ROOT:-/home/debian/ac922-vllm/runtime-525-20260928}
export AC922_VLLM_SITE=$site
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export NCCL_P2P_LEVEL=NVL NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0
export VLLM_SM70_QUANT_BACKEND=turbomind
export VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=0
export VLLM_1CAT_ENABLE_QWEN35_MTP_DEFAULTS=0
export VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

overrides='{"ac922_tp3_head_replication":true,"text_config":{"num_attention_heads":36,"num_key_value_heads":6,"linear_num_key_heads":18,"linear_num_value_heads":54,"intermediate_size":17664}}'
set -- "$runner" -m vllm.entrypoints.cli.main serve "$model" \
    --tensor-parallel-size 3 --disable-custom-all-reduce \
    --quantization compressed-tensors --dtype float16 \
    --hf-overrides "$overrides" \
    --max-model-len "${AC922_MAX_MODEL_LEN:-$default_len}" \
    --max-num-seqs "${AC922_MAX_NUM_SEQS:-4}" \
    --max-num-batched-tokens "${AC922_MAX_NUM_BATCHED_TOKENS:-4096}" \
    --gpu-memory-utilization "${AC922_GPU_MEMORY_UTILIZATION:-$default_util}" \
    --kv-cache-dtype fp8_e4m3 \
    --language-model-only \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --served-model-name "${AC922_SERVED_MODEL_NAME:-qwen3.8-27b-ac922}" \
    --host "${AC922_BIND_HOST:-127.0.0.1}" \
    --port "${AC922_TP3_PORT:-8003}"

case "$mode" in
    baseline) ;;
    dflash)
        test -d "$draft"
        export VLLM_USE_V2_MODEL_RUNNER=1
        export AC922_TP3_DFLASH_REPLICATE=1
        depth=${AC922_DFLASH_DEPTH:-4}
        case "$depth" in 1|2|3|4|5|6|7) ;; *) echo "invalid DFlash depth" >&2; exit 2 ;; esac
        spec="{\"method\":\"dflash\",\"model\":\"$draft\",\"num_speculative_tokens\":$depth,\"draft_tensor_parallel_size\":3,\"draft_sample_method\":\"probabilistic\",\"kv_cache_dtype\":\"auto\"}"
        set -- "$@" --speculative-config "$spec"
        ;;
    mtp)
        export AC922_TP3_MTP_REPLICATE=1
        depth=${AC922_MTP_DEPTH:-4}
        case "$depth" in 1|2|3|4) ;; *) echo "invalid MTP depth" >&2; exit 2 ;; esac
        spec="{\"method\":\"mtp\",\"num_speculative_tokens\":$depth,\"draft_tensor_parallel_size\":3,\"draft_sample_method\":\"greedy\",\"attention_backend\":\"FLASH_ATTN_V100\",\"kv_cache_dtype\":\"fp8_e4m3\"}"
        set -- "$@" --speculative-config "$spec"
        ;;
esac

if [ "${AC922_CODING_API:-0}" = 1 ]; then
    set -- "$@" --reasoning-parser qwen3 \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder
fi
if [ -n "${AC922_MIDDLEWARE:-}" ]; then
    set -- "$@" --middleware "$AC922_MIDDLEWARE"
fi

exec "$@"
