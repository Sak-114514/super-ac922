#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu

root=/home/debian/ac922-vllm
site=$root/runtime-tp3-mtp-20260929/site-packages
key=$root/coding-api.key
test -s "$key"
test -s "$site/vllm/__init__.py"
export VLLM_API_KEY=$(cat "$key")

# The user manager may start before LAN and NVIDIA devices are ready.
attempts=0
while [ "$attempts" -lt 60 ]; do
    if [ -c /dev/nvidia-uvm ] &&
        /usr/bin/ip -4 addr show | grep -q '192\.168\.1\.205/'; then
        gpu_count=$(/usr/local/bin/nvidia-smi \
            --query-gpu=index --format=csv,noheader,nounits 2>/dev/null |
            awk 'NF {count++} END {print count+0}')
        if [ "$gpu_count" -eq 6 ]; then
            break
        fi
    fi
    attempts=$((attempts + 1))
    sleep 5
done
if [ "$attempts" -eq 60 ]; then
    echo 'AC922 TP3 coding API: GPU or LAN readiness timed out' >&2
    exit 1
fi

export AC922_RUN_ROOT=/dev/shm/ac922-tp3-mtp-prod
export AC922_RUNTIME_ROOT=$root/runtime-525-20260928
export AC922_VLLM_SITE=$site
export CUDA_VISIBLE_DEVICES=0,1,2
export AC922_BIND_HOST=127.0.0.1
export AC922_TP3_PORT=8000
export AC922_MAX_MODEL_LEN=${AC922_MAX_MODEL_LEN:-65536}
export AC922_MTP_DEPTH=${AC922_MTP_DEPTH:-4}
export AC922_GPU_MEMORY_UTILIZATION=${AC922_GPU_MEMORY_UTILIZATION:-0.88}
export AC922_MAX_NUM_SEQS=${AC922_MAX_NUM_SEQS:-4}
export AC922_MAX_NUM_BATCHED_TOKENS=${AC922_MAX_NUM_BATCHED_TOKENS:-4096}
export VLLM_SM70_RMSNORM_GATED_EXACT=0
export AC922_CODING_API=1
export AC922_MIDDLEWARE=coding_api_middleware.clamp_oversized_completion
export AC922_SERVED_MODEL_NAME=qwen3.8-27b-ac922
export AC922_ANTHROPIC_MAX_COMPLETION_TOKENS=8192

exec sh "$root/serve_tp3_mtp.sh" mtp
