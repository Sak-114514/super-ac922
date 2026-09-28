#!/bin/sh
set -eu

# Run on AC922 against a locally serving Qwen3.8 model. Copy result JSON
# files to the Mac before shutting down the RAM-backed build environment.
mode=${1:?usage: benchmark_vllm_ac922.sh MODE MODEL_PATH RESULT_DIR [PORT]}
model=${2:?model path required}
result_dir=${3:?result directory required}
port=${4:-8000}
venv=${AC922_VENV:-/home/debian/ac922-vllm/ram-build/fs/venv}

test -d "$model"
mkdir -p "$result_dir"
for concurrency in 4 8; do
    "$venv/bin/vllm" bench serve \
        --backend openai \
        --host 127.0.0.1 --port "$port" \
        --model "$model" --tokenizer "$model" \
        --dataset-name random --seed 20260928 \
        --num-prompts 32 --num-warmups 2 \
        --random-input-len 512 --random-output-len 128 \
        --max-concurrency "$concurrency" --ignore-eos \
        --save-result --save-detailed \
        --result-dir "$result_dir" \
        --result-filename "$mode-c$concurrency.json"
done
