# AC922 TP3 head replication and DFlash2 trial

This branch starts from `db292f9a4` of the 1Cat fork. It targets the local
Qwen3.8-27B AWQ checkpoint with 24 Q / 4 KV attention heads, 16 full-attention
layers, 48 GDN layers, and 17408-wide MLPs. The original checkpoint files are
read without rewriting them. The code is opt-in through `--hf-overrides` and
an isolated runtime package.

## Why the loader changes are specific to this checkpoint

- The attention Q projection also contains output gates: each Q head occupies
  512 rows. Shared q/k norms and the GDN norm are left alone.
- GDN b/a, `A_log`, and `dt_bias` belong to value heads. Two copied key heads
  require six copied value heads and their recurrence parameters.
- Int4 weights pack eight input channels per int32. Zero points pack eight
  output channels per int32. Output-projection source and copied scale groups
  are both halved, preserving their combined contribution.
- MLP width 17408 does not divide by three. Adding 256 zero neurons gives
  17664, or 5888 channels per GPU. The 248320 vocabulary keeps its logical
  size while its internal embedding and LM-head padding uses a 192-row unit.
- The DFlash2 draft checkpoint has 32 Q / 8 KV heads. This fork's DFlash
  loader still constructs the draft with the target TP group even when
  `draft_tensor_parallel_size=1` parses successfully. The opt-in draft path
  copies one KV group and its four Q heads to form 36 Q / 9 KV, and zero-pads
  its MLP. The draft embedding and LM head are shared with the target.

## Isolated host staging

Run from a checkout of this branch **on the AC922**. Adjust the two source
paths if they differ. This creates a separate RAM copy of only the `vllm`
package and symlinks the unchanged Python dependencies; it does not edit the
installed production runtime.

```sh
source_root=$PWD
runtime=/home/debian/ac922-vllm/runtime-525-20260928
stage=/dev/shm/ac922-tp3
mkdir -p "$stage/site-packages"
for package in "$runtime"/site-packages/*; do
    ln -s "$package" "$stage/site-packages/${package##*/}"
done
unlink "$stage/site-packages/vllm"
cp -a "$runtime/site-packages/vllm" "$stage/site-packages/vllm"
for file in \
    vllm/config/speculative.py \
    vllm/model_executor/models/qwen3_5.py \
    vllm/model_executor/models/qwen3_5_tp3_heads.py \
    vllm/model_executor/models/qwen3_dflash.py \
    vllm/model_executor/models/qwen3_dflash_tp3_heads.py \
    vllm/v1/core/kv_cache_utils.py \
    vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py; do
    install -m 644 "$source_root/$file" "$stage/site-packages/$file"
done
```

The staging directory should be new. Use the existing wrapper
`/home/debian/ac922-vllm/run_1cat_525.sh` with
`AC922_RUN_ROOT=$stage`, `AC922_VLLM_SITE=$stage/site-packages`, and
`AC922_RUNTIME_ROOT=$runtime`. `serve_tp3_trial.sh` sets those values by
default for this machine.

## Trial modes

Use GPUs 0/1/2 after confirming they are available. Both modes bind
`127.0.0.1:8003` and disable vLLM custom allreduce on V100.

```sh
sh tools/ac922/serve_tp3_trial.sh baseline
sh tools/ac922/serve_tp3_trial.sh dflash
```

The baseline defaults to a 200k window and 0.90 GPU utilization. The DFlash2
mode defaults to a 64k window, depth 3, 0.93 utilization, and four schedulable
requests. Override with `AC922_MAX_MODEL_LEN`, `AC922_DFLASH_DEPTH`, or
`AC922_GPU_MEMORY_UTILIZATION` for experiments. Depth 7 achieved higher
single-stream decode speed, while the measured four-way configuration uses
depth 3 and a five-group KV layout. Four simultaneous full-64k requests do
not fit in the measured KV pool.

`compare_greedy_tp2_tp3.py` captures and compares deterministic output token
IDs. `bench_stream_token_ids.py` counts token IDs inside streamed chunks;
counting SSE messages understates speculative decode throughput.
`stability_tp3.py` logs completed requests, scheduler Running/Waiting, and
GPU temperatures. Test results for this AC922 are in the companion
`ac922-vllm/bench-results/2026-09-29-tp3/REPORT.md` workspace artifact.
