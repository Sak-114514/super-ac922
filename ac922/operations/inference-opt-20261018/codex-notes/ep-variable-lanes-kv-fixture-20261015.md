# Variable-lane real KV allocator fixture — 2026-10-15

## Result

Implemented and ran the D4 real-KV allocator fixture once each at 2, 3, and 4 lanes against the frozen `runtime-perf-v32`. All three cases passed all retained D4 checks. CUDA was hidden (`CUDA_VISIBLE_DEVICES=`) and the run used `OMP_NUM_THREADS=2`; the model loaded vocab-only, with no model weight tensors or KV layer buffers allocated.

| Lanes | Step-0 rows | Accepted rows per step | Final used cells | Expected `100*(6+2)*lanes` | Result |
|---:|---:|---:|---:|---:|---|
| 2 | 12 | 2 | 1600 | 1600 | PASS |
| 3 | 18 | 3 | 2400 | 2400 | PASS |
| 4 | 24 | 4 | 3200 | 3200 | PASS |

Each run reports `layers=49`, `vocab_only=1`, `n_gpu_layers=0`, `tensor_layers=0`, `buffers=0`, and `cuda_used=0`. The CUDA backend prints that no CUDA-capable device is detected because visibility is intentionally empty.

## Preserved assertions

The fixture parameterizes only lane-scaled batch dimensions and expected counts. It retains checks for unique in-range step-0 K-cell indices; seed-only outputs; correct row/lane and token mapping; seed and future-dummy causal-mask behavior; dummy rollback; step 1 reusing only the rolled-back dummy cell; later accepted rows not overwriting prefix, seed, dummy, or earlier accepted rows; contiguous positions and allocator ranges; and unchanged step-0 indices/output/mask through accepted steps 1–3.

## Artifacts and provenance

- Fixture: `fixtures/mtp-variable-lanes-20261015/real-kv.cpp`
- Build/run scripts and per-lane logs: `fixtures/mtp-variable-lanes-20261015/`
- Runtime: `/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v32`
- Model metadata source: `/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`
- V32 build record: `notes/V32-build.json` (`run_completed=true`)
- Runtime resolution checked with `LD_LIBRARY_PATH=.../runtime-perf-v32/lib ldd real-kv-fixture`; `libllama` and all `libggml*` resolve from V32.
- Matrix command: `bash fixtures/mtp-variable-lanes-20261015/run-matrix.sh`

## Scope limits

This is CPU-side validation of the real KV allocator, indices, masks, and lane-shaped D4 row bookkeeping only. It does not execute model layers, verify generated token/logit parity, validate MTP graph behavior, or measure GPU performance. No core source files were changed and no GPU was used.
