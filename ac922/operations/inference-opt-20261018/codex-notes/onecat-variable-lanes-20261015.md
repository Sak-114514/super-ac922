# Onecat variable-lane XQA fixture — 2026-10-15/16

## Status

The V34 local XQA shape and fallback tests completed. The original full-matrix report remains **FAIL** because its first key-bias fixture supplied only `0/-inf` mask entries, so the asserted stock ALiBi effect was not actually present. A corrected, isolated stock/Onecat key-bias pair then passed. The full matrix was not rerun after correcting that test input.

## Runtime and device provenance

- Runtime: `/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v34`.
- V34 manifest reported `run_completed=true`; V34 `libggml-cuda.so.0` contains the Onecat XQA logger and exported `ggml_onecat_v100::{env_enabled,supported,get_alloc_size,launch}` symbols.
- The fixture linked to V34 directly with no patched Onecat object. `ldd` resolved V34 CUDA/base libraries from `runtime-perf-v34/lib`.
- Only physical GPU2 was visible to test processes (`GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb`). Temperature/free-memory guards abort at 58°C or below 512 MiB free; the matrix observed 37–39°C and at least 15751 MiB free.
- At the end, `nvidia-smi` showed no GPU2 compute PID, 38°C, and 4 MiB residual device memory.

## Shape matrix and routing results

The fixture uses QH=24/KVH=2, D=256, Q8_0 K/V, gathered every-other-row cache views, causal F16 masks, a CPU softmax oracle, and an allocator-adjacent canary. T=1/T=3 cases use NKV=2303 to exercise a partial 256-key partition. T=6/9/10/12 use aligned NKV=2304 because stock V100 FA hit `NO_DEVICE_CODE` for a T=10/NKV=2303 fallback (`fattn-mma-f16.cuh:1858`, Volta `ncols1*ncols2 < 32`).

- Onecat actually dispatched only for T=1, no bias, NKV=2303. V34 emitted `using opt-in Onecat Flash-V100 XQA decode path`; the CPU oracle max absolute error was `7.7158213e-05`, and masked-tail mutation was bit-exact.
- T=3/S=1,2,3 at NKV=2303, and T=6/9/10/12 at NKV=2304, were rejected by Onecat admission and used the stock fallback. The 2-lane/3-lane target shapes are T=3/S=2 and T=3/S=3 (6/9 total query rows). CPU oracle, causal-mask, tail-mutation, canary, and T6→T9→T6 bit-exact ABA checks passed.
- All nine original stock-versus-Onecat F32 files passed elementwise absolute tolerance `0.004` (`rtol=0`). The T=1 Onecat-vs-stock max absolute error was `0.0023601949`; unsupported T>1 and max-bias routes matched stock exactly.
- Onecat admission is intentionally narrow: T>1 verification and any nonzero max-bias case fall back to stock. The fixture establishes correctness of these local synthetic shapes only; it does not establish throughput or full-model correctness.

## Key-bias contract and corrected pair

V34 Onecat `supported()` rejects nonzero `max_bias` (`fattn-volta-onecat.cu:737-741`). Stock CUDA supports ALiBi: `ggml_flash_attn_ext` requires a mask when `max_bias>0` (`ggml.c:5747-5755`), and the stock FA path multiplies per-head ALiBi slope by mask entries (`fattn-mma-f16.cuh:758,1938`). Production mask entries are `-abs(query_pos-key_pos)` for visible keys and `-inf` for masked keys (`llama-graph.cpp:501`; KV mask implementation in `llama-kv-cache.cpp`).

The first matrix’s T=1/max-bias case incorrectly used zero for every visible key. That makes `slope*mask` zero, so stock output did not change; the original full-matrix `compare-strict.txt` correctly remains FAIL on this fixture assertion. I corrected only the bias input and added explicit mask-entry and CPU-oracle gates, then reran only the T=1 stock/Onecat bias pair in separate corrected output directories:

- Stock: `max_bias=8`, real negative-distance mask; CPU oracle max error `0.0013146996`; stock-vs-no-bias max output delta `1.053072` (> `1e-6`).
- Onecat-enabled: T=1/no-bias dispatched Onecat; `max_bias=8` reported `onecat_supported=0` and fell back to stock. Its bias output matched the stock bias output exactly.
- Corrected pair comparison passed: no-bias Onecat-vs-stock max abs `0.0023601949` (<0.004); bias fallback parity max abs `0`.
- The first bias-only attempt accidentally executed T6/T9 ABA because the `bias` suite fell through; those artifacts remain under `results/bias-stock/` and `results/bias-onecat/`. The corrected rerun skipped all non-bias shapes and wrote to `results/bias-stock-corrected/` and `results/bias-onecat-corrected/`.

The production Qwen4Exp model does not appear to use ALiBi: `f_max_alibi_bias` defaults to zero (`llama-hparams.h:218`), model loading enables `use_alibi` only when it is positive (`llama-model.cpp:1318`), and the graph passes that hparam to FlashAttention (`llama-graph.cpp:3201`). QSA’s block-bias path also opts out when `use_alibi` is true (`qwen4exp.cpp:1949`).

## Artifacts and boundary

- Sources/scripts: `fixtures/onecat-variable-lanes-20261015/`
- Original full matrix, including its preserved FAIL: `fixtures/onecat-variable-lanes-20261015/results/compare-strict.txt`
- Corrected key-bias pair: `fixtures/onecat-variable-lanes-20261015/results/compare-bias-corrected-strict.txt`
- Build/run/safety logs and per-case outputs are stored next to those summaries on AC922 and in the local mirror.
- No core files, driver, kernel, BMC, power cap, or fan settings were changed. No model was loaded and no performance claim was measured.
