# SM70 MMVQ 8-row chunk fixture results

Date: 2026-10-15 UTC. Scope: standalone fixture against frozen `runtime-perf-v12`, GPU2 only. No model was loaded and no engine source or build was changed.

## Result

The existing `GGML_SM70_DECODE_LINEAR_CHUNKS=8` route is correct for local T=12 (8+4) and T=16 (8+8) MoE rows on all seven tested GGUF expert formats. For every route canary, candidate `MUL_MAT_ID` output was bitwise identical to explicit MMVQ chunk graphs stitched back to `[N, topk, T]`. Separate dense `MUL_MAT` IQ3_XXS and IQ4_NL checks were also bitwise identical to explicit 8+tail graphs.

Actual runtime dispatch markers were recorded independently in each process:

```text
sm70 decode linear chunks active: moe tokens=12 chunk=8
sm70 decode linear chunks active: moe tokens=16 chunk=8
```

Both logs also show `compiled=1`, CUDA graph `capture`, and `replay` paths. Initial warmup and some changed bindings use eager or re-instantiated graphs; these are visible in the logs. The A/B/A route and activation canaries passed through those transitions.

## Coverage and timing

The fixture used E=16, top-k=10, 3 distinct route/input canaries (balanced, skewed, changed balanced), and real nonzero packed row samples from the existing manifest. Local K/N were block aligned: gate/up K=2560, N=224 for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S; down IQ4_NL K=224, N=2560; down Q2_0 K=256, N=2560. Q2_0 was not forced to use IQ4_NL's block size.

| Format | T=12 exact | T=16 exact |
|---|---:|---:|
| IQ2_XXS | PASS | PASS |
| IQ2_XS | PASS | PASS |
| IQ2_S | PASS | PASS |
| IQ3_XXS | PASS | PASS |
| IQ3_S | PASS | PASS |
| IQ4_NL | PASS | PASS |
| Q2_0 | PASS | PASS |

On the third A/B/A replay, median fixture wall time per full candidate graph was 105 us at T=12 and 129 us at T=16. The explicit two-graph chunk reference medians were 131 us and 153 us. This is a small single-layer fixture, and these figures include host scheduler/synchronization overhead; they are not kernel-only timings or model throughput. In this fixture, T=16 handles 33% more token rows for about 23% more wall time, roughly 8% higher row throughput than T=12. That makes T=16 a reasonable next full-model candidate, subject to semantic and complete-step validation.

Sampled CPU dequantized-dot checks were finite, with relative L2 errors from about 0.13% to 0.55% and maximum absolute error at most 1.02e-3 on sampled output coordinates. These checks are diagnostic, separate from the exact MMVQ-vs-MMVQ parity gate.

## Verification and hardware state

- CPU route-map self-test: PASS for T=8/12/16/20, balanced/skewed, top-k=10, E=16; each compacted expert map was invertible and 8-row slicing preserved token/top-k order.
- Fixture compiled as a standalone C++ runner against the existing V12 shared libraries; no engine build ran.
- Bounded GPU run used only UUID `GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb`; the temperature guard never reached the 58 C stop gate. Final GPU2 readback: 39 C, 0 MiB, 0% utilization; no compute-app process remained. Other GPUs were not selected.
- Initial fixture attempts exited during preflight/scheduler setup before test kernels; `CUDA0` was correctly identified via the device description, and the V12 scheduler requirement for a trailing CPU backend was fixed by using `{CUDA, CPU}` with all test tensors explicitly assigned to CUDA. The corrected run passed.

Raw logs and candidate outputs are under:

`/home/debian/ac922-monster/experiments/standard-opt-20261014/results/mmvq-chunks-gpu2-20261015T065558Z/`

## Limits

This does not validate end-to-end MTP speed, full-model layer coverage, model semantics, or full-sized expert routing. T=8/no-op and T=20 are only CPU-map tested; the CUDA fixture ran T=12 and T=16. The existing backend contains the chunk optimization already; this work only tests it.
