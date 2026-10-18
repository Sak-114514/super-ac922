# Adaptive grouped MMQ J fixture results (2026-10-15)

## Result

The adaptive two-queue prototype was faster than all-J64 in all seven sampled packed formats on the synthetic skewed route, with exact output equality. Kernel-only graph speedup ranged from **1.36× to 1.66×**.

This includes the extra grouped-kernel launch (baseline graph has one node; adaptive graph has two). It excludes route classification, activation quantization, compact-queue construction/reordering, prefix/offset generation, and host/device copy time. Consequently it is a candidate-kernel result, not an end-to-end speedup or a full-model result.

## Measurements

| Type | Tensor role | N×K | J64 baseline (µs) | Cold J32 + hot J64 (µs) | Speedup | Exact vs J64 |
|---|---|---:|---:|---:|---:|---|
| IQ2_XXS | gate/up | 192×2560 | 1479.6202 | 966.0587 | 1.5316× | yes, bitdiff=0 |
| IQ2_XS | gate/up | 224×2560 | 1733.4357 | 1048.7040 | 1.6529× | yes, bitdiff=0 |
| IQ2_S | gate/up | 192×2560 | 1744.7765 | 1053.0389 | 1.6569× | yes, bitdiff=0 |
| IQ3_XXS | gate/up | 224×2560 | 1298.2955 | 954.6752 | 1.3599× | yes, bitdiff=0 |
| IQ3_S | gate/up | 192×2560 | 1357.8922 | 985.1477 | 1.3784× | yes, bitdiff=0 |
| IQ4_NL | down | 768×640, K padded to 768 | 1123.3280 | 824.6955 | 1.3621× | yes, bitdiff=0 |
| Q2_0 | down | 768×640, K padded to 768 | 1325.0304 | 801.0155 | 1.6542× | yes, bitdiff=0 |

Timing used CUDA events around replay of one-node baseline and two-node adaptive CUDA graphs: 8 warmup replays and 120 timed replays per graph. Every output was finite; all seven cases had `max_abs=0`, `nonfinite=0`, and `failed_exact_comparisons=0`.

## Synthetic routing and data limits

The route was Zipf-like and synthetic: top 16 experts received 128 assignments each, next 48 received 64, next 112 received 24, next 160 received 12, next 160 received 3, and last 16 received 2. Total: 10,240 assignments over 512 experts. The threshold sent 448 experts / 5,120 assignments to J32 and 64 experts / 5,120 assignments to J64.

The 16 packed rows for each type were read from expert 0 in the real GGUF sample manifest and repeated across synthetic experts/output rows. Activations used deterministic Q8_1_MMQ blocks with random FP32 scale mantissas; the same per-token packed activations were reordered into the baseline and cold/hot queues. This is not measured production routing and does not use full expert matrices.

## Reproduction

- Label: `grouped-j-adaptive-20261015`
- Device: GPU2 UUID `GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb`, Tesla V100-SXM2-16GB, SM70.
- Command: `CUDA_VISIBLE_DEVICES=GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb timeout 300s fixtures/grouped-j-adaptive-20261015/grouped-j-adaptive-fixture fixtures/volta-tc-real/manifest.json`
- Raw stdout: `notes/grouped-j-adaptive-20261015-results.log`
- Structured output: `notes/grouped-j-adaptive-20261015-results.json`
- Runner: `fixtures/grouped-j-adaptive-20261015/grouped-j-adaptive-fixture.cu`
- Summary parser: `fixtures/grouped-j-adaptive-20261015/summarize.py`
- Build log: `notes/grouped-j-adaptive-20261015-build.log`

GPU2 was free after the fixture exited: 1 MiB usage, 0% utilization, no compute applications. No engine source or runtime was modified.
