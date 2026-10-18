# Grouped MMQ J tile fixture results (2026-10-15)

## Verdict

For this bounded uniform fixture with 20 assigned tokens per expert across 512 experts, **J=32 was the fastest of J=16/32/64 for all seven sampled formats**. Every candidate output was bitwise identical to the J=32 F32 reference. The grouped MMQ kernel ran as one CUDA graph node per replay.

The result supports a targeted next full-model trial of `GGML_CUDA_MOE_GROUPED_J=32` against the current heuristic (which permits J=64 at average 20). It does not establish end-to-end improvement: weights are real packed samples, but rows are repeated to synthetic experts/output rows and routing is uniform synthetic input.

## Measurements

Each number is average GPU kernel time per one-node CUDA graph replay, over 120 timed replays after 8 warmups. All outputs were finite. J16 and J64 each had 0 bit differences and max absolute difference 0 vs J32 for every type.

| Type | Tensor role | N×K | J16 (µs) | J32 (µs) | J64 (µs) | J32 latency reduction vs J64 |
|---|---|---:|---:|---:|---:|---:|
| IQ2_XXS | gate/up | 192×2560 | 1250.0992 | 942.8054 | 1298.0053 | 27.37% |
| IQ2_XS | gate/up | 224×2560 | 1326.0288 | 903.6118 | 1633.8518 | 44.69% |
| IQ2_S | gate/up | 192×2560 | 1349.4357 | 904.2517 | 1655.3387 | 45.37% |
| IQ3_XXS | gate/up | 224×2560 | 1182.3786 | 865.9968 | 1308.1088 | 33.80% |
| IQ3_S | gate/up | 192×2560 | 1269.2907 | 877.0304 | 1358.1056 | 35.42% |
| IQ4_NL | down | 768×640, K padded to 768 | 1002.6923 | 725.1371 | 1104.3157 | 34.34% |
| Q2_0 | down | 768×640, K padded to 768 | 907.0933 | 664.9003 | 1297.2374 | 48.74% |

Compared with J32, J16 was 32.59–49.23% slower and J64 was 37.67–95.10% slower in this fixture.

## Reproduction and boundaries

- Run label: `grouped-j-20261015`
- GPU: UUID `GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb`, Tesla V100-SXM2-16GB, SM70.
- Route: 512 experts × exactly 20 synthetic assignments, 10,240 total, identity output ids.
- Weight data: sixteen actual packed rows from expert 0 in `fixtures/volta-tc-real/manifest.json` repeated across the synthetic experts/output rows. This does not use the full expert matrices or model routing.
- Activations: deterministic synthetic Q8_1_MMQ blocks.
- Command: `CUDA_VISIBLE_DEVICES=GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb timeout 300s fixtures/grouped-j-20261015/grouped-j-fixture fixtures/volta-tc-real/manifest.json`
- Correctness: J16 and J64 vs J32, 7/7 formats, bitdiff=0, max_abs=0, nonfinite=0. Summary reports `failed_exact_comparisons=0`.
- GPU was free after the runner exited: all cards reported 0% utilization and no compute processes; GPU2 returned to 1 MiB usage.

Artifacts:

- Exact raw output: `notes/grouped-j-20261015-results.log`
- Structured output: `notes/grouped-j-20261015-results.json`
- Runner: `fixtures/grouped-j-20261015/grouped-j-fixture.cu`
- Build log: `notes/grouped-j-20261015-build.log`
- Recomputed summary helper: `fixtures/grouped-j-20261015/summarize.py`
