# Grouped MMQ tile-width fixture (2026-10-15)

## Scope

This is a bounded GPU2-only fixture for comparing the production compact grouped MMQ kernel at J=16, 32, and 64. It does not load the model, alter engine sources, or use model routing. It consumes only the seven packed-row samples already recorded in `fixtures/volta-tc-real/manifest.json`.

## Workload

- 512 synthetic experts, exactly 20 token assignments per expert (10,240 total slots).
- Sixteen actual packed rows from expert 0 for each format are repeated across the synthetic expert and output-row dimensions. Gate/up cases use IQ2_XXS (N=192), IQ2_XS (N=224), IQ2_S (N=192), IQ3_XXS (N=224), IQ3_S (N=192), K=2560. Down cases use IQ4_NL and Q2_0, N=768, K=640 padded to the next 256-value MMQ tile (768).
- Activations are deterministic synthetic Q8_1_MMQ blocks, one packed row per token slot and K/128 block; `ids_dst` is identity and expert bounds are uniform. This isolates grouped tile scheduling and the original MMQ arithmetic.
- For each type, J=32 is run first and becomes the exact F32 reference. J=16/J=64 are compared elementwise for bitwise identity, max absolute error, nonfinite count, and first differing output. Each tile width is captured as a one-node CUDA graph, warmed up 8 launches, then timed with CUDA events over 120 graph launches.

The measured results must be described as **packed-real-weights / synthetic-route kernel evidence**, not a model benchmark or end-to-end throughput result. The repeated rows are not a faithful per-expert weight set, and equal 20-token routing does not represent an observed token histogram.

## Build

Compiled as a standalone ppc64le CUDA fixture for `sm_70`; no engine rebuild and no GGML shared library linkage. Build log: `notes/grouped-j-20261015-build.log`.

## Run gate

The runner sets logical CUDA device 0, so only run after the owner releases GPU2, with GPU2 selected by UUID:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
CUDA_VISIBLE_DEVICES=GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb \
  fixtures/grouped-j-20261015/grouped-j-fixture \
  fixtures/volta-tc-real/manifest.json \
  > notes/grouped-j-20261015-results.log 2>&1
```

Before running, verify the UUID mapping and that the root has released GPU2. Do not run on another visible device accidentally. The fixture is expected to allocate at most a few hundred MiB for one case at a time and frees it before the next format.

## Current status

Source is present at `fixtures/grouped-j-20261015/grouped-j-fixture.cu`. Initial compile passed; one host constexpr warning was corrected, and the final build passed after ordering the J=32 correctness reference first. No CUDA context has been initialized and no GPU work has run while the full-model owner holds the GPUs.

## Final update after run

Fixture compiled and ran on the released GPU2. Results are in `notes/grouped-j-20261015-results.md`, exact raw output in `notes/grouped-j-20261015-results.log`, and structured output in `notes/grouped-j-20261015-results.json`. All seven formats matched J32 exactly. This is kernel-only synthetic-route evidence; the scope caveat above still applies.
