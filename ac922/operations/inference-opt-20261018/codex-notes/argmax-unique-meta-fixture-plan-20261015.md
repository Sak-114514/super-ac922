# Argmax uniqueness: real Meta 3-device graph fixture plan

Date: 2026-10-15
Status: fixture source prepared; frozen V12 manifest present. The split callback marks `ARGMAX_SPLIT` output as PARTIAL so the Meta graph must insert the P2P AllReduce. Compile and bounded 3-GPU run are next.

## Why this fixture

The CPU algebra fixture and the single-device/CPU-aggregated split CUDA fixture passed, but neither exercised the actual Meta scheduler, split-state callback, P2P AllReduce graph rewrite, finalizer placement, output views, or Meta-to-host transfer. This fixture will verify those boundaries together without loading a model.

## Ownership / constraints

- Write only under `fixtures/argmax-unique-meta-20261015/` and this note plus its result note/logs.
- Do not edit engine sources or existing fixtures.
- Build only this standalone fixture against the frozen V12 runtime after its manifest arrives; no engine rebuild.
- Do not run until root grants the 3-GPU window. Use only GPU3/4/5 UUIDs, the existing island-B set.
- No model load, driver reset, process cleanup, or power/OS changes.

## Fixture design

1. Load the frozen V12 CUDA, ggml-base, and ggml libraries from the runtime manifest. Initialize three simple CUDA backends and one CUDA Meta backend containing exactly those three devices. Use an explicit Meta split-state callback for one F32 logits tensor `[165546, nrows]`: axis 0, one segment, `nr[0]=1`, and equal local vocab widths `55182` per rank. Mark every other tensor mirrored. This matches the 3-device Meta group used by one `3,3` island.
2. Allocate logits through the Meta backend. For each rank, get its simple member tensor and upload only that member''s contiguous `[55182, nrows]` vocab slice. All logits are synthetic finite values except named NaN/Inf cases; no full model tensors or weights are read.
3. Construct the actual private graph: `ggml_argmax_split_unique(logits, 3)` → `ggml_argmax_unique_finalize(candidates, 3)` → contiguous raw-ID and uniqueness views. Keep the partial node and both final views observable. Build and execute with a scheduler whose backends are `{Meta, CPU}`.
4. Run the same graph shape A/B/A for each `nrows` in `{1,4,8}` so the first cold shape may use fallback and subsequent executions exercise the repeated-shape in-graph P2P path. Change only logits contents between A and B. Use `GGML_CUDA_ALLREDUCE=none`; leave `GGML_CUDA_P2P_AR_DISABLE` unset. Enable P2P launch diagnostics for the bounded run and capture the log.
5. Download the final ID and uniqueness views through the Meta backend to ordinary host arrays. Compare every row against a CPU reference that reproduces the CUDA lower-index tie rule and marks uniqueness false for any non-finite value, all-invalid row, or max tie. Also read final simple-member views and require all three ranks to agree. Check A and A-again outputs exactly match.
6. Inputs cover: unique negative maximum on rank 2 with all three shard maxima negative; equal maxima straddling the rank0/rank1 boundary; equal maxima across a local 4096-key split boundary; NaN on rank1 with a finite global winner on rank2; +Inf on rank0; all `-Inf`; and another unique negative winner. This checks both raw-index offsets and metadata reduction over all ranks.

## Evidence that qualifies as the real Meta/P2P path

Correct answers alone are insufficient. The run log must show:

- Meta device reports three CUDA members.
- Meta backend logs `using in-graph P2P AllReduce`.
- The repeated A/B/A executions produce `p2p allreduce launch` records for each participating rank, not only a copied result from CPU fallback.
- Graph source/outputs confirm `ARGMAX_SPLIT` partial → `ALLREDUCE` → `ARGMAX` split-final → views, with F32 contiguous candidates and I32 contiguous ID/flag planes.

If P2P initialization or graph dispatch is absent, report the Meta graph correctness separately from the P2P-path gate; do not label the run a P2P pass.

## Expected command workflow after V12 manifest is frozen

1. Read `runtime-perf-v12/runtime-manifest.json`; derive paths from the manifest and verify the required private libraries.
2. Compile only the new fixture with CUDA 12.4 / SM70 flags and V12 runtime libraries.
3. Inspect `ldd`/RUNPATH and the binary hash; do not execute yet.
4. Notify root that the fixture binary is ready.
5. After root grants GPUs3/4/5, launch one fresh process pinned by explicit UUID order; set the P2P environment above; save full stdout/stderr.
6. Record output checks, AllReduce/P2P markers, device temperatures/memory after the process, and all limitations.

## Acceptance gates

- All three rank-local vocab shards have size 55182 and are actually split (not mirrored).
- For each `nrows`, downloaded raw IDs equal the CPU lower-index argmax; flags equal the CPU finite-and-unique predicate; A/B/A matches exactly.
- The partial candidate tensor is `[12, nrows]` F32 with one rank slot per member, and final output is `[nrows,2]` I32 split into IDs and flags.
- In-graph P2P markers and per-rank launch logs are present on repeated shapes.
- Canary/sanitizer checks may be added only if they do not broaden source ownership or runtime cost; the core gate is exact IDs/flags, Meta member agreement, and real P2P execution evidence.

The fixture result is kernel/graph plumbing evidence only. It is not a full-model semantic or performance result.
