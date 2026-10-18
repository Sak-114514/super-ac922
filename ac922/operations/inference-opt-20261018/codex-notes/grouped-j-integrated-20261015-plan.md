# Integrated adaptive grouped MMQ fixture plan (2026-10-15)

## Goal and gate

Exercise the private `mmq-moe-adaptive.cuh` prefix helper together with the original grouped MMQ kernels, while keeping the exact same global expert bounds, global ids, activation buffer, and output layout for both implementations. Do not alter the private helper or any engine file.

## Compared graphs

- Baseline: existing `launch_mmq_moe_tile_offsets(bounds, offsets, 512, 64, stream)` then original J64 grouped MMQ. Expected graph nodes: 2.
- Adaptive: actual `launch_mmq_moe_adaptive_offsets(bounds, offsets2, 512, stream)` then original J32 grouped MMQ using the cold offsets and original J64 grouped MMQ using the hot offsets. Both calls use the same global expert bounds, ids_dst, Q8_1_MMQ activation data, packed expert weights, output, and total-token stride. Expected graph nodes: 3.
- CUDA event timing wraps graph replays, so the prefix scan kernel and all grouped MMQ kernels are included. CPU input generation, packed-weight expansion, host-to-device copies, and activation preparation happen before capture and are excluded. There are no cold/hot activation queue copies or per-queue ids/bounds construction in the measured path.

## Workloads

All cases contain 512 experts and 10,240 assignments:

- `skew`: 16×128, 48×64, 112×24, 160×12, 160×3, 16×2.
- `uniform20`: 512×20.
- `zeros`: 128×0, 256×20, 128×40.

Every route shuffles global destination ids after sorting slots by expert. Activations are generated once per global token and repacked into the shared sorted-slot layout, with random FP32 scale mantissas. Adaptive cold/hot boundaries are derived only by the actual GPU prefix helper. The fixture verifies the generated baseline and adaptive prefix arrays against host formulas, then requires full output bitwise equality against the legacy J64 baseline.

Seven sampled packed formats are covered from `fixtures/volta-tc-real/manifest.json`; 16 real packed rows are repeated across synthetic experts/output rows. This is not a full-model or observed routing benchmark.

## Execution controls

- Source/build/output stay under `fixtures/grouped-j-integrated-20261015/` and `notes/grouped-j-integrated-20261015*`.
- Standalone CUDA fixture only; no engine build or source changes.
- CPU compile is allowed while P4 loads, with one compiler process.
- Do not initialize CUDA/run until root explicitly releases a GPU; then use only the assigned UUID and return it immediately after the bounded fixture.
