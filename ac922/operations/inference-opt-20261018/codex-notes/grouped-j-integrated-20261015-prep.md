# Integrated adaptive grouped MMQ fixture prep (2026-10-15)

Plan: `notes/grouped-j-integrated-20261015-plan.md`.

The standalone fixture includes the private adaptive prefix header from `src/ggml/src/ggml-cuda/mmq-moe-adaptive.cuh`, calls its `launch_mmq_moe_adaptive_offsets()` helper, and dispatches the original `mul_mat_q_moe_grouped` kernel template directly. Baseline and adaptive share the same global bounds, shuffled global ids, activation buffer, weight buffer, and output layout. Expected captured graph node counts are two for legacy prefix+J64 and three for adaptive prefix+J32 cold+J64 hot.

CPU-only SM70 compilation passed. The compiler reports only unused helper functions inherited from the previous fixture file. Linking attempted `libggml-base.so.0`, but `--as-needed` removed it: `ldd` and `nm -D` show no GGML or CUDA dynamic-library dependency. No CUDA context or GPU was initialized, and no engine source was modified.

GPU run is pending explicit release from root. Run only on the released GPU2 UUID:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
CUDA_VISIBLE_DEVICES=GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb \
  timeout 600s fixtures/grouped-j-integrated-20261015/grouped-j-integrated-fixture \
  fixtures/volta-tc-real/manifest.json \
  > notes/grouped-j-integrated-20261015-results.log 2>&1
```

The captured graph timing includes the device prefix scan and grouped MMQ kernels; it excludes host-side input creation and upload. No per-queue activation or ids copies occur in the measured path.

## Final run status

The bounded 21-case GPU2 run completed with prefix arrays and outputs exact; GPU2 returned idle. See `notes/grouped-j-integrated-20261015-results.md` and `.log` for per-case timings and the precise graph timing boundary.
