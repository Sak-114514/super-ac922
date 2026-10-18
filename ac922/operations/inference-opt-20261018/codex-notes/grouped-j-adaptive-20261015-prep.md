# Adaptive grouped MMQ fixture prep status (2026-10-15)

Plan: `notes/grouped-j-adaptive-20261015-plan.md`.

Runner: `fixtures/grouped-j-adaptive-20261015/grouped-j-adaptive-fixture.cu`; standalone ppc64le binary: `fixtures/grouped-j-adaptive-20261015/grouped-j-adaptive-fixture`. The runner reuses the existing compact grouped MMQ kernel and packed-row manifest, compares J64-all against two queues (J32 for experts with <=32 assignments; J64 for >32), and reorders one shared deterministic activation set by global token IDs. Scale generation uses random FP32 mantissas.

CPU-only compile completed for SM70. No model load, engine edit/build, CUDA initialization, or GPU test has been performed. Root owns all GPUs until explicit release.

After release, run only on GPU2 UUID:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
CUDA_VISIBLE_DEVICES=GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb \
  timeout 300s fixtures/grouped-j-adaptive-20261015/grouped-j-adaptive-fixture \
  fixtures/volta-tc-real/manifest.json \
  > notes/grouped-j-adaptive-20261015-results.log 2>&1
```

The fixture compares full output exactly before timing and reports baseline one-node J64 vs adaptive two-node J32-cold/J64-hot graph times. CPU route classification, quantization, and prefix building are outside the timed graph; this is a local kernel-scheduling candidate, not a model or end-to-end result.

## Final result

The bounded run completed on released GPU2 and returned it idle afterward. See `notes/grouped-j-adaptive-20261015-results.md` for the exact kernel-only speedups, equality evidence, and the explicit exclusion of route/queue preparation and copies.
