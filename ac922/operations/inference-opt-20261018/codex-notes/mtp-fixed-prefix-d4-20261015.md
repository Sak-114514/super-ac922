# Fixed-prefix D4 common gate and allocator check

## Change scope

`common/speculative.cpp` now selects the unrolled common depth through a small exact-value helper in `common/mtp-fixed-prefix-plan.h`:

- unset `LLAMA_MTP_GPU_CHAIN_UNROLLED_DEPTH` or exact value `2` selects the existing D2 path;
- exact value `4` permits common depth 4;
- any other value rejects the unrolled shape;
- the unrolled path still requires `LLAMA_MTP_GPU_CHAIN_UNROLLED=1`, four lanes, and `LLAMA_MTP_FIXED_PREFIX_ROWS=6`;
- the selected depth must equal the current common depth. A mismatch uses existing fallback handling.

The root-owned fixed-prefix failure rollback remains intact. A padded batch cannot fall through to the legacy per-depth chain.

## Real allocator fixture

The CPU-only fixture uses the actual model vocabulary in `vocab_only=true` mode and the real `llama_kv_cache` allocator with all layer tensors filtered. It confirms no KV tensors or buffers are allocated. Each round:

1. Allocates four lanes with four real catch-up rows, one seed row, and one future pad row per lane (24 allocator rows total).
2. Snapshots the step-0 indices and causal mask, then removes the four synthetic future rows.
3. Allocates three accepted steps, each with four rows at `seed+1`, `seed+2`, and `seed+3`.
4. Checks that step 1 reuses only the released dummy slots; later steps have unique destinations and do not overwrite the prefix, seed, or earlier accepted rows.
5. Checks the per-step sequence maximum, contiguous cell count, compact used-cell range, and unchanged step-0 indices/output mask/causal mask.

This is allocator and mask evidence only; it does not execute the unrolled model graph or establish logits/semantic correctness.

## Verification

Pure helper/policy test:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -Isrc/common \
  fixtures/mtp-fixed-prefix-pad-20261015/planner-test.cpp \
  -o /tmp/mtp-fixed-prefix-planner-d4-test
/tmp/mtp-fixed-prefix-planner-d4-test
```

Result: PASS. It covers unset/default D2, explicit D2, exact D4, malformed depth values, plus the existing fixed-prefix planning cases.

The isolated fixture was compiled against the already frozen V16 runtime libraries; no engine build was run. `LD_LIBRARY_PATH=.../runtime-perf-v16/lib ldd ...` resolved `libllama`, `libggml-base`, `libggml`, `libggml-cpu`, and `libggml-cuda` from V16.

CPU-only run command:

```sh
CUDA_VISIBLE_DEVICES= \
LD_LIBRARY_PATH=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v16/lib \
OMP_NUM_THREADS=1 timeout 120s \
  ./mtp-fixed-prefix-pad-fixture \
  /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf
```

Result: PASS for 100 rounds. The first batch had 24 rows; accepted steps had four rows each; final allocator use was 3200/4096 cells, with `used_min=0`, `used_max_p1=3200`, and final position 799 for each lane. Step-0 indices and mask stayed stable. CUDA was hidden and no model weights were loaded.

Artifacts:

- `fixtures/mtp-fixed-prefix-pad-20261015/fixture.cpp`
- `fixtures/mtp-fixed-prefix-pad-20261015/cpu-only-d4.log`

Current source/fixture hashes:

- `src/common/speculative.cpp`: `77d57bc6345ce10fff66c9d7a010531941affc0d967ead77cf08be463c70da1e`
- `src/common/mtp-fixed-prefix-plan.h`: `f1ed247f538ee62ebd5a631fc132cfea8d850d0b1bfe900a8c3df72ac44c13f8`
- `fixtures/mtp-fixed-prefix-pad-20261015/fixture.cpp`: `be0f29886675baccdc31d44d2a13ff2027e4678959d811a8e8f4a17480d557a9`
- `fixtures/mtp-fixed-prefix-pad-20261015/planner-test.cpp`: `d1c6aa664bdbb7e1ca635b370d7e028d02d2d1f7e30d443426fcbf36aa6fb31d`
