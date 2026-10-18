# MTP catch-up / seed fusion implementation

Status: opt-in source candidate, not runtime validated. `LLAMA_MTP_FUSE_CATCHUP_SEED` defaults off. The existing `LLAMA_MTP_GPU_CHAIN_SINGLE` default and parent changes were preserved.

## Behavior

- The opt-in candidate snapshots the original catch-up-plus-seed input rows, including token, position, sequence, output flag, and hidden embedding values.
- Fusion is attempted only when there is at least one catch-up row, every active lane has the same positive catch-up count, catch-up positions are contiguous and end immediately before the seed, only seed rows request output, and the fused input fits both the draft `n_ubatch` and `n_batch` limits. Existing catch-up caps and lane/depth/context guards still apply.
- A valid fused batch is lane-major. The chain validator accepts its rectangular multi-row prefix only for the first chain decode; subsequent chain decodes remain one row per lane.
- Unequal or unsupported prefixes use the prior separate catch-up decode plus seed decode.
- If chain begin declines before a fused decode, the untouched snapshot is replayed by the generic path. If any chain decode or token-history readback fails, the draft KV is removed from each lane's earliest fused position and the exact input snapshot is restored for generic replay. Non-fused cleanup retains the seed-position rollback boundary.
- A bounded opt-in diagnostic records fusion eligibility/fallback for the first eight observations and then every 128th observation. When the opt-in path compacts to seed-only rows for generic fallback, it refreshes each seed output row index.

## Changes

- `src/common/speculative.cpp`: new default-off environment gate, uniform-prefix candidate construction, exact batch snapshot/replay, first-position rollback, and bounded status logging. With the opt-in enabled, seed row indices are refreshed when a compacted batch is replayed so generic fallback samples the actual seed output row.
- `src/src/llama-context.cpp`: relaxes only the first active chain decode shape from one token per lane to the validated uniform prefix shape.
- `src/src/llama-mtp-chain-layout.h`: uniform-prefix, replay-snapshot, and rollback-position helpers; step-zero validation comment updated.
- `fixtures/mtp-catchup-seed-fusion-fixture.cpp`: bounded CPU regression for shape, output ownership, fallback, rollback, and snapshot replay.
- `notes/MTP-CATCHUP-SEED-FUSION-plan.md`: plan recorded before source edits.

## Verification

- New CPU fixture: PASS. It covers equal per-lane catch-up lengths 0 through 8 across three lanes, exact output flags and sequence mapping, unequal-prefix fallback, ubatch/catch-up-cap rejection, first-position rollback, and snapshot replay after begin rejection, step-zero error, later-step error, and history-readback failure.
- Existing MTP chain layout CPU fixture: PASS.
- `-fsyntax-only` with the frozen V10 compile flags: `src/common/speculative.cpp` PASS; `src/src/llama-context.cpp` PASS. The latter prints existing signedness warnings in unrelated nearby code.
- No full engine build, GPU fixture, model load, or model request was run. CUDA graph output-row semantics and full fallback correctness remain unverified; keep the opt-in flag off until the parent runs the bounded runtime gate.
