# Common-side fixed-prefix D2 plan

## Scope

Modify only `common/speculative.cpp`, add `common/mtp-fixed-prefix-plan.h`, add its pure CPU test, and add this note. The helper maps the original catch-up+seed snapshot into four lane-major groups of six rows. It validates source continuity, sequence ownership, seed-only output flags, context bounds, and total row budget. Padding rows reference the seed source row and get monotonically increasing future positions; the caller copies the seed token and hidden vector from that source.

## Activation and fallback

The fixed planner is eligible only when both `LLAMA_MTP_GPU_CHAIN_UNROLLED=1` and `LLAMA_MTP_FIXED_PREFIX_ROWS=6` are set, and the existing route has exactly four lanes with common depth two. Any planner rejection leaves the original batch untouched and falls through to the existing catch-up decode plus seed-only chain path. Only a successful planner marks the batch as fused, sets each lane's `first_pos` to the actual contiguous prefix start, emits exactly 24 rows and four outputs, and enables unrolled D2.

The original catch-up+seed rows remain in `fused_replay_rows`; dummy rows exist only in the constructed padded batch. A failed begin restores the original snapshot through the existing fallback. If unrolled enablement fails after begin, the code aborts, removes the full padded prefix from `first_pos`, restores the original batch, and lets the generic path continue. The padded batch must never enter the old per-depth GPU chain.

## CPU gates

The independent helper test covers unequal 0/1/4/5-token prefixes, lane-major 24-row output, four seed-only outputs, seed-backed dummy rows, deterministic replay without source mutation, invalid gaps, duplicate positions/lanes, missing seed, extra future source rows, prefix >5, context overflow, wrong row count, and a 23-row budget.

## Implemented and checked

- `common/speculative.cpp` now activates this planner only for the exact `LLAMA_MTP_GPU_CHAIN_UNROLLED=1`, four-lane, depth-2 shape plus `LLAMA_MTP_FIXED_PREFIX_ROWS=6`.
- Planner success constructs lane-major 24-row batches with four seed logits. It keeps the original catch-up+seed snapshot for rollback. If the unrolled graph is unavailable after chain begin, it aborts, removes the full prefix from each real `first_pos`, restores the original batch, and falls through to generic MTP; padded rows never enter the old per-depth chain.
- Local and remote pure CPU test command passed:

  `g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic mtp-fixed-prefix-plan-test.cpp -o /tmp/mtp-fixed-prefix-plan-test && /tmp/mtp-fixed-prefix-plan-test`

  Output: `PASS fixed prefix planner: unequal prefixes 0..5, seed-only outputs, source replay, gap/seed/overflow/budget/duplicate/extra-row rejection`.
- No engine build, model load, shared-tree edit, or GPU run was performed for this implementation slice.
- Final local/remote hashes match: `speculative.cpp` `46add0038101ba54d082a0d61b23f57e9c3844f0d65968b4dfa5860c7b376474`; `mtp-fixed-prefix-plan.h` `8a65af8645f4a95a696b2cbb526df4cac3017a2012043e7539cd8fc62ec70d5d`; test `4688fb5bbeede5f2b2c3e766f82ddfc4eebb4461cac8b185edd8016bddbc54ee`.
