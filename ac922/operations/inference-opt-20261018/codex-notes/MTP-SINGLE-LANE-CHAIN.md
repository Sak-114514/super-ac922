# Single-active-lane MTP GPU chain extension

Date: 2026-10-15
Status: source and CPU policy fixture ready; GPU/operator and model verification pending.

## Scope

The existing opt-in `LLAMA_MTP_GPU_CHAIN_MULTI` path handled two or more active sequences, even when the server context was configured for four sequences. A single active request fell back to the generic draft loop, which samples after each draft decode and synchronizes to obtain the token. This change extends the existing chain slots to one active lane in a multi-sequence context. It does not unroll draft depth into a single graph, change defaults, or enable P5 graph-family caching.

## Changes

- `src/common/speculative.cpp`: allow the existing chain branch to run for one active drafting sequence, retaining the shared-depth, catch-up, greedy, suppression, backend-sampler, adaptive, and max-depth checks.
- `src/src/llama-context.cpp`: allocate and initialize slot 1, accept a one-lane begin/end, keep sequence IDs strictly increasing and below `n_seq_max`, and explicitly check backend support for one-lane and multi-lane `ggml_set_rows` shapes. If either shape is unsupported, chain configuration fails and the existing generic path remains available.
- `src/src/llama-mtp-chain-layout.h`: permit one-lane row validation/reordering and expose an internal pure sequence-ID validator used by the context entry point and CPU fixture.
- `fixtures/mtp-gpu-chain-layout-fixture.cpp`: cover one active sequence ID 3 in a four-sequence context, depths 1 and 4, zero/eight catch-up bounds, legacy two-to-four-lane row order, invalid sequence IDs and row maps, history capacity, and transient snapshot policy.

Original owned source copies are preserved in `notes/single-lane-mtp-before-v9/`.

## Verification

- CPU fixture compiled and passed with `c++ -std=c++17 -O0`; output: `PASS MTP GPU chain layout: lane1 seq_id=3/n_seq_max=4, depths1/4, catch-up0/8, legacy lanes2..4, invalid maps/capacity, transient snapshot policy`.
- `-fsyntax-only` passed for `src/src/llama-context.cpp` and `src/common/speculative.cpp` using the existing V7 compile flags. It reported four pre-existing signedness warnings in `llama-context.cpp`; no new diagnostics were emitted for these changes.
- No engine link/build, GPU execution, or model request was run. The GPU `set_rows` capability probe and single-lane token/history/hidden feedback remain unverified on the actual Meta/CUDA runtime.

## Next gate

After root releases GPU access, run a bounded single-stream parity case with `n_seq_max=4`, active sequence ID 3, depths 1 and 4, and catch-up counts 0 and 8. Confirm the `multi-lane GPU MTP chain completed lanes=1` log, exact draft token IDs against the generic path, and one final history readback per draft call. Then verify that a backend rejecting the one-lane `set_rows` shape declines configuration and falls back cleanly.
