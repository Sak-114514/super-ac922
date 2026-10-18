# Decode executor next-step audit

Date: 2026-10-15
Scope: read-only source and V9b/V10 artifact review. No source edits, builds, GPU launches, or model requests were made for this audit.

## Observed path

V9b (`results/20261015T044252Z`) reports 33.31 shared decode tok/s at one stream and 120.617 aggregate tok/s at four streams. Its server log records both `multi-lane GPU MTP chain completed lanes=4` and `lanes=1`; its CLI semantic run reports 16 pass, 0 fail. The V9b launch had the MTP family cache disabled.

The GPU chain is not a single depth-unrolled CUDA graph. `common_speculative_impl_draft_mtp::draft()` loops over `common_depth` and invokes `llama_decode(ctx_dft, batch)` once per depth. For an eligible chain, each invocation reads the previous step’s device-resident token/hidden carry and writes token history and the next hidden carry on device. `llama_end_mtp_gpu_chain_multi()` synchronizes and reads the small token history once at the end of the draft call so the server can construct its target verification batch.

The server’s current round has these boundaries:

1. **Target verify:** the server batches the previous sampled token plus draft tokens and submits `llama_decode(ctx_tgt, batch_dec)`. When outputs exist, `server-context.cpp` explicitly synchronizes the target context. The target graph has queued argmax IDs and nextn hidden rows for host access.
2. **Speculative process and acceptance:** `common_speculative_process()` reads target nextn hidden rows into `verify_h`; acceptance chooses the accepted prefix and copies the selected hidden row to `pending_h`. Accepted deferred rows and their hidden values remain in host-side per-sequence vectors.
3. **Draft catch-up:** `draft()` appends deferred accepted rows, then the current seed row. It extracts and separately decodes the catch-up rows with one `llama_decode(ctx_dft, batch)`.
4. **Chain begin and draft:** `llama_begin_mtp_gpu_chain_multi()` synchronizes the draft context before resetting the device history cursor. The chain then submits one graph decode per depth; token sampling inside eligible depth steps stays on GPU.
5. **Draft return:** `llama_end_mtp_gpu_chain_multi()` performs the final draft-context synchronization and token-history readback. The server consumes those proposals in the next target verification batch.

This means the eligible chain removes per-depth CPU token/logit sampling, but the target/draft boundary remains host-mediated. Target nextn hidden rows travel GPU→host; the selected seed hidden row is copied into the host `batch.embd` and uploaded for draft. Draft token IDs travel GPU→host once per draft call so the target batch can be built. Those are separate from the chain’s per-depth device feedback.

## Source evidence

- Draft catch-up extraction, separate catch-up decode, chain seed, per-depth `llama_decode`, and end readback: `src/common/speculative.cpp:1986–2159`.
- Multi-chain begin’s synchronize and end’s history transfer: `src/src/llama-context.cpp:1834–1947`.
- Target nextn hidden async copy and MTP chain hidden-feedback handling: `src/src/llama-context.cpp:3374–3395`.
- MTP chain argmax/token-history/step update and hidden feedback: `src/src/llama-graph.cpp:4267–4302`.
- Target decode’s output-boundary synchronize and speculative processing/acceptance: `src/tools/server/server-context.cpp:4745–4757`, `4846–4862`, `5031–5115`.
- Exact greedy sample fast path and masked fallback: `src/common/sampling.cpp:622–655`, `807–889`.
- Async graph submission and optional `wall-event` instrumentation: `src/src/llama-context.cpp:46–68`, `1225–1263`, `2702–2707`.

V10 (`results/20261015T051341Z`) logs `full-step sample ... path=cpu_greedy_masked_unique` for target output rows even with `LLAMA_GREEDY_SUPPRESSION_FAST=1`; those samples are about 0.44–0.47 ms, with first rows around 0.83 ms. The current sampler takes the GPU-argmax branch only when there is no exclusion bias and no suppressed-token list. With exclusion bias, it calls `llama_get_logits_ith()` and uses the CPU masked-argmax path. That accessor flushes any deferred logits tensor to host. This is a bounded candidate for later only if the V10 measurements show sampling/logit staging is material: for pure `-inf` exclusion masks, first test the already-computed raw GPU argmax token against the mask and return it when allowed; transfer/scan full logits only when the raw top candidate is excluded. Keep the existing CPU path for additive biases, ties requiring different ordering, or any unsupported sampler state. Validate exact token parity and tie behavior before enabling.

## Bounded next change: fuse catch-up rows with chain step zero

The current branch often logs 6–8 catch-up rows, then submits a separate seed graph. The existing lane-layout helper already models step zero as rows from each lane’s `first_pos` through `seed_pos`, with output enabled only on the final seed row. The caller currently passes `first_pos == seed_pos`, and `llama_context::decode()` currently enforces one input row per lane for all chain steps.

A narrow candidate is to fuse the MTP catch-up prefix and seed into the first chain decode:

1. Preserve the existing `batch` containing accepted deferred rows followed by the seed row. Set each lane’s `first_pos` to its first deferred position, or to `seed_pos` when it has no catch-up rows.
2. Permit variable-length step-zero prefixes in the chain layout check. Keep later chain steps at exactly one row per lane. Keep the existing checks for lane IDs, contiguous positions, one output per lane on the seed row, ubatch limit, and MTP KV bounds.
3. Because `llama_ubatch` has a uniform `n_seq_tokens` shape (`n_tokens = n_seq_tokens * n_seqs`), only fuse when every active lane has the same catch-up row count. For unequal prefixes, keep the current separate catch-up path.
4. If chain setup declines before decoding, pass the untouched catch-up-plus-seed batch to the generic path. If step zero or later chain execution fails, remove MTP memory from `first_pos` (not merely `seed_pos`) and replay the saved original batch through fallback.
5. Compare V10 baseline against the fused path with exact draft token IDs, accepted-prefix lengths, target token IDs, and `step-timing` rebuild/CUDA-graph capture counts. If varying catch-up row counts cause graph-family churn or the per-step graph count grows, reject the change. Keep P5 family caching disabled.

This removes one MTP `llama_decode` submission and its following begin-time wait for eligible equal-prefix lanes; it does not remove target verification, the final draft-history readback, or cross-context hidden movement. V10’s `wall-event graph_compute`/`synchronize` spans should quantify the actual saved wall time. The change is bounded and reuses the existing MTP chain and row-validation path; it does not require a persistent executor.

## Persistent executor assessment

A persistent cross-context executor could remove both the draft token-history readback and target-hidden GPU→host→GPU transfer. The current separate schedulers do not expose a shared carry tensor contract: target and draft graph/result lifetimes differ, acceptance is determined after target argmax, and draft KV positions must be committed in order. The existing device-side token/hidden feedback is only within the MTP context. A true executor would need a stable owner for shared carry tensors, target-side accepted-prefix selection compatible with sampler semantics, and explicit target/draft KV ordering.

Unrolling depth into one graph is likewise not a switch on the current chain. The present implementation submits one scheduler decode per depth, with recurrent token/hidden and KV state updates between calls. A fixed-depth graph would need repeated MTP subgraphs and explicit dependencies for token embedding, nextn hidden feedback, and per-position KV writes. Treat it as a separate design after the bounded catch-up/seed merge and V10 phase timings; do not re-enable P5 graph-family caching to approximate it.
