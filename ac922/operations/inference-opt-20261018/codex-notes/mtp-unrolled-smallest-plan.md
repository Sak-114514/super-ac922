# Smallest safe MTP D2/D3 single-graph implementation plan

Date: 2026-10-15  
Scope: concrete integration plan for the private AC922 `standard-opt-20261014` source. This narrows the prior feasibility note to the first implementation slice. No engine source, build, model, or GPU work was done here.

## Decision

Use **one MTP graph with `D=2` and four fixed lanes** as the first implementation. Leave `D=3`, catch-up/fused prefix, sampling, and variable lane count on the current loop. The reported 4-lane server path is only about 131.5 shared tok/s, while the 2-bank V12 d16 runner reports 114.5 quad tok/s; an unrolled graph is justified as a bounded way to remove the per-depth CPU scheduling boundary, but does not reduce the MTP block FLOPs and has no measured gain yet.

The first implementation must replace only draft generation. Keep the existing target verification, acceptance, and rollback logic after the draft returns. It must not claim target acceptance as a numerical-parity test.

## Exact integration seams

| Concern | Existing seam | First implementation action |
| --- | --- | --- |
| Call loop / host sync | `common/speculative.cpp:2281-2300` rebuilds one batch per depth and calls `llama_decode()` each time. `llama-context.cpp:2492-2496` syncs before each later MTP graph because persistent carry/token slots are reused. | Replace only the eligible `D=2`, 4-lane loop with one internal `llama_decode_mtp_gpu_chain_unrolled(ctx, seed_batch, depth)` operation. Leave common acceptance/target verification after the call unchanged. Add a side-effect-free eligibility probe before entering it; fall back to current loop on any unsupported shape. |
| Graph allocation | `llama-context.cpp:2703-2739` resets one `llm_graph_result`, calls `model.build_graph()`, then allocates it once. `llm_graph_result::reset()` creates one `ggml_context` and `ggml_cgraph` at `llama-graph.cpp:1599-1634`. | Add one Qwen4exp-specific `build_mtp_unrolled_graph(params, steps)` path that uses one `res->ctx/gf`; do not call `model.build_graph()` repeatedly, and do not reset the result between depths. Each step appends a block to the same graph arena. |
| MTP body | `models/qwen4exp.cpp:626-822` builds one MTP block, writes `t_logits`/`t_h_nextn`, and currently writes hidden feedback to persistent buffers. | Factor the block body into an internal `build_mtp_step(inputs)` result `{logits, hidden_next}`. Step 0 consumes the seed batch's token/hidden inputs. Step 1 consumes step 0's `hidden_next` and exact GPU argmax token tensor directly. Keep the same model weights and same one-layer MTP block; this is depth repetition, not a new MTP architecture. |
| GPU token feedback | `llm_graph_context::build_argmax()` is called once after the architecture builder; its split-vocab path/tie behavior is in `llama-graph.cpp:4280-4297`. Persistent token/history feedback is at `4299-4334`. | Factor an internal helper that returns the same split-aware argmax tensor for any current logits. Intermediate argmax tensors stay internal and directly feed the next token embedding. After depth 1, emit one packed step-major token tensor as the only history output; set the completed chain count to exactly 2 once. No intermediate D2 CPU copies. |
| Positions and KV row reservation | The normal decode makes one memory context from the batch (`llama-context.cpp:3192-3200`), applies the current ubatch at `2570`, and advances memory contexts by `mctx->next()` at `3529`. `llama_kv_cache::prepare()` already accepts a vector of ubatches and sequentially assigns/reserves slot-info for them (`llama-kv-cache.cpp:755-809`); `llama_kv_cache_context` indexes those slots through `i_cur` (`2748-2771`). | For D2, create exactly two `llama_ubatch` descriptors of four rows each: depth 0 at `seed_pos[lane]`, depth 1 at `seed_pos[lane]+1`. Reserve both in one memory preparation so the second row cannot alias the first. Keep the two immutable per-step memory views alive through graph input setup and completion. Do not try to make one `mctx` current at both depths at once. |
| Graph inputs | `res->set_inputs(&ubatch)` passes the same outer ubatch to every input (`llama-graph.cpp:1637-1640`, call at `llama-context.cpp:2782`). `build_inp_pos()`/`build_attn_inp_kv_impl()` size positions, index arrays, and mask from that ubatch (`llama-graph.cpp:2911-2922`, `3278-3309`). | Register a depth-specific input group holding its own immutable `llama_ubatch` plus memory-view/slot-info identity. Its `set_input()` fills only that step's positions, K/V indices, and causal mask. Do not route both groups through generic `set_inputs(&seed_ubatch)`. Step 0's token/hidden remain external; step 1 uses graph SSA values, not dummy host token values. |
| K/V and attention dependency | `build_layer_attn()` expands `cpy_k/cpy_v` then reads cache K/V (`models/qwen4exp.cpp:2659-2704`, `llama-graph.cpp:3335-3359`). `set_input_k_idxs()` binds `sinfos[i_cur]` and asserts the ubatch matches it (`llama-kv-cache.cpp:1519-1532`). | Depth 0's cache-write node must dominate depth 1's cache-read node on every meta rank. Use the depth-specific slot view for each. Construct the depth-1 mask from its actual position and known prior cache contents. If current `ggml_set_rows` aliasing does not give a scheduler-visible dependency, introduce an explicit cache-version edge or a single-stream ordering primitive that is valid under Meta/CUDA Graph capture. Do not assume graph-list order alone proves the write/read hazard is safe. |
| Commit / failure | Current failure removes the draft KV suffix and restores the seed/catch-up rows (`common/speculative.cpp:2290-2297`). End-of-chain reads and validates history once (`llama-context.cpp:1989-2026`). | On graph failure, remove both reserved depth positions from each lane's seed position onward and fall back to generic MTP after restoring the original batch. On success, read the two-token-per-lane history once, validate every ID, then return to the unchanged target-verification code. |

## Smallest state/data flow

For lane `l` and fixed depth 2:

```text
input[l] = seed token[l], seed hidden[l], position[l]
(logits0[l], hidden1[l], kv1[l]) = MTP(input[l], KV before depth 0)
token1[l] = exact_existing_gpu_argmax(logits0[l])
(logits1[l], hidden2[l], kv2[l]) = MTP(token1[l], hidden1[l], position[l] + 1,
                                          KV after depth 0)
tokens_out[l] = [token1[l], exact_existing_gpu_argmax(logits1[l])]
```

`hidden1` and `token1` are tensor dependencies; they must never be copied to/reloaded from the mutable `slot.hidden` or `slot.tokens` buffers between depths. This removes the reason for the existing `process_ubatch()` synchronization. The K/V mutation remains shared state and therefore needs its own explicit ordering proof.

The new `llama-mtp-unrolled-layout.h` and `fixtures/mtp-unrolled-layout-fixture.cpp` define/test only logical mapping: `reservation_row = step * lanes + lane`; `step_input_row = lane`; `history_lane_major_row = lane * depth + step`; `position = seed_pos + step`. These helpers intentionally do **not** claim to compute real KV cell IDs. The memory preparation's `slot_info` remains authoritative for those IDs.

## Gates before any model run

1. Compile and run the new CPU layout fixture. It covers 4×D2, 2×D3 indexing, per-lane positions, step-major to lane-major token packing, catch-up rejection, bad sequence maps, depth bounds, position overflow, and undersized output.
2. Add a tiny single-graph backend fixture with synthetic MTP weights and four lanes. Compare sequential D2 and unrolled D2 exact argmax IDs; compare `hidden1/hidden2` against the existing path; deliberately poison the depth-1 KV slots before launch and prove the depth-0 write happens before depth-1 attention reads. Verify both target and draft KV positions after completion.
3. Exercise CUDA graph replay twice with changed seed IDs/positions, then an injected failure after the depth-0 write. Verify correct slot remapping, no stale external inputs, and rollback of both depths.
4. Only after the fixture gates pass: one full-model 4-lane D2 semantic run on the same prompt/runtime as the current path; exact token sequence plus target verification and KV-state checks. Then measure full-step wall time and count the removed inter-depth synchronizations. Keep the current implementation as automatic fallback.

## Expected scope and performance interpretation

This removes one `llama_decode()` call, one memory-context preparation, and one explicit scheduler sync for D2. It does not remove the second MTP block's compute or the target model's verification. The result can only be judged by direct full-step timing after exact sequence/cache parity. If the target-verification/API hold dominates after the draft graph is combined, this change alone cannot reach the 300 tok/s target.
