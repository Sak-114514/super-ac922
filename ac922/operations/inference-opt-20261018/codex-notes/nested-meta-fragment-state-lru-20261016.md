# Nested Meta3 fragment graph-state LRU: read-only feasibility

**Scope:** static review of current `ggml-backend-meta.cpp`, `ggml-meta-cache.h`, V24 launch/log artifacts, and CUDA graph cache semantics. No core edits, build, model load, or GPU use.

## Verdict

A per-inner-Meta3 graph-state LRU is a **smaller first candidate than full Meta2×Meta3 physical graph flattening**. It can keep the existing three physical backends, streams, P2P3 handle, and workers, while avoiding the current single-UID context from rebuilding the same outer fragment on every pass. It preserves the current numerical graph and outer P2P6 path; it does not remove the per-boundary P2P6 collective or the parent’s per-fragment submission.

The LRU must be selected **before** `needs_rebuild` is computed in `ggml_backend_meta_graph_compute()`. Saving only `cgraphs_merged`/CUDA graph keys after dispatch would be too late: `ctx`, per-backend nodes/cgraph descriptors, merged/chunk contexts, and owner leases are all changed on rebuild.

## Why nested traversal rebuilds the same child context

The live source has one active graph identity in `ggml_backend_meta_context`: `backend_ctx->uid`. `needs_rebuild` compares the incoming child cgraph UID to that one active value (`ggml-backend-meta.cpp:4461-4467`). During parent rebuild, each outer subgraph gets its own child `cgraph_ij->uid = ggml_graph_next_uid()` (`:4865-4880`), and the parent calls the same child Meta backend once for every outer subgraph (`:5472-5485`). After fragment `i` returns, the child’s active UID is `uid[i]`; the next fragment arrives with `uid[i+1]`, so it rebuilds again. On the next hot parent pass, the traversal starts at `uid[0]` while the child is still on `uid[last]`; without a cache, the UID mismatch repeats for each fragment.

`n_subgraphs` is the actual count to measure. The model has 48 layers, but delayed-AllReduce grouping and layouts mean the code must not assume `n_subgraphs == 48` without logging `backend_ctx->n_subgraphs` for the target graph.

Rebuild work includes clearing/cycling `tensor_arenas`, remapping every root node into `backend_configs[j].nodes`, recomputing split boundaries, and rebuilding the per-member cgraphs (`:4522-4552`, `:4566-4798`). P2P merge/capture state is also per active graph UID: `cgraphs_merged` is cleared and `merge_tried` reset when `needs_rebuild`; the shape set can defer the first shape, after which `try_merge` creates merged/chunk graphs (`:4911-4942`, `:4964-5068`). `seen_shapes` is context-wide and must stay shared so the “run a new shape once before merged collectives” policy remains once per shape, not once per LRU entry.

## `GGML_META_PACKED_CACHE` is unrelated to graph-state retention

`ggml-meta-cache.h:9-26` stores only `ggml_backend_meta_split_state` fields. `GGML_META_PACKED_CACHE` trims trailing zero entries from `state.ne`; it does not store cgraphs, context arenas, merge flags, UIDs, P2P handles, or CUDA graph readiness.

The cache key in `ggml-backend-meta.cpp:632-633` is `(tensor pointer, assume_sync)`, with a compact split-state value and descriptor signature. Buffer reset clears the non-frozen split-state cache (`:3060-3073`). This is a lossless metadata-size optimization, not an LRU of prepared graph state.

Likewise, `GGML_FT_GRAPH_CACHE` / `GGML_CUDA_GRAPH_STRUCTURAL` live below Meta graph construction. The structural path canonicalizes a physical CUDA cgraph after Meta has supplied it (`ggml-cuda.cu:4281-4287`). The CUDA variant cache holds captured `ggml_cuda_graph` instances, with a default four-variant cap per physical CUDA context (`:4482-4490`). It cannot restore `backend_ctx->uid`, `backend_configs[].nodes/cgraphs`, `ctx_merged`, `ctx_chunks`, or `merge_tried`. It can still be useful after a Meta LRU hit, but its own four-entry eviction means 48 Meta entries do not imply 48 retained CUDA captures.

## V24 evidence and its limit

The old flat-island V24ep run explicitly enabled both `GGML_META_TIMING=1` and `GGML_META_PACKED_CACHE=1` (`results/20261015T125200Z/launch.json:37-39`). Its server log reports:

| `meta-timing` context / graph size | Rebuilds in 200 calls | Average rebuild time per rebuilt call | Average dispatch per call |
|---|---:|---:|---:|
| `0x1002490c290`, 3,801 nodes | 17/200 | 23.216 ms | 35.621 ms |
| `0x100300c22e0`, 3,449 nodes | 16/200 | 20.515 ms | 63.627 ms |
| Later intervals for those contexts | 4/200 and 2/200; later 9/200 and 9/200 | about 19.568–25.144 ms | 14.685–39.078 ms |

Sources: `results/20261015T125200Z/server.log:5325-5326,6507-6508,7555,7569`. This is useful evidence that Meta graph rebuilds are material (roughly 20–25 ms when they occur) and that `GGML_META_TIMING` is already usable. The run was the **flat `LLAMA_ISLAND_SPLIT=3,3` path**, not the new hierarchical nested Meta2{Meta3,Meta3} path; it cannot establish the nested fragment rebuild rate.

The later V24 hierarchical launcher is explicitly “prepared only; no model or GPU was launched” (`fixtures/dual-context-v24-hierarchical-20261016/README.md:1-3`). Its `v24-exact-env.json` does not enable `GGML_META_TIMING`. Do not report a nested `rebuilt≈48/step` result from that preparation.

To confirm the nested case with the existing diagnostic, run the same nested graph with `GGML_META_TIMING=1`. It reports per-thread, per-Meta-context counts every 200 `graph_compute` calls (`ggml-backend-meta.cpp:4514-4517,5088-5100`). Because each Meta3 context is submitted from its stable parent execution lane, a child showing rebuilds near 200/200 while outer `n_subgraphs` is large is the expected signature of the one-slot UID thrash. Add a temporary one-line trace of `{ctx, incoming child uid, active uid, parent uid, fragment index, LRU hit/miss}` if pointer attribution is ambiguous; the existing summary alone does not identify which context pointer is island A/B.

## Per-context cache inventory

The current `ggml_backend_meta_context` combines long-lived physical-backend resources and one active graph generation (`:3364-3468`). The LRU should split those categories rather than copy or recreate the whole context.

| Member / state | LRU treatment | Reason / constraint |
|---|---|---|
| `backend_configs[j].backend` | **Context-global; never swap/recreate** | These are the same three CUDA backend contexts/streams. Reinitializing them would lose stream/capture/communicator identity and defeats the lightweight design. |
| `workers`, `submit_lanes`, `gpu_spans` and span callbacks | **Context-global** | One stable worker per non-main member; device submit lanes order graph cohorts across contexts. Span sequence may remain diagnostic-global. |
| `p2p_ar`, `comm_ctx`, callbacks, `n_reduce_steps` | **Context-global** | Synthetic AR3 node op_params point to the P2P3 handle. All entries share one communicator and its device flags/epochs. Do not put a new communicator in each entry. |
| `async_copies`, async-copy mutex/policy, CPU-copy limit | **Context-global** | These are backend/stream resources, not a property of one root graph UID. Any in-flight copy must obey the existing stream-tail drain before an entry switch. |
| `name`, `instance_token`, CPU async-copy options, `nccl_large_min_bytes`, timing counters | **Context-global configuration/diagnostics** | Not graph identity. Timing state is already keyed by context pointer and thread. |
| `backend_ctx->uid`, `n_subgraphs`, `replay_ready_uid`, `replay_inputs_uid`, `replay_unused_uid`, `replay_ready_at_us`, `replay_inputs_local` | **Per cached graph state** | These decide whether this specific cgraph can skip repartition/rebuild and whether its child CUDA graphs are still ready. Recheck `graph_ready` after the existing freshness lease expires. |
| `backend_configs[j].nodes`, `.cgraphs[]` (offsets, main and aux cgraphs), `ctx`, `cgraphs_aux`, `nodes_aux` | **Per cached graph state** | Tensor pointers and cgraph descriptors are tied to this UID's translated graph. They are rewritten/reused on rebuild. Package them with the context arena that allocated them. |
| `max_nnodes`, `max_subgraphs`, `graph_node_caps` | **Per state or explicit per-entry capacity record** | Must agree with the associated `ctx` and cgraph capacities. A global max is safe only if every entry's arena was allocated to that max. |
| `ctx_merged`, `cgraphs_merged`, `cgraphs_chunks`, `ctx_chunks`, `merge_tried` | **Per cached graph state** | These are exactly the inner P2P3 merged/chunk descriptors whose rebuild is being avoided. Preserve the first-shape `merge_tried=false` state; on a later hit the existing logic can merge. |
| `tensor_arenas`, `graph_owner_leases` | **Per cached graph state; retain strong references** | External-view descriptors and scheduler owner epochs must remain valid while cached cgraphs/captures refer to them. On owner retirement/epoch change, evict that entry. |
| `nccl_large_graph`, `nccl_large_plan_uid`, `nccl_large_boundaries` | **Per cached graph state or disallow that route** | If the external NCCL-large plan is selected, `cgraphs_merged` is deliberately empty. The first LRU version should cache only states with the expected in-graph P2P3 path; decline external NCCL states. `nccl_large_calls` can remain a context-wide counter. |
| `max_tmp_size`, `backend_configs[j].bufs[]` | **Context-global high-water scratch; not swapped** | Current buffer replacement on growth occurs at `:4800-4805`. Do not let an entry reference scratch that another entry reallocated. Cache only prepared P2P3 merged states; if this high-water grows, invalidate entries that use the old scratch or keep the buffers at the global max. |
| `seen_shapes` | **Context-global; keep shared** | It ensures a shape's first call is deferred before graph merging. Per-entry copies could make every layer fragment look like a new cold shape and reintroduce repeated unmerged calls. |

### Activation and key

Select/activate the LRU entry before the `needs_rebuild` comparison at `:4466-4467`. A robust key is `(Meta context instance_token, parent/root cgraph UID, outer fragment index, child fragment cgraph UID, backend/topology signature, owner epoch set)`. `cgraph_ij->uid` is generated per outer subgraph at `:4868-4880` and remains stable while the parent root graph is hot; root UID and fragment index make the ownership explicit and prevent accidental alias across parent graph generations.

On a hit, restore the entire graph-state bundle atomically, including the active `uid`, then let current `ggml_backend_meta_graph_compute()` take the no-rebuild path. On a miss, save the outgoing state, prepare the new one using the existing path, and insert it only after all cgraphs/owner references are complete. A capacity below the measured `n_subgraphs` working set will thrash every outer sweep; a smaller LRU is useful only if several fragments demonstrably share a retained state/signature.

## Shared communicator reserve and owner invalidation

- P2P3 is shared by all LRU entries. `p2p_ar_reserve()` can grow staging buffers; the kernel launch arguments captured into CUDA graphs include stage/flag/epoch device addresses (`allreduce-p2p.cu:384-446`). A reserve growth therefore invalidates **all** graph captures associated with that handle, not just the active entry. Prefer reserving the maximum across the fragment set before capturing any entry. If the max cannot be known up front, bump a communicator generation and invalidate/recapture every cached entry after growth; do not silently reactivate old captured graphs.
- Re-run/refresh each entry's synthetic AR node params after reserve growth because `p2p_ar_set_params` stores the current cap/handle/rank in `op_params`. Keep rank order and shape checks unchanged.
- Parent/owner epoch change: cache key must include owner IDs/epochs. Keep `owner_ref`s strong in each entry, but also evict if an owner has been explicitly retired/unbound or its backing tensor/buffer generation changes. Before releasing an entry with outstanding GPU work, drain that graph's physical streams; graph metadata may be freed only after the physical CUDA graph cache no longer depends on its raw node key or descriptor storage.
- Physical CUDA graph variants are a separate cache: default `GGML_CUDA_GRAPH_VARIANTS_CAP=4` (`ggml-cuda.cu:4482-4490`). The Meta LRU can keep graph descriptors alive and avoid `needs_rebuild`, but a physical capture can still be evicted and recaptured. Do not claim the LRU pins all 48 CUDA graph instances.

## Minimal next evidence

1. On the existing nested tiny/no-model path, log parent UID, `n_subgraphs`, child context pointer, incoming/active child UID, `needs_rebuild`, LRU hit/miss, `merge_tried`, and graph-state memory bytes. No model launch is needed to validate state transitions.
2. With a nested full-model request when root grants it, set `GGML_META_TIMING=1` and retain current semantic/exact gates. Expect the first traversal to populate states; a repeated same-root UID traversal should show child Meta3 LRU hits and `rebuilt` substantially below the no-LRU case. Compare actual `rebuilt`, `t_rebuild`, dispatch wall time, and entry memory; do not infer a speedup from the old flat V24 run.
3. Exercise root UID change, same-shaped new prompt, owner-epoch retirement, P2P capacity growth, and LRU eviction. On each case, assert fallback/rebuild/recapture rather than replaying an old graph. Keep outer P2P6 calls and output values exactly unchanged.

**Expected scope of benefit:** reduce repeated child graph partition/materialization/merge and associated CUDA graph setup when returning to cached fragment UIDs. It does not remove parent per-fragment graph submission or host P2P6 synchronization. The old V24 data proves rebuild cost can be material; the new nested path's rebuild rate and LRU speed benefit remain unmeasured.
