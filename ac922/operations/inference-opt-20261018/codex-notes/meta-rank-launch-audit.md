# Meta three-rank graph launch audit

Date: 2026-10-15  
Scope: read-only source contract and V11 full-trace summary review. No source edits, builds, GPU work, or model requests were made.

## Finding

There is no single launch mode for all Meta graphs. The current implementation has two paths:

1. **Merged in-graph P2P AllReduce path:** one merged graph (or ordered graph chunks) per simple CUDA backend. For three ranks, rank 1 and rank 2 are posted to two persistent worker threads, while rank 0 launches on the caller thread. The launches therefore fan out concurrently. This is not a serial three-rank launch on one CPU thread.
2. **Unmerged/subgraph fallback path:** each rank's subgraph is submitted in a `for (rank=0..n_backends-1)` loop on the caller thread, then the collective runs separately. That submission is host-serial. CUDA submission is asynchronous, so the source alone does not imply that each GPU stays idle until its turn.

The V11 trace proves that the merged P2P path ran during the profiled workload: the `k_allreduce_p2p_seg` kernels are generated for `GGML_OP_ALLREDUCE`, which the Meta merge code inserts into the rank graphs. The optional cold-P2P API that can call the same CUDA kernel outside a merged graph is gated by `GGML_META_P2P_COLD`; it is absent from the V11 environment. The trace also contains NCCL AllReduce kernels, consistent with unmerged or first-shape fallback collectives. The exported summary does not show the exact graph/chunk/lockstep selection per invocation because `GGML_META_DEBUG` was not enabled.

## Source contract

- The Meta context creates one `ggml_backend_meta_worker` for every simple backend after rank 0 when P2P AllReduce initializes successfully. With three CUDA members, that is two workers plus the caller. See `ggml-backend-meta.cpp:2534-2575, 2615-2638`.
- A new graph shape is deliberately deferred once and first runs unmerged, because lazy driver work can block while a peer GPU is spinning in an AllReduce. Repeated shapes can attempt merge. Merge also requires more than one subgraph, contiguous F32 partials, and reserved P2P workspace. See `ggml-backend-meta.cpp:2572-2575, 3341-3387`.
- The merge inserts one `GGML_OP_ALLREDUCE` after each eligible partial and rewires later nodes to read the reduced tensor. The CUDA kernel thresholds large partials into the segmented implementation. See `ggml-backend-meta.cpp:3388-3445`; `allreduce-p2p.cu:19-26, 116-139, 228-280`.
- Merged graphs are divided at AllReduce boundaries and, by default, around 512 nodes. If a graph has more than one chunk and any chunk is not ready for plain replay, Meta uses a lockstep route: enqueue chunk `k` on all ranks, wait for their host submissions, synchronize every backend, then advance to `k+1`. Otherwise it posts ranks 1..N-1 to worker threads and submits rank 0 on the caller, preserving chunk order per rank. See `ggml-backend-meta.cpp:3450-3493, 3524-3589`.
- If there is no merged graph, the unmerged path submits each rank graph serially on the caller and then invokes the communicator or fallback reduction. See `ggml-backend-meta.cpp:3738-3779`.
- The segmented AllReduce is a real spin barrier: each rank writes data and a peer flag, polls a volatile flag until every peer has arrived, then reduces/copies its segment. Each of its `2*n` rounds has this barrier. A delayed host launch or delayed graph chunk can therefore inflate the observed kernel duration without implying that NVLink bandwidth is the cause. See `allreduce-p2p.cu:140-224`.

## V11 evidence and limits

Artifacts: `results/v11-full-trace-20261015T052734Z/{prefill-trace-summary.json,decode-trace-summary.json,runtime-proof.json,prepared.json}` and server log referenced by `phase-window.json`.

- V11 environment has `GGML_META_ASYNC_COPY=1`, `GGML_META_FIFO_PIPELINE=1`, and `GGML_META_GPU_EVENT_WAIT=1` already enabled. `GGML_META_P2P_COLD` is absent; `GGML_META_CHUNK_NODES` is absent and therefore defaults to 512. `GGML_META_DEBUG` is absent, so exact per-invocation merge/chunk/lockstep outcomes are not logged.
- Prefill summary: 52,449.48 ms window; 5,334 segmented P2P AllReduce kernel events with 11,359.20 ms summed kernel duration across devices (about 2.13 ms per event); 2,157 NCCL AllReduce events with 3,348.98 ms summed duration. These are aggregate per-device kernel sums, not critical-path wall time.
- Presence of the segmented P2P kernel proves that at least some prefill graphs used the merged in-graph path. NCCL events show other collectives also occurred, but the summary alone cannot count which graphs used each path.
- The 2.13 ms P2P kernel average does not measure link-only time and does not establish CPU launch skew. The kernel intentionally waits for all peer flags. The trace summary lacks a rank-by-rank matched-collective launch-skew table, so “mostly spinning on a late host launch” remains a plausible, unconfirmed explanation.

## Existing toggles that are not rank fanout controls

- `GGML_META_ASYNC_COPY=1` allows a supported mirrored activation copy between the two three-GPU Meta islands to use the CUDA helper; it does not schedule the three ranks of a graph.
- `GGML_META_FIFO_PIPELINE=1` provides per-member Meta events; paired with `GGML_META_GPU_EVENT_WAIT=1`, cross-Meta dependencies can become CUDA stream waits instead of host synchronization. These affect dependencies/copies, not the rank-graph executor.
- `LLAMA_PREFILL_FAIR_SHARE=1` affects request scheduling. It is not a switch for Meta rank fanout.
- No separate `PREFILL_PARALLEL` or `META_FANOUT` switch was found in the private source. Merged-graph fanout is already automatic when merge prerequisites pass.

## Bounded next step

Before changing the executor, correlate existing V11 CUDA API/node events for each matched `allreduce_N` and chunk across the three devices. Record `cudaGraphLaunch` host-call intervals and each rank's P2P kernel start/end; use the max-minus-min kernel start time across ranks as launch skew. Compare that skew with the 2.13 ms kernel duration. Do not infer skew from summed per-GPU duration.

If the matched trace shows material launch skew, the narrow candidate is a generation-tagged start gate around the existing per-rank worker jobs in the **merged, non-lockstep** branch. Reuse the two persistent workers and caller thread, release them together per chunk, and keep one ordered job per rank. Do not change graph construction, tensor descriptors, CUDA streams, P2P epoch counters, or lockstep synchronizations. Leave the lockstep route unchanged: it already enqueues all ranks for a chunk and synchronizes all devices before moving to the next chunk.

Risks to guard in any later implementation:

- P2P AllReduce ranks must execute identical graph/chunk and collective order; a missing or reordered rank leaves device threads polling peer flags.
- A start gate can reduce host wake skew but cannot remove blocking CUDA capture/instantiate work or network/device latency. It may add a small host barrier cost.
- The worker path uses separate per-rank backends and status slots; preserve those ownership boundaries and make the gate generation-scoped so one compute cannot release another compute's jobs.
- First execution for a new shape intentionally remains unmerged to avoid lazy-driver deadlocks. Do not remove that fallback based on the steady-state profile.
