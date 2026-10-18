# `GGML_META_P2P_COLD` path audit

Date: 2026-10-15  
Scope: read-only audit of Meta3 P2P fallback, lifecycle, and existing synthetic fixture. No source edits, builds, GPU work, or model requests were made.

## Short answer

`GGML_META_P2P_COLD=1` does **not** skip first-shape deferral, force a graph to merge, disable merge, or change `graph_ready`. It does not replace `cgraphs_merged` or transfer ownership of graph tensors. It opts the unmerged/fallback collective path into a direct, rank-ordered P2P AllReduce callback.

For a graph that successfully uses merged in-graph P2P, the callback is not reached. A graph whose first-seen shape is deferred, whose merge is unsupported, or whose merge attempt fails uses the fallback callback; with this flag set, that callback tries direct P2P before NCCL or the generic butterfly. The actual direct collective is launched outside a synthetic graph node and synchronizes all child streams before it returns. It therefore changes the implementation and synchronization of fallback collectives, while leaving the surrounding subgraph graph-capture decision and pointers alone.

## Call path and lifecycle

1. Meta still constructs the communicator and P2P state. The constructor reads `GGML_META_P2P_COLD` once at backend initialization; changing the environment later has no effect on existing Meta contexts. If `p2p_ar` initialization succeeds and the flag is set, it resolves and stores `ggml_backend_p2p_ar_allreduce` in `p2p_ar_allreduce`. Otherwise the field remains null. See `src/ggml/src/ggml-backend-meta.cpp:2617-2647`.
2. First use of a new graph shape still sets `defer` and suppresses the merge attempt; `try_merge` continues to be driven by the existing `seen_shapes` and `merge_tried` state. The cold flag is not read by this code. See `ggml-backend-meta.cpp:3348-3369`.
3. If `cgraphs_merged` is nonempty, the merged-graph branch dispatches it and returns before the unmerged AllReduce loop. The cold callback is not used for that graph. See `ggml-backend-meta.cpp:3524-3590`.
4. Otherwise the unmerged path submits each simple backend's subgraph, gathers its partial tensors, then tries `p2p_ar_allreduce`; if it returns false, it tries the configured communicator and finally the generic copy/add reduction. See `ggml-backend-meta.cpp:3747-3788`.
5. The CUDA callback validates matching F32, contiguous rank tensors, reserves P2P staging, creates temporary stack tensor descriptors, writes their AllReduce params, and launches one kernel per rank on each child stream. It enqueues all rank kernels before synchronizing the streams. See `src/ggml/src/ggml-cuda/ggml-cuda.cu:1832-1876`.
6. The P2P state and staging remain owned by the Meta context. If `reserve` grows the stage, it synchronizes every device before freeing/reallocating the buffers. See `allreduce-p2p.cu:351-367`; the Meta context frees the P2P state at teardown.

The direct descriptors are stack temporaries used to populate launch parameters; they are not inserted into or retained by `cgraphs_merged`. The direct path does not change the lifetime of the original partial tensors. CUDA child-graph capture is completed inside the child graph-compute call before the Meta fallback loop invokes this callback (`src/ggml/src/ggml-cuda/ggml-cuda.cu:7230-7263, 7288-7365`). Thus the direct collective is outside the child cgraph/Meta merged graph, while surrounding subgraphs retain their normal CUDA graph warmup/capture path. Because it synchronizes each child stream after enqueueing the ranks, it adds a host completion boundary for every fallback collective. That is the main performance risk.

## Difference from `GGML_CUDA_NCCL_F32=1`

These switches affect different parts of the fallback:

- `GGML_CUDA_NCCL_F32=1` keeps the NCCL path but uses F32 wire reduction instead of its large-tensor BF16 scratch path.
- `GGML_META_P2P_COLD=1` tries the direct F32 P2P kernel before the communicator. If successful, NCCL is bypassed for that fallback collective regardless of the NCCL datatype setting; if declined, the communicator still runs.

Neither flag changes merged in-graph P2P AllReduce, which is F32 and graph-captured.

## Existing Meta3 fixture

`fixtures/meta-ar-sequence.cu` can exercise the cold fallback without a source edit. It creates a three-CUDA-member Meta device, four dependent reduction boundaries, and eight calls alternating 16 KiB and 64 KiB shapes. It injects different rank stream delays and checks all intermediate outputs and final partials.

The fixture's mode validation accepts `GGML_CUDA_ALLREDUCE=none` with P2P enabled as `graph-P2P`; it does not reject an additional `GGML_META_P2P_COLD=1`. With that combination, first-use/unmerged collectives exercise the cold callback and repeated eligible shapes can still take the normal merged path. The fixture therefore covers the mixed lifecycle; it does not isolate cold fallback for every call. It is a correctness smoke, not a representative large-prefill benchmark: its shapes are only 16/64 KiB and its optional 64 KiB ring threshold is fixture-specific.

The compiled fixture is already present at `/tmp/meta-ar-sequence` (read-only file check). After exporting `CUDA_VISIBLE_DEVICES` for the intended three-GPU island and `LD_LIBRARY_PATH` as described in `fixtures/meta-ar-sequence.md`, a bounded fresh-process run can use:

```bash
taskset -c 0 env \
  GGML_CUDA_ALLREDUCE=none \
  GGML_META_P2P_COLD=1 \
  GGML_CUDA_P2P_AR_RING_MIN_BYTES=65536 \
  GGML_META_DEBUG=1 \
  /tmp/meta-ar-sequence
```

`validate_mode()` accepts this environment as graph-P2P; no fixture source change is needed. The eight calls alternate two shapes: the first call of each new shape uses the deferred/unmerged route and can exercise direct cold P2P, while repeated shapes can still use the merged route. `GGML_META_DEBUG=1` reports attempted merge results. Use CUDA API/graph-node tracing if direct launches must be distinguished from captured graph nodes. Avoid `GGML_CUDA_P2P_AR_DEBUG` unless needed: that debug option also lowers the kernel spin-abort threshold. This test proves route and arithmetic only; the 16/64 KiB fixture does not establish large-prefill performance.

## Existing evidence and rejection status

No checked-in log or note shows a run with `GGML_META_P2P_COLD=1`, and I found no explicit rejection of the flag. `notes/meta-ar-output-live-none.log` and `notes/meta-ar-output-live-nccl.log` contain passing fallback/NCCL reference runs, but they are not cold-P2P runs. Earlier `meta-ar-fallback*.log` files fail because the older fixture did not keep intermediates live; the updated fixture marks every watched intermediate as `GGML_OUTPUT`. The fixture README's “not run” status appears stale relative to those later reference logs, but none of them establishes cold-flag behavior.

## Decision boundary

This flag can test whether direct P2P is a useful fallback for graphs that already failed/delayed merge. It cannot make those graphs merge, and it is not a way to skip graph capture warmup. Keep it as a separate candidate from `GGML_CUDA_NCCL_F32=1`; compare fallback correctness and time only after V12's current work releases the GPU. The small Meta3 fixture can verify route and arithmetic, but it cannot establish prefill performance for 512 KiB-plus partials or long-context server graphs.
