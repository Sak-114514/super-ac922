# BF16 NCCL versus in-graph P2P AllReduce audit

Date: 2026-10-15  
Scope: read-only source and V11 trace-summary audit. No source edits, builds, GPU work, or model requests were made.

## Answer

The V11 `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` events do **not** by themselves mean a graph contains BF16 partial tensors, and they do **not** make a graph ineligible for the in-graph P2P merge. In the current NCCL fallback, a large F32 partial is converted into a BF16 scratch buffer, reduced by NCCL as BF16, then converted back to F32. The Meta merge eligibility check sees the original partial tensor and requires that it be F32, contiguous, and have non-null data.

Therefore, `P2P_AR=1` and BF16 NCCL events can coexist: P2P is selected for eligible graph shapes; other shapes use the unmerged communicator path. The BF16 NCCL wire format is selected after that fallback path is chosen. It is not the reason the merge gate rejects a graph.

An actual non-F32 partial tensor would fail the P2P merge type gate. The available V11 summary does not identify such a tensor, and the NCCL BF16 kernel name is insufficient evidence for one.

## Source findings

### Meta graph merge gate

In `src/ggml/src/ggml-backend-meta.cpp:3348-3396`, P2P merge is attempted only when a P2P context exists, the graph has more than one subgraph, and every inter-subgraph partial on every rank passes:

- `p->type == GGML_TYPE_F32`
- `ggml_is_contiguous(p)`
- `p->data != nullptr`
- P2P staging reservation succeeds

A newly seen graph shape is deliberately deferred once before merge (`:3352-3369`). A single failed partial rejects the merged graph for that compute; it then uses the separate subgraph/communicator path. The merge inserts `GGML_OP_ALLREDUCE` with the partial's original type (`:3421-3441`). CUDA device support accepts this op only for contiguous F32 (`src/ggml/src/ggml-cuda/ggml-cuda.cu:8589-8590`).

### In-graph P2P data type

`src/ggml/src/ggml-cuda/allreduce-p2p.cu:241-283` asserts a contiguous F32 tensor and uses `float *` staging. The ring threshold defaults to 4 MiB and only chooses the segmented F32 kernel versus the one-shot F32 kernel (`:228-280`). The backing buffers, stage capacity, and kernel arguments are also typed in floats (`:38-58, 351-378`).

I found no BF16 P2P flag or BF16 P2P implementation in this source. Existing P2P controls include disabling P2P and choosing the F32 one-shot/segmented size threshold; the NCCL force-F32 flag is separate.

### NCCL fallback type and threshold

`src/ggml/src/ggml-cuda/ggml-cuda.cu:1586-1667` implements the current fallback:

- Small/exact-row reductions use `ncclFloat` directly (`:1606-1629`).
- Large reductions allocate `nv_bfloat16` scratch, convert each F32 input into that scratch, call `ncclAllReduce(..., ncclBfloat16, ncclSum, ...)`, then convert the result back to F32 (`:1632-1664`).
- `GGML_CUDA_NCCL_F32=1` forces the direct-F32 path. For three ranks, the normal threshold is `ne < 131072` for direct F32; at or above that size the BF16-scratch path is selected unless `exact_rows` applies (`:1606-1615`).

This code treats `tensors[i]->data` as float input before creating BF16 scratch. Thus the NCCL kernel's datatype identifies the communication scratch, not necessarily the ggml partial tensor type. The V11 environment has `GGML_CUDA_ALLREDUCE=nccl`; `GGML_CUDA_NCCL_F32` is unset.

## V11 trace evidence

Artifacts: `results/v11-full-trace-20261015T052734Z/{prefill-trace-summary.json,decode-trace-summary.json,runtime-proof.json}`.

- Prefill window: 52,449.48 ms.
- NCCL: 2,157 kernels, 3,349.40 ms summed duration across devices. The top kernel is `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, 2,154 kernels and 3,348.98 ms.
- Segmented P2P: 5,334 kernels, 11,359.20 ms summed duration across devices.
- Decode NCCL: 1,191 kernels, 37.07 ms summed duration.
- `GGML_CUDA_NCCL_F32` is unset; `GGML_CUDA_ALLREDUCE=nccl` is set.

The per-device kernel sums are **not** the critical-path wall time and cannot be used to infer a throughput gain. The trace summaries do not associate each NCCL kernel with its source graph, partial tensor shape, or merge rejection reason.

## What can explain P2P and NCCL together

The Meta context initializes both the P2P machinery and the communicator. It uses the merged in-graph route only when the graph-level eligibility checks pass. Otherwise, the unmerged route runs rank subgraphs and calls the configured communicator. The observed mixture is therefore expected without any BF16-native partials. Other merge rejection causes include first-seen-shape defer, `n_sub <= 1`, noncontiguous/no-data partials, and failed staging reservation. V11 ran without `GGML_META_DEBUG`, so the current artifacts do not show which cause produced each NCCL call.

## BF16 P2P candidate assessment

A BF16-wire P2P path is technically possible but not a flag-only change. It would need typed staging buffers and capacity bookkeeping separate from the existing F32 `stage`/`cap`, an operation dispatch that can represent the type and output, and a safe graph signature when buffers grow. The existing segmented path uses separate flag cells from its one-shot path because their round schemes differ; a new BF16/round scheme must avoid aliasing flag locations and preserve the shared per-block epoch ordering. All ranks must choose the same dtype, size bucket, and collective order.

A sketch such as “convert each F32 partial to BF16 for transport, accumulate received values in FP32, optionally round the sum to BF16, then convert back to F32” is **not proven equivalent** to current NCCL BF16 behavior. NCCL's BF16 sum may have a different accumulation and rounding path than an F32 sum over rounded inputs. Do not use the in-graph F32 P2P bitwise result as the only oracle, and do not claim exact parity from the dtype label.

Graph-local casts would also alter graph types and lifetime: the Meta split-state and AllReduce nodes would have to keep the reduced output mirrored and correctly reconnect downstream F32 consumers. A cast-only graph experiment could add conversion launches without reducing transport unless the AllReduce kernel consumes and stages BF16 directly.

## Bounded next step before a kernel change

First add or enable a bounded merge-reason trace for a short repeat of a known prefill shape. Record per Meta graph: `n_sub`, each partial's name/type/elements/contiguity/data presence, P2P reserve result, and whether the first-seen-shape defer applied. This will tell whether the BF16 NCCL events correspond to graphs rejected for an actual type/layout reason or simply the deliberate unmerged shape path.

If a BF16 P2P prototype is later authorized, use the current NCCL BF16 path as the numerical reference: preserve actual per-rank F32 partials and compare candidate results against the baseline NCCL outputs on cancellation-heavy, ties/rounding-boundary, zero-contributor, and magnitude-spread fixtures at sizes immediately below/at/above 131072 elements and the 4 MiB P2P bucket. Compare exact output bits to the NCCL reference first; any relaxed tolerance requires a separate downstream logit/token and model-quality gate. Then test one captured model graph and verify that every rank retains matching collective order and staging bounds. The present trace alone cannot set a correctness tolerance or establish speedup.
