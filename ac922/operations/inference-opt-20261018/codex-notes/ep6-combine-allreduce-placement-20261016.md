# EP6 combine-before-AllReduce placement audit

Date: 2026-10-16  
Scope: read-only Qwen4Exp FFN/split-state source audit plus existing V36 V85 logs. No build or GPU work was performed.

## Finding

The outer six-rank reduction is after the MoE contribution has been down-projected, multiplied by the selected top-10 router weights, and summed across the 10 expert slots. It does not reduce the raw per-expert down-MATMUL output. The graph exposes the final `[n_embd, n_tokens]` tensor as PARTIAL; the top-10 intermediate is `[n_embd, 10, n_tokens]` and remains a local partial, so it is not an outer subgraph boundary.

For this model `n_embd=2560`, `n_expert_used=10`, F32 partial output, the outer AR6 payload is:

| Token rows T | Final AR6 `[2560,T]` | Raw down-MMID `[2560,10,T]` |
|---:|---:|---:|
| 16 | 163,840 B (160 KiB) | 1,638,400 B (1.5625 MiB) |
| 1,024 | 10,485,760 B (10 MiB) | 104,857,600 B (100 MiB) |

The T=1,024 result is directly reflected in V36 logs: during the 1,024-token prefill UBatch, the nested six-rank reduction reports `bytes=10485760`, and the P2P implementation logs `relay6 selected ... bytes=10485760` for ranks 0–3. Since a raw `[2560,10,1024]` tensor would be 100 MiB and exceed the configured 10 MiB relay cap, the observed 10 MiB relay selection is consistent with the final combine-before-AR6 path, not raw down-MMID communication.

At T=16, the expected final AR6 is 160 KiB, below the relay6 minimum of 204,800 B; it should use the one-shot P2P route. A raw down-MMID boundary would instead be 1.5625 MiB and be relay6-eligible. The available V36 per-rank logs do not expose a T=16 `kind=0`/`kind=3` record because the debug launch-kind log was not enabled; this T=16 value is a source-derived size, not a separately logged dispatch.

## Source chain

1. The V36 runtime reports `n_expert=512` and `n_expert_used=10` at `runtime.log:146-147`. Qwen4Exp calls `build_moe_ffn(..., n_expert_used, ..., norm_w=true, expert_weights_scale, SOFTMAX, ...)` in `src/models/qwen4exp.cpp:3221-3234`.
2. The generic MoE builder computes the mirrored router probabilities, selects top-k IDs, gathers the top-k weights, and normalizes them (`src/llama-graph.cpp:2560-2688`).
3. `MUL_MAT_ID` with expert weights split on axis 2 yields `ggml_ep_private::local_partial_axis` (`ggml-backend-meta.cpp:1345-1350`). Gate/up/down expert outputs therefore remain rank-local partials through the activation path.
4. The down projection is explicitly `[n_embd, n_expert_used, n_tokens]` (`llama-graph.cpp:2857`). For Qwen4Exp, router weights are applied **after** that projection (`2873-2876`). The builder then creates one `[n_embd,n_tokens]` view per expert and adds the 10 views (`2880-2901`). Only the final `moe_out` is marked with `tensor_output_partial_flag` (`2909-2913`).
5. Split-state handling preserves local partial for `MUL_MAT_ID`, broadcast `MUL`, views/reshapes and matching partial additions; the output flag converts only this final local partial into a standard PARTIAL when queried asynchronously (`ggml-backend-meta.cpp:1290-1310, 1345-1350, 1434-1438, 1493-1496, 1838-1842, 1907-1912, 2064-2068`). Standard PARTIAL nodes terminate an outer subgraph (`5687-5692`).
6. `nested_gpu_allreduce` inspects each island subgraph’s last tensor, requires two PARTIAL islands and six physical leaves, then uses `ggml_nbytes(ref)` as the collective payload (`6402-6477, 6492-6514`). The six-rank nested AR is executed at each outer partial boundary (`6553-6560`).
7. Relay6 selection is purely size-gated for six devices: bytes must be at least 204,800 and no larger than `GGML_CUDA_P2P_AR_RELAY_MAX_BYTES` (`allreduce-p2p.cu:373-380, 415-423`).

## V36 record interpretation

- V36 profile: `GGML_CUDA_P2P_AR_RELAY6=1`, `GGML_CUDA_P2P_AR_RELAY_MAX_BYTES=10485760`; `GGML_CUDA_P2P_AR_DEBUG` is absent.
- In the saved V36 runtime, the prefill interval shows `batch=8192`, `max_ubatch=1024`, `selected_ubatch=1024`, then eight logged `meta-nested-gpu-ar` records at 10,485,760 B; these are six-rank `p2p6-rank-order-async`, `host_stage=0` records. Four `P2P relay6 selected` notices at 10 MiB are printed for ranks 0–3.
- The activation parser says `relay6_kind3_selected_all_ranks_within_10mib=false` and collected no `p2p allreduce launch ... kind=3` records. That gate depends on the debug launch-kind log, which is emitted only when `GGML_CUDA_P2P_AR_DEBUG` is present (`allreduce-p2p.cu:432-437`). Therefore the parser’s false is a missing all-rank kind trace, not evidence that relay6 was not selected. The explicit 10 MiB `P2P relay6 selected` messages prove selection on the logged ranks; they do not individually prove all six kernel launch logs.

## Conclusion and limits

There is no 10× EP6 communication tax from reducing the raw top-10 down-MMID tensor in the current Qwen4Exp source path. The six-rank reduction boundary is the top-10 weighted and combined hidden output `[2560,T]`; at T=1,024 that is exactly 10 MiB, and the V36 log shows this payload entering the relay6-selected path. At T=16 the source-derived payload is 160 KiB and below the relay threshold. No speed estimate is made, and no claim is made that the all-rank relay6 debug gate passed in V36.
