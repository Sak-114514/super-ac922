# One-bank × 4-lane D3 dispatch audit (host-only)

## Scope and result

This is a read-only audit of the V30b four-lane D3 token divergence and the
current V78 source. No engine files were changed. No GPU, CUDA initialization,
model loader, or model tensor payload was used.

The V30b run passed its F32 preclean and native semantic gates but failed its
own-serial token gate: `real8k-0` first diverged at token 2 and `real8k-1` at
token 13 in both diagnostic and clean P4. The other two flows matched. The
original two auxiliary prefix checks also failed, so the run was not quality
certified. The evidence is in
`fixtures/dual-context-v30-onebank4d3-20261016/results/root-p4-v30b-onebank4d3-hierarchical-ep6-shortctx16k-20261016t041327z/v30b-run-analysis.json`
(SHA-256 `13aea5a3f18543de4c9443614b4fe249d8dd7bdacf5c9b465f71973de062a28b`)
and `v30b-real8k-token-review.json` (SHA-256
`9aa09dc62582c32c16f171d7abb93e44ecd9996126202f1a04cc230d1f86494b`).

There is an actual 8-versus-16 target-shape dispatch change in the MoE
gate/up/GLU graph fusion. In V30b's own V71 runtime log, the 8-token graph
selected `moe_gate_up_glu_fusion`, while the 16-token graph declined it as
`shape_or_backend_ineligible`. Separately, the ordinary dense and expert
matmul paths for 16 rows were chunked into two 8-row MMVQ calls. The chunking
does not remove the earlier graph-fusion branch. This fusion transition is a
credible numerical-divergence candidate, not a proven cause. The V78
per-stream QSA helper is a separate attention-path change and does not alter
this MoE gate.

## Run provenance

V30b was run once with V71 (`20261016T040204.430535Z`), one bank × four
16K streams, D3, 16 target verification rows, EP=1, `GGML_SM70_DECODE_LINEAR_CHUNKS=8`,
and `GGML_CUDA_MOE_GROUPED=1`. Runner SHA-256 was
`9f114e7101e909a3d30d1bb7d7c0823160b25e55520294b9dd30b51e3449b24c`; the V71
`libggml-cuda.so.0` SHA-256 was
`3b55d98d385d54dce5ba177050218eb7e26f15b3c424498e7cfc98bb7d44a6da`. The
candidate fixture source was
`dual-context-v30-onebank4d3.cpp` (SHA-256
`0ec67cc846042098f600b3d8fa8924089e4d70b0d98c27711d6ff083f1a3d304`). Its
per-lane target row construction is at lines 638–648; the frame budget and
submission are at lines 650–684.

V30b's retained clean family log is
`results/root-p4-v30b-onebank4d3-hierarchical-ep6-shortctx16k-20261016t041327z/copy1/runtime.log`
(SHA-256 `fa03eb8795f12b76becf7d78736779ba78195caae6c56b4831c04208dde15dc4`).
The clean target-frame histogram in `V30B-DIVERGENCE-TRACE.md` is
`{16:67, 12:2, 8:6, 7:1, 5:1, 4:6, 2:1}`; the diagnostic histogram has 79
frames at 16 rows and its own retained row-count distribution.

Current V78 is build `20261016T061009.431203Z`, report SHA-256
`43cf66b8de860efe35eb891a42ae3b6d57d9b4138d95a0ef61b9e18a25ebcb2b`. Its
`changed_since_runtime` list contains only
`src/llama-qsa-standard-sparse.h` and `src/models/qwen4exp.cpp`. The current
V78 source hashes relevant here are:

| Source | SHA-256 |
| --- | --- |
| `src/ggml/src/ggml-cuda/ggml-cuda.cu` | `b48fd8f0e0c1d9861557ec38f33ccee77dc87b3bf453aa2a72021c8306d288a7` |
| `src/ggml/src/ggml-cuda/mmvq.cu` | `a83ed741f20619008d64839dcb2e3f40ddcc9054df396edda5afe5e6b47d0748` |
| `src/ggml/src/ggml-cuda/mmvq.cuh` | `b37f8359f519ba90ddecd9cb3ca5f8c2ae05685bfae6813d435d75763c2bb006` |
| `src/ggml/src/ggml-cuda/mmq.cu` | `26d53aa56d695b1c0a7f76987e1d67139b9741b283341c1e99e60ce8da2af3de` |
| `src/src/models/qwen4exp.cpp` | `3d4fde36e0c495ba046557458e41a604abaf40e4b44825f327882b3edb679399` |
| `src/src/llama-qsa-standard-sparse.h` | `f00a0518305cb071d93a24d08de8b504b2101dfe8f07a4b308dc832eeeb83230` |

The V71 build report records its `ggml-cuda.cu` source SHA as
`b534167a0483c65717f37eed00e40bdb919f6204446195a7eab9b0fcc1037de7`, versus
V78's `b48fd8…`. The old V71 source snapshot is not present in the tree, so
the V30b runtime path markers are direct binary evidence, while the line-level
explanation below is from the current V78 source. Do not present the old
runtime marker as a V78 runtime result. The V65 baseline used a third
`ggml-cuda.cu` SHA (`0a64714e…`) and a different `libggml-cuda.so.0`; it is
corroborating evidence only.

## Original GGUF metadata (no tensor payload)

The original model is
`/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`.
The audit used the repository's `gguf.GGUFReader` in read-only mode to inspect
GGUF header fields and tensor directory names/types/shapes. The reader uses a
read-only `numpy.memmap`; its `_get` operations make views, and this audit
never accessed a `ReaderTensor.data` array. It did not hash, load, or copy any
tensor payload. File metadata reports 78,615,622,080 bytes, 1,256 tensors,
68 fields, and the tensor data section starts at byte 11,026,368. Metadata
fields were `general.architecture=qwen4exp`, `general.file_type=23`, 49
blocks, embedding width 2560, 512 experts, top 10.

Relevant original tensor metadata (type is from the GGUF directory):

| Weights | Type distribution across trunk plus MTP block |
| --- | --- |
| `output.weight` | Q5_K, shape 2560 × 248320 |
| `token_embd.weight` | IQ3_S, shape 2560 × 248320 |
| Trunk attention `attn_q/k/v/output.weight` | IQ3_S 2, IQ4_XS 8, Q4_K 7, Q5_K 2, Q6_K 29, Q8_0 4 (52 tensors; the remaining blocks are recurrent rather than attention blocks) |
| MoE `ffn_gate_exps.weight` and `ffn_up_exps.weight` (each) | IQ2_S 10, IQ2_XS 10, IQ2_XXS 9, IQ3_S 13, IQ3_XXS 6, plus one Q8_0 MTP tensor |
| MoE `ffn_down_exps.weight` | IQ4_NL 18, Q2_0 30, plus one Q8_0 MTP tensor |
| Router `ffn_gate_inp.weight` | BF16 in 48 trunk blocks; F32 in the MTP block |
| Shared-expert `ffn_gate/up/down_shexp.weight` | mixed IQ/Q formats; see the exact type histogram in the metadata output summarized below |

The shared-expert histograms were gate: IQ3_S 13, IQ4_XS 21, Q4_K 12, Q5_K
1, Q6_K 1, Q8_0 1; up: IQ3_S 27, IQ4_XS 10, Q4_K 7, Q5_K 1, Q6_K 3, Q8_0
1; down: IQ4_NL 40, Q2_0 8, Q8_0 1. The Q8_0 entry in each routed/shared
expert histogram is the layer-48 MTP tensor. A prior `no_alloc=true` loader
smoke also reports `output.weight=Q5_K` and `token_embd.weight=IQ3_S`, both
with `data=none`; it agrees with the GGUF metadata-only inspection.

## Current V78 dispatch seams

### Ordinary dense matmul and final projection

- `ggml-cuda.cu:2663-2687` has the V100 `GGML_SM70_DECODE_LINEAR_CHUNKS=8`
  branch. For eligible CUDA weight matrices with `9 <= ne11 <= 64`, it
  creates contiguous subviews of at most 8 input columns and recursively
  calls `ggml_cuda_mul_mat`. The recursive per-row kernel sees the same
  8-column shape as a normal 8-row verify. Eligibility also requires ordinary
  2-D weight/input geometry and contiguous first-dimension layout; it is not a
  universal row-pad.
- V30b had the switch enabled. Its runtime logs contain actual markers such
  as `sm70 decode linear chunks active: dense columns=16 chunk=8 type=q5_K`,
  as well as BF16, Q6_K, IQ3_S, IQ4_XS, IQ4_NL, Q2_0, and Q4_K 16-column
  shapes. Thus the final Q5_K output projection at 16 rows was split into
  8-column chunks in that run.
- The normal vector/matrix boundary is `ggml_cuda_should_use_mmvq` in
  `mmvq.cu:354-463`, followed by the MMVQ/MMQ branch in
  `ggml-cuda.cu:2788-2795`. `MMVQ_MAX_BATCH_SIZE` is 8 (`mmvq.cuh:3`). V100
  keeps several K-quants below that cap: Q2_K ≤4, Q3_K ≤6, Q4_K ≤5, Q5_K ≤6,
  and Q6_K ≤7. `output.weight` Q5_K therefore uses MMQ for both 8 and 16
  columns; 16 is still chunked into 8 by the explicit V30b option. Default
  MMVQ-eligible types such as IQ3_S and Q8_0 use one MMVQ call at 8 and two
  MMVQ calls at 16. The V30b runtime logged the dense 16-column chunk path.
- The V100 BF16 GEMV branch at `ggml-cuda.cu:2727-2751` only accepts
  `ne11 <= 8` and has an additional long-K/tall-N shape gate. The GGUF BF16
  router tensors are 2560 × 512 (MTP router F32); they do not satisfy either
  long-K (`ne00 >= 10240`) or tall-output (`ne01 >= 8192`) condition. The
  observed V30b BF16 16-column path logs the 8-column splitter. There is no
  source evidence that this BF16 GEMV gate explains the router rows.

### Routed MoE matmul and gate/up/GLU fusion

- For an unfused `MUL_MAT_ID`, `ggml_cuda_mul_mat_id` first tries the V100
  fixed-eight chunk branch (`ggml-cuda.cu` around lines 2830–2890) when
  `ne2` is 9–64 and the canonical expert IDs/strides match. It slices input,
  IDs, and output into chunks of 8 before calling MMVQ. On Volta,
  `get_mmvq_mmid_max_batch()` returns 8 for `MUL_MAT_ID` regardless of
  weight type (`mmvq.cu:322-351`). V30b logged both `sm70 decode linear
  chunks active: moe tokens=16 chunk=8 ...` and
  `ep_mmvq_chunks ... tokens=16 ... selected=1`; the ordinary EP MoE
  16-row GEMM was not forced to an unchunked MMQ shape in that run.
- The gate/up/GLU *graph fusion* is an earlier decision. For the
  `MUL_MAT_ID`, `MUL_MAT_ID`, GLU pattern, the fuse visitor calls
  `ggml_cuda_should_fuse_mul_mat_vec_q(up_n)` before individual matmul
  dispatch (`ggml-cuda.cu:6721-6815` and `:6880-6930`). That helper requires
  quantized weights with F32 input/output and a batch no larger than 8; for `MUL_MAT_ID`, it
  also rejects `dst->ne[2]` above
  `get_mmvq_mmid_max_batch(type, cc)` (`ggml-cuda.cu:2599-2629`).
- The V30b runtime's own `copy1/runtime.log` has both decisions: 8-token
  gate/up/GLU attempts were selected (`tokens=8 selected=1 reason=eligible`,
  51 markers), while 16-token attempts were declined (`tokens=16 selected=0
  reason=shape_or_backend_ineligible`, 12 markers). The declined 16-row graph
  then used the separate per-op 8-row chunk route; the same log records
  `sm70 decode linear chunks active: moe tokens=16 chunk=8 ...` and
  `ep_mmvq_chunks ... tokens=16 ... selected=1`. This is a concrete
  onebank4 shape-dependent dispatch change inside the model. The Q5_K final
  projection follows the separate dense-chunk path above.
- The separate MMVQ MoE kernel handles `has_ids && ncols_dst > 1` (or the
  V100 fixed-decode tail) at `mmvq.cu:1561-1581`. Thus at 8 rows, the fused
  route can combine gate/up and GLU; at 16 rows it can execute two chunked
  gate/up `MUL_MAT_ID` calls and a later GLU operation. The arithmetic
  grouping changes. This is a plausible source of token-sensitive
  differences, but the log does not attribute a specific token divergence to
  this layer or operator.

### Route-preserving 16 = 8 + 8 proposal

If root chooses to normalize this fusion decision, the narrow seam is the
existing `MUL_MAT_ID` gate/up/GLU branch in `ggml_cuda_try_fuse`, after its
`ggml_can_fuse_subgraph` and `ggml_cuda_check_fusion_memory_ranges` checks
(`ggml-cuda.cu:6762-6764`, then `:6791-6808`). Keep the existing one-call path
unchanged for shapes up to 8. For the explicitly opted-in 16-row target verify
only, create two consecutive stack views and call the existing
`ggml_cuda_mul_mat_vec_q_fused_route` twice, each with `ncols_dst=8`:

1. Copy the full input, IDs, and output descriptors. Set their token extents to
   8 and offset `input.data` by `first * src1->nb[2]`, `ids.data` by
   `first * ids->nb[1]`, and `output.data` by `first * dst->nb[2]`, for
   `first=0,8`. This mirrors the canonical V100 `MUL_MAT_ID` chunk slices at
   `ggml-cuda.cu:2873-2885`; it keeps the same row/token order and writes two
   disjoint output ranges.
2. Pass the original routed graph node as the wrapper's `route` argument for
   each chunk. The wrapper makes an output copy and applies the original
   `ggml_ep_private::offset(route)` and global expert count before dispatch
   (`ggml-cuda.cu:6001-6019`). Preserve the unmodified global IDs in each
   token slice. `ggml_cuda_mul_mat_vec_q` derives `mmvq_ep_offset/end` from
   that routed output, and the existing kernel maps global expert IDs to the
   rank-local axis-2 weight slice (`mmvq.cu:1822-1829`, expert ID indexing at
   `:1180-1208`). Do not pass a plain GLU output as the EP route descriptor or
   pre-subtract the local expert offset.
3. Reuse the original gate/up weights, scales, and GLU parameters. Do not
   repack weights, trim experts, or cast activations. Each chunk must call the
   same existing fused MMVQ kernel specialization with eight rows so each
   token sees the same per-row arithmetic and activation quantization as the
   existing 8-row case.
4. Preserve the graph pattern, full-range alias checks, type/layout guards,
   and all seven expected weight-format qualifications. If any format lacks
   the existing fused/MMVQ implementation or the contiguous 16-row views,
   decline this special path and run the existing fallback; do not assert or
   route unsupported weights to another precision.
5. Keep both chunk launches on the same `ctx.stream()`. `ggml_cuda_mul_mat_vec_q`
   quantizes each input view and dispatches its fused kernel on that same
   stream (`mmvq.cu:1836-1849`, `:1911-1945`, `:1985-1989`), so chunk 1's
   quantization cannot overwrite temporary storage before chunk 0's consumer
   has run. The per-context `q8_1_cache_t` two-slot activation cache keys its
   source by `t->data` plus row layout (`common.cuh:1488-1519`; contiguous tensors compare
   `ne0` and row count, while non-contiguous tensors compare full dimensions
   and strides), not by the address of the stack `ggml_tensor`; using the
   proper per-chunk `data` offsets is essential to prevent a false cache hit
   when the stack descriptor is reused.

There is no implicit “prefill versus verify” flag in this generic CUDA fusion
visitor. A `ne2==16` guard alone could also see a 16-token prefill. To meet the
requested no-prefill-change boundary, the eventual owner needs an explicit
MTP-target-verify opt-in/phase marker at this seam (or equivalent context
metadata) in addition to the exact shape/type guards. Do not turn on a global
`allow_batch=16` rule. This proposal has not been implemented or GPU-tested.

## V78 attention candidate is orthogonal

`LLAMA_QSA_STANDARD_SPARSE` is opt-in and defaults off. In V78,
`qwen4exp.cpp:2301-2327` uses the new private adapter only when the flag is
exactly `1`, flash attention and the existing sparse QSA kernel are enabled,
and the adapter accepts the shape. `llama-qsa-standard-sparse.h:15-95`
keeps the original selected IDs, constructs per-stream exact K/V views using
`nb[3]`, gathers the matching mask, and calls the existing explicit-cell
`GGML_OP_FLASH_ATTN_EXT`; it does not add a CUDA kernel or pool/approximate
selected blocks.

The previous single sparse path at `qwen4exp.cpp:2330-2335` requires
`top_k->ne[3] == 1` and `k_all->ne[3] == 1`; a non-unified four-stream KV
layout cannot meet that condition and falls through to the dense attention
path. The V78 adapter removes that stream-count-dependent attention choice
when explicitly enabled. It does not change the GEMM row threshold, MoE
gate/up/GLU fusion, final projection, or the `MMVQ_MAX_BATCH_SIZE=8` cap.
Therefore it may repair a onebank4 attention discrepancy but is not evidence
that all 8→16 linear dispatch differences are fixed.

## Evidence boundary and recommended discriminator

The V30b 16-row `ep_mmvq_chunks` and dense chunk markers show that the explicit
SM70 chunk path was active in its V71 binary. The 8-versus-16 gate/up/GLU
selection marker is the highest-priority remaining dispatch seam. The
current V78 source has the same 8-row condition, but V30b ran V71, whose
`ggml-cuda.cu` source SHA (`b534167a…`) differs from current V78 (`b48fd8f0…`);
the historical V71 source file itself was not retained. Treat the V30b marker
as observed runtime evidence for V71 and the current source review as static
V78 evidence, not as a V78 runtime pass.

For the next bounded test, capture target-verify graph markers for 8- and
16-row cases under V78 with `LLAMA_QSA_STANDARD_SPARSE=1`. Verify
`qsa-standard-sparse ... selected=1 streams=4` separately from
`moe_gate_up_glu_fusion`. If 16 rows still declines fusion while 8 rows
selects it, compare one fixed target input through fused-eligible and
unfused gate/up/GLU routes with the same expert IDs and full F32 output rows.
Keep this as an isolated operator/graph diagnostic; the existing V30b full
model failure must not be relabeled as fixed without the original logits,
token, and canary gates passing.
