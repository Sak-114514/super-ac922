# V100 fused-expert Tensor Core source audit for the seven IQ expert formats

Date: 2026-10-15
Scope: source-only review. No engine edits, build, model load, or GPU use. Root owns expert placement/loader analysis.

## Pinned upstream source

Repository: https://github.com/1115714829/llama.cpp-v100
Observed master HEAD: 9c3e19b5f8e1fdbe0b6c70d193bd5e58515b6faa (2026-10-14 17:32 UTC), version 2.0.0 in README.
The README describes a GPL-3.0 project. fused-expert.cu itself explicitly says its Volta tile/dequant portions are ported from Strata-AC922 under MIT; keep that provenance if code is reused.

Target files saved locally under source-refs/llama-cpp-v100-20261015/:
- fused-expert.cu, 31,176 bytes; SHA-256 61a3a2ce4f8c1f8e32ca21af937b9815702b5afdb750199f4263941590ae13aa
- fused-expert.cuh, 1,371 bytes; SHA-256 00a0d0df848a274bfd41f71830d84cad2bb5f578d7bc0fb3fcfafdba332964a2
- integration-excerpts.txt contains only the relevant dispatcher, CMake glob, and graph-fusion excerpts.

Primary file links pinned to the observed commit:
- [fused-expert.cu](https://github.com/1115714829/llama.cpp-v100/blob/9c3e19b5f8e1fdbe0b6c70d193bd5e58515b6faa/ggml/src/ggml-cuda/fused-expert.cu)
- [fused-expert.cuh](https://github.com/1115714829/llama.cpp-v100/blob/9c3e19b5f8e1fdbe0b6c70d193bd5e58515b6faa/ggml/src/ggml-cuda/fused-expert.cuh)
- [MUL_MAT_ID dispatch](https://github.com/1115714829/llama.cpp-v100/blob/9c3e19b5f8e1fdbe0b6c70d193bd5e58515b6faa/ggml/src/ggml-cuda/ggml-cuda.cu#L4580-L4623)
- [fused launch call](https://github.com/1115714829/llama.cpp-v100/blob/9c3e19b5f8e1fdbe0b6c70d193bd5e58515b6faa/ggml/src/ggml-cuda/ggml-cuda.cu#L4838-L4855)
- [CMake CUDA source glob](https://github.com/1115714829/llama.cpp-v100/blob/9c3e19b5f8e1fdbe0b6c70d193bd5e58515b6faa/ggml/src/ggml-cuda/CMakeLists.txt#L114-L128)
- [graph-level fused gate/up path](https://github.com/1115714829/llama.cpp-v100/blob/9c3e19b5f8e1fdbe0b6c70d193bd5e58515b6faa/src/llama-graph.cpp#L2263-L2283)

## Seven actual expert formats in our GGUF fixture

The existing real-row manifest uses these seven formats; all seven are in our current grouped MMQ dispatch:

| Type | GGUF block | Packed bytes per block | Sample tensor shape K x N x E | Row stride |
|---|---:|---:|---:|---:|
| IQ2_XXS | 256 | 66 | 2560 x 640 x 512 | 660 B |
| IQ2_XS | 256 | 74 | 2560 x 640 x 512 | 740 B |
| IQ2_S | 256 | 82 | 2560 x 640 x 512 | 820 B |
| IQ3_XXS | 256 | 98 | 2560 x 640 x 512 | 980 B |
| IQ3_S | 256 | 110 | 2560 x 640 x 512 | 1100 B |
| IQ4_NL | 32 | 18 | 640 x 2560 x 512 | 360 B |
| Q2_0 | 64 | 18 | 640 x 2560 x 512 | 180 B |

Source evidence: fixtures/volta-tc-real/manifest.json (GGUF tensor-directory metadata plus bounded raw rows; no whole-weight scan). The current grouped MMQ type switch in src/ggml/src/ggml-cuda/mmq-moe-grouped.cu:70-85 includes all seven, plus Q4_0, Q6_K, and IQ4_XS. It quantizes F32 activations to Q8_1 and reuses the regular MMQ tile body; its per-format output is the existing numerical baseline.

## What the V100 fork actually implements

The fused-expert file is a Volta W4A16 kernel. It uses TM=64 tokens, BN=64 output rows, BK=64 K values, four warps, and the Volta mma.sync.m8n8k4 instruction. Each CTA loops K tiles, decodes packed weights into an F16 shared-memory tile, and consumes an F16 activation tile. It reuses expert_bounds, ids_src1, and ids_dst.

Its supported quant types are exactly four, none of them the model's seven:
- Q4_K and Q5_K for the row-paired gate/up path (with optional SwiGLU epilogue);
- Q5_1 and Q8_0 for the plain/down path.

The capability check is restricted to cc 7.0, F32/F16 activations, I32 IDs, N divisible by 64, compatible K and row-stride alignment, no repacked tensor extra pointer, and more than 8 tokens. Small decode/verify batches remain on MMVQ/grouped dp4a. The target dispatch checks fused-expert before grouped MMQ where supported; otherwise the normal grouped/MMQ fallback remains.

This is therefore not a drop-in IQ3_XXS or mixed-IQ kernel. The block readers are hard-coded for Q4_K/Q5_K/Q5_1/Q8_0. For our formats, both block size and row stride differ. All seven sampled row strides are not 16-byte aligned, and five formats have blocks whose byte size is not 16-byte aligned. The existing uint4/vector loads and alignment guards cannot simply be reused. A correct IQ port needs byte-safe packed-block reads and a separate decoder per GGUF format.

## Can it avoid a whole F16 weight cache?

Yes, structurally. The reference kernel dequantizes one BK=64 weight tile into per-CTA shared memory and discards/replaces it on the next K iteration. It has no whole-model F16 weight cache and stores no new quantized format. F32 activations are converted into a launch-local F16 buffer, then staged to shared memory.

That does not preserve the current MMQ arithmetic exactly:
- IQ weights would be decoded to F32 values and rounded to F16 for the shared tile.
- F32 activations are rounded to F16.
- Tensor Core products/FP32 accumulation change reduction order. The source explicitly uses fresh accumulators per K tile because Volta Tensor Core accumulation truncation in a long dependent chain biased sums toward zero.
- The optional SwiGLU epilogue writes F16 H.

Therefore this is an approximate arithmetic route, despite using the original packed GGUF weights and requiring no persistent dequantized cache. It needs exact packed-row decoder fixtures and output comparison against current grouped MMQ, then a semantic-quality gate. Do not claim bit parity from the upstream kernel.

The IQ block geometry is compatible with BK=64 in principle: IQ2/IQ3 blocks span four K tiles, IQ4_NL spans two, and Q2_0 exactly spans one. But a decoder must handle each format's codebooks, signs, scales, and unaligned block starts; the current source contains no IQ decoder.

## Smallest integration path

1. Keep the current grouped MMQ as the default and fallback. Add an opt-in SM70 prefill dispatch before it, guarded by format, token count, K/N shape, activation type, and raw/compatible expert pointer layout.
2. Port only the tile scheduling, expert_bounds/ids tables, 64x64x64 swizzled shared-memory layout, and Volta MMA accumulation. Add seven per-format tile decoders that stream directly from the packed rows into the current CTA's F16 shared tile; do not materialize or cache all experts as F16.
3. Start with the plain F32-output kernel path for each existing MUL_MAT_ID. Do not adopt the target's fused SwiGLU epilogue in the first port: upstream graph code uses a dedicated interleaved gate_up tensor and ggml_mul_mat_id_swiglu API, whereas our private graph currently represents gate/up as separate MUL_MAT_ID nodes followed by GLU. That epilogue would require a graph/weight-layout change.
4. Test each format using the existing fixtures/volta-tc-real payload rows for experts 0/17/511 and rows 0-15. First verify the block decoder's F16 tile against a CPU dequant reference; then compare GEMM outputs against grouped MMQ and run semantic tests before enabling any format.

No microfixture was built because the inspected upstream kernel does not support all seven required formats. No engine build or GPU work was done in this audit.
