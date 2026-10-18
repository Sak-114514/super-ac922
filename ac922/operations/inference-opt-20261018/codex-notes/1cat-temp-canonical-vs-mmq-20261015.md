# Per-MUL_MAT_ID temporary GGUF transcode vs direct MMQ — source review

Date: 2026-10-15. Read-only source review plus one authorized bounded GPU2 run of the independent private fixture. No model was loaded, and no engine source, `_C`, venv, or system configuration was changed.

## Direct MMQ already supports the target shape

The current llama.cpp `ggml_cuda_mul_mat_q_moe_grouped` path is env-gated at `ggml/src/ggml-cuda/mmq.cu:184-190` by `GGML_CUDA_MOE_GROUPED=1`. Its type switch includes IQ3_XXS (`mmq-moe-grouped.cu:20-35, 60-62`), accepts up to `MMQ_MOE_GROUPED_MAX_EXPERTS=1024`, and rejects only unsupported shape/type or shared-memory configurations (`mmq-moe-grouped.cu:78-151`). Therefore `E=512, N=224, K=2560` is inside its declared bounds.

`ggml_mul_mat_id` is a public graph operation; its API documents IDs shaped `[top_k,T]`, activation `b` broadcast across top-k, and output `[N,top_k,T]` (`ggml/src/ggml.c:3560-3590`). The backend path builds route bounds and scatters broadcast activations, quantizes the F32 activation to Q8_1, then dispatches the grouped MMQ kernel directly over original GGUF weights (`mmq-moe-grouped.cu:244-275, 281-295`). That path does not construct a full expert-bank expansion.

There is already a close kernel-only data point in `experiments/standard-opt-20261014/notes/grouped-j-20261015-results.md`: IQ3_XXS, E512/N224/K2560, 20 synthetic routed rows per expert (same aggregate work as T=1024 with top-10), one grouped MMQ kernel node in a CUDA graph. J32 measured **865.9968 µs**, J16 1182.3786 µs, J64 1308.1088 µs; all J outputs were bitwise equal. It used synthetic prebuilt Q8_1 blocks and uniform expert bounds, so it excludes GGML route construction and activation quantize-scatter. There is no matching T=1536/30-rows-per-expert data point.

Use 0.866 ms only as a kernel-stage lower bound at T=1024. The candidate's required route/F16 preparation + producer + pointer-table + GEMM **total** needs to be compared to it; an end-to-end GGML graph comparison is still needed for an apples-to-apples final claim. The existing direct kernel fixture source is `experiments/standard-opt-20261014/fixtures/grouped-j-20261015/grouped-j-fixture.cu`.

## E512 candidate fixture result

`fixtures/1cat-prefill-transcode-20261015/large-geometry/` now contains a bounded run script and Python fixture for E512/N224/K2560/top10 at T=1024 and 1536. It cycles three real IQ3_XXS/type-18 row samples over 512 pseudo-experts; the data rows are copied into distinct GPU allocations to force the producer to read roughly 107 MiB instead of reusing three cached pointers. Routing is synthetic and balanced: exactly 20 or 30 rows per expert. It builds a **single reusable 140 MiB** temporary canonical weight+stats bank, compares every byte against the current `_C` canonical prepare result, and compares candidate/reference grouped-GEMM FP16 outputs exactly with the same activation and offsets.

The fixture ran once on GPU2 at 38°C; peak allocated memory was 610.3 MiB. It reported route+FP16 activation preparation, producer, pointer-table construction, GEMM, combined CUDA-event time, and combined wall time. Candidate weight and stats banks matched the canonical reference byte-for-byte. Same-consumer FP16 outputs were exact at both sizes (`max_abs=0`).

| T | Routed rows/expert | Route+F16 prep | Producer | Pointer table | Grouped GEMM | Combined event | Combined wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 20 | 0.603 ms | 0.390 ms | 0.051 ms | 1.062 ms | 1.997 ms | 2.053 ms |
| 1536 | 30 | 0.607 ms | 0.387 ms | 0.047 ms | 0.789 ms | 1.730 ms | 1.778 ms |

GPU2 returned to 38°C/2 MiB with no compute app; all six cards were idle. The existing `_C.abi3.so` SHA256 and modification time were unchanged.

Limits: the candidate data are repeated existing real rows, not true expert-specific weights; route distribution is synthetic; FP16 consumer output parity does not establish equivalence to GGML's F32→Q8_1→F32 route. The existing 0.866 ms MMQ figure is only a direct kernel timing, not full operator timing.

## Can the TurboMind grouped consumer be called without Torch?

The tensor wrapper at `1Cat-vLLM/csrc/sm70_turbomind/ops/awq_sm70_gemm.cu:3157-3221` checks tensors, then constructs pointer-based matrix descriptors and calls `turbomind::gemm::Gemm::Run`. The public `Gemm::Run`, `MatrixLayout`, `Workspace`, and CUDA-stream signatures are in `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/gemm.h` and `types.h`; the GGUF lattice layout converter is also a raw-pointer CUDA converter.

A pointer-only adapter is technically possible, but not a single-TU, Torch-free drop-in against the current llama build. The wrapper depends on the `Gemm` runtime, dispatch selection, layout-converter registration, and SM70 kernel registry across the vLLM/TurboMind source set. The relevant runtime symbols are visible in `_C.abi3.so`, but linking that extension into llama would also pull Torch/C10 dependencies and private C++ ABI. That is a prototype bridge, not a maintainable Torch-independent launcher.

If integrated into GGML, the clean boundary is a new C ABI accepting raw/source row pointers, temporary canonical banks, FP16 activations, offsets, output, one preallocated scratch descriptor, and `cudaStream_t`. Build/link the necessary TurboMind GEMM/converter sources or extract a small supported static target. Preallocate one ~140 MiB canonical bank per concurrently active stream plus the existing TurboMind scratch (~34 MiB/stream: barriers 1 MiB + partials 32 MiB + tensor maps 1 MiB + flags). This is about 174 MiB per stream (roughly 696 MiB at four streams, 1.36 GiB at eight, per GPU) before activations/output. Reuse it across serialized layer projections; allocate before graph capture and keep pointers stable. This is a larger source/build integration than simply adding one CUDA TU to `mmq-moe-grouped.cu`.

## Decision from this fixture

At T=1024, the candidate's 2.053 ms wall path exceeds the existing 0.866 ms MMQ kernel-only measurement by over 2×; even its grouped-GEMM stage (1.062 ms) is slower than that MMQ kernel-only result. **Do not integrate this temporary-transcode branch for this shape based on current evidence.** The T=1536 candidate total is 1.778 ms, but there is no matching MMQ timing at 30 rows/expert, so it cannot establish a win.

If T=1536 still matters, the next useful comparison is a public-graph GGML `ggml_mul_mat_id` fixture at the same E/N/K/T that includes route-helper and Q8_1 quantize-scatter time; then evaluate FP32/FP16 quality. The direct grouped-MMQ kernel-only lower bound is already sufficient to reject the T=1024 candidate path here.
