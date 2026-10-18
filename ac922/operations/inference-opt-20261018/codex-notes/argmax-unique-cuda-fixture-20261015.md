# CUDA argmax metadata fixture build record (2026-10-15)

## Scope and stop condition

The fixture compiles the current `ggml/src/ggml-cuda/argmax.cu` implementation into a standalone CUDA translation unit wrapper and links against the frozen V10 runtime libraries in `runtime-perf-v10/lib`. The final binary was run only on physical GPU2, selected by UUID with `CUDA_VISIBLE_DEVICES`; it did not load a model or exercise the engine graph. The root granted the bounded window. The process ended normally and left GPU2 at 2 MiB use and 40 C; all other GPUs were at 0 MiB.

The wrapper uses manually constructed GGML tensors because `runtime-perf-v10` predates the new private GGML factory symbols. It calls the actual `ggml_cuda_argmax` and `ggml_cuda_argmax_split` dispatch functions from the current included source. The runtime matrix covered vocab 165546, output rows 1/4/8, direct and split TP3/TP6 paths, and A/B/A raw-ID stability. Split cases aggregate the four partial channels on CPU to model meta AllReduce and verify non-rank slots are zero before final GPU reduction. It checked output canaries and finite negative winners, ties within/across chunks and ranks, NaN, +Inf, and all -Inf. The full result was `PASS cuda_argmax_unique rows=1,4,8 vocab=165546 split=3,6 A/B/A=1 canaries=1 nonrank4channels_zero=1`. A separate TP3/8-row invalid-value case captured each rank's partial channels; the NaN in row 4 yielded `invalid=1` only in rank 1's local slot and final unique=false.

Two initial fixture-only issues were found and corrected before the pass: local per-rank inputs must be repacked from the global row-major tensor, and canary initialization/input updates must use the same nonblocking CUDA stream as the kernels to avoid a default-stream race. No engine source changes were needed.

This is argmax F32 coverage only; it does not retest seven expert quantization formats, GGML graph scheduling, or model semantics.

## Build evidence

Build script: `fixtures/argmax-unique-cuda-build.sh`

Source wrapper: `fixtures/argmax-unique-cuda-fixture.cu`

Object: `fixtures/argmax-unique-cuda-fixture-20261015.o`

Linked binary (not run): `fixtures/argmax-unique-cuda-fixture-20261015`

The NVCC compile command completed with exit code 0 using CUDA 12.4, `-arch=sm_70`, C++17, and `-use_fast_math`. The link completed with exit code 0. `ldd` confirms the executable resolves the frozen runtime libraries:

- `runtime-perf-v10/lib/libggml-cuda.so.0`, SHA256 `f36283edced176d7a04a918c328a22e01adcb95962eb2b4a9a78689a9a092951`
- `runtime-perf-v10/lib/libggml-base.so.0`, SHA256 `4bb34fce3820138aa47dbddc7ecc75485701e2dbdf4b53429ae38cb9a9552a66`

The executable RUNPATH includes `runtime-perf-v10/lib` and CUDA 12.4 ppc64le libraries. Logs are `notes/argmax-unique-cuda-fixture-compile-20261015.log` and `notes/argmax-unique-cuda-fixture-link-20261015.log`.

Fixture SHA256: `4c8088cd8fd66a7eac2978b8bd39da18bef7fdb672b64f484315ac8737bda1c4`
Object SHA256: `993a9fa933aeba752aa2b4e7b63871718a1196874ef0cdb1941aeea8ea0d9dbc`
Binary SHA256: `b0315adc5b2d26ebc5ad2eeff98378dbe334c2627c6ca5fcb31a85b630073eec`

## Remaining verification

The bounded CUDA kernel fixture passed against synthetic F32 logits and CPU-modeled AllReduce. It proves this kernel-level input/output contract only; it does not prove graph scheduling, actual meta-backend AllReduce placement, model-level sampler integration, or performance.
