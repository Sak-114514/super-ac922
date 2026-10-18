# Argmax uniqueness: Meta 3-device graph fixture result

Date: 2026-10-15
Status: PASS for synthetic Meta graph correctness and in-graph P2P execution; not a model-level result.

## Runtime and scope

- Frozen runtime: `/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v12`
- Runtime manifest: `runtime-manifest.json`
- Fixture binary SHA-256: `be43059393ca2cd745d2a00532d8e79f216a8bda075772e09be5670ed529e79e`
- Fixture stdout/stderr: `notes/argmax-unique-meta-fixture-run-v2-20261015.log`
- Log SHA-256: `723477ea13e23a0dec577e473d510ae5359527276b879ec9285140b9e770c7c8`
- Exact GPUs: GPU3/GPU4/GPU5 island-B UUIDs, in that order, all Tesla V100-SXM2-16GB.
- No model loaded; only synthetic F32 logits (maximum tensor 165546 × 8).
- No source outside the new fixture directory or notes was changed.

## Test

The Meta split callback shards the logits vocabulary axis evenly over 3 CUDA devices (55182 columns/device) and marks each `GGML_OP_ARGMAX_SPLIT` candidate tensor as `GGML_BACKEND_SPLIT_AXIS_PARTIAL`. The graph then runs the actual private operators:

`ARGMAX_SPLIT_UNIQUE -> Meta P2P AllReduce -> ARGMAX_UNIQUE_FINALIZE -> ID/unique views -> host download`

For each shape `nrows=1,4,8`, the same graph was run A/B/A. Cases included unique all-negative maxima on different ranks, a rank-boundary tie, a tie crossing the local 4096-key split chunk, NaN plus a finite winner, +Inf, all -Inf, and an additional negative winner. CPU reference output used the lower-index tie rule and marked uniqueness false for any nonfinite row or tie. Every downloaded ID and uniqueness flag matched; the A and A-again outputs matched exactly; the three simple Meta members agreed; and the post-AllReduce four-channel candidate vectors matched the CPU per-rank summaries.

## Evidence

The captured log contains:

- `ggml_backend_meta_context: using in-graph P2P AllReduce`
- P2P launch records for ranks 0, 1, and 2 at all three tested sizes (`n=12,48,96`)
- CUDA Graph warmup-complete/reuse messages
- PASS lines for all A/B/A runs and the full 3-rank graph gate
- `/proc/self/maps` entries for `libggml.so.0`, `libggml-base.so.0`, `libggml-cpu.so.0`, and `libggml-cuda.so.0`, all from `runtime-perf-v12/lib`; CUDA runtime and NCCL mappings are also captured.

The build's bare `ldd` initially resolved the transitive CUDA library through an old RUNPATH. The recorded execution explicitly set `LD_LIBRARY_PATH=runtime-perf-v12/lib`; the saved `ldd-v12.txt` and in-process maps both confirm the intended libraries. This environment override is part of the reproducible run command and must be retained.

After process exit, all six GPUs showed 0 MiB except GPU5 at 3 MiB; no fixture process remained. No reset was performed.

## Command

```sh
D=/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/argmax-unique-meta-20261015
R=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v12/lib
env -u GGML_CUDA_P2P_AR_DISABLE \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-fadb0459-4d30-907b-289a-862b3014f165,GPU-760ce501-32f6-69b6-215b-1020da68bbfc,GPU-d3558623-deca-c702-3cc0-01b4aa9c667a \
  LD_LIBRARY_PATH="$R" \
  GGML_CUDA_ALLREDUCE=none \
  GGML_CUDA_P2P_AR_DEBUG=1 \
  "$D/argmax-unique-meta-fixture"
```

## Fixture-only correction

The initial run aborted before graph execution because the fixture allocated an eight-row input for each 1/4/8-row graph and compared the fixed input's output shape against the active case size. The source now creates a distinct input tensor/context per requested shape. No assertion was weakened; all cases use their actual requested dimensions.

## Limits

This verifies the private argmax metadata operators, three-rank Meta partitioning, P2P AllReduce, graph replay, and host output agreement on synthetic logits. It does not establish production sampler integration, CUDA-Graph performance improvement, multi-stream correctness, or full-model semantic behavior.
