# MTP unrolled KV dependency fixture results (2026-10-15)

## Scope

Bounded synthetic GGML operator/graph test only; no model weights or server. D=2, four lanes, F32 K/V cache with poisoned future cells, 8-wide hidden state, 16-token vocabulary. Depth 0 does K/V `SET_ROWS`, hidden-state update, and GPU `ARGMAX`; depth 1 reads K/V and token1, computes hidden/token2, and writes future rows. This is not model-level correctness or evidence that the complete serving path uses this graph pattern.

## Method

- Sequential reference: two preallocated graphs with synchronization and host transfer of depth-0 hidden/token1 between graphs.
- Unrolled model-order variant: depth-0 `SET_ROWS` expanded before depth-1 outputs; K/V reads use the aliased base cache tensor.
- Deliberate hazard variant: depth-1 read expanded before depth-0 `SET_ROWS`; reads still use the same base-cache alias.
- Versioned-edge variant: depth-1 `GET_ROWS` consumes the depth-0 `SET_ROWS` result (whose data aliases the base cache), creating a direct scheduler-visible write-to-read edge.
- Each unrolled variant replays A/B/A with different inputs and row slots. All inspected outputs are marked `ggml_set_output` to prevent allocator liveness reuse.
- Runtime/library provenance: `runtime-perf-v14b`, build manifest at `runtime-perf-v14b/runtime-manifest.json`; test binary SHA256 `7606ac950bf306a630f31018589f450d8b50b42658ca5b87e8e9d1fc995c30ef`.

## Results

- CPU preflight: sequential, model-order alias, and versioned-edge A/B/A parity all PASS bitwise. Deliberate read-first alias mismatch reproduced.
- Single CUDA, physical GPU3: sequential, model-order alias, and versioned-edge A/B/A parity all PASS bitwise. Deliberate read-first alias mismatch reproduced. CUDA graph path logs show capture and replay on the unrolled graph.
- Meta3, physical GPUs 3/4/5 (PCI bus `0035:03`, `0035:04`, `0035:05`): same bitwise passes and read-first mismatch. All Meta-assigned `SET_ROWS`, `GET_ROWS`, and `ARGMAX` checks passed. Each of the three child CUDA contexts logged graph capture and replay.
- Model-order alias is correct for the observed insertion order (depth-0 SET_ROWS nodes before GET_ROWS). However, the underlying graph has no direct `SET_ROWS -> GET_ROWS` edge when GET_ROWS consumes only the base alias. Deliberate read-first ordering creates nodes `read=6`, `SET_ROWS=18` and returns stale/poisoned rows. The versioned-edge graph reports `direct_write_read_edge_k=1` and `..._v=1` and matches the sequential reference.
- No model, engine build, or full-model request was run. GPU usage ended at 0 MiB on GPUs 0–2/4–5 and 1 MiB CUDA context residue on GPU3; all cards were 38–39 C and 0% utilization at check.

## Commands

Build: `fixtures/mtp-unrolled-kv-dependency-20261015/build.sh`

CPU preflight:
`LD_LIBRARY_PATH=$R/lib OMP_NUM_THREADS=1 mtp-unrolled-kv-dependency cpu`

Single CUDA (physical GPU3 UUID):
`CUDA_VISIBLE_DEVICES=GPU-fadb0459-4d30-907b-289a-862b3014f165 GGML_CUDA_GRAPH_OPT=1 GGML_FT_GRAPH_CACHE=1 GGML_CUDA_GRAPH_PATH_LOG=1 GGML_CUDA_NODE_TRACE=1 LD_LIBRARY_PATH=$R/lib OMP_NUM_THREADS=1 timeout 100s mtp-unrolled-kv-dependency cuda`

Meta3 (physical GPUs3/4/5 UUIDs): same env and command with `CUDA_VISIBLE_DEVICES=GPU-fadb0459-4d30-907b-289a-862b3014f165,GPU-760ce501-32f6-69b6-215b-1020da68bbfc,GPU-d3558623-deca-c702-3cc0-01b4aa9c667a` and mode `meta3`.

## Artifacts

- `fixtures/mtp-unrolled-kv-dependency-20261015/fixture.cpp`
- `fixtures/mtp-unrolled-kv-dependency-20261015/build.sh`
- `fixtures/mtp-unrolled-kv-dependency-20261015/cpu.log`
- `fixtures/mtp-unrolled-kv-dependency-20261015/cuda-single.log`
- `fixtures/mtp-unrolled-kv-dependency-20261015/meta3-island-b.log`
- `fixtures/mtp-unrolled-kv-dependency-20261015/README.md`
