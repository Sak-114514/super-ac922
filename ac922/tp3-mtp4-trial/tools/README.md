# Tools for the TP3 + native MTP4 trial

These complement the [DFlash2 trial tools](../../tp3-dflash2-trial/tools/README.md)
on the same fork (upstream `db292f9a4` + the DFlash2 patch set + the three
MTP patches in [`../patches/`](../patches/)). All scripts default to
`127.0.0.1` and read credentials from a key file; override paths/ports via
environment variables for your own host.

## Weight gate (read-only)

- `tests/check_mtp_tp3_weights.py` — streams the local
  `model-mtp.safetensors` and verifies every tensor that the in-memory TP3
  expansion will touch: the seven expandable tensors' shapes, the fc
  (5120-wide, `disable_tp=True`), and half-scaling of replicated
  projections. Run it **before** starting the server; it never writes.
- `tests/check_tp3_config.py` — prints the effective TP3 config the fork
  will build (36 Q / 6 KV / GDN 18/54 / MLP 17664 / vocab padding 192) for
  target and draft.

## Serving

- `start_coding_api_tp3_gpu012.sh` / `ac922-coding-api-tp3.service` — the
  DFlash2 production unit kept as rollback (disabled while MTP is active).
- `start_coding_api_tp3_mtp_gpu012.sh` / `ac922-coding-api-tp3-mtp.service`
  — the native-MTP4 production unit: TP3 on GPU0/1/2, native MTP depth 4,
  KV `fp8_e4m3`, same API key/parser/middleware as the DFlash unit,
  `AC922_GPU_MEMORY_UTILIZATION` honored (measured production value: 0.88).
- `serve_tp3_trial.sh` — isolated RAM-overlay trial launcher; the `mtp`
  mode binds `127.0.0.1:8003` so it can run while production serves.

## Measurement

- `bench_concurrent_token_ids.py` — the tool this trial was re-run with.
  Streams the response and counts **returned token IDs**, reporting
  per-client TTFT, per-client decode median, the **all-live decode** rate
  (only intervals where every client is decoding simultaneously), and the
  end-to-end output rate. Counting SSE chunks or wall-clock time
  under/over-states speculative throughput and hides prefill/decode
  interleaving. Supports `--key-file` for authenticated production
  endpoints.
- `bench_stream_token_ids.py` — single-request streamed counter (1k/64k
  columns in the reports).
- `stability_tp3.py` — one-wave concurrency checks with scheduler
  Running/Waiting, per-request completion, and GPU-temperature sampling.

The raw per-request JSON files produced by these tools (TTFT, decode_tps,
generated token counts, per-second scheduler samples) are in
[`../results/`](../results/).
