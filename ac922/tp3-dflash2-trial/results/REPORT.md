# AC922 Qwen3.8-27B TP3 + DFlash2 trial (2026-09-29)

Source baseline: `db292f9a4`; isolated Mac worktree:
`<worktree>`. Trial source and temporary
runtime were staged under `/dev/shm/ac922-tp3` on `ac922`; no checkpoint files,
production runtime package, BMC firmware, or hardware fan thresholds were edited.

## TP3 target model

- Original target: 24 Q / 4 KV heads; 16 full-attention and 48 GDN layers.
- Runtime-only exact head replication: 36 Q / 6 KV, GDN 18 key / 54 value.
  Original and copied output-projection columns each receive half their source
  scale. Shared norms stay shared. MLP width 17408→17664 adds 256 zero neurons;
  vocabulary remains 248320 with internal padding to 248448.
- The current checkpoint's Q projection includes a per-head output gate, so
  the copied Q block is 512 channels per head. Quantized int4 zero points are
  packed along the output axis; their aligned blocks are copied intact.
- The 0.95 GPU utilization startup reported 449142 KV tokens but failed during
  CUDA Graph capture. At 0.90 utilization, the full 200k trial served with
  398857 KV tokens.

Greedy parity: all 20 matched token-for-token against the original TP2 server,
with seven 1k, seven 8k, and six 64k deterministic prompts. Each generated 32
tokens. Paired token logprob absolute differences had median 0.00135 and
maximum 0.259; runtime arithmetic is not bitwise identical.

| Prompt | TP2 median total | TP3 median total |
|---|---:|---:|
| 1k | 1.30 s | 1.19 s |
| 8k | 5.11 s | 4.21 s |
| 64k | 42.67 s | 35.53 s |

Cold long-context run at TP3 (0.90 utilization):

| Prompt | TTFT | Prefill | Decode |
|---|---:|---:|---:|
| 100k | 62.15 s | 1609 tok/s | 30.05 tok/s |
| 199k | 167.47 s | 1188 tok/s | 24.71 tok/s |

The previously supplied TP2 figures used different batch-token settings;
elapsed-time differences cannot be attributed to TP topology alone.

## DFlash2 drafter

This fork accepts `draft_tensor_parallel_size=1` in configuration but builds
the DFlash model with the target's TP3 process group. The five-layer BF16
drafter was therefore expanded in memory from 32 Q / 8 KV to 36 Q / 9 KV,
with the same half-output rule, MLP zero padding, and internal vocabulary
padding. It shares the target embedding and LM head. The generic KV planner
requires an LCM page for the target FP8 and draft FP16 caches; this trial has
an explicit, bounded opt-in for that case.

At depth 7, 0.89 utilization, 64k maximum length, the server reported 85333
KV tokens. The three cold, 128-token completions (with distinct cache salts)
yielded:

| Prompt | TTFT | Decode | Accepted / drafted |
|---|---:|---:|---:|
| 1k | 0.50 s | 110.28 tok/s | combined below |
| 8k | 3.49 s | 89.78 tok/s | combined below |
| 64k | 37.89 s | 79.37 tok/s | combined below |

Together they accepted 325/469 proposed tokens (69.3%). The old `bench357.py`
counts SSE messages, which undercounts tokens when a DFlash message carries
several; `bench_stream_token_ids.py` counts the streamed token ID deltas.
Depth 7 could schedule at most two of four simultaneous requests even with
1k prompts. A 0.90 utilization 64k trial served short prompts but OOMed near
59k prefill. A 200k DFlash startup failed its KV budget check (4.79 GiB
required, 3.05 GiB available at 0.86 utilization).

At depth 3, a five-group KV layout and 0.93 utilization reported 122880 KV
tokens. A 35-second pilot completed 23 requests with zero errors and 32 metric
samples at `Running=4`, `Waiting=0`; maximum GPU temperature was 54°C.
Depth-3 cold decode was 64.9 / 46.2 / 53.8 tok/s at 1k / 8k / 64k, with
268/357 draft tokens accepted (75.1%) in that run. The depth-3 configuration
trades long-context single-stream speed for real four-way capacity.

The required four-way stability run completed in **1803 seconds** with 1265
completed requests, 323840 output tokens, zero request errors, max Running=4,
max Waiting=0, and 1594 one-second samples at Running=4. The maximum GPU
temperature was 57°C. Recalculation from the copied request and metrics JSONL
matched every field of `stability-30m.summary.json`; the trial service stayed
HTTP 200 afterward. Its log had no NCCL timeout, fatal engine error, traceback,
or CUDA OOM during the run, and the host kernel log showed no new EEH, I/O
error, or Xid in the checked interval. This proves four-way concurrency for
1k prompts and 256-token completions. It does not prove four simultaneous
64k contexts.

## Operational state and evidence

- The original `ac922-kv-api.service` on port 8001 and ComfyUI on GPU2 were
  stopped for this authorized trial. The user waived automatic restoration.
- The trial binds only `127.0.0.1:8003`. No production switch has been made.
- BMC `ac922-boot-guard.service` had already fallen back to 4000 RPM after a
  60°C event and a `systemctl` timeout; fan watchdog was active, fan alarms 0.
  This trial did not restart the BMC or change fan masks/targets.
- `tp2-baseline-20.jsonl`, `tp3-greedy-20.jsonl`,
  `tp3-long-baseline.jsonl`, `bench-dflash-cold.jsonl`, and the paired
  `dflash-cold-*.prom` snapshots hold the measurements above.
- `stability-pilot-d3-093.*` is the four-way pilot;
  `stability-30m.requests.jsonl`, `.metrics.jsonl`, `.summary.json`, and `.log`
  are the complete 30-minute evidence.
- Further tuning is pending: depth 3 gives real four-way capacity but 8k/64k
  single-stream decode remains below 60 tok/s; depth 7 exceeded 60 tok/s at
  all three sampled lengths but scheduled only two simultaneous requests under
  its tested group-8 memory profile. DFlash2 200k did not fit the measured
  3×16GB memory budget. Port 8001 has not been switched to this trial.
