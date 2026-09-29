# Qwen3.8-27B AWQ TP3 + native MTP4 on IBM AC922 (POWER9 × 3× V100-SXM2-16GB)

The same lossless TP3 head-replication fork as the
[DFlash2 trial](../tp3-dflash2-trial/README.en.md), this time driving the
model's **native single-layer MTP head** instead of the DFlash2 drafter. The
path passed a bounded trial and **replaced DFlash2 in production the same
night**: KV pool 227,555 tokens (2.3× the DFlash2 production pool), 64k
single-stream decode 69.74 tok/s, 4×50k all-live decode 116.30 tok/s, OpenAI
and Anthropic endpoints both HTTP 200, DFlash kept as a one-command rollback.
Measured 2026-09-29/30. Apache-2.0. 中文全程记录见
[README.md](README.md)。

## Why switch: DFlash2's context-price

DFlash2 served at 110 tok/s single stream, but its drafter is a separate
3.58 GiB block-diffusion model, and BF16 draft KV + hybrid GDN paging roughly
doubles per-token KV cost. With DFlash2 attached, the TP3 KV pool fell from
398,857 (bare TP3) to **99,864** tokens in production — fewer than the old
TP2 service (191,030).

The native MTP head is a single BF16 layer, **0.79 GiB**, sharing the
target's embedding / LM head by default
(`VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS=1`). Read-only projection: if TP3 can run
native MTP, the pool should return to the 250k–350k range.

## Three blockers, same fix shape as the target model

1. **MTP re-reads the checkpoint** — the draft config does not inherit the
   in-process TP3 head replication and constructs 24 Q / 4 KV, tripping the
   TP3 Qwen attention assertion (`qwen3_next.py:489`).
2. **`fc` is 5120-wide** — `ColumnParallelLinear` divides by TP degree;
   5120 % 3 ≠ 0. Fix: `disable_tp=True`, every rank holds the full fc.
3. **CLI `--hf-overrides` does not reach the draft** —
   `speculative.py:883` rebuilds the draft config from
   `SpeculativeConfig.hf_config_override` only; the replication flag and
   nested text overrides (36 Q / 6 KV, GDN 18/54, MLP 17664) must be
   propagated before draft architecture construction.

Vocab 248320 % 3 = 1: reuse the target's `padding_size=192` (→ 248448,
82816/rank). Default PP1 MTP shares target IO, so no separate MTP vocab
allocation is needed. Weights are expanded in memory (heads replicated,
MLP zero-padded 17408→17664); **checkpoint files stay untouched**. Diff:
`patches/` (3 commits on top of the DFlash2 trial patch), plus
`tools/tests/check_mtp_tp3_weights.py`, which passed all seven tensor-shape
and half-output checks before startup.

## Bounded trial (isolated, 0.90 utilization, port 8003)

Startup succeeded: **7.79 GiB** model loading per GPU (smaller than
DFlash2's 8.66), KV pool **242,119 tokens**.

| Prompt | TTFT | Prefill | Decode | Total |
|---|---:|---:|---:|---:|
| 1k (128 out) | 0.653 s | 1531.94 tok/s | 61.52 tok/s | 2.717 s |
| 64k (128 out) | 35.693 s | 1793.05 tok/s | **69.69 tok/s** | 37.516 s |

Draft acceptance **198/236 = 83.9%** (DFlash2: 82.2%). vs DFlash2 depth 4:
1k is 21% slower, 64k is **11% faster** — DFlash verification cost grows
with context, MTP does not.

## Streaming concurrency waves and the "all-live decode" metric

Non-streaming waves can only record wall-clock time, so
[`tools/bench_concurrent_token_ids.py`](tools/bench_concurrent_token_ids.py)
consumes streamed token IDs and reports the **all-live decode** rate (only
the interval where every client is decoding), per-client TTFT and decode
medians, and the end-to-end output rate separately.

At 0.90 with longer outputs (512 per 1k request, 256 per long request):

| Wave | All-live decode | Per-client median | E2E output | Min free |
|---|---:|---:|---:|---:|
| 2×1k×512 | 153.38 tok/s | — | 129.75 tok/s | 985 MiB |
| **4×1k×512** | **367.47 tok/s** | 81.03 | 232.93 tok/s | 577 MiB |
| 2×64k×256 | 84.73 tok/s | 37.87 | 7.13 tok/s | 275 MiB |
| 4×60k×256 | (no overlap) | — | 6.24 tok/s | 249 MiB |

Two findings: **4-way short-context aggregate 367 tok/s is ~2× the DFlash2
30-minute figure (~180)**; and the 4×60k "4 running" is an illusion —
chunked prefill starves decode, one client's first token arrived 161 s in,
and no interval had all four decoding simultaneously.

## Utilization scan 0.90 / 0.92 / 0.91

| Util | KV pool | 2×1k | 4×1k | 2×64k | 4×60k | Verdict |
|---|---:|---:|---:|---:|---:|---|
| 0.90 | 242,119 | 153.38 | 367.47 | 84.73 | 6.24 E2E | baseline |
| 0.92 | 256,682 | 179.95 | 352.48 | not run | **skipped** | 237 MiB left after 4×1k |
| 0.91 | 249,400 | 181.50 | 370.35 | 84.67 | **96.22** | **winner** |

0.91 passed 4×60k (133.8 s, all-live 96.22 tok/s, min 183 MiB, ≤61 °C);
0.92's short-input headroom was too narrow to justify the long wave — the
best-looking configuration and the production-safe one are not the same.

## Production switch, and "production ≠ trial"

Install a separate `ac922-coding-api-tp3-mtp.service` (same port 8000, same
key/middleware), DFlash unit kept disabled as rollback. The first 0.91
production start captured **1.13 GiB** of CUDA Graph per GPU (vs 0.37 in the
trial — production middleware changes graph shapes) and left only **37–38
MiB** idle. No request was sent in that state; the unit was backed up and
lowered to **0.88**.

At 0.88: KV pool **227,555 tokens**, CUDA Graph 0.37 GiB, idle **1.29 GiB**
per GPU; unauthorized listing → 401; authorized OpenAI chat and Anthropic
messages → 200. First cold 64k request decoded at 41.10 tok/s; a second
cold-prefix request with a new cache salt decoded at **69.74 tok/s** —
matching the trial, so the 41.10 was cold-start, not production regression.
Production 4×50k×256: all four completed, max Running=4/Waiting=0, all-live
**116.30 tok/s** (trial 123.56; difference = auth/middleware), min free
**359 MiB/card**, max 60 °C.

## Rollback

```bash
ssh ac922 'systemctl --user disable --now ac922-coding-api-tp3-mtp.service \
  && systemctl --user enable --now ac922-coding-api-tp3.service'
```

## Not tested

Production 4×60k wave; 30-minute stability run; token-by-token quality vs
DFlash2/TP2; dynamic speculative depth (fixed at 4).

## Reproduce

```bash
git am ac922/tp3-mtp4-trial/patches/0001-*.patch   # on top of the DFlash2 patch set
git am ac922/tp3-mtp4-trial/patches/0002-*.patch
git am ac922/tp3-mtp4-trial/patches/0003-*.patch
python tools/tests/check_mtp_tp3_weights.py        # read-only shape gate
python tools/bench_concurrent_token_ids.py --clients 4 --prompt-length 50000 \
  --max-tokens 256 --base http://127.0.0.1:8000 --key-file /path/to/key
```

All host/port defaults are `127.0.0.1`; override via environment for your
own deployment. Raw evidence: [`results/`](results/) (per-request JSON with
TTFT/decode_tps, Prometheus counters, server logs) and
[`results/MTP-TP3-1K-64K.md`](results/MTP-TP3-1K-64K.md) (the measured
report this document summarizes).
