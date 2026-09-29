# Qwen3.8-27B AWQ TP3 + DFlash2 on IBM AC922 (POWER9 × 3× V100-SXM2-16GB)

Lossless tensor-parallel-3 serving of a hybrid-attention 27B model on three
16GB V100s, plus DFlash2 speculative decoding — **110 tok/s single-stream,
~180 tok/s sustained at 4 concurrent, bit-identical greedy output to the TP2
baseline**. All numbers below are measured on the hardware listed, on
2026-09-29. Apache-2.0, like the vLLM fork this is based on.

[中文说明见下](#中文说明)

## Why this is non-trivial

Qwen3.8-27B is a hybrid model: 64 layers = 16 full-attention + 48 GDN
(linear attention). Its dimensions do not divide by 3:

| | original | after opt-in replication | TP3 per card |
|---|---:|---:|---:|
| Q heads | 24 | 36 | 12 |
| KV heads | 4 | 6 | 2 |
| GDN key / value heads | 16 / 48 | 18 / 54 | 6 / 18 |
| MLP intermediate | 17408 | 17664 (zero-padded) | 5888 |
| Vocabulary | 248320 | padded to 248448 | — |

The fix is **load-time weight surgery, not TP-code changes**: every replicated
head gets a bitwise-identical copy of the source weights, and BOTH the source
and the copied output-projection columns are scaled by 1/2, so the layer output
is mathematically unchanged (`W·h = (W/2)·h + (W/2)·h′`). Scales of the int4 /
int8 packed quantization are halved in fp16 — an exact operation. The same
treatment is applied to the DFlash2 drafter (32 Q / 8 KV → 36 Q / 9 KV), which
this fork builds on the target's TP3 process group even when configured TP1.

A third blocker was upstream vLLM: the hybrid GDN + draft KV layout has layer
page sizes that the generic planner refuses to unify (`NotImplementedError`).
The patch adds a bounded, opt-in common-page layout (group size 16) that admits
four simultaneous requests.

Everything is opt-in via `hf-overrides` / env flags; checkpoint files are never
modified.

## Results (all measured, cold start unless noted)

Greedy parity vs the production TP2 server: **20/20 prompts token-for-token
identical** (7×1k, 7×8k, 6×64k, 32 tokens each). Paired logprob differences:
median 0.00135, max 0.259 — TP sharding changes summation order; greedy
choices are unaffected.

### Request latency (median total, 32-token generations)

| Prompt | TP2 baseline | TP3 (this work) |
|---|---:|---:|
| 1k | 1.30 s | **1.19 s** |
| 8k | 5.11 s | **4.21 s** |
| 64k | 42.67 s | **35.53 s** |

### Long context (fresh prefill, TP3)

| Prompt | TTFT | Prefill | Decode |
|---|---:|---:|---:|
| 100k | 62.15 s | 1609 tok/s | 30.05 tok/s |
| 199k | 167.47 s | 1188 tok/s | 24.71 tok/s |

KV pool at 0.90 utilization, 200k max length: **398,857 tokens** (fp8_e4m3).

### DFlash2 speculative decoding (depth 7, single stream)

| Prompt | Decode | vs bare decode |
|---|---:|---:|
| 1k | **110.3 tok/s** | 2.9× |
| 8k | 89.8 tok/s | — |
| 64k (prefix-cached) | 79.1 tok/s | — |

Cumulative acceptance: **82.2%** (139,128 / 169,227 drafted), ≈ 5.75 tokens
per step at depth 7. Greedy outputs remain identical to the TP2 baseline with
the drafter enabled.

### Sustained 4-way concurrency (30-minute stability run, depth 3)

- 30 minutes, **1,250+ requests, zero errors**, Running=4 / Waiting=0
  throughout
- Aggregate **≈ 180 tok/s** (≈ 45 tok/s per stream), no drift over time
- GPU 50–55 °C across all three cards

Historical reference on the same machine: TP2+MTP3 4-way = 130.9 tok/s;
llama.cpp (3 GPUs, fp16 KV) single stream = 34.6 tok/s.

## Hardware / software

- IBM Power System AC922 (8335-GTW): 2× POWER9, 3× Tesla V100-SXM2-16GB used
  of 6. GPUs 0/1/2 form one NVLink2 triangle (NUMA 0); GPUs 3/4/5 the other.
  Any 4-GPU TP must cross sockets — hence TP3 on one island.
- Fork of a 1Cat-vLLM tree at `db292f9a4`, Torch 2.12.0a1 (power
  ppc64le / CUDA 12.4), Python 3.11, fp8_e4m3 KV cache, custom SM70
  Flash-V100 E4M3 kernels.

## Repo layout

```
patches/0001-Make-AC922-TP3-inference-trials-reproducible….patch
        the complete change set: opt-in head replication (target + drafter),
        KV page unification, launchers, and tests
tools/  serve_tp3_trial.sh   — isolated launcher (all knobs via env vars)
        compare_greedy_tp2_tp3.py — greedy parity harness
        bench_stream_token_ids.py — streaming decode benchmark
        stability_tp3.py          — concurrency/stability driver
results/REPORT.md             — full measurement report (kept as recorded)
        *.jsonl                — per-request benchmark records
        *.prom                 — Prometheus snapshots (draft/accept counters)
        logs/serve-dflash-*.log, stability-30m.log
docs/   NOTES.md              — pitfalls encountered, in order
```

## Reproduce

1. Start from the same fork commit (`db292f9a4`) and apply the patch:
   `git am patches/0001-*.patch`
2. Weights: an AWQ int4 quant of Qwen3.8-27B (compressed-tensors) and the
   DFlash2 drafter (`z-lab/Qwen3.8-27B-DFlash2`, pinned `50307d4c`). Checksums
   we used are in the drafter MANIFEST upstream; no weights are included here.
3. `tools/serve_tp3_trial.sh` — every path and knob is an env override
   (`AC922_MODEL_PATH`, `AC922_DFLASH_MODEL`, `AC922_MAX_MODEL_LEN`,
   `AC922_GPU_MEMORY_UTILIZATION`, `AC922_DFLASH_DEPTH`, …). Head replication
   is opt-in via `hf-overrides {"ac922_tp3_head_replication": true, ...}`.
4. Parity: `tools/compare_greedy_tp2_tp3.py <url> <prompts> --reference <tp2>`
5. Throughput: `tools/bench_stream_token_ids.py`, concurrency:
   `tools/stability_tp3.py`

The trial runtime is isolated under `/dev/shm` on the target host; nothing in
the production runtime, checkpoint files, BMC configuration, or fan policy is
touched.

## 中文说明

在 IBM AC922(POWER9,6×V100-SXM2-16GB,取 NVLink 三角岛 0/1/2)上,把
Qwen3.8-27B AWQ 混合注意力模型(GDN + 全注意力)以 TP3 无损部署,并叠加
DFlash2 投机解码:

- **无损头复制**:加载期把 24Q/4KV/16GDN-key 复制到 36/6/18,GDN value 48→54,
  MLP 内存中零填充 17408→17664,词表内部补齐;源列与复制列各乘 ½,输出数学
  严格相等,checkpoint 文件不改动。DFlash2 drafter 同样处理(32Q/8KV→36Q/9KV)。
- **KV 页统一**:上游 vLLM 对「GDN 混合层 + drafter」的页大小统一会直接抛
  `NotImplementedError`;补丁增加了有界 opt-in 公共页方案(group 16),
  使四请求可同时运行。
- **结果**:greedy 输出与 TP2 逐 token 一致(20/20);单流有效解码
  110 tok/s(1k)→ 79 tok/s(64k);接受率 82.2%;四并发 30 分钟
  ~180 tok/s 零错误;199k 预填充较 TP2 提速 2.2 倍。
- 所有数字为 2026-09-29 实测,原始逐条记录在 `results/`。

## Credits

- Base: 1Cat-vLLM fork at `db292f9a4` (Apache-2.0, same as upstream vLLM)
- Model: Qwen3.8-27B (AWQ compressed-tensors quant); drafter:
  `z-lab/Qwen3.8-27B-DFlash2`
- Hardware: IBM AC922 (8335-GTW), 3× Tesla V100-SXM2-16GB (NVLink island)

## License

Apache-2.0. The patch touches only vLLM-licensed files plus new files under
the same license.
