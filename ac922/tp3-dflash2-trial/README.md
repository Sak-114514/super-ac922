# Qwen3.8-27B AWQ TP3 + DFlash2 @ IBM AC922(POWER9,3× V100-SXM2-16GB)

> English abstract: lossless tensor-parallel-3 serving of the hybrid-attention
> Qwen3.8-27B AWQ model on three 16GB V100s, plus DFlash2 speculative decoding.
> **110 tok/s single stream, ~180 tok/s sustained at 4-way concurrency
> (30-minute zero-error run), greedy parity 20/20 vs the TP2 baseline.**
> All numbers measured 2026-09-29 on the hardware below. Apache-2.0.

本文记录 2026-09-29 在一台 IBM AC922(8335-GTW)上完成的完整服务战役:
把混合注意力架构(GDN + 全注意力)的 Qwen3.8-27B AWQ 以 **TP3 无损部署**,
并叠加 **DFlash2 投机解码**。所有数字均为当日实测,原始逐条数据在
[`results/`](results/)。

## 为什么这事不平凡

Qwen3.8-27B 是混合架构:64 层 = 16 层全注意力 + 48 层 GDN(线性注意力)。
它的关键维度**都不能被 3 整除**:

| | 原始 | 加载期复制后 | TP3 每卡 |
|---|---:|---:|---:|
| Q 头 | 24 | 36 | 12 |
| KV 头 | 4 | 6 | 2 |
| GDN key / value 头 | 16 / 48 | 18 / 54 | 6 / 18 |
| MLP 中间维 | 17408 | 17664(零填充) | 5888 |
| 词表 | 248320 | 内部补齐 248448 | — |

解法是**加载期权重手术,不动任何 TP 分片代码**:每个被复制的头逐位拷贝
源权重,**源列块与复制列块各乘 ½**,层输出数学严格相等
(`W·h = (W/2)·h + (W/2)·h′`)。int4/int8 的 scale 在 fp16 上乘 ½ 是精确
操作,所以量化域内复制同样无损。DFlash2 drafter 也做了同样手术
(32 Q / 8 KV → 36 Q / 9 KV)——这个 fork 即使配置 TP1 也会把 drafter 建
在目标的 TP3 进程组上,是个隐藏坑。

第三个拦路虎在 vLLM 上游:GDN 混合层 + drafter 的 KV 页大小无法被通用
规划器统一(直接抛 `NotImplementedError`)。补丁增加了一个**有界 opt-in
公共页方案**(组大小 16),让四路请求可以同时运行。

一切通过 `hf-overrides` / 环境变量显式开启;**checkpoint 文件零改动**。

## 实测结果(除注明外均为冷启动)

贪心对照:对现役 TP2 服务 **20/20 条逐 token 完全一致**(7×1k、7×8k、
6×64k,各生成 32 token)。成对 logprob 差:中位 0.00135、最大 0.259——
TP 分片改变求和顺序的正常数值差,贪心选择不受影响。

### 请求时延(中位总耗时,32 token 生成)

| Prompt | TP2 基线 | TP3(本方案) |
|---|---:|---:|
| 1k | 1.30 s | **1.19 s** |
| 8k | 5.11 s | **4.21 s** |
| 64k | 42.67 s | **35.53 s** |

### 长上下文(全新预填充)

| Prompt | TTFT | 预填充 | 解码 |
|---|---:|---:|---:|
| 100k | 62.15 s | 1609 tok/s | 30.05 tok/s |
| 199k | 167.47 s | 1188 tok/s | 24.71 tok/s |

0.90 利用率、200k 窗口下 KV 池:**398,857 tokens**(fp8_e4m3)。

### DFlash2 投机解码(深度 7,单流)

| Prompt | 解码速度 | 对比裸解 |
|---|---:|---:|
| 1k | **110.3 tok/s** | 2.9× |
| 8k | 89.8 tok/s | — |
| 64k(前缀命中) | 79.1 tok/s | — |

累计接受率 **82.2%**(139,128 / 169,227),深度 7 下步均 ~5.75 token。
开启 drafter 后贪心输出仍与 TP2 逐 token 一致。

### 四并发 30 分钟稳定性(深度 3)

- 30 分钟、**1,265+ 请求、零错误**,全程 Running=4 / Waiting=0
- 聚合 **≈ 180 tok/s**(每路 ≈ 45),无时间衰减
- 三卡 GPU 全程 50–55 °C

同机历史参照:TP2+MTP3 四并发 130.9 tok/s;llama.cpp(3 卡,fp16 KV)
单流 34.6 tok/s。

## 深度-并发权衡(参数扫描)

| 深度 | 最大并发 | 单流速度 | 聚合(4路) |
|---:|---:|---:|---:|
| 7 | 2(实测) | 110.3 | — |
| 4 | 3 @0.93 → **4 @0.95**(实测) | ~65-80 | ~169-180 |
| 3 | 4(实测) | 64.9 | ~180 |

规律:每步验证位置数(深度 × 路数)是调度器硬约束;投机解码的 KV 读取
被候选 token 共享,所以长窗下深度反而摊薄带宽成本。

## 硬件 / 软件

- IBM Power System AC922(8335-GTW):2× POWER9,6 卡中用 3× V100-SXM2-16GB。
  GPU 0/1/2 为一个 NVLink2 三角(NUMA 0),3/4/5 为另一个;任何四卡 TP 都要
  跨插槽——所以选单岛 TP3。
- 1Cat-vLLM fork @ `db292f9a4`,Torch 2.12.0a1(ppc64le / CUDA 12.4),
  Python 3.11,fp8_e4m3 KV,自研 SM70 Flash-V100 E4M3 内核。

## 仓库结构

```
patches/0001-*.patch   完整变更集:目标+drafter 的 opt-in 头复制、KV 页统一、
                       启动器与回归测试(对 db292f9a4 直接 git am)
tools/                 serve_tp3_trial.sh(全环境变量可调启动器)
                       compare_greedy_tp2_tp3.py(贪心对照)
                       bench_stream_token_ids.py(流式基准)
                       stability_tp3.py(并发/稳定性驱动)
results/               REPORT.md + 全部逐条记录(jsonl)+ Prometheus 快照 + 日志
docs/NOTES.md          全部踩坑按时间序(视觉塔/词表/MLP/OOM/drafter 词表/页大小/并发上限)
```

## 复现步骤

1. 从同一 fork 提交(`db292f9a4`)开始,应用补丁:`git am patches/0001-*.patch`
2. 权重:Qwen3.8-27B 的 AWQ int4 量化(compressed-tensors)+ DFlash2 drafter
   (`z-lab/Qwen3.8-27B-DFlash2`,锁定 `50307d4c`)。权重不含在本仓库,校验
   和见上游 MANIFEST。
3. `tools/serve_tp3_trial.sh`——所有路径和参数都是环境变量
   (`AC922_MODEL_PATH`、`AC922_DFLASH_MODEL`、`AC922_MAX_MODEL_LEN`、
   `AC922_GPU_MEMORY_UTILIZATION`、`AC922_DFLASH_DEPTH`…)。头复制通过
   `hf-overrides {"ac922_tp3_head_replication": true, ...}` 显式开启。
4. 一致性:`tools/compare_greedy_tp2_tp3.py <url> <prompts> --reference <tp2>`
5. 吞吐:`tools/bench_stream_token_ids.py`;并发:`tools/stability_tp3.py`

试验运行时隔离在目标机的 `/dev/shm`;生产运行时、checkpoint 文件、BMC
配置、风扇策略一律不碰。

## 特别注意(踩坑速查)

- 16GB 卡:0.95 利用率报的 KV 池在 CUDA Graph 捕获时还会再要 ~810 MiB,
  留够头寸或降到 0.90。
- 贪心一致 ≠ 位元相等:分片求和顺序必然带来微小 logprob 差,token 一致
  才是契约。
- 测全新预填充要加 `cache_salt`,否则前缀缓存会把 38 秒变成 4 秒。
- 投机解码下,流式 chunk 可能一次携带 6+ token,数 chunk 会把 110 tok/s
  数成 16 tok/s。
- 完整版见 [`docs/NOTES.md`](docs/NOTES.md)。

## 致谢

- 基线:1Cat-vLLM fork @ `db292f9a4`(Apache-2.0)
- 模型:Qwen3.8-27B(AWQ compressed-tensors 量化);drafter:
  `z-lab/Qwen3.8-27B-DFlash2`
- 硬件:IBM AC922(8335-GTW)

## 许可

Apache-2.0。补丁只涉及 vLLM 许可下的文件与同许可新增文件。
