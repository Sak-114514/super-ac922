# super-ac922 中文说明(AC922 POWER9 / V100 实验记录)

> 本分支(`ac922-ppc64le`)是 [1Cat-vLLM](README.md) 在 IBM AC922
> (POWER9 / Tesla V100-SM70)上的**构建快照与实验记录仓库**。英文原版
> README 见 [README.md](README.md)。

## 本分支导览

| 内容 | 入口 |
|---|---|
| **TP3 + DFlash2 服务战役(2026-09-29)** | [ac922/tp3-dflash2-trial/README.md](ac922/tp3-dflash2-trial/README.md) |
| **TP3 + 原生 MTP4 试车与生产切换(2026-09-29/30)** | [ac922/tp3-mtp4-trial/README.md](ac922/tp3-mtp4-trial/README.md) |
| ppc64le 构建快照、依赖 wheel 清单 | [ac922/README.md](#ac922-ppc64le-构建快照中文版) |
| 家用温控踩坑(风扇策略断电事故复盘) | [ac922/PITFALLS-FAN-THERMAL.md](ac922/PITFALLS-FAN-THERMAL.md) |
| 低内存/内存管理踩坑(幽灵 DIMM、swap、图捕获头寸) | [ac922/PITFALLS-MEMORY-LOWMEM.md](ac922/PITFALLS-MEMORY-LOWMEM.md) |

## 本分支核心成绩(2026-09-29 实测,Qwen3.8-27B AWQ)

在 AC922(POWER9,6×V100-SXM2-16GB,取 NVLink 三角岛 0/1/2)上:

- **TP3 无损部署**:混合注意力模型(16 全注意力 + 48 GDN)的头加载期
  无损复制,贪心输出与现役 TP2 **20/20 逐 token 一致**
- **单流有效解码 110 tok/s**(DFlash2 深度 7,接受率 82.2%)——裸解码的
  2.9 倍
- **四并发 ~180 tok/s 持续 30 分钟,零错误**(1,265 请求,32 万 token)
- **199k 预填充 167s(TP2 基线 375s,2.2×)**;KV 池 398,857 tokens @ fp8
- GPU 全程 43–57 °C,零热事件

快速上手:

```bash
git clone https://github.com/Sak-114514/super-ac922.git
cd super-ac922
git am ac922/tp3-dflash2-trial/patches/0001-*.patch
```

---

# 1Cat-vLLM 中文说明(完整对照)

> 以下是根 [README.md](README.md) 的逐节中文对照,所有数字与链接保持一致。
> 如有歧义以英文原文为准。

## AC922 ppc64le 构建快照

本分支新增 [POWER9/V100 构建笔记、脚本、依赖哈希与实验性 wheel 清单](ac922/README.md),
派生自上游提交 `db292f9a4`。详见下方 [「AC922 ppc64le 构建快照(中文版)」](#ac922-ppc64le-构建快照中文版)。

## 让 Volta 再快起来(Make Volta Fast Again)

### 面向 NVIDIA Tesla V100 / SM70 的现代 LLM 推理

推荐模型:
QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 ·
RadixArk/Qwen3.8-Flash-Next-NVFP4 ·
incoai/Qwen3.8-27B-DFlash2

**4× Tesla V100 16GB · Qwen3.8-27B-NVFP4 + DFlash2 · ≈260 tok/s**

> Tesla V100 发布于 2017 年。它的 Tensor Core 并没有突然失效——
> **是软件栈放弃了对 SM70 的认真优化。**

1Cat-vLLM 是把 **NVIDIA Volta / SM70 / Tesla V100 当作一等优化目标**的
vLLM 工程分支。我们不满足于"最新模型能在 V100 上启动",我们的目标是:
**让现代模型在 V100 上真正跑快。** 如今 4 张 V100 16GB 可通过 1Cat-vLLM
以约 **260 token/s** 运行 Qwen3.8-27B-NVFP4 + DFlash2。
演示视频:[4× V100 运行 Qwen3.8-27B-NVFP4-DFlash2](https://www.bilibili.com/video/BV1kstb6dEaF/)

> ≈260 tok/s 是真机演示的头条数字,不是通用固定解码速率。下文每个基准都
> 保留各自的硬件、模型、上下文长度、批量、KV 精度、采样策略与投机解码
> 契约。注意力 TFLOP/s、预填充 tok/s、纯目标解码 tok/s、投机解码 tok/s
> 不可互相换算。

## 📊 性能优先

SM70 Flash-V100 现已将 `--kv-cache-dtype fp8` 解析为 E4M3。DFlash2 E4M3
验证使用修复后的 FP32 注意力状态;Qwen3.8 DFlash2 配置默认启用 FP32
logits。需重新构建 Flash-V100 至 precision revision 4,参见
[精度契约与验证](docs/design/sm70_dflash2_fp32_defaults.md)。下方历史
E5M2/FP16 部分性能结果保留原始配置,不代表新精度默认值的速度声明。

### 长上下文注意力:17.92 → 47.1 → ≈60.8 TFLOP/s

| 阶段 | 证据 | 有效因果注意力算力 | 备注 |
|---|---|---:|---|
| 旧生产路径 | v1.2.2 时代基线 | **17.92 TFLOP/s** | V100 长前缀注意力基线 |
| D256 Split-D / N32 | [v1.3.0](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.3.0) | **46.63–47.1 TFLOP/s** | 约为旧生产路径 2.6× |
| GQA 打包宽 QK/PV | [PR #286](https://github.com/1CatAI/1Cat-vLLM/pull/286) / 当前 main | **≈60.8 TFLOP/s** | 6 个 GQA 头打包进更宽的 Tensor Core GEMM |
| 实验上限 | [PR #315](https://github.com/1CatAI/1Cat-vLLM/pull/315) | **≈79 TFLOP/s** | 研究结果,**非 Release/默认质量声明** |

同一代硬件上,代表性长上下文 V100 注意力有效算力从 **17.92 → ≈60.8
TFLOP/s,约 3.4×**。这些数字只统计有效因果 QK/PV 工作,不是整模型 TOPS。

## 🚀 真实模型基准

下表优先收录**完整模型 / API / 纯解码 / 投机解码**测量,而非孤立内核
微基准:

| 模型 | 硬件 / 运行时 | 负载 | 实测结果 | 证据 / 状态 |
|---|---|---|---:|---|
| Qwen3.6-27B-AWQ + MTP4 | 4× V100 · TP4 · E5M2 KV · Flash-V100 · CUDA Graph | 64K 解码 | **100.564 tok/s** | v1.2.2 · AL 4.981 / 99.52% |
| Qwen3.6-27B-AWQ + MTP4 | 同上 | 128K 解码 | **85.258 tok/s** | v1.2.2 · 较无 MTP +87.64% |
| Qwen3.6-27B-AWQ + MTP4 | 同上 · 最大 256K | 261,888 上下文解码 | **49.772 tok/s** | v1.2.2 · AL 5.000 / 100% |
| Qwen3.6-35B-A3B NVFP4 | 4× V100 · TP4 · FP8+W4A16_NVFP4 | 4096/1024 · 无 MTP | **116.99 tok/s** | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) |
| Qwen3.6-35B-A3B NVFP4 + MTP4 | 同上 | 对标 MTP4 | **174.76 tok/s** | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) · 1.49× |
| Qwen3.8-27B-NVFP4 | 4× V100 · TP4 · E4M3 KV · 全 CUDA Graph · 无 MTP | 精确 128K 解码 | **61.834 tok/s** | [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285) |
| Qwen3.8-27B-NVFP4 | 同上 | 精确 256K 解码 | **50.376 tok/s** | [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285) · 实测非推算 |
| Qwen3.8-27B-FP8 | 4× V100 · TP4 · E5M2 KV · 无 MTP | 128K / 256K 解码 | **50.68 / 41.11 tok/s** | [#212](https://github.com/1CatAI/1Cat-vLLM/pull/212) |
| Qwen3.8 Flash-Next-NVFP4 | 4× V100 · TP4 · V2 · 全 CUDA Graph | 8K/512 纯解码 | **80.732 tok/s** | [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415) · 质量审计 |
| Flash-Next-NVFP4 + MTP4 | 4× V100 · TP4 · V2 | 冷 JIT 门 | **138.26 tok/s** | [#389](https://github.com/1CatAI/1Cat-vLLM/pull/389) · AL 4.943 / 98.57% |
| Qwen3.8-27B-NVFP4 + DFlash2 | 4× V100 · TP4 · 生产 API | 历史 web prompt · 512 输出 | **206.06 tok/s 流式解码** | [#422](https://github.com/1CatAI/1Cat-vLLM/pull/422) |
| Qwen3.8-27B-NVFP4 + DFlash2 | 4× V100 · TP4 · 实用 API | MBPP #28 · 自然 EOS | **251.60 tok/s** | [#288](https://github.com/1CatAI/1Cat-vLLM/pull/288) · AL 4.686 |
| DFlash2 + 自适应查找 q16 | 4× V100 · TP4 · opt-in 查找增强 | 重复上下文样本 | **316.27 tok/s** | [#366](https://github.com/1CatAI/1Cat-vLLM/pull/366) · 特殊 opt-in 契约 |
| DeepSeek-V4-Flash | 8× V100 · TP8 · FP8+MXFP4 · 无投机 | 1024/256 | **15.357 ms TPOT ≈ 65.1 tok/s** | [#181](https://github.com/1CatAI/1Cat-vLLM/pull/181) |
| DeepSeek-V4-Flash | 8× V100 · PP2×TP4 | 质量受控端点 | **73.613–73.646 tok/s** | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) |
| GLM-5.3-Flash-NVFP4 | 8× V100 · TP4/PP2 · E4M3 KV · 无 MTP | 1K/256 解码 | **53.016 tok/s** | [#402](https://github.com/1CatAI/1Cat-vLLM/pull/402) |

## 🧪 数据集 / 质量 × 吞吐基准

单纯的 tok/s 会把优化变成刷榜游戏。1Cat-vLLM 同时记录**真实模型吞吐、
数据集得分、自然停止健康度、输出有效性与投机接受率**。

### Qwen3.8-27B-NVFP4 + DFlash2 —— 16K 实用编码门

契约:4× V100 TP4 · NVFP4 目标 · 官方 BF16 DFlash2 drafter · FP8 E5M2
目标 KV · FlashAttention-V100 · 全 CUDA Graph · 前缀缓存 · Mamba 对齐 ·
`temperature=1.0` / `top_p=0.95` / `top_k=20` · `xhigh` 推理 · 16K 自然
EOS 输出上限 · 三个预声明采样种子

| 数据集 | 样本 | Base | Plus | 自然停止 | 聚合输出吞吐 | 平均稳态解码 | 接受(池化/单请求) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MBPP / EvalPlus** | 96 · 93 计分 | **89/93** | **80/93** | 95/96 | **213.539 tok/s** | **236.902 tok/s** | 4.061 / 4.318 |
| **HumanEval / EvalPlus** | 96 | **94/96** | **92/96** | 91/96 | **208.978 tok/s** | **245.645 tok/s** | 3.972 / 4.476 |

证据:[PR #346](https://github.com/1CatAI/1Cat-vLLM/pull/346)。首种子对齐
历史无 DFlash 契约;MBPP+HumanEval 两条路线 Base 62/63、Plus 59/63——
**200+ tok/s 的投机路径不是靠放弃任务质量门换来的**。

### 关于长度截断型失败

部分编码失败是**长推理耗尽 16K 输出预算**所致,并非最终解无效。192 个
输出中 6 个触顶,其中 3 个仍可提取且正确。因此 README 把可执行得分、自然
停止率、输出触顶失败与吞吐分开呈现,不把触顶样本一律当作能力退化。

### 可选精确编码配置

同引擎同中间种子,仅改客户端采样为精确编码档
(`temperature=0.6, top_p=0.95, top_k=20`):

| 数据集 | 温度 1.0 | 温度 0.6 |
|---|---|---|
| MBPP | Base 27/31 · Plus 24/31 | **Base 29/31 · Plus 27/31** |
| HumanEval | Base 31/32 · Plus 31/32 | **Base 31/32 · Plus 31/32** |

80 请求合计:稳态解码 233.187 → **244.520 tok/s**;输出吞吐 195.817 →
201.852;平均接受 4.273 → 4.514;自然停止 72/80 → 70/80。这是**可选
精确编码档**,不是强制全局默认。

## 其他全模型质量门

| 模型 / 路线 | 数据集 / 质量 | 契约下真实吞吐 | 状态 |
|---|---|---:|---|
| Flash-Next-NVFP4 · 无 MTP | GSM8K 15/16 严格 · 16/16 自然停止 | **80.935 tok/s 加权纯解码** | [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415) 已合并 |
| Flash-Next-NVFP4 · MTP4 | HumanEval8 8/8 语义执行 | **150.17 tok/s** | [#398](https://github.com/1CatAI/1Cat-vLLM/pull/398) 研究线 |
| Qwen3.6-35B-A3B NVFP4 + MTP4 | GSM8K 122/128(95.31%)· 0 无效 · 0 重复 | **174.76 tok/s** | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) 已合并 |
| Qwen3.6-35B-A3B NVFP4 + MTP4 | ShareGPT16 SHA 负载 | 120.096 纯解码 / 97.678 E2E / 241.973 预填充 | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) 已合并 |
| DeepSeek-V4-Flash · PP2×TP4 | GSM8K 64/64 · HumanEval 29/32 · LongBench 44.740 | **73.539 tok/s 中位** | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) 严格质控 |
| GLM-5.3-Flash-NVFP4 · 无 MTP | Max 推理 6/8;低推理重跑 2/2 | **53.016 解码 · 266.040 1K 预填充** | [#402](https://github.com/1CatAI/1Cat-vLLM/pull/402) |

## 工具调用 / 结构化输出

| 门 | 结果 |
|---|---:|
| **BFCL** | **29/32** |
| **ToolACE** | **12/12** |
| **NexusRaven** | **13/16** |
| **严格 JSON Schema** | **7/8** |
| 结构化 B1 / B4 | **12/12 · 12/12** |
| 长前缀状态隔离 | **5/5** |

同一开发线的运行时示例:普通 q7 → 自适应 q8:168.52 → 170.98 tok/s;
重复上下文 q16:316.27 tok/s(3.162 ms TPOT)——q16 是**特殊的重复上下文
查找命中契约**,不代表所有工具调用请求的预期吞吐。证据:[PR #366](https://github.com/1CatAI/1Cat-vLLM/pull/366)。

## 分布 / PPL 门

八个固定 WikiText 2048-token 段、16,376 个计分 prompt token:
目标单独 PPL 5.4993116;DFlash2 PPL 5.4993622;绝对差 +0.0000506
(相对 +0.00092%);单段最大差 0.0062143。该门用于发现"基准答案看起来
还行,但投机验证已系统性偏移目标分布"的情况。

## ⚡ DFlash2 为什么能到 200+ tok/s

仓库内多条**真实全模型 DFlash2** 记录:生产 web prompt 206.06 流式解码;
高接受 MBPP 请求 251.60(接受长度 4.686);自适应查找 q16 重复上下文
316.27(opt-in 特殊契约);README 头条 ≈260 来自真机演示。
**206、251、260、316 不是同一个基准**——DFlash2 吞吐强依赖接受长度、
prompt 重复度、q8/q16 验证宽度、上下文长度与任务类型。

## 📏 长上下文不止是"能装下 256K"

Qwen3.8-27B-NVFP4 真实 TP4 全模型长上下文解码([PR #285](https://github.com/1CatAI/1Cat-vLLM/pull/285)):
128K:40.561 → **61.834 tok/s**;256K:27.456 → **50.376 tok/s**(实测
端点值;PR 内 52.216 为分解推算,本 README 采用实测值)。DeepSeek-V4 同样
以后续全模型结果为准:TP8 无投机 ≈65.1 tok/s;PP2×TP4 质量受控端点
73.613–73.646 tok/s。

## 🔬 已合并 PR 精选基准

| 领域 | PR / 契约 | 对照 | 1Cat 结果 | 增益 |
|---|---|---:|---:|---:|
| D256 长预填充注意力 | [#198](https://github.com/1CatAI/1Cat-vLLM/pull/198) · Q4096/KV64K | 87.60 ms | **50.45 ms** | **1.74×** |
| D256 长预填充注意力 | #198 · Q4096/KV8K | 11.13 ms | **5.05 ms** | **2.20×** |
| 128 位 E5M2 XQA 加载 | [#268](https://github.com/1CatAI/1Cat-vLLM/pull/268) · B16/17.8K | 0.743 ms | **0.602 ms** | **1.235×** |
| 批量长解码 | #268 · B16/16K 全模型 | 529.071 tok/s | **570.982 tok/s** | **+7.92%** |
| 长上下文解码路由 | [#206](https://github.com/1CatAI/1Cat-vLLM/pull/206) · 128K TP4 | 40.82 tok/s | **48.54 tok/s** | **+18.92%** |
| E4M3 XQA 长解码 | [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285) · 精确 128K / 256K | 40.561 / 27.456 | **61.834 / 50.376** | **+52.5% / +83.5%** |
| 分组 QSA Page4 | [#387](https://github.com/1CatAI/1Cat-vLLM/pull/387) | 55.15 ms | **9.63 ms** | **5.518×** |
| QSA 全模型预填充 | #387 · 64K | 4,446.64 | **5,777.43 tok/s** | **+29.93%** |
| 精确目标解码 | [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415) · 8K/512 | 65.864 | **80.732 tok/s** | **+22.57%** |
| DFlash2 NVFP4 预填充 | [#417](https://github.com/1CatAI/1Cat-vLLM/pull/417) · 32K/64K | 收线前保留 | **4069/3567 tok/s** | **+30.1% / +37.7%** |
| DeepSeek-V4 稀疏 MLA | [#163](https://github.com/1CatAI/1Cat-vLLM/pull/163) | 46.92 ms/token | **4.392 ms/token** | **-90.64%** |
| DeepSeek-V4 TP8 无投机 | [#181](https://github.com/1CatAI/1Cat-vLLM/pull/181) · 1024/256 | 19.342 ms TPOT | **15.357 ms ≈65.1 tok/s** | **TPOT -20.6%** |

## 🧠 128 位加载:不是化妆级的向量化改动

[PR #268](https://github.com/1CatAI/1Cat-vLLM/pull/268) 在真实分页 KV 分区
内:复用 Page ID;合并两个 half8 转换组;发射一次对齐的 128 位缓存加载;
保持 softmax、PV、分区边界与归约顺序不变。NCU 证据:L1 全局加载请求
656,443 → 383,814(**-41.53%**);执行的 warp 指令 -14.71%;长记分牌停顿
39.14% → 30.10%;合格 warp/调度器 0.55 → 0.65;B16/17.8K 内核时长
-22.95%。DRAM 字节几乎不变。收益来自**更少的碎片化加载、更低的地址/依赖
压力、更连续的操作数供给**,而不是凭空缩小模型。

## ✅ 正确性 / 质量门

1Cat-vLLM 不把好看的 TPS 当充分证据。代表性门:#198 全模型 64K A/B/A
64-token ID、文本与 SHA256 匹配;#268 均匀/参差 B4–B16、page256/800、
12K–32K 算子 A/B **位元精确**;#285 128K/256K E4M3 XQA 端点均发出完整
64 token 并保持对照流;#346 结构化 API 24/24、长交替前缀状态 5/5、多种子
MBPP/HumanEval 质量门、目标/DFlash2 WikiText PPL 5.4993116/5.4993622;
#387 分组 QSA 重放确定性、算术/中文/性能用例 token 哈希与保留基线一致;
#415 GSM8K 15/16 严格、自然停止 16/16、零触顶零结构无效;#427 1.5.0 RC
隔离安装通过 `/v1/models`、`/metrics`、普通聊天、流式/非流式工具调用、
JSON Schema 与重复前缀检查;10,017-token 前缀从 **2.642s 冷 → 0.164s 缓存**。

## 🔥 FlashAttention-V100:为 Volta 重建数据流

FlashAttention 本质是 **IO 与调度问题**:减少 HBM 往返、让 Q/K/V 与中间
状态尽量驻留片上、提高复用、减少物化、减少屏障、持续喂饱 Tensor Core。
现代 FlashAttention 实现围绕 Ampere/Hopper 及更新架构设计,而 V100 是
SM70:没有 Ampere `cp.async`、没有新式 `ldmatrix` 数据通路、没有 Hopper
TMA、没有原生 FP8/FP4 Tensor Core。直接兼容移植能跑,但常常让 GPU 吃
不饱。这就是 1Cat-vLLM 围绕 Volta **实际具备的能力**重建执行路径的原因。

## ⚙️ SM70 上的软件重构异步 / 矩阵喂料

我们**不**声称 V100 执行了 `cp.async` 或 `ldmatrix`;我们用
`LDG/STS/LDS`、寄存器预取、双缓冲、Shared Memory swizzle、显式 HMMA
fragment 映射、跨 tile/跨 stage 软件流水线,**重构这些机制背后的设计
目标**:内存搬运与计算重叠 → 提高片上复用 → 缩短依赖链 → 减少屏障与
重放 → 让 HMMA 持续吃满。代表性技术:寄存器预取与双缓冲;下一 K tile
加载与当前 QK 计算重叠;HMMA 执行期间预置 PV 操作数;相位交错布局;
128 位向量化访问;QK/TN 与 PV/TT fragment 显式归属;跨 tile/stage 软件
调度。

## Layer 1 — 把 KV 缓存搬对、搬宽、只搬一次

朴素 SM70 路径反复"读 Page ID → 算地址 → 读窄 FP8 片段 → 转换",浪费
周期。[PR #268](https://github.com/1CatAI/1Cat-vLLM/pull/268) 复用页元数据
并成对对齐加载。全模型批量代表成绩:B16/16K 529.071 → **570.982 tok/s
(+7.92%)**;对应算子在代表性长上下文 XQA 形状上收益约 **21–26.5%**。

## Layer 2 — 把 D=256 注意力重写为 Volta 原生流水线

关键组件:**D256 Split-D**(把 D=256 拆成四个 D64 切片,配对 warp 共享
QK 概率工作的同时提高 PV 并行);**N32 在线 softmax**(保持因果在线
softmax 与 FP32 累加契约,不物化全分数矩阵);**K-stage 乒乓**(K/D64
面板交替进出共享内存 stage,降低屏障压力);**Split-KV3**(长前缀 KV 三
分区再合并 FP32 部分状态);**GQA 多头打包**(6 个 GQA 查询头打包成更宽
的 Tensor Core 工作);**宽 QK/PV**(碎片小算子 → 更大更规整的 GEMM 型
工作);**前缀/因果尾分离**(全可见长前缀与精确因果尾分开调度,再合并
在线 softmax 状态)。该优化族经 [#198](https://github.com/1CatAI/1Cat-vLLM/pull/198)、
D256/Split-KV3、[v1.3.0](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.3.0)
与 [#286](https://github.com/1CatAI/1Cat-vLLM/pull/286) 演化,结果:
17.92 → 46.63–47.1 → **≈60.8 TFLOP/s**。同一代 GPU、同一代 Tensor Core,
是软件不再浪费它们。≈79 TFLOP/s 保留为实验研究上限,不是默认生产质量
声明。

## Layer 3 — 稀疏注意力同样必须是 V100 原生

Qwen3.8 Flash Next QSA 不止"少选些 token":运行时还要处理稀疏块选择、
物理页映射、Page4 K/V 复用、精确逐行掩码与最终 QK/PV 计算。
[PR #387](https://github.com/1CatAI/1Cat-vLLM/pull/387) 把 8 个相邻查询行
分组,重叠的 Page4 K/V 块只加载一次,同时保持逐行精确 4-bit 掩码,再用
Volta WMMA 直接算 QK 与 PV。代表成绩:旧 QSA 路径 55.151 ms/层/卡 →
分组 Page4 **9.632 ms(+0.362 ms 规划器),注意力 5.518×**;全模型纯
预填充 32K +32.36%、64K +29.93%、131K +32.69%。

## 🧩 性能剖析驱动的优化

一个内核快了并不停手:QSA 提速后,剖析显示下一个热点移到了 NVFP4 MoE
预填充,[PR #390](https://github.com/1CatAI/1Cat-vLLM/pull/390) 用索引化
W13 执行移除 `[tokens × topK, hidden]` 输入展开瓶颈(8K 算子链
6.027 → 4.235 ms,1.423×;全模型纯预填充 32K 5998.65 → 6507.10、64K
5777.43 → 6241.48、131K 5450.92 → 5871.47 tok/s);[PR #393](https://github.com/1CatAI/1Cat-vLLM/pull/393)
再融合精确 FP16 SwiGLU、把 N320 W13 尾拆成 N256+N64、砍掉尾部 tile 浪费。
项目哲学:**剖析真实模型,搬走瓶颈,再次剖析。**

## 🎯 投机解码之前,先把目标模型做快

[PR #415](https://github.com/1CatAI/1Cat-vLLM/pull/415):Qwen3.8-Flash-Next-NVFP4
@ 4× V100,8K 入 / 512 出,无 MTP,全 CUDA Graph:对照 65.864 tok/s
(15.183 ms TPOT)→ **80.732 tok/s(12.387 ms TPOT)**——这是**纯目标
吞吐**。同时通过 GSM8K 15/16 严格、自然停止 16/16、加权自然输出解码
80.935 tok/s。

## ⚡ SM70 上的 DFlash2

传统自回归解码每个输出 token 都要一次目标模型前向;DFlash2 改变执行
模型:**块扩散 drafter 提议多个未来 token,目标模型整块验证**,有效服务
循环变成"起草多个候选 → 目标验证一块 → 接受多个 token → 单轮推进多
token"。面向发布的 SM70 栈同时优化:draft 注意力、选择器、分组验证器、
GDN 元数据、稀疏拒绝、NVFP4/QPN 路径、采样、CUDA Graph、前缀状态、
Mamba 对齐、工具/结构化输出状态。draft 注意力本身使用 `FLASH_ATTN_V100`,
而不是退化到无关通用路径。

## DFlash2 长上下文衰减

[PR #328](https://github.com/1CatAI/1Cat-vLLM/pull/328) 让非锚定分页预填充
循环从 drafter 实际使用的第一个滑窗 tile 开始:256K 下 draft 注意力
0.422912 → 0.246784 ms/层,五层投影 2.115 → 1.234 ms;候选中位数 32K
0.252928 / 128K 0.243712 / 256K 0.246784 ms——32K 之后该组件的上下文斜率
几乎被消除。

## 🔢 量化 / 算子栈

V100 早于当前 LLM checkpoint 使用的许多格式,因此 1Cat-vLLM 把量化支持
当作**算子设计问题**而非只是加载器问题。当前 SM70 工作涵盖:AWQ/W4A16、
TurboMind SM70 内核、compressed-tensors、FP8 E4M3/E5M2 KV、ModelOpt
NVFP4、MXFP4、Quark W4A16 INT4/UINT4、QPN8/QPN4/QPN2、分组 MoE、精确形状
解码 GEMV、自定义 SM70 采样路径。目标不是"dtype 能解析",而是**量化格式
成为 Volta 上可用的高性能服务路径**。

### Qwen3.6-35B-A3B NVFP4

[PR #270](https://github.com/1CatAI/1Cat-vLLM/pull/270) 为混合 ModelOpt
NVFP4 checkpoint 添加精确 SM70 路线:FP8 稠密投影、W4A16_NVFP4 路由/共享
专家、分组 TurboMind MoE、重复专家槽保留、混合精度 GDN 路由、MTP 冷启动
预热。对标无 MTP:AWQ 预填充 0.3813s/解码 113.71,NVFP4 预填充
0.4216s/解码 116.99;MTP4 **174.76 tok/s(1.49×)**;质量:GSM8K
122/128(95.31%)、0 无效、0 重复。

### DeepSeek-V4 on V100

SM70 栈涵盖:稀疏 MLA、FP8 稠密投影、MXFP4 专家、分组 MoE、Indexer、
KPool、Q 归一化/RoPE/KV 插入、自定义 TP4 all-reduce、PP2×TP4 执行、精确
GEMV 热路径。代表成绩:TP8 无投机 ≈65.1 tok/s;PP2×TP4 严格质控 73.539;
合并端点 73.613–73.646 tok/s。严格质控:GSM8K 64/64、HumanEval 29/32、
LongBench 44.740。

### GLM-5.3 on V100

当前 GLM-5.3 SM70 路线:ModelOpt NVFP4 MoE、FP16 非专家权重、FP8 E4M3
KV、TP4/PP2、稀疏 MLA、精确 KDA GEMV、融合 KDA f/g、mHC、自定义
all-reduce、全解码 CUDA Graph。保留稳定性结果:解码 53.0131/53.0185/
53.0175 tok/s(均值 **53.0164**,平均 TPOT 18.862 ms);1K 预填充
**266.040 tok/s**。质量审计记录推理模式注意事项:Max 推理可能在简洁代码
任务上耗尽输出预算;定向低推理重跑完成并通过 AST 与外部执行检查。

## 🧠 "让 Volta 再快起来"的含义

我们**不**声称 V100 有 A100/H100/Blackwell 的理论峰值。要点是:大量现代
推理软件已不再认真优化 SM70,造成"硬件代差 + 软件荒废"两个鸿沟,
1Cat-vLLM 只做第二个。当代表性注意力有效算力从 17.92 → 46–47 → ≈60.8
TFLOP/s,而真实 27B 256K 解码仍达 50.376 tok/s,结论不是"V100 变成了
A100",而是:**软件不再浪费 V100。**

## 📦 安装

推荐环境:Python 3.12 / CUDA 12.8 / PyTorch 2.10 / SM70。稳定用户从
GitHub Releases 安装;要最新 DFlash2 1.5.0 服务策略,确保 wheel/源码包含
PR #426、#427 的最新 SM70 DFlash2 运行时改动。当前仓库状态:1.5.0 已完成
RC 构建与隔离 API/运行时冒烟;标签存在之前,本 README 不把 RC 称作正式
Release。安装示例:`pip install ./1cat_vllm-*.whl`;验证脚本见英文原版
(导入 torch/vllm/flash_attn_v100 并打印版本与 grouped verify max Q)。

## ▶ Qwen3.8-27B-NVFP4 + DFlash2 服务示例

TP4 + E5M2 示例命令(完整命令见英文原版):`vllm serve … --tensor-parallel-size 4
--attention-backend FLASH_ATTN_V100 --kv-cache-dtype fp8_e5m2 --max-model-len 262144
--speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","revision":"dedf8df6…","kv_cache_dtype":"auto"}'`。
经验证的 Qwen3.8 DFlash2 契约下,运行时策略自动解析 checkpoint 原生 draft
几何与 SM70 draft 注意力后端:官方 draft block size 8、draft 宽度 7、选择器
Top-K 16、示例目标 KV FP8 E5M2(可选)、draft 注意力后端 FLASH_ATTN_V100、
验证快速路径自动。SM70 DFlash2 验证器默认的开启与目标量化、KV 精度、TP
度、服务容量无关;各算子自行能力检查并独立回退(如单趟分组注意力算子是
E5M2 专用、紧凑 LM-head 重排是 TP4 专用)。`--max-num-seqs`、
`--max-num-batched-tokens`、`--performance-mode` 控制并发与预填充策略。

### DFlash2 发布路径测量

| 契约 | 结果 |
|---|---:|
| 完整 DFlash2 轮 | **≈17.38 ms** |
| 32K 冷预填充 | **≈4,039–4,069 tok/s** |
| 64K 纯预填充 | **≈3,567 tok/s** |
| 对比收线前 | **+30.1% / +37.7%** |
| 历史 web prompt 流式解码 | **206.06 tok/s** |
| 高接受 MBPP 请求 | **251.60 tok/s** |
| 自适应查找 q16 | **316.27 tok/s** |
| 结构化 API / 长交替前缀 | **24/24 · 5/5** |
| WikiText PPL(目标/DFlash2) | **5.4993116 / 5.4993622** |

## 🔨 从源码构建

`git clone` 后为 SM70 构建 FlashAttention-V100:
`export TORCH_CUDA_ARCH_LIST=7.0; export CMAKE_CUDA_ARCHITECTURES=70`,
然后按仓库当前构建说明安装。项目含自定义 CUDA 扩展,注意编译器/工具链
与环境 PyTorch CUDA ABI 匹配。

## 📐 基准测试政策

1Cat-vLLM 刻意区分:内核时延 / 算子吞吐 / 注意力有效 TFLOP/s / 预填充
tok/s / 目标纯解码 / 投机纯解码 / 流式解码 / 端点吞吐 / 任务质量分 /
PPL 分布检查。一条基准声明应保留:精确模型与 checkpoint、GPU 型号数量、
TP/PP 拓扑、上下文与输出长度、批量、KV 精度、量化路线、CUDA Graph 模式、
前缀缓存状态、采样契约、投机方法、接受长度、质量结果、实测还是推算。
本 README 在底层 PR 保留足够信息处均遵循该政策。

## 🛡️ 晋升政策

快速路径不会仅因微基准更快就晋升。视算术改动,晋升可能要求:算子位元
相等;有界数值误差;CUDA Graph 重放稳定;同契约端点提速;数据集质量;
自然停止/输出健康检查;PPL/logprob 分布检查;明确回滚;结构化/运行时
准入而非硬编码模型身份。部分研究 PR 即使速度亮眼,质量门不关闭就保持
Draft。≈79 TFLOP/s 注意力实验即为例子:性能线很强,但 256K 模型质量门
未过,因此不作为默认稳定路径宣传。

## 🧱 运行时,而不只是内核

1Cat-vLLM 的工作覆盖整条服务路径:FlashAttention-V100、分页 KV 工具、
FP8 KV 桥、QSA 稀疏注意力、FlashQLA/GDN、TurboMind SM70 量化内核、分组
MoE、MTP、DFlash2、CUDA Graph、前缀缓存、混合 Mamba 状态、自定义
all-reduce、采样、工具调用、推理解析器、结构化输出、wheel/RPATH/ABI
打包。**快的内核只有被完整模型与服务 API 正确使用才有价值。**

## 🧭 项目方向

1Cat-vLLM 聚焦一个简单问题:**如果软件栈被重新设计而不是被放弃,Volta
里还藏着多少现代 LLM 推理性能?** 当前方向:更深层长上下文注意力、更低
DFlash2 验证器成本、更高接受的投机执行、稀疏注意力、SM70 上的现代量化
格式、融合解码热路径、MoE 路由与分组 GEMM、多模型 SM70 支持、稳定
wheel/发布打包。

## 💬 微信社区

扫码加入 **1Cat-vLLM 开源社区 8 群**(二维码有效期至 2026 年 9 月 20 日;
过期可加微信号 **`YM_isi`** 索取最新邀请)。二维码见英文原版 README。

## ❤️ 致谢

1Cat-vLLM 建立在更广泛的开源推理生态之上,包括 vLLM、NVIDIA CUDA、
FlashAttention、CUTLASS/TurboMind 相关内核、模型作者、量化项目与各位贡献
者:[vLLM](https://github.com/vllm-project/vllm)、
[lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)、
[flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)、
[marlin_v100](https://github.com/zhinianqin/marlin_v100)、
[v100-skinny](https://github.com/dnv2003/v100-skinny)(QPN 四对-N `m8n8k4`
解码布局,MIT)。

特别感谢 [@yangzhuxinyzx](https://github.com/yangzhuxinyzx) 与
[@1CatTCat](https://github.com/1CatTCat) 对 1Cat-vLLM 持续演进与性能突破
的杰出贡献。外部实现或算法被改编时,出处与许可信息应保留在对应源码与 PR
历史中。

## 许可

遵循仓库许可证与捆绑/改编第三方组件的各自许可证。

---

# AC922 ppc64le 构建快照(中文版)

本目录记录 1Cat-vLLM 及其 Python 依赖在 IBM Power System AC922(Tesla V100
SM70)上的**实验性** CUDA 12.4 构建。基于上游 1Cat-vLLM 提交
`db292f9a4…`。构建主机:Debian 12.11、Python 3.11、POWER9、glibc 2.36、
CUDA 12.4、NVIDIA 驱动 550.54.15,并使用自定义内核;wheel **未**验证跨其他
Power 系统或发行版的可移植性。

### 可用产物

[release 清单](release/WHEELS.sha256)列明八个本地编译的 ppc64le wheel。
发布二进制不进 Git 历史,应附于 GitHub Release;下载全部八个资产后用
`sha256sum -c WHEELS.sha256` 校验。

| 构建 | 已验证 | 重要配对 |
| --- | --- | --- |
| PyTorch 2.10.0 CUDA 12.4(源自 PyTorch 提交 `449b1768…`) | wheel 完整性、Python 导入、CPU 张量求和 | 独立锁定 Torch 候选;V100 运行时验证仍待完成 |
| 1Cat-vLLM `0.1.dev1+gdb292f9a4.cu124` | wheel 完整性与原生扩展导入;隔离 SM70 GDN 内核已在 V100 运行 | 针对 [`power-torch-cuda124` 2.12.0a1](https://github.com/RomanMoz/power-torch-cuda124/releases/tag/v2.12.0a1) 构建,**非**上述 PyTorch 2.10 wheel |
| safetensors、tokenizers、tiktoken、msgspec、uvloop、psutil | wheel 哈希与归档完整性 | 本地 Python 3.11/ppc64le 构建;ABI tag 因包而异 |

1Cat wheel 元数据请求 `torch==2.10.0`,而实验 wheel 针对独立的
`power-torch-cuda124` 2.12.0a1 编译。**不要把两个发布行混入同一环境**,
也不要从 wheel 构建成功推断 PyTorch C++ ABI 兼容。实验性 1Cat 安装在隔离
环境使用 `--no-deps`;截至本快照,完整模型加载、服务、MTP/DFlash2 与
4/8 请求基准仍未验收。

### 复现材料

- [`scripts/`](scripts/):主机专属的构建、冒烟与基准命令;默认路径反映
  AC922 原始布局;标注处可用 `AC922_WORK_ROOT` / `AC922_BUILD_ROOT`。
- [`dependencies/WHEEL_CACHE.sha256`](dependencies/WHEEL_CACHE.sha256):
  记录 98 个下载/编译的工具链与缓存文件——**产物清单**,不是解析器锁,
  也不代表 Qwen 服务路径需要其中每个包。
- [`dependencies/README.md`](dependencies/README.md):区分本地构建 wheel、
  厂商 wheel、源码归档、系统包与排除资产。
- [`release/RELEASE_NOTES.md`](release/RELEASE_NOTES.md):二进制出处、兼容
  限制与已做检查。

本分支及其拟发布资产**不包含**模型权重、Hugging Face 令牌、订阅 URL、SSH
材料、BMC 配置、主机日志、NVIDIA 专有归档或第三方 PyTorch 2.12 二进制;
请按各自条款向所有者获取。上游源码与 AC922 构建材料保留仓库的 Apache-2.0
许可;每个重建依赖保留其上游许可,见
[`release/THIRD_PARTY_NOTICES.md`](release/THIRD_PARTY_NOTICES.md)。

### TP3 + DFlash2 服务战役(2026-09-29)

[`tp3-dflash2-trial/`](tp3-dflash2-trial/README.md) 记录了本源码树上的
完整服务战役:混合 Qwen3.8-27B AWQ 模型的**无损 TP3 头复制**(对 TP2 贪心
一致 20/20)、**DFlash2 投机解码单流 110 tok/s(接受率 82.2%)**、
**四并发 30 分钟 ~180 tok/s 聚合零错误**。补丁、测试工具、逐条记录与完整
报告一应俱全。

### TP3 + 原生 MTP4 试车与生产切换(2026-09-29/30)

[`tp3-mtp4-trial/`](tp3-mtp4-trial/README.md) 是续篇:同一个 fork 改用模型
**原生单层 MTP 预测头**替换 DFlash2 drafter。三道配置级障碍(draft 重读
原始 checkpoint、5120 宽 `fc` 无法三等分、`--hf-overrides` 传不进 draft)
在四个文件内修复,checkpoint 零改动。隔离试车实测 **242,119 tokens KV 池、
64k 单流 69.69 tok/s、接受率 83.9%**;三档利用率扫描选定 0.91;遭遇
"生产 CUDA Graph 1.13 GiB ≠ 试车 0.37"后以 **0.88 上生产,KV 池
227,555 tokens**(DFlash2 生产的 2.3 倍),单流 64k **69.74 tok/s**、
4×50k 全场解码 **116.30 tok/s**。DFlash 保留一条命令即可回退。补丁、
工具、逐条记录与完整过程记录一应俱全。

同期与更早的运维踩坑记录:

- [`PITFALLS-FAN-THERMAL.md`](PITFALLS-FAN-THERMAL.md) —— 家用数据中心风扇
  策略:硬 RPM 边界、OCC 激活窗口、风扇策略变更导致编译中途断电的事故。
- [`PITFALLS-MEMORY-LOWMEM.md`](PITFALLS-MEMORY-LOWMEM.md) —— "幽灵 DIMM"
  HBM aperture、swap 耗尽、RAM 镜像构建卫生、16GB 卡的 CUDA Graph 头寸。
