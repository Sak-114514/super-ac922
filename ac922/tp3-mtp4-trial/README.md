# Qwen3.8-27B AWQ TP3 + 原生 MTP4 @ IBM AC922(POWER9,3× V100-SXM2-16GB)

> English abstract: the same lossless TP3 head-replication fork, this time with
> the **native single-layer MTP head** instead of the DFlash2 drafter. The
> native-MTP path passed a bounded trial and **replaced DFlash2 in production
> the same night**: KV pool 227,555 tokens (2.3× the DFlash2 production pool),
> 64k single-stream decode 69.74 tok/s, 4×50k all-live decode 116.30 tok/s,
> both OpenAI and Anthropic endpoints HTTP 200, DFlash kept as a one-command
> rollback. All numbers measured 2026-09-29/30. Apache-2.0.

本文是 [`tp3-dflash2-trial/`](../tp3-dflash2-trial/README.md) 的续篇,记录
2026-09-29 晚另一条投机解码路线的完整过程:在同一个 TP3 无损头复制 fork 上,
**绕开 DFlash2 drafter,让 Qwen3.8 自带的原生单层 MTP 预测头跑在 TP3 上**,
从只读评估、源码适配、隔离试车、三档显存扫描,到最终**替换 DFlash2 成为生产
配置**。全程约 90 分钟,零失败、零重启、每一步可回退。所有数字均为实测,
原始逐条数据在 [`results/`](results/),代码差异在 [`patches/`](patches/)。

## 为什么换路线:DFlash2 的上下文代价

DFlash2 方案以 **110 tok/s 单流**的速度服役,但它的 drafter 是一个 3.58 GiB
的独立块扩散模型,BF16 draft KV 与 GDN 的混合分页还让每个 KV token 的物理
开销翻倍。挂上 DFlash2 后,TP3 的 KV 池从裸 TP3 的 398,857 tokens 掉到生产
配置的 **99,864 tokens**——比旧双卡服务(191,030)还少。

原生 MTP 的只读预估显示另一条路:MTP 权重文件仅 **0.79 GiB**(单层 BF16),
且与目标模型**共享 embedding / 输出头**(默认
`VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS=1`)。若 TP3 能跑原生 MTP,KV 池预估
可回到 25–35 万 tokens 量级。

## 第一步:只读核对发现三道障碍(全部只读,未碰生产)

1. **MTP 会重读 checkpoint**:draft config 不继承目标进程已做的 TP3 头
   复制,仍按原生 24 Q / 4 KV 构造,源 KV=4 直接触发 TP3 的 Qwen attention
   断言(`qwen3_next.py:489`)。
2. **`fc` 投影 5120 维不能按三卡切**:`ColumnParallelLinear` 会除以 TP 数,
   5120 % 3 ≠ 0,构造直接失败。
3. **CLI `--hf-overrides` 不传给 draft**:`speculative.py:883` 重建 draft
   config 时只带 `SpeculativeConfig.hf_config_override`,必须手动把
   `ac922_tp3_head_replication` 与嵌套文本覆盖(36 Q / 6 KV、GDN 18/54、
   MLP 17664)传播到 draft 架构构造之前。

词表 248320 mod 3 余 1,沿用目标模型现成的 `padding_size=192`(补齐到
248448 / 3 = 82816)即可;默认 PP1 下 MTP 共享目标 IO,无需为 MTP 单独分配
词表。

**解法与目标模型同构**:MTP 权重做同样的加载期头复制 + MLP 零填充
(17408→17664,256 个零神经元),`fc` 改为 `disable_tp=True` 每卡持全宽。
改动落在 4 个文件(`vllm/config/speculative.py`、
`vllm/model_executor/models/qwen3_5_mtp.py`、
`vllm/model_executor/models/qwen3_5_tp3_heads.py`、新增权重校验脚本
`tests/models/qwen3_5/check_mtp_tp3_weights.py`),checkpoint 文件依旧
零改动。

## 第二步:隔离试车(单流,0.90 配额)

隔离 RAM 运行包,绑定 `127.0.0.1:8003`,生产 8000 临时停用。启动即成功:
模型加载 **7.79 GiB/卡**(比 DFlash2 的 8.66 还小),KV 池 **242,119
tokens**。七张需扩展张量逐个通过形状/半输出校验后才启动。

| Prompt | TTFT | 预填充 | 解码 | 总耗时 |
|---|---:|---:|---:|---:|
| 1k(128 输出) | 0.653 s | 1531.94 tok/s | 61.52 tok/s | 2.717 s |
| 64k(128 输出) | 35.693 s | 1793.05 tok/s | **69.69 tok/s** | 37.516 s |

draft 接受率 **198/236 = 83.9%**,与 DFlash2 的 82.2% 相当;零 OOM、零
NCCL 超时。对照已测的 DFlash2 depth 4:1k 慢 21%,64k **快 11%**——
DFlash 的验证开销随上下文增长,MTP 不受影响。

## 第三步:并发波次与"流式口径"修正

第一轮四波(128/32 输出)全部完成、零排队,但请求非流式,只能记录完成
时间,算不出准确的并发解码速度。于是补写了
[`tools/bench_concurrent_token_ids.py`](tools/bench_concurrent_token_ids.py):
流式接收 token ID,**只统计所有客户端同时处于解码区间的"全场解码"合计
速度**,并分别给出每路 TTFT / 每路解码中位数 / 端到端输出吞吐——不把
预填充时间混进解码速度。

0.90 配额、加长输出(1k 每路 512、长输入每路 256)后的完整数字:

| 波次 | 全场解码合计 | 每路解码中位 | 端到端输出 | 运行/排队 | 最低空余 |
|---|---:|---:|---:|---|---:|
| 2×1k×512 | 153.38 tok/s | — | 129.75 tok/s | 2/0 | 985 MiB |
| **4×1k×512** | **367.47 tok/s** | 81.03 | 232.93 tok/s | 4/0 | 577 MiB |
| 2×64k×256 | 84.73 tok/s | 37.87 | 7.13 tok/s | 2/0 | 275 MiB |
| 4×60k×256 | (无重叠窗口) | — | 6.24 tok/s | 4/0 | 249 MiB |

两个重要发现:

- **4 路短上下文聚合 367 tok/s,约为 DFlash2 四路压测(~180)的 2 倍**。
  批处理下 MTP 低成本验证的优势完全释放;单路 1k 慢 21% 的印象只适用于
  单请求场景。
- **4×60k 的"4 路同时运行"是表象**:chunked prefill 下预填充与解码互相
  挤占,三路被压到每路 4.5–6.5 tok/s,最后一路等了 161 秒才出首 token
  (之后 77 tok/s),全程没有一个"四路都在解码"的区间。"同时运行数"
  不等于"同时解码"。

## 第四步:三档显存利用率扫描(0.90 / 0.92 / 0.91)

用户要求测 0.92 甚至"挑个极限的"。结果:

| 配额 | KV 池 | 2×1k | 4×1k | 2×64k | 4×60k | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 0.90 | 242,119 | 153.38 | 367.47 | 84.73 | 6.24(挤占) | 基线 |
| 0.92 | 256,682 | 179.95 | 352.48 | 未测 | **主动否掉** | 4×1k 后仅剩 237 MiB/卡,长波必贴死 |
| 0.91 | 249,400 | 181.50 | 370.35 | 84.67 | **96.22** | **胜出**:4×60k 通过,重叠解码 96.22 |

0.91 的 4×60k 全部完成(133.8 秒,重叠解码 96.22 tok/s,最低余量
183 MiB/卡,最高 61°C);它还比 0.90 多 7,281 tokens 池子,让四路长请求
的解码重叠更充分。0.92 短波数字好看但多划的 0.24 GiB 把波次余量吃穿,
长波被主动放弃——**"数字最好看的配置"和"能安全跑生产的配置"不是同一个**。

## 第五步:生产切换,以及"生产 ≠ 试车"的关键一课

切换流程:旧 0.91 unit 先备份(带 SHA256)→ 装独立 unit
`ac922-coding-api-tp3-mtp.service`(同 8000 端口、同 API key、同中间件)
→ DFlash unit 停用保留。

**生产首启就撞出一个陷阱**:隔离试车里 CUDA Graph 只占 0.37 GiB/卡,生产
进程(带完整 API 中间件/工具链)却捕获了 **1.13 GiB/卡**,空闲显存只剩
**37–38 MiB/卡**——健康检查虽然 200,但这个状态连一发 64k 请求都不安全。
处理:**没有把试车的 0.91 直接当生产通过**,降配额到 **0.88** 重启。

0.88 生产实测:

- KV 池 **227,555 tokens**(仍是 DFlash2 生产 99,864 的 **2.3 倍**),
  CUDA Graph 回落到 0.37 GiB/卡,空闲 **1.29 GiB/卡**
- 鉴权:无 key 401;带 key 的 OpenAI `/v1/chat/completions` 与 Anthropic
  `/v1/messages` 均 200
- 单流 64k:首请求(冷启动)仅 41.10 tok/s;**换 cache salt 复测
  69.74 tok/s**——与试车 69.69 一致,41.10 确认为新进程首请求的冷启动
  效应,不是生产缩水
- 生产 4×50k×256:全部成功,零排队,全场解码 **116.30 tok/s**(试车
  123.56,差值为鉴权/中间件开销),最低余量 **359 MiB/卡**,最高 60°C

## 最终状态与回退

| | DFlash2 d4(旧生产) | **MTP4 @0.88(现生产)** |
|---|---:|---:|
| KV 池 | 99,864 | **227,555** |
| 模型加载/卡 | 8.66 GiB | 7.79 GiB |
| 单流 64k 解码 | 62.59 tok/s | **69.74 tok/s** |
| 4×1k 聚合 | ~180(30 分钟均值) | **370.35(波次)** |
| 接受率 | 82.2% | 83.9% |
| 显存配额 | 0.94 | 0.88 |

回退一条命令(DFlash unit 仍在):

```bash
ssh ac922 'systemctl --user disable --now ac922-coding-api-tp3-mtp.service \
  && systemctl --user enable --now ac922-coding-api-tp3.service'
```

## 未测事项(诚实边界)

- 生产 4×60k 波次(试车通过,生产未复测)
- 30 分钟稳定性压测(MTP 线全程只有单波次,无持续负载)
- MTP 输出对 DFlash2 / TP2 的逐 token 质量对照
- MTP 深度固定为 4,无按并发/上下文的动态深度调度(vLLM 源码里
  `num_speculative_tokens` 可作上限的动态机制尚未验证到本机)

## 复现

```bash
# 在 1CatAI/1Cat-vLLM @ db292f9a4 之上依次应用(DFlash2 试验的补丁之后):
git am ac922/tp3-mtp4-trial/patches/0001-*.patch
git am ac922/tp3-mtp4-trial/patches/0002-*.patch
git am ac922/tp3-mtp4-trial/patches/0003-*.patch

# 权重校验(不动 checkpoint,只读形状):
python tests/models/qwen3_5/check_mtp_tp3_weights.py

# 隔离试车 / 生产启动脚本见 tools/,并发流式基准:
python tools/bench_concurrent_token_ids.py --clients 4 --prompt-length 50000 \
  --max-tokens 256 --base http://127.0.0.1:8000 --key-file /path/to/key
```

脚本内所有主机/端口默认 `127.0.0.1`,按自己环境通过环境变量覆盖。

## 原始证据索引

- 报告:[`results/MTP-TP3-1K-64K.md`](results/MTP-TP3-1K-64K.md)(英文,与正文同源)
- 单流试车:`results/mtp4-1k64k.jsonl` + 前后 `prom` 计数器(接受率 198/236)
- 三档扫描:`results/mtp090-*.json`、`mtp091-*.json`、`mtp092-*.json`(逐请求 TTFT/decode_tps)
- 生产验收:`results/mtp088-prod-64k*.jsonl`、`mtp088-prod-c4-50k-o256.json`
- 启动日志:`results/logs/serve-mtp-r{1,2,3-stream,4-util092,5-util091}.log`、`coding-api-tp3-mtp.log`
