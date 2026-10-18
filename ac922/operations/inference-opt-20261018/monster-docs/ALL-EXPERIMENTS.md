# Monster 引擎 · 实验全记录与设置（ALL-EXPERIMENTS）

> **版本 v1.1 · 2026-10-14 16:45 · 状态：当前**。v1.0（2026-10-12 汇编）覆盖 2026-10-11~12 graph-max 全部实验（正确性线、速度线、混杂排除线、协议破案线）+ Codex 老树基准。v1.1 变更：追加 §F 数值乱码案判决臂（2026-10-14）。
> 引擎技术事实见同目录 ENGINE-MASTER.md（v1.1 §0.5 = 本案终审判决），本文只管"做过什么、怎么配、结果、结论"。
> 目录：`/home/debian/ac922-monster/experiments/graph-max-20261011-1757/e0-*`（每臂含 model/{runtime, feature-env.sh, server.log, perf-result.json, environment.txt}）

---

## F. 数值乱码案判决臂（2026-10-14 下午，详见 ENGINE-MASTER §0.5）

统一协议：`launch-p3-arm.sh` 配方（ISLAND=3,3、FAST 全开、P2P_AR=1、KV q8_0、FA on、SLOTS=1）→ 语义探针（"The capital of France is"，24 token，temp=0 seed=7）。

| 臂 | 二进制 | 关键差异 | 输出 | 判定 |
|---|---|---|---|---|
| b3 基座（14:51） | b3-1afa565（llama-v100-v3 脏树构建） | 无 | **" Paris. The capital of Germany is Berlin…"** ✅ | **tensor-split 无罪，b3 系干净** |
| 上午乱码群（11:21-12:00 ×6） | b7-1e542c5（我的 j12） | 各 env 臂 | 全 ❌（"военно"） | j12 乱码，与 env 无关 |
| ringoff（16:16） | b7（runtime-before） | RING_MIN=2⁶⁰ 强制 one-shot AR | ❌ | C1 分段核排除 |
| ncclbypass（16:24） | b7 | P2P_AR=0（NCCL 死路，回退仍是 p2p AR） | ❌ | 无效对照（NCCL 本机不可用） |
| metaoff（16:42） | b7（runtime-a3） | META_FIFO_PIPELINE/COMPACT_ARENA/PACKED_CACHE/KV_PAD=0 | ❌ | META 三 env 排除 |
| a3myrecipe（16:39） | b7（Codex runtime-a3 全新构建） | 无 | ❌ | **构建竞态排除**（双构建同乱码） |
| codexvals（16:33） | b7（runtime-a3） | 六旋钮对齐 Codex 值 | 崩溃（load_model） | 作废，未出文本 |

**结论**：乱码 = monster 树（1e542c5）缺失 v100v3 未提交修复（17 文件+5 头），与配方/构建/切分无关。war 及一切 b3 系历史数字维持有效。

---

## A. 通用设置（所有实验共享）

### A1. 硬件档位（跨档禁止比较）

| 档位 | GPU 上限 | CPU | 使用时期 |
|---|---|---|---|
| 静音 180W | 180W（0x4 SW-cap 节流样本出现） | 2.3GHz 全核 | v8~v22、晚间线 |
| 性能 300W | 300W（默认；零节流，峰值 248-275W/卡，满频 1530MHz） | 3.8GHz 不封顶 | v20、v26b2 之后全部 |

设置：`nvidia-smi -pl <W>`（root，重启失效）+ `echo 3800000 > /sys/.../scaling_max_freq`（跨重启保持）。

### A2. 标准服务启动（run-candidate.sh 固化部分）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=GPU-80452dac...,GPU-ac5b33c1...,GPU-e6529eb8...,GPU-fadb0459...,GPU-760ce501...,GPU-d3558623...  # UUID 固定岛序
LD_LIBRARY_PATH=$run/runtime:/usr/local/nccl/lib:/usr/local/cuda/lib64
LLAMA_ISLAND_SPLIT=3,3  GGML_CUDA_P2P=1  GGML_CUDA_ALLREDUCE=nccl  GGML_CUDA_NCCL_F32=1  GGML_META_P2P_COLD=1
GGML_SM70_BF16_GEMV=1  GGML_SM70_MOE_REWORK=1  GGML_CUDA_MOE_GROUPED=1  GGML_CUDA_VOLTA_GDN_X4=1
LLAMA_QSA_SPARSE_KERNEL=1  LLAMA_QSA_GATHER_MAX_TOKENS=256  LLAMA_QSA_INDEXER_IMPL=fused
LLAMA_MTP_FOLD=1  LLAMA_MTP_GPU_CHAIN=1  LLAMA_MTP_GPU_EMBED=1  LLAMA_MTP_FIFO=1
LLAMA_MTP_SEQ_LAYOUT=1  LLAMA_MTP_FOLD_MULTI=1  LLAMA_MTP_N_UBATCH=64  LLAMA_MTP_PAD=6  LLAMA_DRAFT_DETP=0
LLAMA_GPU_ARGMAX=1  GGML_CUDA_TOPK_STABLE=1
GGML_CUDA_GRAPH_OPT=1  GGML_FT_GRAPH_CACHE=1  GGML_FT_GRAPH_CACHE_MIN_NODES=32  GGML_CUDA_GRAPH_VARIANTS_CAP=4
GGML_META_ASYNC_COPY=1  GGML_META_FIFO_PIPELINE=1  GGML_META_COMPACT_ARENA=1  GGML_META_PACKED_CACHE=1
LLAMA_QWEN4EXP_HC_FUSION=1  LLAMA_HC_FUSED=1  LLAMA_PREFILL_KV_PAD=8192
LLAMA_PLE_SHARD=0  LLAMA_PLE_SHARD_ROWS=0  GGML_CPU_PLE_ROW_CACHE=1  LLAMA_PLE_PREFETCH=0
LLAMA_STEP_TIMING=1  LLAMA_STEP_TIMING_PERIOD=32  LLAMA_DRAFT_TIMING=1  LLAMA_FULL_STEP_TIMING=1
LLAMA_HYBRID_PREFIX_CACHE=0  LLAMA_CHECKPOINT_BUDGET_MIB=0  LLAMA_PREFILL_FAIR_SHARE=1
LLAMA_MTP_MULTI_CELL_SEL=0  LLAMA_QSA_POOLED_MULTI=0
# 之后 source $run/feature-env.sh（按臂覆盖）→ exec:
llama-server --model .../IQ3_XXS-MTP-Q8.gguf --host 127.0.0.1 --port 18098 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 --n-gpu-layers 999 --fit off --ctx-size 65536 \
  --parallel ${MONSTER_PARALLEL:-4} --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 --batch-size 2048 --ubatch-size ${MONSTER_UBATCH:-1024} \
  --threads 16 --threads-batch 32 --threads-http 8 --cache-ram ${MONSTER_CACHE_RAM:-0} \
  --ctx-checkpoints ${MONSTER_CHECKPOINTS:-0} --spec-type draft-mtp --spec-draft-n-max 4 \
  --metrics --slots --log-verbosity 4 [--kv-unified-per-slot ${MONSTER_SLOT_CTX}]
```

### A3. 测量协议

- **旧协议（已废弃，仅历史数字）**：4×8192 prompt × n=128 fixed seed，波间不弃首波 → 全部落在 decode 慢启动区，数字系统性偏低且对"启动类"开关敏感（假阳性）。
- **新协议（v26c 起标准）**：`MONSTER_PARALLEL=4` + `/slots` 确认 4 槽 → warm 4×128（弃）→ 2× 4×512 取稳态；或流式分段取 3/8 段之后的速率。工具 `wave-par.py`（标准）、`wave-nvml.py`（+NVML 200ms/500ms 采样）。
- prompt 库：`~/ac922-llama-next/manifests/kv262k-realtext/prompt_tokens.json`，lane i 取 `[i*8192,(i+1)*8192)`。
- 指标：per-flow TG（predicted_n/predicted_ms）、AGG=ΣTG、墙钟有效值（lanes×n/wall）、PP。

---

## B. 实验总索引

| ID | 档位 | 类型 | 一句话结果 | 状态 |
|---|---|---|---|---|
| teacher-v6~v8 / v11 / v12 | — | fixture | 层激活捕获/前缀种子/seqrm 器具 | 工具，关账 |
| E0 v13/v14 | 180W | 正确性 | GDN head_offset 索引 bug 定位（fixture 8/12 复现） | v13a 修复 |
| v15 | 180W | 正确性 | canonicalPLE 真权重 11/11+14/14 选路证明；live natural 首差 [30,null,null,109] FAIL | 关账 |
| v16/v17 | — | 数值 fixture | MoE batch-vs-T1 8.9e-8、whole==chunks8；PLE mode2 3.6e-7 | 关账 |
| v18-safe | 180W | 正确性 | 全 SM70 快路径 OFF 仍分岔且首差游走 → FP-noise 定性成立；GDN bug 是唯一真 bug | **关账** |
| v19-max | 180W | 速度 | AGG 62-69（n=128 污染）；结构图首次真激活（replay 150/设备）；ONECAT 4 路 assert 崩 | 完成 |
| v20-max300 | 300W | 速度 | AGG 73 均值（77-80 稳态段）；零节流确认软件瓶颈 | 完成 |
| v21-c1/c2 | 180W | 混杂 | MAX_CATCHUP 0/1/2 零差异（52.5-53.2/68.7-69.2） | 关账 |
| v22-timing | 180W | 诊断 | 分配/重建税仅 4.13%；**多槽固定步时 ~88ms 新主导发现** | 完成 |
| v23-nsys | — | 诊断 | nsys 与 CUDA graph 不兼容（CUPTI 无表 + QdstrmImporter 崩），弃 | 关账 |
| v24-par1 | 180W | 混杂 | 单槽同二进制单流 34-45（排除多槽税为主因的简单版） | 完成 |
| v25-nomulti | 180W | 混杂 | 关 FIFO_MULTI 40-46，排除 MULTI 回退 | 完成 |
| v26b2/b3/b4 | 300W | 破案 | 基座 env 单槽：48→68 稳态；timing 三开关无关；"4 路 321" 实为串行假象 | 完成 |
| oldtree-b1 | 300W | 破案 | Codex 冻结老二进制同机 65.5 TG/1326 PP → **无树回归、无硬件回归** | 完成 |
| v26c/v26d | 300W | 破案 | **真 4 槽并行 AGG 88.8-97.4（首个可信稳态）**；target 步 tokens=5..5 逐 lane 轮 | 完成 |
| v27-multi/m2 | 300W | 速度臂 | +GPU_CHAIN_MULTI：合批到 20-token/70ms 轮，AGG 94.4 中性 | 完成 |
| v28-fast | 300W | 速度臂 | +PLE_PREFETCH+结构图：AGG 89.2，爬坡慢 → 双负 | 完成 |
| v29-struct | 300W | 速度臂 | 隔离出**结构图 policy=3 稳态 -6%** | 完成 |
| v30-topk | 300W | 速度臂 | CANONICAL_TOPK+POOLED_MULTI —— **被中断未测** | 待跑 |
| v31-pathlog | 300W | 诊断 | 干净配置 PATH_LOG eager/replay 审计 —— **被中断未测**（用户 CUDA graph 假说判决实验） | 待跑 |

---

## C. 实验详情

### C0. Codex 老树基准（对照锚点）

- 二进制：`~/ac922-llama-next/experiment/codex-multistream-20261010-162940/candidate/runtime/bin/llama-server`（source 1afa5654 冻结）；DSO 在 `results/router-full-20261011-0031/runtime/`
- 启动：`experiment/codex-serving-20261011-0040/baseline/launch.sh`，端口 18096，parallel 6，**ubatch 256、MTP_N_UBATCH=32、QSA_GATHER_MAX_TOKENS=32、QSA_POOLED_MULTI=1**（与 monster 基座 1024/64/256/0 的四大差异）
- 历史锚点：150 TG / 1400 PP（用户口头确认"XBus unlocked"，原始 run 失踪）
- **今日复跑（300W 档）**：n=128 单流 43.8→65.5 TG / 1326-1351 PP；4 路（过 6 槽）12.9-22.6（AGG~67）
- 结论：同协议下与 monster 树打平 → 150 的差距属测量协议，不属树/硬件

### C1. 正确性线（E0 系列，全部 180W）

| 臂 | 设置差异（对 A2 基座） | 结果 |
|---|---|---|
| v13/v13a | GDN head_offset 修复（offset16/32 fixture） | fixture 12/12 绿 + 4 路 natural ×3 绿 → **唯一真 bug 关账** |
| v14 | v15 栈（见下）+ 全 trace | natural 首差 [40,122,114]；fixed/natural 与 v8 数组精确一致 → 形状/槽漂移而非新回归 |
| v15 | **v15 栈**（见 C2 首段）+ `GGML_SM70_BF16_CANONICAL_PLE=1 GGML_SM70_FUSED_PROJ=2` | canonicalPLE 真权重选路 11/11+14/14；live natural 首差 [30,null,null,109] FAIL |
| v16/v17 | 无服务（MoE/PLE 数值 fixture） | MoE b10.t20 vs T1：561/1280 位 exact、max_abs 8.9e-8、whole==chunks8；PLE mode2 shared-vs-T1 3.6e-7 |
| v18-safe | v15 栈 + 末尾 `FUSED_PROJ=0 BF16_CANONICAL_PLE=0 DECODE_LINEAR_CHUNKS=0 MOE_GROUPED=0 SM70_MOE_REWORK=0 VOLTA_GDN_X4=0`（全部 SM70 快路径关） | natural-vs-fixed 仍分岔，首差再游走 [102,123,107,None] → **分岔=形状/调度相关的 F32 归约噪声，容差门获批** |

### C2. 速度线（v19/v20，n=128 旧协议——数字已被协议污染，仅存档）

**v15 栈**（v14~v20 共用的 feature-env 主体，含大量插桩）：
```
LLAMA_MTP_KV_ONLY_PREFILL=0 GPU_CHAIN_MULTI=1 MAX_CATCHUP=0 SNAPSHOT=1 PLE_PREFETCH=1 FULL_STEP_TIMING=0
ARG_LOAD_MODE=mmap ARG_LAZY_MODE=on GRAPH_STRUCTURAL=0(前段) MONSTER_UBATCH=256 CACHE_RAM=512 CHECKPOINTS=2
HYBRID_PREFIX_CACHE=1 CHECKPOINT_BUDGET_MIB=512 CPU_PLE_ROW_CACHE_ENTRIES=524288 QSA_POOLED_MULTI=0 MULTI_CELLSEL=0 MULTI_SPARSE=0
MONSTER_PARALLEL=8→4 SLOT_CTX=16384 HYBRID_CHECKPOINTS_CACHE_ONLY=1
TARGET_FIFO_MULTI=1 META_COPY_TOPOLOGY=1 POLICY_FILE=<dir>/policy  QSA_CANONICAL_TOPK=1
SM70_DECODE_LINEAR_CHUNKS=8  GRAPH_STRUCTURAL=1+TRACE+TEST_POLICY+BOUNDARIES  FT_HOST_TIMING=1 FT_HOST_SNAPSHOT_PERIOD=256
unset META_PACKED_CACHE  MULTI_LOGIT_TRACE=<dir>/logits MIN/MAX_POS=8191/8192→8278/8304 ENABLE_FILE=trace-enable
SCHED_ALLOC_TIMING=1 UPDATE_LAYOUT_TRACE=1 GRAPH_PATH_LOG=1 OPTIMIZATION_PATH_LOG=1 META_INIT_TIMING=1 QSA_PATH_TRACE=1 QSA_INDEXER_TRACE=1
BACKEND_PATH=<dir>/runtime  QSA_INDEXER_DECODE_VECTOR=0
```

- **v19-max**（180W，policy=3）：v15 栈 + ONECAT=1（启动崩，移除后重启）。4×128 波 AGG 53.2/69.2/65.1；per-flow 13.8-22.5；PP~201/slot。结构图证活：built=attempts 100%，cross_descriptor_replays 150/设备，variant_matches 82。NVML：SM util 58-62%，180W 上限生效。
- **v20-max300**（300W+3.8GHz）：同 env。AGG 62.2/80.3/77.3；per-flow 16.4-26.0；PP 222/slot；峰值 233-240W 零节流。测毕恢复静音档。
- **今日复判**：这两个数字与 v26c（干净 env AGG 89-97）不可比——插桩栈压 4 路 + n=128 慢启动双重污染。v19 的"+12-18%"与结构图"+10-11%"结论**作废**。

### C3. 晚间诊断线（v21~v25，180W，v15 栈家族）

- **v21-c1/c2**：E6 MAX_CATCHUP 扫描 {0,1,2} → 全零差异（AGG 52.5-53.2 / 68.7-69.2）→ catchup 参数关账。
- **v22-timing**：+`STEP_TIMING=1 PERIOD=32 DRAFT_TIMING=1 FULL_STEP_TIMING=1` → 分配/重建税 4.13%（87 事件 5.6s；结构图吃掉 v8 时代 25% 的大头）→ 池手术降级。**新主导发现：多槽每步固定 ~88ms**（1-lane-on-4slot TG 34-35，acc 0.509/len 3.0 → step 88.5ms，图复用 513 次非分配问题；专用单槽 ~18ms）。
- **v23-nsys**：nsys 节点级跟踪与 CUDA graph 冲突（CUPTI 无内核表；--cuda-graph-trace=node 生成 157MB qdstrm 但 QdstrmImporter "Wrong event order" 崩）→ 弃用，改用 STEP_TIMING。
- **v24-par1**：MONSTER_PARALLEL=1 单槽同二进制单流 38-45 → 简单版多槽税确认（后被 v26 系列修正解释）。
- **v25-nomulti**：关 FIFO_MULTI → 40-46 → 排除 MULTI 回退假说。

### C4. 协议破案线（2026-10-12 下午，300W+3.8GHz 全程）

全部使用 `e0-20261012-v26-base`（基座：feature-env 仅 `MONSTER_PARALLEL` + timing 开关组合，其余全 A2 默认）。

| 臂 | feature-env | 波 | 结果 | 判读 |
|---|---|---|---|---|
| v26b2 | `MONSTER_PARALLEL=1` | 1 路 warm+2×128+4×128 | 48.7→67.9/68.0；"4 路" 68/75.7/67.7/109.5 | 首次 300W 干净基座；"AGG 321" 疑点 |
| v26b3 | +`STEP/DRAFT/FULL_STEP_TIMING=0` | 2×128 | 48.2→68.3 | timing 三开关排除 |
| v26b4 | timing off，1×512 流式+1×256+NVML 采样 | — | **分段 69/134/113/140/136/138/105**；时钟全 1530、248-275W | **慢启动发现：前 ~128 token 慢 3-5×，n=128 全污染** |
| oldtree-b1 | 老树 launch.sh（18096 口） | 1×128×2 + 4×128 | 43.8→65.5 / 1326 PP；4 路 12.9-22.6 | 无树回归；150 差距=协议 |
| v26c | `MONSTER_PARALLEL=4`（/slots=4 确认） | warm 4×128 + 2× 4×512 | **AGG 88.8 / 97.4**（per-flow 21-25.8） | **首个可信 4 路稳态；单槽"4 路"是串行假象** |
| v26d | +`STEP_TIMING=1 PERIOD=16` | 4×128 | per-flow 16.5-25.9；target 步 tokens=5..5、compute 21.9ms、rebuilt=0；draft 步 6 tok/0.36ms | **target 逐 lane 串行轮；draft 已合批** |
| 4×640 流式 | 同 v26c | 4 线程并发 | 每路 quarters 131/135/101、71/86/74、127/122/90、124/139/122 | q75 回落+lane1 弱=内容接受率差异 |
| 4×1024+NVML | 同 v26c | 1 波 | AGG 415.6（串行假象）；温度 42-51°C、零节流、偶发 1312-1350=岛等待 | 硬件无瓶颈实锤 |

### C5. 速度臂 A/B（300W，新协议 warm 4×128 弃 + 2× 4×512）

| 臂 | feature-env（对基座增量） | AGG m1/m2 | 结论 |
|---|---|---|---|
| v26c 基座 | `MONSTER_PARALLEL=4`（timing off） | 88.8 / 97.4 | 参照 |
| v27-multi(m2) | +`LLAMA_MTP_GPU_CHAIN_MULTI=1` | 94.4 / 94.5 | 日志"compact multi-lane chain lanes=2..4 hidden=10240"；轮从 4×5tok/21.9ms → 1×20tok/70.7ms；**聚合中性**（接受率从 ~2.3 掉到 ~1.6 抵消轮时收益） |
| v28-fast | +GPU_CHAIN_MULTI `PLE_PREFETCH=1` `GRAPH_STRUCTURAL=1+policy3` | 62.8 / 89.2（warm 仅 50.4） | 双负 |
| v29-struct | +GPU_CHAIN_MULTI `GRAPH_STRUCTURAL=1+policy3`（无 prefetch） | 61.4 / 88.4 | **隔离：结构图 policy=3 稳态 -6%、爬坡更慢**；PLE prefetch 贡献 ≈0 |
| v30-topk | +GPU_CHAIN_MULTI `QSA_CANONICAL_TOPK=1 QSA_POOLED_MULTI=1` | — | **被中断，未采数** |
| v31-pathlog | +GPU_CHAIN_MULTI `GGML_CUDA_GRAPH_PATH_LOG=1`（诊断臂） | — | **被中断，未采数**；v19 旧数据：岛A replay 76%/eager 20%（全 warmup 回退）、岛B 46%/34%，fallback 原因 warmup 939-1035 / changed 0-15 / exec_update_failed 3 |
| v32-d2（深度扫描） | `MONSTER_DRAFT_N_MAX=2`（run-candidate.sh 已改支持 `${MONSTER_DRAFT_N_MAX:-4}`） | 93.0 / **99.9** | **历史最佳：深度 2 比深度 4 +4~5%**（短链轮 12 tok/45ms 抵消接受率损失）；d3/d1/d2m 臂被中断未跑 |
| v36-tp6 | `unset LLAMA_ISLAND_SPLIT`（TP6 单岛） | 84.7 / 97.1 | **in-graph P2P AllReduce on 6 devices 实测激活**（跨岛 AR 免费）；decode 持平双岛 → "避跨岛是元凶"对 decode 证伪；PP 降至 149-151（shard 减半代价）→ 拓扑自由度确认但暂不采用 |
| v37-d2topo | d2 + `GGML_CUDA_META_COPY_TOPOLOGY=1`（peer 直通桥） | 93.3 / 100.1 | 桥激活（"peer source=0 staging leader=3"）；性能中性——**保留为防回归保险**（XBus 恢复后异步桥会静默失效） |
| v38-d2bal | d2 + TOPOLOGY + `LLAMA_ISLAND_LAYERS=28`（岛再平衡） | 93.2 / 100.0 | 中性——稳态 4 路下岛负载失衡不是因素 |

---

## D. 跨实验结论（引用最多）

1. **一切 n=128 与单槽"多路"数字作废**；可信参照 = v26c/v27/v29 三臂（AGG 88-97）。
2. **步时模型**：固定 ~6ms + ~3.25ms/token（5→21.9ms、20→70.7ms 拟合）；边际主导=MoE 专家流量（20 tok≈166/512 专家，1024 tok 饱和后 0.6ms/token）。
3. **单流 vs 多路的每路差距主嫌疑 = MTP 接受率**（单流 fold 快链 ~3，4 路逐 lane ~2.3，多路链 ~1.6），根源 `n_seq==1` 硬门（speculative.cpp:1516 一带）。
4. 插桩栈（PATH_TRACE/HOST_TIMING/SNAPSHOT/LOGIT_TRACE/…）本身压 4 路；性能臂禁用。
5. 结构图 policy=3 与 PLE_PREFETCH 在稳态协议下均为负；CANONICAL_TOPK/POOLED_MULTI/SPEC_PAD_MULTI/draft 深度未测。
6. 硬件无瓶颈：300W 档满频零节流；偶发降频=岛间等待空泡（B' 方向的侧证）。

## E. 待跑队列（优先级）

1. v31 重跑：PATH_LOG 干净配置 eager/replay 审计（用户假说判决）
2. **接受率攻击**（当前主攻）：统计单流 ~3 vs 多路 1.6-2.3 的差距来源；放宽 speculative.cpp `n_seq==1` 硬门把 fold/gpu_carry 快链多路化（中工程）
3. v30 重跑：CANONICAL_TOPK + QSA_POOLED_MULTI
4. LLAMA_SPEC_PAD_MULTI=12/24（固定形状实验，d2 下链 3 token → 4 路 12 token）
5. 深度曲线补完：d3 / d1 / d2+GPU_CHAIN_MULTI（v33-35 被中断）
6. Phase C 小件：v9 hashmap / QSA kvlen 钳制 / ONECAT 修复 / PLE 定向预取二期
7. B' 岛交错降级为备选（TP6 证明跨岛 AR 免费，但空转泡在稳态 decode 不主导）
