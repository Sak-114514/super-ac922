# AC922 Monster 推理引擎 · 总览文档（MASTER）

> **版本 v1.1 · 2026-10-14 16:45 · 状态：当前**。v1.0（2026-10-12 汇编）整合：DESIGN-BLUEPRINT-v2 / TWO-TO-FOUR-STREAM-PLAN-v4 / FULL-THROTTLE-PLAN / graph-max CURRENT-STATE 全部条目 / 2026-10-12 下午实测破案。v1.1 变更：**新增 §0.5 数值乱码案终审判决（翻案）**——v1.0 时代及其后会话中一度形成的"tensor-split 切分数值自诞生即坏"假设正式撤回。旧文档不再更新，冲突以本文为准。
> **配套**：全部实验的逐条设置与结果见同目录 ALL-EXPERIMENTS.md（实验台账）。

---

## 0.5 数值乱码案终审判决（2026-10-14 16:45 定案，翻案）

**现象**：10-14 上午起，凡从 monster 树（`/home/debian/ac922-monster/src`，HEAD=1e542c5）构建的二进制，语义探针（`/v1/completions`，"The capital of France is"，temp=0 seed=7）输出乱码（`"!! военно! военно! …"`），涵盖：我的 j12 构建（b7-1e542c5，11:00）、Codex 的 a1/a2/a3（12:06/12:22/13:46，含/不含我的 C1 补丁均乱）。

**根因（2026-10-14 17:15 修订：官方上游溯源确认）**：**上游 llama.cpp 已知 bug——MoE weighted-reduction 融合核破坏投机解码精确性**。链条：上游 PR #25952（2026-09-01 合并）引入融合核，累加写法可被 FMA 收缩 → 上游 Issue **#29168**（"breaks speculative-decoding exactness"，首坏构建 b10751）→ 官方修复 PR **#29381**"preserve MoE weighted reduction rounding"（显式 `__fmul_rn/__fadd_rn`，16445/16445 测试过）**于 2026-09-24 被关闭未合并，上游 master 至今带 bug**（直拉 master 文件第 22 行仍为旧写法）。我们 fork 基线 v0.5.0（7fe450e，2026-09-23）含 #25952 无修复 → 两树俱生带 bug；**v100v3 脏树早已本地回植 #29381 同款修复**（b3 干净的原因），monster 树从未回植（j12/a1/a2/a3 全乱码的原因）。症状与 #29168 完全吻合：MTP 投机解码下 target logits 漂移 → 乱码；sha256 自一致检查必过（确定性错误），语义检查才暴露。次要差异（qsa-attn 显式 cell 模式 / topk-moe / ggml-cuda 回退 AR rank 序——后两者 fork 自有演进，qsa-attn fork 专有文件）与多序列演进（speculative/server 等 15 文件）为 v100v3 并行工作树内容，**非乱码案的必要修复**。

**判决臂证据链**（全部同配方语义探针）：ringoff（分段核 RING_MIN 天文数字关闭）❌ / ncclbypass（NCCL 死路回退，无效对照）❌ / metaoff（META 三 env 关）❌ / a3myrecipe（Codex 全新构建）❌ / **b3 基座 ISLAND 3,3 tensor-split ✅（"Paris. The capital of Germany is Berlin…"）**。chat template 假设排除：两树 `server.cpp` md5 相同。

**已排除嫌疑**（各判决依据）：tensor-split 切分（b3 干净）；我的 C1 AR 分段核 + FIFO 门（RING_MIN 关闭仍乱码；FIFO 门 env 未设恒关闭；a3 含同补丁）；4 个瘦身提交 cut1-4（用户侧审计：PPC/VSX blob 逐字相同、28 行插入全读无一涉及数值）；PPC/VSX 路径（同审计）；构建竞态（双构建同乱）。

**后果重估**：
1. **war / v36 / 1350 PP / 150 TG / 1478 等全部 b3 系历史数字有效**（撤回"历史数字全在乱码模型上测得"的说法）
2. Codex a1-a3 的 1369 PP / 89.4 TG 是**乱码基底上的吞吐**（相对自洽、绝对语义不成立）；其"质量没有问题"依据 = output_sha256 自一致检查（9 处）+ logits 对拍，**均非语义级检查，自洽乱码可通过**
3. 10-10~12 "natural parity 失配"与本案**不同源**（那时只有 b3 系存在；维持 v1.0 §5 的 FP-noise 定性）

**修复路径（2026-10-14 17:15 修订，待执行）**：
1. **外科手术式（首选）**：把 #29381 同款修复（显式 RN 乘加，~11 行）打到 monster 树 `moe-weighted-reduction.cu` → 独立副本重建（不动活树、不占 GPU）→ 语义探针判决；预期单文件即治愈（MTP-off 臂可作交叉验证：monster 二进制关 MTP 应本就干净）
2. 全量对齐（可选后续）：v100v3 脏树其余内容（多序列演进 15 文件 + 5 头）与 Codex carry 做三方合并——那是功能线，不是乱码案的必要条件
3. **上游回报（建议）**：在 #29168 评论或新 issue 附 V100/SM70 + Qwen3.8-Flash-Next MTP 的复现（我们 6 臂判决链现成），推动 #29381 复活——上游 master 至今带 bug，所有 v0.5.0+ 基线的 MoE+MTP 用户都在踩
- 存档：`patches/UPSTREAM-PR29381-moe-weighted-reduction-rounding.diff`（官方原始 49 行）/ `patches/V100V3-FULL-DIRTY-STATE-20261014.patch`（2834 行全量）/ `patches/v100v3-new-headers/`（5 头文件）/ `patches/V100V3-DIRTY-FIXES-missing-from-monster.patch`（数值五件套提取版）

---

## 0. 一句话定位

在 AC922（2×POWER9 + 6×V100-SXM2-16GB，双 3-GPU NVLink 岛）上为 **Qwen3.8-Flash-Next**（48 层线性注意力+MoE 混合，MTP-4 投机）跑专用 llama.cpp fork（monster 树），拓扑 = 层流水双岛 + 岛内 TP3。当前目标：**4 路并发每路 ≥50 tok/s（聚合 ≥200）**，PSU 硬件不动。

## 1. 当前性能真相（2026-10-12 下午定案）

| 场景 | 数字 | 备注 |
|---|---|---|
| **单流稳态 decode**（流式分段后 3/8 段起） | **134-140 tok/s** | 基座 env，300W 软件档+CPU 3.8GHz |
| 单流 n=512 平均 TG | 107 | 前 ~128 token 慢启动摊薄 |
| 单流 n=128 平均 TG | ~68 | 全程在慢启动区——**历史上一切 n=128 数字都被污染** |
| 单流 PP | 1120-1190 | |
| **4 槽真并行**（MONSTER_PARALLEL=4） | 每路 21-26，**聚合 89-97** | n=512×2 波 |
| 4 请求过 1 槽（串行排队） | 每请求独享 71-120，墙钟有效 ~40-68 | 单槽串行吞吐反而 ≈ 真并行 |
| 历史锚点（Codex 树，300W 时代） | 150.7 TG / 1350.8 PP | 原始 run 失踪，"XBus unlocked" 出处不明 |
| 同机今日复跑 Codex 冻结老二进制 | 65.5 TG / 1326 PP（n=128） | **无树回归**；150 与 68 的差主要是测量协议（短波+慢启动），非硬件/软件退化 |

### 1.1 测量协议铁律（血泪教训）

1. **n=128 波全是慢启动**：decode 前 ~2 秒/~128 token 慢 3-5×（冷图/变体捕获/暖机）。一切稳态结论必须 n≥512，或流式分段取后半。
2. **MONSTER_PARALLEL=1 是单槽**：4 个 prompt 会串行过 1 个槽，per-request TG 好看但不是并发。测多路必须 =4 并确认 `/slots` 返回 4。
3. 弃首个波次；同配置 ≥2 波；交错 A/B。
4. 服务器 timing/trace 开关（PATH_TRACE/HOST_TIMING/SNAPSHOT/LOGIT_TRACE 等）本身压 4 路性能（v19 栈把聚合压到 62-80）。性能臂只留 LLAMA_STEP_TIMING（代价 ~0）。

### 1.2 步时成本模型（target 前向，小批量区间线性近似）

- **固定 ~6ms/趟** + **边际 ~3.25ms/token**（5-token 趟 21.9ms、20-token 趟 70.7ms 两点拟合）
- 边际主导项 = **MoE 专家权重流量**：token 数×top-10 激活的不同专家数次线性增长（20 token ≈ 166/512 个专家；1024-token ubatch 时饱和 512 → 边际降到 ~0.6ms/token）。GDN/QSA 注意力随 KV 长度线性、随 token 数可批。
- 由此推论：**每路 ≥50 需要 4 路轮 ≤40ms 且接受率 ≥2**；只靠批不能到——必须同时救接受率（见 §6）。
- **MTP 接受率才是 4 路慢的头号嫌疑**：单流独占（n_seq==1 fold 路径）推断接受 ~3/4；4 路基座（逐 lane 轮）~2.3；4 路多路链 ~1.6。同样的 GPU 每 token 成本，接受率坍塌直接除 2。

## 2. 硬件平台

| 项 | 事实 |
|---|---|
| GPU | 6×V100-SXM2-16GB（SM70）：fp16 TC ✓ dp4a ✓；bf16/fp8/int8-TC ✗ cp.async ✗。满频 1530MHz |
| 互联 | 双 NVLink 岛（GPU0-2 / GPU3-5），岛内 P2P AR；跨岛 = 激活交接（meta 后端异步拷贝）。PCI 顺序按 UUID 固定 |
| CPU/内存 | 2×POWER9（16 线程主循环，弱单核，kernel launch 贵）；31GB RAM/槽，64KiB 页 |
| 盘 | 唯一 SATA BX500：顺序 425MB/s、随机 4K 10-25MB/s，无 NVMe |
| 功耗档 | 软件 GPU 上限 nvidia-smi -pl（默认 300W；静音档 180W 不跨重启）+ CPU scaling_max_freq（2.3 静音 / 3.8 性能，跨重启保持）。300W 档实测满频 1530、温度 46-51°C、**零节流**、峰值 248-275W/卡 |
| PSU/风扇 | PSU0 风扇曾异常（RMA 报告在案，暂缓）；用户红线：**PSU 硬件不动、不满载烤机**；风扇守护正常勿动 |
| 带外 | BMC 仅 IPv6：`ssh -6 ac922-bmc6`（bmc.wuxuexi.top，密码 ~/.ssh/bmc-v6-password.txt）。开机：`busctl set-property xyz.openbmc_project.State.Chassis /xyz/openbmc_project/state/chassis0 xyz.openbmc_project.State.Chassis RequestedPowerTransition s xyz.openbmc_project.State.Chassis.Transition.On`。Mac 断 v6 时热点重连可救 |

## 3. 模型（checkpoint 钉死）

- **Qwen3.8-Flash-Next** IQ3_XXS-MTP-Q8：78.8GB，1541 张量；48 层 = **36 GDN（线性注意力）+ 12 QSA（稀疏注意力）**；**512 专家 MoE top-10**；**MTP-4** 投机（1 层 draft）
- GDN 状态：36 层×48 v-heads×128×128 F32 = 108 MiB/流（岛内 MIRRORED）
- PLE 表：per_layer_token_embd IQ4_NL [160, 320001536]，26.8GiB，90B/行，数据偏移 44.00GiB，每 token 16 行=1440B
- 专家 42.5GiB 随层分岛：岛0（blocks 0-24）20.2GiB / 岛1（25-48+out+MTP）25.9GiB
- 路径：`/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`

## 4. 拓扑与部署

- **层流水双岛**：每层整层归属单岛（MoE 路由/专家永不跨岛），岛内 TP3（tensor-split 1,1,1,1,1,1 + LLAMA_ISLAND_SPLIT=3,3），岛间激活交接
- 启动：`experiments/graph-max-20261011-1757/run-candidate.sh <dir>`（dir 内含 runtime/ + feature-env.sh）；端口 18098；`systemd-run --user --unit=zcode-XXX --property=StandardOutput=append:<dir>/server.log ...`
- 服务参数要点：`--split-mode tensor --kv-unified --ctx-size 65536 --parallel ${MONSTER_PARALLEL:-4} --cache-type-k/v q8_0 --flash-attn on --ubatch-size ${MONSTER_UBATCH:-1024} --spec-type draft-mtp --spec-draft-n-max 4 --threads 16 --numactl cpunodebind=0`
- 基座 env（run-candidate.sh 内固化）：P2P=1、ALLREDUCE=nccl、NCCL_F32=1、BF16_GEMV、MOE_REWORK、MOE_GROUPED、GDN_X4、QSA_SPARSE/GATHER=256/fused-indexer、MTP_FOLD/GPU_CHAIN/GPU_EMBED/FIFO/SEQ_LAYOUT/FOLD_MULTI、N_UBATCH=64、PAD=6、GPU_ARGMAX、TOPK_STABLE、GRAPH_OPT、FT_GRAPH_CACHE、META_ASYNC_COPY/FIFO_PIPELINE/COMPACT_ARENA/PACKED_CACHE、HC_FUSION、PREFILL_KV_PAD=8192、CPU_PLE_ROW_CACHE、PREFILL_FAIR_SHARE
- feature-env.sh 按臂覆盖（在 run-candidate.sh 之后 source）

### 4.1 关键 env/机制语义速查

| 开关 | 语义 | 实测结论 |
|---|---|---|
| `GGML_HARDWARE_TEST_POLICY_FILE`（文件内容 0-4） | 0 遗留 / 1 流水无结构 / 2 串行 / **3 流水+结构图** / 4 串行+结构 | policy=1 会静默 gate 死结构图（陷阱）；**=3 在新协议下 4 路 -6%**（v19 的 +10-11% 是 n=128 污染） |
| `GGML_CUDA_GRAPH_STRUCTURAL=1` | 跨形状结构图复用 | 同上，稳态负收益，暂不启用 |
| `LLAMA_MTP_GPU_CHAIN_MULTI=1` | 多路 GPU MTP 链（compact chain，lanes 2-8） | 日志确认激活（lanes=4，hidden=10240 rows）；**聚合中性 94≈97**，但轮时 4×21.9→1×70.7 |
| `LLAMA_SPEC_PAD_MULTI=N` | 多槽 verify 补齐固定 N token（图形状稳定） | **未测**——用户假说的直接实验 |
| `GGML_CUDA_GRAPH_PATH_LOG=1` | 每 256 趟打 eager/capture/replay 计数 | v19 数据：岛A replay 76%/eager 20%（全是 warmup 回退），岛B replay 46%/eager 34%；干净配置**未测** |
| `LLAMA_PLE_PREFETCH=1` | PLE 行缓存 CPU direct-I/O 暖炉（64MiB/job） | 4 路负收益（抢 POWER9 CPU），剔除 |
| `LLAMA_MTP_FOLD / GPU_CHAIN / GPU_CARRY / PAD=6` | 单流快链（fold 行布局+GPU 携带） | **n_seq==1 限定**——4 槽下自动全关，这是单流与多流 draft 质量差距的根源 |
| `MONSTER_PARALLEL` | 服务槽数（同 --parallel） | =1 单槽串行；测并发必须 =4 |
| `LLAMA_STEP_TIMING=1 PERIOD=N` | ctx 级 apply/build/alloc/inputs/compute 均値（stderr） | 诊断利器，代价≈0；DRAFT/FULL_STEP_TIMING 会互相干扰时再开 |
| `LLAMA_QSA_CANONICAL_TOPK / QSA_POOLED_MULTI` | MoE top-k 规范化（L2 专家驻留）/ 多路 QSA 池化 | **未测**（v30 被中断） |
| `GGML_SM70_DECODE_LINEAR_CHUNKS=8`、`FUSED_PROJ=2`、`BF16_CANONICAL_PLE` | v19 栈的内核快路径 | 未在干净协议下单测 |

### 4.2 代码地图（monster 树 /home/debian/ac922-monster/src）

| 文件 | 关键位置 | 内容 |
|---|---|---|
| `tools/server/server-context.cpp` | :3440-3640 update_slots 批构建 | generating/drafting 向量已支持多槽合批；handle_last_sampled_token 逐槽加 [sampled+draft] |
| `common/speculative.cpp` | :1495-1600 | **fold/gpu_chain/gpu_carry 全部 n_seq==1 限定**；GPU_CHAIN_MULTI 门控多路链 |
| `src/llama-context.cpp` | :567-570 pipeline_parallel 条件（split_mode==LAYER）；:1961/:1977 FIFO 互斥；:2080-2130 step-timing + multi-seq FIFO 资格 | B' 岛交错的两个挡板 |
| `src/llama-graph.h` | :838 allow_reuse | 形状/n_outputs/mtp 指针变化即拒复用 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | :887-920 FT graph cache；:1076-1135 PATH_LOG；:1755-1812 graph_ready + **5 秒未用即逐出** | 图缓存逐出是 warmup churn 候选 |
| `ggml-cuda/ple-conv-sm70.cu` | ONECAT 内核 | 4 路 assert 崩溃未修（contiguous） |

## 5. 正确性基线（已关账）

- **FP-noise 容差门已批准**：4 路 natural 对 fixed 的分岔经三重证据定性为 ULP 级归约噪声（PLE 3.6e-7 / MoE 8.9e-8 / 全关仍分岔且首差游走）
- **唯一真 bug = GDN head_offset 索引错读**（v13a 修复，fixture 12/12 绿，natural ×3 绿）
- 验证链规格：fixed 逐位 OFF/ON 同码 → natural ×3 → 单流锚点
- ONECAT（fused PLE conv）4 路 assert 未修，禁用

## 6. 待验证方向（按优先级，2026-10-12 晚状态）

1. **PATH_LOG 干净配置判决**（v31 被中断未跑）：多路 4×512 一波看 eager/replay 比。若 eager 高 → 用户假说成立（batch>1 图 miss→eager→POWER9 launch-bound），对策 = SPEC_PAD_MULTI 固定形状 + 变体桶化；若 replay 为主 → 瓶颈确认为接受率+边际成本，走 3
2. **MTP 接受率统计**：verbosity 4 的 `accepted x/y` 行或 /metrics——量化单流 ~3 vs 多路 1.6-2.3 的差距，定位 multi 链 draft 质量损失点（fold/gpu_carry 的多路化 = speculative.cpp n_seq==1 门放宽，中工程）
3. **LLAMA_SPEC_PAD_MULTI=24**：固定 20+4 token 形状，直接测图形状稳定收益
4. **CANONICAL_TOPK / QSA_POOLED_MULTI / CHUNKS=8**：v30 中断，重跑
5. draft 深度扫描 --spec-draft-n-max ∈ {4,6}（改 run-candidate.sh）
6. **B' 双岛交错**（大工程，规格在 TWO-TO-FOUR-STREAM-PLAN 附录）：岛空转 ~50% 回收，理论上限 ×2；前置 = 上述小件关账
7. Phase C 小件：v9 hashmap、QSA kvlen 钳制、ONECAT 修复、PLE 定向预取二期
8. **判死清单**（不再碰）：GPU-PLE 零拷、活跃专家走盘、int8 主 GEMM、部分驻留、NCCL 调参、8 流、GPU LRU 专家缓存、PLE_PREFETCH（CPU 暖炉版）、结构图（稳态协议下）

## 7. 运维速查

```bash
# 起服务（例：干净多路臂）
d=$EXP/e0-YYYYMMDD-name; mkdir -p $d/model; cp -r $BASE/runtime $d/model/
printf "export MONSTER_PARALLEL=4 ...\n" > $d/model/feature-env.sh
systemd-run --user --unit=zcode-NAME --property=StandardOutput=append:$d/model/server.log \
  --property=StandardError=append:$d/model/server.log bash $EXP/run-candidate.sh $d/model
# 健康等待 + 测波
curl -sm4 localhost:18098/health ; python3 wave-par.py 18098   # warm 4x128 + 2x 4x512
# 收尾
systemctl --user stop zcode-NAME && systemctl --user reset-failed zcode-NAME
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # 确认 0 MiB
# 功耗档（root）：nvidia-smi -pl 300；CPU: scaling_max_freq 3800000（跨重启保持）
# GPU 泄漏（root）：nvidia-smi --gpu-reset -i 0,1,2,3,4,5；uvm 缺失: sudo nvidia-modprobe -u -c 0
```

- 波脚本：`wave-par.py`（标准协议）、`wave-nvml.py`（带时钟采样）；prompt 库 `manifests/kv262k-realtext/prompt_tokens.json`（每 lane 取 8192）
- sudo 无 NOPASSWD（仅特定命令白名单）；nvidia-smi -pl/-lgc 需 root——**锁频实验（-lgc 1530）未做过**，若做需 root 通道
- 主机 ssh 失联 ≠ 挂机：先 `Tailscale ping ac922`（中继/直连都可能活着），再判断；BMC 是最后手段

## 8. 文件地图

| 位置 | 内容 |
|---|---|
| Mac `/Users/lyx/Desktop/ssh/monster-engine/` | **本文 ENGINE-MASTER.md**、TWO-TO-FOUR-STREAM-PLAN.md（B' 附录）、FULL-THROTTLE-PLAN.md、DESIGN-BLUEPRINT-v2.md、wave-*.py |
| 服务器 `~/ac922-monster/experiments/graph-max-20261011-1757/` | CURRENT-STATE.md（实验日志主档）、run-candidate.sh、run-perf-v21.py、wave-*.py、e0-* 实验目录（v26-base 基座 / v27-multi / v28-fast / v29-struct / v19-max 带全量计数日志） |
| 服务器 `~/ac922-monster/src/` | monster 源码树（monster-v0 + ~3000 行未提交；cumulative.patch 交付物） |
| 服务器 `~/ac922-llama-next/` | Codex 老树（baseline 冻结二进制 source 1afa5654；experiment/codex-multistream-20261010-162940 含 150 锚点 manifest） |
| 服务器 `~/ac922-llama-next/manifests/kv262k-realtext/` | 标准 prompt_tokens.json |

## 9. 历史速度谱系（同一协议不可跨档比较，仅存档）

| 时间点 | 配置 | 4 路聚合 | 备注 |
|---|---|---|---|
| v8 时代（180W） | n=128 波 | ~58.5 | 慢启动污染 |
| v19（180W，全 trace 栈） | n=128 | 62-69 | 插桩压制 |
| v20（300W+3.8GHz，trace 栈） | n=128 | 77-80 | |
| v26c 基座（300W，干净，n=512） | 真并行 | **88.8 / 97.4** | 首个可信稳态 |
| v27 +GPU_CHAIN_MULTI | n=512 | 94.4 / 94.5 | 中性 |
| v29 +结构图 policy3 | n=512 | 88.4 | 结构图 -6% |
| （假象）v26b2 "4 路"过单槽 | n=128/512 | "321-416" | 串行排队，非并发 |

**单流参考**：稳态 134-140（n≥512 后半）/ n=512 均值 107 / n=128 均值 68 / Codex 老树同日 65.5（n=128）。
