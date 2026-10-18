# qwen4exp-monster · 2–4 流主攻最终计划（FINAL v4）

2026-10-12 定稿。经三轮子代理攻防论证（事实审核 → 红队攻击 → 方向侦察 → 用户三轮质询），全部证据对账到 analysis.json / suite.json / 三线日志原文。

---

## 0. 已锁定决策（不再复议）

1. **放弃 8 流，主攻 2-4 流**。8 路聚合 79.1 ≈ 4 路 78.6，批红利已尽且正确性未关。
2. **留在 monster fork**（`/home/debian/ac922-monster/src`，monster-v0 分支 + Codex ~3000 行未提交工作树，交付物 cumulative.patch + 50 文件哈希清单）。不自研引擎、不重置工作树。
3. **正确性优先于一切性能实验**：natural parity 不绿，前缀缓存等方向不动。
4. 模型/硬件固定：Qwen3.8-Flash-Next IQ3_XXS-MTP-Q8；AC922 双岛层流水拓扑。

## 1. 事实基线（全部核实）

| 事实 | 数字 | 出处 |
|---|---|---|
| 单流锚点（**300W 时代，禁跨档比较**） | 1350.8 PP / 150.7 TG | 蓝图 §8 |
| 当前 caps（生效中） | CPU 2.3GHz 全核 / 6×GPU 180W | graph-max CURRENT-STATE |
| 4 路 fixed TG（300W） | 14.4-23.2/流，聚合 78.6 | v5a analysis.json |
| 4 路整波 | 49.9s，其中 **decode 仅 9-10s，波墙 80% 是预填** | v5a suite.json |
| v8 干净 decode 窗口（180W） | 155.2ms = compute 107.7 + **alloc 39.4（100% 来自图重建慢路）** | ASSESSMENT model-v8 节 |
| 结构图 v5a 实测 | 4 路 +10.3~11.0%（单对数据，序效应 +26% 需复跑）；2 路微负；聚合计 replay target 84 / MTP 1329 次每设备 | v5a |
| 4 路 natural 失配 | 首偏 92/109（v5a）→ 40/122/114（v8）；**根因候选 = GDN head_offset 索引 bug**（fixture 8/12 复现；v13 修复 staged，验证待 v13a）；第二嫌疑 = QSA indexer 形状精度（T≥8 分数差 ~2e-4） | CURRENT-STATE |
| 岛间流水气泡 | 单批流下**每岛空转 ~50%**；带宽下限 ~3ms vs 实测 compute 107ms——非带宽瓶颈 | 本计划 §2.4 推演 |
| 预取器有效率 | 仅 11.6%（计划 393,216 行、warmed 45,718）——激活≠有效 | v5a server.log |
| 测量卫生 | MOE_GROUPED / SM70_MOE_REWORK / GDN_X4 / QSA pooled / target FIFO / Meta copy 全部无命中标记；policy 文件逐波可变，终态=2 会 gate 死结构图 | OPTIMIZATION-PATHS |

## 2. 四层论证（浓缩版）

**2.1 图税**：L 流批形状组合 5^L（1/2/4/8 = 5/25/625/390625）。v5a 实测适用区间：2 路基础缓存够（结构图微负），4 路是甜点（+10-11%）。业界同族：vLLM/TensorRT-LLM 形状桶化、piecewise 捕获。
**2.2 非确定性**：业界根因 = 归约序随批维度（Thinking Machines 2025）。**但本案首要嫌疑是索引 bug**——容差门仅限分岔被逐层证明为浮点类之后，禁止用容差放行 fixture 已证明的错读。
**2.3 调度**：长预填独占→解码停摆（Sarathi-Serve OSD'24：分块预填+混合批次，32K 最差 TTFT -68%）。对应物：FAIR_SHARE（旧线预填聚合 +83%，P0 零成本未并入）+ FIFO multi（已证激活）。
**2.4 六卡气泡（第三轮攻防成果）**：双岛层流水单批流每岛空转 ~50%；带宽下限分析（~3ms vs 107ms）证明可回收。手段 = micro-batch 1F1B 流水填充（4 路按 lane 拆 2 微批错峰进岛），吞吐 1/(T₀+T₁) → 1/max(T₀,T₁) ≈ **2×**。佐证：旧线 smlayer 同族弱化版 +14.6% PP；层流水 token-37 旧案与 GDN bug 同族，v13a 关雷后通路打开。
**2.5 批投机甜点**：spec 增益集中中低并发（vLLM 经验）；4 路后批红利已尽。

## 3. 分阶段目标与杠杆乘法账

| 阶段 | 内容 | 目标 | 杠杆 |
|---|---|---|---|
| **A** | 正确性 + 归因仪表 | 4 路 natural 门 ≥3 跑全绿；binder 判定（软件税/功耗墙/带宽） | — |
| **B** | 税清零 | 4 路 TG ≥30/流（聚合 ≥120 > 单流 = 真多流正收益） | 结构图 ×1.11 + 重建慢路消灭 ×~1.3 + FAIR_SHARE（预填） |
| **B'** | 流水填充（结构性） | 4 路 ≥50/流 | ×~1.8（岛气泡回收）[+300W 档 ×1.21 若用户选] |
| **C** | 波墙/TTFT | 共享前缀场景波墙 -30-60% | 前缀快照缓存 + 预填调度 + PLE 预取二期 |
| 合计 | 78.6 × 1.11 × 1.3 × 1.8 × 1.21 ≈ **247 聚合 ≈ 62/流** | **原蓝图 50-70/流可达** | 三杠杆 + 功耗档全落地 |

## 4. 执行铁律（每条实验都适用）

1. **同 caps 档内比较**；每步前后各一次单流短探测（<1min）；锚点每档 ≥2 跑取稳态
2. **A/B 交错复跑 OFF/ON×2，弃首个波次**（实测序效应 +26% > 效应值）
3. **natural 门 ≥3 跑全绿**才算过；fixture 与模型运行各留 **ldd 记录**（字面路径加载错库事故已发生两次）
4. 每波记录：最终 policy 号 + 结构图 capture/replay 计数 + chain-enabled 日志
5. PSU 换件 / BMC 动作 = 强制重锚；PSU 热拔仅限空载
6. 编译 -j≤8；回滚用各阶段 rollback 脚本；不复用被污染的 v13 输出
7. flag 写了 ≠ 内核跑到——归因必须靠 E5 计数器，不靠 env 名单

## 5. 实验矩阵（第一轮执行，顺序：E0→E1→E5→E3→E2→E4→E6→E7→E8）

| # | 实验 | 关键点 | 通过标准 |
|---|---|---|---|
| **E0** | **gdn-head-v13a 验证**（全案第一优先） | fixture（ldd）→ 4 路 natural ×3。**判别分支**：fixture 绿但模型红 → v12 逐层 oracle + `LLAMA_QSA_PATH_TRACE=1` + `LLAMA_QSA_INDEXER_TRACE=1` 按首差层分流；禁止直接走容差门 | offset16/32 fixture 12/12 + natural ×3 绿，或第二根因明确 |
| **E1** | caps 决策 + 工作点审计 | NVML throttle 归因 + 锁频对照 + 墙功率峰值（零成本）；用户拍板档位 | 两档锚点 ×2 + 功耗墙数据 |
| **E5** | 仪表先行 | 计数器全清单（MOE_GROUPED / SM70_MOE_REWORK / GDN_X4 / QSA pooled / target FIFO / Meta copy route）+ 分相位计时（`LLAMA_SPEC_TIMING=1`、PLE 逐波、CUDA event）+ **分卡 NVML 稳态采样（直接拍岛间交替空闲 = §2.4 气泡证据；SM util 判定闲置 vs 饱和）** | 一次构建 + 4 路一跑出全量归因 |
| **E3** | 关图对照 | `GGML_CUDA_GRAPHS=0` × 4 路，交错 2 跑 | 给 E2 同会同档分母 |
| **E2** | 结构图 4 路 | 交错 OFF/ON×2 弃首波；policy 3/4 且逐波回显；**replay 计数 >0**；OFF/ON 臂除结构图 gate 外 policy 等价 | TG ≥ +10% 且超复跑散布；natural 同步查 |
| **E4** | FAIR_SHARE | 先加激活日志（grep 不算验证），再开；预填聚合 ≥ +15% 且超散布 | 激活确认 + 数字 |
| **E6** | MAX_CATCHUP 扫描 | {0,1,2} × 4 路 × fixed，每点 ≥2 跑；GDN X4 计数联动确认（catchup 翻转 n_tokens 分派） | TG 曲线 + 噪声带 |
| **E7** | 单流保底回归 | 同档锚点对照 | -2% 内（红线） |
| **E8** | 大页 + 绑核 | PLE mmap 16GB 大页、主循环独核；E5 计时为前置 | 每步主机时间可归因下降 |

## 6. 第二梯队（阶段 B/B' 按序启动）

| 序 | 方向 | 收益 | 工程 | 前置 |
|---|---|---|---|---|
| 1 | **重建慢路消灭**：高水位预 reserve + 图结果桶键控（`(target/draft, phase, lanes, depth, 布局generation)`） | 4 路 TG +5-15% | 小-中 | E5 计时确认 |
| 2 | **双岛流水填充（阶段 B' 主线）**：micro-batch 1F1B，lane 组错峰进岛 | 聚合 ~2× → ≥50/流 | 中-大 | natural parity 绿 + 税清零 |
| 3 | 混合前缀快照缓存：先接口+拒绝理由，后启用（版本化快照+COW） | 共享前缀波墙 -30-60% | 接口小/启用中 | parity 硬门 |
| 4 | v9 hashmap 重测（判决作废——实际加载了 v8） | init 慢路 -20-30% ≈ 窗口 ~5% | 极小 | 修脚本路径 |
| 5 | goodput 自适应 MTP 深度（2/3/4 桶 + 滞回） | 低接受流 +5-15% | 小-中 | 描述符预算 |
| 6 | PLE 预取二期：定向预取（230 行）+ page-merge + dd 预热 | 冷 TTFT 减半到 1/3 | 小-中 | 存储口径确认 |
| 7 | QSA kvlen 桶钳制（旧线排队件移植） | 8K 预填 +2-5% | 小-中 | 图复用交互验证 |

**明确不做**：GPU 侧 PLE 零拷读（新数值路径，GPU-PLE 家族判死）、树形 draft（无证据）、NCCL 调参家族、图缓存 TTL/cap、连续批处理短上下文（旧线判死）。

## 7. 决策点（执行中问用户）

1. **功耗档**：保持 180W 静音还是回 300W 性能（E1 数据后；是所有目标的乘法系数）
2. **E0 若 fixture 绿模型红**：按判别分支走，容差门仅限浮点类分岔
3. **E2 若红**：先查 replay 计数是否为 0（没吃到图 ≠ 图有害），再决定修 vs 形状桶化（~100 行）
4. 阶段目标确认：A/B/B'/C 分阶段替代一次性 50-70（§3 已给乘法账，62/流可达）

## 8. 风险与回退

- 全部 env/补丁级 + 各阶段 rollback；单流红线同档 -2%
- 流水填充与结构图/MTP fold 交互复杂——先税清零再动结构
- 前缀快照启用被 parity 门挡住是特性不是缺陷（接口先行零风险）
- 16GB 卡显存预算：高水位 reserve 与描述符增殖需先查
- 墙功率峰值未测（922W 只是快照）；BMC 无 1000W 硬上限支持

## 9. 机器与恢复手册

- 开机：`ssh -6 ac922-bmc6`（仅 IPv6）→ Redfish power-on；失败物理电源键
- 开机后：`/dev/nvidia-uvm` 缺失 → `sudo nvidia-modprobe -u -c 0`
- GPU 泄漏：`nvidia-smi --gpu-reset -i 0,1,2,3,4,5`（root）
- 必读：`experiments/graph-max-20261011-1757/CURRENT-STATE.md` + ASSESSMENT + OPTIMIZATION-PATHS
- Mac 镜像：`/Users/lyx/Desktop/ssh/monster-engine/`；PSU0 换货进行中（硬件问题，报告见 ssh 根目录）

## 10. 文献

- 图：vLLM cuda_piecewise_backend；TensorRT-LLM capture；persistent draft metadata RFC（2026-07）
- 非确定性：Thinking Machines《Defeating Nondeterminism in LLM Inference》（2025-09）；vLLM #966；Batch Invariance
- 调度：Sarathi-Serve（OSDI'24, arXiv 2403.02310）；SARATHI（2308.16369）
- 批投机：TETRIS（ACL'25）；Batch Speculative Decoding Done Right（2025-10）；vLLM spec 文档
- 流水：GPipe/1F1B 经典调度；SGLang Unified Radix Cache（前缀缓存边界设计）

---

## 附录 B' 实现定位（2026-10-12 代码勘察结论）

**机制已存在**：`cparams.pipeline_parallel`（llama-context.cpp:567-596）——异步图提交（`graph_compute_async`）+ 前图未完时的新图错峰 + 事件同步（:2039 注释明示）。上游设计给 split-mode=layer 用。

**挡板**：:568 条件 `split_mode == LLAMA_SPLIT_MODE_LAYER`——我们赢家拓扑是 tensor-split 双岛（meta 后端），流水并行从未激活。

**改动方案**（中工程，不动 meta 后端本体）：
1. 放宽 :568 条件：meta 2-岛 tensor-split（`model.n_devices()==2` 且 `n_gpu_layers > n_layer_all`）也允许开（multi-FIFO 本就要求 n_devices==2，:1974）
2. 确认 meta 后端上报 async/events caps（ggml-backend-meta.cpp 设备 props；异步 MIRRORED 岛拷贝已存在=桥已异步）
3. 输入张量双缓冲（:2041 的 reuse-sync 已处理前图读冲突）
4. 验证链：stateful meta fixture → fixed 逐位 → natural ×3 → 单流 150 锚点
5. 风险：sched 内存增长（:564 注释警告）、与结构图双活图交互（结构注册表已支持多活图对象）

**量级预期**：岛空转 ~50% 回收 → 聚合上限 ~2×（300W 档 77-80 → 110-140 理论），受软件税折减。
