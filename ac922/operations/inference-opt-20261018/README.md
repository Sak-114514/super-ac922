# AC922 推理引擎优化战役档案（2026-10-16 → 10-18）

目标：Qwen3.8-Flash-Next（179B MoE，512 专家 top-10，MTP-4 投机，48 层 = 36 GDN + 12 QSA）
在 AC922（2×POWER9 + 6×V100-SXM2-16GB，双 3-GPU NVLink 岛）定制 llama.cpp fork 上的
推理速度穷尽式压榨。红线：PSU 硬件不动、不满载烤机、BMC 风扇守护（night-3000）全程在位。

## 成绩时间线（全部经语义门验证，接受率结构健康）

| 时间点 | decode 4 流 (tok/s) | prefill 聚合 (tok/s) | 关键变化 |
|---|---|---|---|
| 10-16 起点分层线 | 50.5 | 1265 | 加速层未通电 |
| 10-16 14:00 | **140.3** | — | 图组合（graph compose）修复后首跑，反超旧架构全史冠军 138.1 |
| 10-16 18:00 | 133（单 bank 拍平拓扑） | — | Meta 嵌套拍平 +6% |
| 10-17 09:18 | **248.4** | — | layer6-fourbank 四 bank 架构 |
| 10-17 14:39 | — | **1926** | MTP 词表行分片（v203）进全模型 |
| 10-17 18:00 | 282.6 | **2965** | 四 bank + 分片 + 内核军团定型（v234/v237） |
| 10-18 08:01 | 282.6（保持） | **4084** | UB 矩阵扫描 + 算子分块（v373） |

单发（单流）decode：115-131 tok/s（8K 真实语料口径）。MTP 接受率：96.5/90.8/86.5%（衰减结构，
均值步长 3.74/4，v13 假接受案后的真实水位）。

## 原理速览（各项优化为什么有效）

- **图组合（graph compose）**：把 eager 逐步 launch 的算子图捕获为可重放 CUDA 图，跨卡归约
  以 in-graph P2P AllReduce 融合。首跑（v31）崩于 cuGraphLaunch 空指针 → 最小探针复现为
  AR 组合死锁 → v76/v77 修复父图遍历+组合核 → v32 全模型全绿。见 `run-outcomes/V31-run-outcome.md`。
- **拍平拓扑（flat Meta6）**：Meta2{Meta3,Meta3} 嵌套层级摊平为六后端平铺，消除嵌套调度税（+6%）。
- **四 bank 缓存调度**：把单条同步大队列拆为四条独立 KV 池/执行上下文的传送带，各 bank 按自身
  节奏推进——显存带宽与张量核心时间上互相喂饱，图形状稳定命中，MTP 状态隔离不再串味。
- **MTP 词表行分片**（v203）：output 头（Q5_K 2560×248320）按行切四份摊到 GPU0/1/3/5，
  logits 逐位一致（bitdiff=0）；解放 GPU5（全机最挤卡）的 1GiB 重复 embedding。
- **UB 矩阵 + 算子分块**：锁页预算帽（512MiB）内扫描 ubatch 工作点（v373 ub512/256），
  配合 GDN/QSA prefill 分块策略（V375），prefill 2965→4084。
- **relay 10MiB 跨卡归约**（已探针验证、待接入）：大载荷归约从分段 ring（5.24ms）换整包接力
  （2.00ms），2.62×；六卡 FP64 oracle 逐位一致。见 `run-outcomes/RESULTS.md`。

## 目录

- `monster-docs/` — 引擎设计蓝图与总账（DESIGN-BLUEPRINT-v2 三路审核版、ENGINE-MASTER v1.1
  含数值乱码案终审、ALL-EXPERIMENTS 实验台账、P3 战报、2→4 流计划）
- `codex-notes/` — 优化会话的滚动计划文档（V101 分配器修复、v110 长稀疏 admit、v118 图执行
  fresh-exec 诊断、v119 virtual TopK 等）与 P4 阶段状态
- `run-outcomes/` — 关键轮次结案报告（V31 图组合崩溃案、V30B 分叉追踪、relay/qsa/Q5K RESULTS）
- `build-drivers/` — 各运行时版本的构建驱动日志与清单（V100→V310，二进制不入库，
  sha256 见 build json；完整产物在服务器 `~/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v*/`）

## 未决事项

- serial-vs-P4 分叉案：五审未结（第 1 层入口差异 16/16、内核 21/21 无罪、T4/T8 对照 exact、
  拓扑无关），压着多流全开的质量门
- relay 10MiB / TurboPrefill 跨岛（形态漂移 splits=7 vs 检查器三段式）/ 图组合接四 bank（等
  fresh-exec 验收）——三张已验证的牌待收
- nsys 大轮 ABRT 归因、monster 树 #29381 补丁回植
