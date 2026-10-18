# monster-engine 工作区（Mac 镜像）

> **版本 v1.2 · 2026-10-14 16:45 · 状态：当前**。v1.1 及更早未版本化。
> **权威入口：`ENGINE-MASTER.md`（v1.1，含 2026-10-14 数值乱码案终审判决 §0.5）**。实验逐条台账：`ALL-EXPERIMENTS.md`（v1.1，含 §F 判决臂）。

服务器主战场：`/home/debian/ac922-monster/`（src/ = monster 树 HEAD 1e542c5 + 未提交补丁）。
**两棵树两个世界**：`/home/debian/ac922-llama-next/sources/llama-v100-v3`（= 干净产线 b3 的源，含 17+5 未提交修复）与 monster 树分叉，monster 系二进制 2026-10-14 判语义乱码，修复移植补丁在本区 `patches/`。

## 文档（按时间，全部已版本化）
- `ENGINE-MASTER.md` — **唯一权威总览**（v1.1）
- `ALL-EXPERIMENTS.md` — 实验台账（v1.1）
- `CURRENT-SESSION-LOG.md` — 会话日志（v3）
- `DESIGN-BLUEPRINT-v2.md` — 总设计（三路审核版，2026-10-10；v1 已归档）
- `TWO-TO-FOUR-STREAM-PLAN.md` — B' 双岛交错主计划（v4，2026-10-11 夜）
- `P3-BLUEPRINT.md` / `P3-REPACKER-DESIGN.md` — monpack 线（2026-10-13；-iq3xxs-v1 已归档）
- `P3-WARREPORT-20261013.md` — monpack 战报（数字全部 b3 系干净基底，有效）
- `SCHED-AUDIT-20261014.md` — 调度审计（当日）
- `VLLM-ROUTE-*.md` — 路线裁决四件套（2026-10-12，判"留守"）

## patches/（补丁存档，2026-10-14 起建立）
- `V100V3-FULL-DIRTY-STATE-20261014.patch` — **v100v3 全量脏状态（17 文件，2834 行）**——干净语义的唯一来源
- `v100v3-new-headers/` — 配套 5 个新头文件（speculative-mtp-layout / llama-mtp-row-order / server-prompt-cache-policy / server-scheduling / server-token-stream）
- `V100V3-DIRTY-FIXES-missing-from-monster.patch` — 数值五件套提取版（qsa-attn / moe-weighted-reduction / ggml-cuda / meta / speculative）
- `C1-tp6-fixes.patch` — C1 两补丁（AR 分桶核 + FIFO 单岛门；判清白，待干净基底重测）
- `P2-PER-SEQ-CARRY-draft.patch` / `P2-SHAPE-TAX-draft.patch` — 二期草案（carry 已被 Codex 独立实现并领先；形状税未实现）

## 工具脚本
- `launch-p3-arm.sh`（臂式启动器：ISLAND/UB/P2P_AR/RING_MIN/EXTRA/BINARY/SLOTS/FAST）→ 服务器 `/home/debian/ac922-monster/`
- `c1-run-arm.sh`（全自动测速臂）· `text-probe.sh` / `long-probe.sh` / `ringoff-probe.sh` 等判决探针
- 服务器 `run_test_server.sh` 已参数化（AC922_TEST_KV_K/KV_V/FA/BINARY/SLOTS，.bak-kv 备份在）

## 机器状态（2026-10-14 16:45）
- GPU 空闲、p3-ab inactive；Codex hw-orchestrate 矩阵已收卷（final-state.json 14:52）
- BMC 三通道全部在线（DDNS 31e0 前缀 / Tailscale / 直连）；静音档保持
- PSU0 风扇异常是硬件问题（换货报告见 ../PSU0-风扇异常-换货报告.md）；用户红线：PSU 硬件不动、不满载烤机、风扇守护勿动
