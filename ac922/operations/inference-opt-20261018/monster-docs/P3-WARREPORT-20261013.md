# P3 monpack 战报·终版（2026-10-13/14 通宵施工）

> 一句话：一夜完成 IQ 家族离线重打包 + V100 张量核化全栈施工（转换器/内核/加载器/分派/切片重打包）。
> **正确性：IQ4_NL+Q8_0 族已在引擎内端到端逐位验证通过；grid 族（IQ2/IQ3 五类）GEMV 路径待复刻（病根已定位，与 Q2_0 已修案例同模式）。
> 性能：TC 内核 v1 功能完整不再崩溃，但暂慢于 fork 调优过的 grouped-dp4a 基线（无加速）；终局加速待 grid 族修复 + TC 性能迭代。**
> 部署状态：**边车已改名禁用**（`.disabled-by-p3`），任何发射=100% 旧路径零风险；恢复=改回原名。

## 1. 终局数字（全部 28K-token 预填 TTFT 法，贪心 sha 对照）

| 臂 | 配置 | 预填 tok/s（热） | gen_sha16 | 判定 |
|---|---|---|---|---|
| 基线@512 | MONSTER_MONPACK=0, UBATCH=512 | **1478.9** | e39035627dbc3f7a | 锚点（三发确定） |
| 基线@1024 | =0, UBATCH=1024 | 1438.8 | e39035627dbc3f7a | 批加大在旧路径**无益**（−2.7%）；贪心与批大小正交性 ✓ |
| GEMV TYPES=20 | 边车+GEMV，仅 IQ4_NL | 1474.6 | e39035627dbc3f7a | **逐位通过，零回退** |
| GEMV TYPES=20,8 | +Q8_0 | 1380.5 | e39035627dbc3f7a | **逐位通过** |
| GEMV 全量/no42 | 含 grid 族 | 999-1120 | 垃圾(len≈378) | grid 族 GEMV 发散（见 §3） |
| TC v1 @512 | TC_MIN=16 | 712-863 | 确定性 | 不崩、确定性，但**慢于基线**（权重错位期测得，量级可信——计算与数值无关） |
| TC v1 FAM-B | TYPES=16,17,21,22 | 825-863 | — | 同上，族间差 ~100 |

## 2. 交付物（全部落盘）
- 边车 `IQ3_XXS-MTP-Q8.gguf.monpack.disabled-by-p3`（45.6G，T0 147/147 逐位往返，T1 8/8 C 参考逐位）+ manifest
- `monpack_format.h` 格式宪法（layout_rev=1）；`~/monpack/mrt-repack.py`+`mrt-t1.py`+`ref.c`（机器抽取参考）
- 内核 `ggml/src/ggml-cuda/monpack/`：GEMV 8 类型 + SM70-moe-rework Q2_0 复刻 + TC grouped（per-device work-list、两阶段 scatter、lane 相对片段装载）——单测全绿 + sanitizer 干净
- 加载器 v2 `src/src/llama-monpack-loader.cpp`：meta 岛缓冲切片感知重打包（K 切+N 切、先验证后注册再上传、失败整张量回退）+ `ggml-backend-meta.cpp` 访问器 + `ggml-cuda.cu` 两钩子
- A/B 工具：`launch-p3.sh`（MONPACK_ARM/TC_MIN/UB/TYPES 四旋钮）+ `~/monpack/p3-probe.py`
- 快照：`pre-p3-uncommitted.patch` + 各文件 .pre-p3.bak/.i2bak

## 3. 侦探链与病根存档（按发现序）
1. 三臂同速同 sha → 新内核没跑（I 的 override 拒了 META 型 buffer + async 分支跳过）
2. 岛内 TP3 切片：down=K 切(192/224/224 或 256/192/192)、gate/up=N 切(256/192/192)——tile 布局下切片字节不连续 → I2 的 per-slice 重打包解决
3. index family 字段 R 侧硬编码 1（129 条原地修补+CRC）；透传过校验；blob_off 绝对/相对口径（K2 发现 :515 双重加基 → 已修，恢复逐位）
4. TC 三连 bug（K2）：进程级单设备 g_tc；mma tile 32 线程契约被 256 线程违反（越界读 smem）；work-list rows[] 槽位碰撞（原子秩当全局槽）
5. Q2_0 GEMV 非逐位真因：fork 在 GGML_SM70_MOE_REWORK=1 下走 kernel_sm70_moe.cuh 的不同求和结构 → K2 已复刻（单测逐位）
6. **现存唯一拦路虎（下一场）**：grid 族（16/17/18/21/22）同病——引擎走的向量结构（mmvq 的 GLU/gate 融合或 rework 变体）≠ K 复刻的标准 vec_dot。修复模式已由 Q2_0 案例示范：逐行复刻引擎实际路径。types 20+8+42 已逐位/无罪。
7. 224 份额 57 张保守回退（224%64=32）——v2 用 N-padding 补零行可解（格式已支持 d=0 行）。

## 4. 性能判定（为什么今晚没有加速）
- **旧基线已经很强**：fork 的 grouped-dp4a MMQ 直读压缩字节（P15 证据），P3 预设的"消灭反量化写回 9.5×"在此基线不适用（它没有写回）。
- M̄≈10 带宽区，TC 与 dp4a 理论趋同；TC v1 的实现开销（M-tile 16 浪费 37%、smem 装配、J=16 细粒度）使其实测 ~1.8× 慢于调优多年的 fork 路径。
- **加速路径仍在**：①grid 族修复后全量逐位 + TC@UBATCH=1024（M̄=20 协同假设待验）②TC v2 性能迭代（寄存器直装、M-tile 8、k-外层复用加强）③224 份额 padding 化补齐 39% 字节覆盖。
- UBATCH 结论：旧路径 512>1024（1478 vs 1439），fork 历史 A/B 胜者复核成立。

## 5. 下一场开工清单（按优先级）
1. grid 族 GEMV 引擎路径复刻（病根已定位，照 Q2_0 模式；预计半天）
2. 全量逐位门 → TC@512/1024 复测（真权重）
3. TC v2 性能迭代（对照 grouped-dp4a 的 profile 找差在哪一段）
4. 224 padding、T3 全量 200 条语料质量门、上游 issue 回报（monster fork 生态无此义务，略）

## 6. 关机
按用户指令于收尾后执行 sudo poweroff。1cat-tp2pp3 常备单元保持 stopped（整机已关）；下次开机后 `systemd-run` 原命令可拉回。
