# AC922 低内存与内存管理踩坑实录(Memory / low-mem pitfalls)

> 中文为主。AC922 的内存姿势和 x86 差异很大:HBM aperture 会被认成 DIMM、
> 编译和加载都容易把 swap 吃穿。以下是本机实际发生过的每一条。

## English abstract

POWER9 + 6× V100 memory pitfalls on one machine: six high-address NUMA nodes
that are actually V100 HBM apertures (offlining them breaks CUDA init with
cudaErrorInitializationError=3), swap exhaustion when all GPUs load weights at
once, the RAM-image build workflow that keeps a flaky SATA root disk alive,
systemd memory caps for compile jobs, and the CUDA-graph headroom that
32-byte-counting engineers always forget.

## 一、"幽灵 DIMM":六个高地址内存节点其实是 HBM

- 6 个高地址 NUMA 节点是 **V100 HBM aperture**,不是空槽 DIMM。旧脚本把它们
  当坏槽 offline,**后果是 CUDA 初始化直接失败**(`cudaErrorInitialization
  Error=3`)。
- 排查口诀:正常启动后 CUDA 报 err=3,别查库/权限/UVM 模块——直接查六个
  GPU HBM 节点是否 `Movable` 在线,以及 nvidia-startup/persistenced 的启动
  顺序。
- `nvidia-persistenced` 持有 `/dev/nvidia0..5` 是**正常现象**;手动让 HBM
  回 offline 是有意杠杆,不是故障。
- 旧 bench 结果(幽灵内存时期)全部作废,不要引用。

## 二、低内存防泄漏:把每次 OOM 拒之门外

1. **六卡同时加载权重会把 swap 吃穿**(TP4 容量探测时实测接近满额)。
   多模型/多实例加载必须错峰,一次只让一个 runtime 分配显存。
2. **编译任务用 systemd 限内存**:`MemoryHigh=12G`、`MemoryMax=18G`,
   配合 `MAX_JOBS=1~3`。POWER9 上链接器的内存峰值很容易失控。
3. **构建临时文件放 RAM**(`/dev/shm`),完成后把 wheel 复制出来并校验。
   这不是为了快,是为了保护根盘(本机 SATA 控制器有 EEH 病史,减少根盘
   I/O = 减少触发面)。
4. **模型权重不要搬进 RAM 镜像**:drafter 3.85 GB、目标模型 20 GB,放进去
   会挤掉编译和页缓存的空间。
5. **CUDA Graph 捕获要留显存头寸**:权重装下 ≠ 能启动。0.95 利用率报
   449k KV tokens,图捕获时仍差 ~810 MiB OOM;降到 0.90 才稳。16GB 卡
   记住:最后 ~800 MiB 是图捕获的,不是你的。
6. **投机解码的 KV 池要单独算账**:drafter 权重(BF16,~4 GB)+ 草稿 KV
   都从同一池子扣;同一张卡上 depth 7 + 200k 窗口 + 0.90 利用率装不下,
   实测可行组合见 `tp3-dflash2-trial/REPORT.md`。
7. **显存读数在加载暂存期会虚高**(瞬时冲到 14–16 GiB/卡),判断稳定占用
   要等加载结束,不要在加载中途做 OOM 判断。

## 三、与存储故障的交叉验证(防止错误归因)

- 四次 SATA EEH 故障都发生在同一 PHB 分支,且故障瞬间 HBM 均在线——
  NCQ、6 Gb/s 链路速率都不是必要条件;不能把内存热插拔状态错配到故障
  时刻来定罪(我们就犯过一次,已废弃该结论)。
- `SErr=0xffffffff` 单独不能证明任何根因;被隔离设备的寄存器读取就是
  全 1。先换线/换口/换盘做对照,再下结论。
- `mem=32G` 启动参数与 DMA bypass、HBM 热插拔的交互**未经证明**,不要
  当修复方案部署。

## 四、一键自查清单

- [ ] 六个 HBM 节点在线、`Movable`?
- [ ] nvidia-startup / persistenced 启动顺序正确?
- [ ] swap 当前用量 vs 六卡加载峰值?
- [ ] 编译服务的 systemd 内存上限生效?
- [ ] /dev/shm 构建产物已迁出并校验?
- [ ] vLLM 利用率参数给图捕获留了 ≥0.05 余量?
