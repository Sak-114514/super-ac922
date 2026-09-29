# AC922 家用温控踩坑实录(Fan / Thermal pitfalls)

> 中文为主。适用对象:把 AC922(8335-GTW)放在家里/办公室、自己接管 BMC
> 风扇策略的人。每一条都是本机实际发生过的事,不是理论风险。

## English abstract

Running an AC922 outside a datacenter means you own the BMC fan policy. This
note records the hard boundaries we found: the 2549 RPM low-speed fault line,
the 3100 RPM practical quiet floor, the OCC activation window with phantom
zero-RPM reads, a real incident where a fan-policy change timed out and the
BMC killed power during a compile (mid-campaign, 2026-09-29), and the operating
rules that kept a 3×V100 speculative-decoding campaign at 43–57 °C with zero
thermal events.

## 硬边界(调风扇前必读)

1. **低速故障线 2549 RPM**。fan-monitor 的故障判定线;静音目标永远不要
   往这条线以下压。
2. **3100 RPM 是实际静音下限**。2900 档看似更安静,但 fan0 余量不足时会
   触发自保、整组跳 4300——比 3100 吵得多。
3. **OCC 激活窗口**:主机上电后约 75–90 秒 OCC 才接管,期间副转速通道
   (fan5/7/8)会读零毛刺。监控脚本若对所有通道采样,会在这个窗口看到瞬时
   "故障"跳 4000 RPM——这是正常时序现象,别当 bug 修,更别让脚本在这个
   窗口自动出手。
4. **60 °C 是本机的敏感线**:GPU 到 60 °C 曾触发过误报警(见下)。整个
   TP3+DFlash2 战役(含四并发 30 分钟长测)GPU 实测 43–57 °C,说明负载
   本身不危险,危险的是风扇策略切换的瞬间。

## 事故复盘:风扇策略切换 → 编译中途断电(2026-09-29)

- **经过**:agent 启动 fan-monitor 后,一条 legacy `systemctl` 操作超时,
  BMC 把低转速误判为三路风扇失效,watchdog 直接断电。当时编译正在跑,
  被打断。
- **根因**:风扇策略变更没有走已验证的 BMC 途径;超时的中间态恰好撞上
  BMC 的故障判定。
- **后续防线**(全部已落地):
  1. 风扇/温控**彻底移出服务路径**——推理试验只读温度,不再写风扇;
  2. `/etc` 持久化例外与 watchdog 依赖已文档化,改策略前先核对;
  3. 试验期间只监控温度(本轮全程 43–57 °C),异常即暂停试验流程。

## 操作规则(血泪版)

- **两条 PSU 均衡供电**,不做单电源操作;编译等高负载时确认两条 AC 输入
  电压、负载可比。
- **不写未经核实的 PMBus 寄存器**。
- 挂死用 **BMC soft shutdown**,不要拔整机电源:BMC 冷启动可能切换到主
  flash bank(coral 固件已坏、bootcount 无安全清法),整套配置要重来。
- 开机后 90 秒内部署的任何守护进程,**不要**对风扇通道做自动动作(先过
  OCC 激活窗口)。
- 风扇策略的任何变更:改之前记录当前档位,改之后立即核对三路实际转速
  与报警计数,不对劲立刻回滚。

## 当前稳定档(供参考)

三路 target 3000 RPM、实际 2749–2849 RPM、报警 0;watchdog active;
GPU 满载(四并发投机解码)51–57 °C。这组参数经过了 30 分钟持续压力验证。
