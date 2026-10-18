# BMC fan guard

These files are the source snapshot used on one IBM Power System AC922 8335-GTW
water-cooled Witherspoon BMC. The installed code was verified byte-for-byte
against the local copies before publication.

## Files

- `boot_guard.py`: Python 2.7 control loop. Reads CPU/GPU/DIMM/VDD/PCIe/ambient
  sensors and MAX31785 tach/target values; keeps the six controlled tachs,
  monitor, and watchdog in its safety policy.
- `boot_pin.sh`: holds runtime masks on the original target writer while the
  custom guard owns the RPM target.
- `restore.sh`: returns targets to the configured safe value and restores the
  4000-RPM guarded fallback after the custom service stops. It keeps the
  factory writer masked; it does not restart original fan control.
- `ac922-boot-guard.service`: systemd unit that orders the guard with chassis
  power-on and invokes both scripts.

## AC922 GTW-specific details

The current mapping controls MAX31785 fan targets `fan1_target`, `fan3_target`,
and `fan4_target` (chassis fan0/fan2/fan3). The water-cooled filler fan has no
target. The Witherspoon hardware watchdog remains active. **In the deployed
night-3000 profile, `phosphor-fan-monitor@0.service` is masked.** The stock
monitor expects the factory 4300-RPM target: when it was unmasked while these
fans targeted 3000 RPM, it marked all three fans nonfunctional and OpenBMC
immediately powered the host off. Do not enable this quiet profile and the
stock monitor together unless that monitor's thresholds have first been
reviewed and adjusted for the target RPM. The custom guard reads both tach
channels itself, but that is not a replacement for the stock monitor or the
independent hardware watchdog.

This policy uses explicit temperature tiers and a 2650-RPM primary-tach
safety floor; it does not modify MAX31785 permanent hardware thresholds.

The published guard starts at 3000 RPM, raises to 3500 when an air-cooled
temperature reaches 45°C, and to 4000 when any monitored temperature reaches
55°C. At 60°C it enters its emergency handoff. CPU/GPU temperatures still
trigger 55/60°C protection but do not promote the fan-moving tier on this
water-cooled GTW. Downshifts use a 90-second hold at 42°C and 52°C. Startup
retries OCC temperatures and transient secondary tach zeros; repeated sensor,
tach, target-writer, or service failures select the 4000-RPM fallback.

The service is not a generic OpenBMC fan manager. Before adapting it, verify
the BMC firmware generation, hwmon indices, D-Bus sensor names, water/air
cooling profile, watchdog takeover behavior, and the actual unit writers.
Firmware updates can change every one of those interfaces.

## Manual deployment outline

1. Back up the current BMC service, masks, and source files.
2. Copy scripts to `/etc/ac922-fan-quiet/` and the unit to
   `/etc/systemd/system/` with root ownership and restrictive write access.
   The deployed configuration also has persistent masks for the factory
   `phosphor-fan-control@0` and `phosphor-fan-control-init@0` writers, plus a
   persistent mask for `phosphor-fan-monitor@0`. The source package does not
   install these masks automatically; review the monitor incompatibility above
   before reproducing the low-RPM profile.
3. Run the BMC's installed Python 2.7 syntax check and review the D-Bus/sysfs
   mapping.
4. Keep factory targets available until a preflight proves sensor coverage,
   all tach channels, watchdog, and target readback. The low-RPM deployment
   does not run the original fan-monitor because its factory threshold caused
   a chassis power-off at 3000 RPM.
5. Test stop and failure restoration at factory RPM before enabling at boot.

Do not copy the test procedure in [`../experiments/fan_trial_bmc.py`](../experiments/fan_trial_bmc.py)
into a boot service; it deliberately writes temporary targets for supervised
low-load experiments.
