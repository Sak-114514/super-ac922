# AC922 operations source provenance

Snapshot date: 2026-09-28. This note distinguishes sources copied from the
deployed AC922 from templates intended for local adaptation. It is not a
claim that every service should be enabled on every machine.

## Deployed source snapshots

The following source hashes were read from the live AC922/BMC and match the
published files at snapshot time. The current quiet BMC profile intentionally
has persistent and runtime `/dev/null` masks for `phosphor-fan-monitor@0.service`;
this is documented as a specific compatibility workaround, not as a standard
OpenBMC recommendation. During publication review the mask was briefly removed,
the stock monitor marked three fans nonfunctional and OpenBMC powered the host
off; the mask and guarded 3000-RPM state were restored afterward. The watchdog
remained active. See [`bmc/README.md`](bmc/README.md).

| Deployed file | Published file | SHA-256 |
| --- | --- | --- |
| BMC `/etc/ac922-fan-quiet/boot_guard.py` | `bmc/boot_guard.py` | `1e4af1408e465036ac2523f6e8fa83c1e4177e9c446c7ab79aec7054b8ceb45c` |
| BMC `/etc/ac922-fan-quiet/boot_pin.sh` | `bmc/boot_pin.sh` | `cd48345074913ff0ac99831f2dd4604b569de931c546409cba0d438f37736b17` |
| BMC `/etc/ac922-fan-quiet/boot_guard_restore.sh` | `bmc/restore.sh` | `e12a13de97d5cb21ed5044bd264d6df5290a0035fd4ef5e640b85cf9fc2280cf` |
| BMC `/etc/systemd/system/ac922-boot-guard.service` | `bmc/ac922-boot-guard.service` | `4e89005a0ff5c448a4a9044232329379df1321014494fb6e9fc3fe623a20c9ad` |
| Host `/usr/local/sbin/ac922-memory-guard.py` | `host/memory_guard.py` | `e919e73ee18a10d9893244dcbb22c4b6855fad3ac59855ba71efed8d9a07bf03` |
| Host `/usr/local/bin/with-ac922-cuda525` | `host/with-ac922-cuda525` | `8622b9a4cccf27375e886a3b7a023b4b9f81f3c2be886079451b8e044f11cedc` |
| Host `/etc/systemd/system/nvidia-startup.service` | `host/nvidia-startup.service` | `d4a71f0fa2b473ba75e48ed224fac872fc435185c648bbec1832000e641810f8` |
| Host `/etc/systemd/system/nvidia-startup.timer` | `host/nvidia-startup.timer` | `4ca2b1752f21985336113414aaf58655ae0243c472f645467a8e225f6c2e826d` |
| Host `/usr/local/bin/nvidia-smi` | `host/nvidia-smi.wrapper` | `426321f8d5b06495f5f6dd53512f823a068e4e2ae30cffd406aa8ff4e6eb33ee` |
| Host `/usr/local/sbin/ac922-sata32-rollback` | `kernel/ac922-sata32-rollback.reference.sh` | `1a72b3d287661b8d11d641bf8062cdaac62bbb3fe12a891dbd9d6c4dbba5bd77` |
| AHCI source change | `kernel/88se9235-dma-mask32.patch` | `4b4d2eb920ab3c1b9e8e1bc3b04fe178624ff36625c8e7ba12eb218a04a0111f` |
| Debian kernel config | `kernel/config-6.1.0-53-powerpc64le` | `9d42a31bfc2ae1f46c67f0922626a4caec6aeff5701642a91fff18d4d1c71ec0` |
| Sanitized boot entry template | `kernel/09_ac922_sata32.template` | `0c5edf7ee64d0239838f84a2444981a39656ea41b2a8aecd8ce40ba0d08358c3` |

`kernel/09_ac922_sata32.template` deliberately replaces the live boot and root
filesystem UUIDs with placeholders. The live `/etc/grub.d/09_ac922_sata32`
script is not published verbatim.

## Experimental or diagnostic sources

- `experiments/fan_trial_bmc.py` is the temporary supervised RPM-stage trial.
  It directly writes fan targets and stops only the factory fan-control
  writer. It is not a service and must not be installed at boot.
- `proxy/` is an opt-in attempt that did not pass real node connectivity on
  the AC922. The BMC and host proxy services remain disabled. Do not interpret
  successful config parsing as proof of a working subscription.
- `kernel/ac922-sata32-rollback.reference.sh` is copied from the host's root
  rollback helper. Only its `--check` path was exercised; `--reboot` and its
  destructive cleanup were not. Review local boot entries and backups before
  adapting it.
- The workspace retains private recovery tarballs, logs, built modules, model
  files, and vendor driver archives. They are intentionally outside this
  repository and are not needed to understand the published source.
