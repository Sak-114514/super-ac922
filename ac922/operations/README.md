# AC922 operations source kit

This directory publishes the small pieces of AC922-specific software that
were previously scattered across deployment notes and recovery folders. The
POWER9/ppc64le build instructions, wheel manifests, and release artifacts are
already documented in [`../README.md`](../README.md). This kit adds the fan,
memory, GPU-startup, network-helper, and SATA32 source needed to understand or
rebuild the rest of the setup.

The root repository is Apache-2.0. The files in `kernel/` derive from Debian
Linux source marked GPL-2.0-or-later; see [`kernel/COPYING`](kernel/COPYING).
Third-party programs and firmware keep their upstream licenses.

## Contents

| Area | Source | Scope |
| --- | --- | --- |
| BMC fan guard | [`bmc/`](bmc/) | Python 2.7 guard and systemd unit for Witherspoon/AC922 GTW fan control |
| Host memory guard | [`host/memory_guard.py`](host/memory_guard.py) | Keeps phantom/high-address blocks offline while preserving NVIDIA-managed V100 HBM |
| NVIDIA 525 startup | [`host/`](host/) | Unit/timer and environment wrappers for the separately installed NVIDIA 525 files |
| SATA32 kernel fix | [`kernel/`](kernel/) | One-source-file AHCI patch, boot-entry template, and machine-bound rollback reference |
| Selective proxy | [`proxy/`](proxy/) | Local-only, opt-in wrapper and subscription renderer; current subscription did not pass connection checks |
| Fan trial | [`experiments/fan_trial_bmc.py`](experiments/fan_trial_bmc.py) | Short, supervised low-load RPM experiment; not an unattended controller |

## AC922 requirements and boundaries

- The BMC guard targets the IBM Witherspoon/AC922 GTW topology, MAX31785 tach
  channels, and OpenBMC Python 2.7/dbus environment. The deployed quiet profile
  masks the original fan-monitor because it shut this host down at 3000 RPM;
  this is a hardware-specific safety tradeoff, not a general recommendation.
  Verify the cooling profile, service writers, independent watchdog, and
  temperature/tach paths before installation. The files are not a portable
  drop-in for other boards.
- `memory_guard.py` relies on POWER9 device-tree memory-region phandles and
  six V100 NUMA/HBM ranges. It deliberately stops when that mapping is
  incomplete. Do not generalize its `32 GiB` CPU-memory boundary or GPU HBM
  assumptions to another AC922 configuration without reviewing the memory
  topology.
- NVIDIA startup files refer to kernel `6.1.0-53-powerpc64le` and binaries under
  `/opt/ac922-nvidia525`. They do not include the NVIDIA driver or CUDA
  compatibility libraries. Obtain those from NVIDIA under their license.
- The SATA32 patch is a candidate for the Marvell 88SE9235/AC922 combination.
  It is not a general AHCI fix. The rollback reference contains destructive
  uninstall and reboot operations; read every command and verify local backup
  paths before adapting it. Its `--reboot` path was not exercised as part of
  this publication.
- The proxy example listens on loopback only and does not alter system proxy,
  TUN, firewall, or routing configuration. Put a current subscription in a
  mode-0600 local file. Never commit subscription URLs, node credentials,
  generated configs, or downloaded client profiles. The provider subscription
  used for the original trial failed actual connection tests, so the proxy
  service remains opt-in and is not presented as a working node list.

## Reproduction outline

### 1. Build the userspace stack

Follow [`../README.md`](../README.md) and its scripts. The current branch
records the exact upstream source revision, compiler/toolchain assumptions,
dependency-cache hashes, wheel hashes, license notices, and separate ABI
lanes. Install only one compatible stack at a time; the experimental 1Cat
wheel and standalone PyTorch 2.10 wheel are not a matched pair.

### 2. Rebuild the AHCI module

Start from the Debian `linux-source-6.1` package version recorded in the
machine's release notes, use the matching kernel config and headers, apply
`kernel/88se9235-dma-mask32.patch`, and rebuild only after confirming the
PCI ID and current AHCI source context. Keep the original kernel, initrd,
NVIDIA modules, and boot entry as rollback artifacts. Compiled kernel objects,
initrds, NVIDIA modules, firmware blobs, and local root/boot UUIDs are not
committed here.

### 3. Install guards manually

Copy the host memory guard and unit to the target only after checking the
device-tree mappings and CPU NUMA memory total. On the BMC, use its Python 2.7
runtime and verify the fan objects, tach channels, original monitor
compatibility, and hardware watchdog before enabling
`bmc/ac922-boot-guard.service`. The published night-3000 policy leaves the
stock monitor masked and retains the hardware watchdog; operators who require
the stock monitor must first retune and validate its RPM thresholds.

### 4. Use the proxy only for selected commands

The proxy files are optional examples. Render credentials locally, validate
the sing-box config, start the user service, pass the health check, then use
`with-ac922-proxy.sh COMMAND ...`. A failed health check exits before starting
the requested command. Do not globally export proxy variables or run the
wrapper until a node from your own subscription has passed an actual request.

## Public-source hygiene

The release intentionally contains source, documentation, and hashes only.
Exclude model weights, subscription files, system logs, BMC backups, disk
images, initrds, built kernel modules, proprietary NVIDIA/CUDA files, local
hardware identifiers, and credential-bearing generated configuration.

The repository's `ac922/operations/PROVENANCE.md` records which source files
match the currently deployed host/BMC helpers and which files are templates or
bench-only. Files listed as references are not implied to be safe, portable, or
validated on other revisions.
