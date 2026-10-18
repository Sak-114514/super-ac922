# AC922 Debian host guards and NVIDIA startup

These files are separate from the vLLM build scripts. They support the
specific six-V100 AC922 host and must be reviewed against the running kernel,
device tree, NUMA map, and installed NVIDIA directories before installation.

## Memory guard

`memory_guard.py` is the deployed `ac922-memory-guard.py` source. It keeps
phantom/high-address memory offline while allowing the six firmware-identified
V100 HBM ranges to be onlined by the NVIDIA driver. The policy uses device-tree
phandles and CPU NUMA nodes 0 and 8. If the GPU map changes or is incomplete,
the program fails rather than guessing. It is not a general memory cap for
other POWER9 or GPU layouts.

`ac922-memory-guard.service` starts the loop before `multi-user.target`,
restarts on failure, and sends systemd watchdog notifications. The 32-GiB
boundary is specific to the AC922's installed 32-GiB CPU DIMM configuration.

## NVIDIA startup and launch wrappers

`nvidia-startup.service` and `.timer` initialize the separately installed
NVIDIA 525 kernel/user stack only when kernel `6.1.0-53-powerpc64le` is active
and the memory guard is healthy. `with-ac922-cuda525` selects the matching
CUDA compatibility libraries for an application. `nvidia-smi.wrapper` selects
the management tool by running kernel.

The NVIDIA driver, CUDA toolkit, cuDNN/CUPTI libraries, and compatibility
archives are not included. Download them directly from NVIDIA and follow their
licenses. Do not place vendor binaries in Git or mix the 525 and 550 user-space
libraries without reproducing the matching kernel/ABI combination.

`ensure-ipv4-default-route.sh` is an optional cron helper for a host with an
existing narrowly scoped `sudo -n /usr/bin/ip -4 route` permission. Set
`AC922_INTERFACE` and `AC922_GATEWAY` for the site. It adds a route only when
the expected DHCP gateway is missing; it does not change interface addresses
or restart networking. A cron entry can supply those values explicitly:

```cron
@reboot AC922_INTERFACE=eth0 AC922_GATEWAY=192.0.2.1 /usr/local/sbin/ensure-ipv4-default-route.sh
* * * * * AC922_INTERFACE=eth0 AC922_GATEWAY=192.0.2.1 /usr/local/sbin/ensure-ipv4-default-route.sh
```
