#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo 'Run as root.' >&2
    exit 1
fi

backup=/root/ac922-sata32-backup/nvidia-startup.before525.service
test -s "$backup"
test -s /boot/vmlinux-8.8.8-ac922-gw
test -s /boot/initrd.img-8.8.8-ac922-gw

if [ "${1:-}" = '--check' ]; then
    echo 'Factory boot files and service backup are present.'
    exit 0
fi
if [ "${1:-}" != '--reboot' ]; then
    echo 'Usage: ac922-sata32-rollback --check|--reboot' >&2
    exit 2
fi

systemctl disable --now nvidia-startup.timer || true
rm -f /etc/grub.d/09_ac922_sata32
update-grub
grep -m 1 '^menuentry ' /boot/grub/grub.cfg | grep -q '8.8.8-ac922-gw'

install -m 644 "$backup" /etc/systemd/system/nvidia-startup.service
rm -f /etc/systemd/system/nvidia-startup.timer
rm -f /etc/modprobe.d/ac922-disable-nouveau.conf
rm -f /usr/local/bin/nvidia-smi /usr/local/bin/with-ac922-cuda525
rm -f /boot/initrd.img-6.1.0-53-ac922-sata32
rm -f /lib/modules/6.1.0-53-powerpc64le/updates/ac922-sata32/ahci.ko
rm -f /lib/modules/6.1.0-53-powerpc64le/updates/ac922-sata32/nvidia.ko
rm -f /lib/modules/6.1.0-53-powerpc64le/updates/ac922-sata32/nvidia-uvm.ko
rmdir /lib/modules/6.1.0-53-powerpc64le/updates/ac922-sata32 2>/dev/null || true
depmod -a 6.1.0-53-powerpc64le

rm -f /opt/ac922-nvidia525/bin/nvidia-modprobe /opt/ac922-nvidia525/bin/nvidia-persistenced
rm -f /opt/ac922-nvidia525/sbin/nvidia-smi
rm -f /opt/ac922-nvidia525/lib/libnvidia-ml.so.525.147.05 /opt/ac922-nvidia525/lib/libnvidia-ml.so.1
rm -f /opt/ac922-nvidia525/lib/libcuda.so.525.147.05 /opt/ac922-nvidia525/lib/libcuda.so.1
rm -f /opt/ac922-nvidia525/lib/libnvidia-cfg.so.525.147.05 /opt/ac922-nvidia525/lib/libnvidia-cfg.so.1
rm -f /opt/ac922-nvidia525/lib/libnvidia-ptxjitcompiler.so.525.147.05 /opt/ac922-nvidia525/lib/libnvidia-ptxjitcompiler.so.1
rm -f /opt/ac922-nvidia525/lib/libnvidia-nvvm.so.525.147.05 /opt/ac922-nvidia525/lib/libnvidia-nvvm.so.1
rmdir /opt/ac922-nvidia525/bin /opt/ac922-nvidia525/sbin /opt/ac922-nvidia525/lib /opt/ac922-nvidia525 2>/dev/null || true

rm -f /opt/ac922-cuda-compat-12-4/lib/libcuda.so*
rm -f /opt/ac922-cuda-compat-12-4/lib/libcudadebugger.so*
rm -f /opt/ac922-cuda-compat-12-4/lib/libnvidia-nvvm.so*
rm -f /opt/ac922-cuda-compat-12-4/lib/libnvidia-ptxjitcompiler.so*
rmdir /opt/ac922-cuda-compat-12-4/lib /opt/ac922-cuda-compat-12-4 2>/dev/null || true

systemctl daemon-reload
systemctl enable nvidia-startup.service
sync
systemctl reboot
