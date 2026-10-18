#!/bin/sh
set -eu
hwmon=
for d in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$d/name" 2>/dev/null)" = max31785a ] || continue
    hwmon=$d
    break
done
if [ -z "$hwmon" ]; then
    echo 'AC922 4000 RPM fallback unavailable: max31785a sysfs missing' >&2
    exit 1
fi
for n in 1 3 4; do
    [ -r "$hwmon/fan${n}_target" ] && [ -w "$hwmon/fan${n}_target" ] || exit 1
done

status=0
for service in phosphor-fan-control-init@0.service phosphor-fan-control@0.service; do
    state=$(systemctl show "$service" -p LoadState 2>/dev/null || true)
    if [ "$state" != 'LoadState=masked' ]; then
        if ! systemctl mask --runtime --now "$service"; then
            echo "Failed to mask $service; still trying 4000 RPM" >&2
            status=1
        fi
    fi
done
for n in 1 3 4; do
    f="$hwmon/fan${n}_target"
    if ! echo 4000 > "$f"; then
        status=1
    fi
    readback=$(cat "$f" 2>/dev/null || echo unavailable)
    [ "$readback" = 4000 ] || status=1
done
exit "$status"
