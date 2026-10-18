#!/bin/sh
# Pin a quiet fan target while OCC sensors initialize at chassis power-on.
set -eu

hwmon=
for dir in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$dir/name" 2>/dev/null)" = max31785a ] || continue
    hwmon=$dir
    break
done
[ -n "$hwmon" ] || exit 1
for fan in 1 3 4; do
    [ -r "$hwmon/fan${fan}_target" ] && [ -w "$hwmon/fan${fan}_target" ] || exit 1
done

ambient_hwmon=
for dir in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$dir/name" 2>/dev/null)" = iio_hwmon_dps310 ] || continue
    ambient_hwmon=$dir
    break
done
[ -n "$ambient_hwmon" ] || exit 1
[ -r "$ambient_hwmon/temp1_input" ] || exit 1

for service in phosphor-fan-control-init@0.service phosphor-fan-control@0.service; do
    state=$(systemctl show "$service" -p LoadState 2>/dev/null || true)
    if [ "$state" != 'LoadState=masked' ]; then
        if ! systemctl mask --runtime --now "$service"; then
            for fan in 1 3 4; do echo 4000 > "$hwmon/fan${fan}_target"; done
            exit 1
        fi
    fi
done

start=$(cut -d. -f1 /proc/uptime)
deadline=$((start + 180))
while [ "$(cut -d. -f1 /proc/uptime)" -lt "$deadline" ]; do
    ambient_mc=$(cat "$ambient_hwmon/temp1_input")
    [ "$ambient_mc" -ge 5000 ] && [ "$ambient_mc" -lt 45000 ] || exit 1
    for fan in 1 3 4; do
        echo 3100 > "$hwmon/fan${fan}_target"
        [ "$(cat "$hwmon/fan${fan}_target")" -eq 3100 ] || exit 1
    done

    now=$(cut -d. -f1 /proc/uptime)
    if [ $((now - start)) -ge 15 ]; then
        for fan in 1 3 4; do
            attempt=0
            rpm=0
            while [ "$attempt" -lt 3 ]; do
                rpm=$(cat "$hwmon/fan${fan}_input")
                [ "$rpm" -ge 2650 ] && break
                attempt=$((attempt + 1))
                sleep 1
            done
            [ "$rpm" -ge 2650 ] || exit 1
            [ "$(cat "$hwmon/fan${fan}_alarm")" -eq 0 ] || exit 1
            [ "$(cat "$hwmon/fan${fan}_fault")" -eq 0 ] || exit 1
        done
    fi

    occ0=$(busctl get-property org.open_power.OCC.Control \
        /org/open_power/control/occ0 org.open_power.OCC.Status OccActive 2>/dev/null || true)
    occ1=$(busctl get-property org.open_power.OCC.Control \
        /org/open_power/control/occ1 org.open_power.OCC.Status OccActive 2>/dev/null || true)
    if [ "$occ0" = 'b true' ] && [ "$occ1" = 'b true' ]; then
        exit 0
    fi
    sleep 1
done

exit 1
