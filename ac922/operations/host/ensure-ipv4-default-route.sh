#!/bin/sh
# Rootless periodic guard when sudoers grants only `ip route`.
# Configure AC922_INTERFACE and AC922_GATEWAY for the target machine.
set -eu

interface=${AC922_INTERFACE:-}
gateway=${AC922_GATEWAY:-}
if [ -z "$interface" ] || [ -z "$gateway" ]; then
    echo 'Set AC922_INTERFACE and AC922_GATEWAY before use.' >&2
    exit 2
fi
if /usr/bin/ip -4 route show default dev "$interface" | \
    /usr/bin/grep -Fq "via $gateway "; then
    exit 0
fi
exec sudo -n /usr/bin/ip -4 route replace default via "$gateway" \
    dev "$interface" metric 100
