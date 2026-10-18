#!/usr/bin/env python
"""AC922 BMC cooling guard; a failed/stopped unit selects 4000 RPM."""
from __future__ import print_function

import json
import datetime
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time

import dbus


SENSOR_IFACE = 'xyz.openbmc_project.Sensor.Value'
PROPERTIES = 'org.freedesktop.DBus.Properties'
FAN_IDS = (1, 3, 4)
SECONDARY_IDS = (5, 7, 8)
MIN_RPM = 2650
CPU_TEMP_RE = re.compile(r'p[01]_core\d+_temp$')
GPU_TEMP_RE = re.compile(r'gpu[0-5]_core_temp$')
state = {'target': 3000, 'last_ok': 0.0, 'running': True}
mutex = threading.Lock()


def uptime():
    with open('/proc/uptime') as stream:
        return float(stream.read().split()[0])


def log(event, **details):
    details['event'] = event
    details['uptime_s'] = uptime()
    details['time_utc'] = datetime.datetime.utcnow().isoformat() + 'Z'
    print(json.dumps(details, sort_keys=True))
    sys.stdout.flush()


def notify():
    path = os.environ.get('NOTIFY_SOCKET')
    if not path:
        return
    if path.startswith('@'):
        path = '\0' + path[1:]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.sendto('WATCHDOG=1', path)
    finally:
        sock.close()


def controller_path():
    found = []
    for entry in os.listdir('/sys/class/hwmon'):
        path = '/sys/class/hwmon/' + entry
        try:
            with open(path + '/name') as stream:
                if stream.read().strip() == 'max31785a':
                    found.append(path)
        except IOError:
            pass
    if len(found) != 1:
        raise RuntimeError('expected one max31785a, found %d' % len(found))
    return found[0]


def read_int(path):
    with open(path) as stream:
        return int(stream.read().strip())


def air_cooled_temperature_maximum(values):
    """Return the maximum temperature allowed to select the quiet tiers.

    CPU and GPU packages are water-cooled on the GTW chassis.  They remain in
    the all-sensor 55/60C safety checks below, but their package/core readings
    must not promote the air-moving fans from 3000 to 3500 RPM.
    """
    candidates = [(name, value) for name, value in values.items()
                  if not CPU_TEMP_RE.match(name) and
                  not GPU_TEMP_RE.match(name)]
    if not candidates:
        raise RuntimeError('no air-cooled temperature sensors available')
    return max(value for name, value in candidates)


def temperature_names_at(values, maximum):
    return sorted(name for name, value in values.items()
                  if value == maximum)


def set_targets(hwmon, target):
    for fan in FAN_IDS:
        path = '%s/fan%d_target' % (hwmon, fan)
        with open(path, 'w') as stream:
            stream.write('%d\n' % target)
    actual = [read_int('%s/fan%d_target' % (hwmon, fan)) for fan in FAN_IDS]
    if actual != [target] * 3:
        raise RuntimeError('target readback mismatch: %r' % actual)


def emergency(hwmon, reason):
    mask_errors = []
    for service in ('phosphor-fan-control-init@0.service',
                    'phosphor-fan-control@0.service'):
        try:
            process = subprocess.Popen(['systemctl', 'mask', '--runtime', '--now', service],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            deadline = uptime() + 3
            while process.poll() is None and uptime() < deadline:
                time.sleep(0.1)
            if process.poll() is None:
                process.kill()
                process.communicate()
                raise RuntimeError('mask timed out')
            output, errors = process.communicate()
            if process.returncode:
                raise RuntimeError('mask failed: %s' % errors.strip())
        except Exception as exc:
            mask_errors.append('%s: %s' % (service, exc))
    try:
        set_targets(hwmon, 4000)
        log('safe_speed_fallback', reason=reason, target=4000,
            mask_errors=mask_errors)
    except Exception as exc:
        log('restore_failed', reason=reason, error=str(exc))
    os._exit(1)


def writers_masked():
    # /etc masks have priority over /run units and cannot be activated by a
    # transient systemd job. Avoid per-second systemctl/D-Bus queries: on this
    # old BMC those occasionally time out while the manager is reloading.
    for service in ('phosphor-fan-control-init@0.service',
                    'phosphor-fan-control@0.service'):
        path = '/etc/systemd/system/' + service
        if not os.path.islink(path) or os.readlink(path) != '/dev/null':
            return False
    return True


def pin_loop(hwmon):
    mismatches = 0
    while True:
        with mutex:
            target = state['target']
            age = uptime() - state['last_ok']
        if age > 35:
            emergency(hwmon, 'temperature monitor stalled for %.1fs' % age)
        if not writers_masked():
            emergency(hwmon, 'persistent factory-writer mask disappeared')
        try:
            actual = [read_int('%s/fan%d_target' % (hwmon, fan)) for fan in FAN_IDS]
            if actual != [target] * 3:
                set_targets(hwmon, target)
                log('target_reapplied', target=target, old_targets=actual)
            mismatches = 0
        except Exception as exc:
            mismatches += 1
            if mismatches >= 3:
                emergency(hwmon, 'three target write/read failures: %s' % exc)
        time.sleep(1)


def temperature_values(bus, expected):
    mapper = dbus.Interface(bus.get_object('xyz.openbmc_project.ObjectMapper',
                                            '/xyz/openbmc_project/object_mapper'),
                            'xyz.openbmc_project.ObjectMapper')
    tree = mapper.GetSubTree('/xyz/openbmc_project/sensors/temperature', 0,
                             [SENSOR_IFACE])
    values = {}
    for path, owners in tree.items():
        name = str(path).split('/')[-1]
        wanted = (re.match(r'p[01]_core\d+_temp$', name) or
                  re.match(r'gpu[0-5]_core_temp$', name) or
                  name.startswith('dimm') or
                  re.match(r'p[01]_vddr?_temp$', name) or
                  name in ('pcie', 'ambient'))
        if not wanted:
            continue
        service = str(list(owners.keys())[0])
        props = dbus.Interface(bus.get_object(service, path), PROPERTIES)
        value = float(props.Get(SENSOR_IFACE, 'Value')) / 1000.0
        if math.isnan(value) or math.isinf(value) or value < 5 or value > 105:
            raise RuntimeError('implausible temperature %s=%.1f' % (name, value))
        values[name] = value
    required = set(['gpu%d_core_temp' % i for i in range(6)] +
                   ['dimm1_temp', 'dimm3_temp', 'p0_vdd_temp', 'p1_vdd_temp',
                    'p0_vddr_temp', 'p1_vddr_temp', 'pcie', 'ambient'])
    if required - set(values):
        raise RuntimeError('required temperature sensors missing')
    if sum(name.startswith('p0_core') for name in values) < 20 or \
            sum(name.startswith('p1_core') for name in values) < 20:
        raise RuntimeError('CPU temperature coverage below 20 cores per CPU')
    if expected is not None and not expected.issubset(set(values)):
        raise RuntimeError('previously present temperature sensor disappeared')
    return values


def tach_values(hwmon, bus):
    result = []
    mapper = dbus.Interface(bus.get_object('xyz.openbmc_project.ObjectMapper',
                                            '/xyz/openbmc_project/object_mapper'),
                            'xyz.openbmc_project.ObjectMapper')
    for primary, secondary in zip(FAN_IDS, SECONDARY_IDS):
        if read_int('%s/fan%d_alarm' % (hwmon, primary)) or \
                read_int('%s/fan%d_fault' % (hwmon, primary)):
            raise RuntimeError('fan%d alarm/fault' % primary)
        for suffix in ('alarm', 'fault'):
            optional = '%s/fan%d_%s' % (hwmon, secondary, suffix)
            if os.path.exists(optional) and read_int(optional):
                raise RuntimeError('fan%d %s' % (secondary, suffix))
        path = '/xyz/openbmc_project/sensors/fan_tach/fan%d_0' % (primary - 1)
        owner = mapper.GetObject(path, ['xyz.openbmc_project.State.Decorator.OperationalStatus'])
        service = str(list(owner.keys())[0])
        props = dbus.Interface(bus.get_object(service, path), PROPERTIES)
        if not bool(props.Get('xyz.openbmc_project.State.Decorator.OperationalStatus',
                              'Functional')):
            raise RuntimeError('fan%d marked nonfunctional' % primary)
        readings = []
        for fan in (primary, secondary):
            value = 0
            for attempt in range(3):
                value = read_int('%s/fan%d_input' % (hwmon, fan))
                if value >= MIN_RPM:
                    break
                if attempt < 2:
                    time.sleep(1)
            if value < MIN_RPM:
                raise RuntimeError('fan%d tach stayed at %d RPM' % (fan, value))
            readings.append(value)
        result.append(readings)
    return result


def startup_tach_values(hwmon, bus):
    # The MAX31785 secondary tach registers can briefly return zero after
    # boot_pin changes the target. Keep the startup pin and watchdog active
    # while retrying; a persistent failure still enters the 4000 RPM fallback.
    deadline = uptime() + 20
    while True:
        try:
            return tach_values(hwmon, bus)
        except Exception as exc:
            if uptime() >= deadline:
                raise
            log('startup_tach_retry', error=str(exc))
            time.sleep(1)


def startup_temperature_values(bus, expected):
    # OCC temperature objects can be republished several seconds after boot.
    # Keep the startup pin and watchdog active while retrying; a persistent
    # failure still enters the 4000 RPM fallback.
    deadline = uptime() + 20
    while True:
        try:
            return temperature_values(bus, expected)
        except Exception as exc:
            if uptime() >= deadline:
                raise
            log('startup_temperature_retry', error=str(exc))
            time.sleep(1)


def main():
    hwmon = controller_path()
    bus = dbus.SystemBus()
    if not writers_masked():
        raise RuntimeError('factory fan writers lack persistent masks')
    expected = None
    # Validate before any new write; fan target is already 3100 on entry.
    temps = startup_temperature_values(bus, None)
    tach = startup_tach_values(hwmon, bus)
    maximum = max(temps.values())
    air_maximum = air_cooled_temperature_maximum(temps)
    if maximum >= 60:
        emergency(hwmon, 'startup temperature >=60C')
    target = 4000 if maximum >= 55 else 3500 if air_maximum >= 45 else 3000
    set_targets(hwmon, target)
    state['target'] = target
    state['last_ok'] = uptime()
    expected = set(temps)
    log('guard_started', target=target, max_temp_c=maximum,
        max_temp_sensors=temperature_names_at(temps, maximum),
        air_control_max_temp_c=air_maximum,
        air_control_max_sensors=temperature_names_at(
            dict((name, value) for name, value in temps.items()
                 if not CPU_TEMP_RE.match(name) and
                 not GPU_TEMP_RE.match(name)), air_maximum),
        temperature_count=len(temps), tach_rpm=tach)
    writer = threading.Thread(target=pin_loop, args=(hwmon,))
    writer.daemon = True
    writer.start()
    cool_since = None
    intermediate_cool_since = None
    while True:
        notify()
        if not writer.isAlive():
            emergency(hwmon, 'pinning thread exited')
        errors = []
        for attempt in range(3):
            try:
                temps = temperature_values(bus, expected)
                tach = tach_values(hwmon, bus)
                break
            except Exception as exc:
                errors.append(str(exc))
                notify()
                if attempt < 2:
                    time.sleep(1)
        else:
            emergency(hwmon, 'three sensor/tach failures: %r' % errors)
        maximum = max(temps.values())
        air_maximum = air_cooled_temperature_maximum(temps)
        if maximum >= 60:
            emergency(hwmon, 'temperature %.1fC >=60C' % maximum)
        with mutex:
            old = state['target']
            if maximum >= 55:
                state['target'] = 4000
                cool_since = None
                intermediate_cool_since = None
            elif state['target'] == 4000:
                if maximum <= 52:
                    if intermediate_cool_since is None:
                        intermediate_cool_since = uptime()
                    elif uptime() - intermediate_cool_since >= 90:
                        state['target'] = 3500
                        intermediate_cool_since = None
                        cool_since = None
                else:
                    intermediate_cool_since = None
            elif air_maximum >= 45:
                state['target'] = 3500
                cool_since = None
                intermediate_cool_since = None
            elif air_maximum <= 42 and state['target'] == 3500:
                if cool_since is None:
                    cool_since = uptime()
                elif uptime() - cool_since >= 90:
                    state['target'] = 3000
                    cool_since = None
            else:
                cool_since = None
                intermediate_cool_since = None
            state['last_ok'] = uptime()
            new = state['target']
        if new != old:
            set_targets(hwmon, new)
            log('tier_change', target=new, previous_target=old,
                max_temp_c=maximum,
                max_temp_sensors=temperature_names_at(temps, maximum),
                air_control_max_temp_c=air_maximum,
                air_control_max_sensors=temperature_names_at(
                    dict((name, value) for name, value in temps.items()
                         if not CPU_TEMP_RE.match(name) and
                         not GPU_TEMP_RE.match(name)), air_maximum),
                tach_rpm=tach)
        time.sleep(4)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        try:
            emergency(controller_path(), 'controller exception: %s' % exc)
        except Exception:
            sys.exit(1)
