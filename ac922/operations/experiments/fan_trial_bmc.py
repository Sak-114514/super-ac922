#!/usr/bin/env python
"""Temporary AC922 low-load fan trial; Python 2.7 for the installed BMC."""

from __future__ import print_function

import json
import os
import re
import subprocess
import sys
import time
import traceback

import dbus


HWMON = None
FAN_PATH = '/xyz/openbmc_project/sensors/fan_tach/'
SENSOR_ROOT = '/xyz/openbmc_project/sensors/temperature'
SENSOR_IFACE = 'xyz.openbmc_project.Sensor.Value'
FAN_IFACE = 'xyz.openbmc_project.Control.FanSpeed'
PROP_IFACE = 'org.freedesktop.DBus.Properties'
LOG = '/tmp/ac922-fan-trial.jsonl'
STAGES = [(3300, 180), (3100, 180), (2900, 180)]
SERVICE_NAMES = ['phosphor-fan-control@0.service',
                 'phosphor-fan-monitor@0.service',
                 'witherspoon-fan-watchdog.service']
BUS = dbus.SystemBus()


def record(kind, **data):
    item = {'time_epoch': time.time(), 'kind': kind}
    item.update(data)
    line = json.dumps(item, sort_keys=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')
    print(line)
    sys.stdout.flush()


def sysfs_int(name):
    with open(os.path.join(hwmon_path(), name)) as f:
        return int(f.read().strip())


def hwmon_path():
    global HWMON
    if HWMON is not None:
        return HWMON
    matches = []
    for entry in os.listdir('/sys/class/hwmon'):
        path = os.path.join('/sys/class/hwmon', entry)
        try:
            with open(os.path.join(path, 'name')) as stream:
                if stream.read().strip() == 'max31785a':
                    matches.append(path)
        except IOError:
            continue
    if len(matches) != 1:
        raise RuntimeError('expected one max31785a controller, found %r' % matches)
    HWMON = matches[0]
    return HWMON


def get_temp_objects():
    obj = BUS.get_object('xyz.openbmc_project.ObjectMapper',
                         '/xyz/openbmc_project/object_mapper')
    mapper = dbus.Interface(obj, 'xyz.openbmc_project.ObjectMapper')
    subtree = mapper.GetSubTree(SENSOR_ROOT, 0, [SENSOR_IFACE])
    selected = {}
    for path, owners in subtree.items():
        name = str(path).split('/')[-1]
        if (re.match(r'p[01]_core\d+_temp$', name) or
                re.match(r'gpu[0-5]_core_temp$', name) or
                name.startswith('dimm') or
                re.match(r'p[01]_vddr?_temp$', name) or
                name in ('pcie', 'ambient')):
            service = str(list(owners.keys())[0])
            selected[name] = dbus.Interface(BUS.get_object(service, path),
                                            PROP_IFACE)
    return selected


def temperatures(objects):
    vals = {}
    for name, props in objects.items():
        vals[name] = float(props.Get(SENSOR_IFACE, 'Value')) / 1000.0
    counts = {
        'cpu0': len([n for n in vals if n.startswith('p0_core')]),
        'cpu1': len([n for n in vals if n.startswith('p1_core')]),
        'gpu': len([n for n in vals if re.match(r'gpu[0-5]_core_temp$', n)]),
        'dimm': len([n for n in vals if n.startswith('dimm')]),
        'vdd': len([n for n in vals if re.match(r'p[01]_vddr?_temp$', n)]),
        'pcie': int('pcie' in vals),
        'ambient': int('ambient' in vals),
    }
    if (counts['cpu0'] < 20 or counts['cpu1'] < 20 or counts['gpu'] != 6 or
            counts['dimm'] < 2 or counts['vdd'] < 2 or
            not counts['pcie'] or not counts['ambient']):
        raise RuntimeError('mandatory temperature coverage missing: %r' % counts)
    return vals


def fan_state():
    fans = {}
    for label, a, b in [('fan0', 1, 5), ('fan2', 3, 7), ('fan3', 4, 8)]:
        fans[label] = {
            'target': sysfs_int('fan%d_target' % a),
            'primary_rpm': sysfs_int('fan%d_input' % a),
            'secondary_rpm': sysfs_int('fan%d_input' % b),
            'alarm': sysfs_int('fan%d_alarm' % a),
            'fault': sysfs_int('fan%d_fault' % a),
        }
    # The MAX31785 occasionally exposes a zero tach sample immediately
    # after a target transition. Confirm it promptly; persistent zero still
    # fails validation and restores factory control.
    if any(f['secondary_rpm'] == 0 for f in fans.values()):
        for attempt in range(2):
            time.sleep(1)
            for label, b in [('fan0', 5), ('fan2', 7), ('fan3', 8)]:
                if fans[label]['secondary_rpm'] == 0:
                    fans[label]['secondary_rpm'] = sysfs_int('fan%d_input' % b)
            record('tach_zero_retry', attempt=attempt + 1,
                   secondary={name: f['secondary_rpm'] for name, f in fans.items()})
            if all(f['secondary_rpm'] > 0 for f in fans.values()):
                break
    return fans


def services():
    proc = subprocess.Popen(['systemctl', 'is-active'] + SERVICE_NAMES,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, _ = proc.communicate()
    return dict(zip(SERVICE_NAMES, output.decode('ascii').splitlines()))


def sample(objects):
    return {'temps_c': temperatures(objects), 'fans': fan_state(),
            'services': services()}


def set_targets(target):
    mapper_obj = BUS.get_object('xyz.openbmc_project.ObjectMapper',
                                '/xyz/openbmc_project/object_mapper')
    mapper = dbus.Interface(mapper_obj, 'xyz.openbmc_project.ObjectMapper')
    for name, idx in [('fan0_0', 1), ('fan2_0', 3), ('fan3_0', 4)]:
        path = FAN_PATH + name
        try:
            owners = mapper.GetObject(path, [FAN_IFACE])
            service = str(list(owners.keys())[0])
            obj = BUS.get_object(service, path)
            dbus.Interface(obj, PROP_IFACE).Set(FAN_IFACE, 'Target',
                                                dbus.UInt64(target))
        except Exception:
            # A recovery path if D-Bus is unavailable. The same hwmon
            # target backs the writable FanSpeed.Target property.
            with open(os.path.join(hwmon_path(), 'fan%d_target' % idx), 'w') as f:
                f.write(str(target) + '\n')
    actual = [sysfs_int('fan%d_target' % i) for i in (1, 3, 4)]
    if actual != [target] * 3:
        raise RuntimeError('target readback mismatch: %r' % actual)
    record('target_set', target=target, readback=actual)


def restore():
    for attempt in range(3):
        try:
            set_targets(4300)
            subprocess.check_call(['systemctl', 'start', SERVICE_NAMES[0]])
            state = services()
            if any(v != 'active' for v in state.values()):
                raise RuntimeError('service restoration failed: %r' % state)
            record('restored', attempt=attempt + 1)
            return True
        except Exception as exc:
            record('restore_error', attempt=attempt + 1, error=str(exc))
            time.sleep(1)
    return False


def validate(state, baseline, target, control_active):
    expected = 'active' if control_active else 'inactive'
    if state['services'].get(SERVICE_NAMES[0]) != expected:
        raise RuntimeError('unexpected fan control state: %r' % state['services'])
    if any(state['services'].get(n) != 'active' for n in SERVICE_NAMES[1:]):
        raise RuntimeError('fan monitor/watchdog became inactive: %r' %
                           state['services'])
    for name, fan in state['fans'].items():
        if fan['alarm'] or fan['fault']:
            raise RuntimeError('%s alarm/fault: %r' % (name, fan))
        if min(fan['primary_rpm'], fan['secondary_rpm']) < 2700:
            raise RuntimeError('%s tach below 2700: %r' % (name, fan))
        if target is not None and fan['target'] != target:
            raise RuntimeError('%s target changed from %s to %s' %
                               (name, target, fan['target']))
    for name, value in state['temps_c'].items():
        if value >= 45.0:
            raise RuntimeError('%s reached %.3f C' % (name, value))
        if name in baseline and value - baseline[name] >= 5.0:
            raise RuntimeError('%s rose %.3f C from baseline' %
                               (name, value - baseline[name]))


def trial(stages):
    objects = get_temp_objects()
    base = sample(objects)
    record('baseline', **base)
    validate(base, base['temps_c'], 4300, True)
    baseline = base['temps_c']
    try:
        subprocess.check_call(['systemctl', 'stop', SERVICE_NAMES[0]])
        record('control_paused')
        state = sample(objects)
        record('paused_baseline', **state)
        validate(state, baseline, 4300, False)
        for target, seconds in stages:
            set_targets(target)
            record('stage_start', target=target, duration_s=seconds)
            end = time.time() + seconds
            n = 0
            while time.time() < end:
                state = sample(objects)
                record('sample', stage=target, **state)
                validate(state, baseline, target, False)
                n += 1
                time.sleep(10)
            record('stage_complete', target=target, samples=n)
    except Exception as exc:
        record('abort', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if not restore():
            record('FATAL', error='automatic restoration failed')
            raise RuntimeError('automatic restoration failed')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: fan_trial_bmc.py sample|restore|trial [target seconds]')
    if sys.argv[1] == 'sample':
        print(json.dumps(sample(get_temp_objects()), sort_keys=True))
    elif sys.argv[1] == 'restore':
        sys.exit(0 if restore() else 2)
    elif sys.argv[1] == 'trial':
        stages = STAGES
        if len(sys.argv) == 4:
            stages = [(int(sys.argv[2]), int(sys.argv[3]))]
        trial(stages)
    else:
        sys.exit('unknown command')
