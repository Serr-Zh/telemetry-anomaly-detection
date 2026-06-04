import numpy as np
import pandas as pd
import os, json, warnings
warnings.filterwarnings('ignore')

SEED = 42
rng = np.random.default_rng(SEED)

OUT_DIR = 'data_simulated'
os.makedirs(OUT_DIR, exist_ok=True)

PARAMS = {
    'battery_voltage':   {'base': 28.0, 'amp': 1.2, 'phase': 0.0,   'noise': 0.15},
    'solar_current':     {'base': 4.5,  'amp': 1.0, 'phase': 0.5,   'noise': 0.10},
    'temperature_int':   {'base': 22.0, 'amp': 3.0, 'phase': 0.5,   'noise': 0.5},
    'temperature_ext':   {'base': -20.0,'amp': 40.0,'phase': 1.0,   'noise': 1.0},
    'thruster_pressure': {'base': 220.0,'amp': 0.0, 'phase': 0.0,   'noise': 2.0},
    'attitude_error':    {'base': 0.05, 'amp': 0.0, 'phase': 0.0,   'noise': 0.03},
    'data_rate':         {'base': 2.0,  'amp': 0.3, 'phase': 0.0,   'noise': 0.05},
    'cpu_load':          {'base': 45.0, 'amp': 10.0,'phase': 1.5,   'noise': 2.0},
}
PARAM_NAMES = list(PARAMS.keys())
N_PARAMS = len(PARAM_NAMES)

DRIFT_PARAMS = [
    ['battery_voltage', 'solar_current'],
    ['temperature_int', 'temperature_ext'],
    ['thruster_pressure', 'attitude_error'],
    ['battery_voltage', 'cpu_load'],
    ['solar_current', 'data_rate'],
]

STEP_PARAMS = [
    ['thruster_pressure', 'attitude_error'],
    ['temperature_int', 'cpu_load'],
    ['battery_voltage', 'solar_current'],
    ['data_rate', 'cpu_load'],
    ['temperature_ext', 'temperature_int'],
]

SPIKE_PARAMS = [
    ['battery_voltage'],
    ['temperature_ext'],
    ['thruster_pressure'],
    ['cpu_load'],
    ['attitude_error'],
]


def generate_normal(n_points, rng):
    t = np.linspace(0, 4 * np.pi, n_points)
    data = np.zeros((n_points, N_PARAMS), dtype=np.float64)
    for j, name in enumerate(PARAM_NAMES):
        p = PARAMS[name]
        signal = p['base'] + p['amp'] * np.sin(t + p['phase'])
        data[:, j] = signal + rng.normal(0, p['noise'], n_points)
    if name == 'attitude_error':
        data[:, PARAM_NAMES.index('attitude_error')] = np.abs(data[:, PARAM_NAMES.index('attitude_error')])
    return data


def inject_drift(data, start, length, affected_params, rng):
    data = data.copy()
    end = min(start + length, len(data))
    for param_name in affected_params:
        j = PARAM_NAMES.index(param_name)
        std = PARAMS[param_name]['noise'] * 3
        slope = rng.uniform(0.5, 2.0) * std / length
        for i in range(start, end):
            data[i, j] += slope * (i - start)
    return data


def inject_step(data, start, length, affected_params, rng):
    data = data.copy()
    end = min(start + length, len(data))
    for param_name in affected_params:
        j = PARAM_NAMES.index(param_name)
        std = PARAMS[param_name]['noise'] * 3
        offset = rng.choice([-1, 1]) * rng.uniform(1.5, 4.0) * std
        data[start:end, j] += offset
    return data


def inject_spike(data, start, length, affected_params, rng):
    data = data.copy()
    end = min(start + length, len(data))
    for param_name in affected_params:
        j = PARAM_NAMES.index(param_name)
        std = PARAMS[param_name]['noise'] * 3
        amplitude = rng.choice([-1, 1]) * rng.uniform(3.0, 8.0) * std
        mid = (start + end) // 2
        for i in range(start, end):
            dist = abs(i - mid) / max(1, (end - start) / 2)
            envelope = np.exp(-3 * dist ** 2)
            data[i, j] += amplitude * envelope
    return data


def generate_scenario(scenario_id, n_normal=7000, n_test=5000, anomaly_rate=0.10, rng=None):
    if rng is None:
        rng = np.random.default_rng(scenario_id * 100 + SEED)

    train = generate_normal(n_normal, rng)

    test = generate_normal(n_test, rng)
    labels = np.zeros(n_test, dtype=int)
    anomaly_info = []

    n_anom_points = int(n_test * anomaly_rate)

    base_lengths = {'drift': (150, 300), 'step': (50, 150), 'spike': (10, 30)}
    avg_lens = {'drift': 225, 'step': 100, 'spike': 20}

    n_anom_target = int(n_test * anomaly_rate)
    types_cycle = ['drift', 'step', 'spike'] * 2

    if anomaly_rate >= 0.12:
        types_cycle = types_cycle + ['drift', 'drift', 'step', 'step', 'spike', 'spike']

    scheduled = []
    budget_left = n_anom_target
    for atype in types_cycle:
        min_len, max_len = base_lengths[atype]
        if budget_left <= min_len:
            length = min_len
        else:
            share = avg_lens[atype] / sum(avg_lens[t] for t in types_cycle)
            target = int(n_anom_target * share)
            lo = max(min_len, target - 30)
            hi = min(max_len, target + 30)
            if lo > hi: lo, hi = min_len, max_len
            length = int(rng.integers(lo, hi + 1))
        scheduled.append((atype, length))
        budget_left -= length

    occupied = []
    anomaly_info = []
    labels = np.zeros(n_test, dtype=int)

    for atype, length in scheduled:
        placed = False
        for attempt in range(50):
            start = int(rng.integers(100, n_test - length - 50))
            overlap = False
            for s_start, s_end in occupied:
                if not (start + length < s_start or start > s_end):
                    overlap = True
                    break
            if not overlap:
                placed = True
                break
        if not placed:
            continue

        if atype == 'drift':
            params = DRIFT_PARAMS[int(rng.integers(0, len(DRIFT_PARAMS)))]
            test = inject_drift(test, start, length, params, rng)
        elif atype == 'step':
            params = STEP_PARAMS[int(rng.integers(0, len(STEP_PARAMS)))]
            test = inject_step(test, start, length, params, rng)
        else:
            params = SPIKE_PARAMS[int(rng.integers(0, len(SPIKE_PARAMS)))]
            test = inject_spike(test, start, length, params, rng)

        labels[start:start + length] = 1
        occupied.append((start, start + length))
        anomaly_info.append({
            'type': atype,
            'start': int(start),
            'end': int(start + length - 1),
            'length': int(length),
            'affected_params': params,
        })

    return train, test, labels, anomaly_info


scenarios = []
for sid in range(3):
    rate = [0.05, 0.10, 0.15][sid]
    n_test = [10000, 8000, 8000][sid]
    train, test, labels, info = generate_scenario(sid, n_normal=10000, n_test=n_test, anomaly_rate=rate)

    np.save(os.path.join(OUT_DIR, f'scenario_{sid}_train.npy'), train)
    np.save(os.path.join(OUT_DIR, f'scenario_{sid}_test.npy'), test)
    np.save(os.path.join(OUT_DIR, f'scenario_{sid}_labels.npy'), labels)

    n_anom = int(labels.sum())
    actual_rate = n_anom / len(labels)

    scenario_data = {
        'scenario_id': sid,
        'anomaly_rate_target': rate,
        'anomaly_rate_actual': round(actual_rate, 4),
        'train_shape': list(train.shape),
        'test_shape': list(test.shape),
        'n_anomaly_points': n_anom,
        'params': PARAM_NAMES,
        'anomalies': info,
    }
    scenarios.append(scenario_data)

    drift_count = sum(1 for a in info if a['type'] == 'drift')
    step_count = sum(1 for a in info if a['type'] == 'step')
    spike_count = sum(1 for a in info if a['type'] == 'spike')

    print(f"Scenario {sid} (target={rate:.0%}, actual={actual_rate:.1%}): "
          f"train={train.shape}, test={test.shape}, "
          f"drift={drift_count}, step={step_count}, spike={spike_count}")

with open(os.path.join(OUT_DIR, 'scenarios.json'), 'w') as f:
    json.dump(scenarios, f, indent=2)

print(f"\nSaved to {OUT_DIR}/")
