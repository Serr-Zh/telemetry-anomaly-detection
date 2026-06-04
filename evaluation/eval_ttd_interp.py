import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import ast, json, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
EVAL_CHANNELS = ['P-1', 'S-1', 'E-1', 'E-2', 'F-1', 'G-1', 'D-1', 'M-5']


def parse_labels(labels_df, channel_id, T):
    arr = np.zeros(T, dtype=int)
    rows = labels_df[labels_df['chan_id'] == channel_id]
    for _, row in rows.iterrows():
        try:
            seqs = ast.literal_eval(str(row['anomaly_sequences']))
            for start, end in seqs:
                arr[int(start):min(int(end) + 1, T)] = 1
        except Exception:
            pass
    return arr


def get_anomaly_intervals(labels):
    intervals = []
    in_anom = False
    start = 0
    for i in range(len(labels)):
        if labels[i] == 1 and not in_anom:
            in_anom = True
            start = i
        elif labels[i] == 0 and in_anom:
            intervals.append((start, i - 1))
            in_anom = False
    if in_anom:
        intervals.append((start, len(labels) - 1))
    return intervals


def time_to_detection(pred, labels):
    intervals = get_anomaly_intervals(labels)
    ttd_list = []
    for a_start, a_end in intervals:
        detected_at = -1
        for t in range(a_start, min(a_end + 1, len(pred))):
            if pred[t] == 1:
                detected_at = t - a_start
                break
        ttd_list.append((a_start, a_end, detected_at))
    detected = [t for _, _, t in ttd_list if t >= 0]
    return {
        'mean_ttd': float(np.mean(detected)) if detected else -1,
        'median_ttd': float(np.median(detected)) if detected else -1,
        'detection_rate': len(detected) / len(ttd_list) if ttd_list else 0,
        'n_detected': len(detected),
        'n_intervals': len(ttd_list),
        'ttd_list': ttd_list,
    }


def per_dimension_contribution(test_arr, pred, useful_dims=None, min_std=0.01):
    if useful_dims is None:
        useful_dims = np.where(test_arr.std(axis=0) > min_std)[0]
    results = {}
    total_z = 0
    for d in useful_dims:
        vals = test_arr[:, d]
        normal_vals = vals[pred == 0]
        anomaly_vals = vals[pred == 1]
        n_mean = normal_vals.mean() if len(normal_vals) > 0 else 0
        n_std = normal_vals.std() if len(normal_vals) > 1 else 1e-8
        a_mean = anomaly_vals.mean() if len(anomaly_vals) > 0 else 0
        deviation_z = abs(a_mean - n_mean) / n_std if n_std > 1e-8 else 0
        results[d] = {
            'dim': int(d),
            'normal_mean': float(n_mean),
            'anomaly_mean': float(a_mean),
            'deviation_z': float(deviation_z),
        }
        total_z += deviation_z
    for d in results:
        results[d]['relative_contribution'] = results[d]['deviation_z'] / total_z if total_z > 0 else 0
    sorted_dims = sorted(results.values(), key=lambda x: x['deviation_z'], reverse=True)
    return {'ranking': sorted_dims, 'top_dim': sorted_dims[0]['dim'] if sorted_dims else None}


def select_useful_dims(arr_2d, min_std=0.01):
    return np.where(arr_2d.std(axis=0) > min_std)[0]


if __name__ == '__main__':
    labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labeled_anomalies.csv'))
    channels = [ch for ch in EVAL_CHANNELS
                if ch in labels_df['chan_id'].values
                and os.path.exists(os.path.join(DATA_DIR, 'train', f'{ch}.npy'))
                and os.path.exists(os.path.join(DATA_DIR, 'test', f'{ch}.npy'))]

    print("=" * 70)
    print("  TIME-TO-DETECTION + INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    from pipelines.lof_pipeline import run_lof_channel
    from pipelines.ocsvm_pipeline import run_ocsvm_channel

    saved = {}
    for fname in ['if_metrics', 'lof_metrics', 'ocsvm_metrics', 'lstm_lof_metrics']:
        path = os.path.join('reports', f'{fname}.json')
        if os.path.exists(path):
            with open(path) as f:
                saved[fname] = json.load(f)

    all_ttd = {}
    all_interp = {}

    for ch in channels:
        test_arr = np.load(os.path.join(DATA_DIR, 'test', f'{ch}.npy'))
        if test_arr.ndim == 1:
            test_arr = test_arr.reshape(-1, 1)
        T = test_arr.shape[0]
        labels = parse_labels(labels_df, ch, T)
        anom_rate = labels.sum() / len(labels)
        intervals = get_anomaly_intervals(labels)

        print(f"\n=== {ch} (anom={anom_rate:.1%}, intervals={len(intervals)}) ===")

        ch_ttd = {}
        ch_interp = {}

        # LOF (moderate speed, has per-channel function)
        print(f"  Running LOF...")
        lof_res = run_lof_channel(ch, labels_df)
        ttd_lof = time_to_detection(lof_res['pred'], labels)
        interp_lof = per_dimension_contribution(test_arr, lof_res['pred'])
        ch_ttd['LOF'] = ttd_lof
        ch_interp['LOF'] = interp_lof
        print(f"    LOF: F1={lof_res['f1']:.1%}, mean_TTD={ttd_lof['mean_ttd']:.0f}, "
              f"det_rate={ttd_lof['detection_rate']:.1%}, top_dim={interp_lof['top_dim']}")

        # OCSVM (moderate speed, has per-channel function)
        print(f"  Running OCSVM...")
        ocsvm_res = run_ocsvm_channel(ch, labels_df)
        ttd_ocsvm = time_to_detection(ocsvm_res['pred'], labels)
        interp_ocsvm = per_dimension_contribution(test_arr, ocsvm_res['pred'])
        ch_ttd['OCSVM'] = ttd_ocsvm
        ch_interp['OCSVM'] = interp_ocsvm
        print(f"    OCSVM: F1={ocsvm_res['f1']:.1%}, mean_TTD={ttd_ocsvm['mean_ttd']:.0f}, "
              f"det_rate={ttd_ocsvm['detection_rate']:.1%}, top_dim={interp_ocsvm['top_dim']}")

        all_ttd[ch] = ch_ttd
        all_interp[ch] = ch_interp

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: TIME-TO-DETECTION")
    print("=" * 70)
    for method in ['LOF', 'OCSVM']:
        ttds = [all_ttd[ch][method]['mean_ttd'] for ch in channels
                if method in all_ttd.get(ch, {}) and all_ttd[ch][method]['mean_ttd'] >= 0]
        det_rates = [all_ttd[ch][method]['detection_rate'] for ch in channels
                     if method in all_ttd.get(ch, {})]
        if ttds:
            print(f"  {method:10s}: mean_TTD={np.mean(ttds):.0f}, "
                  f"det_rate={np.mean(det_rates):.1%}")

    print("\n  IF (from saved metrics):")
    if_m = saved.get('if_metrics', {})
    for ch in channels:
        if ch in if_m.get('per_channel', {}):
            r = if_m['per_channel'][ch]
            print(f"    {ch}: F1={r['f1']:.1%} P={r['p']:.1%} R={r['r']:.1%}")

    print("\n  LSTM-LOF+IF (from saved metrics):")
    lstm_lof = saved.get('lstm_lof_metrics', {})
    for ch in channels:
        if ch in lstm_lof.get('per_channel', {}):
            r = lstm_lof['per_channel'][ch]
            print(f"    {ch}: F1={r['f1']:.1%}")

    # Save
    save_data = {}
    for ch in channels:
        save_data[ch] = {}
        for method in all_ttd.get(ch, {}):
            ttd = all_ttd[ch][method]
            interp = all_interp.get(ch, {}).get(method, {})
            save_data[ch][method] = {
                'mean_ttd': ttd['mean_ttd'],
                'median_ttd': ttd['median_ttd'],
                'detection_rate': ttd['detection_rate'],
                'n_intervals': ttd['n_intervals'],
                'n_detected': ttd['n_detected'],
                'top_contributing_dim': interp.get('top_dim', None),
            }
        if ch in if_m.get('per_channel', {}):
            save_data[ch]['IF'] = {
                'f1': if_m['per_channel'][ch]['f1'],
                'note': 'TTD not computed, F1 from saved metrics',
            }
        if ch in lstm_lof.get('per_channel', {}):
            save_data[ch]['LSTM-LOF+IF'] = {
                'f1': lstm_lof['per_channel'][ch]['f1'],
                'note': 'TTD not computed, F1 from saved metrics',
            }

    os.makedirs('reports', exist_ok=True)
    with open('reports/ttd_interpretability.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved to reports/ttd_interpretability.json")
