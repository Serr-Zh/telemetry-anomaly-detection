import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import ast, json, time, warnings
warnings.filterwarnings('ignore')

from evaluation.evaluation_utils import (
    parse_labels, get_anomaly_intervals, eval_pred, point_adjust, eval_pred_adjusted,
    time_to_detection, early_detection_fraction, EVAL_CHANNELS, DATA_DIR
)
from pipelines.if_pipeline import _run_single_channel as run_if_ch, _parse_labels as if_parse
from pipelines.lof_pipeline import run_lof_channel
from pipelines.ocsvm_pipeline import run_ocsvm_channel
from pipelines.lstm_vae_pipeline import run_channel as run_lstm_vae_ch
from pipelines.lstm_lof_pipeline import run_channel as run_lstm_lof_ch

METHODS = ['IF', 'LOF', 'OCSVM', 'LSTM-VAE', 'LSTM-LOF']


def run_if_timed(ch, labels_df, data_dir):
    t0 = time.perf_counter()
    r = run_if_ch(ch, labels_df, data_dir)
    elapsed = time.perf_counter() - t0
    if r is None:
        return None
    y_pred = (r['preds'] == -1).astype(int)
    T_test = len(y_pred)
    labels = if_parse(labels_df, ch, T_test)
    return {
        'pred': y_pred, 'labels': labels,
        'train_time': 0, 'inference_time': 0, 'total_time': elapsed,
    }


def run_lof_timed(ch, labels_df, data_dir):
    t0 = time.perf_counter()
    r = run_lof_channel(ch, labels_df, data_dir)
    elapsed = time.perf_counter() - t0
    return {
        'pred': r['pred'], 'labels': r['labels'],
        'train_time': 0, 'inference_time': 0, 'total_time': elapsed,
    }


def run_ocsvm_timed(ch, labels_df, data_dir):
    t0 = time.perf_counter()
    r = run_ocsvm_channel(ch, labels_df, data_dir)
    elapsed = time.perf_counter() - t0
    return {
        'pred': r['pred'], 'labels': r['labels'],
        'train_time': 0, 'inference_time': 0, 'total_time': elapsed,
    }


def run_lstm_vae_timed(ch, labels_df, data_dir):
    t0 = time.perf_counter()
    r = run_lstm_vae_ch(ch, labels_df, data_dir)
    elapsed = time.perf_counter() - t0
    return {
        'pred': r['pred'], 'labels': r['labels'],
        'train_time': 0, 'inference_time': 0, 'total_time': elapsed,
    }


def run_lstm_lof_timed(ch, labels_df, data_dir):
    t0 = time.perf_counter()
    r = run_lstm_lof_ch(ch, labels_df, data_dir)
    elapsed = time.perf_counter() - t0
    return {
        'pred': r['pred'], 'labels': r['labels'],
        'train_time': 0, 'inference_time': 0, 'total_time': elapsed,
    }


RUNNERS = {
    'IF': run_if_timed,
    'LOF': run_lof_timed,
    'OCSVM': run_ocsvm_timed,
    'LSTM-VAE': run_lstm_vae_timed,
    'LSTM-LOF': run_lstm_lof_timed,
}


def ensemble_predictions(preds_dict, labels, strategy='majority'):
    """
    Combine predictions from multiple methods.
    strategies:
      'majority': >= ceil(n/2) methods agree -> anomaly
      'any': any method flags -> anomaly (union)
      'weighted': weighted by method's macro F1
    """
    methods = list(preds_dict.keys())
    n_methods = len(methods)
    T = len(labels)
    stacked = np.column_stack([preds_dict[m] for m in methods])

    if strategy == 'any':
        return stacked.max(axis=1)
    elif strategy == 'majority':
        threshold = np.ceil(n_methods / 2)
        return (stacked.sum(axis=1) >= threshold).astype(int)
    elif strategy == 'weighted':
        weights = np.array([preds_dict.get(f'{m}_weight', 1.0) for m in methods])
        weighted = stacked * weights
        return (weighted.sum(axis=1) >= weights.sum() * 0.5).astype(int)
    else:
        return stacked.max(axis=1)


def main():
    labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labeled_anomalies.csv'))
    channels = [ch for ch in EVAL_CHANNELS
                if ch in labels_df['chan_id'].values
                and os.path.exists(os.path.join(DATA_DIR, 'train', f'{ch}.npy'))
                and os.path.exists(os.path.join(DATA_DIR, 'test', f'{ch}.npy'))]

    print("=" * 70)
    print("  FULL EVALUATION: Point-wise F1 + Point-adjust F1 + Timing + Ensemble")
    print("=" * 70)
    print(f"  Channels: {channels}")
    print(f"  Methods: {METHODS}")
    print()

    all_results = {}

    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")
        runner = RUNNERS[method]
        method_results = {}

        for ch in channels:
            print(f"\n  Running {method} on {ch}...")
            try:
                r = runner(ch, labels_df, DATA_DIR)
                if r is None:
                    print(f"    SKIP: {ch}")
                    continue

                pred = r['pred']
                labels = r['labels']
                total_time = r['total_time']

                pw = eval_pred(pred, labels)
                pa = eval_pred_adjusted(pred, labels)
                ttd = time_to_detection(pred, labels)
                edf = early_detection_fraction(pred, labels)

                method_results[ch] = {
                    'pw_f1': pw['f1'], 'pw_p': pw['p'], 'pw_r': pw['r'],
                    'pa_f1': pa['f1'], 'pa_p': pa['p'], 'pa_r': pa['r'],
                    'total_time_s': round(total_time, 2),
                    'mean_ttd': ttd['mean_ttd'],
                    'detection_rate': ttd['detection_rate'],
                    'early_detection_50': edf,
                    'pred': pred, 'labels': labels,
                }

                print(f"    PW-F1={pw['f1']:.1%}  PA-F1={pa['f1']:.1%}  "
                      f"time={total_time:.1f}s  TTD={ttd['mean_ttd']:.0f}")

            except Exception as e:
                print(f"    ERROR on {ch}: {e}")

        all_results[method] = method_results

    # --- Ensemble ---
    print(f"\n{'='*60}")
    print(f"  Ensemble (all methods)")
    print(f"{'='*60}")

    ensemble_results = {}
    for ch in channels:
        preds_ch = {}
        labels_ch = None
        for method in METHODS:
            if ch in all_results.get(method, {}):
                preds_ch[method] = all_results[method][ch]['pred']
                labels_ch = all_results[method][ch]['labels']

        if len(preds_ch) < 2 or labels_ch is None:
            continue

        for strategy in ['majority', 'any']:
            ens_pred = ensemble_predictions(preds_ch, labels_ch, strategy=strategy)
            pw = eval_pred(ens_pred, labels_ch)
            pa = eval_pred_adjusted(ens_pred, labels_ch)

            key = f"Ensemble-{strategy}"
            if ch not in ensemble_results:
                ensemble_results[ch] = {}
            ensemble_results[ch][strategy] = {
                'pw_f1': pw['f1'], 'pw_p': pw['p'], 'pw_r': pw['r'],
                'pa_f1': pa['f1'], 'pa_p': pa['p'], 'pa_r': pa['r'],
                'pred': ens_pred, 'labels': labels_ch,
            }
            print(f"  {ch} [{strategy}]: PW-F1={pw['f1']:.1%}  PA-F1={pa['f1']:.1%}")

    # --- Aggregate ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    summary = {}
    for method in METHODS + ['Ensemble-majority', 'Ensemble-any']:
        if method.startswith('Ensemble'):
            strategy = method.split('-')[1]
            ch_results = ensemble_results
        else:
            ch_results = all_results.get(method, {})

        pw_f1s, pa_f1s, times = [], [], []
        pw_micro_tp, pw_micro_fp, pw_micro_fn = 0, 0, 0
        pa_micro_tp, pa_micro_fp, pa_micro_fn = 0, 0, 0

        for ch in channels:
            if method.startswith('Ensemble'):
                if ch not in ch_results or strategy not in ch_results[ch]:
                    continue
                cr = ch_results[ch][strategy]
            else:
                if ch not in ch_results:
                    continue
                cr = ch_results[ch]

            pw_f1s.append(cr['pw_f1'])
            pa_f1s.append(cr['pa_f1'])
            if 'total_time_s' in cr:
                times.append(cr['total_time_s'])

            pw_r = eval_pred(cr['pred'], cr['labels'])
            pa_r = eval_pred_adjusted(cr['pred'], cr['labels'])
            pw_micro_tp += pw_r['tp']
            pw_micro_fp += pw_r['fp']
            pw_micro_fn += pw_r['fn']
            pa_micro_tp += pa_r['tp']
            pa_micro_fp += pa_r['fp']
            pa_micro_fn += pa_r['fn']

        if not pw_f1s:
            continue

        pw_macro_f1 = np.mean(pw_f1s)
        pa_macro_f1 = np.mean(pa_f1s)

        pw_micro_p = pw_micro_tp / (pw_micro_tp + pw_micro_fp) if (pw_micro_tp + pw_micro_fp) > 0 else 0
        pw_micro_r = pw_micro_tp / (pw_micro_tp + pw_micro_fn) if (pw_micro_tp + pw_micro_fn) > 0 else 0
        pw_micro_f1 = 2 * pw_micro_p * pw_micro_r / (pw_micro_p + pw_micro_r) if (pw_micro_p + pw_micro_r) > 0 else 0

        pa_micro_p = pa_micro_tp / (pa_micro_tp + pa_micro_fp) if (pa_micro_tp + pa_micro_fp) > 0 else 0
        pa_micro_r = pa_micro_tp / (pa_micro_tp + pa_micro_fn) if (pa_micro_tp + pa_micro_fn) > 0 else 0
        pa_micro_f1 = 2 * pa_micro_p * pa_micro_r / (pa_micro_p + pa_micro_r) if (pa_micro_p + pa_micro_r) > 0 else 0

        mean_time = np.mean(times) if times else 0
        total_time = sum(times) if times else 0

        summary[method] = {
            'pw_macro_f1': round(pw_macro_f1, 4),
            'pw_micro_f1': round(pw_micro_f1, 4),
            'pw_micro_p': round(pw_micro_p, 4),
            'pw_micro_r': round(pw_micro_r, 4),
            'pa_macro_f1': round(pa_macro_f1, 4),
            'pa_micro_f1': round(pa_micro_f1, 4),
            'pa_micro_p': round(pa_micro_p, 4),
            'pa_micro_r': round(pa_micro_r, 4),
            'mean_time_per_channel_s': round(mean_time, 2),
            'total_time_s': round(total_time, 2),
        }

        print(f"  {method:20s}: PW-Macro={pw_macro_f1:.1%}  PA-Macro={pa_macro_f1:.1%}  "
              f"PW-Micro={pw_micro_f1:.1%}  PA-Micro={pa_micro_f1:.1%}  "
              f"Time={mean_time:.1f}s/ch")

    # --- Per-channel detail ---
    per_channel = {}
    for method in METHODS:
        for ch in channels:
            if ch in all_results.get(method, {}):
                cr = all_results[method][ch]
                if ch not in per_channel:
                    per_channel[ch] = {}
                per_channel[ch][method] = {
                    'pw_f1': cr['pw_f1'],
                    'pa_f1': cr['pa_f1'],
                    'time_s': cr['total_time_s'],
                }

    for ch in channels:
        if ch in ensemble_results:
            if ch not in per_channel:
                per_channel[ch] = {}
            for strategy in ['majority', 'any']:
                cr = ensemble_results[ch][strategy]
                per_channel[ch][f'Ensemble-{strategy}'] = {
                    'pw_f1': cr['pw_f1'],
                    'pa_f1': cr['pa_f1'],
                }

    # --- Save ---
    output = {
        'summary': summary,
        'per_channel': per_channel,
    }

    os.makedirs('reports', exist_ok=True)
    with open('reports/full_evaluation.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to reports/full_evaluation.json")

    # --- Print comparison table ---
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE (Point-adjust Macro F1)")
    print(f"{'='*70}")
    print(f"  {'Method':20s} {'PW-Macro':>10s} {'PA-Macro':>10s} {'PW-Micro':>10s} {'PA-Micro':>10s} {'Time/ch':>10s}")
    print(f"  {'-'*70}")
    for method in METHODS + ['Ensemble-majority', 'Ensemble-any']:
        if method in summary:
            s = summary[method]
            print(f"  {method:20s} {s['pw_macro_f1']:>9.1%} {s['pa_macro_f1']:>9.1%} "
                  f"{s['pw_micro_f1']:>9.1%} {s['pa_micro_f1']:>9.1%} "
                  f"{s['mean_time_per_channel_s']:>9.1f}s")


if __name__ == '__main__':
    main()
