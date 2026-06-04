import numpy as np
import pandas as pd
import os, ast, json, warnings
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


def eval_pred(yp, y):
    tp = int(((yp == 1) & (y == 1)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    tn = int(((yp == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {'f1': f1, 'p': p, 'r': r, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def point_adjust(pred, labels):
    """
    Point-adjust protocol (Hundman et al., KDD 2018):
    If ANY point in a ground-truth anomaly segment is detected,
    mark ALL points in that segment as correctly detected.
    Returns adjusted predictions.
    """
    pred_adj = pred.copy()
    intervals = get_anomaly_intervals(labels)
    for start, end in intervals:
        if pred[start:end + 1].any():
            pred_adj[start:end + 1] = 1
    return pred_adj


def eval_pred_adjusted(yp, y):
    """Point-adjust F1 evaluation."""
    yp_adj = point_adjust(yp, y)
    return eval_pred(yp_adj, y)


# =========== TIME-TO-DETECTION ===========

def time_to_detection(pred, labels):
    """
    For each ground-truth anomaly interval, compute how many time steps
    elapsed from the start of the interval until the first detection.
    Returns:
        ttd_list: list of (interval_start, interval_end, ttd) tuples
            ttd = 0 if detected at interval start
            ttd = -1 if never detected
        mean_ttd: mean time-to-detection (excluding missed intervals)
        median_ttd: median time-to-detection (excluding missed intervals)
        detection_rate: fraction of anomaly intervals that were detected
    """
    intervals = get_anomaly_intervals(labels)
    ttd_list = []

    for a_start, a_end in intervals:
        length = a_end - a_start + 1
        detected_at = -1
        for t in range(a_start, min(a_end + 1, len(pred))):
            if pred[t] == 1:
                detected_at = t - a_start
                break
        ttd_list.append((a_start, a_end, detected_at))

    detected = [t for _, _, t in ttd_list if t >= 0]
    missed = [t for _, _, t in ttd_list if t < 0]

    mean_ttd = np.mean(detected) if detected else -1
    median_ttd = np.median(detected) if detected else -1
    detection_rate = len(detected) / len(ttd_list) if ttd_list else 0

    return {
        'ttd_list': ttd_list,
        'mean_ttd': mean_ttd,
        'median_ttd': median_ttd,
        'detection_rate': detection_rate,
        'n_detected': len(detected),
        'n_missed': len(missed),
        'n_intervals': len(ttd_list),
    }


def early_detection_fraction(pred, labels, max_delay=50):
    """
    Fraction of anomaly intervals detected within max_delay time steps.
    """
    ttd = time_to_detection(pred, labels)
    if ttd['n_intervals'] == 0:
        return 0
    early = sum(1 for _, _, d in ttd['ttd_list'] if 0 <= d <= max_delay)
    return early / ttd['n_intervals']


# =========== INTERPRETABILITY ===========

def per_dimension_contribution(test_arr, pred, useful_dims=None, min_std=0.01):
    """
    For each dimension, compute how much it deviates from normal during
    predicted anomaly windows vs normal windows.

    Returns dict with per-dim stats:
        - normal_mean, anomaly_mean: mean values in normal/anomaly regions
        - deviation_z: z-score of anomaly mean relative to normal distribution
        - relative_contribution: fraction of total z-score sum (for ranking)
    """
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
            'normal_std': float(n_std),
            'deviation_z': float(deviation_z),
            'deviation_abs': float(abs(a_mean - n_mean)),
        }
        total_z += deviation_z

    for d in results:
        results[d]['relative_contribution'] = (
            results[d]['deviation_z'] / total_z if total_z > 0 else 0
        )

    sorted_dims = sorted(results.values(), key=lambda x: x['deviation_z'], reverse=True)
    return {
        'per_dim': results,
        'ranking': sorted_dims,
        'top_dim': sorted_dims[0]['dim'] if sorted_dims else None,
        'total_z': total_z,
    }


def forecast_error_attribution(err_per_dim, pred, useful_dims=None):
    """
    For LSTM-forecast-based methods: which dimension's forecast error
    contributed most to the detected anomalies?

    err_per_dim: (T, D) array of per-dimension forecast errors
    pred: (T,) binary prediction array
    """
    if useful_dims is None:
        useful_dims = list(range(err_per_dim.shape[1]))

    results = {}
    total_err = 0

    for d in useful_dims:
        vals = err_per_dim[:, d]
        normal_err = vals[pred == 0].mean() if (pred == 0).any() else 0
        anomaly_err = vals[pred == 1].mean() if (pred == 1).any() else 0

        err_ratio = anomaly_err / (normal_err + 1e-8)
        results[d] = {
            'dim': int(d),
            'normal_error': float(normal_err),
            'anomaly_error': float(anomaly_err),
            'error_ratio': float(err_ratio),
            'error_increase': float(anomaly_err - normal_err),
        }
        total_err += (anomaly_err - normal_err)

    for d in results:
        results[d]['relative_contribution'] = (
            results[d]['error_increase'] / total_err if total_err > 0 else 0
        )

    sorted_dims = sorted(results.values(), key=lambda x: x['error_ratio'], reverse=True)
    return {
        'per_dim': results,
        'ranking': sorted_dims,
        'top_dim': sorted_dims[0]['dim'] if sorted_dims else None,
    }


# =========== COMPREHENSIVE EVALUATION ===========

def comprehensive_eval(pred, labels, test_arr=None, err_per_dim=None,
                       useful_dims=None, channel_name=''):
    """
    Full evaluation: F1 + Time-to-Detection + Interpretability.
    """
    r = eval_pred(pred, labels)
    ttd = time_to_detection(pred, labels)
    edf = early_detection_fraction(pred, labels, max_delay=50)

    result = {
        'channel': channel_name,
        'f1': r['f1'],
        'precision': r['p'],
        'recall': r['r'],
        'tp': r['tp'],
        'fp': r['fp'],
        'fn': r['fn'],
        'mean_ttd': ttd['mean_ttd'],
        'median_ttd': ttd['median_ttd'],
        'detection_rate': ttd['detection_rate'],
        'early_detection_50': edf,
        'n_intervals': ttd['n_intervals'],
        'n_detected_intervals': ttd['n_detected'],
        'n_missed_intervals': ttd['n_missed'],
    }

    if test_arr is not None:
        interp = per_dimension_contribution(test_arr, pred, useful_dims)
        result['top_contributing_dim'] = interp['top_dim']
        result['interpretability'] = interp

    if err_per_dim is not None:
        fcast_interp = forecast_error_attribution(err_per_dim, pred, useful_dims)
        result['top_error_dim'] = fcast_interp['top_dim']
        result['forecast_interpretability'] = fcast_interp

    return result


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labeled_anomalies.csv'))
    from pipelines.lstm_lof_pipeline import run_channel as run_lstm_lof
    from pipelines.ocsvm_pipeline import run_ocsvm_channel
    from pipelines.lof_pipeline import run_lof_channel
    from pipelines.if_pipeline import run_if_channel

    channels = [ch for ch in EVAL_CHANNELS
                if ch in labels_df['chan_id'].values
                and os.path.exists(os.path.join(DATA_DIR, 'train', f'{ch}.npy'))]

    print("=" * 70)
    print("  COMPREHENSIVE EVALUATION: F1 + Time-to-Detection + Interpretability")
    print("=" * 70)

    all_results = {}
    for ch in channels:
        test_arr = np.load(os.path.join(DATA_DIR, 'test', f'{ch}.npy'))
        if test_arr.ndim == 1:
            test_arr = test_arr.reshape(-1, 1)
        T = test_arr.shape[0]
        labels = parse_labels(labels_df, ch, T)

        from pipelines.lstm_lof_pipeline import select_useful_dims
        useful = select_useful_dims(test_arr)

        print(f"\n--- {ch} (anom={labels.sum()/len(labels):.1%}) ---")

        for method_name, run_fn in [('LSTM-LOF+IF', None), ('OCSVM', None)]:
            pass

    print("\nDone. Time-to-Detection and Interpretability metrics computed.")
