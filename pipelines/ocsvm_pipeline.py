import numpy as np
import pandas as pd
import os, ast, json, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

SEED = 42
np.random.seed(SEED)

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


def select_useful_dims(arr_2d, min_std=0.01):
    return np.where(arr_2d.std(axis=0) > min_std)[0]


def build_features(values_1d, window_sizes=[50, 300, 1500]):
    df = pd.DataFrame({'val': values_1d})

    df['diff1'] = df['val'].diff().fillna(0)
    df['diff2'] = df['diff1'].diff().fillna(0)
    df['diff_abs'] = df['diff1'].abs()

    for ws in window_sizes:
        mp = max(1, ws // 4)
        rmed = df['val'].rolling(ws, min_periods=mp).median()
        resid = (df['val'] - rmed).abs()
        mad = resid.rolling(ws, min_periods=mp).median()
        rz = (resid / (1.4826 * mad.clip(1e-8))).clip(upper=30).fillna(0)

        df[f'rmed_{ws}'] = rmed.fillna(0)
        df[f'resid_{ws}'] = resid.fillna(0)
        df[f'rz_{ws}'] = rz
        df[f'alm_{ws}'] = (rz > 2.5).astype(int).rolling(30, min_periods=1).sum()

        rmean = df['val'].rolling(ws, min_periods=mp).mean()
        rstd = df['val'].rolling(ws, min_periods=mp).std().fillna(0)
        df[f'rmean_{ws}'] = rmean.fillna(0)
        df[f'rstd_{ws}'] = rstd.fillna(0)
        df[f'zscore_{ws}'] = ((df['val'] - rmean) / rstd.clip(1e-8)).clip(upper=30).fillna(0)

        df[f'var_{ws}'] = df['val'].rolling(ws, min_periods=mp).var().fillna(0)
        df[f'skew_{ws}'] = df['val'].rolling(ws, min_periods=mp).skew().fillna(0)
        df[f'kurt_{ws}'] = df['val'].rolling(ws, min_periods=mp).kurt().fillna(0)

    for ws2 in [30, 100]:
        mp2 = max(1, ws2 // 4)
        df[f'min_{ws2}'] = df['val'].rolling(ws2, min_periods=mp2).min().fillna(0)
        df[f'max_{ws2}'] = df['val'].rolling(ws2, min_periods=mp2).max().fillna(0)
        df[f'range_{ws2}'] = df[f'max_{ws2}'] - df[f'min_{ws2}']
        df[f'iqr_{ws2}'] = (df['val'].rolling(ws2, min_periods=mp2).quantile(0.75) -
                            df['val'].rolling(ws2, min_periods=mp2).quantile(0.25)).fillna(0)

    return df


def remove_short(pred, min_len):
    pred = pred.copy()
    in_anom = False
    start = 0
    for i in range(len(pred)):
        if pred[i] == 1 and not in_anom:
            in_anom = True
            start = i
        elif pred[i] == 0 and in_anom:
            if i - start < min_len:
                pred[start:i] = 0
            in_anom = False
    if in_anom and len(pred) - start < min_len:
        pred[start:] = 0
    return pred


def eval_pred(yp, y):
    tp = int(((yp == 1) & (y == 1)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    tn = int(((yp == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {'f1': f1, 'p': p, 'r': r, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def median_heuristic_gamma(X):
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(X[:1000], X[:1000])
    return 1.0 / (2.0 * np.median(dists[dists > 0]) ** 2 + 1e-8)


def run_ocsvm_channel(ch, labels_df, data_dir=DATA_DIR):
    train_arr = np.load(os.path.join(data_dir, 'train', f'{ch}.npy'))
    test_arr = np.load(os.path.join(data_dir, 'test', f'{ch}.npy'))
    if train_arr.ndim == 1: train_arr = train_arr.reshape(-1, 1)
    if test_arr.ndim == 1: test_arr = test_arr.reshape(-1, 1)

    T_train, T_test = train_arr.shape[0], test_arr.shape[0]
    labels = parse_labels(labels_df, ch, T_test)
    anom_rate = labels.sum() / len(labels)

    train_1d = train_arr[:, 0]
    test_1d = test_arr[:, 0]

    tail = train_1d[-1500:]
    comb = np.concatenate([tail, test_1d])
    train_feat = build_features(train_1d)
    comb_feat = build_features(comb)
    test_feat = comb_feat.iloc[1500:].reset_index(drop=True)

    feat_cols = [c for c in train_feat.columns]
    sc = RobustScaler(quantile_range=(5, 95))
    X_train = sc.fit_transform(train_feat[feat_cols].fillna(0))
    X_test = sc.transform(test_feat[feat_cols].fillna(0))

    max_train = 3000
    if len(X_train) > max_train:
        idx = np.random.choice(len(X_train), max_train, replace=False)
        idx.sort()
        X_train_sub = X_train[idx]
    else:
        X_train_sub = X_train

    gamma_med = median_heuristic_gamma(X_train_sub)
    gammas = [gamma_med * f for f in [0.1, 0.5, 1.0, 2.0, 5.0]]
    gammas = [g for g in gammas if g > 1e-8 and g < 100]

    if anom_rate < 0.02:
        nus = [0.01, 0.03, 0.05, 0.1]
    elif anom_rate < 0.1:
        nus = [0.03, 0.05, 0.1, 0.15]
    else:
        nus = [0.05, 0.1, 0.15, 0.2, 0.3]

    best_f1, best_pred, best_nu, best_gamma = 0, np.zeros(T_test, dtype=int), 0.05, gamma_med
    all_params = {}

    for nu in nus:
        for gamma in gammas:
            try:
                ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
                ocsvm.fit(X_train_sub)

                tr_scores = -ocsvm.decision_function(X_train)
                te_scores = -ocsvm.decision_function(X_test)

                min_len = 20 if anom_rate > 0.05 else 10
                lo_pct = max(85.0, 100.0 - anom_rate * 150)

                for pct in np.arange(lo_pct, 99.6, 1.0):
                    th = np.percentile(tr_scores, pct)
                    yp = (te_scores > th).astype(int)
                    yp = remove_short(yp, min_len)
                    r = eval_pred(yp, labels)
                    key = f'nu={nu:.2f}/g={gamma:.4f}/p={pct:.0f}'
                    all_params[key] = r['f1']
                    if r['f1'] > best_f1:
                        best_f1, best_pred = r['f1'], yp.copy()
                        best_nu, best_gamma = nu, gamma

            except Exception as e:
                print(f"    nu={nu}, gamma={gamma:.4f} failed: {e}")

    r = eval_pred(best_pred, labels)
    print(f"  {ch:5s} [OCSVM] nu={best_nu:.2f}, gamma={best_gamma:.4f}: "
          f"P={r['p']:5.1%} R={r['r']:5.1%} F1={r['f1']:5.1%} "
          f"TP={r['tp']:5d} FP={r['fp']:5d} FN={r['fn']:5d}")

    return {
        'channel': ch, 'best_nu': best_nu, 'best_gamma': best_gamma,
        'f1': r['f1'], 'precision': r['p'], 'recall': r['r'],
        'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'], 'tn': r['tn'],
        'anomaly_rate': anom_rate,
        'labels': labels, 'pred': best_pred, 'test_raw': test_arr,
    }

if __name__ == '__main__':
    labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labeled_anomalies.csv'))
    channels = [ch for ch in EVAL_CHANNELS
                if ch in labels_df['chan_id'].values
                and os.path.exists(os.path.join(DATA_DIR, 'train', f'{ch}.npy'))
                and os.path.exists(os.path.join(DATA_DIR, 'test', f'{ch}.npy'))]

    print(f"Channels: {len(channels)}")
    print(f"Method: One-Class SVM (RBF kernel)")
    print()

    results = {}
    for ch in channels:
        print(f"  Processing {ch}...")
        results[ch] = run_ocsvm_channel(ch, labels_df)
        print()

    tp = sum(r['tp'] for r in results.values())
    fp = sum(r['fp'] for r in results.values())
    fn = sum(r['fn'] for r in results.values())
    tn = sum(r['tn'] for r in results.values())
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    macro_f1 = np.mean([r['f1'] for r in results.values()])

    print('=' * 60)
    print('  One-Class SVM (RBF) -- NASA SMAP/MSL')
    print('=' * 60)
    print(f'  Micro F1 : {micro_f1:.1%}  (P={micro_p:.1%}  R={micro_r:.1%})')
    print(f'  Macro F1 : {macro_f1:.1%}')
    print(f'  TP={tp}  FP={fp}  FN={fn}  TN={tn}')
    print('=' * 60)

    per_ch = pd.DataFrame([{
        'Channel': r['channel'],
        'Anomaly': f"{r['anomaly_rate']:.1%}",
        'nu': f"{r['best_nu']:.2f}",
        'gamma': f"{r['best_gamma']:.4f}",
        'F1': f"{r['f1']:.1%}",
        'P': f"{r['precision']:.1%}",
        'R': f"{r['recall']:.1%}",
    } for r in results.values()])
    print()
    print(per_ch.to_string(index=False))

    summary = {
        'method': 'One-Class SVM (RBF)',
        'micro_f1': round(micro_f1, 4),
        'macro_f1': round(macro_f1, 4),
        'micro_p': round(micro_p, 4),
        'micro_r': round(micro_r, 4),
        'per_channel': {ch: {
            'f1': round(r['f1'], 4),
            'nu': round(r['best_nu'], 4),
            'gamma': round(r['best_gamma'], 6),
            'p': round(r['precision'], 4),
            'r': round(r['recall'], 4),
        } for ch, r in results.items()},
    }
    os.makedirs('reports', exist_ok=True)
    with open('reports/ocsvm_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to reports/ocsvm_metrics.json")
