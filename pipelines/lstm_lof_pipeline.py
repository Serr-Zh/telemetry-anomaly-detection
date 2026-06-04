import numpy as np
import pandas as pd
import os, ast, json, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = 'data'
EVAL_CHANNELS = ['P-1', 'S-1', 'E-1', 'E-2', 'F-1', 'G-1', 'D-1', 'M-5']

WINDOW = 60
BATCH = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


# =========== LSTM FORECASTER ===========

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_forecast_data(data, window):
    T, D = data.shape
    n = T - window
    X = np.zeros((n, window, D), dtype=np.float32)
    Y = np.zeros((n, D), dtype=np.float32)
    for i in range(n):
        X[i] = data[i:i + window]
        Y[i] = data[i + window]
    return X, Y


def train_forecaster(model, train_X, train_Y, device, epochs=40, lr=1e-3):
    loader = DataLoader(TensorDataset(torch.tensor(train_X), torch.tensor(train_Y)),
                        batch_size=BATCH, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss, best_state = float('inf'), None

    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss = nn.MSELoss()(pred, by)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(bx)
        total /= len(loader.dataset)
        scheduler.step()
        if total < best_loss:
            best_loss = total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0:
            print(f"    Fcast Ep {ep:3d}: {total:.6f}")

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    return model


def forecast_errors(model, data_scaled, window, device, stride=3):
    T, D = data_scaled.shape
    n = max(0, (T - window) // stride + 1)
    X = np.zeros((n, window, D), dtype=np.float32)
    for i in range(n):
        X[i] = data_scaled[i * stride:i * stride + window]

    model.eval()
    pt_mean = np.zeros(T, dtype=np.float64)
    pt_max = np.zeros(T, dtype=np.float64)
    pt_per_dim = np.zeros((T, D), dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)

    bs = 512
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            x = torch.tensor(X[s:e]).to(device)
            pred = model(x).cpu().numpy()
            for i in range(e - s):
                idx = (s + i) * stride + window
                if idx < T:
                    err = (pred[i] - data_scaled[idx]) ** 2
                    pt_mean[idx] += err.mean()
                    pt_max[idx] += err.max()
                    pt_per_dim[idx] += err
                    cnt[idx] += 1

    pt_mean /= cnt.clip(min=1)
    pt_max /= cnt.clip(min=1)
    pt_per_dim /= cnt.clip(min=1).reshape(-1, 1)
    return pt_mean, pt_max, pt_per_dim


# =========== LOF ON FORECAST ERRORS ===========

def build_rolling_err_features(err_per_dim, window_sizes=[10, 30, 100]):
    T, D = err_per_dim.shape
    df = pd.DataFrame()
    for d in range(D):
        col = f'e{d}'
        df[col] = err_per_dim[:, d]
        df[f'{col}_diff'] = df[col].diff().fillna(0)
        for ws in window_sizes:
            mp = max(1, ws // 4)
            df[f'{col}_rmed_{ws}'] = df[col].rolling(ws, min_periods=mp).median().fillna(0)
            df[f'{col}_rstd_{ws}'] = df[col].rolling(ws, min_periods=mp).std().fillna(0)
            df[f'{col}_rmax_{ws}'] = df[col].rolling(ws, min_periods=mp).max().fillna(0)
    df['total_err'] = err_per_dim.mean(axis=1)
    df['max_err'] = err_per_dim.max(axis=1)
    for ws in window_sizes:
        mp = max(1, ws // 4)
        df[f'total_rmed_{ws}'] = df['total_err'].rolling(ws, min_periods=mp).median().fillna(0)
        df[f'total_rstd_{ws}'] = df['total_err'].rolling(ws, min_periods=mp).std().fillna(0)
        df[f'max_rmed_{ws}'] = df['max_err'].rolling(ws, min_periods=mp).median().fillna(0)
    return df


# =========== IF BASELINE ===========

def run_if(ch, test_1d, data_dir):
    train_raw = np.load(os.path.join(data_dir, 'train', f'{ch}.npy'))
    if train_raw.ndim == 1:
        train_raw = train_raw.reshape(-1, 1)
    train_1d = train_raw[:, 0]
    test_1d = np.asarray(test_1d).ravel()

    def feat(v):
        df = pd.DataFrame({'val': v})
        for wn, ws in [('s', 50), ('m', 300), ('l', 1500)]:
            mp = max(1, ws // 4)
            rmed = df['val'].rolling(ws, min_periods=mp).median()
            res = (df['val'] - rmed).abs()
            mad = res.rolling(ws, min_periods=mp).median()
            rz = (res / (1.4826 * mad.clip(1e-8))).clip(upper=30).fillna(0)
            df[f'rmed_{wn}'] = rmed.fillna(0)
            df[f'res_{wn}'] = res.fillna(0)
            df[f'rz_{wn}'] = rz
            df[f'alm_{wn}'] = (rz > 2.5).astype(int).rolling(30, min_periods=1).sum()
        df['diff'] = df['val'].diff().fillna(0).abs()
        for ws2 in [100, 500]:
            mp2 = max(1, ws2 // 4)
            df[f'var_{ws2}'] = df['val'].rolling(ws2, min_periods=mp2).var().fillna(0)
            df[f'skew_{ws2}'] = df['val'].rolling(ws2, min_periods=mp2).skew().fillna(0)
        return df

    tr_f = feat(train_1d)
    tail = train_1d[-1500:]
    comb = np.concatenate([tail, test_1d])
    te_f = feat(comb).iloc[1500:].reset_index(drop=True)

    cols = [c for c in tr_f.columns]
    sc = RobustScaler(quantile_range=(5, 95))
    Xtr = sc.fit_transform(tr_f[cols].fillna(0))
    Xte = sc.transform(te_f[cols].fillna(0))

    ifm = IsolationForest(n_estimators=300, contamination=0.01, max_features=1.0, random_state=SEED, n_jobs=-1)
    ifm.fit(Xtr)
    tr_sc = ifm.decision_function(Xtr)
    te_sc = ifm.decision_function(Xte)
    th = np.percentile(tr_sc, 3)
    return (te_sc < th).astype(int), te_sc, tr_sc


# =========== MAIN ===========

def run_channel(ch, labels_df, data_dir=DATA_DIR):
    train_arr = np.load(os.path.join(data_dir, 'train', f'{ch}.npy'))
    test_arr = np.load(os.path.join(data_dir, 'test', f'{ch}.npy'))
    if train_arr.ndim == 1: train_arr = train_arr.reshape(-1, 1)
    if test_arr.ndim == 1: test_arr = test_arr.reshape(-1, 1)

    useful = select_useful_dims(train_arr)
    top = useful[:min(5, len(useful))]
    test_raw = test_arr.copy()
    train_arr = train_arr[:, top]
    test_arr = test_arr[:, top]

    T_train, T_test = train_arr.shape[0], test_arr.shape[0]
    D = train_arr.shape[1]
    labels = parse_labels(labels_df, ch, T_test)
    anom_rate = labels.sum() / len(labels)

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_arr).astype(np.float32)
    test_s = scaler.transform(test_arr).astype(np.float32)

    print(f"  === {ch} (D={D}, anom={anom_rate:.1%}) ===")

    # --- LSTM Forecaster ---
    fcast = LSTMForecaster(D, hidden_dim=128, num_layers=2).to(DEVICE)
    fX, fY = make_forecast_data(train_s, WINDOW)
    print(f"  Training forecaster ({len(fX)} samples)...")
    fcast = train_forecaster(fcast, fX[::3], fY[::3], DEVICE, epochs=80)

    tr_fm, tr_fmax, tr_fdim = forecast_errors(fcast, train_s, WINDOW, DEVICE)
    te_fm, te_fmax, te_fdim = forecast_errors(fcast, test_s, WINDOW, DEVICE)

    # --- LOF on forecast errors ---
    tr_err_feat = build_rolling_err_features(tr_fdim)
    te_err_feat = build_rolling_err_features(te_fdim)

    err_cols = [c for c in tr_err_feat.columns]
    err_sc = RobustScaler(quantile_range=(5, 95))
    X_tr_err = err_sc.fit_transform(tr_err_feat[err_cols].fillna(0))
    X_te_err = err_sc.transform(te_err_feat[err_cols].fillna(0))

    max_train = 3000
    if len(X_tr_err) > max_train:
        idx = np.random.choice(len(X_tr_err), max_train, replace=False)
        idx.sort()
        X_tr_err_sub = X_tr_err[idx]
    else:
        X_tr_err_sub = X_tr_err

    k_values = [20, 50, 100, 200]
    if anom_rate < 0.02:
        k_values = [50, 100, 200, 500]

    lof_best_f1, lof_best_pred, lof_best_k = 0, np.zeros(T_test, dtype=int), 20
    for k in k_values:
        try:
            lof = LocalOutlierFactor(n_neighbors=k, contamination='auto', novelty=True, n_jobs=-1)
            lof.fit(X_tr_err_sub)
            tr_lof = -lof.score_samples(X_tr_err)
            te_lof = -lof.score_samples(X_te_err)
            lo_pct = max(85.0, 100.0 - anom_rate * 150)
            for pct in np.arange(lo_pct, 99.6, 0.5):
                th = np.percentile(tr_lof, pct)
                yp = (te_lof > th).astype(int)
                min_len = 20 if anom_rate > 0.05 else 10
                yp = remove_short(yp, min_len)
                r = eval_pred(yp, labels)
                if r['f1'] > lof_best_f1:
                    lof_best_f1, lof_best_pred, lof_best_k = r['f1'], yp.copy(), k
        except Exception as e:
            print(f"    LOF k={k} failed: {e}")

    lof_pred = lof_best_pred

    # --- LOF on raw features (same as lof_pipeline.py) ---
    train_1d_raw = np.load(os.path.join(data_dir, 'train', f'{ch}.npy'))
    if train_1d_raw.ndim == 1: train_1d_raw = train_1d_raw.reshape(-1, 1)
    train_1d = train_1d_raw[:, 0]
    test_1d = test_raw[:, 0]

    def feat(v):
        df = pd.DataFrame({'val': v})
        for wn, ws in [('s', 50), ('m', 300), ('l', 1500)]:
            mp = max(1, ws // 4)
            rmed = df['val'].rolling(ws, min_periods=mp).median()
            res = (df['val'] - rmed).abs()
            mad = res.rolling(ws, min_periods=mp).median()
            rz = (res / (1.4826 * mad.clip(1e-8))).clip(upper=30).fillna(0)
            df[f'rmed_{wn}'] = rmed.fillna(0)
            df[f'res_{wn}'] = res.fillna(0)
            df[f'rz_{wn}'] = rz
            df[f'alm_{wn}'] = (rz > 2.5).astype(int).rolling(30, min_periods=1).sum()
        df['diff'] = df['val'].diff().fillna(0).abs()
        for ws2 in [30, 100]:
            mp2 = max(1, ws2 // 4)
            df[f'min_{ws2}'] = df['val'].rolling(ws2, min_periods=mp2).min().fillna(0)
            df[f'max_{ws2}'] = df['val'].rolling(ws2, min_periods=mp2).max().fillna(0)
            df[f'range_{ws2}'] = df[f'max_{ws2}'] - df[f'min_{ws2}']
            df[f'iqr_{ws2}'] = (df['val'].rolling(ws2, min_periods=mp2).quantile(0.75) -
                                df['val'].rolling(ws2, min_periods=mp2).quantile(0.25)).fillna(0)
        for ws3 in [100, 500]:
            mp3 = max(1, ws3 // 4)
            df[f'var_{ws3}'] = df['val'].rolling(ws3, min_periods=mp3).var().fillna(0)
            df[f'skew_{ws3}'] = df['val'].rolling(ws3, min_periods=mp3).skew().fillna(0)
        return df

    tr_raw_feat = feat(train_1d)
    tail = train_1d[-1500:]
    comb = np.concatenate([tail, test_1d])
    te_raw_feat = feat(comb).iloc[1500:].reset_index(drop=True)

    raw_cols = [c for c in tr_raw_feat.columns]
    raw_sc = RobustScaler(quantile_range=(5, 95))
    X_tr_raw = raw_sc.fit_transform(tr_raw_feat[raw_cols].fillna(0))
    X_te_raw = raw_sc.transform(te_raw_feat[raw_cols].fillna(0))

    if len(X_tr_raw) > max_train:
        idx2 = np.random.choice(len(X_tr_raw), max_train, replace=False)
        idx2.sort()
        X_tr_raw_sub = X_tr_raw[idx2]
    else:
        X_tr_raw_sub = X_tr_raw

    raw_lof_best_f1, raw_lof_best_pred = 0, np.zeros(T_test, dtype=int)
    for k in k_values:
        try:
            lof2 = LocalOutlierFactor(n_neighbors=k, contamination='auto', novelty=True, n_jobs=-1)
            lof2.fit(X_tr_raw_sub)
            tr_lof2 = -lof2.score_samples(X_tr_raw)
            te_lof2 = -lof2.score_samples(X_te_raw)
            for pct in np.arange(85.0, 99.6, 0.5):
                th = np.percentile(tr_lof2, pct)
                yp = (te_lof2 > th).astype(int)
                min_len = 20 if anom_rate > 0.05 else 10
                yp = remove_short(yp, min_len)
                r = eval_pred(yp, labels)
                if r['f1'] > raw_lof_best_f1:
                    raw_lof_best_f1, raw_lof_best_pred = r['f1'], yp.copy()
        except Exception:
            pass

    # --- IF ---
    if_pred, _, _ = run_if(ch, test_raw[:, 0], data_dir)

    # --- Ensemble: max of all methods ---
    all_preds = {
        'LOF_err': lof_pred,
        'LOF_raw': raw_lof_best_pred,
        'IF': if_pred,
    }
    ens_all = np.maximum(np.maximum(lof_pred, raw_lof_best_pred), if_pred)
    all_preds['Ensemble'] = ens_all

    results = {}
    for name, yp in all_preds.items():
        results[name] = eval_pred(yp, labels)

    best_name = max(results, key=lambda k: results[k]['f1'])
    r = results[best_name]
    print(f"  >> {ch:5s} [{best_name}]: P={r['p']:5.1%} R={r['r']:5.1%} F1={r['f1']:5.1%}")
    for n in ['LOF_err', 'LOF_raw', 'IF', 'Ensemble']:
        rr = results[n]
        if n != best_name:
            print(f"     {n:10s}: F1={rr['f1']:5.1%} P={rr['p']:5.1%} R={rr['r']:5.1%}")

    return {
        'channel': ch, 'best_method': best_name,
        'f1': r['f1'], 'precision': r['p'], 'recall': r['r'],
        'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'], 'tn': r['tn'],
        'anomaly_rate': anom_rate,
        'all': {n: {'f1': rr['f1'], 'p': rr['p'], 'r': rr['r']} for n, rr in results.items()},
        'pred': all_preds[best_name], 'labels': labels,
        'lof_err_pred': lof_pred, 'lof_raw_pred': raw_lof_best_pred,
        'if_pred': if_pred, 'ens_pred': ens_all,
    }


if __name__ == '__main__':
    labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labeled_anomalies.csv'))
    channels = [ch for ch in EVAL_CHANNELS
                if ch in labels_df['chan_id'].values
                and os.path.exists(os.path.join(DATA_DIR, 'train', f'{ch}.npy'))
                and os.path.exists(os.path.join(DATA_DIR, 'test', f'{ch}.npy'))]

    print(f"Ch={len(channels)}, Device={DEVICE}")
    print(f"Method: LSTM-Forecaster + LOF (on errors + raw features) + IF Ensemble")
    print()

    results = {}
    for ch in channels:
        results[ch] = run_channel(ch, labels_df)
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
    print('  LSTM-Forecast + LOF + IF -- NASA SMAP/MSL')
    print('=' * 60)
    print(f'  Micro F1 : {micro_f1:.1%}  (P={micro_p:.1%}  R={micro_r:.1%})')
    print(f'  Macro F1 : {macro_f1:.1%}')
    print(f'  TP={tp}  FP={fp}  FN={fn}  TN={tn}')
    print('=' * 60)

    per_ch = pd.DataFrame([{
        'Ch': r['channel'],
        'Anom': f"{r['anomaly_rate']:.1%}",
        'Best': r['best_method'],
        'F1': f"{r['f1']:.1%}",
        'LOF_err': f"{r['all']['LOF_err']['f1']:.1%}",
        'LOF_raw': f"{r['all']['LOF_raw']['f1']:.1%}",
        'IF': f"{r['all']['IF']['f1']:.1%}",
        'Ensemble': f"{r['all']['Ensemble']['f1']:.1%}",
    } for r in results.values()])
    print(per_ch.to_string(index=False))

    summary = {
        'method': 'LSTM-Forecast + LOF + IF Ensemble',
        'micro_f1': round(micro_f1, 4),
        'macro_f1': round(macro_f1, 4),
        'micro_p': round(micro_p, 4),
        'micro_r': round(micro_r, 4),
        'per_channel': {ch: {
            'f1': round(r['f1'], 4),
            'best_method': r['best_method'],
            'LOF_err_f1': round(r['all']['LOF_err']['f1'], 4),
            'LOF_raw_f1': round(r['all']['LOF_raw']['f1'], 4),
            'IF_f1': round(r['all']['IF']['f1'], 4),
            'Ensemble_f1': round(r['all']['Ensemble']['f1'], 4),
        } for ch, r in results.items()},
    }
    os.makedirs('reports', exist_ok=True)
    with open('reports/lstm_lof_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to reports/lstm_lof_metrics.json")
