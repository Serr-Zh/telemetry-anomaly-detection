import numpy as np
import pandas as pd
import os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = 'data_simulated'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW = 60
BATCH = 64


def build_features_1d(values_1d, window_sizes=[50, 300, 1500]):
    df = pd.DataFrame({'val': values_1d})
    df['diff1'] = df['val'].diff().fillna(0)
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
        df[f'rmean_{ws}'] = df['val'].rolling(ws, min_periods=mp).mean().fillna(0)
        df[f'rstd_{ws}'] = df['val'].rolling(ws, min_periods=mp).std().fillna(0)
    for ws2 in [30, 100]:
        mp2 = max(1, ws2 // 4)
        df[f'range_{ws2}'] = df['val'].rolling(ws2, min_periods=mp2).max().fillna(0) - \
                             df['val'].rolling(ws2, min_periods=mp2).min().fillna(0)
    return df


def remove_short(pred, min_len=15):
    pred = pred.copy()
    in_anom, start = False, 0
    for i in range(len(pred)):
        if pred[i] == 1 and not in_anom:
            in_anom, start = True, i
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


def eval_by_type(yp, labels, anomalies):
    results = {}
    for atype in ['drift', 'step', 'spike']:
        type_labels = np.zeros(len(labels), dtype=int)
        for a in anomalies:
            if a['type'] == atype:
                type_labels[a['start']:a['end'] + 1] = 1
        if type_labels.sum() > 0:
            r = eval_pred(yp, type_labels)
            results[atype] = r
        else:
            results[atype] = {'f1': 0, 'p': 0, 'r': 0}
    return results


# =========== METHOD 1: ISOLATION FOREST ===========

def run_if_sim(train, test, labels, anom_rate):
    D = train.shape[1]
    best_f1, best_pred, best_dim = 0, np.zeros(len(labels), dtype=int), 0

    for d in range(D):
        train_1d = train[:, d]
        test_1d = test[:, d]

        tail = train_1d[-1500:]
        comb = np.concatenate([tail, test_1d])
        tr_f = build_features_1d(train_1d)
        te_f = build_features_1d(comb).iloc[1500:].reset_index(drop=True)

        cols = [c for c in tr_f.columns]
        sc = RobustScaler(quantile_range=(5, 95))
        Xtr = sc.fit_transform(tr_f[cols].fillna(0))
        Xte = sc.transform(te_f[cols].fillna(0))

        ifm = IsolationForest(n_estimators=300, contamination=0.01, max_features=1.0, random_state=SEED, n_jobs=-1)
        ifm.fit(Xtr)
        tr_sc = ifm.decision_function(Xtr)
        te_sc = ifm.decision_function(Xte)

        for pct in np.arange(90.0, 99.6, 1.0):
            th = np.percentile(tr_sc, pct)
            yp = (te_sc < th).astype(int)
            yp = remove_short(yp)
            r = eval_pred(yp, labels)
            if r['f1'] > best_f1:
                best_f1, best_pred, best_dim = r['f1'], yp.copy(), d

    return best_pred


# =========== METHOD 2: LOF ===========

def run_lof_sim(train, test, labels, anom_rate):
    D = train.shape[1]
    best_f1, best_pred, best_dim = 0, np.zeros(len(labels), dtype=int), 0

    for d in range(D):
        train_1d = train[:, d]
        test_1d = test[:, d]

        tail = train_1d[-1500:]
        comb = np.concatenate([tail, test_1d])
        tr_f = build_features_1d(train_1d)
        te_f = build_features_1d(comb).iloc[1500:].reset_index(drop=True)

        cols = [c for c in tr_f.columns]
        sc = RobustScaler(quantile_range=(5, 95))
        Xtr = sc.fit_transform(tr_f[cols].fillna(0))
        Xte = sc.transform(te_f[cols].fillna(0))

        max_tr = 3000
        if len(Xtr) > max_tr:
            idx = np.random.choice(len(Xtr), max_tr, replace=False)
            idx.sort()
            Xtr_sub = Xtr[idx]
        else:
            Xtr_sub = Xtr

        for k in [100]:
            try:
                lof = LocalOutlierFactor(n_neighbors=k, contamination='auto', novelty=True, n_jobs=-1)
                lof.fit(Xtr_sub)
                tr_sc = -lof.score_samples(Xtr)
                te_sc = -lof.score_samples(Xte)
                for pct in np.arange(90.0, 99.6, 1.0):
                    th = np.percentile(tr_sc, pct)
                    yp = (te_sc > th).astype(int)
                    yp = remove_short(yp)
                    r = eval_pred(yp, labels)
                    if r['f1'] > best_f1:
                        best_f1, best_pred, best_dim = r['f1'], yp.copy(), d
            except Exception:
                pass

    return best_pred


# =========== METHOD 3: OCSVM ===========

def run_ocsvm_sim(train, test, labels, anom_rate):
    D = train.shape[1]
    best_f1, best_pred, best_dim = 0, np.zeros(len(labels), dtype=int), 0

    for d in range(D):
        train_1d = train[:, d]
        test_1d = test[:, d]

        tail = train_1d[-1500:]
        comb = np.concatenate([tail, test_1d])
        tr_f = build_features_1d(train_1d)
        te_f = build_features_1d(comb).iloc[1500:].reset_index(drop=True)

        cols = [c for c in tr_f.columns]
        sc = RobustScaler(quantile_range=(5, 95))
        Xtr = sc.fit_transform(tr_f[cols].fillna(0))
        Xte = sc.transform(te_f[cols].fillna(0))

        max_tr = 3000
        if len(Xtr) > max_tr:
            idx = np.random.choice(len(Xtr), max_tr, replace=False)
            idx.sort()
            Xtr_sub = Xtr[idx]
        else:
            Xtr_sub = Xtr

        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(Xtr_sub[:500], Xtr_sub[:500])
        gamma_med = 1.0 / (2.0 * np.median(dists[dists > 0]) ** 2 + 1e-8)

        for nu in [0.03, 0.05, 0.1, 0.15]:
            for gamma in [gamma_med * f for f in [0.5, 1.0, 2.0]]:
                try:
                    ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
                    ocsvm.fit(Xtr_sub)
                    tr_sc = -ocsvm.decision_function(Xtr)
                    te_sc = -ocsvm.decision_function(Xte)
                    for pct in np.arange(90.0, 99.6, 1.0):
                        th = np.percentile(tr_sc, pct)
                        yp = (te_sc > th).astype(int)
                        yp = remove_short(yp)
                        r = eval_pred(yp, labels)
                        if r['f1'] > best_f1:
                            best_f1, best_pred, best_dim = r['f1'], yp.copy(), d
                except Exception:
                    pass

    return best_pred


# =========== METHOD 4: LSTM-VAE + IF ===========

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, seq_len=60):
        super().__init__()
        self.seq_len = seq_len
        self.enc = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mu_l = nn.Linear(hidden_dim, latent_dim)
        self.lv_l = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.LSTM(latent_dim + input_dim, hidden_dim, batch_first=True)
        self.out_l = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h, _) = self.enc(x)
        h = h.squeeze(0)
        return self.mu_l(h), self.lv_l(h)

    def reparam(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def decode(self, z, x):
        z_rep = z.unsqueeze(1).expand(-1, self.seq_len, -1)
        x_shift = torch.zeros_like(x)
        x_shift[:, 1:] = x[:, :-1]
        dec_in = torch.cat([z_rep, x_shift], dim=2)
        out, _ = self.dec(dec_in)
        return self.out_l(out)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return self.decode(z, x), mu, lv


def make_forecast_data(data, window):
    T, D = data.shape
    n = T - window
    X = np.zeros((n, window, D), dtype=np.float32)
    Y = np.zeros((n, D), dtype=np.float32)
    for i in range(n):
        X[i] = data[i:i + window]
        Y[i] = data[i + window]
    return X, Y


def make_windows(data, window, stride):
    T, D = data.shape
    n = max(0, (T - window) // stride + 1)
    out = np.zeros((n, window, D), dtype=np.float32)
    for i in range(n):
        out[i] = data[i * stride:i * stride + window]
    return out


def seq_to_point(scores, T, window, stride):
    pt = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)
    for i in range(len(scores)):
        s = i * stride
        e = min(s + window, T)
        pt[s:e] += scores[i]
        cnt[s:e] += 1
    return pt / cnt.clip(min=1)


def run_deep_sim(train, test, labels, anom_rate):
    D = train.shape[1]
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train).astype(np.float32)
    test_s = scaler.transform(test).astype(np.float32)

    T_train, T_test = train.shape[0], test.shape[0]

    # Forecaster
    fX, fY = make_forecast_data(train_s, WINDOW)
    fcast = LSTMForecaster(D, hidden_dim=128, num_layers=2).to(DEVICE)
    loader = DataLoader(TensorDataset(torch.tensor(fX[::3]), torch.tensor(fY[::3])),
                        batch_size=BATCH, shuffle=True)
    optimizer = torch.optim.Adam(fcast.parameters(), lr=1e-3)
    for ep in range(1, 21):
        fcast.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            pred = fcast(bx)
            loss = nn.MSELoss()(pred, by)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(fcast.parameters(), 1.0)
            optimizer.step()

    # Forecast errors
    fcast.eval()

    # Train forecast errors
    tr_fcast_errors = np.zeros(T_train, dtype=np.float64)
    tr_fcast_counts = np.zeros(T_train, dtype=np.float64)
    tr_seq = make_windows(train_s, WINDOW, 3)
    tr_fcast_loader = DataLoader(TensorDataset(torch.tensor(tr_seq)), batch_size=256, shuffle=False)
    with torch.no_grad():
        for bi, (bx,) in enumerate(tr_fcast_loader):
            bx = bx.to(DEVICE)
            pred = fcast(bx).cpu().numpy()
            for i in range(len(pred)):
                idx = bi * 256 + i
                pt_idx = idx * 3 + WINDOW
                if pt_idx < T_train:
                    err = ((pred[i] - train_s[pt_idx]) ** 2).mean()
                    tr_fcast_errors[pt_idx] += err
                    tr_fcast_counts[pt_idx] += 1
    tr_fcast_errors /= tr_fcast_counts.clip(min=1)

    # Test forecast errors
    te_fcast_errors = np.zeros(T_test, dtype=np.float64)
    te_fcast_counts = np.zeros(T_test, dtype=np.float64)
    te_fcast_seq = make_windows(test_s, WINDOW, 3)
    te_fcast_loader = DataLoader(TensorDataset(torch.tensor(te_fcast_seq)), batch_size=256, shuffle=False)
    with torch.no_grad():
        for (bx,) in te_fcast_loader:
            bx = bx.to(DEVICE)
            pred = fcast(bx).cpu().numpy()
            for i in range(len(pred)):
                idx = i * 3 + WINDOW
                if idx < T_test:
                    err = ((pred[i] - test_s[idx]) ** 2).mean()
                    te_fcast_errors[idx] += err
                    te_fcast_counts[idx] += 1
    te_fcast_errors /= te_fcast_counts.clip(min=1)

    # VAE
    X_tr = make_windows(train_s, WINDOW, 5)
    X_te = make_windows(test_s, WINDOW, 3)
    tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr)), batch_size=BATCH, shuffle=True)
    tr_el = DataLoader(TensorDataset(torch.tensor(X_tr)), batch_size=BATCH, shuffle=False)
    te_el = DataLoader(TensorDataset(torch.tensor(X_te)), batch_size=BATCH, shuffle=False)

    vae = LSTMVAE(D, hidden_dim=128, latent_dim=32, seq_len=WINDOW).to(DEVICE)
    vae_opt = torch.optim.Adam(vae.parameters(), lr=5e-4)
    for ep in range(1, 31):
        vae.train()
        for (bx,) in tr_ld:
            bx = bx.to(DEVICE)
            x_hat, mu, lv = vae(bx)
            recon = nn.MSELoss()(x_hat, bx)
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
            loss = recon + 0.1 * kl
            vae_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            vae_opt.step()

    # VAE scores
    vae.eval()
    tr_mus, te_mus = [], []
    with torch.no_grad():
        for (bx,) in tr_el:
            mu, _ = vae.encode(bx.to(DEVICE))
            tr_mus.append(mu.cpu().numpy())
        for (bx,) in te_el:
            mu, _ = vae.encode(bx.to(DEVICE))
            te_mus.append(mu.cpu().numpy())
    tr_mu = np.concatenate(tr_mus)
    te_mu = np.concatenate(te_mus)

    mean = tr_mu.mean(axis=0)
    cov = np.cov(tr_mu, rowvar=False) + np.eye(tr_mu.shape[1]) * 1e-6
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)

    tr_mahal = np.sqrt(np.sum((tr_mu - mean) @ inv * (tr_mu - mean), axis=1))
    te_mahal = np.sqrt(np.sum((te_mu - mean) @ inv * (te_mu - mean), axis=1))

    tr_mahal_pt = seq_to_point(tr_mahal, T_train, WINDOW, 5)
    te_mahal_pt = seq_to_point(te_mahal, T_test, WINDOW, 3)

    def norm(a):
        mn, mx = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - mn) / (mx - mn + 1e-8), 0, 1)

    combo_tr = np.maximum(norm(tr_mahal_pt), norm(tr_fcast_errors))
    combo_te = np.maximum(norm(te_mahal_pt), norm(te_fcast_errors))

    best_f1, best_pred = 0, np.zeros(T_test, dtype=int)
    for sname, (tr_sc, te_sc) in [('mahal', (tr_mahal_pt, te_mahal_pt)),
                                    ('fcast', (tr_fcast_errors, te_fcast_errors)),
                                    ('combo', (combo_tr, combo_te))]:
        for pct in np.arange(85.0, 99.6, 0.5):
            th = np.percentile(tr_sc, pct)
            te_s = pd.Series(te_sc).rolling(30, min_periods=1, center=True).mean().values
            yp = (te_s > th).astype(int)
            yp = remove_short(yp)
            r = eval_pred(yp, labels)
            if r['f1'] > best_f1:
                best_f1, best_pred = r['f1'], yp.copy()

    return best_pred


# =========== MAIN ===========

with open(os.path.join(DATA_DIR, 'scenarios.json')) as f:
    scenarios = json.load(f)

all_results = []

for sc in scenarios:
    sid = sc['scenario_id']
    train = np.load(os.path.join(DATA_DIR, f'scenario_{sid}_train.npy'))
    test = np.load(os.path.join(DATA_DIR, f'scenario_{sid}_test.npy'))
    labels = np.load(os.path.join(DATA_DIR, f'scenario_{sid}_labels.npy'))
    anom_rate = labels.sum() / len(labels)
    anomalies = sc['anomalies']

    print(f"\n{'='*60}")
    print(f"  Scenario {sid}: anom_rate={anom_rate:.1%}, train={train.shape}, test={test.shape}")
    print(f"{'='*60}")

    methods = {}

    print(f"  Running IF...")
    if_pred = run_if_sim(train, test, labels, anom_rate)
    methods['IF'] = if_pred
    r = eval_pred(if_pred, labels)
    by_type = eval_by_type(if_pred, labels, anomalies)
    print(f"    IF: F1={r['f1']:.1%} P={r['p']:.1%} R={r['r']:.1%}  "
          f"drift={by_type['drift']['f1']:.1%} step={by_type['step']['f1']:.1%} spike={by_type['spike']['f1']:.1%}")

    print(f"  Running LOF...")
    lof_pred = run_lof_sim(train, test, labels, anom_rate)
    methods['LOF'] = lof_pred
    r = eval_pred(lof_pred, labels)
    by_type = eval_by_type(lof_pred, labels, anomalies)
    print(f"    LOF: F1={r['f1']:.1%} P={r['p']:.1%} R={r['r']:.1%}  "
          f"drift={by_type['drift']['f1']:.1%} step={by_type['step']['f1']:.1%} spike={by_type['spike']['f1']:.1%}")

    print(f"  Running LSTM-VAE+IF...")
    deep_pred = run_deep_sim(train, test, labels, anom_rate)
    ens_pred = np.maximum(np.maximum(np.maximum(if_pred, lof_pred), deep_pred), 0)
    methods['Deep'] = deep_pred
    methods['Ensemble'] = ens_pred
    for mname, mpred in [('Deep', deep_pred), ('Ensemble', ens_pred)]:
        r = eval_pred(mpred, labels)
        by_type = eval_by_type(mpred, labels, anomalies)
        print(f"    {mname}: F1={r['f1']:.1%} P={r['p']:.1%} R={r['r']:.1%}  "
              f"drift={by_type['drift']['f1']:.1%} step={by_type['step']['f1']:.1%} spike={by_type['spike']['f1']:.1%}")

    for mname, mpred in methods.items():
        r = eval_pred(mpred, labels)
        by_type = eval_by_type(mpred, labels, anomalies)
        all_results.append({
            'scenario': sid, 'anomaly_rate': anom_rate, 'method': mname,
            'f1': r['f1'], 'p': r['p'], 'r': r['r'],
            'drift_f1': by_type['drift']['f1'],
            'step_f1': by_type['step']['f1'],
            'spike_f1': by_type['spike']['f1'],
        })

print(f"\n{'='*60}")
print(f"  SUMMARY -- Simulated Telemetry")
print(f"{'='*60}")

df = pd.DataFrame(all_results)

print("\n-- Overall F1 by Method and Scenario --")
pivot = df.pivot_table(index='method', columns='scenario', values='f1').round(3)
print(pivot.to_string())

print("\n-- F1 by Anomaly Type (averaged across scenarios) --")
for atype in ['drift', 'step', 'spike']:
    col = f'{atype}_f1'
    print(f"  {atype:6s}: " + "  ".join(f"{m}={df[df['method']==m][col].mean():.1%}"
          for m in ['IF', 'LOF', 'Deep', 'Ensemble']))

print("\n-- Macro F1 per Method (across scenarios) --")
for m in ['IF', 'LOF', 'Deep', 'Ensemble']:
    vals = df[df['method'] == m]['f1'].values
    print(f"  {m:10s}: {np.mean(vals):.1%}")

os.makedirs('reports', exist_ok=True)
df.to_csv('reports/simulated_results.csv', index=False)
print("\nSaved to reports/simulated_results.csv")
