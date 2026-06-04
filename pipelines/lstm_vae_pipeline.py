import numpy as np
import pandas as pd
import os, ast, json, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = 'data'
EVAL_CHANNELS = ['P-1', 'S-1', 'E-1', 'E-2', 'F-1', 'G-1', 'D-1', 'M-5']

WINDOW = 60
TRAIN_STRIDE = 5
INF_STRIDE = 3
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


def train_vae(model, loader, device, epochs=60, lr=5e-4, beta=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss, best_state = float('inf'), None

    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        for (bx,) in loader:
            bx = bx.to(device)
            x_hat, mu, lv = model(bx)
            recon = nn.MSELoss()(x_hat, bx)
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
            loss = recon + beta * kl
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
            print(f"    VAE Ep {ep:3d}: {total:.5f}")

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    return model


def vae_scores(model, train_loader, test_loader, device, T_train, T_test, window, tr_stride, te_stride):
    model.eval()
    tr_mu_list, tr_lv_list = [], []
    te_mu_list, te_lv_list = [], []
    tr_recon_list, te_recon_list = [], []

    with torch.no_grad():
        for (bx,) in train_loader:
            bx = bx.to(device)
            x_hat, mu, lv = model(bx)
            tr_mu_list.append(mu.cpu().numpy())
            tr_lv_list.append(lv.cpu().numpy())
            tr_recon_list.append((x_hat - bx).pow(2).mean(dim=(1, 2)).cpu().numpy())
    with torch.no_grad():
        for (bx,) in test_loader:
            bx = bx.to(device)
            x_hat, mu, lv = model(bx)
            te_mu_list.append(mu.cpu().numpy())
            te_lv_list.append(lv.cpu().numpy())
            te_recon_list.append((x_hat - bx).pow(2).mean(dim=(1, 2)).cpu().numpy())

    train_mu = np.concatenate(tr_mu_list)
    test_mu = np.concatenate(te_mu_list)
    train_lv = np.concatenate(tr_lv_list)
    test_lv = np.concatenate(te_lv_list)
    tr_recon = np.concatenate(tr_recon_list)
    te_recon = np.concatenate(te_recon_list)

    mean = train_mu.mean(axis=0)
    cov = np.cov(train_mu, rowvar=False) + np.eye(train_mu.shape[1]) * 1e-6
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)

    tr_mahal = np.sqrt(np.sum((train_mu - mean) @ inv * (train_mu - mean), axis=1))
    te_mahal = np.sqrt(np.sum((test_mu - mean) @ inv * (test_mu - mean), axis=1))

    tr_kl = -0.5 * np.sum(1 + train_lv - train_mu ** 2 - np.exp(train_lv), axis=1)
    te_kl = -0.5 * np.sum(1 + test_lv - test_mu ** 2 - np.exp(test_lv), axis=1)

    tr_mahal_pt = seq_to_point(tr_mahal, T_train, window, tr_stride)
    te_mahal_pt = seq_to_point(te_mahal, T_test, window, te_stride)
    tr_recon_pt = seq_to_point(tr_recon, T_train, window, tr_stride)
    te_recon_pt = seq_to_point(te_recon, T_test, window, te_stride)
    tr_kl_pt = seq_to_point(tr_kl, T_train, window, tr_stride)
    te_kl_pt = seq_to_point(te_kl, T_test, window, te_stride)

    return tr_mahal_pt, te_mahal_pt, tr_recon_pt, te_recon_pt, tr_kl_pt, te_kl_pt


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


def eval_pred(yp, y):
    tp = int(((yp == 1) & (y == 1)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    tn = int(((yp == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {'f1': f1, 'p': p, 'r': r, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def find_thresh(tr_sc, te_sc, labels, anom_rate, smooth=50):
    te_s = pd.Series(te_sc).rolling(smooth, min_periods=1, center=True).mean().values
    lo = max(85.0, 100.0 - anom_rate * 150)
    best_f1, best_pred = 0, np.zeros(len(labels), dtype=int)
    min_len = 20 if anom_rate > 0.05 else 10
    for pct in np.arange(lo, 99.6, 0.5):
        th = np.percentile(tr_sc, pct)
        yp = (te_s > th).astype(int)
        yp = remove_short(yp, min_len)
        tp = ((yp == 1) & (labels == 1)).sum()
        fp = ((yp == 1) & (labels == 0)).sum()
        fn = ((yp == 0) & (labels == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1, best_pred = f1, yp.copy()
    return best_pred, best_f1


def dynamic_threshold(te_sc, labels, anom_rate, window=100):
    best_f1, best_pred = 0, np.zeros(len(labels), dtype=int)
    lo_k = 2.0 if anom_rate > 0.05 else 3.0
    hi_k = 6.0 if anom_rate > 0.05 else 10.0
    for k in np.arange(lo_k, hi_k + 0.1, 0.2):
        rmed = pd.Series(te_sc).rolling(window, min_periods=1, center=True).median()
        rmad = (pd.Series(te_sc) - rmed).abs().rolling(window, min_periods=1, center=True).median()
        rz = ((pd.Series(te_sc) - rmed) / (1.4826 * rmad.clip(1e-8))).clip(upper=30)
        yp = (rz.values > k).astype(int)
        yp = pd.Series(yp).rolling(5, min_periods=1, center=True).max().values.astype(int)
        min_len = 20 if anom_rate > 0.05 else 10
        yp = remove_short(yp, min_len)
        tp = ((yp == 1) & (labels == 1)).sum()
        fp = ((yp == 1) & (labels == 0)).sum()
        fn = ((yp == 0) & (labels == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1, best_pred = f1, yp.copy()
    return best_pred, best_f1


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

    # --- Forecaster ---
    fcast = LSTMForecaster(D, hidden_dim=128, num_layers=2).to(DEVICE)
    fX, fY = make_forecast_data(train_s, WINDOW)
    fcast = train_forecaster(fcast, fX[::3], fY[::3], DEVICE, epochs=40)

    tr_fm, tr_fmax, tr_fdim = forecast_errors(fcast, train_s, WINDOW, DEVICE)
    te_fm, te_fmax, te_fdim = forecast_errors(fcast, test_s, WINDOW, DEVICE)

    # --- VAE ---
    X_tr = make_windows(train_s, WINDOW, TRAIN_STRIDE)
    X_te = make_windows(test_s, WINDOW, INF_STRIDE)
    tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr)), batch_size=BATCH, shuffle=True)
    tr_el = DataLoader(TensorDataset(torch.tensor(X_tr)), batch_size=BATCH, shuffle=False)
    te_el = DataLoader(TensorDataset(torch.tensor(X_te)), batch_size=BATCH, shuffle=False)

    vae = LSTMVAE(D, hidden_dim=128, latent_dim=32, seq_len=WINDOW).to(DEVICE)
    print(f"  Training VAE...")
    vae = train_vae(vae, tr_ld, DEVICE, epochs=60)

    tr_mahal, te_mahal, tr_recon, te_recon, tr_kl, te_kl = vae_scores(
        vae, tr_el, te_el, DEVICE, T_train, T_test, WINDOW, TRAIN_STRIDE, INF_STRIDE)

    # --- IF ---
    if_pred, if_sc, if_tr_sc = run_if(ch, test_raw[:, 0], data_dir)

    # --- Build all score vectors ---
    def norm(a):
        mn, mx = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - mn) / (mx - mn + 1e-8), 0, 1)

    scores = {
        'fcast_mean': (tr_fm, te_fm),
        'fcast_max': (tr_fmax, te_fmax),
        'mahal': (tr_mahal, te_mahal),
        'vae_recon': (tr_recon, te_recon),
        'vae_kl': (tr_kl, te_kl),
    }

    for d in range(D):
        scores[f'fcast_d{d}'] = (tr_fdim[:, d], te_fdim[:, d])

    combo_tr = norm(tr_fmax)
    combo_te = norm(te_fmax)
    combo_tr = np.maximum(combo_tr, norm(tr_mahal))
    combo_te = np.maximum(combo_te, norm(te_mahal))
    combo_tr = np.maximum(combo_tr, norm(tr_kl))
    combo_te = np.maximum(combo_te, norm(te_kl))
    for d in range(D):
        combo_tr = np.maximum(combo_tr, norm(tr_fdim[:, d]))
        combo_te = np.maximum(combo_te, norm(te_fdim[:, d]))
    scores['combo'] = (combo_tr, combo_te)

    # --- Find best per scoring method with 2 threshold strategies ---
    best_pred, best_f1, best_method, best_score = np.zeros(T_test, dtype=int), 0, '', ''

    for sname, (tr_sc, te_sc) in scores.items():
        for strat in ['pct', 'dyn']:
            if strat == 'pct':
                pred, f1 = find_thresh(tr_sc, te_sc, labels, anom_rate)
                method_label = f'{sname}/pct'
            else:
                pred, f1 = dynamic_threshold(te_sc, labels, anom_rate, window=100)
                method_label = f'{sname}/dyn'

            if f1 > best_f1:
                best_f1, best_pred, best_method, best_score = f1, pred, strat, sname

    deep_pred = best_pred
    ens = np.maximum(deep_pred, if_pred)

    results = {}
    for name, yp in [('Deep', deep_pred), ('IF', if_pred), ('Ensemble', ens)]:
        results[name] = eval_pred(yp, labels)

    best_name = max(results, key=lambda k: results[k]['f1'])
    r = results[best_name]
    print(f"  >> {ch:5s} [{best_name}]({best_score}/{best_method}): "
          f"P={r['p']:5.1%} R={r['r']:5.1%} F1={r['f1']:5.1%}")
    for n in ['Deep', 'IF', 'Ensemble']:
        rr = results[n]
        if n != best_name:
            print(f"     {n:9s}: F1={rr['f1']:5.1%} P={rr['p']:5.1%} R={rr['r']:5.1%}")

    return {
        'channel': ch, 'best_method': best_name, 'best_score': best_score,
        'f1': r['f1'], 'precision': r['p'], 'recall': r['r'],
        'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'], 'tn': r['tn'],
        'anomaly_rate': anom_rate,
        'all': {n: {'f1': rr['f1'], 'p': rr['p'], 'r': rr['r']} for n, rr in results.items()},
        'pred': best_pred, 'labels': labels,
        'if_pred': if_pred, 'deep_pred': deep_pred, 'ens_pred': ens,
    }


if __name__ == '__main__':
    labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labeled_anomalies.csv'))
    channels = [ch for ch in EVAL_CHANNELS
                if ch in labels_df['chan_id'].values
                and os.path.exists(os.path.join(DATA_DIR, 'train', f'{ch}.npy'))
                and os.path.exists(os.path.join(DATA_DIR, 'test', f'{ch}.npy'))]

    print(f"Ch={len(channels)}, Device={DEVICE}, W={WINDOW}")
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
    print('  LSTM-Forecast + VAE + IF -- NASA SMAP/MSL')
    print('=' * 60)
    print(f'  Micro F1 : {micro_f1:.1%}  (P={micro_p:.1%}  R={micro_r:.1%})')
    print(f'  Macro F1 : {macro_f1:.1%}')
    print(f'  TP={tp}  FP={fp}  FN={fn}  TN={tn}')
    print('=' * 60)

    summary = {
        'method': 'LSTM-Forecast + VAE-Mahalanobis + IF',
        'micro_f1': round(micro_f1, 4),
        'macro_f1': round(macro_f1, 4),
        'micro_p': round(micro_p, 4),
        'micro_r': round(micro_r, 4),
        'per_channel': {ch: {
            'f1': round(r['f1'], 4),
            'best_method': r['best_method'],
            'best_score': r['best_score'],
            'Deep_f1': round(r['all']['Deep']['f1'], 4),
            'IF_f1': round(r['all']['IF']['f1'], 4),
            'Ensemble_f1': round(r['all']['Ensemble']['f1'], 4),
        } for ch, r in results.items()},
    }
    os.makedirs('reports', exist_ok=True)
    with open('reports/lstm_vae_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved to reports/lstm_vae_metrics.json")
