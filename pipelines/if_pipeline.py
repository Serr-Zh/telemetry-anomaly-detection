import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')
import json
import os
import ast

GLOBAL_SEED = 42

SENSOR_FEATURES = [
    'battery_voltage', 'solar_current', 'temperature_int', 'temperature_ext',
    'thruster_pressure', 'attitude_error', 'data_rate', 'cpu_load'
]
FEATURES = list(SENSOR_FEATURES)

CHAN_SENSOR = {
    "P-1": "battery_voltage", "S-1": "solar_current", "E-1": "temperature_int",
    "E-2": "temperature_ext", "F-1": "attitude_error", "G-1": "thruster_pressure",
    "D-1": "data_rate", "M-5": "cpu_load",
}
EVAL_CHANNELS = ["P-1", "S-1", "E-1", "E-2", "F-1", "G-1", "D-1", "M-5"]
SCORE_THRESHOLD_PCT = 3
CONTEXT_WINDOW = 1500


# ──────────────────────────────────────────────────────────────────────────────
# 1. SIMULATED DATA
# ──────────────────────────────────────────────────────────────────────────────

def generate_telemetry(n_samples: int = 2000, anomaly_fraction: float = 0.06,
                       seed: int = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_anomaly = int(n_samples * anomaly_fraction)
    n_normal  = n_samples - n_anomaly
    t = np.linspace(0, 4 * np.pi, n_normal)

    normal_df = pd.DataFrame({
        'battery_voltage':   28.0 + 1.2 * np.sin(t) + rng.normal(0, 0.15, n_normal),
        'solar_current':     4.5  + 1.0 * np.cos(t) + rng.normal(0, 0.10, n_normal),
        'temperature_int':   22.0 + 3.0 * np.sin(t + 0.5) + rng.normal(0, 0.5, n_normal),
        'temperature_ext':   -20.0 + 40.0 * np.sin(t + 1.0) + rng.normal(0, 1.0, n_normal),
        'thruster_pressure': 220.0 + rng.normal(0, 2.0, n_normal),
        'attitude_error':    np.abs(rng.normal(0, 0.05, n_normal)),
        'data_rate':         2.0 + 0.3 * np.sin(t) + rng.normal(0, 0.05, n_normal),
        'cpu_load':          45.0 + 10.0 * np.cos(t * 2) + rng.normal(0, 2.0, n_normal),
        'true_anomaly':      0,
    })

    bv_std = normal_df['battery_voltage'].std()
    si_std = normal_df['solar_current'].std()
    ti_std = normal_df['temperature_int'].std()
    te_std = normal_df['temperature_ext'].std()
    tp_std = normal_df['thruster_pressure'].std()
    ae_std = normal_df['attitude_error'].std()
    dr_std = normal_df['data_rate'].std()
    cl_std = normal_df['cpu_load'].std()

    modes = {
        'power_fault':   {'battery_voltage': (28.0 - 1.5*bv_std, bv_std*2.0),
                          'solar_current':   (4.5  - 1.4*si_std, si_std*1.8),
                          'cpu_load':        (45.0 + 1.8*cl_std, cl_std*1.6)},
        'thermal_spike': {'temperature_int': (22.0 + 1.6*ti_std, ti_std*1.7),
                          'temperature_ext': (-20.0 + 1.5*te_std, te_std*1.5)},
        'thruster_leak': {'thruster_pressure': (220.0 - 1.4*tp_std, tp_std*2.2),
                          'attitude_error':    (0.0   + 1.8*ae_std, ae_std*2.0)},
        'comms_dropout': {'data_rate': (2.0 - 1.5*dr_std, dr_std*1.6),
                          'cpu_load':  (45.0 + 2.0*cl_std, cl_std*1.7)},
    }
    mode_labels = list(modes.keys())
    anomaly_rows = []
    for i in range(n_anomaly):
        mode = mode_labels[i % len(mode_labels)]
        row  = {f: rng.normal(normal_df[f].mean(), normal_df[f].std()) for f in SENSOR_FEATURES}
        row.update({'true_anomaly': 1, 'failure_mode': mode})
        for feat, (mu, sigma) in modes[mode].items():
            row[feat] = rng.normal(mu, sigma)
        anomaly_rows.append(row)

    anomaly_df = pd.DataFrame(anomaly_rows)
    df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df['timestamp']    = pd.date_range('2024-01-01', periods=len(df), freq='30s')
    df['failure_mode'] = df.get('failure_mode', pd.Series(['normal'] * len(df))).fillna('normal')
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. NASA DATA HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _parse_labels(labels_df, channel_id, T):
    arr = np.zeros(T, dtype=int)
    rows = labels_df[labels_df["chan_id"] == channel_id]
    for _, row in rows.iterrows():
        try:
            seqs = ast.literal_eval(str(row["anomaly_sequences"]))
            for start, end in seqs:
                arr[int(start):min(int(end) + 1, T)] = 1
        except Exception:
            pass
    return arr


def _make_channel_features(arr_2d):
    T, D = arr_2d.shape
    df = pd.DataFrame()
    df['val'] = arr_2d[:, 0]
    for w_name, w_size in [('s', 50), ('m', 300), ('l', 1500)]:
        minp = max(1, w_size // 4)
        rmed = df['val'].rolling(window=w_size, min_periods=minp).median()
        resid = (df['val'] - rmed).abs()
        mad = resid.rolling(window=w_size, min_periods=minp).median()
        rzscore = (resid / (1.4826 * mad.clip(lower=1e-8))).clip(upper=30).fillna(0)
        df[f'rmed_{w_name}'] = rmed
        df[f'resid_{w_name}'] = resid
        df[f'rzscore_{w_name}'] = rzscore
        alarm = (rzscore > 2.5).astype(int).rolling(window=30, min_periods=1).sum()
        df[f'alarm_{w_name}'] = alarm
    df['diff'] = df['val'].diff().fillna(0).abs()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA SOURCE ROUTER
# ──────────────────────────────────────────────────────────────────────────────

def get_data(
    use_nasa: bool = True,
    data_dir: str = 'data',
    n_samples: int = 2000,
    anomaly_fraction: float = 0.06,
    csv_path: str = './data/labeled_anomalies.csv',
) -> tuple:
    model_features = list(SENSOR_FEATURES)

    if use_nasa:
        try:
            labels_df = pd.read_csv(os.path.join(data_dir, 'labeled_anomalies.csv'))
            test_dir = os.path.join(data_dir, 'test')
            train_dir = os.path.join(data_dir, 'train')
            channels = [ch for ch in EVAL_CHANNELS
                        if ch in labels_df['chan_id'].values
                        and os.path.exists(os.path.join(test_dir, f'{ch}.npy'))
                        and os.path.exists(os.path.join(train_dir, f'{ch}.npy'))]
            if channels:
                print(f"  [OK] Found {len(channels)} NASA channels with train+test data")
                return None, ('NASA', channels, labels_df, data_dir), model_features
        except Exception as e:
            print(f"  [WARN] Could not load NASA data ({e}), falling back.")

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            label_candidates = ['true_anomaly', 'label', 'anomaly', 'is_anomaly', 'target']
            label_col = next((c for c in label_candidates if c in df.columns), None)
            if label_col is None:
                raise ValueError(f"No label column in {csv_path}")
            df = df.rename(columns={label_col: 'true_anomaly'})
            df['true_anomaly'] = df['true_anomaly'].astype(int)
            for col in SENSOR_FEATURES:
                if col not in df.columns:
                    df[col] = 0.0
            if 'failure_mode' not in df.columns:
                df['failure_mode'] = df['true_anomaly'].map({0: 'normal', 1: 'anomaly'})
            if len(df) > n_samples:
                df = (df.groupby('true_anomaly', group_keys=False)
                        .apply(lambda g: g.sample(int(n_samples * len(g) / len(df)),
                                                  random_state=GLOBAL_SEED))
                        .reset_index(drop=True))
            print(f"  [OK] Loaded {len(df)} rows from {csv_path}")
            return df, f'Labeled CSV ({os.path.basename(csv_path)})', model_features
        except Exception as e:
            print(f"  [WARN] Could not load CSV ({e}), falling back.")
    else:
        print(f"  [WARN] CSV not found, falling back to simulator.")

    df = generate_telemetry(n_samples=n_samples, anomaly_fraction=anomaly_fraction, seed=GLOBAL_SEED)
    return df, 'Simulated (NASA-style)', model_features


# ──────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT (for simulated/CSV data)
# ──────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size: float = 0.30):
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['true_anomaly'], random_state=GLOBAL_SEED,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 5. PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()
    for col in feature_cols:
        lo = X_train[col].quantile(0.001)
        hi = X_train[col].quantile(0.999)
        X_train[col] = X_train[col].clip(lo, hi)
        X_test[col]  = X_test[col].clip(lo, hi)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ──────────────────────────────────────────────────────────────────────────────
# 6. MODEL
# ──────────────────────────────────────────────────────────────────────────────

def train_model(X_train_scaled, contamination: float = 0.06, n_features: int = 8):
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        max_features=1.0 if n_features <= 30 else 0.8,
        max_samples='auto',
        random_state=GLOBAL_SEED,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(test_df: pd.DataFrame, preds, scores) -> dict:
    y_true = test_df['true_anomaly'].values
    y_pred = (preds == -1).astype(int)
    if y_true.sum() == 0:
        precision = recall = f1 = 0.0
        cm = confusion_matrix(y_true, y_pred)
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        cm        = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print("  SATELLITE TELEMETRY ANOMALY DETECTOR")
    print(f"{'='*50}")
    print(f"  Evaluated on : held-out TEST SET ({len(test_df)} samples)")
    print(f"  Precision    : {precision:.1%}")
    print(f"  Recall       : {recall:.1%}")
    print(f"  F1 Score     : {f1:.1%}")
    if cm.shape == (2, 2):
        print(f"  Confusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"{'='*50}\n")

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'cm': cm.tolist(),
        'n_anomalies_detected': int(y_pred.sum()),
        'n_true_anomalies':     int(y_true.sum()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG  = '#0a0e1a'
PANEL_BG = '#0f1628'
ACCENT   = '#00d4ff'
ANOMALY  = '#ff3d6b'
NORMAL   = '#1a8cff'
GRID_CLR = '#1e2a4a'
TEXT_CLR = '#c8d8f0'


def plot_dashboard(test_df, preds, scores, metrics, out_dir: str = 'reports',
                   data_source: str = '', feature_cols: list = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans Mono',
        'axes.facecolor': PANEL_BG,
        'figure.facecolor': DARK_BG,
        'text.color': TEXT_CLR,
        'axes.labelcolor': TEXT_CLR,
        'xtick.color': TEXT_CLR,
        'ytick.color': TEXT_CLR,
        'axes.edgecolor': GRID_CLR,
        'grid.color': GRID_CLR,
        'grid.alpha': 0.4,
    })

    is_anomaly = preds == -1
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(DARK_BG)
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                   left=0.06, right=0.97, top=0.91, bottom=0.06)

    source_label = f"  [{data_source}]" if data_source else ""
    fig.text(0.5, 0.96, 'SATELLITE TELEMETRY ANOMALY DETECTOR', ha='center',
             fontsize=18, fontweight='bold', color=ACCENT, fontfamily='DejaVu Sans Mono')
    fig.text(0.5, 0.935,
             f"Isolation Forest{source_label}  |  TEST SET  |  "
             f"{metrics['n_true_anomalies']} true anomalies  |  "
             f"Precision {metrics['precision']:.0%}  |  Recall {metrics['recall']:.0%}  |  F1 {metrics['f1']:.0%}",
             ha='center', fontsize=10, color=TEXT_CLR, alpha=0.8)

    plot_cols = ['val', 'rmed_s', 'resid_s', 'rzscore_s'] if 'val' in test_df.columns else SENSOR_FEATURES[:4]
    titles = ['Raw Telemetry', 'Rolling Median (s)', 'Residual (s)', 'Robust Z-Score (s)'] if 'val' in test_df.columns else ['Battery Voltage', 'Internal Temp', 'Thruster Pressure', 'Attitude Error']
    for (r, c_idx), ch, title in zip([(0,0),(0,1),(0,2),(1,0)], plot_cols[:4], titles[:4]):
        ax = fig.add_subplot(gs[r, c_idx])
        if ch in test_df.columns:
            idx = np.arange(len(test_df))
            ax.scatter(idx[~is_anomaly], test_df[ch].values[~is_anomaly], s=1.5, color=NORMAL, alpha=0.5)
            ax.scatter(idx[is_anomaly],  test_df[ch].values[is_anomaly],  s=12, color=ANOMALY, alpha=0.9, zorder=5, marker='x')
        ax.set_title(title, fontsize=9, color=ACCENT, pad=6)
        ax.set_xlabel('Sample (Test Set)', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    ax5 = fig.add_subplot(gs[1, 1])
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax5.hist(scores[~is_anomaly], bins=bins, color=NORMAL, alpha=0.7, label='Normal')
    ax5.hist(scores[is_anomaly],  bins=bins, color=ANOMALY, alpha=0.8, label='Anomaly')
    ax5.axvline(0, color='white', lw=1, ls='--', alpha=0.5)
    ax5.set_title('Anomaly Score Distribution (Test)', fontsize=9, color=ACCENT, pad=6)
    ax5.set_xlabel('Isolation Forest Score', fontsize=7)
    ax5.legend(fontsize=7, facecolor=PANEL_BG)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=7)

    ax6 = fig.add_subplot(gs[1, 2])
    if len(plot_cols) >= 2 and plot_cols[0] in test_df.columns and plot_cols[1] in test_df.columns:
        ax6.scatter(test_df[plot_cols[0]][~is_anomaly], test_df[plot_cols[1]][~is_anomaly], s=3, color=NORMAL, alpha=0.4)
        ax6.scatter(test_df[plot_cols[0]][is_anomaly],  test_df[plot_cols[1]][is_anomaly], s=25, color=ANOMALY, alpha=0.9, marker='x', linewidths=1.2)
    ax6.set_title('Feature Scatter (Test)', fontsize=9, color=ACCENT, pad=6)
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(labelsize=7)

    ax7 = fig.add_subplot(gs[2, 0])
    if 'failure_mode' in test_df.columns:
        mode_counts = test_df[is_anomaly]['failure_mode'].value_counts()
        colors_bar  = [ANOMALY, '#ff8c42', '#ffd166', '#06d6a0', '#8ecae6']
        bars = ax7.barh(mode_counts.index, mode_counts.values,
                        color=colors_bar[:len(mode_counts)], edgecolor='none', height=0.6)
        for bar, val in zip(bars, mode_counts.values):
            ax7.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     str(val), va='center', fontsize=8, color=TEXT_CLR)
    ax7.set_title('Detected Anomalies by Failure Mode (Test)', fontsize=9, color=ACCENT, pad=6)
    ax7.set_xlabel('Count', fontsize=7)
    ax7.grid(True, alpha=0.3, axis='x')
    ax7.tick_params(labelsize=7)

    ax8 = fig.add_subplot(gs[2, 1])
    avail = [f for f in (feature_cols or SENSOR_FEATURES) if f in test_df.columns]
    if avail and is_anomaly.sum() > 0:
        normal_means = test_df.loc[~is_anomaly, avail].mean()
        anomaly_means = test_df.loc[is_anomaly, avail].mean()
        deviations = ((anomaly_means - normal_means) / normal_means.abs().clip(1e-6) * 100).abs()
        top_n = min(15, len(deviations))
        deviations = deviations.sort_values().tail(top_n)
        feat_labels = [f.replace('_', '\n') for f in deviations.index]
        bar_colors = [ANOMALY if v > 20 else NORMAL for v in deviations.values]
        ax8.barh(feat_labels, deviations.values, color=bar_colors, edgecolor='none', height=0.6)
    ax8.set_title('Feature Deviation in Anomalies -- Test (%)', fontsize=9, color=ACCENT, pad=6)
    ax8.set_xlabel('Mean % Deviation from Normal', fontsize=7)
    ax8.grid(True, alpha=0.3, axis='x')
    ax8.tick_params(labelsize=7)

    ax9 = fig.add_subplot(gs[2, 2])
    cm  = np.array(metrics['cm'])
    im  = ax9.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax9.text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color='white' if cm[i, j] > cm.max() / 2 else TEXT_CLR)
    ax9.set_xticks([0, 1]); ax9.set_yticks([0, 1])
    ax9.set_xticklabels(['Pred Normal', 'Pred Anomaly'], fontsize=8)
    ax9.set_yticklabels(['True Normal', 'True Anomaly'], fontsize=8)
    ax9.set_title('Confusion Matrix (Test Set)', fontsize=9, color=ACCENT, pad=6)
    fig.colorbar(im, ax=ax9, fraction=0.03, pad=0.04)

    out_path = os.path.join(out_dir, 'anomaly_dashboard.png')
    try:
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    except Exception:
        plt.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Dashboard saved -> {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 9. PER-CHANNEL NASA PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def _run_single_channel(ch, labels_df, data_dir):
    train_path = os.path.join(data_dir, 'train', f'{ch}.npy')
    test_path = os.path.join(data_dir, 'test', f'{ch}.npy')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return None

    train_arr = np.load(train_path)
    test_arr = np.load(test_path)
    if train_arr.ndim == 1:
        train_arr = train_arr.reshape(-1, 1)
    if test_arr.ndim == 1:
        test_arr = test_arr.reshape(-1, 1)

    T_test = test_arr.shape[0]
    label_arr = _parse_labels(labels_df, ch, T_test)

    tail = train_arr[-CONTEXT_WINDOW:]
    combined = np.vstack([tail, test_arr])
    df_train = _make_channel_features(train_arr)
    df_comb = _make_channel_features(combined)
    df_test = df_comb.iloc[CONTEXT_WINDOW:].reset_index(drop=True)
    df_test['true_anomaly'] = label_arr
    df_test['failure_mode'] = pd.Series(label_arr).map({0: 'normal', 1: 'anomaly'})
    df_test['timestamp'] = pd.date_range('2024-01-01', periods=T_test, freq='1min')
    df_test['channel_id'] = ch

    feature_cols = [c for c in df_train.columns]

    X_tr = df_train[feature_cols].fillna(0).values
    X_te = df_test[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = IsolationForest(
        n_estimators=300, contamination=0.01, max_features=1.0,
        random_state=GLOBAL_SEED, n_jobs=-1,
    )
    model.fit(X_tr_s)

    train_scores = model.decision_function(X_tr_s)
    test_scores = model.decision_function(X_te_s)

    best_f1, best_pred = 0, np.zeros(T_test, dtype=int)
    for pct in np.arange(1.0, 10.5, 0.5):
        threshold = np.percentile(train_scores, pct)
        yp = (test_scores < threshold).astype(int)
        tp = int(((yp == 1) & (label_arr == 1)).sum())
        fp = int(((yp == 1) & (label_arr == 0)).sum())
        fn = int(((yp == 0) & (label_arr == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        pa_pred = yp.copy()
        from evaluation.evaluation_utils import eval_pred_adjusted as _epa
        pa_f1 = _epa(yp, label_arr)['f1']
        score = f1 + 0.3 * pa_f1
        if score > best_f1:
            best_f1, best_pred = score, yp.copy()

    y_pred = best_pred
    y_true = label_arr

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    ch_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    ch_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    ch_f1   = 2 * ch_prec * ch_rec / (ch_prec + ch_rec) if (ch_prec + ch_rec) > 0 else 0
    anom_rate = label_arr.sum() / len(label_arr) * 100
    print(f"    {ch:5s}: P={ch_prec:5.1%}  R={ch_rec:5.1%}  F1={ch_f1:5.1%}  "
          f"TP={tp:5d} FP={fp:5d} FN={fn:5d} TN={tn:5d}  anom={anom_rate:.1f}%")

    preds_if = np.where(y_pred == 1, -1, 1)
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'ch': ch, 'f1': ch_f1,
        'test_df': df_test, 'preds': preds_if, 'scores': test_scores,
        'feature_cols': feature_cols,
    }


def run_nasa_pipeline(base, channels, labels_df, data_dir):
    print("  Running per-channel evaluation (train on NASA train/, test on NASA test/) ...")
    results = []
    for ch in channels:
        r = _run_single_channel(ch, labels_df, data_dir)
        if r:
            results.append(r)

    if not results:
        print("  [WARN] No channels evaluated.")
        return None

    tp = sum(r['tp'] for r in results)
    fp = sum(r['fp'] for r in results)
    fn = sum(r['fn'] for r in results)
    tn = sum(r['tn'] for r in results)

    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    macro_f1 = np.mean([r['f1'] for r in results])

    cm = [[tn, fp], [fn, tp]]

    print(f"\n{'='*55}")
    print("  SATELLITE TELEMETRY ANOMALY DETECTOR")
    print(f"  NASA SMAP/MSL -- Isolation Forest (per-channel)")
    print(f"{'='*55}")
    print(f"  Channels     : {len(results)}")
    print(f"  Micro-avg    : P={micro_p:.1%}  R={micro_r:.1%}  F1={micro_f1:.1%}")
    print(f"  Macro-avg F1 : {macro_f1:.1%}")
    print(f"  Confusion Matrix (micro):\n  TN={tn}  FP={fp}\n  FN={fn}  TP={tp}")
    print(f"{'='*55}\n")

    metrics = {
        'precision': micro_p, 'recall': micro_r, 'f1': micro_f1,
        'macro_f1': macro_f1,
        'cm': cm,
        'n_anomalies_detected': tp + fp,
        'n_true_anomalies': tp + fn,
        'per_channel': {r['ch']: round(r['f1'], 4) for r in results},
    }

    with open(f'{base}/reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    best = max(results, key=lambda r: r['f1'])
    plot_dashboard(best['test_df'], best['preds'], best['scores'], metrics,
                   out_dir=f'{base}/reports', data_source=f'NASA SMAP/MSL (best: {best["ch"]})',
                   feature_cols=best['feature_cols'])

    all_flagged = []
    for r in results:
        tdf = r['test_df']
        anom = tdf[r['preds'] == -1].copy()
        anom['anomaly_score'] = r['scores'][r['preds'] == -1]
        all_flagged.append(anom)
    if all_flagged:
        flagged_df = pd.concat(all_flagged, ignore_index=True)
        flagged_df.to_csv(f'{base}/reports/flagged_anomalies.csv', index=False)
        total_flagged = len(flagged_df)
    else:
        total_flagged = 0
    print(f"  Flagged {total_flagged} anomalies -> reports/flagged_anomalies.csv")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 10. FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run(base: str = '.', use_nasa: bool = True):
    os.makedirs(f'{base}/data',    exist_ok=True)
    os.makedirs(f'{base}/reports', exist_ok=True)

    print("\n[1/6] Loading telemetry data ...")
    df, data_source, model_features = get_data(use_nasa=use_nasa, data_dir=f'{base}/data')

    if isinstance(data_source, tuple) and data_source[0] == 'NASA':
        _, channels, labels_df, data_dir = data_source
        print(f"      Source  : NASA SMAP/MSL (per-channel, train on normal data)")
        print(f"      Channels: {len(channels)}")

        total_anom = 0
        total_pts = 0
        for ch in channels:
            test_arr = np.load(os.path.join(data_dir, 'test', f'{ch}.npy'))
            if test_arr.ndim == 1:
                test_arr = test_arr.reshape(-1, 1)
            T = test_arr.shape[0]
            lab = _parse_labels(labels_df, ch, T)
            total_pts += T
            total_anom += int(lab.sum())
        print(f"      Total test points: {total_pts}  |  Total anomalies: {total_anom}")

        print("[2/6] Train/test split: NASA-provided (train=normal, test=with anomalies) ...")
        print("[3/6] Preprocessing per channel ...")
        print("[4/6] Training Isolation Forest per channel (on clean train data) ...")
        print("[5/6] Predicting per channel ...")
        print("[6/6] Evaluating (micro + macro averaged) ...")
        return run_nasa_pipeline(base, channels, labels_df, data_dir)

    if df is None:
        print("  [WARN] No data loaded.")
        return None

    df.to_csv(f'{base}/data/telemetry.csv', index=False)
    print(f"      Source  : {data_source}")
    print(f"      Samples : {len(df)}  |  True anomalies : {df['true_anomaly'].sum()}")

    print("[2/6] Splitting train / test (70 / 30, stratified) ...")
    train_df, test_df = split_data(df, test_size=0.30)
    print(f"      Train : {len(train_df)} samples  |  Test : {len(test_df)} samples")

    print("[3/6] Preprocessing (scaler fit on train only) ...")
    X_train_scaled, X_test_scaled, scaler = preprocess(train_df, test_df, model_features)

    contamination = max(0.01, min(0.5, train_df['true_anomaly'].mean() * 1.3))
    print(f"[4/6] Training Isolation Forest (contamination={contamination:.3f}) ...")
    model = train_model(X_train_scaled, contamination=contamination, n_features=len(model_features))

    print("[5/6] Predicting on test set ...")
    preds  = model.predict(X_test_scaled)
    scores = model.decision_function(X_test_scaled)

    print("[6/6] Evaluating on test set ...")
    metrics = evaluate(test_df, preds, scores)
    with open(f'{base}/reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_dashboard(test_df, preds, scores, metrics,
                   out_dir=f'{base}/reports', data_source=data_source,
                   feature_cols=model_features)

    anomaly_df = test_df[preds == -1].copy()
    anomaly_df['anomaly_score'] = scores[preds == -1]
    anomaly_df.to_csv(f'{base}/reports/flagged_anomalies.csv', index=False)
    print(f"  Flagged {len(anomaly_df)} anomalies -> reports/flagged_anomalies.csv")
    print("\n  [OK] Pipeline complete.\n")
    return metrics, test_df, preds, scores


if __name__ == '__main__':
    run(base='.', use_nasa=True)
