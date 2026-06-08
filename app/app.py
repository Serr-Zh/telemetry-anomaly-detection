import os, sys, ast, json, webbrowser, threading, warnings
warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, jsonify, request

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics.pairwise import euclidean_distances

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
SIM_DIR = os.path.join(_PROJECT_ROOT, "data_simulated")
REPORTS_DIR = os.path.join(_PROJECT_ROOT, "reports")
EVAL_CHANNELS = ["P-1", "S-1", "E-1", "E-2", "F-1", "G-1", "D-1", "M-5"]
SIM_PARAMS = ["battery_voltage", "solar_current", "temperature_int",
              "temperature_ext", "thruster_pressure", "attitude_error",
              "data_rate", "cpu_load"]
METHOD_KEYS = ["IF", "LOF", "OCSVM", "LSTM_VAE_IF", "LSTM_LOF_IF", "Prophet", "SARIMA"]
METHOD_LABELS = {"IF": "Isolation Forest", "LOF": "Local Outlier Factor",
                 "OCSVM": "One-Class SVM", "LSTM_VAE_IF": "LSTM-VAE + IF",
                 "LSTM_LOF_IF": "LSTM-Forecast + LOF + IF",
                 "Prophet": "Prophet", "SARIMA": "SARIMA"}
METHOD_COLORS = {"IF": "#ef5350", "LOF": "#42a5f5", "OCSVM": "#66bb6a",
                 "LSTM_VAE_IF": "#ffa726", "LSTM_LOF_IF": "#ab47bc",
                 "Prophet": "#26c6da", "SARIMA": "#8d6e63"}

HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Satellite Anomaly Detector</title>
<script src="/static/plotly.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #1a1a2e; }
.app { display: flex; min-height: 100vh; }
.sidebar { width: 280px; background: #fff; border-right: 1px solid #ddd; padding: 20px; position: fixed; top: 0; bottom: 0; overflow-y: auto; }
.sidebar h1 { font-size: 16px; margin-bottom: 4px; }
.sidebar .subtitle { font-size: 11px; color: #888; margin-bottom: 16px; }
.sidebar h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #666; margin: 16px 0 8px; border-bottom: 1px solid #eee; padding-bottom: 4px; }
.sidebar label { display: block; font-size: 13px; margin: 6px 0; cursor: pointer; }
.sidebar select, .sidebar button { width: 100%; padding: 8px; font-size: 13px; border: 1px solid #ccc; border-radius: 6px; margin: 4px 0; }
.sidebar select { background: #fff; }
.sidebar button { background: #4361ee; color: #fff; border: none; font-weight: 600; cursor: pointer; margin-top: 12px; }
.sidebar button:hover { background: #3a56d4; }
.sidebar button:disabled { background: #aaa; cursor: wait; }
.sidebar .info { font-size: 11px; color: #888; margin: 8px 0; line-height: 1.4; }
.sidebar .checkbox-row { display: flex; align-items: center; gap: 6px; }
.sidebar .checkbox-row input { margin: 0; }
.main { margin-left: 280px; padding: 24px 32px; flex: 1; }
.section-title { font-size: 14px; font-weight: 700; margin: 20px 0 8px; color: #1a1a2e; }
.metrics-row { display: flex; gap: 12px; margin: 12px 0; }
.metric-card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; text-align: center; flex: 1; }
.metric-card .val { font-size: 22px; font-weight: 700; }
.metric-card .lbl { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 16px; background: #fff; border-radius: 8px; overflow: hidden; font-size: 13px; }
th { background: #4361ee; color: #fff; padding: 8px 12px; text-align: center; font-size: 12px; font-weight: 600; }
td { padding: 7px 12px; text-align: center; border-bottom: 1px solid #eee; }
tr:hover td { background: #f0f0ff; }
.best-row td { background: #e8f5e9; font-weight: 700; }
.plot-container { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 8px; margin: 8px 0 16px; }
#log { font-family: monospace; font-size: 12px; color: #666; margin: 8px 0; min-height: 20px; }
.footer { text-align: center; padding: 16px; font-size: 11px; color: #aaa; border-top: 1px solid #eee; margin-top: 32px; }
</style>
</head>
<body>
<div class="app">
<div class="sidebar">
  <h1>Anomaly Detector</h1>
  <div class="subtitle">Unsupervised Methods for Satellite Telemetry</div>

  <h2>Data Source</h2>
  <label><input type="radio" name="mode" value="nasa" checked onchange="switchMode()"> NASA SMAP/MSL (real)</label>
  <label><input type="radio" name="mode" value="sim" onchange="switchMode()"> Simulated telemetry</label>

  <div id="nasa-controls">
    <h2>Channel</h2>
    <select id="channel" onchange="update()">
      {% for ch in channels %}<option value="{{ch}}">{{ch}}</option>{% endfor %}
    </select>
    <div class="info">Pre-computed results from pipeline runs.<br>LSTM methods require long training.</div>
  </div>

  <div id="sim-controls" style="display:none">
    <h2>Scenario</h2>
    <select id="scenario" onchange="update()">
      <option value="0">Scenario 0 (5% anomalies)</option>
      <option value="1">Scenario 1 (10% anomalies)</option>
      <option value="2">Scenario 2 (15% anomalies)</option>
    </select>
    <h2>Parameter</h2>
    <select id="param" onchange="update()">
      {% for i, p in enumerate_params %}<option value="{{i}}">{{p}}</option>{% endfor %}
    </select>
    <button id="runBtn" onclick="runSynthetic()">Run Methods</button>
    <div class="info">IF, LOF, OCSVM run in real-time.<br>LSTM-VAE and LSTM-LOF use pre-computed results.</div>
  </div>

  <h2>View</h2>
  <div class="checkbox-row"><input type="checkbox" id="chkCompare" checked onchange="update()"> All-method comparison</div>
  <div class="checkbox-row"><input type="checkbox" id="chkTTD" checked onchange="update()"> Time-to-Detection</div>
</div>

<div class="main" id="content">
  <p>Select a channel or scenario, then view results.</p>
</div>
</div>

<script>
function switchMode() {
  var mode = document.querySelector('input[name=mode]:checked').value;
  document.getElementById('nasa-controls').style.display = mode === 'nasa' ? '' : 'none';
  document.getElementById('sim-controls').style.display = mode === 'sim' ? '' : 'none';
  update();
}

function update() {
  var mode = document.querySelector('input[name=mode]:checked').value;
  var params = new URLSearchParams();
  if (mode === 'nasa') {
    params.set('channel', document.getElementById('channel').value);
  } else {
    params.set('scenario', document.getElementById('scenario').value);
    params.set('param', document.getElementById('param').value);
  }
  params.set('compare', document.getElementById('chkCompare').checked);
  params.set('ttd', document.getElementById('chkTTD').checked);
  fetch('/data?' + params.toString()).then(r=>r.json()).then(render);
}

function runSynthetic() {
  var btn = document.getElementById('runBtn');
  btn.disabled = true; btn.textContent = 'Running...';
  var log = document.getElementById('log');
  if (log) log.textContent = 'Running methods...';
  var params = new URLSearchParams();
  params.set('scenario', document.getElementById('scenario').value);
  params.set('param', document.getElementById('param').value);
  params.set('compare', document.getElementById('chkCompare').checked);
  params.set('ttd', document.getElementById('chkTTD').checked);
  fetch('/run?' + params.toString()).then(r=>r.json()).then(function(d) {
    btn.disabled = false; btn.textContent = 'Run Methods';
    render(d);
  });
}

function render(d) {
  var html = '';
  if (d.error) { html = '<p>' + d.error + '</p>'; document.getElementById('content').innerHTML = html; return; }

  if (d.info) {
    html += '<div class="metrics-row">';
    for (var k in d.info) {
      html += '<div class="metric-card"><div class="val">' + d.info[k] + '</div><div class="lbl">' + k + '</div></div>';
    }
    html += '</div>';
  }

  if (d.telemetry_plot) {
    html += '<div class="section-title">Telemetry with Ground Truth</div>';
    html += '<div class="plot-container" id="plot-telemetry"></div>';
  }

  if (d.method_table) {
    html += '<div class="section-title">Method Results</div>';
    html += '<table><tr>';
    d.method_table.headers.forEach(function(h){ html += '<th>' + h + '</th>'; });
    html += '</tr>';
    d.method_table.rows.forEach(function(row, i) {
      var cls = i === d.method_table.best_idx ? ' class="best-row"' : '';
      html += '<tr' + cls + '>';
      row.forEach(function(cell){ html += '<td>' + cell + '</td>'; });
      html += '</tr>';
    });
    html += '</table>';
    if (d.method_table.best) html += '<div style="color:#2e7d32;font-weight:600;margin-bottom:8px">' + d.method_table.best + '</div>';
  }

  if (d.ttd_table) {
    html += '<div class="section-title">Time-to-Detection</div>';
    html += '<table><tr>';
    d.ttd_table.headers.forEach(function(h){ html += '<th>' + h + '</th>'; });
    html += '</tr>';
    d.ttd_table.rows.forEach(function(row){
      html += '<tr>';
      row.forEach(function(cell){ html += '<td>' + cell + '</td>'; });
      html += '</tr>';
    });
    html += '</table>';
  }

  if (d.events_table) {
    html += '<div class="section-title">Anomaly Events</div>';
    html += '<table><tr>';
    d.events_table.headers.forEach(function(h){ html += '<th>' + h + '</th>'; });
    html += '</tr>';
    d.events_table.rows.forEach(function(row){
      html += '<tr>';
      row.forEach(function(cell){ html += '<td>' + cell + '</td>'; });
      html += '</tr>';
    });
    html += '</table>';
  }

  if (d.run_results) {
    html += '<div id="log"></div>';
    html += '<div class="section-title">Run Results</div>';
    html += '<table><tr>';
    d.run_results.headers.forEach(function(h){ html += '<th>' + h + '</th>'; });
    html += '</tr>';
    d.run_results.rows.forEach(function(row, i) {
      var cls = i === d.run_results.best_idx ? ' class="best-row"' : '';
      html += '<tr' + cls + '>';
      row.forEach(function(cell){ html += '<td>' + cell + '</td>'; });
      html += '</tr>';
    });
    html += '</table>';
    if (d.run_results.best) html += '<div style="color:#2e7d32;font-weight:600;margin-bottom:8px">' + d.run_results.best + '</div>';
  }

  var plotIds = [];
  if (d.channel_plot) { html += '<div class="section-title">Per-Channel F1 Comparison</div><div class="plot-container" id="plot-channel"></div>'; plotIds.push('channel'); }
  if (d.macro_plot) { html += '<div class="section-title">Macro F1 Summary</div><div class="plot-container" id="plot-macro"></div>'; plotIds.push('macro'); }
  if (d.sim_plot) { html += '<div class="section-title">Simulated: F1 by Scenario</div><div class="plot-container" id="plot-sim"></div>'; plotIds.push('sim'); }
  if (d.type_plot) { html += '<div class="section-title">F1 by Anomaly Type</div><div class="plot-container" id="plot-type"></div>'; plotIds.push('type'); }

  if (d.summary_table) {
    html += '<div class="section-title">Summary (All Channels)</div>';
    html += '<table><tr>';
    d.summary_table.headers.forEach(function(h){ html += '<th>' + h + '</th>'; });
    html += '</tr>';
    d.summary_table.rows.forEach(function(row, i) {
      var cls = i === d.summary_table.best_idx ? ' class="best-row"' : '';
      html += '<tr' + cls + '>';
      row.forEach(function(cell){ html += '<td>' + cell + '</td>'; });
      html += '</tr>';
    });
    html += '</table>';
  }

  html += '<div class="footer">Satellite Telemetry Anomaly Detector | IF, LOF, OCSVM, LSTM-VAE+IF, LSTM-Forecast+LOF+IF | NASA SMAP/MSL + Synthetic</div>';

  document.getElementById('content').innerHTML = html;

  if (d.telemetry_plot) Plotly.newPlot('plot-telemetry', d.telemetry_plot.data, d.telemetry_plot.layout);
  if (d.channel_plot) Plotly.newPlot('plot-channel', d.channel_plot.data, d.channel_plot.layout);
  if (d.macro_plot) Plotly.newPlot('plot-macro', d.macro_plot.data, d.macro_plot.layout);
  if (d.sim_plot) Plotly.newPlot('plot-sim', d.sim_plot.data, d.sim_plot.layout);
  if (d.type_plot) Plotly.newPlot('plot-type', d.type_plot.data, d.type_plot.layout);
}
</script>
</body>
</html>"""


def parse_labels(labels_df, channel_id, T):
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


def eval_pred(yp, y):
    tp = int(((yp == 1) & (y == 1)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"f1": f1, "p": p, "r": r, "tp": tp, "fp": fp, "fn": fn}


def remove_short(pred, min_len=10):
    pred = pred.copy()
    in_a, s = False, 0
    for i in range(len(pred)):
        if pred[i] == 1 and not in_a:
            in_a, s = True, i
        elif pred[i] == 0 and in_a:
            if i - s < min_len:
                pred[s:i] = 0
            in_a = False
    if in_a and len(pred) - s < min_len:
        pred[s:] = 0
    return pred


def build_features_1d(v, window_sizes=[50, 300, 1500]):
    df = pd.DataFrame({"val": v})
    for wn, ws in [("s", 50), ("m", 300), ("l", 1500)]:
        mp = max(1, ws // 4)
        rmed = df["val"].rolling(ws, min_periods=mp).median()
        res = (df["val"] - rmed).abs()
        mad = res.rolling(ws, min_periods=mp).median()
        rz = (res / (1.4826 * mad.clip(1e-8))).clip(upper=30).fillna(0)
        df[f"rmed_{wn}"] = rmed.fillna(0)
        df[f"res_{wn}"] = res.fillna(0)
        df[f"rz_{wn}"] = rz
        df[f"alm_{wn}"] = (rz > 2.5).astype(int).rolling(30, min_periods=1).sum()
    df["diff"] = df["val"].diff().fillna(0).abs()
    for ws2 in [100, 500]:
        mp2 = max(1, ws2 // 4)
        df[f"var_{ws2}"] = df["val"].rolling(ws2, min_periods=mp2).var().fillna(0)
        df[f"skew_{ws2}"] = df["val"].rolling(ws2, min_periods=mp2).skew().fillna(0)
    return df


def load_nasa_results():
    results = {}
    mapping = {"IF": "if_metrics.json", "LOF": "lof_metrics.json",
               "OCSVM": "ocsvm_metrics.json", "LSTM_VAE_IF": "lstm_vae_metrics.json",
               "LSTM_LOF_IF": "lstm_lof_metrics.json"}
    for key, fname in mapping.items():
        path = os.path.join(REPORTS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            results[key] = {"micro_f1": data.get("micro_f1", 0),
                            "macro_f1": data.get("macro_f1", 0),
                            "micro_p": data.get("micro_p", 0),
                            "micro_r": data.get("micro_r", 0),
                            "per_channel": {}}
            for ch, cd in data.get("per_channel", {}).items():
                f1 = cd.get("f1", 0)
                p = cd.get("p", cd.get("precision", 0))
                r = cd.get("r", cd.get("recall", 0))
                results[key]["per_channel"][ch] = {"f1": f1, "p": p, "r": r}
    bl_path = os.path.join(REPORTS_DIR, "baselines_metrics.json")
    if os.path.exists(bl_path):
        with open(bl_path) as f:
            bl = json.load(f)
        for bkey in ["Prophet", "SARIMA"]:
            if bkey in bl:
                bd = bl[bkey]
                per_ch = {}
                for ch, cd in bd.get("per_channel", {}).items():
                    per_ch[ch] = {"f1": cd.get("pw_f1", 0), "p": cd.get("p", 0), "r": cd.get("r", 0)}
                results[bkey] = {"micro_f1": 0, "macro_f1": bd.get("pw_macro_f1", 0),
                                "micro_p": 0, "micro_r": 0, "per_channel": per_ch}
    return results


def load_ttd():
    path = os.path.join(REPORTS_DIR, "ttd_interpretability.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_sim():
    path = os.path.join(REPORTS_DIR, "simulated_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def run_if_synthetic(train, test, labels, param_idx=0):
    tr1, te1 = train[:, param_idx], test[:, param_idx]
    tr_f = build_features_1d(tr1)
    tail = tr1[-1500:]
    comb = np.concatenate([tail, te1])
    te_f = build_features_1d(comb).iloc[1500:].reset_index(drop=True)
    cols = [c for c in tr_f.columns]
    sc = RobustScaler(quantile_range=(5, 95))
    Xtr = sc.fit_transform(tr_f[cols].fillna(0))
    Xte = sc.transform(te_f[cols].fillna(0))
    ifm = IsolationForest(n_estimators=300, contamination=0.01, max_features=1.0, random_state=42, n_jobs=-1)
    ifm.fit(Xtr)
    tr_sc = ifm.decision_function(Xtr)
    te_sc = ifm.decision_function(Xte)
    best_f1, best_pred = 0, np.zeros(len(labels), dtype=int)
    for pct in np.arange(1.0, 10.1, 0.5):
        th = np.percentile(tr_sc, pct)
        yp = remove_short((te_sc < th).astype(int), 10)
        r = eval_pred(yp, labels)
        if r["f1"] > best_f1:
            best_f1, best_pred = r["f1"], yp.copy()
    return best_pred


def run_lof_synthetic(train, test, labels, param_idx=0):
    tr1, te1 = train[:, param_idx], test[:, param_idx]
    tr_f = build_features_1d(tr1)
    tail = tr1[-1500:]
    comb = np.concatenate([tail, te1])
    te_f = build_features_1d(comb).iloc[1500:].reset_index(drop=True)
    cols = [c for c in tr_f.columns]
    sc = RobustScaler(quantile_range=(5, 95))
    Xtr = sc.fit_transform(tr_f[cols].fillna(0))
    Xte = sc.transform(te_f[cols].fillna(0))
    Xtr_sub = Xtr[np.sort(np.random.choice(len(Xtr), min(3000, len(Xtr)), replace=False))]
    best_f1, best_pred = 0, np.zeros(len(labels), dtype=int)
    for k in [50, 100, 200]:
        try:
            lof = LocalOutlierFactor(n_neighbors=k, contamination="auto", novelty=True, n_jobs=-1)
            lof.fit(Xtr_sub)
            tr_sc = -lof.score_samples(Xtr)
            te_sc = -lof.score_samples(Xte)
            for pct in np.arange(90.0, 99.6, 1.0):
                yp = remove_short((te_sc > np.percentile(tr_sc, pct)).astype(int), 10)
                r = eval_pred(yp, labels)
                if r["f1"] > best_f1:
                    best_f1, best_pred = r["f1"], yp.copy()
        except Exception:
            pass
    return best_pred


def run_ocsvm_synthetic(train, test, labels, param_idx=0):
    tr1, te1 = train[:, param_idx], test[:, param_idx]
    tr_f = build_features_1d(tr1)
    tail = tr1[-1500:]
    comb = np.concatenate([tail, te1])
    te_f = build_features_1d(comb).iloc[1500:].reset_index(drop=True)
    cols = [c for c in tr_f.columns]
    sc = RobustScaler(quantile_range=(5, 95))
    Xtr = sc.fit_transform(tr_f[cols].fillna(0))
    Xte = sc.transform(te_f[cols].fillna(0))
    Xtr_sub = Xtr[np.sort(np.random.choice(len(Xtr), min(3000, len(Xtr)), replace=False))]
    dists = euclidean_distances(Xtr_sub[:500], Xtr_sub[:500])
    gm = 1.0 / (2.0 * np.median(dists[dists > 0]) ** 2 + 1e-8)
    gammas = [gm * f for f in [0.5, 1.0, 2.0] if gm * f < 100]
    best_f1, best_pred = 0, np.zeros(len(labels), dtype=int)
    for nu in [0.03, 0.05, 0.1]:
        for gamma in gammas:
            try:
                ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
                ocsvm.fit(Xtr_sub)
                tr_sc = -ocsvm.decision_function(Xtr)
                te_sc = -ocsvm.decision_function(Xte)
                for pct in np.arange(90.0, 99.6, 1.0):
                    yp = remove_short((te_sc > np.percentile(tr_sc, pct)).astype(int), 10)
                    r = eval_pred(yp, labels)
                    if r["f1"] > best_f1:
                        best_f1, best_pred = r["f1"], yp.copy()
            except Exception:
                pass
    return best_pred


LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              font=dict(color="#1a1a2e", size=12),
              margin=dict(l=50, r=20, t=40, b=40),
              xaxis=dict(gridcolor="#ddd", showgrid=True, tickfont=dict(color="#555")),
              yaxis=dict(gridcolor="#ddd", showgrid=True, tickfont=dict(color="#555")))


def _layout(**kw):
    import copy
    d = copy.deepcopy(LAYOUT)
    for k, v in kw.items():
        if k in ("xaxis", "yaxis", "title", "legend") and isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k].update(v)
        else:
            d[k] = v
    return d


def make_telemetry_plot(values, labels_arr):
    x = list(range(len(values)))
    anom_x, anom_y = [], []
    for i in range(len(labels_arr)):
        if labels_arr[i] == 1:
            anom_x.append(i)
            anom_y.append(values[i])
    data = [{"x": x, "y": values.tolist(), "mode": "lines",
             "line": {"width": 0.5, "color": "#42a5f5"}, "name": "Telemetry"},
            {"x": anom_x, "y": anom_y, "mode": "markers",
             "marker": {"size": 3, "color": "red", "opacity": 0.4}, "name": "Anomaly"}]
    layout = _layout(height=250, xaxis_title="Time index", yaxis_title="Value",
                     title=dict(text="Telemetry with Ground Truth", font=dict(size=13)))
    return {"data": data, "layout": layout}


def make_channel_barchart(nasa_results):
    data = []
    x = list(range(len(EVAL_CHANNELS)))
    n = len([mk for mk in METHOD_KEYS if mk in nasa_results])
    w = 0.8 / max(n, 1)
    i = 0
    for mk in METHOD_KEYS:
        if mk in nasa_results:
            f1s = [nasa_results[mk]["per_channel"].get(c, {}).get("f1", 0) for c in EVAL_CHANNELS]
            data.append({"x": [xi + i * w for xi in x], "y": f1s, "type": "bar",
                         "width": w, "name": METHOD_LABELS[mk],
                         "marker": {"color": METHOD_COLORS[mk]}})
            i += 1
    layout = _layout(height=350, barmode="group",
                     xaxis=dict(tickvals=[xi + (n - 1) * w / 2 for xi in x], ticktext=EVAL_CHANNELS),
                     yaxis=dict(title="F1 Score", range=[0, 1.05]),
                     title=dict(text="Per-Channel F1 Score", font=dict(size=13)),
                     legend=dict(orientation="h", y=1.1, font=dict(size=9)))
    return {"data": data, "layout": layout}


def make_macro_barchart(nasa_results):
    names, vals, colors = [], [], []
    for mk in METHOD_KEYS:
        if mk in nasa_results:
            names.append(METHOD_LABELS[mk])
            vals.append(nasa_results[mk]["macro_f1"])
            colors.append(METHOD_COLORS[mk])
    data = [{"x": names, "y": vals, "type": "bar", "marker": {"color": colors},
             "text": [f"{v:.1%}" for v in vals], "textposition": "outside",
             "textfont": {"color": "#1a1a2e", "size": 11}}]
    layout = _layout(height=280, yaxis=dict(title="Macro F1", range=[0, 0.85]),
                     title=dict(text="Macro F1 Summary", font=dict(size=13)))
    return {"data": data, "layout": layout}


def make_sim_barchart(sim_df):
    data = []
    for mk, label in METHOD_LABELS.items():
        short = {"LSTM_VAE_IF": "Deep", "LSTM_LOF_IF": "Ensemble"}.get(mk, mk)
        sub = sim_df[sim_df["method"] == short]
        if not sub.empty:
            data.append({"x": [f"S{r['scenario']}" for _, r in sub.iterrows()],
                         "y": sub["f1"].tolist(), "type": "bar", "name": label,
                         "marker": {"color": METHOD_COLORS.get(mk, "#999")}})
    layout = _layout(height=300, barmode="group", yaxis=dict(title="F1 Score", range=[0, 1.05]),
                     title=dict(text="F1 by Scenario", font=dict(size=13)),
                     legend=dict(orientation="h", y=1.1, font=dict(size=9)))
    return {"data": data, "layout": layout}


def make_type_barchart(sim_df):
    types, tlabels = ["drift_f1", "step_f1", "spike_f1"], ["Drift", "Step", "Spike"]
    data = []
    for mk, label in METHOD_LABELS.items():
        short = {"LSTM_VAE_IF": "Deep", "LSTM_LOF_IF": "Ensemble"}.get(mk, mk)
        sub = sim_df[sim_df["method"] == short]
        if not sub.empty:
            means = [sub[t].mean() for t in types]
            data.append({"x": tlabels, "y": means, "type": "bar", "name": label,
                         "marker": {"color": METHOD_COLORS.get(mk, "#999")}})
    layout = _layout(height=280, barmode="group", yaxis=dict(title="Mean F1", range=[0, 0.7]),
                     title=dict(text="F1 by Anomaly Type", font=dict(size=13)),
                     legend=dict(orientation="h", y=1.1, font=dict(size=9)))
    return {"data": data, "layout": layout}


app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
nasa_results = load_nasa_results()
ttd_data = load_ttd()
sim_results = load_sim()
full_eval = {}
_fe_path = os.path.join(REPORTS_DIR, "full_evaluation.json")
if os.path.exists(_fe_path):
    with open(_fe_path) as _f:
        full_eval = json.load(_f)


@app.route("/")
def index():
    ep = [(i, p) for i, p in enumerate(SIM_PARAMS)]
    return render_template_string(HTML, channels=EVAL_CHANNELS, enumerate_params=ep)


@app.route("/data")
def get_data():
    ch = request.args.get("channel")
    sid = request.args.get("scenario", "0")
    pidx = request.args.get("param", "0")
    show_compare = request.args.get("compare", "true") == "true"
    show_ttd = request.args.get("ttd", "true") == "true"

    if ch:
        return jsonify(_nasa_data(ch, show_compare, show_ttd))
    else:
        return jsonify(_sim_data(int(sid), int(pidx), show_compare, show_ttd))


@app.route("/run")
def run_methods():
    sid = int(request.args.get("scenario", "0"))
    pidx = int(request.args.get("param", "0"))
    show_compare = request.args.get("compare", "true") == "true"
    show_ttd = request.args.get("ttd", "true") == "true"

    train = np.load(os.path.join(SIM_DIR, f"scenario_{sid}_train.npy"))
    test = np.load(os.path.join(SIM_DIR, f"scenario_{sid}_test.npy"))
    labels = np.load(os.path.join(SIM_DIR, f"scenario_{sid}_labels.npy"))

    results = {}
    pred_if = run_if_synthetic(train, test, labels, pidx)
    results["IF"] = eval_pred(pred_if, labels)
    pred_lof = run_lof_synthetic(train, test, labels, pidx)
    results["LOF"] = eval_pred(pred_lof, labels)
    pred_ocsvm = run_ocsvm_synthetic(train, test, labels, pidx)
    results["OCSVM"] = eval_pred(pred_ocsvm, labels)

    if not sim_results.empty:
        for mk, short in [("LSTM_VAE_IF", "Deep"), ("LSTM_LOF_IF", "Ensemble")]:
            sub = sim_results[(sim_results["method"] == short) & (sim_results["scenario"] == sid)]
            if not sub.empty:
                rr = sub.iloc[0]
                results[mk] = {"f1": rr["f1"], "p": rr["p"], "r": rr["r"]}

    base = _sim_data(sid, pidx, show_compare, show_ttd)

    headers = ["Method", "F1", "Precision", "Recall"]
    rows, best_mk = [], max(results, key=lambda k: results[k]["f1"])
    for mk in METHOD_KEYS:
        if mk in results:
            r = results[mk]
            rows.append([METHOD_LABELS[mk], f"{r['f1']:.1%}", f"{r['p']:.1%}", f"{r['r']:.1%}"])
    base["run_results"] = {"headers": headers, "rows": rows,
                           "best_idx": list(results.keys()).index(best_mk),
                           "best": f"Best: {METHOD_LABELS[best_mk]} (F1={results[best_mk]['f1']:.1%})"}
    return jsonify(base)


def _nasa_data(ch, show_compare, show_ttd):
    out = {}
    test_path = os.path.join(DATA_DIR, "test", f"{ch}.npy")
    if not os.path.exists(test_path):
        return {"error": f"No data for {ch}"}

    test_arr = np.load(test_path)
    if test_arr.ndim == 1:
        test_arr = test_arr.reshape(-1, 1)
    T = test_arr.shape[0]
    labels_df = pd.read_csv(os.path.join(DATA_DIR, "labeled_anomalies.csv"))
    labels = parse_labels(labels_df, ch, T)
    anom_pct = labels.sum() / T * 100

    out["info"] = {"Channel": ch, "Samples": f"{T:,}", "Anomaly %": f"{anom_pct:.1f}%",
                   "Anomaly points": f"{int(labels.sum()):,}"}
    out["telemetry_plot"] = make_telemetry_plot(test_arr[:, 0], labels)

    ch_data = {}
    for mk in METHOD_KEYS:
        if mk in nasa_results and ch in nasa_results[mk]["per_channel"]:
            ch_data[mk] = nasa_results[mk]["per_channel"][ch]

    fe_ch = full_eval.get("per_channel", {}).get(ch, {})

    if ch_data:
        headers = ["Method", "PW-F1", "PA-F1", "Precision", "Recall"]
        rows, best_mk = [], max(ch_data, key=lambda k: ch_data[k]["f1"])
        for mk, d in ch_data.items():
            p_s = f"{d['p']:.1%}" if d.get("p", 0) > 0 else "-"
            r_s = f"{d['r']:.1%}" if d.get("r", 0) > 0 else "-"
            mk_key = {"LSTM_VAE_IF": "LSTM-VAE", "LSTM_LOF_IF": "LSTM-LOF"}.get(mk, mk)
            pw_s = f"{d['f1']:.1%}"
            pa_s = "-"
            if mk_key in fe_ch:
                pw_s = f"{fe_ch[mk_key].get('pw_f1', d['f1']):.1%}"
                pa_s = f"{fe_ch[mk_key]['pa_f1']:.1%}"
            rows.append([METHOD_LABELS[mk], pw_s, pa_s, p_s, r_s])
        out["method_table"] = {"headers": headers, "rows": rows,
                               "best_idx": list(ch_data.keys()).index(best_mk),
                               "best": f"Best for {ch}: {METHOD_LABELS[best_mk]} (F1={ch_data[best_mk]['f1']:.1%})"}

    if show_ttd and ch in ttd_data:
        headers = ["Method", "Mean TTD", "Median TTD", "Detection Rate"]
        rows = []
        for mk in METHOD_KEYS:
            if mk in ttd_data[ch]:
                td = ttd_data[ch][mk]
                if "mean_ttd" in td:
                    rows.append([METHOD_LABELS[mk], f"{td['mean_ttd']:.0f}",
                                 f"{td['median_ttd']:.0f}", f"{td['detection_rate']:.0%}"])
                else:
                    rows.append([METHOD_LABELS[mk], "N/A", "N/A", "N/A"])
        if rows:
            out["ttd_table"] = {"headers": headers, "rows": rows}

    if show_compare and nasa_results:
        out["channel_plot"] = make_channel_barchart(nasa_results)
        out["macro_plot"] = make_macro_barchart(nasa_results)

        headers = ["Method", "PW Macro F1", "PA Macro F1", "Time/ch (s)"]
        fe_summ = full_eval.get("summary", {})
        fe_order = ["IF", "LOF", "OCSVM", "LSTM-VAE", "LSTM-LOF", "Prophet", "SARIMA"]
        rows, best_pa = [], 0
        for mk_key in fe_order:
            if mk_key not in fe_summ:
                continue
            d = fe_summ[mk_key]
            pw = d.get("pw_macro_f1", 0)
            pa = d.get("pa_macro_f1", 0)
            tc = d.get("mean_time_per_channel_s", 0)
            label = {"LSTM-VAE": "LSTM-VAE + IF", "LSTM-LOF": "LSTM-Forecast + LOF + IF"}.get(mk_key, mk_key)
            rows.append([label, f"{pw:.1%}", f"{pa:.1%}", f"{tc:.1f}"])
            if pa > best_pa:
                best_pa = pa
        out["summary_table"] = {"headers": headers, "rows": rows,
                                "best_idx": max(range(len(rows)), key=lambda i: float(rows[i][2].rstrip('%')) / 100) if rows else 0}

    return out


def _sim_data(sid, pidx, show_compare, show_ttd):
    out = {}
    train = np.load(os.path.join(SIM_DIR, f"scenario_{sid}_train.npy"))
    test = np.load(os.path.join(SIM_DIR, f"scenario_{sid}_test.npy"))
    labels = np.load(os.path.join(SIM_DIR, f"scenario_{sid}_labels.npy"))

    pidx = max(0, min(int(pidx), test.shape[1] - 1))

    with open(os.path.join(SIM_DIR, "scenarios.json")) as f:
        scenarios = json.load(f)
    info = scenarios[sid]
    anom_pct = labels.sum() / len(labels) * 100

    out["info"] = {"Scenario": str(sid), "Parameter": SIM_PARAMS[pidx],
                   "Train": f"{train.shape[0]:,}", "Test": f"{test.shape[0]:,}",
                   "Anomaly %": f"{anom_pct:.1f}%",
                   "Events": str(len(info.get("anomalies", [])))}
    out["telemetry_plot"] = make_telemetry_plot(test[:, pidx], labels)

    if info.get("anomalies"):
        headers = ["Type", "Start", "End", "Length", "Affected Params"]
        rows = [[a["type"].capitalize(), str(a["start"]), str(a["end"]),
                 str(a["length"]), ", ".join(a["affected_params"])]
                for a in info["anomalies"]]
        out["events_table"] = {"headers": headers, "rows": rows}

    if show_compare and not sim_results.empty:
        out["sim_plot"] = make_sim_barchart(sim_results)
        out["type_plot"] = make_type_barchart(sim_results)

    return out


if __name__ == "__main__":
    port = 5000
    threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    print(f"Starting server at http://localhost:{port}")
    app.run(port=port, debug=False)
