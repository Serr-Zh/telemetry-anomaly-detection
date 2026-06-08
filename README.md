# Telemetry Anomaly Detection

Unsupervised anomaly detection in spacecraft telemetry using IF, LOF, OCSVM, LSTM-VAE and LSTM-LOF hybrid — evaluated on NASA SMAP/MSL benchmark with Flask web dashboard.

---

## Methods

### Classical ML

| Method | Description | Key settings |
|--------|-------------|--------------|
| **Isolation Forest (IF)** | Isolation via random splits | Per-channel threshold tuning (PW+PA scoring) |
| **Local Outlier Factor (LOF)** | Local density vs. neighbors | Rolling window 50, k=20 |
| **One-Class SVM (OCSVM)** | Hyperplane separating normal from anomaly | RBF kernel, nu=0.05 |

### Deep Learning

| Method | Description | Architecture |
|--------|-------------|--------------|
| **LSTM-VAE + IF** | Variational autoencoder on LSTM, residuals analyzed by IF | Encoder 40 epochs + Decoder 60 epochs, latent=8 |
| **LSTM-Forecast + LOF + IF** | LSTM forecaster predicts next step, forecast errors + raw residuals analyzed by LOF, IF on features, best selected | Forecaster 80 epochs, LOF on errors, IF on features |

### Classical Baselines

| Method | Description |
|--------|-------------|
| **Prophet** | Facebook Prophet additive model with confidence interval detection |
| **SARIMA** | Seasonal ARIMA with residual-based detection |

---

## Results

### NASA SMAP/MSL (8 channels)

| Method | PW Macro F1 | PA Macro F1 | Time/ch (s) |
|--------|-------------|-------------|-------------|
| Isolation Forest | 35.5% | 82.3% | 0.5 |
| LOF | 65.8% | 86.6% | 1.8 |
| OCSVM | 65.0% | 85.6% | 7.9 |
| LSTM-VAE + IF | 50.3% | 73.4% | 56.3 |
| **LSTM-LOF + IF** | **71.5%** | **92.8%** | 59.9 |
| Prophet | 25.8% | 39.6% | 1.4 |
| SARIMA | 14.6% | 14.6% | 0.1 |

- **PA (Point-Adjust) F1** — NASA SMAP/MSL standard (Hundman et al., KDD 2018): if any point of an anomaly segment is detected, the whole segment counts as found.
- **PW (Point-Wise) F1** — strict: exact point-level matching.
- LSTM-LOF + IF achieves best results among single methods on both metrics.
- Classical ML (LOF, OCSVM) significantly outperforms Prophet/SARIMA with low runtime.

### Ensemble (union/any)

| Strategy | PW Macro F1 | PA Macro F1 | Time/ch (s) |
|----------|-------------|-------------|-------------|
| Ensemble (any) | 74.6% | 96.3% | 101.2 |

Union of all methods (anomaly = at least one method flagged) yields PA-F1 = 96.3%.

### Synthetic Data

| Method | Macro F1 |
|--------|----------|
| LSTM-VAE + IF (Deep) | 74.9% |
| LOF | 36.3% |
| IF | 19.1% |

### Time-to-Detection

| Method | Mean TTD | Detection Rate |
|--------|----------|----------------|
| LOF | 44 | 93.8% |
| OCSVM | 47 | 93.8% |

---

## Project Structure

```
telemetry-anomaly-detection/
├── app/                            ← Flask web application
│   ├── app.py                      ← Main Flask app
│   └── static/
│       └── plotly.min.js           ← Local Plotly (no CDN required)
├── pipelines/                      ← Anomaly detection pipelines
│   ├── if_pipeline.py              ← Isolation Forest
│   ├── lof_pipeline.py             ← Local Outlier Factor
│   ├── ocsvm_pipeline.py           ← One-Class SVM
│   ├── lstm_vae_pipeline.py        ← LSTM-VAE + IF
│   ├── lstm_lof_pipeline.py        ← LSTM-Forecast + LOF + IF
│   └── baselines_pipeline.py       ← Prophet / SARIMA / GARCH
├── evaluation/                     ← Evaluation utilities and runners
│   ├── evaluation_utils.py         ← PA-F1, TTD, interpretability
│   ├── run_full_eval.py            ← Full evaluation + ensemble
│   ├── run_simulated.py            ← Synthetic data evaluation
│   ├── eval_ttd_interp.py          ← TTD + interpretability
│   └── generate_simulated.py       ← Synthetic data generator
│
├── data/                           ← NASA SMAP/MSL data
│   ├── train/                      ← Normal telemetry (no anomalies)
│   ├── test/                       ← Test telemetry (with anomalies)
│   ├── labeled_anomalies.csv       ← Anomaly segment labels
│   └── 2018-05-19_15.00.10/        ← Pre-trained models (Hundman et al.)
│
├── data_simulated/                 ← Synthetic telemetry (3 scenarios)
│   ├── scenario_*_train.npy
│   ├── scenario_*_test.npy
│   ├── scenario_*_labels.npy
│   └── scenarios.json
│
├── reports/                        ← Evaluation results
│   ├── full_evaluation.json        ← Full results (PW, PA, timing, per-channel)
│   ├── baselines_metrics.json      ← Prophet/SARIMA/GARCH
│   ├── if_metrics.json
│   ├── lof_metrics.json
│   ├── ocsvm_metrics.json
│   ├── lstm_vae_metrics.json
│   ├── lstm_lof_metrics.json
│   ├── simulated_results.csv
│   └── ttd_interpretability.json
│
├── notebooks/                      ← Jupyter notebooks per method
│   ├── IF_NASA_SMAP_MSL.ipynb
│   ├── LOF_NASA_SMAP_MSL.ipynb
│   ├── OCSVM_NASA_SMAP_MSL.ipynb
│   ├── LSTM_VAE_NASA_SMAP_MSL.ipynb
│   └── LSTM_LOF_NASA_SMAP_MSL.ipynb
│
├── requirements.txt
├── .gitignore
└── readme.md
```

---

## Setup & Run

### Requirements

- Python 3.9+
- CPU (GPU optional for LSTM acceleration)

### Install

```bash
git clone https://github.com/<username>/telemetry-anomaly-detection.git
cd telemetry-anomaly-detection
pip install -r requirements.txt
```

### Run Web App

```bash
python -m app.app
```

Opens browser at `http://localhost:5000`. Plotly is served locally — no internet required.

### Run Full Evaluation

```bash
python -m evaluation.run_full_eval
```

Runs all methods on 8 NASA SMAP/MSL channels, saves results to `reports/full_evaluation.json`.

---

## Web Application

The Flask app (`app/app.py`) provides:

**NASA SMAP/MSL tab:**
- Channel selection (P-1, S-1, E-1, E-2, F-1, G-1, D-1, M-5)
- Telemetry visualization with anomaly overlay
- Per-channel table: PW-F1, PA-F1, Precision, Recall
- Summary table: PW Macro F1, PA Macro F1, Time/ch for all 7 methods
- Bar charts: per-channel F1, macro F1
- Time-to-Detection table

**Synthetic Data tab:**
- Scenario selection (5%, 10%, 15% anomaly rate)
- Parameter selection (8 telemetry parameters, charts update on switch)
- Real-time IF/LOF/OCSVM detection with "Run Methods" button
- Anomaly event table with type, affected parameters
- F1 by scenario and anomaly type bar charts

---

## Run Pipelines Individually

```bash
python -m pipelines.if_pipeline           # Isolation Forest
python -m pipelines.lof_pipeline           # LOF
python -m pipelines.ocsvm_pipeline         # OCSVM
python -m pipelines.lstm_vae_pipeline      # LSTM-VAE + IF
python -m pipelines.lstm_lof_pipeline      # LSTM-Forecast + LOF + IF
python -m pipelines.baselines_pipeline     # Prophet, SARIMA, GARCH
python -m evaluation.run_simulated         # Synthetic evaluation
python -m evaluation.eval_ttd_interp       # TTD + interpretability
python -m evaluation.generate_simulated    # Generate synthetic data
```

Each pipeline saves metrics to `reports/`.

---

## Data

### NASA SMAP/MSL

Benchmark from Hundman et al. (KDD 2018). Telemetry from two spacecraft:

| Channel | Spacecraft | Subsystem | Sensor |
|---------|-----------|-----------|--------|
| P-1 | SMAP | Power | battery_voltage |
| S-1 | SMAP | Solar Array | solar_current |
| E-1 | SMAP | Thermal (internal) | temperature_int |
| E-2 | SMAP | Thermal (external) | temperature_ext |
| F-1 | SMAP | Attitude Control | attitude_error |
| G-1 | SMAP | Propulsion | thruster_pressure |
| D-1 | SMAP | Data Handling | data_rate |
| M-5 | MSL | CPU | cpu_load |

- `data/train/` — normal telemetry
- `data/test/` — telemetry with anomalies
- `data/labeled_anomalies.csv` — anomaly segment labels

### Synthetic Data

Generated by `evaluation/generate_simulated.py`. 3 scenarios with 5%, 10%, 15% anomaly contamination across 8 telemetry parameters. Anomaly types:

| Type | Description |
|------|-------------|
| Drift | Gradual parameter deviation |
| Step | Sudden offset change |
| Spike | Short-lived extreme value |

---

## Evaluation Metrics

### Point-Wise F1 (PW-F1)

Classic F1-score: exact "anomaly/normal" match at each point. Strict evaluation — penalizes imprecise anomaly boundaries.

### Point-Adjust F1 (PA-F1)

NASA SMAP/MSL benchmark standard (Hundman et al., KDD 2018). If at least one point of an anomaly segment is detected, the entire segment counts as correctly found. Softer evaluation reflecting practical significance: a single alert is enough for an operator to start investigation.

### Time-to-Detection (TTD)

Mean number of time steps from anomaly segment start to first detection. Lower = faster reaction.

### Detection Rate

Fraction of anomaly segments where the method detected at least one point.

---

## References

1. Hundman K., Constantinou V., Laporte C., Colwell I., Soderstrom T. **Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding** // KDD 2018. [arXiv:1802.04431](https://arxiv.org/abs/1802.04431)
2. Liu F.T., Ting K.M., Zhou Z.H. **Isolation Forest** // ICDM 2008.
3. Breunig M.M., Kriegel H.P., Ng R.T., Sander J. **LOF: Identifying Density-Based Local Outliers** // SIGMOD 2000.
4. Scholkopf B., Platt J.C., Shawe-Taylor J. et al. **Estimating the Support of a High-Dimensional Distribution** // Neural Computation, 2001.
5. Taylor S.J., Letham B. **Forecasting at Scale** // The American Statistician, 2018. (Prophet)
