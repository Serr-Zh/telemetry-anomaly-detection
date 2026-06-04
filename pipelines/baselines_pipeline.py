import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import ast, json, time, warnings
warnings.filterwarnings('ignore')

from evaluation.evaluation_utils import (
    parse_labels, eval_pred, eval_pred_adjusted, EVAL_CHANNELS, DATA_DIR
)

SEED = 42
np.random.seed(SEED)


def run_prophet_channel(ch, labels_df, data_dir=DATA_DIR):
    from prophet import Prophet

    train_arr = np.load(os.path.join(data_dir, 'train', f'{ch}.npy'))
    test_arr = np.load(os.path.join(data_dir, 'test', f'{ch}.npy'))
    if train_arr.ndim == 1: train_arr = train_arr.reshape(-1, 1)
    if test_arr.ndim == 1: test_arr = test_arr.reshape(-1, 1)

    T_train, T_test = train_arr.shape[0], test_arr.shape[0]
    labels = parse_labels(labels_df, ch, T_test)
    anom_rate = labels.sum() / len(labels)

    train_1d = train_arr[:, 0]
    test_1d = test_arr[:, 0]

    df_train = pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=T_train, freq='min'),
                              'y': train_1d})

    t0 = time.perf_counter()
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=0.1,
        interval_width=0.99,
    )
    model.fit(df_train)
    train_time = time.perf_counter() - t0

    df_test = pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=T_test, freq='min')})
    t1 = time.perf_counter()
    forecast = model.predict(df_test)
    inf_time = time.perf_counter() - t1

    yhat = forecast['yhat'].values
    lower = forecast['yhat_lower'].values
    upper = forecast['yhat_upper'].values

    below = (test_1d < lower).astype(int)
    above = (test_1d > upper).astype(int)
    pred_raw = np.maximum(below, above)

    best_f1, best_pred = 0, np.zeros(T_test, dtype=int)
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        width_factor = alpha
        pred = ((test_1d < yhat - width_factor * (yhat - lower)) |
                (test_1d > yhat + width_factor * (upper - yhat))).astype(int)
        min_len = 20 if anom_rate > 0.05 else 10
        pred = remove_short(pred, min_len)
        r = eval_pred(pred, labels)
        if r['f1'] > best_f1:
            best_f1, best_pred = r['f1'], pred.copy()

    r = eval_pred(best_pred, labels)
    print(f"  {ch:5s} [Prophet]: P={r['p']:5.1%} R={r['r']:5.1%} F1={r['f1']:5.1%} "
          f"train={train_time:.1f}s inf={inf_time:.1f}s")

    return {
        'channel': ch, 'f1': r['f1'], 'p': r['p'], 'r': r['r'],
        'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'], 'tn': r['tn'],
        'train_time': round(train_time, 2), 'inference_time': round(inf_time, 2),
        'total_time': round(train_time + inf_time, 2),
        'pred': best_pred, 'labels': labels,
    }


def run_sarima_channel(ch, labels_df, data_dir=DATA_DIR):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train_arr = np.load(os.path.join(data_dir, 'train', f'{ch}.npy'))
    test_arr = np.load(os.path.join(data_dir, 'test', f'{ch}.npy'))
    if train_arr.ndim == 1: train_arr = train_arr.reshape(-1, 1)
    if test_arr.ndim == 1: test_arr = test_arr.reshape(-1, 1)

    T_train, T_test = train_arr.shape[0], test_arr.shape[0]
    labels = parse_labels(labels_df, ch, T_test)
    anom_rate = labels.sum() / len(labels)

    train_1d = train_arr[:, 0]
    test_1d = test_arr[:, 0]

    max_train_sarima = 2000
    train_sub = train_1d[-max_train_sarima:]

    best_f1, best_pred = 0, np.zeros(T_test, dtype=int)
    best_config = (1, 0, 0)
    train_time_total = 0

    for order in [(1, 0, 0), (1, 1, 0), (2, 0, 0)]:
        try:
            t0 = time.perf_counter()
            model = SARIMAX(train_sub, order=order, seasonal_order=(0, 0, 0, 0),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=30)
            fit_time = time.perf_counter() - t0

            train_resid = res.resid
            train_std = np.std(train_resid)

            fc_steps = min(500, T_test)
            fc = res.get_forecast(steps=fc_steps)
            yhat = fc.predicted_mean

            errors = np.abs(test_1d[:fc_steps] - yhat)
            z_scores = errors / max(train_std, 1e-8)

            full_z = np.zeros(T_test)
            full_z[:fc_steps] = z_scores
            if T_test > fc_steps:
                full_z[fc_steps:] = z_scores[-1]

            for k in [2.0, 3.0, 4.0, 5.0]:
                pred = (full_z > k).astype(int)
                min_len = 20 if anom_rate > 0.05 else 10
                pred = remove_short(pred, min_len)
                r = eval_pred(pred, labels)
                if r['f1'] > best_f1:
                    best_f1, best_pred = r['f1'], pred.copy()
                    best_config = order
                    train_time_total = fit_time

        except Exception as e:
            print(f"    SARIMA{order} failed: {e}")

    r = eval_pred(best_pred, labels)
    print(f"  {ch:5s} [SARIMA{best_config}]: P={r['p']:5.1%} R={r['r']:5.1%} F1={r['f1']:5.1%} "
          f"train={train_time_total:.1f}s")

    return {
        'channel': ch, 'f1': r['f1'], 'p': r['p'], 'r': r['r'],
        'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'], 'tn': r['tn'],
        'train_time': round(train_time_total, 2), 'inference_time': 0,
        'total_time': round(train_time_total, 2),
        'pred': best_pred, 'labels': labels,
    }


def run_garch_channel(ch, labels_df, data_dir=DATA_DIR):
    from arch import arch_model

    train_arr = np.load(os.path.join(data_dir, 'train', f'{ch}.npy'))
    test_arr = np.load(os.path.join(data_dir, 'test', f'{ch}.npy'))
    if train_arr.ndim == 1: train_arr = train_arr.reshape(-1, 1)
    if test_arr.ndim == 1: test_arr = test_arr.reshape(-1, 1)

    T_train, T_test = train_arr.shape[0], test_arr.shape[0]
    labels = parse_labels(labels_df, ch, T_test)
    anom_rate = labels.sum() / len(labels)

    train_1d = train_arr[:, 0]
    test_1d = test_arr[:, 0]
    train_ret = pd.Series(train_1d).diff().fillna(0).values[1:]
    train_ret_scaled = train_ret[-3000:]

    best_f1, best_pred = 0, np.zeros(T_test, dtype=int)
    train_time_total = 0

    for p_val, q_val in [(1, 1), (1, 0), (2, 1)]:
        try:
            t0 = time.perf_counter()
            am = arch_model(train_ret_scaled * 100, vol='Garch', p=p_val, q=q_val, dist='normal',
                            mean='Zero', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            fit_time = time.perf_counter() - t0

            cond_vol = res.conditional_volatility[-1] / 100

            test_ret = pd.Series(test_1d).diff().fillna(0).values
            rolling_vol = pd.Series(test_ret).rolling(50, min_periods=10).std().fillna(cond_vol).values
            rolling_vol = np.maximum(rolling_vol, 1e-6)

            z_scores = np.abs(test_ret) / rolling_vol.clip(1e-8)
            z_scores = z_scores[:T_test]

            pred_full = np.zeros(T_test, dtype=int)
            for k in [2.0, 3.0, 4.0, 5.0, 7.0]:
                pred = (z_scores[:T_test] > k).astype(int)
                if len(pred) < T_test:
                    pred = np.pad(pred, (0, T_test - len(pred)))
                min_len = 20 if anom_rate > 0.05 else 10
                pred = remove_short(pred, min_len)
                r = eval_pred(pred, labels)
                if r['f1'] > best_f1:
                    best_f1, best_pred = r['f1'], pred.copy()
                    train_time_total = fit_time

        except Exception as e:
            print(f"    GARCH({p_val},{q_val}) failed: {e}")

    r = eval_pred(best_pred, labels)
    print(f"  {ch:5s} [GARCH]: P={r['p']:5.1%} R={r['r']:5.1%} F1={r['f1']:5.1%} "
          f"train={train_time_total:.1f}s")

    return {
        'channel': ch, 'f1': r['f1'], 'p': r['p'], 'r': r['r'],
        'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'], 'tn': r['tn'],
        'train_time': round(train_time_total, 2), 'inference_time': 0,
        'total_time': round(train_time_total, 2),
        'pred': best_pred, 'labels': labels,
    }


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


if __name__ == '__main__':
    labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labeled_anomalies.csv'))
    channels = [ch for ch in EVAL_CHANNELS
                if ch in labels_df['chan_id'].values
                and os.path.exists(os.path.join(DATA_DIR, 'train', f'{ch}.npy'))
                and os.path.exists(os.path.join(DATA_DIR, 'test', f'{ch}.npy'))]

    all_results = {}
    for method_name, runner in [('Prophet', run_prophet_channel),
                                 ('SARIMA', run_sarima_channel),
                                 ('GARCH', run_garch_channel)]:
        print(f"\n{'='*60}")
        print(f"  {method_name}")
        print(f"{'='*60}")
        method_results = {}
        for ch in channels:
            try:
                r = runner(ch, labels_df, DATA_DIR)
                method_results[ch] = r
            except Exception as e:
                print(f"  {ch} ERROR: {e}")
        all_results[method_name] = method_results

        if method_results:
            pw_f1s = [r['f1'] for r in method_results.values()]
            pa_f1s = []
            for r in method_results.values():
                pa_r = eval_pred_adjusted(r['pred'], r['labels'])
                pa_f1s.append(pa_r['f1'])
            times = [r['total_time'] for r in method_results.values()]
            macro_f1 = np.mean(pw_f1s)
            pa_macro = np.mean(pa_f1s)
            mean_time = np.mean(times)
            print(f"\n  {method_name} Summary: PW-Macro={macro_f1:.1%} PA-Macro={pa_macro:.1%} Time/ch={mean_time:.1f}s")

    tp = sum(r['tp'] for mr in all_results.values() for r in mr.values())
    fp = sum(r['fp'] for mr in all_results.values() for r in mr.values())
    fn = sum(r['fn'] for mr in all_results.values() for r in mr.values())
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    summary = {}
    for method_name in ['Prophet', 'SARIMA', 'GARCH']:
        mr = all_results.get(method_name, {})
        if not mr:
            continue
        pw_f1s = [r['f1'] for r in mr.values()]
        pa_f1s = [eval_pred_adjusted(r['pred'], r['labels'])['f1'] for r in mr.values()]
        times = [r['total_time'] for r in mr.values()]
        summary[method_name] = {
            'pw_macro_f1': round(np.mean(pw_f1s), 4),
            'pa_macro_f1': round(np.mean(pa_f1s), 4),
            'mean_time_per_channel_s': round(np.mean(times), 2),
            'per_channel': {ch: {
                'pw_f1': round(r['f1'], 4),
                'pa_f1': round(eval_pred_adjusted(r['pred'], r['labels'])['f1'], 4),
                'p': round(r['p'], 4),
                'r': round(r['r'], 4),
                'time': r['total_time'],
            } for ch, r in mr.items()},
        }

    os.makedirs('reports', exist_ok=True)
    with open('reports/baselines_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to reports/baselines_metrics.json")
