# Satellite Telemetry Anomaly Detector

Детекция аномалий в телеметрии космических аппаратов методами обучения без учителя. Проект реализован как прототип дипломной работы и включает 5 методов обнаружения аномалий, 2 классических базовых метода, гибридный ансамбль, а также веб-интерфейс на Flask для визуализации и сравнения результатов.

---

## Содержание

- [Постановка задачи](#постановка-задачи)
- [Методы обнаружения аномалий](#методы-обнаружения-аномалий)
- [Результаты](#результаты)
- [Структура проекта](#структура-проекта)
- [Установка и запуск](#установка-и-запуск)
- [Веб-приложение](#веб-приложение)
- [Запуск пайплайнов](#запуск-пайплайнов)
- [Данные](#данные)
- [Метрики оценки](#метрики-оценки)
- [Литература](#литература)

---

## Постановка задачи

Космические аппараты передают телеметрические данные (напряжение батарей, токи солнечных панелей, температуры, давление и др.), которые необходимо анализировать для своевременного выявления аномального поведения подсистем. Задача осложняется тем, что:

- Разметка аномалий скудная — обучение без учителя
- Аномалии редки — сильный дисбаланс классов
- Телеметрия нестационарна — сезонность, тренды, переключения режимов
- Требуется быстрая реакция — важна не только точность, но и время обнаружения

---

## Методы обнаружения аномалий

### Классические ML-методы

| Метод | Описание | Контаминация | Особенности |
|-------|----------|--------------|-------------|
| **Isolation Forest (IF)** | Изоляция аномалий через случайные разбиения | Автоподбор по порогу | Per-channel threshold tuning, combined PW+PA scoring |
| **Local Outlier Factor (LOF)** | Локальная плотность относительно соседей | 5% | Rolling window 50, k=20 |
| **One-Class SVM (OCSVM)** | Гиперплоскость, отделяющая норму от аномалий | 5% | RBF kernel, nu=0.05 |

### Глубинные методы

| Метод | Описание | Архитектура |
|-------|----------|-------------|
| **LSTM-VAE + IF** | Вариационный автоэнкодер на LSTM для реконструкции, остатки анализируются IF | LSTM encoder (40 эпох) + LSTM decoder (60 эпох), latent=8 |
| **LSTM-Forecast + LOF + IF** | LSTM-предиктор прогнозирует следующий шаг, ошибки прогноза + raw residuals анализируются LOF, дополнительно IF на окнах | LSTM forecaster (80 эпох), LOF на ошибках, IF на признаках, выбор лучшего |

### Классические базовые методы

| Метод | Описание |
|-------|----------|
| **Prophet** | Аддитивная модель Facebook Prophet с детекцией по доверительному интервалу |
| **SARIMA** | Сезонная ARIMA с детекцией по остаткам |

---

## Результаты

### NASA SMAP/MSL (8 каналов)

**Point-Wise F1 (строгая оценка, точное совпадение точек):**

| Метод | PW Macro F1 | PA Macro F1 | Time/ch (s) |
|-------|-------------|-------------|-------------|
| Isolation Forest | 35.5% | 82.3% | 0.5 |
| LOF | 65.8% | 86.6% | 1.8 |
| OCSVM | 65.0% | 85.6% | 7.9 |
| LSTM-VAE + IF | 50.3% | 73.4% | 56.3 |
| **LSTM-LOF + IF** | **71.5%** | **92.8%** | 59.9 |
| Prophet | 25.8% | 39.6% | 1.4 |
| SARIMA | 14.6% | 14.6% | 0.1 |

- **PA (Point-Adjust) F1** — стандартная метрика NASA SMAP/MSL (Hundman et al., KDD 2018): если хотя бы одна точка аномального сегмента обнаружена, весь сегмент считается найденным.
- **PW (Point-Wise) F1** — строгая оценка, требует точного совпадения каждой точки.
- LSTM-LOF + IF показывает лучший результат по обеим метрикам среди одиночных методов.
- Классические ML-методы (LOF, OCSVM) значительно превосходят Prophet/SARIMA при малом времени работы.

### Ансамбль (union/any strategy)

| Стратегия | PW Macro F1 | PA Macro F1 | Time/ch (s) |
|-----------|-------------|-------------|-------------|
| Ensemble (any) | 74.6% | 96.3% | 101.2 |

Объединение предсказаний всех методов (аномалия = хотя бы один метод детектировал) даёт максимальный PA-F1 = 96.3%.

### Синтетические данные

| Метод | Macro F1 |
|-------|----------|
| LSTM-VAE + IF (Deep) | 74.9% |
| LOF | 36.3% |
| IF | 19.1% |

### Time-to-Detection

| Метод | Mean TTD | Detection Rate |
|-------|----------|----------------|
| LOF | 44 | 93.8% |
| OCSVM | 47 | 93.8% |

---

## Структура проекта

```
telemetry-anomaly-detection/
├── app/                            ← Flask веб-приложение
│   └── app.py
├── pipelines/                      ← Пайплайны обнаружения аномалий
│   ├── if_pipeline.py              ← Isolation Forest
│   ├── lof_pipeline.py             ← Local Outlier Factor
│   ├── ocsvm_pipeline.py           ← One-Class SVM
│   ├── lstm_vae_pipeline.py        ← LSTM-VAE + IF
│   ├── lstm_lof_pipeline.py        ← LSTM-Forecast + LOF + IF
│   └── baselines_pipeline.py       ← Prophet / SARIMA / GARCH
├── evaluation/                     ← Утилиты оценки и запуска
│   ├── evaluation_utils.py         ← PA-F1, TTD, интерпретируемость
│   ├── run_full_eval.py            ← Полная оценка + ансамбль
│   ├── run_simulated.py            ← Оценка на синтетических данных
│   ├── eval_ttd_interp.py          ← TTD + интерпретируемость
│   └── generate_simulated.py       ← Генератор синтетических данных
│
├── data/                           ← NASA SMAP/MSL данные
│   ├── train/                      ← Обучающая выборка (норма)
│   ├── test/                       ← Тестовая выборка (с аномалиями)
│   ├── labeled_anomalies.csv       ← Разметка аномальных сегментов
│   └── 2018-05-19_15.00.10/        ← Предобученные модели Hundman et al.
│
├── data_simulated/             ← Синтетические данные (3 сценария)
│   ├── scenario_0_*.npy
│   ├── scenario_1_*.npy
│   ├── scenario_2_*.npy
│   └── scenarios.json
│
├── reports/                    ← Результаты оценки
│   ├── full_evaluation.json    ← Полные результаты (PW, PA, timing, per-channel)
│   ├── baselines_metrics.json  ← Prophet/SARIMA/GARCH
│   ├── if_metrics.json
│   ├── lof_metrics.json
│   ├── ocsvm_metrics.json
│   ├── lstm_vae_metrics.json
│   ├── lstm_lof_metrics.json
│   ├── simulated_results.csv
│   └── ttd_interpretability.json
│
└── notebooks/                  ← Jupyter notebooks по каждому методу
    ├── IF_NASA_SMAP_MSL.ipynb
    ├── LOF_NASA_SMAP_MSL.ipynb
    ├── OCSVM_NASA_SMAP_MSL.ipynb
    ├── LSTM_VAE_NASA_SMAP_MSL.ipynb
    └── LSTM_LOF_NASA_SMAP_MSL.ipynb
```

---

## Установка и запуск

### Требования

- Python 3.9+
- CPU (GPU опционально для ускорения LSTM)

### Установка

```bash
git clone https://github.com/<username>/telemetry-anomaly-detection.git
cd telemetry-anomaly-detection
pip install -r requirements.txt
```

### Запуск веб-приложения

```bash
python -m app.app
```

Откроется браузер: `http://localhost:5000`

### Запуск полной оценки

```bash
python -m evaluation.run_full_eval
```

Выполняет все методы на 8 каналах NASA SMAP/MSL, сохраняет результаты в `reports/full_evaluation.json`.

---

## Веб-приложение

Flask-приложение (`diploma_app.py`) предоставляет:

- **Вкладка NASA SMAP/MSL:**
  - Выбор канала телеметрии (P-1, S-1, E-1, E-2, F-1, G-1, D-1, M-5)
  - Визуализация телеметрии с подсветкой аномалий
  - Per-channel таблица: PW-F1, PA-F1, Precision, Recall
  - Сравнительная таблица всех методов: PW Macro F1, PA Macro F1, Time/ch
  - Bar-charts: per-channel F1, macro F1
  - Time-to-Detection таблица

- **Вкладка Synthetic Data:**
  - Выбор сценария (0, 1, 2) и телеметрического параметра
  - Real-time IF/LOF/OCSVM детекция
  - Breakdown по типам аномалий (drift, step, spike)

---

## Запуск пайплайнов

```bash
python -m pipelines.if_pipeline           # Isolation Forest
python -m pipelines.lof_pipeline           # LOF
python -m pipelines.ocsvm_pipeline         # OCSVM
python -m pipelines.lstm_vae_pipeline      # LSTM-VAE + IF
python -m pipelines.lstm_lof_pipeline      # LSTM-Forecast + LOF + IF
python -m pipelines.baselines_pipeline     # Prophet, SARIMA, GARCH
python -m evaluation.run_simulated         # Оценка на синтетических данных
python -m evaluation.eval_ttd_interp       # TTD + интерпретируемость
```

Каждый пайплайн сохраняет метрики в `reports/`.

---

## Данные

### NASA SMAP/MSL

Бенчмарк от Hundman et al. (KDD 2018). Содержит телеметрию двух космических аппаратов:

| Канал | Аппарат | Подсистема | Датчик |
|-------|---------|-----------|--------|
| P-1 | SMAP | Энергоснабжение | battery_voltage |
| S-1 | SMAP | Солнечные панели | solar_current |
| E-1 | SMAP | Термоконтроль (внутр.) | temperature_int |
| E-2 | SMAP | Термоконтроль (внеш.) | temperature_ext |
| F-1 | SMAP | Система ориентации | attitude_error |
| G-1 | SMAP | Двигательная установка | thruster_pressure |
| D-1 | SMAP | Обработка данных | data_rate |
| M-5 | MSL | Вычислительная система | cpu_load |

- `data/train/` — нормальная телеметрия (без аномалий)
- `data/test/` — тестовая телеметрия (с аномалиями)
- `data/labeled_anomalies.csv` — разметка аномальных сегментов

### Синтетические данные

Генерируются `generate_simulated.py`. 3 сценария с уровнем аномальности 5%, 10%, 15%. 8 телеметрических параметров. Типы аномалий:

| Тип | Описание |
|-----|----------|
| Drift | Постепенное отклонение параметра |
| Step | Внезапный сдвиг уровня |
| Spike | Кратковременный выброс |

---

## Метрики оценки

### Point-Wise F1 (PW-F1)

Классический F1-score: точное совпадение «аномалия/норма» в каждой точке. Строгая оценка — штрафует за неточное определение границ аномалии.

### Point-Adjust F1 (PA-F1)

Метрика, принятая в NASA SMAP/MSL бенчмарке (Hundman et al., KDD 2018). Если хотя бы одна точка аномального сегмента была обнаружена, весь сегмент считается найденным верно. Более мягкая оценка, отражающая практическую значимость: оператору достаточно одного сигнала для начала расследования.

### Time-to-Detection (TTD)

Среднее количество отсчётов от начала аномального сегмента до первого обнаружения. Меньше = быстрее реакция.

### Detection Rate

Доля аномальных сегментов, в которых метод обнаружил хотя бы одну точку.

---

## Литература

1. Hundman K., Constantinou V., Laporte C., Colwell I., Soderstrom T. **Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding** // KDD 2018. [arXiv:1802.04431](https://arxiv.org/abs/1802.04431)
2. Liu F.T., Ting K.M., Zhou Z.H. **Isolation Forest** // ICDM 2008.
3. Breunig M.M., Kriegel H.P., Ng R.T., Sander J. **LOF: Identifying Density-Based Local Outliers** // SIGMOD 2000.
4. Scholkopf B., Platt J.C., Shawe-Taylor J. et al. **Estimating the Support of a High-Dimensional Distribution** // Neural Computation, 2001.
5. Taylor S.J., Letham B. **Forecasting at Scale** // The American Statistician, 2018. (Prophet)
