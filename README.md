# ICU Bed Decision Support

Forecasting ICU bed availability and shortage risk, with an interactive dashboard for capacity planning.

The project forecasts **available adult and pediatric ICU beds** 7, 14 and 28 days ahead. It shows the expected value, a likely low–high range (p10–p90) and the chance that availability falls below a safe threshold.

> **Disclaimer:** decision support only. Confirm with local policy and real-time context.

---

## Repository contents

```
icu-bed-decision-support/
├── icu_dashboard_v2.py        # Dash dashboard (decision support UI)
├── train_models.py            # Trains the deep learning models (DeepAR, TFT, PatchTST)
├── icu_beds.csv               # Daily ICU bed data (Ontario open data)
├── requirements.txt           # Python packages for the dashboard and models
├── artifacts/
│   ├── models/                # Saved model settings (deepar, tft, patchtst)
│   └── outputs/               # Forecasts and backtest metrics read by the dashboard
│       ├── test_forecast_quantiles_7d.csv
│       ├── test_forecast_quantiles_14d.csv
│       ├── test_forecast_quantiles_28d.csv
│       └── backtest_metrics.csv
└── traditional_models/        # Traditional baselines (Seasonal Naive, ARIMA, XGBoost)
    ├── ICU-Traditional.py     # Streamlit app that runs the baselines
    ├── icu_beds.csv
    └── artifacts/outputs/     # Baseline metrics and forecasts
```

---

## Data

Source: Government of Ontario Data Catalogue, *Availability of adult ICU beds and occupancy for COVID-related critical illness (CRCI)*
https://data.ontario.ca/dataset/availability-of-adult-icu-beds-and-occupancy-for-covid-related-critical-illness-crci

The data is aggregated daily counts. It contains no individual-level patient information.

The dashboard needs these columns in `icu_beds.csv`:

`date`, `available_adult_icu_beds`, `total_adult_icu_patients`, `total_adult_icu_beds`, `available_ped_icu_beds`, `total_ped_icu_patients`, `total_ped_icu_beds`

---

## Setup

Python 3.11 was used.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

---

## 1. Run the dashboard

```bash
python icu_dashboard_v2.py
```

Then open http://127.0.0.1:8050 (the browser opens automatically).

The dashboard reads `icu_beds.csv` and the files in `artifacts/outputs/` from the same folder as the script.

**Main features**

- **Today's status:** available beds and occupancy for adult and pediatric ICU.
- **Forecast charts:** past data, today and the forecast, with the p10–p90 uncertainty band and the shortage threshold line.
- **Action Board (next 72 hours):** expected beds, shortage risk, confidence and an action label (OK / WATCH / ESCALATE).
- **Confidence:** based on how much the models disagree and how wide the uncertainty band is.
- **Scenario builder:** what-if tests for extra adult or pediatric beds and a higher safety buffer, with one-click presets.
- **Shortage outlook tables:** day-by-day expected beds, likely range, shortage chance and risk level.
- **Model & Metrics tab:** backtest MAE and RMSE per model and horizon.
- **Data Quality tab:** missing values and negative counts in the raw data.
- **CSV downloads:** forecast, risk and metrics.

**How shortage risk is calculated**

Shortage risk is the probability that available beds fall below the threshold τ (default: adult 200, pediatric 10). It uses a normal approximation from the forecast quantiles:

```
σ ≈ (p90 − p10) / (2 × 1.2816)
z = (τ − p50) / σ
risk = Φ(z)
```

Risk levels: **LOW** < 5% (OK), **MEDIUM** 5–20% (WATCH), **HIGH** ≥ 20% (ESCALATE).

The default forecast is **EnsembleMean**, the average of DeepAR, TFT and PatchTST. Each individual model can also be viewed.

---

## 2. Train the deep learning models

```bash
python train_models.py
```

This creates the forecast and metrics files in `artifacts/outputs/` that the dashboard uses.

---

## 3. Run the traditional baselines

The baselines are **Seasonal Naive** (weekly), **ARIMA** (SARIMAX with weekly seasonality) and **XGBoost** (lag and calendar features, recursive multi-step forecast).

```bash
cd traditional_models
pip install streamlit numpy pandas scikit-learn xgboost statsmodels matplotlib
streamlit run ICU-Traditional.py
```

**Before running**, open `ICU-Traditional.py` and set the two paths at the top to your own folder:

```python
CSV_PATH = r"C:\path\to\traditional_models\icu_beds.csv"
OUT_DIR = Path(r"C:\path\to\traditional_models\artifacts\outputs")
```

The app holds out the last 28 days as a test set (adjustable). It reports MAE and RMSE at 7, 14 and 28 days and saves:

- `backtest_metrics_traditional.csv`
- `test_forecast_traditional_28d.csv`

---

## Funding

Supported by Research Nova Scotia (Grant No. RNS-NHIG-2021-1968) and partially by an NSERC Discovery Grant (RGPIN-2020-05869).
