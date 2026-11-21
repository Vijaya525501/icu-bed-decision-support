# streamlit_traditional_baselines_ui.py
# Simple Streamlit UI (file path is hard-coded in code itself)
#
# Run:
#   pip install streamlit numpy pandas scikit-learn xgboost statsmodels matplotlib
#   streamlit run streamlit_traditional_baselines_ui.py

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# HARD-CODED PATHS (EDIT THESE)
# =========================================================
CSV_PATH = r"C:\ICU-Traditional\icu_beds.csv"  # <-- put your CSV path here
OUT_DIR = Path(r"C:\ICU-Traditional\artifacts\outputs")  # <-- outputs saved here


# -------------------------
# DEFAULTS
# -------------------------
DEFAULT_TEST_DAYS = 28
DEFAULT_HORIZONS = [7, 14, 28]
DEFAULT_SEASONAL_PERIOD = 7
DEFAULT_MAX_LAG = 28

DEFAULT_EVAL_SERIES = [
    "available_adult_icu_beds",
    "available_ped_icu_beds",
]


# -------------------------
# DATA LOADING + CLEAN
# -------------------------
def load_wide_csv_from_df(df: pd.DataFrame, eval_series: List[str]) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")

    for c in eval_series:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[eval_series] = df[eval_series].interpolate(limit_direction="both").ffill().bfill()
    return df.reset_index(drop=True)


# -------------------------
# BASELINE 1: Seasonal Naive (weekly)
# -------------------------
def seasonal_naive_forecast(y: np.ndarray, steps: int, season: int = 7) -> np.ndarray:
    y = list(map(float, y))
    if len(y) == 0:
        return np.full(steps, np.nan)
    if len(y) < season:
        return np.array([y[-1]] * steps, dtype=float)

    last_season = y[-season:]
    out = [last_season[i % season] for i in range(steps)]
    return np.array(out, dtype=float)


# -------------------------
# BASELINE 2: ARIMA (SARIMAX weekly)
# -------------------------
def arima_forecast(y: np.ndarray, steps: int, season: int = 7) -> np.ndarray:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        # fallback if statsmodels missing
        return seasonal_naive_forecast(y, steps, season=season)

    y = pd.Series(y).astype(float)
    try:
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, season),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=steps).predicted_mean
        return np.array(fc, dtype=float)
    except Exception:
        # fallback: non-seasonal
        try:
            model = SARIMAX(
                y,
                order=(1, 1, 1),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            fc = res.get_forecast(steps=steps).predicted_mean
            return np.array(fc, dtype=float)
        except Exception:
            return seasonal_naive_forecast(y.values, steps, season=season)


# -------------------------
# BASELINE 3: XGBoost (lag + calendar, recursive multi-step)
# -------------------------
def make_supervised(y: np.ndarray, dates: pd.Series, max_lag: int = 28) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    d = pd.DataFrame({"date": pd.to_datetime(dates)})
    d["dow"] = d["date"].dt.dayofweek.astype(int)
    d["month"] = d["date"].dt.month.astype(int)
    d["doy"] = d["date"].dt.dayofyear.astype(int)

    s = pd.Series(y).astype(float)
    for lag in range(1, max_lag + 1):
        d[f"lag{lag}"] = s.shift(lag)

    d["y"] = s.values
    d = d.dropna().reset_index(drop=True)

    X = d.drop(columns=["y", "date"])
    y_target = d["y"].values.astype(float)
    return X, y_target, list(X.columns)


def xgb_recursive_forecast(
    y: np.ndarray,
    dates: pd.Series,
    steps: int,
    max_lag: int = 28,
    seasonal_period: int = 7,
) -> np.ndarray:
    try:
        from xgboost import XGBRegressor
    except Exception:
        # fallback if xgboost missing
        return seasonal_naive_forecast(y, steps, season=seasonal_period)

    X_train, y_train, feat_cols = make_supervised(y, dates, max_lag=max_lag)
    if len(y_train) < 50:
        return seasonal_naive_forecast(y, steps, season=seasonal_period)

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=1,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    hist = list(map(float, y))
    last_date = pd.to_datetime(dates).iloc[-1]

    out = []
    for s in range(1, steps + 1):
        d = last_date + pd.Timedelta(days=s)
        row = {"dow": int(d.dayofweek), "month": int(d.month), "doy": int(d.dayofyear)}
        for lag in range(1, max_lag + 1):
            row[f"lag{lag}"] = hist[-lag] if len(hist) >= lag else hist[0]

        X_pred = pd.DataFrame([row])[feat_cols]
        yhat = float(model.predict(X_pred)[0])
        out.append(yhat)
        hist.append(yhat)

    return np.array(out, dtype=float)


# -------------------------
# CORE RUN
# -------------------------
def run_baselines(
    df_raw: pd.DataFrame,
    eval_series: List[str],
    test_days: int,
    horizons: List[int],
    seasonal_period: int,
    max_lag: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, np.ndarray]], pd.DataFrame, pd.DataFrame]:
    df = load_wide_csv_from_df(df_raw, eval_series)

    n = len(df)
    min_needed = test_days + max_lag + 10
    if n < min_needed:
        raise ValueError(f"Not enough rows. Need at least ~{min_needed} rows, found {n}.")

    # split: last TEST_DAYS is test, everything before is train+val
    split_idx = n - test_days
    train_val = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    steps = test_days  # forecast length

    forecasts: Dict[str, Dict[str, np.ndarray]] = {
        "SeasonalNaive": {},
        "ARIMA": {},
        "XGBoost": {},
    }

    for series in eval_series:
        y_train = train_val[series].values
        dates_train = train_val["date"]

        forecasts["SeasonalNaive"][series] = seasonal_naive_forecast(y_train, steps, season=seasonal_period)
        forecasts["ARIMA"][series] = arima_forecast(y_train, steps, season=seasonal_period)
        forecasts["XGBoost"][series] = xgb_recursive_forecast(
            y_train, dates_train, steps, max_lag=max_lag, seasonal_period=seasonal_period
        )

    # metrics (mean over series)
    rows = []
    horizons = sorted(set(int(h) for h in horizons))
    for h in horizons:
        if h <= 0 or h > steps:
            continue
        for model_name, per_series in forecasts.items():
            maes, rmses = [], []
            for series in eval_series:
                y_true = test[series].values[:h]
                y_pred = per_series[series][:h]
                maes.append(mean_absolute_error(y_true, y_pred))
                rmses.append(math.sqrt(mean_squared_error(y_true, y_pred)))
            rows.append(
                {
                    "horizon_days": h,
                    "model": model_name,
                    "mae_mean_over_series": float(np.mean(maes)),
                    "rmse_mean_over_series": float(np.mean(rmses)),
                }
            )

    metrics_df = pd.DataFrame(rows).sort_values(["horizon_days", "model"]).reset_index(drop=True)

    # forecast table (point forecasts + actuals for the test window)
    out_rows = []
    future_dates = test["date"].iloc[:steps].dt.date.astype(str).tolist()
    for i, d in enumerate(future_dates):
        for series in eval_series:
            out_rows.append(
                {
                    "date": d,
                    "series": series,
                    "SeasonalNaive": float(forecasts["SeasonalNaive"][series][i]),
                    "ARIMA": float(forecasts["ARIMA"][series][i]),
                    "XGBoost": float(forecasts["XGBoost"][series][i]),
                    "actual": float(test[series].values[i]),
                }
            )

    fc_df = pd.DataFrame(out_rows)
    return metrics_df, fc_df, forecasts, train_val, test


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="ICU Traditional Baselines", layout="wide")
st.title("ICU Bed Availability — Traditional Baselines (Simple UI)")

st.caption(f"Using CSV_PATH: {CSV_PATH}")
st.caption(f"Saving outputs to OUT_DIR: {str(OUT_DIR)}")

# Load automatically from hard-coded path
try:
    df_raw = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"Failed to read CSV_PATH.\n\nCSV_PATH = {CSV_PATH}\n\nError: {e}")
    st.stop()

# Preview
st.subheader("Data Preview")
st.write(f"Rows: **{len(df_raw):,}** | Columns: **{len(df_raw.columns)}**")
st.dataframe(df_raw.head(20), use_container_width=True)

# Series selection
st.subheader("Series Selection")
candidates = [c for c in df_raw.columns if c != "date"]
default_pick = [c for c in DEFAULT_EVAL_SERIES if c in df_raw.columns]
if not default_pick:
    default_pick = candidates[:2] if len(candidates) >= 2 else candidates

eval_series = st.multiselect(
    "Choose series to evaluate/forecast",
    options=candidates,
    default=default_pick,
)

# Settings (simple)
with st.expander("Settings", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        test_days = st.number_input("Test days (forecast length)", min_value=7, max_value=90, value=DEFAULT_TEST_DAYS, step=1)
    with col2:
        seasonal_period = st.number_input("Seasonal period (days)", min_value=1, max_value=30, value=DEFAULT_SEASONAL_PERIOD, step=1)
    with col3:
        max_lag = st.number_input("Max lag (XGBoost)", min_value=7, max_value=90, value=DEFAULT_MAX_LAG, step=1)

    horizons = st.multiselect(
        "Evaluation horizons (days)",
        options=list(range(1, int(test_days) + 1)),
        default=[h for h in DEFAULT_HORIZONS if h <= test_days],
    )

run = st.button("Run Baselines", type="primary", use_container_width=True)

if run:
    if not eval_series:
        st.error("Please select at least one series.")
        st.stop()
    if not horizons:
        st.error("Please select at least one evaluation horizon.")
        st.stop()

    try:
        metrics_df, fc_df, forecasts, train_val, test = run_baselines(
            df_raw=df_raw,
            eval_series=eval_series,
            test_days=int(test_days),
            horizons=horizons,
            seasonal_period=int(seasonal_period),
            max_lag=int(max_lag),
        )

        # Save outputs
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = OUT_DIR / "backtest_metrics_traditional.csv"
        fc_path = OUT_DIR / f"test_forecast_traditional_{int(test_days)}d.csv"
        metrics_df.to_csv(metrics_path, index=False)
        fc_df.to_csv(fc_path, index=False)

        st.success("Done! Metrics + forecasts generated and saved.")

        colA, colB = st.columns(2)

        with colA:
            st.subheader("Backtest Metrics (MAE/RMSE)")
            st.dataframe(metrics_df, use_container_width=True)
            st.download_button(
                "Download backtest_metrics_traditional.csv",
                data=df_to_csv_bytes(metrics_df),
                file_name="backtest_metrics_traditional.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"Saved: {metrics_path}")

        with colB:
            st.subheader(f"Forecasts ({int(test_days)} days) + Actuals")
            st.dataframe(fc_df.head(60), use_container_width=True)
            st.download_button(
                f"Download test_forecast_traditional_{int(test_days)}d.csv",
                data=df_to_csv_bytes(fc_df),
                file_name=f"test_forecast_traditional_{int(test_days)}d.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"Saved: {fc_path}")

        st.divider()
        st.subheader("Plot: Actual vs Forecast")

        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            plot_series = st.selectbox("Series", options=eval_series, index=0)
        with pcol2:
            plot_model = st.selectbox("Model", options=["SeasonalNaive", "ARIMA", "XGBoost"], index=0)
        with pcol3:
            plot_h = st.selectbox("Horizon (days)", options=sorted(set(int(h) for h in horizons)), index=0)

        h = int(plot_h)
        test_dates = pd.to_datetime(test["date"].iloc[:h])
        actual = test[plot_series].values[:h]
        pred = forecasts[plot_model][plot_series][:h]

        fig = plt.figure()
        plt.plot(test_dates, actual, label="Actual")
        plt.plot(test_dates, pred, label=plot_model)
        plt.xlabel("Date")
        plt.ylabel(plot_series)
        plt.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.error(f"Run failed: {e}")
