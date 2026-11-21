from __future__ import annotations

import time
import warnings
import re
import functools
from pathlib import Path

import numpy as np
import pandas as pd

print("OK")

# =========================
# HARD CODE SETTINGS
# =========================
BASE_DIR = Path(__file__).resolve().parent  # ✅ FIX: portable folder-based paths

# ✅ FIX: dataset must be in the SAME folder as this script
DATA_PATH = str(BASE_DIR / "icu_beds.csv")

OUT_DIR = BASE_DIR / "artifacts"  # ✅ FIX: portable artifacts path

FREQ = "D"
CONTEXT_LEN = 56

PRED_LEN = 28
FORECAST_HORIZONS = [7, 14, 28]

VAL_LEN = 28
TEST_LEN = 28

DEEPAR_EPOCHS = 12
TFT_EPOCHS = 8
PATCHTST_MAX_STEPS = 800

BATCH_SIZE = 32
NUM_SAMPLES = 300

# ✅ Quantiles we will output everywhere
Q_LEVELS = [0.1, 0.5, 0.9]

TARGET_COLS = [
    "available_adult_icu_beds",
    "total_adult_icu_patients",
    "available_ped_icu_beds",
    "total_ped_icu_patients",
]

# -------------------------
# Reduce noisy warnings
# -------------------------
warnings.filterwarnings(
    "ignore",
    message=r"Using a non-tuple sequence for multidimensional indexing is deprecated.*",
    category=UserWarning,
)

# -------------------------
# Torch safety (only relevant for torch>=2.6)
# -------------------------
try:
    import torch
except Exception as e:
    raise RuntimeError(f"torch import failed: {e}")

# ✅ FIX (safe-side): allowlist globals if torch uses safe unpickling mode
if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([functools.partial, getattr, object])


# -------------------------
# Utilities
# -------------------------
def now() -> float:
    return time.time()


def fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}min"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_wide_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {p}\n"
            f"✅ Put 'icu_beds.csv' in: {BASE_DIR}"
        )

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")

    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")

    wide = df[["date"] + TARGET_COLS].copy()
    for c in TARGET_COLS:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")

    if wide[TARGET_COLS].isna().any().any():
        wide[TARGET_COLS] = wide[TARGET_COLS].interpolate(limit_direction="both")
        wide[TARGET_COLS] = wide[TARGET_COLS].ffill().bfill()

    return wide.set_index("date")


def to_long_nf(wide: pd.DataFrame) -> pd.DataFrame:
    long_df = wide.reset_index().melt(
        id_vars="date", value_vars=TARGET_COLS, var_name="unique_id", value_name="y"
    )
    return long_df.rename(columns={"date": "ds"})


def split_wide(wide: pd.DataFrame):
    n = len(wide)
    need = VAL_LEN + TEST_LEN + CONTEXT_LEN + PRED_LEN
    if n <= need:
        raise ValueError(
            f"Not enough rows ({n}) for VAL_LEN={VAL_LEN}, TEST_LEN={TEST_LEN}, "
            f"CONTEXT_LEN={CONTEXT_LEN}, PRED_LEN={PRED_LEN}. Need > {need}."
        )

    train_end = n - (VAL_LEN + TEST_LEN)
    val_end = n - TEST_LEN

    train_wide = wide.iloc[:train_end]
    trainval_wide = wide.iloc[:val_end]
    test_wide = wide.iloc[val_end: val_end + TEST_LEN]

    return train_wide, trainval_wide, test_wide, train_end, val_end


def make_gluonts_listdataset(wide_slice: pd.DataFrame):
    from gluonts.dataset.common import ListDataset

    start = wide_slice.index[0]
    data = []
    for col in TARGET_COLS:
        data.append({"start": start, "target": wide_slice[col].to_numpy(dtype=np.float32)})
    return ListDataset(data, freq=FREQ)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def slice_horizon(arr_2d: np.ndarray, h: int) -> np.ndarray:
    return arr_2d[:, :h]


def safe_deserialize_predictor(model_dir: Path):
    from gluonts.model.predictor import Predictor
    pred_json = model_dir / "predictor.json"
    if model_dir.exists() and pred_json.exists():
        return Predictor.deserialize(model_dir)
    return None


# -------------------------
# ✅ Quantile extraction for GluonTS (DeepAR / TFT)
# -------------------------
def _forecast_to_quantiles(fc, q_levels: list[float]) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    for q in q_levels:
        try:
            out[q] = np.asarray(fc.quantile(q), dtype=np.float32)
        except Exception:
            try:
                s = np.asarray(fc.samples, dtype=np.float32)  # [num_samples, PRED_LEN]
                out[q] = np.quantile(s, q, axis=0).astype(np.float32)
            except Exception:
                out[q] = np.asarray(fc.mean, dtype=np.float32)
    return out


def predict_gluonts_quantiles(predictor, history_ds, num_samples: int, q_levels: list[float]) -> dict[str, np.ndarray]:
    forecasts = list(predictor.predict(history_ds, num_samples=num_samples))

    q_maps = {q: [] for q in q_levels}
    for fc in forecasts:
        qdict = _forecast_to_quantiles(fc, q_levels)
        for q in q_levels:
            q_maps[q].append(qdict[q])

    out = {}
    for q in q_levels:
        out[f"p{int(q*100):02d}"] = np.stack(q_maps[q], axis=0)
    return out


# -------------------------
# Model: DeepAR
# -------------------------
def train_or_load_deepar(train_ds, val_ds, model_dir: Path):
    from gluonts.torch.model.deepar import DeepAREstimator
    from lightning.pytorch.callbacks import EarlyStopping

    loaded = safe_deserialize_predictor(model_dir)
    if loaded is not None:
        print(f"Load DeepAR from: {model_dir}")
        return loaded

    ensure_dir(model_dir)

    trainer_kwargs = dict(
        max_epochs=DEEPAR_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir=str(model_dir),
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
    )

    est = DeepAREstimator(
        freq=FREQ,
        prediction_length=PRED_LEN,
        context_length=CONTEXT_LEN,
        batch_size=BATCH_SIZE,
        trainer_kwargs=trainer_kwargs,
    )

    predictor = est.train(training_data=train_ds, validation_data=val_ds)
    predictor.serialize(model_dir)
    return predictor


# -------------------------
# Model: TFT
# -------------------------
def train_or_load_tft(train_ds, val_ds, model_dir: Path):
    from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
    from lightning.pytorch.callbacks import EarlyStopping

    loaded = safe_deserialize_predictor(model_dir)
    if loaded is not None:
        print(f"Load TFT from: {model_dir}")
        return loaded

    ensure_dir(model_dir)

    trainer_kwargs = dict(
        max_epochs=TFT_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir=str(model_dir),
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, mode="min")],
    )

    est = TemporalFusionTransformerEstimator(
        freq=FREQ,
        prediction_length=PRED_LEN,
        context_length=CONTEXT_LEN,
        batch_size=BATCH_SIZE,
        hidden_dim=32,
        variable_dim=16,
        num_heads=4,
        dropout_rate=0.1,
        trainer_kwargs=trainer_kwargs,
    )

    predictor = est.train(training_data=train_ds, validation_data=val_ds)
    predictor.serialize(model_dir)
    return predictor


# -------------------------
# Model: PatchTST (NeuralForecast)
# -------------------------
def train_or_load_patchtst(trainval_long: pd.DataFrame, model_dir: Path):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST
    from neuralforecast.losses.pytorch import MQLoss  # ✅ FIX: quantile loss

    ensure_dir(model_dir)

    try:
        if (model_dir / "config.json").exists():
            print(f"Load PatchTST from: {model_dir}")
            return NeuralForecast.load(path=str(model_dir))
    except Exception:
        pass

    print("Train PatchTST (fast settings, quantile loss)")
    model = PatchTST(
        h=PRED_LEN,
        input_size=CONTEXT_LEN,
        max_steps=PATCHTST_MAX_STEPS,
        loss=MQLoss(quantiles=Q_LEVELS),
        scaler_type="standard",
        batch_size=BATCH_SIZE,
    )

    nf = NeuralForecast(models=[model], freq=FREQ)
    nf.fit(df=trainval_long, val_size=VAL_LEN)
    nf.save(path=str(model_dir), overwrite=True)
    return nf


# ✅ FIX: robust parser for NeuralForecast quantile columns
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _parse_quantile_from_col(col: str) -> float | None:
    s = col.lower()

    # patterns: q0.1, q10, p10, p90 etc
    m = re.search(r"(?:^|[_\-])q(?P<v>\d+(?:\.\d+)?)", s)
    if m:
        v = float(m.group("v"))
        return v if v <= 1.0 else v / 100.0

    m = re.search(r"(?:^|[_\-])p(?P<v>\d+(?:\.\d+)?)", s)
    if m:
        v = float(m.group("v"))
        return v if v <= 1.0 else v / 100.0

    # sometimes "quantile0.1"
    m = re.search(r"quantile(?P<v>\d+(?:\.\d+)?)", s)
    if m:
        v = float(m.group("v"))
        return v if v <= 1.0 else v / 100.0

    return None


def _parse_interval_level(col: str) -> tuple[str, int] | None:
    s = col.lower()
    m = re.search(r"(?:^|[_\-])(lo|lower)[_\-]*(?P<lvl>\d+)", s)
    if m:
        return ("lo", int(m.group("lvl")))
    m = re.search(r"(?:^|[_\-])(hi|upper)[_\-]*(?P<lvl>\d+)", s)
    if m:
        return ("hi", int(m.group("lvl")))
    return None


def predict_patchtst_quantiles(nf, history_long: pd.DataFrame) -> dict[str, np.ndarray]:
    # ✅ Prefer quantiles API
    try:
        fcst = nf.predict(df=history_long, quantiles=Q_LEVELS)
    except TypeError:
        # older NF: use level for intervals
        fcst = nf.predict(df=history_long, level=[80])

    model_name = getattr(nf.models[0], "name", nf.models[0].__class__.__name__)
    cols = [c for c in fcst.columns if c not in ("ds", "unique_id")]

    print(f"PatchTST predict columns: {cols}")  # ✅ preflight visibility (no guessing)

    # point column
    point_col = None
    for c in cols:
        if _norm(c) == _norm(model_name):
            point_col = c
            break
    if point_col is None:
        # fallback: first non-interval, non-quantile column containing model name
        for c in cols:
            if model_name.lower() in c.lower():
                if _parse_quantile_from_col(c) is None and _parse_interval_level(c) is None:
                    point_col = c
                    break

    # quantile columns
    qcols: dict[float, str] = {}
    lo80 = None
    hi80 = None

    for c in cols:
        if model_name.lower() not in c.lower():
            continue

        q = _parse_quantile_from_col(c)
        if q is not None:
            qcols[q] = c
            continue

        iv = _parse_interval_level(c)
        if iv is not None:
            side, lvl = iv
            if lvl == 80 and side == "lo":
                lo80 = c
            if lvl == 80 and side == "hi":
                hi80 = c

    def pick_q(desired: float) -> str | None:
        # exact match
        for q, c in qcols.items():
            if abs(q - desired) < 1e-9:
                return c
        # near match (handles rounding like 0.1000000002)
        if qcols:
            best_q = min(qcols.keys(), key=lambda x: abs(x - desired))
            if abs(best_q - desired) < 0.02:
                return qcols[best_q]
        return None

    q10_col = pick_q(0.1) or lo80
    q50_col = pick_q(0.5) or point_col
    q90_col = pick_q(0.9) or hi80

    print(f"PatchTST column map -> p10={q10_col}, p50={q50_col}, p90={q90_col}")  # ✅ explicit mapping

    if q10_col is None or q50_col is None or q90_col is None:
        raise RuntimeError(
            "❌ PatchTST quantile columns not found.\n"
            f"Columns available: {cols}\n"
            "Fix: upgrade NeuralForecast, or share the printed columns with me and I will map them exactly."
        )

    out = {"p10": [], "p50": [], "p90": []}
    for uid in TARGET_COLS:
        out["p10"].append(fcst.loc[fcst["unique_id"] == uid, q10_col].to_numpy(dtype=np.float32))
        out["p50"].append(fcst.loc[fcst["unique_id"] == uid, q50_col].to_numpy(dtype=np.float32))
        out["p90"].append(fcst.loc[fcst["unique_id"] == uid, q90_col].to_numpy(dtype=np.float32))

    out = {k: np.stack(v, axis=0) for k, v in out.items()}

    # ✅ SAFETY: never allow p10==p50 everywhere again
    eq_rate = float((out["p10"] == out["p50"]).mean())
    if eq_rate > 0.98:
        raise RuntimeError(
            f"❌ PatchTST p10==p50 rate is {eq_rate:.2f} (too high). This means wrong column mapping or missing quantiles.\n"
            "I already printed the columns + selected mapping above — share that text and I will lock the mapping."
        )

    return out


# -------------------------
# Main
# -------------------------
def main():
    print(f"Running script: {Path(__file__).resolve()}")
    print("Version: 2026-02-17")

    ensure_dir(OUT_DIR)
    model_root = OUT_DIR / "models"
    ensure_dir(model_root)
    ensure_dir(OUT_DIR / "outputs")

    print("\n===============================")
    print("STEP 0 — Load + split dataset")
    print("===============================")

    wide = load_wide_csv(DATA_PATH)
    train_wide, trainval_wide, test_wide, train_end, val_end = split_wide(wide)

    total_days = len(wide)
    series_count = len(TARGET_COLS)
    print(f"Date range: {wide.index.min().date()} to {wide.index.max().date()}")
    print(f"Days (dates): {total_days} | Series: {series_count} | Long rows: {total_days*series_count}")
    print(f"Train days: {len(train_wide)}")
    print(f"Val days: {VAL_LEN}")
    print(f"Test days: {TEST_LEN}")
    print(
        f"Long rows split -> Train: {len(train_wide)*series_count}  "
        f"Val: {VAL_LEN*series_count}  Test: {TEST_LEN*series_count}"
    )

    train_ds = make_gluonts_listdataset(train_wide)
    trainval_ds = make_gluonts_listdataset(trainval_wide)

    history_for_test_ds = trainval_ds
    y_test_true = np.stack([test_wide[c].to_numpy(dtype=np.float32) for c in TARGET_COLS], axis=0)

    trainval_long = to_long_nf(trainval_wide)

    preds_q: dict[str, dict[str, np.ndarray]] = {}

    print("\n===============================")
    print("MODEL 1/3 — DeepAR")
    print("===============================")
    t0 = now()
    deepar_dir = model_root / "deepar"
    deepar = train_or_load_deepar(train_ds, trainval_ds, deepar_dir)
    preds_q["DeepAR"] = predict_gluonts_quantiles(deepar, history_for_test_ds, NUM_SAMPLES, Q_LEVELS)
    print(f"DeepAR done in {fmt_secs(now() - t0)}")

    print("\n===============================")
    print("MODEL 2/3 — TFT")
    print("===============================")
    t0 = now()
    tft_dir = model_root / "tft"
    tft = train_or_load_tft(train_ds, trainval_ds, tft_dir)
    preds_q["TFT"] = predict_gluonts_quantiles(tft, history_for_test_ds, NUM_SAMPLES, Q_LEVELS)
    print(f"TFT done in {fmt_secs(now() - t0)}")

    print("\n===============================")
    print("MODEL 3/3 — PatchTST")
    print("===============================")
    t0 = now()
    patch_dir = model_root / "patchtst"
    nf = train_or_load_patchtst(trainval_long, patch_dir)
    preds_q["PatchTST"] = predict_patchtst_quantiles(nf, trainval_long)
    print(f"PatchTST done in {fmt_secs(now() - t0)}")

    preds_q["EnsembleMean"] = {
        "p10": (preds_q["DeepAR"]["p10"] + preds_q["TFT"]["p10"] + preds_q["PatchTST"]["p10"]) / 3.0,
        "p50": (preds_q["DeepAR"]["p50"] + preds_q["TFT"]["p50"] + preds_q["PatchTST"]["p50"]) / 3.0,
        "p90": (preds_q["DeepAR"]["p90"] + preds_q["TFT"]["p90"] + preds_q["PatchTST"]["p90"]) / 3.0,
    }

    print("\n===============================")
    print("STEP 1 — Backtest metrics on TEST window (using p50)")
    print("===============================")
    rows = []
    for h in FORECAST_HORIZONS:
        y_true_h = y_test_true[:, :h]
        for name, qdict in preds_q.items():
            y_pred_h = slice_horizon(qdict["p50"], h)
            rows.append(
                {
                    "horizon_days": h,
                    "model": name,
                    "MAE_mean_over_series": np.mean([mae(y_true_h[i], y_pred_h[i]) for i in range(series_count)]),
                    "RMSE_mean_over_series": np.mean([rmse(y_true_h[i], y_pred_h[i]) for i in range(series_count)]),
                }
            )

    metrics = pd.DataFrame(rows).sort_values(["horizon_days", "model"])
    print(metrics.to_string(index=False))

    metrics_path = OUT_DIR / "outputs" / "backtest_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics -> {metrics_path}")

    print("\n===============================")
    print("STEP 2 — Save TEST forecasts (7/14/28) with p10/p50/p90")
    print("===============================")
    for h in FORECAST_HORIZONS:
        dates = test_wide.index[:h]
        out_rows = []
        for si, series in enumerate(TARGET_COLS):
            for di, d in enumerate(dates):
                r = {"date": d.date().isoformat(), "series": series}
                for model_name, qdict in preds_q.items():
                    r[f"{model_name}_p10"] = float(qdict["p10"][si, di])
                    r[f"{model_name}_p50"] = float(qdict["p50"][si, di])
                    r[f"{model_name}_p90"] = float(qdict["p90"][si, di])
                out_rows.append(r)

        out_df = pd.DataFrame(out_rows)
        out_path = OUT_DIR / "outputs" / f"test_forecast_quantiles_{h}d.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")

    print("\nDONE ✅")
    print("Open artifacts/outputs/ to see CSVs (metrics + 7/14/28 test quantile forecasts).")


if __name__ == "__main__":
    main()
