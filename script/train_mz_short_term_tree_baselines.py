# -*- coding: utf-8 -*-
"""Short-term tree-model baselines for Miaozhuang wind power.

The script trains Persistence, DummyMean, LightGBM and XGBoost baselines for
15-minute, 30-minute and 60-minute ahead power prediction.

It uses only information available at timestamp t:
- current and historical station power
- measured wind speed/direction up to timestamp t
- calendar features

Targets are future power values at t+h.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Iterable


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "sklearn",
    "lightgbm",
    "xgboost",
    "joblib",
    "matplotlib",
]


def check_dependencies() -> None:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing dependencies. No packages were installed automatically.")
        print("Missing:", ", ".join(missing))
        sys.exit(2)


check_dependencies()

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "mz_short_term_tree_baselines"

CAPACITY_MW = 76.0
RANDOM_STATE = 42

POWER_DIR = ROOT / "风电厂" / "苗庄风电厂" / "实发数据"
POWER_FILES = ["311.csv", "312.csv", "313.csv", "314.csv"]

JK_ROOT = ROOT / "data" / "weather" / "jk_data" / "苗庄风电站"
WIND_SPEED_DIR = JK_ROOT / "25年风速"
WIND_DIR_DIR = JK_ROOT / "25年风向"

TRAIN_END = pd.Timestamp("2025-06-01 00:00:00")
VALID_START = pd.Timestamp("2025-06-01 00:00:00")
VALID_END = pd.Timestamp("2025-06-16 00:00:00")
TEST_START = pd.Timestamp("2025-06-16 00:00:00")
TEST_END = pd.Timestamp("2025-07-01 00:00:00")

HORIZONS = {
    "15min": 1,
    "30min": 2,
    "60min": 4,
}

POWER_LAGS = [1, 2, 4, 8, 16, 32, 96]
ROLL_WINDOWS = [4, 8, 16, 96]


def fail(message: str) -> None:
    raise RuntimeError(message)


def turbine_id_from_name(path: Path) -> str:
    return path.name.split("_", 1)[0]


def detect_scada_header_row(file_path: Path, max_lines: int = 30) -> int:
    with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            normalized = [str(cell).strip().lower() for cell in row]
            if len(normalized) >= 2 and normalized[0] == "objectid" and normalized[1] == "objecttimestamp":
                return i
            if i >= max_lines - 1:
                break
    fail(f"Could not find SCADA header row in {file_path}")


def read_power_file(file_path: Path, name: str) -> pd.DataFrame:
    if not file_path.exists():
        fail(f"Power file not found: {file_path}")

    header_row = detect_scada_header_row(file_path)
    usecols = ["ObjectTimeStamp", "Latest value", "Quality", "Operation status"]
    df = pd.read_csv(file_path, header=header_row, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df["ObjectTimeStamp"], errors="coerce"),
            f"{name}_power": pd.to_numeric(df["Latest value"], errors="coerce"),
            f"{name}_quality": pd.to_numeric(df["Quality"], errors="coerce"),
            f"{name}_operation_status": pd.to_numeric(df["Operation status"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="first")
    return out.sort_values("time").reset_index(drop=True)


def load_power() -> tuple[pd.DataFrame, dict]:
    frames = []
    for file_name in POWER_FILES:
        line_name = Path(file_name).stem
        frames.append(read_power_file(POWER_DIR / file_name, line_name))

    data = frames[0]
    for frame in frames[1:]:
        data = data.merge(frame, on="time", how="outer")
    data = data.sort_values("time").reset_index(drop=True)

    line_names = [Path(name).stem for name in POWER_FILES]
    power_cols = [f"{name}_power" for name in line_names]
    quality_cols = [f"{name}_quality" for name in line_names]
    op_cols = [f"{name}_operation_status" for name in line_names]

    data["power_sum_raw_mw"] = data[power_cols].sum(axis=1, min_count=len(power_cols))
    data["power_mw"] = (-data["power_sum_raw_mw"]).clip(lower=0.0, upper=CAPACITY_MW)

    data["power_missing_count"] = data[power_cols].isna().sum(axis=1)
    data["power_quality_missing_count"] = data[quality_cols].isna().sum(axis=1)
    data["power_quality_bad_count"] = data[quality_cols].ne(0).sum(axis=1)
    data["power_operation_missing_count"] = data[op_cols].isna().sum(axis=1)
    data["power_operation_bad_count"] = data[op_cols].ne(4100).sum(axis=1)

    complete = data.dropna(subset=power_cols + ["power_mw"]).copy()
    summary = {
        "rows_raw": int(len(data)),
        "rows_complete": int(len(complete)),
        "time_start": str(complete["time"].min()),
        "time_end": str(complete["time"].max()),
        "missing_timestamps": int(len(data) - len(complete)),
        "power_min_mw": float(complete["power_mw"].min()),
        "power_max_mw": float(complete["power_mw"].max()),
        "power_mean_mw": float(complete["power_mw"].mean()),
    }
    return complete, summary


def expected_turbine_files(directory: Path) -> list[Path]:
    if not directory.exists():
        fail(f"Directory not found: {directory}")
    files = sorted(directory.glob("N??_*.csv"))
    if len(files) != 19:
        fail(f"Expected 19 turbine files in {directory}, found {len(files)}")
    return files


def load_speed_series(file_path: Path) -> pd.Series:
    df = pd.read_csv(file_path, encoding="utf-8-sig", usecols=["时间", "平均风速"])
    time = pd.to_datetime(df["时间"], errors="coerce")
    speed = pd.to_numeric(df["平均风速"], errors="coerce")
    frame = pd.DataFrame({"time": time, "speed": speed}).dropna(subset=["time"])
    frame = frame[(frame["time"] >= pd.Timestamp("2025-01-01")) & (frame["time"] <= TEST_END)]
    frame = frame.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return (
        frame.set_index("time")["speed"]
        .resample("15min", label="right", closed="right")
        .mean()
        .rename(turbine_id_from_name(file_path))
    )


def load_direction_sincos(file_path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(file_path, encoding="utf-8-sig", usecols=["时间", "平均风向"])
    time = pd.to_datetime(df["时间"], errors="coerce")
    degrees = pd.to_numeric(df["平均风向"], errors="coerce")
    frame = pd.DataFrame({"time": time, "direction": degrees}).dropna(subset=["time"])
    frame = frame[(frame["time"] >= pd.Timestamp("2025-01-01")) & (frame["time"] <= TEST_END)]
    frame = frame.sort_values("time").drop_duplicates(subset=["time"], keep="last")

    radians = np.deg2rad(frame["direction"] % 360.0)
    frame["sin"] = np.sin(radians)
    frame["cos"] = np.cos(radians)
    indexed = frame.set_index("time")
    tid = turbine_id_from_name(file_path)
    sin_series = indexed["sin"].resample("15min", label="right", closed="right").mean().rename(tid)
    cos_series = indexed["cos"].resample("15min", label="right", closed="right").mean().rename(tid)
    return sin_series, cos_series


def load_wind_features() -> tuple[pd.DataFrame, dict]:
    speed_files = expected_turbine_files(WIND_SPEED_DIR)
    direction_files = expected_turbine_files(WIND_DIR_DIR)

    speed_wide = pd.concat([load_speed_series(path) for path in speed_files], axis=1).sort_index()
    dir_sin_wide, dir_cos_wide = [], []
    for path in direction_files:
        sin_series, cos_series = load_direction_sincos(path)
        dir_sin_wide.append(sin_series)
        dir_cos_wide.append(cos_series)
    sin_wide = pd.concat(dir_sin_wide, axis=1).sort_index()
    cos_wide = pd.concat(dir_cos_wide, axis=1).sort_index()

    out = pd.DataFrame(index=speed_wide.index.union(sin_wide.index).union(cos_wide.index).sort_values())
    out.index.name = "time"

    out["wind_speed_mean"] = speed_wide.mean(axis=1)
    out["wind_speed_std"] = speed_wide.std(axis=1)
    out["wind_speed_min"] = speed_wide.min(axis=1)
    out["wind_speed_max"] = speed_wide.max(axis=1)
    out["wind_speed_missing_ratio"] = speed_wide.isna().mean(axis=1)
    out["wind_speed_available_count"] = speed_wide.notna().sum(axis=1)
    out["wind_speed_mean_sq"] = out["wind_speed_mean"] ** 2
    out["wind_speed_mean_cu"] = out["wind_speed_mean"] ** 3

    out["wind_dir_sin"] = sin_wide.mean(axis=1)
    out["wind_dir_cos"] = cos_wide.mean(axis=1)
    norm = np.sqrt(out["wind_dir_sin"] ** 2 + out["wind_dir_cos"] ** 2)
    out["wind_dir_sin"] = np.where(norm > 0, out["wind_dir_sin"] / norm, 0.0)
    out["wind_dir_cos"] = np.where(norm > 0, out["wind_dir_cos"] / norm, 1.0)
    out["wind_dir_missing_ratio"] = sin_wide.isna().mean(axis=1)
    out["wind_dir_available_count"] = sin_wide.notna().sum(axis=1)

    out = out.reset_index()
    summary = {
        "speed_files": len(speed_files),
        "direction_files": len(direction_files),
        "rows": int(len(out)),
        "time_start": str(out["time"].min()),
        "time_end": str(out["time"].max()),
    }
    return out, summary


def add_power_history_features(data: pd.DataFrame) -> list[str]:
    feature_cols = ["power_mw"]

    for lag in POWER_LAGS:
        name = f"power_lag_{lag}"
        data[name] = data["power_mw"].shift(lag)
        feature_cols.append(name)

    for window in ROLL_WINDOWS:
        roll = data["power_mw"].rolling(window=window, min_periods=window)
        for stat_name, series in [
            ("mean", roll.mean()),
            ("std", roll.std()),
            ("min", roll.min()),
            ("max", roll.max()),
        ]:
            name = f"power_roll_{window}_{stat_name}"
            data[name] = series
            feature_cols.append(name)

    return feature_cols


def add_time_features(data: pd.DataFrame) -> list[str]:
    hour = data["time"].dt.hour + data["time"].dt.minute / 60.0
    minute = data["time"].dt.minute.astype(float)
    weekday = data["time"].dt.weekday.astype(float)
    doy = data["time"].dt.dayofyear.astype(float)

    features = {}
    cycles = {
        "hour": (hour, 24.0),
        "minute": (minute, 60.0),
        "weekday": (weekday, 7.0),
        "doy": (doy, 365.25),
    }
    for name, (value, period) in cycles.items():
        features[f"{name}_sin"] = np.sin(2.0 * math.pi * value / period)
        features[f"{name}_cos"] = np.cos(2.0 * math.pi * value / period)

    for name, values in features.items():
        data[name] = values
    return list(features.keys())


def build_dataset(power: pd.DataFrame, wind: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    data = power.merge(wind, on="time", how="left").sort_values("time").reset_index(drop=True)

    feature_cols = []
    feature_cols.extend(add_power_history_features(data))

    wind_cols = [
        "wind_speed_mean",
        "wind_speed_std",
        "wind_speed_min",
        "wind_speed_max",
        "wind_speed_missing_ratio",
        "wind_speed_available_count",
        "wind_speed_mean_sq",
        "wind_speed_mean_cu",
        "wind_dir_sin",
        "wind_dir_cos",
        "wind_dir_missing_ratio",
        "wind_dir_available_count",
    ]
    feature_cols.extend(wind_cols)

    quality_cols = [
        "power_missing_count",
        "power_quality_missing_count",
        "power_quality_bad_count",
        "power_operation_missing_count",
        "power_operation_bad_count",
    ]
    feature_cols.extend(quality_cols)
    feature_cols.extend(add_time_features(data))

    for horizon_name, steps in HORIZONS.items():
        data[f"target_{horizon_name}"] = data["power_mw"].shift(-steps)

    before = len(data)
    data = data.dropna(subset=feature_cols).reset_index(drop=True)
    summary = {
        "rows_before_feature_drop": int(before),
        "rows_after_feature_drop": int(len(data)),
        "dropped_feature_rows": int(before - len(data)),
        "time_start": str(data["time"].min()),
        "time_end": str(data["time"].max()),
        "feature_count": int(len(feature_cols)),
    }
    return data, feature_cols, summary


def split_for_horizon(data: pd.DataFrame, target_col: str, feature_cols: list[str]) -> dict[str, pd.DataFrame]:
    usable = data.dropna(subset=feature_cols + [target_col]).copy()
    splits = {
        "train": usable[usable["time"] < TRAIN_END].copy(),
        "valid": usable[(usable["time"] >= VALID_START) & (usable["time"] < VALID_END)].copy(),
        "test": usable[(usable["time"] >= TEST_START) & (usable["time"] < TEST_END)].copy(),
    }
    for split_name, frame in splits.items():
        if frame.empty:
            fail(
                f"{target_col}: {split_name} split is empty. "
                f"Usable time range is {usable['time'].min()} -> {usable['time'].max()}"
            )
    return splits


def clipped(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).clip(0.0, CAPACITY_MW)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pred = clipped(y_pred)
    mae = mean_absolute_error(y_true, pred)
    rmse = math.sqrt(mean_squared_error(y_true, pred))
    return {
        "MAE_MW": float(mae),
        "RMSE_MW": float(rmse),
        "nMAE": float(mae / CAPACITY_MW),
        "nRMSE": float(rmse / CAPACITY_MW),
        "R2": float(r2_score(y_true, pred)) if len(y_true) >= 2 else None,
    }


def model_factories() -> dict[str, object]:
    return {
        "LightGBM": LGBMRegressor(
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_horizon(
    data: pd.DataFrame,
    horizon_name: str,
    feature_cols: list[str],
) -> tuple[list[dict], pd.DataFrame, dict[str, object]]:
    target_col = f"target_{horizon_name}"
    splits = split_for_horizon(data, target_col, feature_cols)
    train, valid, test = splits["train"], splits["valid"], splits["test"]

    x_train = train[feature_cols]
    y_train = train[target_col].to_numpy()

    rows = []
    predictions = test[["time", "power_mw", target_col]].copy()
    predictions = predictions.rename(columns={"power_mw": "power_current", target_col: "y_true"})

    for split_name, frame in splits.items():
        y = frame[target_col].to_numpy()
        persistence_pred = frame["power_mw"].to_numpy()
        dummy_pred = np.full(len(frame), float(np.mean(y_train)))

        for model_name, pred in [
            ("Persistence", persistence_pred),
            ("DummyMean", dummy_pred),
        ]:
            row = {
                "horizon": horizon_name,
                "split": split_name,
                "model": model_name,
                "rows": int(len(frame)),
                **metrics(y, pred),
            }
            rows.append(row)

    fitted_models: dict[str, object] = {}
    for model_name, model in model_factories().items():
        model.fit(x_train, y_train)
        fitted_models[model_name] = model

        for split_name, frame in splits.items():
            y = frame[target_col].to_numpy()
            pred = model.predict(frame[feature_cols])
            row = {
                "horizon": horizon_name,
                "split": split_name,
                "model": model_name,
                "rows": int(len(frame)),
                **metrics(y, pred),
            }
            rows.append(row)

        predictions[f"y_pred_{model_name}"] = clipped(model.predict(test[feature_cols]))

    predictions["y_pred_Persistence"] = clipped(test["power_mw"].to_numpy())
    predictions["y_pred_DummyMean"] = np.full(len(test), float(np.mean(y_train))).clip(0.0, CAPACITY_MW)
    predictions["horizon"] = horizon_name
    return rows, predictions, fitted_models


def write_plot(predictions: pd.DataFrame, horizon_name: str, output_path: Path) -> None:
    plot_df = predictions.sort_values("time")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df["time"], plot_df["y_true"], label="Actual", linewidth=1.7)
    ax.plot(plot_df["time"], plot_df["y_pred_Persistence"], label="Persistence", linewidth=1.1, alpha=0.8)
    ax.plot(plot_df["time"], plot_df["y_pred_LightGBM"], label="LightGBM", linewidth=1.3)
    ax.plot(plot_df["time"], plot_df["y_pred_XGBoost"], label="XGBoost", linewidth=1.3)
    ax.set_title(f"Miaozhuang short-term power prediction ({horizon_name})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (MW)")
    ax.set_ylim(bottom=0.0, top=CAPACITY_MW * 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def best_test_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    test = metrics_df[metrics_df["split"] == "test"].copy()
    return test.sort_values(["horizon", "RMSE_MW"]).groupby("horizon", as_index=False).first()


def write_report(
    metrics_df: pd.DataFrame,
    data_summary: dict,
    power_summary: dict,
    wind_summary: dict,
    output_path: Path,
) -> None:
    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    best = best_test_rows(metrics_df)

    def metric_line(horizon: str, model: str) -> str:
        row = test_metrics[(test_metrics["horizon"] == horizon) & (test_metrics["model"] == model)]
        if row.empty:
            return "-"
        r = row.iloc[0]
        return f"MAE {r['MAE_MW']:.2f} MW, RMSE {r['RMSE_MW']:.2f} MW, nRMSE {r['nRMSE']:.2%}"

    lines = [
        "# 苗庄短期树模型基线结果",
        "",
        "## 数据与任务",
        "",
        "- 任务：用当前及历史功率、实测风速/风向和时间特征，预测未来 15min、30min、60min 全场功率。",
        "- 功率标签：`power_mw = clip(-(311 + 312 + 313 + 314), 0, 76)`。",
        "- 功率数据范围：`{}` 至 `{}`。".format(power_summary["time_start"], power_summary["time_end"]),
        "- 风速/风向文件：19 台风机风速 + 19 台风机风向。",
        "- 特征表样本：`{}` 条，特征数 `{}`。".format(data_summary["rows_after_feature_drop"], data_summary["feature_count"]),
        "",
        "## 测试集结果",
        "",
        "| Horizon | Persistence RMSE | LightGBM RMSE | XGBoost RMSE | 最优模型 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for horizon in HORIZONS:
        rows = {
            model: test_metrics[(test_metrics["horizon"] == horizon) & (test_metrics["model"] == model)]
            for model in ["Persistence", "LightGBM", "XGBoost"]
        }
        values = {}
        for model, frame in rows.items():
            values[model] = float(frame.iloc[0]["RMSE_MW"]) if not frame.empty else float("nan")
        best_row = best[best["horizon"] == horizon].iloc[0]
        lines.append(
            f"| {horizon} | {values['Persistence']:.2f} | {values['LightGBM']:.2f} | "
            f"{values['XGBoost']:.2f} | {best_row['model']} |"
        )

    lines.extend(
        [
            "",
            "## 分模型说明",
            "",
        ]
    )

    for horizon in HORIZONS:
        lines.append(f"- `{horizon}`：Persistence（{metric_line(horizon, 'Persistence')}），LightGBM（{metric_line(horizon, 'LightGBM')}），XGBoost（{metric_line(horizon, 'XGBoost')}）。")

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "本轮已经跑通短期功率预测的树模型基线流程。结果需要重点和 Persistence 对比：短期预测里当前功率本身就是很强的基线，复杂模型如果不能稳定超过它，应优先检查数据对齐、功率口径、实测风速/风向与功率时间戳是否一致。",
            "",
            "本轮没有使用未来时刻的实测风速/风向，避免了明显的数据泄漏。5min 预测暂未做，因为当前半年功率数据是 15 分钟粒度。",
            "",
            "## 输出文件",
            "",
            "- `features_short_term.csv`：完整特征表。",
            "- `metrics_by_horizon.csv`：各 horizon 和模型的指标。",
            "- `predictions_15min.csv`、`predictions_30min.csv`、`predictions_60min.csv`：测试集预测结果。",
            "- `actual_vs_pred_15min.png`、`actual_vs_pred_30min.png`、`actual_vs_pred_60min.png`：预测对比图。",
            "- `model_lightgbm_*.joblib`、`model_xgboost_*.joblib`：模型文件。",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading power data...")
    power, power_summary = load_power()

    print("Loading and aggregating wind speed/direction data...")
    wind, wind_summary = load_wind_features()

    print("Building feature table...")
    data, feature_cols, data_summary = build_dataset(power, wind)

    data.to_csv(OUTPUT_DIR / "features_short_term.csv", index=False, encoding="utf-8-sig")

    all_metrics = []
    all_predictions = {}
    model_summary = {}

    for horizon_name in HORIZONS:
        print(f"Training horizon {horizon_name}...")
        rows, predictions, models = evaluate_horizon(data, horizon_name, feature_cols)
        all_metrics.extend(rows)
        all_predictions[horizon_name] = predictions

        predictions.to_csv(OUTPUT_DIR / f"predictions_{horizon_name}.csv", index=False, encoding="utf-8-sig")
        write_plot(predictions, horizon_name, OUTPUT_DIR / f"actual_vs_pred_{horizon_name}.png")

        for model_name, model in models.items():
            model_key = model_name.lower()
            joblib.dump(
                {
                    "model": model,
                    "feature_columns": feature_cols,
                    "capacity_mw": CAPACITY_MW,
                    "horizon": horizon_name,
                    "target_definition": f"target_{horizon_name}=power_mw shifted -{HORIZONS[horizon_name]} rows",
                },
                OUTPUT_DIR / f"model_{model_key}_{horizon_name}.joblib",
            )
        model_summary[horizon_name] = {"models": sorted(models)}

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig")

    run_summary = {
        "power_summary": power_summary,
        "wind_summary": wind_summary,
        "data_summary": data_summary,
        "feature_columns": feature_cols,
        "horizons": HORIZONS,
        "split": {
            "train": f"time < {TRAIN_END}",
            "valid": f"{VALID_START} <= time < {VALID_END}",
            "test": f"{TEST_START} <= time < {TEST_END}",
        },
        "model_summary": model_summary,
    }
    save_json(OUTPUT_DIR / "run_summary.json", run_summary)
    write_report(metrics_df, data_summary, power_summary, wind_summary, OUTPUT_DIR / "短期树模型基线结果.md")

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Test RMSE summary:")
    test = metrics_df[metrics_df["split"] == "test"].copy()
    print(test.pivot(index="horizon", columns="model", values="RMSE_MW").round(4).to_string())


if __name__ == "__main__":
    main()
