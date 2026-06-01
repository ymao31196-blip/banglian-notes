# -*- coding: utf-8 -*-
"""Small-batch wind power training for Miaozhuang wind farm.

This script intentionally uses only dependencies already available in the
project Python environment. It builds next-day 15-minute samples from local
NetCDF weather forecasts and SCADA active-power exports, trains a sklearn
RandomForest baseline, and writes reproducible artifacts.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "netCDF4",
    "sklearn",
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
from netCDF4 import Dataset, num2date
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "mz_small_batch"

WEATHER_DIRS = ["2501_2502", "2503_2504", "2505_2506", "2507_2508"]
WEATHER_VARS = ["u10", "v10", "fg10", "u100", "v100", "t2m", "d2m", "ssrd", "tp", "msl"]

STATION_LON = 117.8406861
STATION_LAT = 39.40238056
CAPACITY_MW = 76.0

POWER_DIR = ROOT / "风电厂" / "苗庄风电厂" / "实发数据"
POWER_FILES = ["311.csv", "312.csv", "313.csv", "314.csv"]

TRAIN_INIT_END = pd.Timestamp("2025-01-06 12:00:00")
TEST_INIT_START = pd.Timestamp("2025-01-07 12:00:00")

FEATURE_COLUMNS = [
    "u10",
    "v10",
    "fg10",
    "u100",
    "v100",
    "t2m",
    "d2m",
    "ssrd",
    "tp",
    "msl",
    "ws10",
    "ws100",
    "wd10_sin",
    "wd10_cos",
    "wd100_sin",
    "wd100_cos",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "lead_minutes",
    "lead_hours",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def decode_nc_datetime(var, values) -> pd.DatetimeIndex:
    dates = num2date(values, units=var.units, calendar=getattr(var, "calendar", "standard"))
    if np.ndim(dates) == 0:
        return pd.DatetimeIndex([pd.Timestamp(str(dates))])
    return pd.DatetimeIndex([pd.Timestamp(str(item)) for item in dates])


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
    df = pd.read_csv(file_path, header=header_row, encoding="utf-8-sig", low_memory=False)
    required = {"ObjectTimeStamp", "Latest value"}
    missing = required.difference(df.columns)
    if missing:
        fail(f"{file_path} missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(df["ObjectTimeStamp"], errors="coerce"),
            name: pd.to_numeric(df["Latest value"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["valid_time"]).drop_duplicates(subset=["valid_time"], keep="first")
    return out.sort_values("valid_time").reset_index(drop=True)


def load_power() -> tuple[pd.DataFrame, dict]:
    frames = []
    per_file_summary = {}
    for file_name in POWER_FILES:
        line_name = Path(file_name).stem
        frame = read_power_file(POWER_DIR / file_name, line_name)
        per_file_summary[line_name] = {
            "rows": int(len(frame)),
            "time_start": str(frame["valid_time"].min()),
            "time_end": str(frame["valid_time"].max()),
            "missing_values": int(frame[line_name].isna().sum()),
        }
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="valid_time", how="outer")

    line_cols = [Path(name).stem for name in POWER_FILES]
    merged = merged.sort_values("valid_time").reset_index(drop=True)
    merged["power_sum_raw_mw"] = merged[line_cols].sum(axis=1, min_count=len(line_cols))
    merged["power_mw"] = (-merged["power_sum_raw_mw"]).clip(lower=0.0, upper=CAPACITY_MW)

    complete = merged.dropna(subset=line_cols + ["power_mw"]).copy()
    summary = {
        "per_file": per_file_summary,
        "merged_rows": int(len(merged)),
        "complete_rows": int(len(complete)),
        "dropped_incomplete_rows": int(len(merged) - len(complete)),
        "time_start": str(complete["valid_time"].min()),
        "time_end": str(complete["valid_time"].max()),
        "power_mw_min": float(complete["power_mw"].min()),
        "power_mw_max": float(complete["power_mw"].max()),
        "power_mw_mean": float(complete["power_mw"].mean()),
    }
    return complete[["valid_time", "power_mw", "power_sum_raw_mw"] + line_cols], summary


def weather_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in WEATHER_DIRS:
        root = ROOT / directory
        if not root.exists():
            fail(f"Weather directory not found: {root}")
        paths.extend(sorted(root.glob("*.nc")))
    if len(paths) != 8:
        fail(f"Expected 8 weather NetCDF files, found {len(paths)}")
    return paths


def masked_to_float_array(values) -> np.ndarray:
    arr = np.ma.asarray(values)
    return np.asarray(arr.filled(np.nan), dtype=float)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ws10"] = np.sqrt(out["u10"] ** 2 + out["v10"] ** 2)
    out["ws100"] = np.sqrt(out["u100"] ** 2 + out["v100"] ** 2)

    out["wd10_sin"] = np.where(out["ws10"] > 0, out["v10"] / out["ws10"], 0.0)
    out["wd10_cos"] = np.where(out["ws10"] > 0, out["u10"] / out["ws10"], 0.0)
    out["wd100_sin"] = np.where(out["ws100"] > 0, out["v100"] / out["ws100"], 0.0)
    out["wd100_cos"] = np.where(out["ws100"] > 0, out["u100"] / out["ws100"], 0.0)

    hour = out["valid_time"].dt.hour + out["valid_time"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2.0 * math.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * math.pi * hour / 24.0)

    doy = out["valid_time"].dt.dayofyear.astype(float)
    out["doy_sin"] = np.sin(2.0 * math.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2.0 * math.pi * doy / 365.25)

    lead_minutes = (out["valid_time"] - out["init_time"]).dt.total_seconds() / 60.0
    out["lead_minutes"] = lead_minutes
    out["lead_hours"] = lead_minutes / 60.0
    return out


def load_weather_next_day() -> tuple[pd.DataFrame, dict]:
    rows = []
    file_summaries = []
    nearest_grid = None

    for path in weather_paths():
        nc_path = path.relative_to(ROOT).as_posix()
        with Dataset(nc_path) as ds:
            missing_vars = [name for name in WEATHER_VARS if name not in ds.variables]
            if missing_vars:
                fail(f"{path} missing weather variables: {missing_vars}")

            for axis_name in ["latitude", "longitude", "time", "valid_time"]:
                if axis_name not in ds.variables:
                    fail(f"{path} missing axis variable: {axis_name}")

            latitudes = np.asarray(ds.variables["latitude"][:], dtype=float)
            longitudes = np.asarray(ds.variables["longitude"][:], dtype=float)
            if not (min(latitudes) <= STATION_LAT <= max(latitudes)):
                fail(f"Station latitude {STATION_LAT} outside grid in {path}")
            if not (min(longitudes) <= STATION_LON <= max(longitudes)):
                fail(f"Station longitude {STATION_LON} outside grid in {path}")

            lat_idx = int(np.abs(latitudes - STATION_LAT).argmin())
            lon_idx = int(np.abs(longitudes - STATION_LON).argmin())
            grid_info = {
                "latitude": float(latitudes[lat_idx]),
                "longitude": float(longitudes[lon_idx]),
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
            }
            if nearest_grid is None:
                nearest_grid = grid_info

            time_var = ds.variables["time"]
            init_value = np.asarray(time_var[...]).item()
            init_time = decode_nc_datetime(time_var, init_value)[0]

            valid_var = ds.variables["valid_time"]
            valid_times = decode_nc_datetime(valid_var, valid_var[:])
            next_day_start = (init_time + pd.Timedelta(days=1)).normalize()
            next_day_end = next_day_start + pd.Timedelta(hours=23, minutes=45)
            selected_indices = np.flatnonzero((valid_times >= next_day_start) & (valid_times <= next_day_end))

            if len(selected_indices) != 96:
                fail(f"{path.name}: expected 96 next-day steps, found {len(selected_indices)}")

            data = {
                "source_file": path.name,
                "init_time": pd.Series([init_time] * len(selected_indices), dtype="datetime64[ns]"),
                "valid_time": pd.Series(valid_times[selected_indices], dtype="datetime64[ns]"),
                "grid_lat": grid_info["latitude"],
                "grid_lon": grid_info["longitude"],
            }

            for name in WEATHER_VARS:
                data[name] = masked_to_float_array(ds.variables[name][selected_indices, lat_idx, lon_idx])

            frame = pd.DataFrame(data)
            rows.append(frame)
            file_summaries.append(
                {
                    "file": path.name,
                    "init_time": str(init_time),
                    "next_day_start": str(next_day_start),
                    "next_day_end": str(next_day_end),
                    "rows": int(len(frame)),
                }
            )

    weather = pd.concat(rows, ignore_index=True)
    weather = add_engineered_features(weather)
    summary = {
        "files": file_summaries,
        "nearest_grid": nearest_grid,
        "rows": int(len(weather)),
        "time_start": str(weather["valid_time"].min()),
        "time_end": str(weather["valid_time"].max()),
    }
    return weather, summary


def build_features(weather: pd.DataFrame, power: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    expected_rows = len(weather)
    data = weather.merge(power, on="valid_time", how="left")
    missing_label_rows = int(data["power_mw"].isna().sum())
    data = data.dropna(subset=FEATURE_COLUMNS + ["power_mw"]).copy()
    data = data.sort_values(["init_time", "valid_time"]).reset_index(drop=True)
    summary = {
        "expected_weather_rows": int(expected_rows),
        "rows_after_join": int(len(data)),
        "missing_label_rows": missing_label_rows,
        "dropped_rows_total": int(expected_rows - len(data)),
    }
    return data, summary


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE_MW": float(mae),
        "RMSE_MW": float(rmse),
        "nMAE": float(mae / CAPACITY_MW),
        "nRMSE": float(rmse / CAPACITY_MW),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
    }


def train_and_evaluate(data: pd.DataFrame) -> tuple[dict, pd.DataFrame, RandomForestRegressor]:
    train_mask = data["init_time"] <= TRAIN_INIT_END
    test_mask = data["init_time"] >= TEST_INIT_START

    train_df = data.loc[train_mask].copy()
    test_df = data.loc[test_mask].copy()
    if train_df.empty:
        fail("Training set is empty")
    if test_df.empty:
        fail("Test set is empty")

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["power_mw"]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["power_mw"]

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(x_train, y_train)
    y_dummy = dummy.predict(x_test)

    metrics = {
        "capacity_mw": CAPACITY_MW,
        "feature_columns": FEATURE_COLUMNS,
        "split": {
            "train_init_end": str(TRAIN_INIT_END),
            "test_init_start": str(TEST_INIT_START),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_init_times": [str(x) for x in sorted(train_df["init_time"].unique())],
            "test_init_times": [str(x) for x in sorted(test_df["init_time"].unique())],
        },
        "random_forest": metric_dict(y_test.to_numpy(), y_pred),
        "dummy_mean": metric_dict(y_test.to_numpy(), y_dummy),
    }
    metrics["random_forest_beats_dummy_rmse"] = (
        metrics["random_forest"]["RMSE_MW"] < metrics["dummy_mean"]["RMSE_MW"]
    )

    predictions = test_df[["source_file", "init_time", "valid_time", "lead_hours"]].copy()
    predictions["y_true"] = y_test.to_numpy()
    predictions["y_pred"] = y_pred
    predictions["y_dummy"] = y_dummy
    predictions["error_mw"] = predictions["y_pred"] - predictions["y_true"]
    return metrics, predictions, model


def write_plot(predictions: pd.DataFrame, path: Path) -> None:
    plot_df = predictions.sort_values("valid_time")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df["valid_time"], plot_df["y_true"], label="Actual", linewidth=1.8)
    ax.plot(plot_df["valid_time"], plot_df["y_pred"], label="RandomForest", linewidth=1.5)
    ax.plot(plot_df["valid_time"], plot_df["y_dummy"], label="Dummy mean", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Valid time")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Miaozhuang next-day power: actual vs predicted")
    ax.set_ylim(bottom=0, top=CAPACITY_MW * 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(
    path: Path,
    metrics: dict,
    weather_summary: dict,
    power_summary: dict,
    feature_summary: dict,
) -> None:
    rf = metrics["random_forest"]
    dummy = metrics["dummy_mean"]
    split = metrics["split"]
    beats_dummy = "优于" if metrics["random_forest_beats_dummy_rmse"] else "未优于"

    text = f"""# 苗庄小批量训练结果

## 数据口径
- 任务：每日 12:00 起报，预测次日 00:00-23:45 的 15 分钟全场功率。
- 气象输入：苗庄最近网格点，坐标约 `{weather_summary['nearest_grid']['longitude']:.4f}, {weather_summary['nearest_grid']['latitude']:.4f}`。
- 功率标签：`power_mw = clip(-(311 + 312 + 313 + 314), 0, 76)`。
- 装机容量：`76 MW`。

## 样本情况
- 气象次日样本：`{feature_summary['expected_weather_rows']}` 条。
- 可训练样本：`{feature_summary['rows_after_join']}` 条。
- 因功率标签缺失丢弃：`{feature_summary['missing_label_rows']}` 条。
- 训练集样本：`{split['train_rows']}` 条。
- 测试集样本：`{split['test_rows']}` 条。
- 测试起报日：`{', '.join(split['test_init_times'])}`。

## 评估结果
| 模型 | MAE(MW) | RMSE(MW) | nMAE | nRMSE | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | {rf['MAE_MW']:.4f} | {rf['RMSE_MW']:.4f} | {rf['nMAE']:.4f} | {rf['nRMSE']:.4f} | {rf['R2']:.4f} |
| DummyMean | {dummy['MAE_MW']:.4f} | {dummy['RMSE_MW']:.4f} | {dummy['nMAE']:.4f} | {dummy['nRMSE']:.4f} | {dummy['R2']:.4f} |

RandomForest 的 RMSE {beats_dummy} DummyMean 基线。

## 说明
当前只有 8 个气象起报日，适合验证读取、抽点、对齐、训练和评估流程，不适合作为正式精度结论。后续若要评估模型稳定性，需要补充更长时间的连续气象起报文件和同口径功率数据。
"""
    path.write_text(text, encoding="utf-8")


def save_outputs(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: dict,
    model: RandomForestRegressor,
    weather_summary: dict,
    power_summary: dict,
    feature_summary: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(OUTPUT_DIR / "features_next_day.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "predictions_next_day.csv", index=False, encoding="utf-8-sig")

    metrics_out = {
        **metrics,
        "weather_summary": weather_summary,
        "power_summary": power_summary,
        "feature_summary": feature_summary,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "capacity_mw": CAPACITY_MW,
            "station_lon": STATION_LON,
            "station_lat": STATION_LAT,
            "target_definition": "power_mw = clip(-(311 + 312 + 313 + 314), 0, 76)",
        },
        OUTPUT_DIR / "model_random_forest.joblib",
    )

    write_plot(predictions, OUTPUT_DIR / "actual_vs_pred.png")
    write_report(
        OUTPUT_DIR / "小批量训练结果.md",
        metrics,
        weather_summary,
        power_summary,
        feature_summary,
    )


def main() -> None:
    os.chdir(ROOT)

    print("Loading power labels...")
    power, power_summary = load_power()

    print("Loading weather forecasts and extracting station grid...")
    weather, weather_summary = load_weather_next_day()

    print("Joining features with labels...")
    data, feature_summary = build_features(weather, power)

    print("Training and evaluating sklearn baselines...")
    metrics, predictions, model = train_and_evaluate(data)

    print("Saving outputs...")
    save_outputs(data, predictions, metrics, model, weather_summary, power_summary, feature_summary)

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Rows: total={len(data)}, train={metrics['split']['train_rows']}, test={metrics['split']['test_rows']}")
    print(
        "RandomForest RMSE={:.4f} MW, Dummy RMSE={:.4f} MW".format(
            metrics["random_forest"]["RMSE_MW"],
            metrics["dummy_mean"]["RMSE_MW"],
        )
    )


if __name__ == "__main__":
    main()
