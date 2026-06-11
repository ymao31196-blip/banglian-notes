from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset, num2date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import train_mz_short_term_tree_baselines as baseline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "mz_windfm_10h_backtest"
DEFAULT_WINDFM_REPO = Path(tempfile.gettempdir()) / "WindFM-official"

CAPACITY_MW = 76.0
STATION_LAT = 39.40238056
STATION_LON = 117.8406861
FREQ = "15min"
FREQ_DELTA = pd.Timedelta(minutes=15)
FEATURE_COLUMNS = [
    "wind_speed",
    "wind_direction",
    "power",
    "density",
    "temperature",
    "pressure",
]
WEATHER_DIRS = ["2501_2502", "2503_2504", "2505_2506", "2507_2508"]


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small rolling-origin WindFM backtest for Miaozhuang."
    )
    parser.add_argument("--windfm-repo", type=Path, default=DEFAULT_WINDFM_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--pred-len", type=int, default=40)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--anchor-stride-hours", type=int, default=12)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scada-timezone", default="Asia/Shanghai")
    parser.add_argument("--nwp-timezone", default="UTC")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ["lookback", "pred_len", "sample_count", "anchor_stride_hours", "max_context"]:
        if getattr(args, name) <= 0:
            fail(f"--{name.replace('_', '-')} must be positive")
    if args.max_anchors < 0:
        fail("--max-anchors must be zero or positive")


def select_device(requested: str) -> str:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            fail(f"Requested device {requested}, but CUDA is unavailable")
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def localize_naive(series: pd.Series, timezone: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(
            timezone,
            ambiguous="NaT",
            nonexistent="shift_forward",
        ).dt.tz_convert("UTC")
    return parsed.dt.tz_convert("UTC")


def decode_nc_datetime(var, values) -> pd.DatetimeIndex:
    dates = num2date(
        values,
        units=var.units,
        calendar=getattr(var, "calendar", "standard"),
    )
    if np.ndim(dates) == 0:
        return pd.DatetimeIndex([pd.Timestamp(str(dates))])
    return pd.DatetimeIndex([pd.Timestamp(str(item)) for item in dates])


def masked_to_float(values) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        return np.asarray(values.filled(np.nan), dtype=float)
    return np.asarray(values, dtype=float)


def interpolate_isolated_time_gaps(series: pd.Series) -> pd.Series:
    missing = series.isna()
    isolated = (
        missing
        & ~missing.shift(1, fill_value=True)
        & ~missing.shift(-1, fill_value=True)
    )
    interpolated = series.interpolate(method="time", limit_area="inside")
    result = series.copy()
    result.loc[isolated] = interpolated.loc[isolated]
    return result


def weather_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in WEATHER_DIRS:
        paths.extend(sorted((ROOT / directory).glob("*.nc")))
    if len(paths) != 8:
        fail(f"Expected 8 Tianjin weather files, found {len(paths)}")
    return paths


def load_weather_asof(nwp_timezone: str) -> tuple[pd.DataFrame, dict]:
    rows: list[pd.DataFrame] = []
    file_summaries = []
    nearest_grid = None

    for path in weather_paths():
        nc_path = path.relative_to(ROOT).as_posix()
        with Dataset(nc_path) as ds:
            for name in ["latitude", "longitude", "time", "valid_time", "t2m", "msl"]:
                if name not in ds.variables:
                    fail(f"{path} is missing NetCDF variable {name}")

            latitudes = np.asarray(ds.variables["latitude"][:], dtype=float)
            longitudes = np.asarray(ds.variables["longitude"][:], dtype=float)
            if not latitudes.min() <= STATION_LAT <= latitudes.max():
                fail(f"Station latitude is outside {path.name}")
            if not longitudes.min() <= STATION_LON <= longitudes.max():
                fail(f"Station longitude is outside {path.name}")

            lat_idx = int(np.abs(latitudes - STATION_LAT).argmin())
            lon_idx = int(np.abs(longitudes - STATION_LON).argmin())
            grid = {
                "latitude": float(latitudes[lat_idx]),
                "longitude": float(longitudes[lon_idx]),
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
            }
            nearest_grid = nearest_grid or grid

            init_var = ds.variables["time"]
            init_value = np.asarray(init_var[...]).item()
            init_naive = decode_nc_datetime(init_var, init_value)[0]

            valid_var = ds.variables["valid_time"]
            valid_naive = decode_nc_datetime(valid_var, valid_var[:])
            frame = pd.DataFrame(
                {
                    "time_naive": valid_naive,
                    "init_time_naive": init_naive,
                    "temperature": masked_to_float(
                        ds.variables["t2m"][:, lat_idx, lon_idx]
                    ),
                    "pressure": masked_to_float(
                        ds.variables["msl"][:, lat_idx, lon_idx]
                    ),
                    "weather_source_file": path.name,
                }
            )
            frame["time"] = localize_naive(frame["time_naive"], nwp_timezone)
            frame["weather_init_time"] = localize_naive(
                frame["init_time_naive"], nwp_timezone
            )
            frame = frame.drop(columns=["time_naive", "init_time_naive"])
            rows.append(frame)
            file_summaries.append(
                {
                    "file": path.name,
                    "init_time_assumed_utc": str(frame["weather_init_time"].iloc[0]),
                    "valid_start_assumed_utc": str(frame["time"].min()),
                    "valid_end_assumed_utc": str(frame["time"].max()),
                }
            )

    weather_all = pd.concat(rows, ignore_index=True)
    weather_all = weather_all[
        weather_all["weather_init_time"] <= weather_all["time"]
    ].copy()
    weather_all = weather_all.sort_values(
        ["time", "weather_init_time", "weather_source_file"]
    )
    weather = weather_all.drop_duplicates(subset=["time"], keep="last").copy()
    weather["density"] = weather["pressure"] / (
        287.05 * weather["temperature"]
    )

    invalid = (
        weather[["temperature", "pressure", "density"]]
        .replace([np.inf, -np.inf], np.nan)
        .isna()
        .any(axis=1)
    )
    weather = weather.loc[~invalid].sort_values("time").reset_index(drop=True)

    summary = {
        "files": file_summaries,
        "nearest_grid": nearest_grid,
        "nwp_timezone_assumption": nwp_timezone,
        "rows_all_runs": int(len(weather_all)),
        "rows_asof": int(len(weather)),
        "time_start_utc": str(weather["time"].min()),
        "time_end_utc": str(weather["time"].max()),
        "last_init_time_utc": str(weather["weather_init_time"].max()),
        "temperature_min_k": float(weather["temperature"].min()),
        "temperature_max_k": float(weather["temperature"].max()),
        "pressure_min_pa": float(weather["pressure"].min()),
        "pressure_max_pa": float(weather["pressure"].max()),
        "density_min": float(weather["density"].min()),
        "density_max": float(weather["density"].max()),
    }
    return weather, summary


def load_station_data(scada_timezone: str) -> tuple[pd.DataFrame, dict]:
    power, power_summary = baseline.load_power()
    wind, wind_summary = baseline.load_wind_features()

    station = power[["time", "power_mw"]].merge(
        wind[
            [
                "time",
                "wind_speed_mean",
                "wind_dir_sin",
                "wind_dir_cos",
                "wind_speed_missing_ratio",
                "wind_dir_missing_ratio",
            ]
        ],
        on="time",
        how="inner",
    )
    station["time"] = localize_naive(station["time"], scada_timezone)
    station["wind_direction"] = (
        np.degrees(
            np.arctan2(station["wind_dir_sin"], station["wind_dir_cos"])
        )
        + 360.0
    ) % 360.0
    station = station.rename(
        columns={
            "power_mw": "power",
            "wind_speed_mean": "wind_speed",
        }
    )
    station = station.sort_values("time").drop_duplicates("time", keep="last")

    summary = {
        "scada_timezone_assumption": scada_timezone,
        "rows": int(len(station)),
        "time_start_utc": str(station["time"].min()),
        "time_end_utc": str(station["time"].max()),
        "power_summary": power_summary,
        "wind_summary": wind_summary,
    }
    return station, summary


def build_feature_table(
    station: pd.DataFrame,
    weather: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    start = max(station["time"].min(), weather["time"].min())
    end = min(station["time"].max(), weather["time"].max())
    full_index = pd.date_range(start, end, freq=FREQ, name="time")

    station_indexed = station.set_index("time").reindex(full_index)
    weather_indexed = weather.set_index("time").reindex(full_index)
    station_indexed["power_observed"] = station_indexed["power"].notna()
    station_indexed["wind_observed"] = station_indexed[
        ["wind_speed", "wind_direction"]
    ].notna().all(axis=1)
    weather_indexed["weather_observed"] = weather_indexed[
        ["temperature", "pressure", "density"]
    ].notna().all(axis=1)

    weather_cols = [
        "temperature",
        "pressure",
        "density",
        "weather_init_time",
        "weather_source_file",
        "weather_observed",
    ]
    data = station_indexed.join(weather_indexed[weather_cols], how="left")
    data["input_observed"] = (
        data["power_observed"]
        & data["wind_observed"]
        & data["weather_observed"]
    )

    for column in FEATURE_COLUMNS:
        data[column] = interpolate_isolated_time_gaps(data[column])
    data["weather_init_time"] = data["weather_init_time"].ffill().bfill()
    data["weather_source_file"] = data["weather_source_file"].ffill().bfill()
    data["input_imputed"] = ~data["input_observed"]
    data = data.reset_index()

    required = FEATURE_COLUMNS + ["time"]
    before = len(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    data = data[
        (data["power"] >= 0.0)
        & (data["power"] <= CAPACITY_MW)
        & (data["wind_speed"] >= 0.0)
        & (data["temperature"] > 150.0)
        & (data["pressure"] > 50_000.0)
        & (data["density"] > 0.5)
    ].copy()

    expected = pd.date_range(data["time"].min(), data["time"].max(), freq=FREQ)
    missing_timestamps = int(
        len(expected.difference(pd.DatetimeIndex(data["time"])))
    )
    summary = {
        "rows_before_quality_drop": int(before),
        "rows": int(len(data)),
        "dropped_rows": int(before - len(data)),
        "input_imputed_rows": int(data["input_imputed"].sum()),
        "power_unobserved_rows": int((~data["power_observed"]).sum()),
        "wind_unobserved_rows": int((~data["wind_observed"]).sum()),
        "weather_unobserved_rows": int((~data["weather_observed"]).sum()),
        "time_start_utc": str(data["time"].min()),
        "time_end_utc": str(data["time"].max()),
        "missing_15min_timestamps": missing_timestamps,
        "power_min_mw": float(data["power"].min()),
        "power_max_mw": float(data["power"].max()),
        "wind_speed_min": float(data["wind_speed"].min()),
        "wind_speed_max": float(data["wind_speed"].max()),
    }
    return data, summary


def import_windfm(repo: Path):
    repo = repo.resolve()
    if not (repo / "model" / "windfm.py").exists():
        fail(
            f"WindFM repository not found at {repo}. "
            "Clone https://github.com/shiyu-coder/WindFM.git first."
        )
    sys.path.insert(0, str(repo))
    try:
        from model import WindFM, WindFMPredictor, WindFMTokenizer
        from model.windfm import auto_regressive_inference
    except Exception as exc:
        fail(f"Could not import WindFM from {repo}: {exc}")
    return WindFM, WindFMTokenizer, WindFMPredictor, auto_regressive_inference


def make_fixed_predictor_class(base_class, auto_regressive_inference):
    class FixedWindFMPredictor(base_class):
        def generate(
            self,
            x,
            x_stamp,
            y_stamp,
            pred_len,
            T,
            top_k,
            top_p,
            sample_count,
            verbose,
        ):
            x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(
                self.device
            )
            x_stamp_tensor = torch.from_numpy(
                np.asarray(x_stamp, dtype=np.float32)
            ).to(self.device)
            y_stamp_tensor = torch.from_numpy(
                np.asarray(y_stamp, dtype=np.float32)
            ).to(self.device)

            preds = auto_regressive_inference(
                self.tokenizer,
                self.model,
                x_tensor,
                x_stamp_tensor,
                y_stamp_tensor,
                self.max_context,
                pred_len,
                self.clip,
                T,
                top_k,
                top_p,
                sample_count,
                verbose,
            )
            return preds[:, :, -pred_len:, :]

    return FixedWindFMPredictor


def ceil_to_stride(timestamp: pd.Timestamp, hours: int) -> pd.Timestamp:
    naive = timestamp.tz_convert("UTC").tz_localize(None)
    epoch = pd.Timestamp("1970-01-01")
    step = pd.Timedelta(hours=hours)
    units = math.ceil((naive - epoch) / step)
    return (epoch + units * step).tz_localize("UTC")


def build_anchors(
    data: pd.DataFrame,
    weather: pd.DataFrame,
    lookback: int,
    pred_len: int,
    stride_hours: int,
    max_anchors: int,
) -> list[pd.Timestamp]:
    earliest = data["time"].min() + (lookback - 1) * FREQ_DELTA
    latest_target = data["time"].max() - pred_len * FREQ_DELTA
    latest_weather_init = weather["weather_init_time"].max()
    latest = min(latest_target, latest_weather_init)
    start = ceil_to_stride(earliest, stride_hours)
    if start > latest:
        fail(f"No valid backtest anchors: earliest={start}, latest={latest}")

    candidates = list(pd.date_range(start, latest, freq=f"{stride_hours}h"))
    time_set = set(data["time"])
    valid = []
    for anchor in candidates:
        history_times = pd.date_range(
            anchor - (lookback - 1) * FREQ_DELTA,
            anchor,
            freq=FREQ,
        )
        future_times = pd.date_range(
            anchor + FREQ_DELTA,
            anchor + pred_len * FREQ_DELTA,
            freq=FREQ,
        )
        history_complete = all(t in time_set for t in history_times)
        future_complete = all(t in time_set for t in future_times)
        future_power_observed = (
            future_complete
            and data.set_index("time").loc[future_times, "power_observed"].all()
        )
        if history_complete and future_power_observed:
            valid.append(anchor)

    if max_anchors:
        valid = valid[:max_anchors]
    if not valid:
        fail("No anchors have complete history and future target windows")
    return valid


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pred = np.asarray(y_pred, dtype=float).clip(0.0, CAPACITY_MW)
    true = np.asarray(y_true, dtype=float)
    mae = float(mean_absolute_error(true, pred))
    rmse = float(math.sqrt(mean_squared_error(true, pred)))
    return {
        "MAE_MW": mae,
        "RMSE_MW": rmse,
        "nMAE": mae / CAPACITY_MW,
        "nRMSE": rmse / CAPACITY_MW,
        "R2": float(r2_score(true, pred)) if len(true) >= 2 else None,
        "accuracy_1_minus_nRMSE": 1.0 - rmse / CAPACITY_MW,
    }


def run_backtest(
    data: pd.DataFrame,
    anchors: list[pd.Timestamp],
    predictor,
    args: argparse.Namespace,
    device: str,
) -> tuple[pd.DataFrame, dict]:
    indexed = data.set_index("time").sort_index()
    rows = []
    window_seconds = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for index, anchor in enumerate(anchors, start=1):
        history_times = pd.date_range(
            anchor - (args.lookback - 1) * FREQ_DELTA,
            anchor,
            freq=FREQ,
        )
        future_times = pd.date_range(
            anchor + FREQ_DELTA,
            anchor + args.pred_len * FREQ_DELTA,
            freq=FREQ,
        )
        history = indexed.loc[history_times]
        future = indexed.loc[future_times]

        x_df = history[FEATURE_COLUMNS].copy()
        x_timestamp = pd.Series(history.index, name="time")
        y_timestamp = pd.Series(future.index, name="time")

        np.random.seed(args.random_seed + index)
        torch.manual_seed(args.random_seed + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed + index)

        start = time.perf_counter()
        pred_samples = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=args.pred_len,
            T=1.0,
            top_p=1.0,
            sample_count=args.sample_count,
            verbose=False,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        window_seconds.append(elapsed)

        if pred_samples.shape != (args.pred_len, args.sample_count):
            fail(
                f"Unexpected prediction shape {pred_samples.shape}; expected "
                f"({args.pred_len}, {args.sample_count})"
            )

        clipped = pred_samples.clip(lower=0.0, upper=CAPACITY_MW)
        quantiles = {
            "q05": clipped.quantile(0.05, axis=1).to_numpy(),
            "q25": clipped.quantile(0.25, axis=1).to_numpy(),
            "q50": clipped.quantile(0.50, axis=1).to_numpy(),
            "q75": clipped.quantile(0.75, axis=1).to_numpy(),
            "q95": clipped.quantile(0.95, axis=1).to_numpy(),
        }
        y_true = future["power"].to_numpy(dtype=float)
        persistence = np.full(args.pred_len, history["power"].iloc[-1])

        for step in range(args.pred_len):
            row = {
                "anchor_time_utc": anchor,
                "valid_time_utc": future.index[step],
                "anchor_time_beijing": anchor.tz_convert("Asia/Shanghai"),
                "valid_time_beijing": future.index[step].tz_convert(
                    "Asia/Shanghai"
                ),
                "lead_step": step + 1,
                "lead_minutes": (step + 1) * 15,
                "lead_hours": (step + 1) * 0.25,
                "y_true_mw": y_true[step],
                "y_pred_persistence_mw": persistence[step],
                "window_seconds": elapsed,
                **{name: values[step] for name, values in quantiles.items()},
            }
            rows.append(row)

        print(
            f"[{index}/{len(anchors)}] anchor={anchor} "
            f"seconds={elapsed:.2f} shape={pred_samples.shape}",
            flush=True,
        )

    predictions = pd.DataFrame(rows)
    runtime = {
        "windows": len(anchors),
        "seconds_total": float(sum(window_seconds)),
        "seconds_mean_per_window": float(np.mean(window_seconds)),
        "seconds_min_window": float(np.min(window_seconds)),
        "seconds_max_window": float(np.max(window_seconds)),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if device.startswith("cuda")
            else None
        ),
    }
    return predictions, runtime


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for model, column in [
        ("Persistence", "y_pred_persistence_mw"),
        ("WindFM_P50", "q50"),
    ]:
        rows.append(
            {
                "scope": "overall",
                "model": model,
                "rows": int(len(predictions)),
                **regression_metrics(
                    predictions["y_true_mw"].to_numpy(),
                    predictions[column].to_numpy(),
                ),
            }
        )

    windfm = rows[-1]
    windfm["P05_P95_coverage"] = float(
        (
            (predictions["y_true_mw"] >= predictions["q05"])
            & (predictions["y_true_mw"] <= predictions["q95"])
        ).mean()
    )
    windfm["P05_P95_mean_width_mw"] = float(
        (predictions["q95"] - predictions["q05"]).mean()
    )

    metrics_df = pd.DataFrame(rows)
    lead_rows = []
    for lead_minutes, group in predictions.groupby("lead_minutes"):
        for model, column in [
            ("Persistence", "y_pred_persistence_mw"),
            ("WindFM_P50", "q50"),
        ]:
            lead_rows.append(
                {
                    "lead_minutes": int(lead_minutes),
                    "lead_hours": float(lead_minutes / 60.0),
                    "model": model,
                    "rows": int(len(group)),
                    **regression_metrics(
                        group["y_true_mw"].to_numpy(),
                        group[column].to_numpy(),
                    ),
                }
            )
    segmented = predictions.copy()
    segmented["segment"] = pd.cut(
        segmented["lead_hours"],
        bins=[0.0, 2.0, 5.0, 10.0],
        labels=["0-2h", "2-5h", "5-10h"],
        include_lowest=True,
        right=True,
    )
    segment_rows = []
    for segment, group in segmented.groupby("segment", observed=True):
        for model, column in [
            ("Persistence", "y_pred_persistence_mw"),
            ("WindFM_P50", "q50"),
        ]:
            segment_rows.append(
                {
                    "segment": str(segment),
                    "model": model,
                    "rows": int(len(group)),
                    **regression_metrics(
                        group["y_true_mw"].to_numpy(),
                        group[column].to_numpy(),
                    ),
                }
            )
    return metrics_df, pd.DataFrame(lead_rows), pd.DataFrame(segment_rows)


def write_plot(
    predictions: pd.DataFrame,
    metrics_by_lead: pd.DataFrame,
    output_path: Path,
) -> None:
    first_anchor = predictions["anchor_time_utc"].min()
    example = predictions[predictions["anchor_time_utc"] == first_anchor]
    windfm_lead = metrics_by_lead[metrics_by_lead["model"] == "WindFM_P50"]
    persistence_lead = metrics_by_lead[
        metrics_by_lead["model"] == "Persistence"
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax = axes[0]
    x = example["lead_hours"].to_numpy()
    ax.fill_between(
        x,
        example["q05"].to_numpy(),
        example["q95"].to_numpy(),
        alpha=0.15,
        label="WindFM P05-P95",
    )
    ax.fill_between(
        x,
        example["q25"].to_numpy(),
        example["q75"].to_numpy(),
        alpha=0.25,
        label="WindFM P25-P75",
    )
    ax.plot(x, example["y_true_mw"], color="black", label="Actual")
    ax.plot(x, example["q50"], label="WindFM P50")
    ax.plot(
        x,
        example["y_pred_persistence_mw"],
        linestyle="--",
        label="Persistence",
    )
    ax.set_title(f"First rolling window: {first_anchor}")
    ax.set_ylabel("Power (MW)")
    ax.set_ylim(0.0, CAPACITY_MW * 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(
        windfm_lead["lead_hours"],
        windfm_lead["RMSE_MW"],
        marker="o",
        markersize=3,
        label="WindFM P50",
    )
    ax.plot(
        persistence_lead["lead_hours"],
        persistence_lead["RMSE_MW"],
        marker="o",
        markersize=3,
        label="Persistence",
    )
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("RMSE (MW)")
    ax.set_title("RMSE by forecast lead")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def format_metric(metrics: pd.DataFrame, model: str) -> str:
    row = metrics[metrics["model"] == model].iloc[0]
    return (
        f"MAE {row['MAE_MW']:.2f} MW，RMSE {row['RMSE_MW']:.2f} MW，"
        f"nRMSE {row['nRMSE']:.2%}，R2 {row['R2']:.3f}"
    )


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    device: str,
    anchors: list[pd.Timestamp],
    metrics: pd.DataFrame,
    metrics_by_segment: pd.DataFrame,
    runtime: dict,
    weather_summary: dict,
    station_summary: dict,
    feature_summary: dict,
) -> None:
    windfm_row = metrics[metrics["model"] == "WindFM_P50"].iloc[0]
    persistence_row = metrics[metrics["model"] == "Persistence"].iloc[0]
    better = windfm_row["RMSE_MW"] < persistence_row["RMSE_MW"]
    comparison = (
        "本轮 WindFM P50 的整体 RMSE 低于 Persistence。"
        if better
        else "本轮 WindFM P50 没有超过 Persistence。"
    )
    accuracy_note = (
        "文中的“1-nRMSE”仅按此前假设公式换算，不能在客户公式确认前"
        "直接视为正式准确率。"
    )
    timezone_note = (
        f"本轮将SCADA解释为{args.scada_timezone}，将NWP解释为"
        f"{args.nwp_timezone}，随后统一转换为UTC。该口径仍需数据提供方确认。"
    )
    segment_lines = []
    for segment in ["0-2h", "2-5h", "5-10h"]:
        windfm = metrics_by_segment[
            (metrics_by_segment["segment"] == segment)
            & (metrics_by_segment["model"] == "WindFM_P50")
        ].iloc[0]
        persistence = metrics_by_segment[
            (metrics_by_segment["segment"] == segment)
            & (metrics_by_segment["model"] == "Persistence")
        ].iloc[0]
        segment_lines.append(
            f"- {segment}：WindFM RMSE {windfm['RMSE_MW']:.2f} MW，"
            f"Persistence RMSE {persistence['RMSE_MW']:.2f} MW。"
        )

    lines = [
        "# 苗庄 WindFM 未来10小时小批量回测结果",
        "",
        "## 一、实验定位",
        "",
        "- 任务：使用过去60小时的功率、实测风速风向和气象温压，滚动预测未来10小时全场功率。",
        "- 粒度：15分钟，单个窗口预测40点。",
        f"- 回测窗口：{len(anchors)}个，起报间隔{args.anchor_stride_hours}小时。",
        f"- 起报范围：{anchors[0]} 至 {anchors[-1]}。",
        "- 模型：WindFM 零样本概率预测，不使用苗庄数据重新训练。",
        "- 对照：Persistence，即未来10小时功率均等于起报时刻功率。",
        "",
        "本实验用于验证苗庄数据适配和WindFM工作流程，不作为正式业务精度结论。",
        "",
        "## 二、总体结果",
        "",
        f"- WindFM P50：{format_metric(metrics, 'WindFM_P50')}。",
        f"- Persistence：{format_metric(metrics, 'Persistence')}。",
        f"- WindFM P05-P95覆盖率：{windfm_row['P05_P95_coverage']:.2%}。",
        f"- WindFM P05-P95平均区间宽度：{windfm_row['P05_P95_mean_width_mw']:.2f} MW。",
        f"- 初步判断：{comparison}",
        f"- 注意：{accuracy_note}",
        "",
        "### 分提前量结果",
        "",
        *segment_lines,
        "",
        "## 三、输入数据",
        "",
        "- 功率：311、312、313、314四路相加后取负号，并裁剪到0-76 MW。",
        "- 风速：19台风机一分钟实测风速聚合为15分钟场站均值。",
        "- 风向：19台风机风向先转sin/cos后做圆形平均。",
        "- 温度：天津区域NetCDF中苗庄最近格点的t2m。",
        "- 压力：天津区域NetCDF中苗庄最近格点的msl。",
        "- 空气密度：使用pressure / (287.05 × temperature)近似计算。",
        f"- 最近气象格点：{weather_summary['nearest_grid']}",
        f"- SCADA时区假设：{station_summary['scada_timezone_assumption']}。",
        f"- NWP时区假设：{weather_summary['nwp_timezone_assumption']}。",
        f"- 对齐后样本：{feature_summary['rows']}行，"
        f"{feature_summary['time_start_utc']} 至 {feature_summary['time_end_utc']}。",
        f"- 短缺口插值行数：{feature_summary['input_imputed_rows']}；"
        "未来评价标签仍要求原始功率观测存在。",
        "",
        "## 四、运行情况",
        "",
        f"- 运行设备：{device}。",
        f"- WindFM概率样本数：{args.sample_count}。",
        f"- 总推理时间：{runtime['seconds_total']:.2f}秒。",
        f"- 平均每个10小时窗口：{runtime['seconds_mean_per_window']:.2f}秒。",
        f"- 峰值CUDA显存：{runtime['peak_cuda_memory_mb']:.2f} MB。"
        if runtime["peak_cuda_memory_mb"] is not None
        else "- 使用CPU运行，无CUDA显存记录。",
        "",
        "## 五、已经修正的官方接口问题",
        "",
        "官方预测器在四维生成结果上裁剪了概率样本轴。项目脚本没有修改官方仓库，"
        "而是在本地预测器子类中将切片改为 `preds[:, :, -pred_len:, :]`，"
        "并强制校验最终输出必须为 `pred_len × sample_count`。",
        "",
        "## 六、结果边界",
        "",
        "1. 场站目录没有独立实测温度和气压，本轮使用的是区域预报文件中的t2m和msl。",
        "2. msl是海平面气压，不一定等同于WindFM预训练数据中的场站压力。",
        f"3. {timezone_note}",
        "4. 气象重复时只选择在该有效时刻之前已经起报、且起报时间最新的一份，"
        "避免使用未来发布的文件。",
        "5. 本轮只有2025年1月初的少量滚动窗口，不能据此判断跨季节稳定性。",
        "6. WindFM未来区间只输入时间戳，没有输入未来NWP，因此它不是当前10天短期预测的主模型。",
        "",
        "## 七、下一步",
        "",
        "1. 确认SCADA与NetCDF的时区，分别按“同一时区”和“SCADA北京时间、NWP UTC”复跑。",
        "2. 获取实测温度、气压或更合适的场站气象数据，替换当前近似输入。",
        "3. 扩展回测日期和季节，按0-2小时、2-5小时、5-10小时分别评价。",
        "4. 在同一批起报窗口上训练并评价NWP驱动的LightGBM/XGBoost，形成公平对比。",
        "5. 如果WindFM在部分场景有效，再研究与NWP主模型的加权集成和残差订正。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    print("Loading station power and turbine wind data...", flush=True)
    station, station_summary = load_station_data(args.scada_timezone)
    print("Loading and selecting as-of weather data...", flush=True)
    weather, weather_summary = load_weather_asof(args.nwp_timezone)
    data, feature_summary = build_feature_table(station, weather)
    data.to_csv(
        output_dir / "features_windfm_15min.csv",
        index=False,
        encoding="utf-8-sig",
    )

    anchors = build_anchors(
        data,
        weather,
        lookback=args.lookback,
        pred_len=args.pred_len,
        stride_hours=args.anchor_stride_hours,
        max_anchors=args.max_anchors,
    )
    print(f"Selected {len(anchors)} rolling anchors.", flush=True)

    WindFM, WindFMTokenizer, WindFMPredictor, inference = import_windfm(
        args.windfm_repo
    )
    FixedPredictor = make_fixed_predictor_class(WindFMPredictor, inference)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    tokenizer = WindFMTokenizer.from_pretrained("NeoQuasar/WindFM-Tokenizer")
    model = WindFM.from_pretrained("NeoQuasar/WindFM")
    predictor = FixedPredictor(
        model,
        tokenizer,
        device=device,
        max_context=args.max_context,
        clip=5,
    )

    predictions, runtime = run_backtest(
        data,
        anchors,
        predictor,
        args,
        device,
    )
    metrics, metrics_by_lead, metrics_by_segment = evaluate_predictions(
        predictions
    )

    predictions.to_csv(
        output_dir / "predictions_10h.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        output_dir / "metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_by_lead.to_csv(
        output_dir / "metrics_by_lead.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_by_segment.to_csv(
        output_dir / "metrics_by_segment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_plot(
        predictions,
        metrics_by_lead,
        output_dir / "windfm_10h_backtest.png",
    )

    run_summary = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": device,
        "anchors_utc": [str(anchor) for anchor in anchors],
        "runtime": runtime,
        "station": station_summary,
        "weather": weather_summary,
        "features": feature_summary,
        "metrics": metrics.replace({np.nan: None}).to_dict(orient="records"),
        "metrics_by_segment": metrics_by_segment.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir / "苗庄WindFM十小时回测结果.md",
        args,
        device,
        anchors,
        metrics,
        metrics_by_segment,
        runtime,
        weather_summary,
        station_summary,
        feature_summary,
    )

    print(metrics.to_string(index=False), flush=True)
    print(f"Outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
