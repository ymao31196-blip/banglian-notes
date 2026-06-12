from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from netCDF4 import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import run_mz_windfm_10h_backtest as windfm_data


ROOT = Path(__file__).resolve().parents[1]
WINDFM_OUTPUT = ROOT / "outputs" / "mz_windfm_10h_backtest"
DEFAULT_OUTPUT = ROOT / "outputs" / "mz_nwp_tree_same_windows"

CAPACITY_MW = 76.0
FREQ = "15min"
FREQ_DELTA = pd.Timedelta(minutes=15)
PRED_LEN = 40
POWER_LAGS = [1, 2, 4, 8, 16, 32, 96]
ROLL_WINDOWS = [4, 8, 16, 96]
WIND_ROLL_WINDOWS = [4, 16, 96]
NWP_VARIABLES = [
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
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train expanding-window NWP-driven LightGBM/XGBoost models and "
            "evaluate them on the exact WindFM rolling windows."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scada-timezone", default="Asia/Shanghai")
    parser.add_argument("--nwp-timezone", default="UTC")
    parser.add_argument("--nwp-latency-hours", type=float, default=0.0)
    parser.add_argument("--training-origin-stride-minutes", type=int, default=15)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.training_origin_stride_minutes <= 0:
        fail("--training-origin-stride-minutes must be positive")
    if args.training_origin_stride_minutes % 15 != 0:
        fail("--training-origin-stride-minutes must be a multiple of 15")
    if args.nwp_latency_hours < 0:
        fail("--nwp-latency-hours cannot be negative")


def load_test_windows() -> tuple[list[pd.Timestamp], pd.DataFrame]:
    summary_path = WINDFM_OUTPUT / "run_summary.json"
    prediction_path = WINDFM_OUTPUT / "predictions_10h.csv"
    if not summary_path.exists() or not prediction_path.exists():
        fail(
            "WindFM reference output is missing. Run "
            "script/run_mz_windfm_10h_backtest.py first."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    anchors = [pd.Timestamp(value) for value in summary["anchors_utc"]]
    reference = pd.read_csv(prediction_path)
    for column in [
        "anchor_time_utc",
        "valid_time_utc",
        "anchor_time_beijing",
        "valid_time_beijing",
    ]:
        reference[column] = pd.to_datetime(reference[column], utc=True)
    if len(reference) != len(anchors) * PRED_LEN:
        fail(
            f"WindFM reference has {len(reference)} rows; expected "
            f"{len(anchors) * PRED_LEN}"
        )
    return anchors, reference


def load_station_frame(scada_timezone: str) -> tuple[pd.DataFrame, dict]:
    station, summary = windfm_data.load_station_data(scada_timezone)
    start = station["time"].min()
    end = station["time"].max()
    full_index = pd.date_range(start, end, freq=FREQ, name="time")
    frame = station.set_index("time").reindex(full_index)
    frame["power_observed"] = frame["power"].notna()
    frame["wind_observed"] = frame[["wind_speed", "wind_direction"]].notna().all(
        axis=1
    )

    for column in ["power", "wind_speed", "wind_direction"]:
        frame[column] = windfm_data.interpolate_isolated_time_gaps(frame[column])

    radians = np.deg2rad(frame["wind_direction"] % 360.0)
    frame["station_wind_dir_sin"] = np.sin(radians)
    frame["station_wind_dir_cos"] = np.cos(radians)

    for lag in POWER_LAGS:
        frame[f"power_lag_{lag}"] = frame["power"].shift(lag)
    for window in ROLL_WINDOWS:
        rolling = frame["power"].rolling(window=window, min_periods=window)
        frame[f"power_roll_{window}_mean"] = rolling.mean()
        frame[f"power_roll_{window}_std"] = rolling.std()
        frame[f"power_roll_{window}_min"] = rolling.min()
        frame[f"power_roll_{window}_max"] = rolling.max()
    for window in WIND_ROLL_WINDOWS:
        rolling = frame["wind_speed"].rolling(window=window, min_periods=window)
        frame[f"station_wind_roll_{window}_mean"] = rolling.mean()
        frame[f"station_wind_roll_{window}_std"] = rolling.std()

    frame["power_current"] = frame["power"]
    frame["station_wind_speed"] = frame["wind_speed"]
    frame = frame.drop(columns=["wind_dir_sin", "wind_dir_cos"], errors="ignore")

    feature_columns = [
        "power_current",
        *[f"power_lag_{lag}" for lag in POWER_LAGS],
        *[
            f"power_roll_{window}_{stat}"
            for window in ROLL_WINDOWS
            for stat in ["mean", "std", "min", "max"]
        ],
        "station_wind_speed",
        "station_wind_dir_sin",
        "station_wind_dir_cos",
        *[
            f"station_wind_roll_{window}_{stat}"
            for window in WIND_ROLL_WINDOWS
            for stat in ["mean", "std"]
        ],
    ]
    summary = {
        **summary,
        "feature_columns": feature_columns,
        "rows_reindexed": int(len(frame)),
        "power_unobserved_rows": int((~frame["power_observed"]).sum()),
        "wind_unobserved_rows": int((~frame["wind_observed"]).sum()),
    }
    return frame, summary


def load_nwp_runs(nwp_timezone: str) -> tuple[pd.DataFrame, dict]:
    frames = []
    summaries = []
    nearest_grid = None

    for path in windfm_data.weather_paths():
        nc_path = path.relative_to(ROOT).as_posix()
        with Dataset(nc_path) as ds:
            missing = [name for name in NWP_VARIABLES if name not in ds.variables]
            if missing:
                fail(f"{path.name} is missing NWP variables: {missing}")

            latitudes = np.asarray(ds.variables["latitude"][:], dtype=float)
            longitudes = np.asarray(ds.variables["longitude"][:], dtype=float)
            lat_idx = int(
                np.abs(latitudes - windfm_data.STATION_LAT).argmin()
            )
            lon_idx = int(
                np.abs(longitudes - windfm_data.STATION_LON).argmin()
            )
            nearest_grid = nearest_grid or {
                "latitude": float(latitudes[lat_idx]),
                "longitude": float(longitudes[lon_idx]),
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
            }

            init_var = ds.variables["time"]
            init_value = np.asarray(init_var[...]).item()
            init_naive = windfm_data.decode_nc_datetime(init_var, init_value)[0]
            valid_var = ds.variables["valid_time"]
            valid_naive = windfm_data.decode_nc_datetime(
                valid_var, valid_var[:]
            )

            data = {
                "valid_time_naive": valid_naive,
                "init_time_naive": init_naive,
                "nwp_source_file": path.name,
            }
            for name in NWP_VARIABLES:
                data[name] = windfm_data.masked_to_float(
                    ds.variables[name][:, lat_idx, lon_idx]
                )
            frame = pd.DataFrame(data)
            frame["valid_time"] = windfm_data.localize_naive(
                frame["valid_time_naive"], nwp_timezone
            )
            frame["nwp_init_time"] = windfm_data.localize_naive(
                frame["init_time_naive"], nwp_timezone
            )
            frame = frame.drop(
                columns=["valid_time_naive", "init_time_naive"]
            )
            frame = frame.sort_values("valid_time")
            frame["ssrd_increment"] = frame["ssrd"].diff().clip(lower=0.0)
            frame["tp_increment"] = frame["tp"].diff().clip(lower=0.0)
            frame["ssrd_increment"] = frame["ssrd_increment"].fillna(
                frame["ssrd"].clip(lower=0.0)
            )
            frame["tp_increment"] = frame["tp_increment"].fillna(
                frame["tp"].clip(lower=0.0)
            )
            frames.append(frame)
            summaries.append(
                {
                    "file": path.name,
                    "init_time_utc": str(frame["nwp_init_time"].iloc[0]),
                    "valid_start_utc": str(frame["valid_time"].min()),
                    "valid_end_utc": str(frame["valid_time"].max()),
                    "rows": int(len(frame)),
                }
            )

    all_runs = pd.concat(frames, ignore_index=True)
    all_runs = all_runs.replace([np.inf, -np.inf], np.nan)
    all_runs = all_runs.dropna(subset=NWP_VARIABLES)
    all_runs["ws10"] = np.hypot(all_runs["u10"], all_runs["v10"])
    all_runs["ws100"] = np.hypot(all_runs["u100"], all_runs["v100"])

    for height, u_name, v_name in [
        ("10", "u10", "v10"),
        ("100", "u100", "v100"),
    ]:
        direction = (
            np.degrees(np.arctan2(-all_runs[u_name], -all_runs[v_name]))
            + 360.0
        ) % 360.0
        radians = np.deg2rad(direction)
        all_runs[f"wd{height}_sin"] = np.sin(radians)
        all_runs[f"wd{height}_cos"] = np.cos(radians)

    all_runs["t2m_c"] = all_runs["t2m"] - 273.15
    all_runs["d2m_c"] = all_runs["d2m"] - 273.15
    all_runs["temp_dew_spread"] = all_runs["t2m"] - all_runs["d2m"]
    all_runs["density_approx"] = all_runs["msl"] / (
        287.05 * all_runs["t2m"]
    )
    all_runs["nwp_lead_hours"] = (
        all_runs["valid_time"] - all_runs["nwp_init_time"]
    ).dt.total_seconds() / 3600.0

    summary = {
        "files": summaries,
        "nearest_grid": nearest_grid,
        "nwp_timezone_assumption": nwp_timezone,
        "rows": int(len(all_runs)),
        "init_times_utc": [
            str(value)
            for value in sorted(all_runs["nwp_init_time"].unique())
        ],
    }
    return all_runs.sort_values(
        ["nwp_init_time", "valid_time"]
    ).reset_index(drop=True), summary


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame["valid_time"]
    hour = valid.dt.hour + valid.dt.minute / 60.0
    day_of_year = valid.dt.dayofyear.astype(float)
    weekday = valid.dt.weekday.astype(float)
    frame["valid_hour_sin"] = np.sin(2.0 * math.pi * hour / 24.0)
    frame["valid_hour_cos"] = np.cos(2.0 * math.pi * hour / 24.0)
    frame["valid_doy_sin"] = np.sin(
        2.0 * math.pi * day_of_year / 365.25
    )
    frame["valid_doy_cos"] = np.cos(
        2.0 * math.pi * day_of_year / 365.25
    )
    frame["valid_weekday_sin"] = np.sin(
        2.0 * math.pi * weekday / 7.0
    )
    frame["valid_weekday_cos"] = np.cos(
        2.0 * math.pi * weekday / 7.0
    )
    return frame


def latest_available_run(
    nwp_runs: pd.DataFrame,
    origin: pd.Timestamp,
    latency_hours: float,
) -> pd.DataFrame | None:
    cutoff = origin - pd.Timedelta(hours=latency_hours)
    available = nwp_runs[nwp_runs["nwp_init_time"] <= cutoff]
    if available.empty:
        return None
    latest_init = available["nwp_init_time"].max()
    return available[available["nwp_init_time"] == latest_init].copy()


def build_rows_for_origin(
    origin: pd.Timestamp,
    station: pd.DataFrame,
    nwp_runs: pd.DataFrame,
    station_feature_columns: list[str],
    latency_hours: float,
) -> pd.DataFrame | None:
    if origin not in station.index:
        return None
    origin_row = station.loc[origin]
    if origin_row[station_feature_columns].isna().any():
        return None

    future_times = pd.date_range(
        origin + FREQ_DELTA,
        origin + PRED_LEN * FREQ_DELTA,
        freq=FREQ,
    )
    if not future_times.isin(station.index).all():
        return None
    targets = station.loc[future_times]
    if not targets["power_observed"].all():
        return None

    run = latest_available_run(nwp_runs, origin, latency_hours)
    if run is None:
        return None
    run = run.set_index("valid_time")
    if not future_times.isin(run.index).all():
        return None
    future_nwp = run.loc[future_times].reset_index(drop=True)
    future_nwp.insert(0, "valid_time", future_times)

    frame = future_nwp.copy()
    frame["origin_time"] = origin
    frame["target_power_mw"] = targets["power"].to_numpy(dtype=float)
    frame["lead_step"] = np.arange(1, PRED_LEN + 1)
    frame["lead_minutes"] = frame["lead_step"] * 15
    frame["lead_hours"] = frame["lead_minutes"] / 60.0
    frame["nwp_age_at_origin_hours"] = (
        origin - frame["nwp_init_time"]
    ).dt.total_seconds() / 3600.0

    for name in station_feature_columns:
        frame[name] = float(origin_row[name])
    frame = add_time_features(frame)
    return frame


def feature_columns(station_feature_columns: list[str]) -> list[str]:
    return [
        *station_feature_columns,
        *NWP_VARIABLES,
        "ssrd_increment",
        "tp_increment",
        "ws10",
        "ws100",
        "wd10_sin",
        "wd10_cos",
        "wd100_sin",
        "wd100_cos",
        "t2m_c",
        "d2m_c",
        "temp_dew_spread",
        "density_approx",
        "lead_step",
        "lead_minutes",
        "lead_hours",
        "nwp_lead_hours",
        "nwp_age_at_origin_hours",
        "valid_hour_sin",
        "valid_hour_cos",
        "valid_doy_sin",
        "valid_doy_cos",
        "valid_weekday_sin",
        "valid_weekday_cos",
    ]


def build_training_pool(
    station: pd.DataFrame,
    nwp_runs: pd.DataFrame,
    test_anchors: list[pd.Timestamp],
    station_feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    first_run = nwp_runs["nwp_init_time"].min()
    earliest_station = station.dropna(
        subset=station_feature_columns
    ).index.min()
    start = max(first_run, earliest_station)
    end = max(test_anchors) - FREQ_DELTA
    origins = pd.date_range(
        start,
        end,
        freq=f"{args.training_origin_stride_minutes}min",
    )

    frames = []
    skipped = 0
    for origin in origins:
        frame = build_rows_for_origin(
            origin,
            station,
            nwp_runs,
            station_feature_columns,
            args.nwp_latency_hours,
        )
        if frame is None:
            skipped += 1
        else:
            frames.append(frame)
    if not frames:
        fail("No training rows could be generated")

    pool = pd.concat(frames, ignore_index=True)
    summary = {
        "origin_start_utc": str(start),
        "origin_end_utc": str(end),
        "origin_candidates": int(len(origins)),
        "origin_windows_built": int(len(frames)),
        "origin_windows_skipped": int(skipped),
        "rows": int(len(pool)),
        "target_start_utc": str(pool["valid_time"].min()),
        "target_end_utc": str(pool["valid_time"].max()),
    }
    return pool, summary


def build_test_frame(
    station: pd.DataFrame,
    nwp_runs: pd.DataFrame,
    test_anchors: list[pd.Timestamp],
    station_feature_columns: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    frames = []
    for anchor in test_anchors:
        frame = build_rows_for_origin(
            anchor,
            station,
            nwp_runs,
            station_feature_columns,
            args.nwp_latency_hours,
        )
        if frame is None:
            fail(f"Could not build complete test features for {anchor}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def model_factories(random_seed: int) -> dict[str, object]:
    return {
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=25,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.2,
            random_state=random_seed,
            n_jobs=-1,
            verbosity=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.035,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float).clip(0.0, CAPACITY_MW)
    mae = float(mean_absolute_error(true, pred))
    rmse = float(math.sqrt(mean_squared_error(true, pred)))
    return {
        "MAE_MW": mae,
        "RMSE_MW": rmse,
        "nMAE": mae / CAPACITY_MW,
        "nRMSE": rmse / CAPACITY_MW,
        "R2": float(r2_score(true, pred)),
        "accuracy_1_minus_nRMSE": 1.0 - rmse / CAPACITY_MW,
    }


def train_expanding_models(
    training_pool: pd.DataFrame,
    test_frame: pd.DataFrame,
    test_anchors: list[pd.Timestamp],
    features: list[str],
    output_dir: Path,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    training_summaries = []
    importance_rows = []

    for anchor_index, anchor in enumerate(test_anchors, start=1):
        train = training_pool[
            (training_pool["origin_time"] < anchor)
            & (training_pool["valid_time"] <= anchor)
        ].copy()
        test = test_frame[test_frame["origin_time"] == anchor].copy()
        if train.empty or len(test) != PRED_LEN:
            fail(
                f"Invalid split for {anchor}: train={len(train)}, "
                f"test={len(test)}"
            )

        x_train = train[features]
        y_train = train["target_power_mw"].to_numpy()
        x_test = test[features]
        anchor_predictions = test[
            [
                "origin_time",
                "valid_time",
                "lead_step",
                "lead_minutes",
                "lead_hours",
                "nwp_init_time",
                "nwp_source_file",
                "target_power_mw",
                "power_current",
            ]
        ].copy()

        started = time.perf_counter()
        fitted = model_factories(random_seed + anchor_index)
        model_times = {}
        for model_name, model in fitted.items():
            model_start = time.perf_counter()
            model.fit(x_train, y_train)
            model_times[model_name] = time.perf_counter() - model_start
            anchor_predictions[f"y_pred_{model_name}"] = np.asarray(
                model.predict(x_test), dtype=float
            ).clip(0.0, CAPACITY_MW)
            joblib.dump(
                model,
                output_dir
                / f"model_{model_name.lower()}_{anchor.strftime('%Y%m%d%H%M')}.joblib",
            )

            importance = getattr(model, "feature_importances_", None)
            if importance is not None:
                for feature, value in zip(features, importance):
                    importance_rows.append(
                        {
                            "anchor_time_utc": anchor,
                            "model": model_name,
                            "feature": feature,
                            "importance": float(value),
                        }
                    )

        elapsed = time.perf_counter() - started
        anchor_predictions["training_rows"] = len(train)
        predictions.append(anchor_predictions)
        training_summaries.append(
            {
                "anchor_time_utc": anchor,
                "training_rows": int(len(train)),
                "training_origin_count": int(train["origin_time"].nunique()),
                "training_target_start_utc": str(train["valid_time"].min()),
                "training_target_end_utc": str(train["valid_time"].max()),
                "elapsed_seconds": elapsed,
                **{
                    f"{name}_seconds": seconds
                    for name, seconds in model_times.items()
                },
            }
        )
        print(
            f"[{anchor_index}/{len(test_anchors)}] anchor={anchor} "
            f"train_rows={len(train)} elapsed={elapsed:.2f}s",
            flush=True,
        )

    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(training_summaries),
        pd.DataFrame(importance_rows),
    )


def merge_reference_predictions(
    tree_predictions: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    reference_columns = [
        "anchor_time_utc",
        "valid_time_utc",
        "y_true_mw",
        "q50",
        "q05",
        "q95",
        "y_pred_persistence_mw",
    ]
    ref = reference[reference_columns].rename(
        columns={
            "anchor_time_utc": "origin_time",
            "valid_time_utc": "valid_time",
            "q50": "y_pred_WindFM_P50",
            "q05": "windfm_q05",
            "q95": "windfm_q95",
        }
    )
    merged = tree_predictions.merge(
        ref,
        on=["origin_time", "valid_time"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(reference):
        fail(
            f"Prediction merge returned {len(merged)} rows; "
            f"expected {len(reference)}"
        )
    if not np.allclose(
        merged["target_power_mw"],
        merged["y_true_mw"],
        atol=1e-8,
    ):
        fail("Tree and WindFM target power values do not match")
    return merged


def evaluate(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_columns = {
        "Persistence": "y_pred_persistence_mw",
        "WindFM_P50": "y_pred_WindFM_P50",
        "LightGBM": "y_pred_LightGBM",
        "XGBoost": "y_pred_XGBoost",
    }
    overall_rows = []
    for model, column in model_columns.items():
        overall_rows.append(
            {
                "model": model,
                "rows": int(len(predictions)),
                **metrics(
                    predictions["y_true_mw"].to_numpy(),
                    predictions[column].to_numpy(),
                ),
            }
        )
    overall = pd.DataFrame(overall_rows).sort_values("RMSE_MW")

    segment_frame = predictions.copy()
    segment_frame["segment"] = pd.cut(
        segment_frame["lead_hours"],
        bins=[0.0, 2.0, 5.0, 10.0],
        labels=["0-2h", "2-5h", "5-10h"],
        include_lowest=True,
    )
    segment_rows = []
    for segment, group in segment_frame.groupby("segment", observed=True):
        for model, column in model_columns.items():
            segment_rows.append(
                {
                    "segment": str(segment),
                    "model": model,
                    "rows": int(len(group)),
                    **metrics(
                        group["y_true_mw"].to_numpy(),
                        group[column].to_numpy(),
                    ),
                }
            )

    lead_rows = []
    for lead_minutes, group in predictions.groupby("lead_minutes"):
        for model, column in model_columns.items():
            lead_rows.append(
                {
                    "lead_minutes": int(lead_minutes),
                    "lead_hours": float(lead_minutes / 60.0),
                    "model": model,
                    "rows": int(len(group)),
                    **metrics(
                        group["y_true_mw"].to_numpy(),
                        group[column].to_numpy(),
                    ),
                }
            )
    return overall, pd.DataFrame(segment_rows), pd.DataFrame(lead_rows)


def aggregate_importance(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    normalized = raw.copy()
    totals = normalized.groupby(["anchor_time_utc", "model"])[
        "importance"
    ].transform("sum")
    normalized["importance_normalized"] = np.where(
        totals > 0.0,
        normalized["importance"] / totals,
        0.0,
    )
    return (
        normalized.groupby(["model", "feature"], as_index=False)
        .agg(
            importance_mean=("importance_normalized", "mean"),
            importance_std=("importance_normalized", "std"),
        )
        .sort_values(["model", "importance_mean"], ascending=[True, False])
    )


def write_plot(
    predictions: pd.DataFrame,
    metrics_by_lead: pd.DataFrame,
    output_path: Path,
) -> None:
    first_anchor = predictions["origin_time"].min()
    example = predictions[predictions["origin_time"] == first_anchor]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    axes[0].plot(
        example["lead_hours"],
        example["y_true_mw"],
        color="black",
        linewidth=1.8,
        label="Actual",
    )
    for model, column in [
        ("Persistence", "y_pred_persistence_mw"),
        ("WindFM P50", "y_pred_WindFM_P50"),
        ("LightGBM", "y_pred_LightGBM"),
        ("XGBoost", "y_pred_XGBoost"),
    ]:
        axes[0].plot(example["lead_hours"], example[column], label=model)
    axes[0].set_title(f"Same-window comparison: {first_anchor}")
    axes[0].set_ylabel("Power (MW)")
    axes[0].set_ylim(0.0, CAPACITY_MW * 1.05)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    for model in ["Persistence", "WindFM_P50", "LightGBM", "XGBoost"]:
        frame = metrics_by_lead[metrics_by_lead["model"] == model]
        axes[1].plot(
            frame["lead_hours"],
            frame["RMSE_MW"],
            marker="o",
            markersize=3,
            label=model,
        )
    axes[1].set_title("RMSE by forecast lead")
    axes[1].set_xlabel("Lead time (hours)")
    axes[1].set_ylabel("RMSE (MW)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def metric_text(overall: pd.DataFrame, model: str) -> str:
    row = overall[overall["model"] == model].iloc[0]
    return (
        f"MAE {row['MAE_MW']:.2f} MW，RMSE {row['RMSE_MW']:.2f} MW，"
        f"nRMSE {row['nRMSE']:.2%}，R2 {row['R2']:.3f}"
    )


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    anchors: list[pd.Timestamp],
    overall: pd.DataFrame,
    segment: pd.DataFrame,
    pool_summary: dict,
    training_summary: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    best = overall.iloc[0]
    lines = [
        "# 苗庄同窗口 NWP 树模型与 WindFM 对比结果",
        "",
        "## 一、实验设计",
        "",
        "- 测试窗口：与WindFM完全相同的6个起报时刻、240个未来功率点。",
        "- 预测长度：未来10小时，15分钟粒度，共40步。",
        "- 模型：Persistence、WindFM P50、LightGBM、XGBoost。",
        "- 树模型输入：起报时刻之前的功率与场站风况，以及未来40步NWP。",
        "- 训练方式：按测试起报时刻扩展训练，每次只使用当时已经发生的实际功率标签。",
        f"- 训练起点：{pool_summary['origin_start_utc']}。",
        f"- NWP假设时区：{args.nwp_timezone}；SCADA假设时区：{args.scada_timezone}。",
        f"- NWP发布延迟假设：{args.nwp_latency_hours:.2f}小时。",
        "",
        "该设计保证四种模型使用相同测试目标，但树模型能够使用未来NWP，"
        "用于检验WindFM零样本历史自回归是否提供额外价值。",
        "",
        "## 二、总体结果",
        "",
        "| 模型 | MAE | RMSE | nRMSE | R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ["Persistence", "WindFM_P50", "LightGBM", "XGBoost"]:
        row = overall[overall["model"] == model].iloc[0]
        lines.append(
            f"| {model} | {row['MAE_MW']:.2f} MW | "
            f"{row['RMSE_MW']:.2f} MW | {row['nRMSE']:.2%} | "
            f"{row['R2']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- 最低整体RMSE：{best['model']}，{best['RMSE_MW']:.2f} MW。",
            "- 指标只代表2025年1月初的小批量同窗口实验，不代表正式业务准确率。",
            "",
            "## 三、分提前量结果",
            "",
            "| 提前量 | Persistence | WindFM | LightGBM | XGBoost |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for segment_name in ["0-2h", "2-5h", "5-10h"]:
        values = {}
        for model in ["Persistence", "WindFM_P50", "LightGBM", "XGBoost"]:
            row = segment[
                (segment["segment"] == segment_name)
                & (segment["model"] == model)
            ].iloc[0]
            values[model] = row["RMSE_MW"]
        lines.append(
            f"| {segment_name} | {values['Persistence']:.2f} MW | "
            f"{values['WindFM_P50']:.2f} MW | "
            f"{values['LightGBM']:.2f} MW | "
            f"{values['XGBoost']:.2f} MW |"
        )

    lines.extend(
        [
            "",
            "## 四、训练样本",
            "",
            f"- 可构建训练起报窗口：{pool_summary['origin_windows_built']}个。",
            f"- 训练池总行数：{pool_summary['rows']}行。",
            f"- 第一个测试窗口训练行数：{int(training_summary['training_rows'].min())}行。",
            f"- 最后一个测试窗口训练行数：{int(training_summary['training_rows'].max())}行。",
            "- 每个测试时刻单独重新拟合模型，后面的窗口可以使用此前已经发生的真实功率。",
            "",
            "## 五、主要特征",
            "",
            "- 历史功率：当前值、15分钟至24小时滞后、滚动均值/标准差/最小值/最大值。",
            "- 场站风况：当前实测风速风向及滚动统计。",
            "- 未来NWP：10米/100米风矢量、阵风、温度、露点、辐射、降水、气压。",
            "- 派生变量：未来风速、风向sin/cos、空气密度、温露差、提前量和时间周期。",
            "",
            "### 平均特征重要性前十",
            "",
        ]
    )
    for model in ["LightGBM", "XGBoost"]:
        lines.extend([f"**{model}**", ""])
        top = importance[importance["model"] == model].head(10)
        for _, row in top.iterrows():
            lines.append(
                f"- `{row['feature']}`：{row['importance_mean']:.4f}"
            )
        lines.append("")

    lines.extend(
        [
            "## 六、结果边界",
            "",
            "1. 只有8个NWP起报文件，时间范围很短，树模型训练样本高度重叠。",
            "2. 默认假设NWP在名义起报时刻即可获得，尚未加入企业侧实际文件到达延迟。",
            "3. 训练采用滚动扩展方式，后续窗口比首个窗口拥有更多已发生数据。",
            "4. 多步预测共用一个模型，通过lead_hours区分提前量，尚未按40个提前量分别建模。",
            "5. LightGBM和XGBoost使用固定参数，没有用测试窗口调参。",
            "6. 客户准确率公式仍未确认，报告以MAE、RMSE和nRMSE为主。",
            "",
            "## 七、下一步",
            "",
            "1. 确认NWP文件实际到达延迟，并增加1-6小时延迟敏感性实验。",
            "2. 扩大到至少1-3个月连续NWP起报文件，减少窗口重叠造成的偶然性。",
            "3. 分别训练0-2小时、2-5小时、5-10小时模型，比较统一模型与分段模型。",
            "4. 用验证集确定LightGBM/XGBoost参数，不在测试窗口上调参。",
            "5. 在树模型基础上加入WindFM P50、区间宽度等特征，验证模型融合是否有效。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WindFM reference windows...", flush=True)
    anchors, reference = load_test_windows()
    print("Loading station history...", flush=True)
    station, station_summary = load_station_frame(args.scada_timezone)
    station_features = station_summary["feature_columns"]
    print("Loading full NWP forecast runs...", flush=True)
    nwp_runs, nwp_summary = load_nwp_runs(args.nwp_timezone)

    print("Building origin/lead training pool...", flush=True)
    training_pool, pool_summary = build_training_pool(
        station,
        nwp_runs,
        anchors,
        station_features,
        args,
    )
    test_frame = build_test_frame(
        station,
        nwp_runs,
        anchors,
        station_features,
        args,
    )
    features = feature_columns(station_features)
    training_pool.to_csv(
        output_dir / "training_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_frame.to_csv(
        output_dir / "test_features_same_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions, training_summary, raw_importance = train_expanding_models(
        training_pool,
        test_frame,
        anchors,
        features,
        output_dir,
        args.random_seed,
    )
    merged = merge_reference_predictions(predictions, reference)
    overall, segment, lead = evaluate(merged)
    importance = aggregate_importance(raw_importance)

    merged.to_csv(
        output_dir / "predictions_same_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall.to_csv(
        output_dir / "metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    segment.to_csv(
        output_dir / "metrics_by_segment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lead.to_csv(
        output_dir / "metrics_by_lead.csv",
        index=False,
        encoding="utf-8-sig",
    )
    training_summary.to_csv(
        output_dir / "training_summary_by_anchor.csv",
        index=False,
        encoding="utf-8-sig",
    )
    importance.to_csv(
        output_dir / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_plot(
        merged,
        lead,
        output_dir / "same_window_model_comparison.png",
    )
    write_report(
        output_dir / "同窗口NWP树模型对比结果.md",
        args,
        anchors,
        overall,
        segment,
        pool_summary,
        training_summary,
        importance,
    )

    run_summary = {
        "arguments": vars(args)
        | {"output_dir": str(output_dir)},
        "test_anchors_utc": [str(anchor) for anchor in anchors],
        "features": features,
        "station": station_summary,
        "nwp": nwp_summary,
        "training_pool": pool_summary,
        "training_by_anchor": training_summary.to_dict(orient="records"),
        "metrics": overall.to_dict(orient="records"),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(overall.to_string(index=False), flush=True)
    print(f"Outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
