# -*- coding: utf-8 -*-
"""AutoGluon-TimeSeries rolling-window baselines for Miaozhuang.

This script uses the already aligned short-term feature table produced by
train_mz_short_term_tree_baselines.py. It builds many rolling windows:

- context: previous 96 points, i.e. 24 hours at 15-minute resolution
- target: future 15min, 30min, and 60min power

The first run is intentionally target-only. It tests whether AutoGluon can beat
the strong persistence baseline before adding more covariates.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "outputs" / "mz_short_term_tree_baselines" / "features_short_term.csv"
OUTPUT_DIR = ROOT / "outputs" / "mz_autogluon_timeseries"

CAPACITY_MW = 76.0
FREQ = "15min"
CONTEXT_STEPS = 96
TRAIN_STRIDE = 4
TEST_STRIDE = 1
TIME_LIMIT_PER_HORIZON = 180

TRAIN_END = pd.Timestamp("2025-06-01 00:00:00")
TEST_START = pd.Timestamp("2025-06-16 00:00:00")
TEST_END = pd.Timestamp("2025-07-01 00:00:00")

HORIZONS = {
    "15min": 1,
    "30min": 2,
    "60min": 4,
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_series() -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        fail(f"Feature table not found: {FEATURE_PATH}")

    data = pd.read_csv(FEATURE_PATH, usecols=["time", "power_mw"], parse_dates=["time"])
    data = data.dropna(subset=["time", "power_mw"]).sort_values("time").reset_index(drop=True)
    if data.empty:
        fail("Feature table has no usable power rows.")
    return data


def clipped(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=float).clip(0.0, CAPACITY_MW)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = clipped(y_pred)
    mae = mean_absolute_error(y_true, pred)
    rmse = math.sqrt(mean_squared_error(y_true, pred))
    return {
        "MAE_MW": float(mae),
        "RMSE_MW": float(rmse),
        "nMAE": float(mae / CAPACITY_MW),
        "nRMSE": float(rmse / CAPACITY_MW),
        "R2": float(r2_score(y_true, pred)) if len(y_true) >= 2 else float("nan"),
    }


def anchor_indices(data: pd.DataFrame, split: str, horizon_steps: int) -> list[int]:
    times = data["time"]
    last_index = len(data) - horizon_steps - 1
    candidates = range(CONTEXT_STEPS - 1, last_index + 1)

    if split == "train":
        selected = [i for i in candidates if times.iloc[i] < TRAIN_END]
        return selected[::TRAIN_STRIDE]
    if split == "test":
        selected = [i for i in candidates if TEST_START <= times.iloc[i] < TEST_END]
        return selected[::TEST_STRIDE]
    fail(f"Unknown split: {split}")


def build_window_frame(
    data: pd.DataFrame,
    anchors: list[int],
    horizon_steps: int,
    include_future: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta_rows = []
    base_time = pd.Timestamp("2000-01-01 00:00:00")
    y = data["power_mw"].to_numpy(dtype=float)
    times = data["time"].to_numpy()
    length = CONTEXT_STEPS + (horizon_steps if include_future else 0)

    for anchor in anchors:
        item_id = f"w_{anchor}"
        start = anchor - CONTEXT_STEPS + 1
        for offset in range(length):
            source_index = start + offset
            rows.append(
                {
                    "item_id": item_id,
                    "timestamp": base_time + pd.Timedelta(minutes=15 * offset),
                    "target": float(y[source_index]),
                }
            )

        meta_rows.append(
            {
                "item_id": item_id,
                "anchor_index": int(anchor),
                "anchor_time": pd.Timestamp(times[anchor]),
                "target_time": pd.Timestamp(times[anchor + horizon_steps]),
                "power_current": float(y[anchor]),
                "y_true": float(y[anchor + horizon_steps]),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(meta_rows)


def to_tsdf(frame: pd.DataFrame) -> TimeSeriesDataFrame:
    return TimeSeriesDataFrame.from_data_frame(
        frame,
        id_column="item_id",
        timestamp_column="timestamp",
    )


def extract_last_step_predictions(pred: TimeSeriesDataFrame, horizon_steps: int) -> pd.DataFrame:
    pred_df = pred.reset_index()
    mean_col = "mean" if "mean" in pred_df.columns else pred_df.columns[-1]
    pred_df = pred_df.sort_values(["item_id", "timestamp"])
    last = pred_df.groupby("item_id", as_index=False).tail(1)
    return last[["item_id", mean_col]].rename(columns={mean_col: "y_pred"})


def plot_predictions(predictions: pd.DataFrame, horizon_name: str, output_path: Path) -> None:
    plot_df = predictions.sort_values("target_time")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df["target_time"], plot_df["y_true"], label="Actual", linewidth=1.7)
    ax.plot(plot_df["target_time"], plot_df["y_pred_persistence"], label="Persistence", linewidth=1.2)
    ax.plot(plot_df["target_time"], plot_df["y_pred"], label="AutoGluon", linewidth=1.3)
    ax.set_title(f"Miaozhuang AutoGluon-TimeSeries ({horizon_name})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (MW)")
    ax.set_ylim(bottom=0.0, top=CAPACITY_MW * 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_horizon(data: pd.DataFrame, horizon_name: str, horizon_steps: int) -> tuple[dict, pd.DataFrame]:
    train_anchors = anchor_indices(data, "train", horizon_steps)
    test_anchors = anchor_indices(data, "test", horizon_steps)
    if not train_anchors or not test_anchors:
        fail(f"{horizon_name}: empty train or test anchors")

    train_frame, train_meta = build_window_frame(data, train_anchors, horizon_steps, include_future=True)
    test_context, test_meta = build_window_frame(data, test_anchors, horizon_steps, include_future=False)

    horizon_dir = OUTPUT_DIR / f"predictor_{horizon_name}"
    if horizon_dir.exists():
        shutil.rmtree(horizon_dir)

    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=horizon_steps,
        freq=FREQ,
        eval_metric="RMSE",
        path=horizon_dir,
        verbosity=2,
    )
    predictor.fit(
        to_tsdf(train_frame),
        presets="fast_training",
        time_limit=TIME_LIMIT_PER_HORIZON,
        random_seed=42,
        num_val_windows=1,
        refit_every_n_windows=1,
        enable_ensemble=True,
    )

    pred = predictor.predict(to_tsdf(test_context))
    pred_last = extract_last_step_predictions(pred, horizon_steps)
    predictions = test_meta.merge(pred_last, on="item_id", how="left")
    predictions["y_pred"] = clipped(predictions["y_pred"])
    predictions["y_pred_persistence"] = clipped(predictions["power_current"])
    predictions["horizon"] = horizon_name

    auto_metrics = metrics(predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy())
    persistence_metrics = metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_pred_persistence"].to_numpy(),
    )

    leaderboard = predictor.leaderboard(silent=True)
    leaderboard.to_csv(OUTPUT_DIR / f"leaderboard_{horizon_name}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "horizon": horizon_name,
        "horizon_steps": int(horizon_steps),
        "train_windows": int(len(train_meta)),
        "test_windows": int(len(test_meta)),
        "train_rows": int(len(train_frame)),
        "test_context_rows": int(len(test_context)),
        "autogluon": auto_metrics,
        "persistence": persistence_metrics,
        "best_model": str(leaderboard.iloc[0]["model"]) if not leaderboard.empty else None,
    }
    return summary, predictions


def write_report(results: list[dict], output_path: Path) -> None:
    lines = [
        "# AutoGluon-TimeSeries 实验结果",
        "",
        "## 口径",
        "",
        "- 使用已有短期特征表中的 `power_mw`，先做 target-only 时间序列基线。",
        "- 每个样本用过去 24 小时功率窗口，预测未来 15min、30min、60min。",
        "- 训练窗口按 1 小时间隔抽样，测试窗口按 15 分钟全量滚动。",
        "- 本轮不使用未来实测风速/风向，避免数据泄漏。",
        "",
        "## 测试集指标",
        "",
        "| Horizon | AutoGluon RMSE | Persistence RMSE | AutoGluon nRMSE | 最优内部模型 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for row in results:
        ag = row["autogluon"]
        ps = row["persistence"]
        lines.append(
            f"| {row['horizon']} | {ag['RMSE_MW']:.3f} | {ps['RMSE_MW']:.3f} | "
            f"{ag['nRMSE']:.2%} | {row['best_model']} |"
        )

    lines.extend(
        [
            "",
            "## 初步结论",
            "",
            "AutoGluon 这轮用于验证框架可跑通，以及 target-only 时间序列模型能否超过 Persistence。",
            "如果没有稳定超过 Persistence，下一步应增加已知时间特征、当前实测风速/风向静态特征，或者改成残差预测。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_series()

    results = []
    metrics_rows = []
    for horizon_name, horizon_steps in HORIZONS.items():
        print(f"Running AutoGluon horizon {horizon_name}...")
        summary, predictions = run_horizon(data, horizon_name, horizon_steps)
        results.append(summary)

        predictions.to_csv(
            OUTPUT_DIR / f"predictions_{horizon_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_predictions(predictions, horizon_name, OUTPUT_DIR / f"actual_vs_pred_{horizon_name}.png")

        for model_name in ["autogluon", "persistence"]:
            row = {
                "horizon": horizon_name,
                "model": model_name,
                "rows": int(summary["test_windows"]),
                **summary[model_name],
            }
            metrics_rows.append(row)

    pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(results, OUTPUT_DIR / "AutoGluon_TimeSeries实验结果.md")

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(pd.DataFrame(metrics_rows).pivot(index="horizon", columns="model", values="RMSE_MW").round(4))


if __name__ == "__main__":
    main()
