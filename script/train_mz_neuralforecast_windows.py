# -*- coding: utf-8 -*-
"""NeuralForecast rolling-window baselines for Miaozhuang.

The script trains lightweight NHITS and NBEATSx baselines on rolling power
windows. It is target-only for this first experiment, so it does not use future
measured wind speed or direction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch  # Import torch before numpy/pandas/matplotlib on this Windows environment.
import matplotlib
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx, NHITS
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "outputs" / "mz_short_term_tree_baselines" / "features_short_term.csv"
OUTPUT_DIR = ROOT / "outputs" / "mz_neuralforecast_windows"

CAPACITY_MW = 76.0
FREQ = "15min"
CONTEXT_STEPS = 96
TRAIN_STRIDE = 4
TEST_STRIDE = 1
MAX_STEPS = 80

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


def build_nf_frame(
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
        unique_id = f"w_{anchor}"
        start = anchor - CONTEXT_STEPS + 1
        for offset in range(length):
            source_index = start + offset
            rows.append(
                {
                    "unique_id": unique_id,
                    "ds": base_time + pd.Timedelta(minutes=15 * offset),
                    "y": float(y[source_index]),
                }
            )
        meta_rows.append(
            {
                "unique_id": unique_id,
                "anchor_index": int(anchor),
                "anchor_time": pd.Timestamp(times[anchor]),
                "target_time": pd.Timestamp(times[anchor + horizon_steps]),
                "power_current": float(y[anchor]),
                "y_true": float(y[anchor + horizon_steps]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(meta_rows)


def build_models(horizon_steps: int) -> list:
    common = {
        "h": horizon_steps,
        "input_size": CONTEXT_STEPS,
        "max_steps": MAX_STEPS,
        "learning_rate": 0.001,
        "batch_size": 64,
        "windows_batch_size": 512,
        "scaler_type": "robust",
        "random_seed": 42,
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "enable_checkpointing": False,
        "logger": False,
    }
    return [
        NHITS(
            **common,
            alias="NHITS",
            mlp_units=[[64, 64], [64, 64], [64, 64]],
        ),
        NBEATSx(
            **common,
            alias="NBEATSx",
            stack_types=["identity", "identity", "identity"],
            mlp_units=[[64, 64], [64, 64], [64, 64]],
        ),
    ]


def plot_predictions(predictions: pd.DataFrame, horizon_name: str, output_path: Path) -> None:
    plot_df = predictions.sort_values("target_time")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df["target_time"], plot_df["y_true"], label="Actual", linewidth=1.7)
    ax.plot(plot_df["target_time"], plot_df["y_pred_persistence"], label="Persistence", linewidth=1.2)
    for col in ["y_pred_NHITS", "y_pred_NBEATSx"]:
        if col in plot_df.columns:
            ax.plot(plot_df["target_time"], plot_df[col], label=col.replace("y_pred_", ""), linewidth=1.2)
    ax.set_title(f"Miaozhuang NeuralForecast ({horizon_name})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (MW)")
    ax.set_ylim(bottom=0.0, top=CAPACITY_MW * 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_horizon(data: pd.DataFrame, horizon_name: str, horizon_steps: int) -> tuple[list[dict], pd.DataFrame]:
    train_anchors = anchor_indices(data, "train", horizon_steps)
    test_anchors = anchor_indices(data, "test", horizon_steps)
    if not train_anchors or not test_anchors:
        fail(f"{horizon_name}: empty train or test anchors")

    train_df, train_meta = build_nf_frame(data, train_anchors, horizon_steps, include_future=True)
    test_context, test_meta = build_nf_frame(data, test_anchors, horizon_steps, include_future=False)

    nf = NeuralForecast(models=build_models(horizon_steps), freq=FREQ)
    nf.fit(df=train_df, val_size=0)
    forecast = nf.predict(df=test_context).reset_index()

    forecast = forecast.sort_values(["unique_id", "ds"]).groupby("unique_id", as_index=False).tail(1)
    predictions = test_meta.merge(forecast, on="unique_id", how="left")
    predictions["y_pred_persistence"] = clipped(predictions["power_current"])
    predictions["horizon"] = horizon_name

    metric_rows = []
    y_true = predictions["y_true"].to_numpy()
    metric_rows.append(
        {
            "horizon": horizon_name,
            "model": "Persistence",
            "rows": int(len(predictions)),
            **metrics(y_true, predictions["y_pred_persistence"].to_numpy()),
        }
    )

    for model_name in ["NHITS", "NBEATSx"]:
        pred_col = f"y_pred_{model_name}"
        predictions[pred_col] = clipped(predictions[model_name])
        metric_rows.append(
            {
                "horizon": horizon_name,
                "model": model_name,
                "rows": int(len(predictions)),
                **metrics(y_true, predictions[pred_col].to_numpy()),
            }
        )

    summary_path = OUTPUT_DIR / f"model_summary_{horizon_name}.txt"
    summary_path.write_text(str(nf.models), encoding="utf-8")
    return metric_rows, predictions


def write_report(metrics_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# NeuralForecast 实验结果",
        "",
        "## 口径",
        "",
        "- 使用 `NHITS` 和 `NBEATSx` 做轻量深度学习基线。",
        "- 输入为过去 24 小时功率窗口，预测未来 15min、30min、60min。",
        "- 本轮先不加未来天气预报协变量，也不使用未来实测风速/风向。",
        "- 训练窗口按 1 小时间隔抽样，测试窗口按 15 分钟全量滚动。",
        "",
        "## 测试集指标",
        "",
        "| Horizon | Persistence RMSE | NHITS RMSE | NBEATSx RMSE | 最优模型 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for horizon_name in HORIZONS:
        part = metrics_df[metrics_df["horizon"] == horizon_name]
        best = part.sort_values("RMSE_MW").iloc[0]

        def val(model: str) -> float:
            row = part[part["model"] == model]
            return float(row.iloc[0]["RMSE_MW"]) if not row.empty else float("nan")

        lines.append(
            f"| {horizon_name} | {val('Persistence'):.3f} | {val('NHITS'):.3f} | "
            f"{val('NBEATSx'):.3f} | {best['model']} |"
        )

    lines.extend(
        [
            "",
            "## 初步结论",
            "",
            "这轮实验用于确认 NeuralForecast 在当前数据上是否能跑通，并与强 Persistence 基线对比。",
            "如果深度模型没有超过 Persistence，不能直接说明模型无效，更可能说明当前任务短期惯性很强，且 target-only 输入信息不足。",
            "下一步应优先加入未来气象预报、历史实测风速/风向，以及更长训练期后，再尝试 TFT 或 PatchTST。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = load_series()

    all_metrics = []
    run_summary = {}
    for horizon_name, horizon_steps in HORIZONS.items():
        print(f"Running NeuralForecast horizon {horizon_name}...")
        rows, predictions = run_horizon(data, horizon_name, horizon_steps)
        all_metrics.extend(rows)
        predictions.to_csv(OUTPUT_DIR / f"predictions_{horizon_name}.csv", index=False, encoding="utf-8-sig")
        plot_predictions(predictions, horizon_name, OUTPUT_DIR / f"actual_vs_pred_{horizon_name}.png")
        run_summary[horizon_name] = {
            "horizon_steps": horizon_steps,
            "rows": int(len(predictions)),
            "models": ["NHITS", "NBEATSx"],
        }

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(metrics_df, OUTPUT_DIR / "NeuralForecast实验结果.md")

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(metrics_df.pivot(index="horizon", columns="model", values="RMSE_MW").round(4))


if __name__ == "__main__":
    main()
