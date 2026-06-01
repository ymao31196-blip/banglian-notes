from pathlib import Path
import pandas as pd
import csv
import re

# =========================================================
# 0. 路径
# =========================================================
root = Path(r"C:\Users\86183\Desktop\大学\大四\帮联\苗庄风电场-01")
out_dir = root / "_processed_v2"
single_dir = out_dir / "single_files"

out_dir.mkdir(exist_ok=True)
single_dir.mkdir(exist_ok=True)


# =========================================================
# 1. 基础工具
# =========================================================
def clean_colname(x):
    return str(x).replace("\ufeff", "").replace("\n", " ").replace("\r", " ").strip()

def norm_col(x):
    x = clean_colname(x).lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def detect_scada_header_row(file_path, max_lines=30):
    with open(file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            row_norm = [norm_col(x) for x in row]
            if len(row_norm) >= 2 and row_norm[0] == "objectid" and row_norm[1] == "objecttimestamp":
                return i
            if i >= max_lines - 1:
                break
    return None

def parse_time_series(series):
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    t = pd.to_datetime(s, errors="coerce")
    return t

def save_single_df(df, name):
    out_file = single_dir / f"{name}.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")

def summarize_time(df, name):
    return {
        "dataset": name,
        "rows": len(df),
        "time_non_null": int(df["time"].notna().sum()) if "time" in df.columns else 0,
        "time_min": df["time"].min() if "time" in df.columns and len(df) > 0 else pd.NaT,
        "time_max": df["time"].max() if "time" in df.columns and len(df) > 0 else pd.NaT,
        "duplicated_time_count": int(df["time"].duplicated().sum()) if "time" in df.columns else 0
    }


# =========================================================
# 2. 读取函数
# =========================================================
def read_scada_raw(file_path, prefix):
    """
    读取SCADA导出文件：
    - mz-311p-26.csv
    - mz-312p-26.csv
    - mz-fs8-26.csv
    - mz-fs9-26.csv
    """
    header_row = detect_scada_header_row(file_path)
    if header_row is None:
        raise ValueError(f"{file_path.name} 未自动识别到SCADA表头行")

    df = pd.read_csv(
        file_path,
        encoding="utf-8",
        header=header_row,
        low_memory=False
    )
    df.columns = [clean_colname(c) for c in df.columns]

    # 去掉整行全空
    df = df.dropna(how="all").copy()

    col_map = {norm_col(c): c for c in df.columns}
    time_col = col_map.get("objecttimestamp")
    if time_col is None:
        raise ValueError(f"{file_path.name} 未找到 ObjectTimeStamp 列。当前列名：{list(df.columns)}")

    out = pd.DataFrame()
    out["time"] = parse_time_series(df[time_col])

    exact_fields = {
        "latest value": f"{prefix}_latest",
        "avg": f"{prefix}_avg",
        "max": f"{prefix}_max",
        "min": f"{prefix}_min",
        "quality": f"{prefix}_quality",
        "load factor": f"{prefix}_load_factor",
    }

    for std_name, out_name in exact_fields.items():
        if std_name in col_map:
            out[out_name] = pd.to_numeric(df[col_map[std_name]], errors="coerce")

    # 只保留有时间的行
    out = out.dropna(subset=["time"]).copy()

    dup_count = out["time"].duplicated().sum()
    if dup_count > 0:
        print(f"[WARN] {file_path.name} 存在重复时间 {dup_count} 条，将保留第一条")

    out = (
        out.drop_duplicates(subset=["time"], keep="first")
           .sort_values("time")
           .reset_index(drop=True)
    )

    print(
        f"[OK] {file_path.name} -> header_row={header_row}, "
        f"rows={len(out)}, time_range=({out['time'].min()} -> {out['time'].max()})"
    )
    save_single_df(out, prefix)
    return out


def read_simple_two_col(file_path, value_name):
    """
    读取两列表：
    - mz-313p-26.csv
    - mz-314p-26.csv
    """
    df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    df.columns = [clean_colname(c) for c in df.columns]

    df = df.dropna(how="all").copy()

    col_map = {norm_col(c): c for c in df.columns}
    time_col = col_map.get("time")
    if time_col is None:
        raise ValueError(f"{file_path.name} 未找到 time 列，当前列名：{list(df.columns)}")

    value_cols = [c for c in df.columns if c != time_col]
    if len(value_cols) != 1:
        raise ValueError(f"{file_path.name} 不是标准两列表，当前值列：{value_cols}")

    out = pd.DataFrame()
    out["time"] = parse_time_series(df[time_col])
    out[value_name] = pd.to_numeric(df[value_cols[0]], errors="coerce")

    out = out.dropna(subset=["time"]).copy()

    dup_count = out["time"].duplicated().sum()
    if dup_count > 0:
        print(f"[WARN] {file_path.name} 存在重复时间 {dup_count} 条，将保留第一条")

    out = (
        out.drop_duplicates(subset=["time"], keep="first")
           .sort_values("time")
           .reset_index(drop=True)
    )

    print(
        f"[OK] {file_path.name} -> rows={len(out)}, "
        f"time_range=({out['time'].min()} -> {out['time'].max()})"
    )
    save_single_df(out, value_name)
    return out


def read_wind_dir(file_path, value_name):
    """
    读取风向文件：
    - N08_原始数据_*.csv
    - N09_原始数据_*.csv
    """
    df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    df.columns = [clean_colname(c) for c in df.columns]

    df = df.dropna(how="all").copy()

    col_map = {norm_col(c): c for c in df.columns}
    time_col = col_map.get("时间")
    dir_col = col_map.get("平均风向")

    if time_col is None or dir_col is None:
        raise ValueError(f"{file_path.name} 缺少必要列，当前列名：{list(df.columns)}")

    out = pd.DataFrame()
    out["time"] = parse_time_series(df[time_col])
    out[value_name] = pd.to_numeric(df[dir_col], errors="coerce")

    if "风机" in col_map:
        out[f"{value_name}_turbine"] = df[col_map["风机"]]
    if "风机类型" in col_map:
        out[f"{value_name}_type"] = df[col_map["风机类型"]]

    out = out.dropna(subset=["time"]).copy()

    dup_count = out["time"].duplicated().sum()
    if dup_count > 0:
        print(f"[WARN] {file_path.name} 存在重复时间 {dup_count} 条，将保留第一条")

    out = (
        out.drop_duplicates(subset=["time"], keep="first")
           .sort_values("time")
           .reset_index(drop=True)
    )

    print(
        f"[OK] {file_path.name} -> rows={len(out)}, "
        f"time_range=({out['time'].min()} -> {out['time'].max()})"
    )
    save_single_df(out, value_name)
    return out


# =========================================================
# 3. 读取所有文件
# =========================================================
df_311 = read_scada_raw(root / "集电线-01" / "mz-311p-26.csv", "line311_p")
df_312 = read_scada_raw(root / "集电线-01" / "mz-312p-26.csv", "line312_p")
df_313 = read_simple_two_col(root / "集电线-01" / "mz-313p-26.csv", "line313_p")
df_314 = read_simple_two_col(root / "集电线-01" / "mz-314p-26.csv", "line314_p")

df_ws08 = read_scada_raw(root / "风速-01" / "mz-fs8-26.csv", "ws08")
df_ws09 = read_scada_raw(root / "风速-01" / "mz-fs9-26.csv", "ws09")

df_wd08 = read_wind_dir(root / "风向-01" / "mz-fx8-26" / "N08_原始数据_2026-03-06_16-26-18.csv", "wd08")
df_wd09 = read_wind_dir(root / "风向-01" / "mz-fx9-26" / "N09_原始数据_2026-03-06_16-39-48.csv", "wd09")

dfs = [df_311, df_312, df_313, df_314, df_ws08, df_ws09, df_wd08, df_wd09]


# =========================================================
# 4. 单文件时间汇总
# =========================================================
file_summary = pd.DataFrame([
    summarize_time(df_311, "line311_p"),
    summarize_time(df_312, "line312_p"),
    summarize_time(df_313, "line313_p"),
    summarize_time(df_314, "line314_p"),
    summarize_time(df_ws08, "ws08"),
    summarize_time(df_ws09, "ws09"),
    summarize_time(df_wd08, "wd08"),
    summarize_time(df_wd09, "wd09"),
])

file_summary.to_csv(out_dir / "mz_file_time_summary.csv", index=False, encoding="utf-8-sig")


# =========================================================
# 5. 构造“公共重叠时段”
# =========================================================
common_start = max(df["time"].min() for df in dfs)
common_end = min(df["time"].max() for df in dfs)

print(f"\n[COMMON OVERLAP] {common_start} -> {common_end}")

common_time = pd.date_range(common_start, common_end, freq="1min")
merged_common = pd.DataFrame({"time": common_time})

for df in dfs:
    merged_common = merged_common.merge(df, on="time", how="left")


# =========================================================
# 6. 缺测统计（公共时段）
# =========================================================
summary_rows = []
for col in merged_common.columns:
    if col == "time":
        continue
    missing_count = int(merged_common[col].isna().sum())
    summary_rows.append({
        "column": col,
        "total_rows": len(merged_common),
        "non_null_rows": int(merged_common[col].notna().sum()),
        "missing_rows": missing_count,
        "missing_ratio": missing_count / len(merged_common)
    })

missing_summary = (
    pd.DataFrame(summary_rows)
      .sort_values(["missing_rows", "column"], ascending=[False, True])
      .reset_index(drop=True)
)


# =========================================================
# 7. 保存输出
# =========================================================
merged_common.to_csv(out_dir / "mz_merged_common_period.csv", index=False, encoding="utf-8-sig")
missing_summary.to_csv(out_dir / "mz_missing_summary_common_period.csv", index=False, encoding="utf-8-sig")

print("\n已输出文件：")
print(out_dir / "mz_merged_common_period.csv")
print(out_dir / "mz_missing_summary_common_period.csv")
print(out_dir / "mz_file_time_summary.csv")
print(single_dir)

print("\n公共时段合并表前10行：")
print(merged_common.head(10).to_string())

print("\n公共时段缺测统计：")
print(missing_summary.to_string(index=False))

print("\n单文件时间范围汇总：")
print(file_summary.to_string(index=False))