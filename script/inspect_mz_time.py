from pathlib import Path
import pandas as pd
import csv
import re

root = Path(r"C:\Users\86183\Desktop\大学\大四\帮联\苗庄风电场-01")
out_dir = root / "_processed" / "time_debug"
out_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# 基础工具
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

def guess_time_col(df):
    col_map = {norm_col(c): c for c in df.columns}

    # 优先顺序
    for key in ["objecttimestamp", "time", "时间"]:
        if key in col_map:
            return col_map[key]

    # 模糊兜底
    for c in df.columns:
        nc = norm_col(c)
        if "time" in nc or "时间" in nc:
            return c
    return None

def sample_indices(n):
    idx = set()

    # 前几行
    idx.update(range(0, min(8, n)))

    # 关键位置
    for start in [1438, 1439, 1440, 1441, 2878, 2879, 2880, 4318, 4319, 4320, 5758, 5759, 5760]:
        if 0 <= start < n:
            idx.add(start)

    # 最后几行
    idx.update(range(max(0, n - 8), n))

    return sorted(idx)

def print_series_samples(s, title):
    print(f"\n--- {title} ---")
    for i in sample_indices(len(s)):
        val = s.iloc[i]
        print(f"[{i}] {repr(val)}")

def inspect_raw_lines(file_path, line_numbers, max_fields=4):
    print("\n--- 原始文件行抽样（直接读文本）---")
    wanted = set(line_numbers)
    with open(file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i in wanted:
                preview = row[:max_fields]
                print(f"line {i}: len={len(row)} preview={preview}")
            if i > max(wanted):
                break


# =========================================================
# 单文件检查
# =========================================================
def inspect_file(file_path, file_type):
    print("\n" + "=" * 120)
    print(f"文件: {file_path}")

    if file_type == "scada":
        header_row = detect_scada_header_row(file_path)
        print(f"识别到SCADA表头行: {header_row}")
        if header_row is None:
            print("未识别到SCADA表头，跳过")
            return

        df = pd.read_csv(file_path, encoding="utf-8", header=header_row, low_memory=False)
        raw_df = pd.read_csv(file_path, encoding="utf-8", header=None, low_memory=False)

    elif file_type == "simple":
        df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        raw_df = pd.read_csv(file_path, encoding="utf-8", header=None, low_memory=False)

    elif file_type == "winddir":
        df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        raw_df = pd.read_csv(file_path, encoding="utf-8", header=None, low_memory=False)

    else:
        print("未知文件类型")
        return

    df.columns = [clean_colname(c) for c in df.columns]
    time_col = guess_time_col(df)

    print(f"DataFrame形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"识别到时间列: {time_col}")

    if time_col is None:
        print("未找到时间列，跳过")
        return

    s = df[time_col].astype(str)
    s_nonnull = df[time_col].notna().sum()
    s_unique = s.nunique(dropna=True)
    parsed = pd.to_datetime(s, errors="coerce")
    parsed_valid = parsed.notna().sum()
    parsed_unique = parsed.dropna().nunique()

    print(f"总行数: {len(df)}")
    print(f"时间列非空数: {s_nonnull}")
    print(f"时间列原始唯一值数: {s_unique}")
    print(f"可解析时间数: {parsed_valid}")
    print(f"可解析唯一时间数: {parsed_unique}")

    print_series_samples(s, "时间列样本")
    print_series_samples(parsed.astype(str), "解析后时间样本")

    # 长度分布
    lens = s.str.len().value_counts(dropna=False).sort_index()
    print("\n--- 时间字符串长度分布 ---")
    print(lens.to_string())

    # 包含冒号的数量
    has_colon = s.str.contains(":", regex=False, na=False).sum()
    has_slash = s.str.contains("/", regex=False, na=False).sum()
    print(f"\n包含 ':' 的行数: {has_colon}")
    print(f"包含 '/' 的行数: {has_slash}")

    # 导出调试表
    debug_df = pd.DataFrame({
        "row_idx": range(len(df)),
        "raw_time": s,
        "raw_time_repr": s.map(repr),
        "parsed_time": parsed.astype(str),
    })
    out_file = out_dir / f"{file_path.stem}_time_debug.csv"
    debug_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n已导出: {out_file}")

    # 直接抽原始文本行，避免被 pandas 美化
    if file_type == "scada":
        header_row = detect_scada_header_row(file_path)
        line_numbers = [
            0, 1, 2, 3, 4, 5, 6, 7,
            (header_row or 0) + 1,
            (header_row or 0) + 1439,
            (header_row or 0) + 1440,
            (header_row or 0) + 1441,
            (header_row or 0) + 2879,
            (header_row or 0) + 2880,
            (header_row or 0) + 2881,
        ]
    else:
        line_numbers = [0, 1, 2, 3, 4, 1439, 1440, 1441, 2879, 2880, 2881]

    inspect_raw_lines(file_path, [x for x in line_numbers if x >= 0])


# =========================================================
# 执行
# =========================================================
files_to_check = [
    (root / "集电线-01" / "mz-311p-26.csv", "scada"),
    (root / "集电线-01" / "mz-312p-26.csv", "scada"),
    (root / "集电线-01" / "mz-313p-26.csv", "simple"),
    (root / "集电线-01" / "mz-314p-26.csv", "simple"),
    (root / "风速-01" / "mz-fs8-26.csv", "scada"),
    (root / "风速-01" / "mz-fs9-26.csv", "scada"),
    (root / "风向-01" / "mz-fx8-26" / "N08_原始数据_2026-03-06_16-26-18.csv", "winddir"),
    (root / "风向-01" / "mz-fx9-26" / "N09_原始数据_2026-03-06_16-39-48.csv", "winddir"),
]

for fp, ftype in files_to_check:
    if fp.exists():
        inspect_file(fp, ftype)
    else:
        print(f"\n文件不存在: {fp}")