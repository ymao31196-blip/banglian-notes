from pathlib import Path
import pandas as pd
import numpy as np

root = Path(r"C:\Users\86183\Desktop\大学\大四\帮联\苗庄风电场-01\_processed_v2")

if not root.exists():
    raise FileNotFoundError(f"文件夹不存在：{root}")

csv_files = sorted(root.rglob("*.csv"))

print(f"\n检查文件夹：{root}")
print(f"共发现 {len(csv_files)} 个 CSV 文件\n")

for file_path in csv_files:
    print("=" * 120)
    print(f"文件：{file_path}")

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
    except Exception as e:
        print(f"读取失败：{e}")
        continue

    print(f"形状：{df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"列名：{list(df.columns)}")

    # -------------------------
    # time列检查
    # -------------------------
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], errors="coerce")
        valid_t = t.dropna()

        print("\n[时间列检查]")
        print(f"time 非空数：{t.notna().sum()}")
        print(f"time 缺失数：{t.isna().sum()}")
        print(f"time 重复数：{t.duplicated().sum()}")
        if len(valid_t) > 0:
            print(f"time 起点：{valid_t.min()}")
            print(f"time 终点：{valid_t.max()}")

            if len(valid_t) >= 2:
                dt = valid_t.sort_values().diff().dropna()
                dt_counts = dt.value_counts().sort_values(ascending=False)
                print("time 间隔分布（前5项）：")
                print(dt_counts.head(5).to_string())
        else:
            print("time 列没有可解析时间")

    # -------------------------
    # 缺失值检查
    # -------------------------
    print("\n[缺失值统计]")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        print("无缺失值")
    else:
        print(missing.to_string())

    # -------------------------
    # 整行全空检查
    # -------------------------
    all_null_rows = df.isna().all(axis=1).sum()
    print(f"\n整行全空数量：{all_null_rows}")

    # -------------------------
    # 数值列统计
    # -------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n[数值列统计] 数值列数量：{len(num_cols)}")

    if len(num_cols) > 0:
        stats_rows = []
        for col in num_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            stats_rows.append({
                "column": col,
                "non_null": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "mean": s.mean(),
                "min": s.min(),
                "max": s.max(),
            })
        stats_df = pd.DataFrame(stats_rows)
        print(stats_df.to_string(index=False))
    else:
        print("没有数值列")

    # -------------------------
    # 前3行
    # -------------------------
    print("\n[前3行预览]")
    print(df.head(3).to_string())

print("\n检查完成。")