# # # # from pathlib import Path
# # # # from datetime import datetime

# # # # # ====== 改成你的文件夹路径 ======
# # # # folder_path = r"C:\Users\86183\Desktop\大学\大四\帮联\苗庄风电场-01"
# # # # # ==============================

# # # # folder = Path(folder_path)

# # # # if not folder.exists():
# # # #     print(f"路径不存在：{folder}")
# # # # elif not folder.is_dir():
# # # #     print(f"这不是文件夹：{folder}")
# # # # else:
# # # #     print(f"\n递归检查文件夹：{folder}\n")
# # # #     print("-" * 130)
# # # #     print(f"{'类型':<8}{'相对路径':<70}{'大小(MB)':<12}{'修改时间'}")
# # # #     print("-" * 130)

# # # #     for item in sorted(folder.rglob("*")):
# # # #         rel_path = item.relative_to(folder)
# # # #         item_type = "文件夹" if item.is_dir() else "文件"

# # # #         if item.is_file():
# # # #             size_mb = item.stat().st_size / (1024 * 1024)
# # # #             size_str = f"{size_mb:.2f}"
# # # #         else:
# # # #             size_str = "-"

# # # #         mtime = datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

# # # #         print(f"{item_type:<8}{str(rel_path):<70}{size_str:<12}{mtime}")

# # # #     print("-" * 130)

# # # from pathlib import Path
# # # import pandas as pd
# # # import xarray as xr

# # # folder = Path(r"C:\Users\86183\Desktop\大学\大四\帮联\苗庄风电场-01")

# # # def try_read_csv(file_path):
# # #     encodings = ["utf-8", "gbk", "gb18030", "utf-8-sig"]
# # #     last_error = None
# # #     for enc in encodings:
# # #         try:
# # #             df = pd.read_csv(file_path, encoding=enc)
# # #             return df, enc, None
# # #         except Exception as e:
# # #             last_error = e
# # #     return None, None, last_error

# # # for file in sorted(folder.rglob("*")):
# # #     if file.name == ".DS_Store":
# # #         continue

# # #     print("\n" + "=" * 120)
# # #     print(f"文件: {file}")

# # #     if file.suffix.lower() == ".csv":
# # #         df, enc, err = try_read_csv(file)
# # #         if df is None:
# # #             print(f"CSV读取失败: {err}")
# # #             continue

# # #         print(f"类型: CSV")
# # #         print(f"编码: {enc}")
# # #         print(f"形状: {df.shape[0]} 行 x {df.shape[1]} 列")
# # #         print(f"列名: {list(df.columns)}")
# # #         print("前8行:")
# # #         print(df.head(8).to_string())

# # #     elif file.suffix.lower() == ".nc":
# # #         try:
# # #             ds = xr.open_dataset(file)
# # #             print("类型: NetCDF")
# # #             print(f"维度: {dict(ds.dims)}")
# # #             print(f"坐标: {list(ds.coords)}")
# # #             print(f"变量: {list(ds.data_vars)}")
# # #             print(ds)
# # #             ds.close()
# # #         except Exception as e:
# # #             print(f"NetCDF读取失败: {e}")

# # from pathlib import Path
# # import xarray as xr
# # import shutil
# # import tempfile

# # nc_file = Path(r"C:\Users\86183\Desktop\大学\大四\帮联\苗庄风电场-01\tianjin-1-119.0_38.8_117.1_36.0_2026030100.nc")

# # engines = ["netcdf4", "h5netcdf", "scipy"]

# # print("原路径存在吗：", nc_file.exists())

# # for eng in engines:
# #     try:
# #         ds = xr.open_dataset(nc_file, engine=eng)
# #         print(f"\n原路径读取成功，engine={eng}")
# #         print(ds)
# #         ds.close()
# #         break
# #     except Exception as e:
# #         print(f"\n原路径失败，engine={eng}，错误：{e}")
# # else:
# #     tmp_dir = Path(tempfile.gettempdir()) / "nc_ascii_test"
# #     tmp_dir.mkdir(exist_ok=True)
# #     tmp_file = tmp_dir / nc_file.name
# #     shutil.copy2(nc_file, tmp_file)

# #     print(f"\n已复制到纯英文临时目录：{tmp_file}")

# #     for eng in engines:
# #         try:
# #             ds = xr.open_dataset(tmp_file, engine=eng)
# #             print(f"\n临时目录读取成功，engine={eng}")
# #             print(ds)
# #             ds.close()
# #             break
# #         except Exception as e:
# #             print(f"\n临时目录失败，engine={eng}，错误：{e}")

# from pathlib import Path

# nc_file = Path(r"C:\Users\86183\Desktop\大学\大四\帮联\苗庄风电场-01\tianjin-1-119.0_38.8_117.1_36.0_2026030100.nc")

# with open(nc_file, "rb") as f:
#     head = f.read(16)

# print("文件大小(字节):", nc_file.stat().st_size)
# print("前16字节:", head)
# print("十六进制:", head.hex())

# if head.startswith(b"CDF"):
#     print("判断：像 NetCDF3")
# elif head.startswith(b"\x89HDF\r\n\x1a\n"):
#     print("判断：像 HDF5 / NetCDF4")
# elif head.startswith(b"GRIB"):
#     print("判断：像 GRIB 文件，只是后缀写成了 .nc")
# elif head.startswith(b"PK\x03\x04"):
#     print("判断：像 ZIP 压缩包，只是后缀写成了 .nc")
# else:
#     print("判断：文件头不符合常见 NetCDF/GRIB/ZIP 特征")
from pathlib import Path
import h5py

nc_file = Path(r"C:\Users\86183\Desktop\大学\大四\帮联\苗庄风电场-01\tianjin-1-119.0_38.8_117.1_36.0_2026030100.nc")

with h5py.File(nc_file, "r") as f:
    print("根节点 keys:", list(f.keys()))

    def show(name, obj):
        obj_type = type(obj).__name__
        shape = getattr(obj, "shape", None)
        print(f"{name} | {obj_type} | shape={shape}")

    f.visititems(show)