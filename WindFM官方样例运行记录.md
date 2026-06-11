# WindFM 官方样例运行记录

> 运行日期：2026-06-12  
> 运行解释器：`D:\miniconda\envs\moment\python.exe`  
> 官方仓库：<https://github.com/shiyu-coder/WindFM>  
> 仓库 commit：`f0fc5ec210c3f11caf06948c75a9e42503d114e3`

## 一、运行结论

WindFM 官方样例已在本机成功完成模型下载、GPU 推理和预测输出。

本次验证说明：

- 官方模型和 tokenizer 可以正常加载。
- RTX 3050 Laptop GPU 可以完成样例推理。
- 官方示例数据的 240 点历史窗口可以预测后续 80 点。
- 模型能够输出多条概率预测路径，并计算 P50 和预测区间。
- 当前官方代码存在一处概率样本维度切片问题，需要在正式苗庄实验前修正并复测。

## 二、环境处理

运行前检查发现 `moment` 环境只缺少 `einops`，经用户确认后安装：

```powershell
& 'D:\miniconda\envs\moment\python.exe' -m pip install einops==0.8.1
```

没有主动降级其他已安装依赖。

| 项目 | 实际状态 |
|---|---|
| Python | 3.11.14 |
| PyTorch | 2.5.1+cu121 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| CUDA | 可用 |
| `einops` | 已补装 0.8.1 |
| 官方模型 | `NeoQuasar/WindFM` |
| 官方 tokenizer | `NeoQuasar/WindFM-Tokenizer` |

当前 `pandas`、`huggingface_hub`、`matplotlib` 等版本高于官方 `requirements.txt` 中的固定版本。本次样例可以正常运行，因此暂未降级。

## 三、官方样例参数

| 参数 | 数值 |
|---|---:|
| 历史窗口 `lookback` | 240 点 |
| 预测长度 `pred_len` | 80 点 |
| 请求概率样本数 `sample_count` | 100 |
| `max_context` | 512 |
| 温度参数 `T` | 1.0 |
| `top_p` | 1.0 |

输入变量顺序为：

```text
wind_speed
wind_direction
power
density
temperature
pressure
```

时间戳使用 UTC。

## 四、运行结果

使用固定随机种子 42 进行复核，得到：

| 指标 | 结果 |
|---|---:|
| 实际返回形状 | `80 × 80` |
| 推理及缓存模型加载时间 | 62.14 秒 |
| 峰值 CUDA 显存 | 1207.82 MB |
| P50 MAE | 3.5483 |
| P50 RMSE | 4.5109 |
| P50 R² | 0.1906 |
| P05-P95 区间覆盖率 | 62.5% |

这些指标只用于确认官方样例能够运行，不代表苗庄场站模型精度，也不能直接与客户 90% 或 95% 的准确率目标比较。官方样例采用小时级数据，80 个预测点对应约 80 小时，不是本项目正式的 10 小时或 10 天业务样本。

生成文件：

- `outputs/windfm_official_sample/official_sample_predictions.csv`
- `outputs/windfm_official_sample/official_sample_metrics.json`
- `outputs/windfm_official_sample/official_sample_forecast.png`

## 五、发现的官方代码问题

官方 `WindFMPredictor.generate()` 接收到的原始生成结果是四维数组：

```text
批次 × 概率样本 × 时间 × 特征
```

当前代码使用：

```python
preds = preds[:, -pred_len:, :]
```

该表达式在四维数组上裁剪的是第二维，即概率样本维，而不是时间维。后续代码又按时间维切片。

因此本次设置：

```text
sample_count = 100
pred_len = 80
```

最终只返回了 80 条概率样本，而不是请求的 100 条。使用 `sample_count=10、pred_len=8` 做独立检查时，也只返回 8 条样本，验证了这个问题。

正式实验前建议改为显式裁剪时间轴：

```python
preds = preds[:, :, -pred_len:, :]
```

修改后必须重新检查：

- 返回形状是否为 `pred_len × sample_count`。
- P05、P50、P95 是否按概率样本轴计算。
- 修正前后的点预测和区间指标是否一致。

本次没有修改临时目录中的官方源码，保留了原始仓库状态。

## 六、从样例确认的数据处理细节

### 1. 功率不强制预先归一化到 0-1

官方样例直接使用功率原始尺度，图中单位标为 MW。`WindFMPredictor` 会根据历史窗口的均值和标准差，对六个输入变量分别进行内部标准化，并在输出后还原尺度。

苗庄第一轮实验建议直接输入 MW：

```text
power = clip(-(311 + 312 + 313 + 314), 0, 76)
```

模型输出再按物理范围裁剪到 `[0, 76] MW`。

### 2. 字段顺序不能随意改变

官方代码最后使用特征索引 2 取得功率预测，说明 `power` 必须位于六变量中的第三列。即使字段名称都存在，改变顺序也可能造成错误结果。

### 3. 未来输入中没有 NWP 数值

官方接口给未来区间传入的是时间戳，不是未来风速、风向、温度和压力。因此当前实现可以用于历史状态驱动的零样本预测，但不能直接视为未来多源 NWP 融合模型。

### 4. 输入不允许存在缺失值

预测器会直接检查六个字段中的 NaN。苗庄数据适配前必须完成时间对齐、缺失统计和合理填补，不能把缺失值直接送入模型。

## 七、下一步

建议下一步先编写苗庄 WindFM 数据适配和回测脚本，完成：

1. 合成全场 MW 功率。
2. 将 19 台风机风速聚合为场站风速。
3. 对 19 台风机风向做圆形平均。
4. 补齐温度、压力和空气密度。
5. 将北京时间正确转换为 UTC。
6. 修正概率样本维度切片问题。
7. 按 15 分钟粒度滚动预测未来 10 小时，即 40 点。
8. 与 Persistence、LightGBM、XGBoost 做同一测试集对比。

WindFM 能否用于苗庄，不应以官方样例图是否好看判断，而应以苗庄滚动回测、分提前量误差和概率区间覆盖率判断。
