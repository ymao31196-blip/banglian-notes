# AutoGluon-TimeSeries 实验结果

## 口径

- 使用已有短期特征表中的 `power_mw`，先做 target-only 时间序列基线。
- 每个样本用过去 24 小时功率窗口，预测未来 15min、30min、60min。
- 训练窗口按 1 小时间隔抽样，测试窗口按 15 分钟全量滚动。
- 本轮不使用未来实测风速/风向，避免数据泄漏。

## 测试集指标

| Horizon | AutoGluon RMSE | Persistence RMSE | AutoGluon nRMSE | 最优内部模型 |
| --- | ---: | ---: | ---: | --- |
| 15min | 3.580 | 3.494 | 4.71% | WeightedEnsemble |
| 30min | 5.585 | 5.275 | 7.35% | WeightedEnsemble |
| 60min | 7.864 | 7.183 | 10.35% | WeightedEnsemble |

## 初步结论

AutoGluon 这轮用于验证框架可跑通，以及 target-only 时间序列模型能否超过 Persistence。
如果没有稳定超过 Persistence，下一步应增加已知时间特征、当前实测风速/风向静态特征，或者改成残差预测。
