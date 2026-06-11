# WindFM 与开源风电功率预测模型调研

> 预测时长口径更新（2026-06-03）：后续统一按客户最新说法，**超短期预测=未来 10 小时，目标准确率 95%**；**短期预测=未来 10 天，目标准确率 90%**。历史文档中 `15min/30min/60min` 或 `1h` 实验统一视为“分钟级/小时级流程验证实验”，不再等同于客户口径下的正式“短期/超短期”任务；“次日 24 小时”相关实验视为日前/日内样本构造验证。
> 调研日期：2026-06-03  
> 项目背景：京津冀风电场功率预测，部分场站位于丘陵、山地地形。客户口径中，短期预测为未来 10 天，目标准确率 90%；超短期预测为未来 10 小时，目标准确率 95%。  
> 本文定位：供技术调研和后续模型选型使用，重点说明 WindFM 以及其他开源模型是什么、适合解决什么问题、在本项目中如何落地实验。

---

## 一、总体判断

当前风电功率预测项目不应简单理解为“找一个模型替换玖天算法”。更合理的技术路线是：

1. 先明确客户考核口径。
2. 建立可复现、可解释的开源基线。
3. 用同一批数据对玖天预测结果和开源基线进行独立评估。
4. 再逐步引入多源气象预报、地形特征、物理约束、持续校准和模型融合。

WindFM 可以作为一个重要探索项，因为它是面向风电预测的基础模型方向。但它不应直接替代 LightGBM、XGBoost、TFT 等常规基线。原因是 WindFM 更像是“风电领域预训练模型”，适合做零样本、少样本或概率预测对照；而客户要求的 10 小时和 10 天预测，尤其是 10 天短期预测，核心仍然依赖未来气象预报和场站本地化订正。

简要结论如下。

| 模型路线 | 当前定位 | 是否建议做 | 主要价值 |
|---|---|---:|---|
| LightGBM、XGBoost | 第一优先级强基线 | 是 | 快速、稳定、可解释，适合多源气象和业务特征 |
| WindFM | 风电专用基础模型探索 | 是 | 可作为风电专门模型对照，关注零样本、概率预测能力 |
| NeuralForecast | 深度时间序列模型库 | 是 | NHITS、NBEATSx、TFT、PatchTST 可系统比较 |
| AutoGluon-TimeSeries | 自动时间序列建模 | 是 | 快速出强基线，但黑盒程度较高 |
| TimesFM、Chronos、Moirai、TTM、TabPFN-TS、Lag-Llama、MOMENT | 通用时间序列基础模型 | 可选 | 可做技术储备和对照，重点关注未来气象预报融合能力 |
| GraphCast、Pangu-Weather、FourCastNet | AI 天气模型 | 谨慎 | 更偏气象预报源，不是直接功率预测模型 |
| FLORIS、windpowerlib、OpenOA | 物理与工程模型 | 辅助 | 可用于物理约束、功率曲线、风场诊断 |

---

## 二、先校正预测任务口径

客户口径中：

- 超短期预测：未来 10 小时。
- 短期预测：未来 10 天。
- 超短期目标准确率：95%。
- 短期目标准确率：90%。

这和我们之前小实验中的 15 分钟、30 分钟、1 小时预测不完全一致。

之前实验的价值主要是流程验证：

- 功率标签口径是否能处理。
- 时间戳能否对齐。
- 特征表能否构造。
- 模型能否训练。
- 指标能否输出。
- 多类模型能否跑通。

但后续正式任务必须重新按客户口径构造样本。

### 1. 超短期 10 小时预测

超短期预测可以充分利用：

- 当前功率。
- 过去几小时功率变化。
- 当前和历史实测风速、风向。
- 未来 10 小时数值天气预报。
- 场站运行状态。
- 限电、停机、检修标记。

这个任务里，Persistence，也就是“当前功率延续”，仍然是非常强的基线。但 10 小时时间跨度已经比 15 分钟、1 小时更长，气象预报和功率变化趋势会更重要。

适合优先实验：

- Persistence。
- LightGBM/XGBoost。
- NHITS/NBEATSx。
- TFT。
- WindFM。

### 2. 短期 10 天预测

短期 10 天预测几乎不可能只靠历史功率完成。未来 10 天内，真实风速、真实功率都还没有发生，所以核心输入必须是未来气象预报。

适合重点投入：

- 多源气象预报融合。
- 气象预报本地化订正。
- 地形特征。
- 场站可用容量和运行状态。
- LightGBM/XGBoost 强表格基线。
- TFT、PatchTST 等多协变量深度模型。
- WindFM 作为对照探索。

### 3. 准确率 90% 和 95% 必须确认公式

风电行业中的“准确率”通常不是分类任务中的 accuracy。常见口径可能是：

```text
准确率 = 1 - RMSE / 装机容量
```

如果客户按这个口径：

- 超短期准确率 95%，约等价于 nRMSE 小于等于 5%。
- 短期准确率 90%，约等价于 nRMSE 小于等于 10%。

但不同企业和考核规则可能有差异，例如：

- 按单点误差统计。
- 按全日平均误差统计。
- 按考核时段统计。
- 剔除限电、停机、通信异常时段。
- 按装机容量归一化。
- 按可用容量归一化。

因此后续必须向客户确认准确率公式。否则模型优化目标可能和客户考核目标不一致。

---

## 三、WindFM 重点调研

### 1. WindFM 是什么

WindFM 是一个面向风电功率预测的基础模型。根据其论文和开源仓库信息，它的目标是建立一个可以在不同风电场之间迁移的风电预测模型，而不是每个场站都从零训练一个模型。

资料来源：

- 论文：<https://arxiv.org/abs/2509.06311>
- GitHub：<https://github.com/shiyu-coder/WindFM>
- Hugging Face 模型页：<https://huggingface.co/shiyu-coder>

WindFM 的核心定位可以概括为：

> 利用大规模风电时序数据进行预训练，使模型学习风电功率、风速、风向等变量之间的通用规律，然后迁移到新风电场进行预测。

通俗解释：

> 它类似于风电领域的“预训练模型”。不是只看一个场站的数据，而是先从大量风电场和长时间序列中学习通用规律，再用于新场站预测。

### 2. WindFM 的训练数据背景

WindFM 论文和仓库中提到，模型使用 NREL WIND Toolkit 数据进行大规模预训练。

WIND Toolkit 是美国 NREL 提供的风能相关数据集，覆盖美国大陆范围，包含大量时空点位的风资源和功率相关数据。

资料来源：

- NREL WIND Toolkit：<https://www.nrel.gov/grid/wind-toolkit>

这说明 WindFM 的一个明显优势是，它不是在很小的数据集上训练的普通模型，而是在大规模风电相关数据上预训练得到的。

但这也带来一个需要注意的问题：

> WindFM 的预训练数据主要来自美国风资源数据集，而本项目区域是京津冀，且包含丘陵、山地地形。区域气候、地形、机型、调度规则可能不同，因此不能默认 WindFM 零样本结果一定优于本地训练模型。

### 3. WindFM 的模型思想

根据论文和仓库介绍，WindFM 采用类似大语言模型的思路处理时间序列。

可以理解为三步：

1. 把连续的风电时间序列数值转成离散 token。
2. 使用 Transformer 类模型学习 token 序列中的规律。
3. 根据历史序列生成未来功率预测。

通俗解释：

> 大语言模型是看前面的词预测后面的词，WindFM 则是看过去的风电时间序列片段，预测未来的功率变化。

### 4. WindFM 的输入输出

从 GitHub 示例看，WindFM 输入包括历史时间序列数据和未来时间戳。

历史输入中通常需要包含：

- `power`
- 风速相关变量
- 风向相关变量
- 其他气象或状态变量
- 时间戳

输出是未来时刻的功率预测。

WindFM 还支持概率预测，即不只是输出一个点预测值，也可以输出多个样本或不确定性结果。

这对风电预测很重要，因为风电功率天然受天气不确定性影响。概率预测可以回答：

- 未来功率大概是多少。
- 预测有多不确定。
- 低功率风险有多大。
- 高功率波动风险有多大。

### 5. WindFM 的优势

WindFM 对本项目有几个潜在价值。

#### 第一，风电专用

它不是通用时间序列模型，而是面向风电预测提出的基础模型。这一点在技术交流和方案评审中具有较高关注度。

可以这样说：

> WindFM 的优势在于它不是普通时间序列模型，而是针对风电功率预测进行大规模预训练的模型，理论上更容易学习风速、风向、功率之间的领域规律。

#### 第二，适合做零样本或少样本对照

如果本地场站数据较少，WindFM 可能通过预训练知识提供一个初始预测能力。

这和从零训练 LightGBM 或深度模型不同。

#### 第三，支持概率预测

风电业务中，单点预测并不能完整表达风险。概率预测可以输出预测区间，例如：

- 低估风险。
- 高估风险。
- 置信区间。

如果客户后续关注调度风险，概率预测会有价值。

#### 第四，可以作为“风电基础模型”技术储备

即使 WindFM 当前不一定直接成为主模型，也可以作为技术储备写进后续方案：

- 和 LightGBM/XGBoost 对比。
- 和 TFT/PatchTST 对比。
- 和玖天结果对比。
- 评估其零样本、少样本、本地微调潜力。

### 6. WindFM 在本项目中的限制

WindFM 不能直接被过度包装成最终答案，主要有以下限制。

#### 第一，预训练区域和本项目区域不同

WindFM 预训练主要基于美国风资源数据。本项目在京津冀，且有丘陵、山地。

区域差异可能包括：

- 风资源分布不同。
- 地形影响不同。
- 风机机型不同。
- 调度规则不同。
- 限电和停机规则不同。
- 数据采集口径不同。

因此 WindFM 零样本结果必须实测评估。

#### 第二，客户目标是 10 小时和 10 天预测

WindFM 是否适合长达 10 天的预测，需要看：

- 输入序列长度是否足够。
- 输出长度是否支持。
- 是否能有效利用未来天气预报。
- 长提前量下误差是否快速累积。

如果 WindFM 主要依赖历史序列自回归，那么它对 10 天预测未必优于未来气象驱动模型。

#### 第三，未来气象预报融合能力需要验证

客户短期 10 天预测必须使用未来气象预报。

经核查当前官方示例和开源接口，预测端输入的是历史多变量序列、历史时间戳和未来时间戳；示例没有向未来预测区间传入 NWP 数值。因此，当前开源实现不能直接视为“未来多源 NWP 融合模型”。它更适合先作为依赖历史状态的零样本、概率预测对照；面向未来 10 天的正式短期预测时，需要另行使用 NWP 驱动模型，或者研究 WindFM 与 NWP 模型的融合和残差订正。

#### 第四，功率单位和归一化必须严格处理

WindFM 类模型通常对数据尺度敏感。需要确认：

- 功率是 MW、kW 还是归一化值。
- 是否按装机容量归一化。
- 是否要求功率字段名固定为 `power`。
- 时间戳是否要求 UTC。
- 是否需要固定采样间隔。

如果这些处理不一致，模型结果可能没有意义。

### 7. WindFM 在本项目中的实验方案

建议把 WindFM 作为单独实验路线，不和主环境混在一起。

#### 阶段 1：环境和样例跑通

目标：

- 按官方 GitHub 示例跑通最小样例。
- 确认依赖版本。
- 确认输入字段。
- 确认输出格式。
- 确认是否支持 CPU，是否需要 GPU。

输出：

- WindFM 可运行记录。
- 示例预测结果。
- 依赖环境说明。

#### 阶段 2：苗庄数据格式适配

把苗庄数据整理成 WindFM 需要的格式。

至少包括：

- 时间戳。
- 归一化功率。
- 历史风速。
- 历史风向。
- 历史气象或状态变量。

注意：

- 不能使用未来实测风速风向。
- 如果用于短期 10 天，必须确认未来气象预报如何输入。
- 如果不能输入未来气象预报，则 WindFM 主要作为历史序列基线。

#### 阶段 3：超短期 10 小时对比

预测未来 10 小时，按 15 分钟粒度就是 40 个点。

对比模型：

- Persistence。
- LightGBM。
- XGBoost。
- NHITS。
- NBEATSx。
- WindFM。

评价指标：

- MAE。
- RMSE。
- nRMSE。
- 客户准确率。
- 分时段误差，例如 0 到 2 小时、2 到 5 小时、5 到 10 小时。

#### 阶段 4：短期 10 天对比

预测未来 10 天，按 15 分钟粒度就是 960 个点。

这个任务必须加入未来气象预报。

对比模型：

- 单源 NWP + LightGBM/XGBoost。
- 多源 NWP + LightGBM/XGBoost。
- TFT。
- PatchTST。
- WindFM。
- WindFM + 后处理或残差修正。

如果 WindFM 不能直接使用未来多源 NWP，那么应把它定位为对照项，而不是主模型。

#### 阶段 5：与玖天结果对比

如果能拿到玖天同一时间段的预测结果，建议建立独立评估表。

每一行包含：

- 场站。
- 起报时间。
- 有效时间。
- 预测提前量。
- 实际功率。
- 玖天预测。
- 我们的 LightGBM/XGBoost 预测。
- WindFM 预测。
- TFT/PatchTST 预测。
- 是否限电。
- 是否停机。
- 气象源。

这样即使不知道玖天内部算法，也能判断：

- 玖天在哪些提前量好。
- 哪些天气条件下差。
- 山地丘陵场站是否误差更高。
- 我们的开源基线距离客户目标还有多远。

### 8. WindFM 正式表述口径

建议正式表述如下：

> WindFM 是目前比较值得关注的风电专用基础模型，它把风电功率预测从传统单场站训练提升到预训练模型迁移的方向。它的优势是风电领域针对性强、可能具备零样本或少样本能力，并且支持概率预测。但它是否适合我们这个项目，还需要结合京津冀丘陵山地场站、未来 10 小时和 10 天预测口径、多源气象预报输入方式进行实际验证。因此 WindFM 适合作为技术储备和对照实验，不建议直接替代当前应优先建立的 LightGBM/XGBoost 多源气象强基线。

---

## 四、其他开源模型和工具路线

### 1. LightGBM

资料来源：

- 官方文档：<https://lightgbm.readthedocs.io/>
- GitHub：<https://github.com/microsoft/LightGBM>

LightGBM 是梯度提升树模型，适合结构化表格数据。

在风电功率预测中，它适合处理：

- 多源气象变量。
- 历史功率滞后。
- 风速平方、三次方。
- 风向 sin/cos。
- 时间周期特征。
- 场站静态特征。
- 运行状态标记。
- 限电、停机、检修标记。

优势：

- 训练快。
- 对表格数据强。
- 可解释性较好。
- 对中小样本友好。
- 很适合作为工业基线。

限制：

- 不能天然处理很长时间序列，需要人工构造滞后和滚动特征。
- 对多步预测需要设计策略，例如直接多输出、逐步递推、按提前量建模。

本项目优先级：

> 第一优先级。建议作为超短期和短期预测的主基线。

### 2. XGBoost

资料来源：

- 官方文档：<https://xgboost.readthedocs.io/>
- GitHub：<https://github.com/dmlc/xgboost>

XGBoost 和 LightGBM 类似，也是梯度提升树模型。

优势：

- 稳定。
- 工业界使用广。
- 表格数据表现强。
- 对特征工程友好。

限制：

- 大数据训练速度有时不如 LightGBM。
- 同样需要人工构造时间序列特征。

本项目优先级：

> 第一优先级。建议和 LightGBM 同步作为强基线。

### 3. CatBoost

资料来源：

- 官方文档：<https://catboost.ai/>
- GitHub：<https://github.com/catboost/catboost>

CatBoost 也是梯度提升树模型，对类别特征处理能力较强。

在风电项目中，如果后续有大量类别变量，例如：

- 场站 ID。
- 风机型号。
- 地形类别。
- 气象源类别。
- 运行状态类别。

CatBoost 可能有价值。

本项目优先级：

> 第二优先级。若多场站建模并包含大量类别特征，可以加入对比。

### 4. AutoGluon-TimeSeries

资料来源：

- 官方文档：<https://auto.gluon.ai/stable/tutorials/timeseries/index.html>
- GitHub：<https://github.com/autogluon/autogluon>

AutoGluon-TimeSeries 是自动时间序列建模框架。

它的特点是：

- 可以自动训练多种时间序列模型。
- 可以自动选择或集成模型。
- 上手快。
- 适合快速建立强基线。

适合本项目的原因：

- 数据口径稳定后，可以快速测试多种模型组合。
- 可以作为人工模型之外的自动化对照组。

限制：

- 依赖较重。
- 黑盒程度较高。
- 对业务特征和物理约束的控制不如手写模型。
- 不适合作为第一步排查数据问题的工具。

本项目优先级：

> 第二优先级。建议在 LightGBM/XGBoost 样本表稳定后运行。

### 5. NeuralForecast

资料来源：

- 官方文档：<https://nixtlaverse.nixtla.io/neuralforecast/>
- GitHub：<https://github.com/Nixtla/neuralforecast>

NeuralForecast 是 Nixtla 提供的深度时间序列预测库，包含多种神经网络模型。

适合关注：

- NHITS。
- NBEATSx。
- TFT。
- PatchTST。
- Informer、Autoformer 等长序列模型。

本项目中已经初步跑过：

- NHITS。
- NBEATSx。

结果显示，在短预测尺度上二者小幅超过 Persistence，说明深度时间序列路线可行，但不能夸大。

后续更有价值的用法是加入：

- 未来气象预报。
- 多源气象特征。
- 场站静态特征。
- 时间特征。

本项目优先级：

> 第二到第三优先级。等未来气象预报样本表稳定后系统实验。

### 6. NHITS

NHITS 是 NeuralForecast 中常用的多步时间序列预测模型。

适合：

- 稳健深度学习基线。
- 多步预测。
- 从历史功率曲线中学习多尺度模式。

优点：

- 比复杂 Transformer 更容易跑稳。
- 训练成本相对可控。
- 对历史功率序列建模能力较好。

限制：

- 如果不加入未来气象预报，对 10 天短期预测能力有限。

本项目用法：

- 超短期 10 小时：可以作为主力深度基线。
- 短期 10 天：需要配合未来气象预报或作为对照。

### 7. NBEATSx

NBEATSx 是 N-BEATS 的扩展版本，支持外生变量。

适合：

- 历史功率。
- 时间特征。
- 外部协变量。
- 多步预测。

本项目用法：

- 在加入未来气象协变量后重新评估。
- 和 NHITS、TFT、PatchTST 对比。

### 8. TFT

TFT 全称 Temporal Fusion Transformer。

资料来源：

- PyTorch Forecasting 文档：<https://pytorch-forecasting.readthedocs.io/>
- PyTorch Forecasting GitHub：<https://github.com/sktime/pytorch-forecasting>

TFT 的特点是可以同时处理：

- 历史已知变量。
- 未来已知变量。
- 静态变量。
- 多步预测。
- 注意力机制。

这和本项目非常契合，因为正式预测中会有：

- 历史功率。
- 历史实测风速风向。
- 未来气象预报。
- 场站 ID。
- 地形。
- 气象源。
- 时间特征。

优势：

- 多协变量能力强。
- 适合未来气象预报参与建模。
- 有一定可解释性。

限制：

- 训练成本高于树模型。
- 依赖数据量。
- 参数调试复杂。

本项目优先级：

> 第三优先级，但对正式 10 天短期预测很重要。建议在表格模型基线稳定后做。

### 9. PatchTST

资料来源：

- 论文和代码：<https://github.com/yuqinie98/PatchTST>

PatchTST 是基于 Transformer 的长序列时间序列模型。它把时间序列切成 patch，再学习 patch 之间的关系。

适合：

- 长历史窗口。
- 长预测步长。
- 多变量时间序列。

对本项目的意义：

- 10 天预测属于长预测任务。
- 如果有足够长的历史功率和气象序列，PatchTST 可作为长窗口深度模型候选。

限制：

- 对数据量和训练环境要求较高。
- 需要仔细处理未来气象协变量。

本项目优先级：

> 第三优先级。数据量充足后做系统对比。

### 10. Darts

资料来源：

- 官方文档：<https://unit8co.github.io/darts/>
- GitHub：<https://github.com/unit8co/darts>

Darts 是一个通用时间序列预测库，集成了很多传统模型和深度模型。

它支持：

- ARIMA。
- Exponential Smoothing。
- Prophet。
- RNN。
- NBEATS。
- TFT。
- Transformer。
- XGBoost、LightGBM 等回归模型。

优势：

- 模型类型丰富。
- 适合快速试验。
- 文档比较完整。

限制：

- 工业项目中需要注意数据格式和训练效率。
- 不一定比手写 LightGBM/XGBoost 管线更可控。

本项目优先级：

> 可作为实验框架备选。如果已有 NeuralForecast 和自写树模型脚本，Darts 不是必需。

### 11. sktime

资料来源：

- 官方文档：<https://www.sktime.net/>
- GitHub：<https://github.com/sktime/sktime>

sktime 是 Python 时间序列机器学习工具库。

适合：

- 时间序列基线。
- 传统统计模型。
- 时序回归。
- 模型评估框架。

本项目中，它更适合作为传统基线和评估工具，不是主力风电模型。

### 12. MLForecast 和 StatsForecast

资料来源：

- MLForecast：<https://nixtlaverse.nixtla.io/mlforecast/>
- StatsForecast：<https://nixtlaverse.nixtla.io/statsforecast/>

MLForecast 适合把时间序列问题转成特征工程加机器学习问题，例如：

- 滞后特征。
- 滚动特征。
- LightGBM。
- XGBoost。

StatsForecast 更偏传统统计预测，例如：

- ARIMA。
- ETS。
- Seasonal Naive。

本项目中：

- MLForecast 可以辅助自动构造正式 10 小时超短期和 10 天短期功率预测特征。
- StatsForecast 可作为传统时间序列基线，但对风电 10 天预测的主价值有限。

---

## 五、通用时间序列基础模型

这些模型不是风电专用，但在技术调研中也值得关注，因为它们代表“时间序列大模型”方向。

从本项目角度看，判断这类模型是否值得投入，不能只看“是不是大模型”或“是不是预训练模型”，更要看它是否支持以下能力：

- 能否输入未来已知协变量，例如未来 10 小时或 10 天的气象预报。
- 能否处理多变量时间序列，例如功率、风速、风向、温度、气压等。
- 能否处理不同预测提前量，例如 1 小时、10 小时、1 天、10 天。
- 能否进行少样本迁移或微调。
- 能否输出概率预测或预测区间。

这里有一个重要判断：

> WindFM 的突出优势是风电领域专用预训练，但如果其开源接口不能稳定融合未来 NWP 气象预报，那么在客户要求的 10 天短期预测任务上，它未必优于支持 future covariates 的通用时序基础模型或多协变量深度模型。

通俗说，WindFM 可能更懂“风电功率曲线长什么样”，但短期 10 天预测更需要模型知道“未来 10 天风会怎么变”。如果一个模型能直接输入未来天气预报，那么它在业务任务上可能更实用。

### 通用基础模型与 WindFM 的关键差异

| 模型 | 是否风电专用 | 预训练/基础模型属性 | 未来气象预报融合潜力 | 相对 WindFM 的潜在优势 | 主要风险 |
|---|---:|---:|---:|---|---|
| WindFM | 是 | 是 | 需要实际验证 | 风电领域针对性强，可能适合少样本迁移 | 可能不擅长直接融合多源未来 NWP |
| TimesFM | 否 | 是 | 有扩展方向，需验证接口 | 通用零样本能力强，生态活跃 | 不是风电专用 |
| Chronos / Chronos-2 | 否 | 是 | Chronos-2 更关注外生变量和多变量场景 | 通用基础模型能力强，适合做大模型对照 | 领域物理知识不足 |
| Moirai / Uni2TS | 否 | 是 | 对多变量和协变量更友好 | 更适合多变量时间序列统一建模 | 工程适配复杂 |
| TinyTimeMixer / TTM | 否 | 是 | 支持多变量、外生变量方向 | 模型较轻，适合工程实验 | 风电专用性不足 |
| TabPFN-TS | 否 | 是 | 适合表格化时序和协变量场景 | 可能更适合小样本、强特征表场景 | 长序列和长预测步长需验证 |
| Lag-Llama | 否 | 是 | 需要验证外生变量能力 | 类似语言模型的概率预测思路，适合做基础模型对照 | 不是风电专用，工程适配需测试 |
| MOMENT | 否 | 是 | 需要额外设计 | 可做时间序列表征和迁移学习 | 不直接面向功率预测 |

### 1. TimesFM

资料来源：

- GitHub：<https://github.com/google-research/timesfm>

TimesFM 是 Google Research 提出的时间序列基础模型。

特点：

- 面向通用时间序列预测。
- 支持零样本预测。
- 可以作为通用基础模型对照。

对本项目的价值：

- 可测试其对风电功率序列的零样本能力。
- 可作为 WindFM 的通用基础模型对照。
- 如果其外生变量或协变量接口可稳定使用，可测试“历史功率 + 未来气象预报”的预测能力。

限制：

- 不是风电专用。
- 是否能有效使用多源未来气象预报，需要看具体接口和工程适配。
- 对 10 天风电预测不一定优于专门的 NWP 融合模型。

### 2. Chronos / Chronos-2

资料来源：

- GitHub：<https://github.com/amazon-science/chronos-forecasting>

Chronos 是 Amazon Science 提出的时间序列基础模型。它把时间序列数值离散化后，用类似语言模型的方式预测未来。

从项目选型角度，Chronos 系列值得关注的原因是：较新的 Chronos-2 方向开始强调更通用的时间序列预测能力，包括多变量、外生变量或更复杂的预测任务支持。这类能力正是 WindFM 可能不具备或需要额外验证的部分。

和 WindFM 的相似点：

- 都有 token 化时间序列的思想。
- 都借鉴了大语言模型的生成式预测思路。

区别：

- Chronos 是通用时间序列模型。
- WindFM 是风电领域模型。

本项目价值：

- 可作为通用基础模型对照。
- 适合验证“风电专用基础模型是否比通用基础模型更适合本项目”。
- 如果 Chronos-2 能稳定融合外生变量，可重点测试未来 NWP 气象预报输入。
- 可用于比较“领域专用预训练”和“通用大规模预训练”在风电预测上的差异。

### 3. Moirai

资料来源：

- GitHub：<https://github.com/SalesforceAIResearch/uni2ts>

Moirai 是 Salesforce AI Research 的通用时间序列基础模型方向，属于 Uni2TS 项目的一部分。

特点：

- 面向多领域时间序列。
- 支持概率预测。
- 关注统一时间序列建模。

本项目价值：

- 可作为概率预测对照。
- 可和 WindFM 比较领域专用模型和通用模型的差异。
- 对多变量时间序列和协变量建模更友好，适合测试“功率 + 未来气象预报 + 时间特征”的统一建模。

限制：

- 不是风电专用。
- 工程依赖和数据格式需要单独适配。

### 4. TinyTimeMixer / TTM

资料来源：

- GitHub：<https://github.com/ibm-granite/granite-tsfm>

TinyTimeMixer，简称 TTM，是 IBM 开源的轻量级时间序列基础模型方向。

它的特点是：

- 模型相对轻量。
- 面向多变量时间序列预测。
- 支持预训练和迁移应用。
- 更适合工程侧快速实验。

对本项目的价值：

- 可作为 WindFM 之外的轻量级预训练模型对照。
- 如果能输入外生变量，可测试未来气象预报融合能力。
- 对服务器资源要求可能低于更大的基础模型。

限制：

- 不是风电专用。
- 需要验证对 10 小时和 10 天预测步长的适配效果。
- 需要验证对多源 NWP 的输入格式支持。

### 5. TabPFN-TS

资料来源：

- TabPFN-Time-Series GitHub：<https://github.com/PriorLabs/tabpfn-time-series>
- TabPFN GitHub：<https://github.com/PriorLabs/TabPFN>

TabPFN-TS 是将 TabPFN 思路扩展到时间序列预测的方向。它更偏“表格化时间序列 + 预训练推断”的路线。

对本项目的潜在价值：

- 对小样本场景可能有优势。
- 更容易与手工特征表结合。
- 如果把未来气象预报、历史功率、地形、状态标记整理成结构化特征，它可能比纯序列模型更容易融合这些信息。

与 WindFM 的差异：

- WindFM 更偏风电序列基础模型。
- TabPFN-TS 更偏通用表格/时序预训练推断。
- 对未来气象预报这类结构化协变量，TabPFN-TS 可能更容易接入。

限制：

- 风电专用性不足。
- 长预测步长能力需要验证。
- 对大规模多场站数据的训练和推理效率需要测试。

### 6. Lag-Llama

资料来源：

- GitHub：<https://github.com/time-series-foundation-models/lag-llama>

Lag-Llama 是一种时间序列基础模型，思路上借鉴了大语言模型的序列生成方式，用于概率时间序列预测。

对本项目的价值：

- 可作为 Chronos、TimesFM、WindFM 之外的基础模型对照。
- 适合评估概率预测能力。
- 可用于比较“通用预训练模型”和“风电专用预训练模型”的差异。

限制：

- 不是风电专用。
- 对未来气象预报、多源 NWP、长达 10 天预测步长的支持能力需要实际验证。
- 工程适配成本可能高于 LightGBM/XGBoost 等表格模型。

### 7. MOMENT

资料来源：

- GitHub：<https://github.com/moment-timeseries-foundation-model/moment>

MOMENT 是通用时间序列基础模型，关注时间序列表征学习、预测、分类、异常检测等任务。

本项目价值：

- 可用于时间序列表征。
- 可尝试功率序列 embedding 加树模型或融合模型。

限制：

- 不是直接面向风电功率预测。
- 对未来气象预报协变量的支持需要额外设计。

### 8. 通用基础模型在本项目中的定位

TimesFM、Chronos、Moirai、TTM、TabPFN-TS、Lag-Llama、MOMENT 等模型可以作为技术储备，但不要替代表格强基线。

原因：

- 客户目标是明确的风电功率预测。
- 10 天短期预测必须依赖未来气象预报。
- 通用基础模型未必理解风电功率曲线、限电、停机、地形订正等业务因素。

建议定位：

> 作为 WindFM 之外的通用时间序列基础模型对照，用于评估“领域专用基础模型”和“通用时序基础模型”在风电功率预测上的差异。特别需要关注这些模型是否能够稳定融合未来 NWP 气象预报；如果可以，它们在 10 天短期预测上可能具备 WindFM 所不具备的业务优势。

---

## 六、AI 天气模型和上游气象源

这些模型不是直接预测风电功率，而是预测天气。它们对项目的价值在于提供或增强气象预报源。

### 1. GraphCast

资料来源：

- GitHub：<https://github.com/google-deepmind/graphcast>

GraphCast 是 Google DeepMind 提出的机器学习天气预报模型。

作用：

- 生成全球中期天气预报。
- 可作为未来气象预报来源之一。

对本项目的意义：

- 它不是直接输出风电功率。
- 如果能获得 GraphCast 类天气预报结果，可以作为多源气象之一输入功率预测模型。

### 2. Pangu-Weather

资料来源：

- GitHub：<https://github.com/198808xc/Pangu-Weather>

Pangu-Weather 是华为提出的 AI 天气预报模型方向。

对本项目的意义：

- 如果企业侧已有华为或盘古相关气象数据，可以作为多源气象输入。
- 关键不是直接使用 Pangu-Weather 代码，而是把它的预报结果和 EC 等其他气象源融合。

### 3. FourCastNet

资料来源：

- GitHub：<https://github.com/NVlabs/FourCastNet>

FourCastNet 是 NVIDIA 提出的全球天气预测模型方向。

对本项目的意义：

- 属于上游气象模型。
- 可作为气象预报源或技术储备。

### 4. AI 天气模型和功率预测模型的关系

可以这样理解：

```text
AI 天气模型 / 数值天气预报
        ↓
未来风速、风向、温度、气压等气象预报
        ↓
风电功率预测模型
        ↓
未来 10 小时 / 10 天功率预测
```

也就是说，GraphCast、Pangu-Weather、FourCastNet 这类模型主要解决“未来天气是什么”，而 LightGBM、TFT、WindFM 等模型解决“未来天气对应多少风电功率”。

---

## 七、风电物理与工程开源工具

### 1. FLORIS

资料来源：

- NREL FLORIS：<https://github.com/NREL/floris>

FLORIS 是 NREL 的风电场控制和尾流建模工具。

它主要用于：

- 风电场尾流模拟。
- 风机布局和控制分析。
- 风机间相互影响建模。

对功率预测的价值：

- 可辅助理解风场物理规律。
- 可用于构造物理约束或尾流相关特征。
- 对复杂地形和多风机相互影响分析有参考意义。

限制：

- 它不是直接的机器学习预测模型。
- 需要较完整的风机布局、机型和地形信息。

### 2. windpowerlib

资料来源：

- GitHub：<https://github.com/wind-python/windpowerlib>

windpowerlib 是用于根据天气数据计算风机或风电场功率输出的 Python 库。

它可以用于：

- 风速到功率的物理转换。
- 功率曲线建模。
- 构造物理基线。

对本项目的价值：

- 可建立“气象风速 + 功率曲线”的物理基线。
- 可用于模型输出后处理。
- 可辅助判断机器学习预测是否违反基本功率曲线规律。

### 3. OpenOA

资料来源：

- GitHub：<https://github.com/NREL/OpenOA>

OpenOA 是 NREL 的风电运行分析工具，主要用于风电场运营数据分析。

对本项目的价值：

- 可借鉴其数据清洗、可用率分析、能量评估思路。
- 对限电、停机、异常样本识别有参考意义。

### 4. WIND Toolkit

资料来源：

- NREL WIND Toolkit：<https://www.nrel.gov/grid/wind-toolkit>

WIND Toolkit 是大规模风能数据集。WindFM 预训练数据来源与此相关。

对本项目的价值：

- 可作为理解 WindFM 数据背景的资料。
- 可作为风资源研究参考。

限制：

- 它不是中国区域数据。
- 不能直接替代企业京津冀气象和功率数据。

---

## 八、针对京津冀丘陵山地的模型注意点

客户场站主要在京津冀，部分是丘陵、山地。这会影响模型选择和特征设计。

### 1. 最近格点可能不够

复杂地形下，气象网格点和实际风机位置之间差异可能很大。

仅使用最近格点可能存在问题：

- 网格分辨率不足。
- 山脊、山谷风况差异大。
- 局地风向受地形影响强。
- 风切变明显。

建议后续使用：

- 最近格点。
- 周边多格点均值。
- 周边多格点最大值、最小值。
- 风速梯度。
- 不同高度风速差。
- 地形高度、坡度、坡向。

### 2. 需要本地化订正

气象预报是区域模型输出，场站功率预测需要把区域天气转成场站风况。

本地化订正可以包括：

- 对预报风速做偏差修正。
- 按风向区间修正。
- 按季节修正。
- 按地形类别修正。
- 按气象源和提前量修正。

### 3. 需要区分限电和自然低风

低功率不一定是因为风小，也可能是：

- 限电。
- 停机。
- 检修。
- 通信异常。
- 风机故障。

如果不区分，模型会把业务原因误学成气象原因。

因此后续数据需求中必须强调：

- 限电标记。
- 停机标记。
- 检修记录。
- 可用容量。
- 风机在线数量。

---

## 九、推荐实验路线

### 1. 第一阶段：统一评估口径

目标：

- 明确客户准确率公式。
- 明确超短期和短期预测粒度。
- 明确是否剔除限电、停机异常。
- 明确装机容量和可用容量口径。

输出：

- 统一评价脚本。
- 统一样本表字段。
- 统一训练集、验证集、测试集划分。

### 2. 第二阶段：树模型强基线

模型：

- LightGBM。
- XGBoost。
- CatBoost 可选。

输入：

- 多源未来气象预报。
- 历史功率滞后。
- 历史实测风速风向。
- 时间周期特征。
- 地形特征。
- 运行状态。
- 质量标记。

预测：

- 超短期未来 10 小时。
- 短期未来 10 天。

目标：

- 建立可解释强基线。
- 发现数据问题。
- 作为后续深度模型和 WindFM 的比较对象。

### 3. 第三阶段：WindFM 实验

目标：

- 跑通官方样例。
- 适配苗庄或京津冀场站数据。
- 做零样本和少样本测试。
- 和 LightGBM/XGBoost、玖天结果对比。

重点问题：

- 是否支持未来气象预报作为输入。
- 是否支持 10 小时和 10 天输出。
- 是否需要功率归一化。
- 对京津冀场站零样本效果如何。
- 是否支持概率预测。

### 4. 第四阶段：深度多协变量模型

模型：

- TFT。
- PatchTST。
- NHITS。
- NBEATSx。

重点：

- 把未来气象预报作为 known future covariates。
- 把场站信息作为 static covariates。
- 把历史功率和历史实测气象作为 past covariates。

适用：

- 数据量达到 3 个月、6 个月、1 年后逐步评估。

### 5. 第五阶段：通用基础模型对照

模型：

- TimesFM。
- Chronos。
- Moirai。
- MOMENT。

目标：

- 对比通用时间序列基础模型和 WindFM 的差异。
- 判断风电专用预训练是否有优势。

### 6. 第六阶段：融合和持续校准

最终业务模型不一定只用一个模型。

可以做融合：

```text
最终预测 = w1 * LightGBM + w2 * XGBoost + w3 * TFT + w4 * WindFM
```

权重可以按以下维度动态调整：

- 预测提前量。
- 气象源误差。
- 风速区间。
- 风向区间。
- 季节。
- 地形类别。
- 场站。

持续校准逻辑：

1. 每次预测保存模型版本和输入数据。
2. 等实测功率回来后计算误差。
3. 分析不同提前量、气象源、风速区间误差。
4. 调整偏差修正和融合权重。
5. 定期滚动重训。

---

## 十、推荐优先级

### 当前最优先

1. 明确客户准确率公式。
2. 构建 10 小时、10 天统一样本表。
3. LightGBM/XGBoost 多源气象基线。
4. 与玖天预测结果做同口径评估。

### 第二优先级

1. WindFM 官方样例跑通。
2. 苗庄数据适配 WindFM。
3. WindFM 与树模型、玖天结果对比。
4. NeuralForecast 中 TFT、NHITS、NBEATSx 对比。

### 第三优先级

1. PatchTST 长序列建模。
2. AutoGluon 自动基线。
3. TimesFM、Chronos、Moirai、MOMENT 通用基础模型对照。
4. FLORIS、windpowerlib 等物理工具辅助约束。

---

## 十一、正式报告中的简短表述

可采用如下表述：

> 经调研，WindFM 是值得关注的风电专用基础模型。它与普通时间序列模型不同，主要价值在于利用大规模风电数据进行预训练，具备零样本、少样本和概率预测方面的潜在能力。但 WindFM 是否适合本项目，需要结合客户的 10 小时超短期和 10 天短期预测口径进行验证，尤其需要评估其对未来气象预报的使用能力。当前更稳妥的路线是先用 LightGBM、XGBoost 建立多源气象强基线，再用 WindFM、TFT、PatchTST、NHITS 等模型做对照实验。这样既能保持工程可解释性，也能探索风电基础模型的先进性。

关于“是否应该押宝 WindFM”，可表述为：

> 不建议一开始押宝 WindFM。WindFM 很有研究价值，但项目要先满足客户准确率考核，因此需要可解释、可控、能融合多源气象的强基线。WindFM 更适合作为风电专用基础模型探索项，与 LightGBM/XGBoost、TFT 和玖天结果做同口径比较。

关于“开源模型和企业自研模型的关系”，可表述为：

> 企业一般不会完全从零发明模型，也不会直接裸用开源模型。更常见的是基于成熟开源模型或论文方法，结合自己的场站数据、气象源、运行状态、地形、考核口径做定制训练和持续校准。真正的壁垒不只是模型结构，而是数据口径、气象订正、物理约束、业务规则和长期校准。

---

## 十二、建议进一步确认的问题

1. 客户所说准确率 90%、95% 的计算公式是什么？
2. 评价时是否剔除限电、停机、检修、通信异常时段？
3. 评价归一化使用装机容量还是实时可用容量？
4. 超短期 10 小时和短期 10 天的时间粒度是多少，15 分钟还是 1 小时？
5. 是否能拿到玖天同时间段预测结果，用于同口径对比？
6. 是否能提供多源气象预报，包括 EC、华为、盘古或其他来源？
7. 气象预报的起报时间、有效时间、提前量结构是否完整？
8. 是否能提供场站地形信息、风机坐标、风机型号和轮毂高度？
9. 是否能提供限电、停机、检修、可用容量记录？
10. 未来是否允许在企业服务器上直接训练或调用 WindFM、TFT、PatchTST、TimesFM、Chronos、Moirai、TTM 等深度模型或预训练模型？

---

## 十三、资料来源

### WindFM

- WindFM 论文：<https://arxiv.org/abs/2509.06311>
- WindFM GitHub：<https://github.com/shiyu-coder/WindFM>
- WindFM Hugging Face 作者页：<https://huggingface.co/shiyu-coder>
- NREL WIND Toolkit：<https://www.nrel.gov/grid/wind-toolkit>

### 表格模型

- LightGBM 文档：<https://lightgbm.readthedocs.io/>
- LightGBM GitHub：<https://github.com/microsoft/LightGBM>
- XGBoost 文档：<https://xgboost.readthedocs.io/>
- XGBoost GitHub：<https://github.com/dmlc/xgboost>
- CatBoost 文档：<https://catboost.ai/>
- CatBoost GitHub：<https://github.com/catboost/catboost>

### 时间序列和深度学习框架

- AutoGluon-TimeSeries：<https://auto.gluon.ai/stable/tutorials/timeseries/index.html>
- AutoGluon GitHub：<https://github.com/autogluon/autogluon>
- NeuralForecast 文档：<https://nixtlaverse.nixtla.io/neuralforecast/>
- NeuralForecast GitHub：<https://github.com/Nixtla/neuralforecast>
- MLForecast 文档：<https://nixtlaverse.nixtla.io/mlforecast/>
- StatsForecast 文档：<https://nixtlaverse.nixtla.io/statsforecast/>
- Darts 文档：<https://unit8co.github.io/darts/>
- Darts GitHub：<https://github.com/unit8co/darts>
- sktime 文档：<https://www.sktime.net/>
- sktime GitHub：<https://github.com/sktime/sktime>
- PyTorch Forecasting 文档：<https://pytorch-forecasting.readthedocs.io/>
- PyTorch Forecasting GitHub：<https://github.com/sktime/pytorch-forecasting>
- PatchTST GitHub：<https://github.com/yuqinie98/PatchTST>

### 通用时间序列基础模型

- TimesFM GitHub：<https://github.com/google-research/timesfm>
- Chronos GitHub：<https://github.com/amazon-science/chronos-forecasting>
- Moirai / Uni2TS GitHub：<https://github.com/SalesforceAIResearch/uni2ts>
- TinyTimeMixer / TTM GitHub：<https://github.com/ibm-granite/granite-tsfm>
- TabPFN-Time-Series GitHub：<https://github.com/PriorLabs/tabpfn-time-series>
- TabPFN GitHub：<https://github.com/PriorLabs/TabPFN>
- Lag-Llama GitHub：<https://github.com/time-series-foundation-models/lag-llama>
- MOMENT GitHub：<https://github.com/moment-timeseries-foundation-model/moment>

### AI 天气模型

- GraphCast GitHub：<https://github.com/google-deepmind/graphcast>
- Pangu-Weather GitHub：<https://github.com/198808xc/Pangu-Weather>
- FourCastNet GitHub：<https://github.com/NVlabs/FourCastNet>

### 风电物理与工程工具

- FLORIS GitHub：<https://github.com/NREL/floris>
- windpowerlib GitHub：<https://github.com/wind-python/windpowerlib>
- OpenOA GitHub：<https://github.com/NREL/OpenOA>
