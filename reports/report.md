# 基于机器学习的肥料推荐系统研究报告

## 1. 项目背景

农业施肥决策需要综合考虑作物类型、土壤类型、温度、湿度、土壤水分以及氮磷钾含量等因素。不同作物和不同土壤条件下，适合使用的肥料类型不同。

本项目基于 Kaggle Predicting Optimal Fertilizers 数据集，构建一个肥料推荐模型，根据环境、土壤、作物和养分特征预测合适的肥料类型。

## 2. 数据来源

数据来源于 Kaggle Playground Series - Predicting Optimal Fertilizers 竞赛。

项目使用的数据文件包括：

- train.csv：训练集，包含输入特征和目标变量 Fertilizer Name
- test.csv：测试集，只包含输入特征
- sample_submission.csv：提交格式示例

由于原始数据文件来自 Kaggle，本仓库不直接上传原始 CSV 数据，只在 data/README-data.md 中说明数据来源和下载方式。

## 3. 数据字段说明

本项目主要使用以下字段：

| 字段名 | 含义 |
|---|---|
| id | 样本编号 |
| Temparature | 温度 |
| Humidity | 湿度 |
| Moisture | 土壤含水量 |
| Soil Type | 土壤类型 |
| Crop Type | 作物类型 |
| Nitrogen | 氮含量 |
| Potassium | 钾含量 |
| Phosphorous | 磷含量 |
| Fertilizer Name | 肥料名称，模型预测目标 |

## 4. 数据分析

通过 `eda.py` 对训练数据进行了探索性分析。

训练集规模为：

```text
750000 行，10 列

缺失值统计结果显示，各字段缺失值数量均为 0，说明数据完整性较好。

肥料类别分布如下：

14-35-14    114436
10-26-26    113887
17-17-17    112453
28-28       111158
20-20       110889
DAP          94860
Urea         92317

从类别分布看，不同肥料类别样本数量存在一定差异，但整体没有出现极端类别稀缺的情况。

生成的可视化结果位于：

outputs/figures/

主要图表包括：

fertilizer_distribution.png：肥料类别分布
soil_type_distribution.png：土壤类型分布
crop_type_distribution.png：作物类型分布
temparature_distribution.png：温度分布
humidity_distribution.png：湿度分布
moisture_distribution.png：土壤水分分布
nitrogen_distribution.png：氮含量分布
potassium_distribution.png：钾含量分布
phosphorous_distribution.png：磷含量分布
5. 数据预处理

模型训练前进行了以下处理：

删除不参与训练的 id 字段。
将 Soil Type 和 Crop Type 作为类别特征处理。
使用 OneHotEncoder 对类别变量进行编码。
使用 StandardScaler 对数值变量进行标准化。
将 Fertilizer Name 作为分类预测目标。

本项目没有直接使用原始字符串特征训练模型，而是通过编码方式将类别变量转换为机器学习模型可以处理的数值形式。

6. 模型方法

本项目使用 Logistic Regression 作为基线多分类模型。

模型流程如下：

输入特征
→ 类别变量 OneHotEncoder
→ 数值变量 StandardScaler
→ Logistic Regression 分类模型
→ 输出前三个肥料预测结果

选择 Logistic Regression 的原因是：

模型训练速度较快。
便于在普通电脑上复现。
可以作为肥料推荐任务的基线模型。
输出类别概率后，可以根据概率排序得到 Top 3 推荐结果。
7. 评价指标

本项目使用 MAP@3 作为评价指标。

MAP@3 适合推荐任务，因为模型不只输出一个肥料类别，而是为每个样本输出前三个可能的肥料类别。

如果真实肥料类别排在第 1 位，得分最高；如果排在第 2 或第 3 位，得分相应降低；如果前三个预测都不包含真实类别，则得分为 0。

8. 实验结果

本项目使用 10000 条训练样本进行快速可复现实验，采用 3 折分层交叉验证。

实验结果如下：

Fold 1 MAP@3: 0.2710
Fold 2 MAP@3: 0.2718
Fold 3 MAP@3: 0.2753
Mean MAP@3: 0.27285

模型输出文件包括：

outputs/submission.csv
outputs/model_metrics.txt
outputs/model/fertilizer_model.joblib

其中：

submission.csv：测试集预测结果
model_metrics.txt：模型训练指标记录
fertilizer_model.joblib：训练后的模型文件
9. 结果分析

从实验结果看，基线模型可以完成肥料类别预测任务，但 MAP@3 分数仍有提升空间。

可能原因包括：

Logistic Regression 是线性模型，对复杂非线性关系表达能力有限。
肥料推荐可能受到作物、土壤和养分之间复杂交互关系影响。
当前实验只使用了 10000 条样本进行快速训练，没有使用完整训练集。
没有加入更复杂的特征工程，例如氮磷钾比例、温湿度交互项等高级特征。

后续可以尝试：

使用 LightGBM、XGBoost、Random Forest 等非线性模型。
使用完整训练集训练模型。
构造更多农业相关特征。
对不同土壤类型和作物类型分别分析肥料分布。
10. 可复现说明

在本地电脑中可按以下步骤复现实验。

10.1 安装环境
python -m pip install -r requirements.txt
10.2 准备数据

从 Kaggle 下载数据，并放入：

data/

目录结构应为：

data/
├── README-data.md
├── train.csv
├── test.csv
└── sample_submission.csv
10.3 运行 EDA
python eda.py

运行后生成：

outputs/data_summary.csv
outputs/figures/
10.4 运行模型训练
python train.py --sample_size 10000

运行后生成：

outputs/submission.csv
outputs/model_metrics.txt
outputs/model/fertilizer_model.joblib

如需使用更多样本，可调整参数：

python train.py --sample_size 50000
11.总结

本项目完成了从数据准备、数据分析、可视化、模型训练到结果输出的完整流程。通过本项目，可以了解农业肥料推荐任务的基本建模流程，并掌握 GitHub 仓库整理、代码复现和实验报告撰写的方法。

当前模型作为可复现的基线模型已经能够输出肥料推荐结果。后续可以进一步尝试更复杂的机器学习模型和特征工程方法，以提高 MAP@3 评价分数。
