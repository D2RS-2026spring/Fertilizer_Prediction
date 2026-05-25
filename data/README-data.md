# 数据说明

## 数据来源

本项目数据来自 Kaggle Playground Series - Predicting Optimal Fertilizers 竞赛数据集。

原始数据地址：

https://www.kaggle.com/competitions/playground-series-s5e6

本项目使用的数据文件包括：

- train.csv：训练集，包含输入特征和目标变量
- test.csv：测试集，只包含输入特征
- sample_submission.csv：Kaggle 提交格式示例

## 任务目标

本项目的目标是根据环境、土壤、作物和养分信息，预测最适合的肥料类型。

预测目标变量为：

- Fertilizer Name

## 字段说明

| 字段名 | 含义 |
|---|---|
| Temparature | 温度 |
| Humidity | 湿度 |
| Moisture | 土壤含水量 |
| Soil Type | 土壤类型 |
| Crop Type | 作物类型 |
| Nitrogen | 氮含量 |
| Potassium | 钾含量 |
| Phosphorous | 磷含量 |
| Fertilizer Name | 肥料名称，模型预测目标 |

## 数据处理说明

本项目主要进行以下数据处理：

1. 检查缺失值
2. 统计不同肥料类别的样本数量
3. 对 Soil Type 和 Crop Type 进行类别编码
4. 对数值型变量进行标准化
5. 构造氮、磷、钾比例特征
6. 构造温湿度交互特征
7. 使用 LightGBM 模型进行多分类预测

## 评价指标

本任务使用 MAP@3 作为评价指标。模型需要为每个测试样本预测最可能的前三个肥料类别。
