# Fertilizer Prediction 肥料推荐系统

## 项目简介

本项目基于 Kaggle Predicting Optimal Fertilizers 数据集，构建一个可复现的肥料推荐系统。

项目根据温度、湿度、土壤水分、土壤类型、作物类型以及氮磷钾含量等信息，预测适合使用的肥料类型。

## 数据来源

数据来自 Kaggle Playground Series - Predicting Optimal Fertilizers。

数据说明见：

```text
data/README-data.md

原始数据文件包括：

train.csv
test.csv
sample_submission.csv

由于数据来自 Kaggle，本仓库不直接上传原始 CSV 数据。复现实验时需要自行从 Kaggle 下载数据，并放入 data/ 文件夹。

项目结构
Fertilizer_Prediction/
├── data/
│   └── README-data.md
├── outputs/
│   ├── data_summary.csv
│   ├── model_metrics.txt
│   ├── submission.csv
│   ├── figures/
│   └── model/
├── reports/
│   └── report.md
├── eda.py
├── train.py
├── requirements.txt
└── README.md
环境安装
python -m pip install -r requirements.txt
数据分析

运行：

python eda.py

生成结果：

outputs/data_summary.csv
outputs/figures/

主要可视化图表包括：

肥料类别分布
土壤类型分布
作物类型分布
温度、湿度、土壤水分分布
氮、磷、钾含量分布
模型训练

运行：

python train.py --sample_size 10000

训练完成后生成：

outputs/submission.csv
outputs/model_metrics.txt
outputs/model/fertilizer_model.joblib
模型结果

本项目使用 Logistic Regression 作为基线多分类模型，采用 3 折分层交叉验证。

实验结果：

Fold 1 MAP@3: 0.2710
Fold 2 MAP@3: 0.2718
Fold 3 MAP@3: 0.2753
Mean MAP@3: 0.27285
研究报告

完整研究报告见：

reports/report.md

报告内容包括：

项目背景
数据来源
数据字段说明
数据分析与可视化
数据预处理
模型方法
MAP@3 评价指标
实验结果
可复现步骤
总结与改进方向
