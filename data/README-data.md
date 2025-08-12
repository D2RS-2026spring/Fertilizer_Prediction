# **预测最佳肥料**



**您的目标：**您的目标是为不同的天气、土壤条件和作物选择最佳肥料。



## 数据集描述

本次竞赛的数据集（训练和测试）是由在[肥料预测](https://www.kaggle.com/datasets/irakozekelly/fertilizer-prediction)数据集上训练的深度学习模型生成的。特征分布与原始分布接近，但并不完全相同。请随意使用原始数据集作为本次竞赛的一部分，既可以探索差异，也可以看看将原始数据集纳入训练是否可以提高模型性能。

## 文件

- **train.csv** - 训练数据集; 是分类目标`Fertilizer Name`
- **test.csv** - 测试数据集;您的目标是预测每行最多三个值，以空格分隔。`Fertilizer Name`
- **sample_submission.csv** - 格式正确的示例提交文件。



## 评估

根据平均精度 @ 3 （MAP@3） 对提交进行评估：
$$
\mathrm{MAP}@3 = \frac{1}{U} \sum_{u=1}^{U} \sum_{k=1}^{\min(n, 3)} P(k) \times \mathrm{rel}(k)
$$

其中 $U$ 是观测数，$P(k)$ 是截止时的精度 $k$,$n$ 是每个观测值的预测数，并且 $rel(k)$ 是一个指标函数，如果排名中的项目 $k$ 是相关（正确）标签，否则为零。