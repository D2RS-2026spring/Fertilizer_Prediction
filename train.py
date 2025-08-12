import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
import os
import time

# 修正后的MAP@3计算函数
def map3(y_true, y_pred_proba, k=3):
    """
    计算Mean Average Precision @ k
    y_true: 真实标签的整数编码 (n_samples,)
    y_pred_proba: 预测概率矩阵 (n_samples, n_classes)
    k: 考虑的前k个预测
    """
    n_samples = y_true.shape[0]
    map_val = 0.0
    
    for i in range(n_samples):
        # 获取当前样本的预测概率
        proba = y_pred_proba[i]
        # 获取top k的索引
        topk_idx = np.argsort(proba)[::-1][:k]
        
        # 计算精度@k
        precision_sum = 0.0
        correct_count = 0
        
        for j, idx in enumerate(topk_idx):
            if idx == y_true[i]:
                correct_count += 1
                precision_at_j = correct_count / (j + 1)
                precision_sum += precision_at_j
        
        # 如果样本有正确预测，则添加到MAP
        if correct_count > 0:
            map_val += precision_sum / min(correct_count, k)
        else:
            # 没有正确预测，添加0
            map_val += 0
    
    return map_val / n_samples

# 数据预处理 - 添加特征工程
def preprocess_data(df, label_encoders=None, scaler=None, is_train=True):
    df = df.copy()
    categorical_cols = ['Soil Type', 'Crop Type']
    
    # 添加特征工程
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
    df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)
    df['Temp_Humidity'] = df['Temparature'] * df['Humidity'] / 100
    df['Nutrient_Sum'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
    
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        le_fertilizer = LabelEncoder()
        df['Fertilizer Name'] = le_fertilizer.fit_transform(df['Fertilizer Name'])
        label_encoders['fertilizer'] = le_fertilizer
        
        numeric_cols = [
            'Temparature', 'Humidity', 'Moisture', 
            'Nitrogen', 'Potassium', 'Phosphorous',
            'N_P_ratio', 'N_K_ratio', 'P_K_ratio',
            'Temp_Humidity', 'Nutrient_Sum'
        ]
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        X = df.drop(['id', 'Fertilizer Name'], axis=1)
        y = df['Fertilizer Name']
        return X, y, label_encoders, scaler
    else:
        for col in categorical_cols:
            le = label_encoders[col]
            df[col] = le.transform(df[col])
        
        numeric_cols = [
            'Temparature', 'Humidity', 'Moisture', 
            'Nitrogen', 'Potassium', 'Phosphorous',
            'N_P_ratio', 'N_K_ratio', 'P_K_ratio',
            'Temp_Humidity', 'Nutrient_Sum'
        ]
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        
        X = df.drop(['id'], axis=1)
        return X, label_encoders, scaler

# 加载数据
train_df = pd.read_csv('/kaggle/input/optimal-fertilizers/data/train.csv')
test_df = pd.read_csv('/kaggle/input/optimal-fertilizers/data/test.csv')

# 数据预处理 - 添加特征工程
X, y, label_encoders, scaler = preprocess_data(train_df, is_train=True)

# 分析类别分布
class_counts = np.bincount(y)
print("Class distribution:")
for i, count in enumerate(class_counts):
    class_name = label_encoders['fertilizer'].inverse_transform([i])[0]
    print(f"{class_name}: {count} samples ({count/len(y):.2%})")

# 使用分层K折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = []
val_scores = []
train_times = []

# 设置LightGBM参数 - 更强大的模型
params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y)),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 512,  # 增加模型复杂度
    'max_depth': 12,    # 增加深度
    'learning_rate': 0.005,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'seed': 42,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0
}

# 训练多个模型并进行交叉验证
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold+1}/5")
    print(f"{'='*50}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 添加记录器
    evals_result = {}
    
    # 训练模型
    start_time = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(evals_result)
        ]
    )
    fold_time = time.time() - start_time
    train_times.append(fold_time)
    
    # 在验证集上评估
    val_proba = model.predict(X_val, num_iteration=model.best_iteration)
    val_map3 = map3(y_val.values, val_proba)
    val_scores.append(val_map3)
    
    # 保存模型
    models.append(model)
    print(f"Fold {fold+1} MAP@3: {val_map3:.4f} | Training time: {fold_time:.2f} seconds")
    
    # 绘制本折的训练历史
    plt.figure(figsize=(10, 6))
    lgb.plot_metric(evals_result, metric='multi_logloss')
    plt.title(f'Fold {fold+1} Training Metrics')
    plt.grid(True)
    plt.savefig(f'/kaggle/working/training_metrics_fold{fold+1}.png')
    plt.close()

# 打印交叉验证结果
print("\nCross-validation results:")
print(f"Average MAP@3: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
print(f"Total training time: {sum(train_times):.2f} seconds")

# 创建模型目录
os.makedirs('/kaggle/working/model', exist_ok=True)

# 保存整体模型和预处理对象
for i, model in enumerate(models):
    model.save_model(f'/kaggle/working/model/lgb_model_fold{i+1}.txt')
joblib.dump(label_encoders, '/kaggle/working/model/label_encoders.pkl')
joblib.dump(scaler, '/kaggle/working/model/scaler.pkl')

# 使用整个训练集重新训练一个最终模型
print("\nTraining final model on full dataset...")
start_time = time.time()
final_model = lgb.train(
    params,
    lgb.Dataset(X, label=y),
    num_boost_round=2000,
    callbacks=[
        lgb.log_evaluation(period=100),
    ]
)
final_model.save_model('/kaggle/working/model/lgb_model_final.txt')
print(f"Final model training time: {time.time() - start_time:.2f} seconds")

# 在验证集上评估最终模型 - 使用第一个折的验证集作为示例
val_idx = list(skf.split(X, y))[0][1]
X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

val_proba = final_model.predict(X_val)
val_map3 = map3(y_val.values, val_proba)
print(f"\nFinal Model Validation MAP@3: {val_map3:.4f}")

# 随机选取10个样本展示结果
le_fertilizer = label_encoders['fertilizer']
sample_indices = np.random.choice(len(X_val), 10, replace=False)

print("\nSample Validation Predictions:")
print("=" * 60)
for i, idx in enumerate(sample_indices):
    true_label = y_val.iloc[idx]
    proba = val_proba[idx]
    
    # 获取top3预测
    top3_idx = np.argsort(proba)[::-1][:3]
    top3_labels = le_fertilizer.inverse_transform(top3_idx)
    top3_proba = proba[top3_idx]
    
    true_name = le_fertilizer.inverse_transform([true_label])[0]
    
    print(f"Sample {i+1}:")
    print(f"  True Fertilizer: {true_name}")
    print("  Top 3 Predictions:")
    for j, (label, prob) in enumerate(zip(top3_labels, top3_proba)):
        print(f"    {j+1}. {label} ({prob:.4f})")
    print("-" * 60)

# 处理测试集
X_test, label_encoders, scaler = preprocess_data(
    test_df, 
    label_encoders=label_encoders, 
    scaler=scaler,
    is_train=False
)

# 预测测试集 - 使用模型集成
test_probas = []
for model in models:
    test_proba = model.predict(X_test, num_iteration=model.best_iteration)
    test_probas.append(test_proba)

# 平均概率
avg_test_proba = np.mean(test_probas, axis=0)

# 创建提交文件
submission = pd.DataFrame({'id': test_df['id']})

# 获取top3预测结果
le_fertilizer = label_encoders['fertilizer']
fertilizer_names = []

for proba in avg_test_proba:
    top3_idx = np.argsort(proba)[::-1][:3]
    top3_labels = le_fertilizer.inverse_transform(top3_idx)
    fertilizer_names.append(" ".join(top3_labels))

submission['Fertilizer Name'] = fertilizer_names

# 保存结果
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nSubmission file saved as '/kaggle/working/submission.csv'")