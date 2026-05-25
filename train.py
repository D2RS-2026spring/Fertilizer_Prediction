import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def map_at_3(y_true, y_proba, classes):
    top3_idx = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]
    top3_labels = classes[top3_idx]

    scores = []
    for true_label, preds in zip(y_true, top3_labels):
        score = 0.0
        for rank, pred in enumerate(preds, start=1):
            if pred == true_label:
                score = 1.0 / rank
                break
        scores.append(score)

    return float(np.mean(scores))


def build_model(categorical_cols, numeric_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_onehot_encoder(), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    model = LogisticRegression(
        max_iter=300,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--sample_size", type=int, default=50000)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    model_dir = out_dir / "model"

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError("未找到 data/train.csv")
    if not test_path.exists():
        raise FileNotFoundError("未找到 data/test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "Fertilizer Name"

    if target_col not in train_df.columns:
        raise ValueError("训练集缺少 Fertilizer Name 目标列")

    print("训练集规模：", train_df.shape)
    print("测试集规模：", test_df.shape)

    if args.sample_size > 0 and args.sample_size < len(train_df):
        train_df, _ = train_test_split(
            train_df,
            train_size=args.sample_size,
            stratify=train_df[target_col],
            random_state=42,
        )
        train_df = train_df.reset_index(drop=True)
        print("使用抽样训练集规模：", train_df.shape)

    feature_cols = [c for c in train_df.columns if c not in ["id", target_col]]

    categorical_cols = [c for c in ["Soil Type", "Crop Type"] if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X = train_df[feature_cols]
    y = train_df[target_col].astype(str).values

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n========== Fold {fold} ==========")

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        model = build_model(categorical_cols, numeric_cols)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_valid)
        classes = model.named_steps["classifier"].classes_

        score = map_at_3(y_valid, y_proba, classes)
        fold_scores.append(score)

        print(f"Fold {fold} MAP@3: {score:.5f}")

    mean_score = float(np.mean(fold_scores))
    print("\n========== 交叉验证结果 ==========")
    print("各折 MAP@3：", fold_scores)
    print(f"平均 MAP@3：{mean_score:.5f}")

    final_model = build_model(categorical_cols, numeric_cols)
    final_model.fit(X, y)

    joblib.dump(final_model, model_dir / "fertilizer_model.joblib")

    test_X = test_df[feature_cols]
    test_proba = final_model.predict_proba(test_X)
    classes = final_model.named_steps["classifier"].classes_

    top3_idx = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]
    top3_labels = classes[top3_idx]
    pred_strings = [" ".join(row) for row in top3_labels]

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "Fertilizer Name": pred_strings,
        }
    )

    submission_path = out_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    metrics_path = out_dir / "model_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Fertilizer Prediction Model Results\n")
        f.write("=================================\n")
        f.write(f"Training data shape: {train_df.shape}\n")
        f.write(f"Test data shape: {test_df.shape}\n")
        f.write(f"Features: {feature_cols}\n")
        f.write(f"Categorical features: {categorical_cols}\n")
        f.write(f"Numeric features: {numeric_cols}\n")
        f.write(f"Fold MAP@3 scores: {fold_scores}\n")
        f.write(f"Mean MAP@3: {mean_score:.5f}\n")
        f.write("Model: LogisticRegression with OneHotEncoder and StandardScaler\n")

    print("\n训练完成。")
    print(f"模型保存到：{model_dir / 'fertilizer_model.joblib'}")
    print(f"预测结果保存到：{submission_path}")
    print(f"指标结果保存到：{metrics_path}")


if __name__ == "__main__":
    main()
