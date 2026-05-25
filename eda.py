import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

train_path = DATA_DIR / "train.csv"

if not train_path.exists():
    raise FileNotFoundError("未找到 data/train.csv，请先把数据文件放入 data 文件夹。")

df = pd.read_csv(train_path)

print("========== 数据基本信息 ==========")
print("数据规模：", df.shape)
print("字段列表：")
print(df.columns.tolist())

print("\n========== 缺失值统计 ==========")
print(df.isnull().sum())

print("\n========== 数值字段描述统计 ==========")
print(df.describe())

OUT_DIR.mkdir(parents=True, exist_ok=True)
df.describe(include="all").to_csv(OUT_DIR / "data_summary.csv", encoding="utf-8-sig")

target_col = "Fertilizer Name"

if target_col in df.columns:
    print("\n========== 肥料类别分布 ==========")
    print(df[target_col].value_counts())

    plt.figure(figsize=(10, 6))
    df[target_col].value_counts().plot(kind="bar")
    plt.title("Fertilizer Class Distribution")
    plt.xlabel("Fertilizer Name")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fertilizer_distribution.png", dpi=300)
    plt.close()

if "Soil Type" in df.columns:
    plt.figure(figsize=(8, 5))
    df["Soil Type"].value_counts().plot(kind="bar")
    plt.title("Soil Type Distribution")
    plt.xlabel("Soil Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "soil_type_distribution.png", dpi=300)
    plt.close()

if "Crop Type" in df.columns:
    plt.figure(figsize=(10, 6))
    df["Crop Type"].value_counts().plot(kind="bar")
    plt.title("Crop Type Distribution")
    plt.xlabel("Crop Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "crop_type_distribution.png", dpi=300)
    plt.close()

numeric_cols = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
]

for col in numeric_cols:
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        df[col].hist(bins=30)
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{col.lower()}_distribution.png", dpi=300)
        plt.close()

print("\nEDA 分析完成。")
print("统计结果保存到 outputs/data_summary.csv")
print("图表保存到 outputs/figures 文件夹。")
