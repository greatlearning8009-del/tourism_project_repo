# ==========================
# Import Libraries
# ==========================
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# ==========================
# Hugging Face Authentication
# ==========================
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is not set.")

api = HfApi(token=HF_TOKEN)

# ==========================
# Load Dataset
# ==========================
DATASET_PATH = "hf://datasets/SRGL/machine-failure-prediction/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully.")

# ==========================
# Drop Unnecessary Columns
# ==========================
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# ==========================
# Encode Categorical Columns
# ==========================
cat_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

label_encoders = {}

for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    else:
        print(f"Warning: Column '{col}' not found. Skipping...")

# ==========================
# Define Features and Target
# ==========================
target_col = "ProdTaken"

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

X = df.drop(columns=[target_col])
y = df[target_col]

# ==========================
# Train-Test Split
# ==========================
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("✅ Train-Test split completed.")

# ==========================
# Save Files
# ==========================
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("✅ CSV files saved locally.")

# ==========================
# Upload to Hugging Face Dataset
# ==========================
files = [
    "Xtrain.csv",
    "Xtest.csv",
    "ytrain.csv",
    "ytest.csv",
]

for file_path in files:
    print(f"Uploading {file_path}...")

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="SRGL/machine-failure-prediction",
        repo_type="dataset",
    )

print("✅ All files uploaded successfully.")
