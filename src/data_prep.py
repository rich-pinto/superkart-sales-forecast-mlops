import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_DATASET_ID = os.environ["HF_DATASET_ID"]
TARGET = "Product_Store_Sales_Total"

def main():
    ds = load_dataset(HF_DATASET_ID, data_files={"raw": "data/SuperKart_raw.csv"})
    df = ds["raw"].to_pandas()

    df = df.drop_duplicates()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs("data", exist_ok=True)
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    api = HfApi(token=HF_TOKEN)
    api.upload_file(train_path, "processed/train.csv", repo_id=HF_DATASET_ID, repo_type="dataset")
    api.upload_file(test_path, "processed/test.csv", repo_id=HF_DATASET_ID, repo_type="dataset")

    print("✅ Uploaded processed train/test to HF dataset:", HF_DATASET_ID)

if __name__ == "__main__":
    main()
