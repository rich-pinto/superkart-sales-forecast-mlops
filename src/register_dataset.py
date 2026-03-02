
import os
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]
HF_DATASET_REPO = os.environ["HF_DATASET_REPO"]

RAW_LOCAL = "data/SuperKart_raw.csv"   # ensure this exists in repo
HF_PATH_IN_REPO = "data/SuperKart_raw.csv"

def main():
    login(token=HF_TOKEN)
    api = HfApi()

    repo_id = f"{HF_USERNAME}/{HF_DATASET_REPO}"
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    if not os.path.exists(RAW_LOCAL):
        raise FileNotFoundError(f"Missing raw dataset at {RAW_LOCAL}. Commit it OR download it in workflow.")

    api.upload_file(
        path_or_fileobj=RAW_LOCAL,
        path_in_repo=HF_PATH_IN_REPO,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print("✅ Uploaded raw dataset:", f"https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
