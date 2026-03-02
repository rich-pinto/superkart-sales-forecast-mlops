
import os, json
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]
HF_MODEL_REPO = os.environ["HF_MODEL_REPO"]

def main():
    login(token=HF_TOKEN)
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{HF_MODEL_REPO}"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Upload model + metrics
    api.upload_file("artifacts/best_model.joblib", "best_model.joblib", repo_id=repo_id, repo_type="model")
    api.upload_file("artifacts/metrics.json", "metrics.json", repo_id=repo_id, repo_type="model")

    # Build model card safely (NO triple quotes)
    card = (
        "---\n"
        "license: mit\n"
        "tags:\n"
        "- tabular-regression\n"
        "- mlops\n"
        "---\n\n"
        "# SuperKart Sales Forecast Model\n\n"
        "This model predicts Product_Store_Sales_Total from product + store attributes.\n"
    )

    with open("artifacts/README.md", "w") as f:
        f.write(card)

    api.upload_file("artifacts/README.md", "README.md", repo_id=repo_id, repo_type="model")

    print("Model registered:", f"https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
