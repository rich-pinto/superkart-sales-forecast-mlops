
import os
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]
HF_SPACE_REPO = os.environ["HF_SPACE_REPO"]

def main():
    login(token=HF_TOKEN)
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{HF_SPACE_REPO}"

    # IMPORTANT: space_sdk must be one of: gradio | docker | static
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    api.upload_folder(folder_path="deployment", repo_id=repo_id, repo_type="space")
    print("✅ Space deployed:", f"https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    main()
