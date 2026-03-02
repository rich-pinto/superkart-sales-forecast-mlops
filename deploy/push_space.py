import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_SPACE_ID = os.environ["HF_SPACE_ID"]  # e.g. username/space-name

api = HfApi(token=HF_TOKEN)

# Create space repo if it doesn't exist (Docker Space)
api.create_repo(
    repo_id=HF_SPACE_ID,
    repo_type="space",
    exist_ok=True,
    space_sdk="docker"
)

# Upload ALL files in current folder (Dockerfile + app.py + requirements.txt)
api.upload_folder(
    folder_path=".",
    repo_id=HF_SPACE_ID,
    repo_type="space"
)

print("✅ Deployed/Updated HF Space:", HF_SPACE_ID)
print("🔗 Space URL: https://huggingface.co/spaces/" + HF_SPACE_ID)
