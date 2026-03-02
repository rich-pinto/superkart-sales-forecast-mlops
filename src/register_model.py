import os
from huggingface_hub import HfApi, login

def main():
    HF_TOKEN = os.environ['HF_TOKEN']
    HF_USERNAME = os.environ['HF_USERNAME']
    HF_MODEL_REPO = os.environ['HF_MODEL_REPO']

    login(token=HF_TOKEN)
    api = HfApi()

    repo_id = f"{HF_USERNAME}/{HF_MODEL_REPO}"
    api.create_repo(repo_id=repo_id, repo_type='model', exist_ok=True)

    # Upload ALL model artifacts in one commit (avoids API positional args + reduces 412 conflicts)
    # Expected local files produced by train.py:
    #   artifacts/best_model.joblib
    #   artifacts/metrics.json
    os.makedirs('artifacts', exist_ok=True)
    if not os.path.exists('artifacts/best_model.joblib'):
        raise FileNotFoundError('Missing artifacts/best_model.joblib. Ensure train.py ran and saved artifacts.')

    # Minimal model card with YAML to avoid metadata warnings
    card = (
        '---\n'
        'license: mit\n'
        'tags:\n'
        '- tabular-regression\n'
        '- mlops\n'
        '---\n\n'
        '# SuperKart Sales Forecast Model\n\n'
        'Predicts Product_Store_Sales_Total from product + store attributes.\n'
    )
    with open('artifacts/README.md', 'w') as f:
        f.write(card)

    # Single commit upload of artifacts folder
    api.upload_folder(
        folder_path='artifacts',
        path_in_repo='.',
        repo_id=repo_id,
        repo_type='model',
    )

    print('✅ Model registered:', f'https://huggingface.co/{repo_id}')

if __name__ == '__main__':
    main()
