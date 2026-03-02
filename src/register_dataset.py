import os
from huggingface_hub import HfApi, login, hf_hub_download

RAW_LOCAL_PATH = 'data/SuperKart_raw.csv'
HF_DATASET_FILE_PATH = 'data/SuperKart_raw.csv'

def main():
    HF_TOKEN = os.environ['HF_TOKEN']
    HF_USERNAME = os.environ['HF_USERNAME']
    HF_DATASET_REPO = os.environ['HF_DATASET_REPO']

    login(token=HF_TOKEN)
    api = HfApi()
    dataset_id = f"{HF_USERNAME}/{HF_DATASET_REPO}"

    # Create dataset repo if not exists
    api.create_repo(repo_id=dataset_id, repo_type='dataset', exist_ok=True)

    os.makedirs('data', exist_ok=True)

    # If dataset missing locally (GitHub runner case), download from HF
    if not os.path.exists(RAW_LOCAL_PATH):
        print(f'Dataset not found locally. Attempting download from HF: {dataset_id}')
        try:
            downloaded = hf_hub_download(
                repo_id=dataset_id,
                repo_type='dataset',
                filename=HF_DATASET_FILE_PATH
            )
            with open(downloaded, 'rb') as src, open(RAW_LOCAL_PATH, 'wb') as dst:
                dst.write(src.read())
            print('Downloaded dataset to local data folder.')
        except Exception as e:
            raise FileNotFoundError(
                f'Raw dataset not found locally AND not present in HF dataset repo. Error: {e}'
            )

    # Upload raw dataset to HF (safe to re-upload)
    api.upload_file(
        path_or_fileobj=RAW_LOCAL_PATH,
        path_in_repo=HF_DATASET_FILE_PATH,
        repo_id=dataset_id,
        repo_type='dataset'
    )

    print('Raw dataset registered at:')
    print(f'https://huggingface.co/datasets/{dataset_id}')

if __name__ == '__main__':
    main()
