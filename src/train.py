
import os, json
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, login, hf_hub_download
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]
HF_DATASET_REPO = os.environ["HF_DATASET_REPO"]

TARGET = "Product_Store_Sales_Total"

def main():
    login(token=HF_TOKEN)
    dataset_repo_id = f"{HF_USERNAME}/{HF_DATASET_REPO}"

    # Download raw dataset from HF
    raw_path = hf_hub_download(repo_id=dataset_repo_id, repo_type="dataset", filename="data/SuperKart_raw.csv")
    df = pd.read_csv(raw_path)

    # Basic cleaning
    df = df.dropna(subset=[TARGET]).copy()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Faster than GridSearchCV
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10],
    }

    pipe = Pipeline([("preprocess", preprocess), ("model", model)])

    rs = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=6,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    rs.fit(X_train, y_train)

    best_model = rs.best_estimator_
    preds = best_model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/best_model.joblib")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "best_params": rs.best_params_,
    }
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save train/test locally (rubric)
    os.makedirs("data", exist_ok=True)
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    # Upload train/test back to HF Dataset Hub (rubric)

    api = HfApi()
    api.upload_folder(folder_path="data", path_in_repo="data", repo_id=dataset_repo_id, repo_type="dataset")
    print("✅ Uploaded train/test to HF dataset repo in a single commit.")

    print("✅ Training complete | RMSE:", rmse, "| MAE:", mae, "| R2:", r2)

if __name__ == "__main__":
    main()
