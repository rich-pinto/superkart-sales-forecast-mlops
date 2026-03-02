import os, json
import joblib
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_DATASET_ID = os.environ["HF_DATASET_ID"]
HF_MODEL_ID = os.environ["HF_MODEL_ID"]
TARGET = "Product_Store_Sales_Total"

def main():
    ds = load_dataset(HF_DATASET_ID, data_files={"train": "processed/train.csv", "test": "processed/test.csv"})
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].values
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].values

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    models = {
        "DecisionTree": (DecisionTreeRegressor(random_state=42), {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 10, 30]
        }),
        "RandomForest": (RandomForestRegressor(random_state=42, n_jobs=-1), {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 10]
        }),
        "GradientBoosting": (GradientBoostingRegressor(random_state=42), {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3]
        }),
        "AdaBoost": (AdaBoostRegressor(random_state=42), {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1, 0.2]
        }),
    }

    results = []
    best_pipe = None
    best_rmse = None
    best_row = None

    for name, (estimator, grid) in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        gs = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)
        gs.fit(X_train, y_train)

        pred = gs.best_estimator_.predict(X_test)
        rmse = mean_squared_error(y_test, pred, squared=False)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        row = {
            "model": name,
            "best_params": gs.best_params_,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }
        results.append(row)

        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_pipe = gs.best_estimator_
            best_row = row

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_pipe, "artifacts/best_model.joblib")
    with open("artifacts/metrics.json", "w") as f:
        json.dump({"best": best_row, "all_results": results}, f, indent=2)

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_MODEL_ID, repo_type="model", exist_ok=True)
    api.upload_file("artifacts/best_model.joblib", "best_model.joblib", repo_id=HF_MODEL_ID, repo_type="model")
    api.upload_file("artifacts/metrics.json", "metrics.json", repo_id=HF_MODEL_ID, repo_type="model")

    print("✅ Uploaded best model to HF Model Hub:", HF_MODEL_ID)
    print("✅ Best:", best_row)

if __name__ == "__main__":
    main()
