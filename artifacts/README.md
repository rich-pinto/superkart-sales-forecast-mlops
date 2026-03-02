---
language: en
license: mit
tags:
- tabular-regression
- sales-forecasting
- mlops
library_name: scikit-learn
pipeline_tag: tabular-regression
---

# SuperKart Sales Forecast Model

## Target
Predict **Product_Store_Sales_Total**

## Best Model (by RMSE)
- Model: **RandomForest**
- RMSE: **280.0752**
- MAE: **83.1635**
- R²: **0.9313**

## Dataset
This model is trained on the SuperKart dataset uploaded to Hugging Face Datasets as part of the MLOps pipeline.

## Preprocessing
- Numeric features: missing values imputed
- Categorical features: one-hot encoded
- Final artifact: a single scikit-learn **Pipeline** (`preprocess` + `model`)

## How to Use (Inference)
Load the model artifact (`best_model.joblib`) and call `.predict()` on a dataframe with the same feature columns used during training.

## Notes
This repository is created as part of an automated CI/CD pipeline for sales forecasting.
