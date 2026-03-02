---
title: SuperKart Sales Forecast
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SuperKart Sales Forecast (Streamlit)

This Space loads the trained `best_model.joblib` from the Hugging Face **Model Hub**
and predicts `Product_Store_Sales_Total` from product + store attributes.

## How it works
- Streamlit UI collects input features
- Builds a single-row dataframe
- Runs `model.predict(df)`
