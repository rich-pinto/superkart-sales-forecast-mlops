import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")
st.write("Predict **Product_Store_Sales_Total** using product + store attributes.")

MODEL_ID = "rpinto123/superkart-sales-forecast-model"

@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=MODEL_ID, filename="best_model.joblib")
    return joblib.load(path)

model = load_model()

st.subheader("Input Features")

FEATURE_COLS = ["Product_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_Type", "Product_MRP", "Store_Id", "Store_Establishment_Year", "Store_Size", "Store_Location_City_Type", "Store_Type"]
NUM_COLS = set(["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"])

inputs = {}
for col in FEATURE_COLS:
    if col in NUM_COLS:
        inputs[col] = st.number_input(col, value=0.0)
    else:
        inputs[col] = st.text_input(col, value="Unknown")

df_in = pd.DataFrame([inputs])

if st.button("Predict"):
    pred = model.predict(df_in)[0]
    st.success(f"Predicted Sales Total: {pred:,.2f}")
