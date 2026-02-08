import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="dararaje/Tourism_Package_Prediction", filename="best_model_tourism_pred_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer taking a tourism package.
Please enter the data below to get a prediction.
""")

# User input
passport = st.selectbox("Passport", ["0", "1"])
MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
CityTier = st.selectbox("CityTier", ["1", "2", "3"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'MaritalStatus': MaritalStatus,
    'CityTier': CityTier,
    'Designation': Designation,
    'Passport': passport
}])


if st.button("Predict Tourism Package"):
    prediction = model.predict(input_data)[0]
    result = "Not Taking Tourism Package" if prediction == 1 else "Taking Tourism Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
