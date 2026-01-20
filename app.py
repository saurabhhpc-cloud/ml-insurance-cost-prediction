import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("rf_model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="medical Insurance Prediction", layout="centered")

st.title("🏥 medical Insurance Cost Prediction")
st.write("Enter patient details to predict insurance charges")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)

sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encode inputs
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# Create input dataframe
input_data = pd.DataFrame([[
    age, bmi, children, sex_val, smoker_val,
    region_northwest, region_southeast, region_southwest
]], columns=features)

# Predict
if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Insurance Cost: ₹ {prediction:,.2f}")
