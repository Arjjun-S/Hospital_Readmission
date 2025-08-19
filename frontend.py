import streamlit as st
import pandas as pd
import joblib
from app import preprocess_input, model   # assumes you structured app.py with these

st.set_page_config(page_title="Hospital Readmission Predictor", layout="centered")

st.title("Hospital Readmission Prediction")
st.write("Enter patient details below to predict the risk of readmission.")

# Example inputs - update based on your dataset
age = st.selectbox("Age Group", ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"])
num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=200, value=40)
num_medications = st.number_input("Number of Medications", min_value=0, max_value=100, value=10)
time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=30, value=5)

if st.button("Predict Readmission"):
    try:
        input_df = pd.DataFrame([{ 
            "age": age,
            "num_lab_procedures": num_lab_procedures,
            "num_medications": num_medications,
            "time_in_hospital": time_in_hospital
        }])

        processed = preprocess_input(input_df)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        if prediction == 1:
            st.error(f"High Risk of Readmission (Probability: {probability:.2f})")
        else:
            st.success(f"Low Risk of Readmission (Probability: {probability:.2f})")
    except Exception as e:
        st.warning(f"Error during prediction: {e}")
