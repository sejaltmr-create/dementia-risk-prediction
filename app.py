import streamlit as st
import pandas as pd
import joblib

# Load Model and Scaler
model = joblib.load("dementia_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Dementia Risk Prediction System")

st.write("Enter Patient Information")

# Input Fields
Age = st.number_input("Age", 20, 100, 60)
Sex = st.number_input("Sex (Encoded)", 0, 5, 1)
Education_Level = st.number_input("Education Level (Encoded)", 0, 10, 1)
Height_cm = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
Weight_kg = st.number_input("Weight (kg)", 20.0, 200.0, 70.0)
BMI = st.number_input("BMI", 10.0, 60.0, 25.0)
Waist_Circumference_cm = st.number_input("Waist Circumference", 40.0, 200.0, 90.0)

Duration_of_Diabetes_Years = st.number_input("Duration of Diabetes (Years)", 0.0, 50.0, 10.0)

Systolic_BP_mmHg = st.number_input("Systolic BP", 80.0, 250.0, 120.0)
Diastolic_BP_mmHg = st.number_input("Diastolic BP", 40.0, 150.0, 80.0)

Fasting_Glucose_mg_dL = st.number_input("Fasting Glucose", 50.0, 500.0, 100.0)
Postprandial_Glucose_mg_dL = st.number_input("Postprandial Glucose", 50.0, 600.0, 140.0)

HbA1c_percent = st.number_input("HbA1c (%)", 3.0, 20.0, 6.5)
Fasting_Insulin_uIU_mL = st.number_input("Fasting Insulin", 0.0, 100.0, 10.0)
HOMA_IR = st.number_input("HOMA IR", 0.0, 20.0, 2.0)

Total_Cholesterol_mg_dL = st.number_input("Total Cholesterol", 50.0, 500.0, 180.0)
Triglycerides_mg_dL = st.number_input("Triglycerides", 20.0, 1000.0, 150.0)

HDL_mg_dL = st.number_input("HDL", 10.0, 150.0, 50.0)
LDL_mg_dL = st.number_input("LDL", 10.0, 300.0, 100.0)

Serum_Creatinine_mg_dL = st.number_input("Serum Creatinine", 0.1, 10.0, 1.0)
eGFR_mL_min_1_73m2 = st.number_input("eGFR", 1.0, 200.0, 90.0)

Smoking_Status = st.number_input("Smoking Status (Encoded)", 0, 5, 0)
Alcohol_Status = st.number_input("Alcohol Status (Encoded)", 0, 5, 0)
Physical_Activity = st.number_input("Physical Activity (Encoded)", 0, 5, 1)

# Prediction
if st.button("Predict Dementia Risk"):

    data = [[
        Age, Sex, Education_Level, Height_cm, Weight_kg, BMI,
        Waist_Circumference_cm, Duration_of_Diabetes_Years,
        Systolic_BP_mmHg, Diastolic_BP_mmHg,
        Fasting_Glucose_mg_dL, Postprandial_Glucose_mg_dL,
        HbA1c_percent, Fasting_Insulin_uIU_mL, HOMA_IR,
        Total_Cholesterol_mg_dL, Triglycerides_mg_dL,
        HDL_mg_dL, LDL_mg_dL,
        Serum_Creatinine_mg_dL,
        eGFR_mL_min_1_73m2,
        Smoking_Status,
        Alcohol_Status,
        Physical_Activity
    ]]

    input_df = pd.DataFrame(data)

    scaled_data = scaler.transform(input_df)

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1] * 100

    if prediction == 1:
        st.error(f"High Dementia Risk ({probability:.2f}%)")
    else:
        st.success(f"Low Dementia Risk ({100-probability:.2f}%)")
