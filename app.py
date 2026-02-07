import streamlit as st
import pandas as pd
import pickle

# Load trained ML pipeline
with open("HealthExpense.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("Medical Expense Prediction in India")

st.markdown("Enter the details below to predict medical expenses:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker?", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "northeast", "southwest", "southeast"])

# Prediction button
if st.button("Predict"):
    # Create input dataframe for model
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])
    
    # Make prediction
    prediction = model.predict(input_df)[0]

    # Display result
    st.success(f"Predicted Medical Expense: ₹{round(prediction, 2)}")

    # Optional: Show input summary
    st.write("**Your Input:**")
    st.write(input_df)
