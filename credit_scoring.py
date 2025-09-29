import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Load model and scaler
# -------------------------------
# You can save your trained model & scaler as pickle files in your notebook first
# Example in notebook:
#   with open("model.pkl", "wb") as f: pickle.dump(model, f)
#   with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)

with open("credit_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("credit_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("credit_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("💳 Credit Scoring Prediction App")
st.write("Enter applicant details to predict the probability of loan repayment.")

# User inputs
income = st.number_input("Monthly Income (£)", min_value=0, value=2000, step=100)
credit_history = st.slider("Credit History Score (0 = worst, 10 = best)", 0, 10, 5, 1)
balance = st.number_input("Loan Amount (£)", min_value=0, value=500, step=100)
employment_status = st.radio("Employment Status", ("Employed", "Self-Employed", "Unemployed"))

# Convert employment status to dummies
if employment_status == "Employed":
    employed, self_employed, unemployed = 1, 0, 0
elif employment_status == "Self-Employed":
    employed, self_employed, unemployed = 0, 1, 0
else:
    employed, self_employed, unemployed = 0, 0, 1

DTI = balance/ income

# Create dataframe for input
input_data = pd.DataFrame([[DTI, credit_history, employed, self_employed, unemployed]], 
columns=['DTI', 'Credit_History','Employed', 'Self-Employed', 'Unemployed'])

# Scale input
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prob_repay = model.predict_proba(input_data_scaled)[0][1]  # class 1 = repay
    prob_default = model.predict_proba(input_data_scaled)[0][0]

    prediction = "YES ✅ (Likely to Repay)" if prob_repay >= 0.5 else "NO ❌ (Likely to Default)"

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Repayment Probability:** {prob_repay*100:.2f}%")
    st.write(f"**Default Probability:** {prob_default*100:.2f}%")

    # Get SHAP values
    shap_values = explainer(input_data_scaled)

    st.subheader("🔎 What drove the decision?")
    st.write("Red bars push the prediction higher towards repayment, whereas blue bars push the prediction lower towards default. The contributions add up to the final prediction shown on the right (f(x)).")
    st.write("The gray value on the left (E[f(X)]) is the model’s average prediction (baseline).")

    # Create a waterfall plot and render in Streamlit
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
