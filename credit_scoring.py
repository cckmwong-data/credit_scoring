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
st.title("💳 Loan Approval Prediction")
st.write("Enter applicant details to predict the probability of loan repayment.")

# User inputs
age = st.number_input("Age", min_value = 18, value = 30, max_value = 100 ,step = 1)
income = st.number_input("Monthly Income (£)", min_value=0, value = 2000, step=100)
credit_history = st.slider("CrediAt History Score (0 = worst, 10 = best)", 0, 10, 5, 1)
balance = st.number_input("Loan Amount (£)", min_value=0, value = 200, step=100)
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
input_data = pd.DataFrame([[age, DTI, credit_history, employed, self_employed, unemployed]], 
columns=['Age', 'DTI', 'Credit_History','Employed', 'Self-Employed', 'Unemployed'])

# Scale input
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prob_default = model.predict_proba(input_data_scaled)[0][1]   # class 1 = default
    prob_repay   = model.predict_proba(input_data_scaled)[0][0]   # class 0 = non-default

    apply_result = "APPROVED ✅." if prob_default <= 0.35 else "DECLINED ❌."
    apply_reason = "As the default probability falls below this thresold, the loan application is " if prob_default <= 0.35 else "As the repayment probability is above this thresold, the loan application is "

    st.subheader("Prediction Result")
    st.write(f"The lender adopts a prudent approach by setting a lower decision threshold of 35% compared to the standard 50%. {apply_reason}**{apply_result}**")
    st.write(f"Default Probability: **{prob_default*100:.2f}%**")
    st.write(f"Repayment Probability: **{prob_repay*100:.2f}%**")


    # Get SHAP values
    shap_values = explainer(input_data_scaled)
         
    st.subheader("🔎 What drove the decision?")

    # Create a waterfall plot and render in Streamlit
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    st.write("Remarks: Blue bars push the prediction higher towards repayment, whereas red bars push the prediction lower towards default. The contributions add up to the final prediction f(x). The gray value on the left (E[f(X)]) is the model’s average prediction.")
    st.write("DTI (debt-to-income ratio) is to assess a borrower’s ability to manage monthly payments and repay debts. DTI = Total Monthly Debt Payments/ Monthly Income")
