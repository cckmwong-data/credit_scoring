# ------------------------------- Import Libraries -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# ------------------------------- Load model and scaler -------------------------------

with open("credit_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("credit_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("credit_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

# ------------------------------- UI -------------------------------
st.title("💳 Loan Approval Prediction")

with st.expander("ℹ️ About This App"):
    st.markdown("""
This application demonstrates a Loan Approval Prediction model that estimates the probability of an applicant successfully repaying a loan. By entering key applicant details such as **age**, **monthly income**, **credit history score**, **loan amount**, and **employment status**, the model predicts both **default** and **repayment probabilities**.

The goal of the model is to assist lenders in making more data-driven and consistent lending decisions by quantifying the likelihood of default. To maintain prudent risk management, a custom decision threshold (lower than 35%) is applied to account for the lender’s risk tolerance.

In addition to the prediction result, the app provides a visual explanation of what factors most influenced the decision. Blue bars indicate factors that increased the likelihood of loan repayment, while red bars show factors that contributed to default risk. This transparency helps both lenders and applicants understand how individual variables impact the final outcome.
""")

st.write("Enter applicant details to predict the probability of loan repayment.")

# User inputs
age = st.number_input("Age", min_value = 18, value = 30, max_value = 100 ,step = 1)
income = st.number_input("Monthly Income (£)", min_value=0, value = 2000, step=100)
credit_history = st.slider("Credit History Score (0 = worst, 10 = best)", 0, 10, 5, 1)
balance = st.number_input("Loan Amount (£)", min_value=0, value = 200, step=100)
employment_status = st.radio("Employment Status", ("Employed", "Self-Employed", "Unemployed"))

# Convert employment status to dummies
if employment_status == "Employed":
    employed, self_employed, unemployed = 1, 0, 0
elif employment_status == "Self-Employed":
    employed, self_employed, unemployed = 0, 1, 0
else:
    employed, self_employed, unemployed = 0, 0, 1

# Define Debt-to-Income ratio
DTI = balance/ income

# Create dataframe for input
input_data = pd.DataFrame([[age, DTI, credit_history, employed, self_employed, unemployed]], 
columns=['Age', 'DTI', 'Credit_History','Employed', 'Self-Employed', 'Unemployed'])

# Scale inputs
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prob_default = model.predict_proba(input_data_scaled)[0][1]   # class 1 = default
    prob_repay   = model.predict_proba(input_data_scaled)[0][0]   # class 0 = non-default

    # Results of the loan application
    apply_result = "APPROVED ✅." if prob_default <= 0.35 else "DECLINED ❌."
    # Reasons
    apply_reason = "Given that the applicant’s default probability falls below this threshold, the loan application is " if prob_default <= 0.35 else "Given that the applicant’s default probability surpasses this threshold, the loan application is "

    st.subheader("Prediction Result")
    st.write(f"To maintain prudent risk management, the lender applies a reduced decision threshold of 35% rather than the standard 50%. {apply_reason}**{apply_result}**")
    st.write(f"Default Probability: **{prob_default*100:.2f}%**")
    st.write(f"Repayment Probability: **{prob_repay*100:.2f}%**")

    # Get SHAP values
    shap_values = explainer(input_data_scaled)
         
    st.subheader("🔎 What drove the decision?")

    # Create a waterfall plot and render in Streamlit
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    st.write("Blue bars push the prediction higher towards repayment, whereas red bars push the prediction lower towards default. The contributions add up to the final prediction f(x). E[f(X)] is the model’s average prediction.")
    st.write("Remarks: DTI (debt-to-income ratio) = Total Monthly Debt Payments/ Monthly Income, is to assess a borrower’s ability to manage monthly payments and repay debts.")
