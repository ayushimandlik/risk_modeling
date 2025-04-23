# app.py

import numpy as np

import streamlit as st
import pandas as pd
#import joblib  # for loading your trained model


# Load the model
#model = joblib.load("your_model.pkl")  # replace with your actual model file path

from keras.saving import load_model

model = load_model('weights.08-0.26.keras')  # or .h5 depending on how you saved it


st.title("Loan Default Prediction App")
st.write("This app predicts the likelihood of a loan default based on customer inputs.")

# Sidebar inputs
st.sidebar.header("Input Customer Data")

def user_input():
    loan_amnt = st.sidebar.slider("Loan Amount", 500, 40000, 15000)
    income = st.sidebar.slider("Annual Income", 10000, 200000, 60000)
    credit_utilisation = st.sidebar.slider("Credit Utilisation", 0.0, 1.5, 0.5)
    term_60_months = st.sidebar.radio("Term is 60 Months?", ["Yes", "No"])

    # Add more features as needed
    data = {
        'loan_amnt': loan_amnt,
        'annual_inc': income,
        'credit_utilisation': credit_utilisation,
        'term_60_months': 1 if term_60_months == "Yes" else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input()

# Show user input
st.subheader("User Input Features")
st.write(input_df)

# Prediction
input_df = np.array([-1.3627165e-02,  1.9019926e-01, -4.0888897e-01,  1.7524152e-01,
        5.9214455e-01, -3.5455707e-02, -2.5520524e-01, -3.9257029e-01,
       -2.0933819e-01,  4.0760717e-01, -7.9202181e-01,  1.2820402e-01,
       -3.5855809e-01, -1.2526691e+00, -2.0501226e-01, -4.0072712e-01,
        1.0661038e+00,  1.9700000e+03,  1.0000000e+00,  1.9700000e+03,
        1.0000000e+00,  0.0000000e+00,  1.0000000e+00,  0.0000000e+00,
        1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
        0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  0.0000000e+00,
        0.0000000e+00,  0.0000000e+00,  1.0000000e+00,  0.0000000e+00,
        1.0000000e+00,  1.0000000e+00,  0.0000000e+00,  0.0000000e+00,
        0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  0.0000000e+00,
        0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00])
st.write(input_df.shape)
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write("Default" if prediction[0] == 1 else "No Default")

st.subheader("Prediction Probability")
st.write(f"Probability of Default: {prediction_proba[0][1]:.2f}")
