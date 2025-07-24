import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Title
st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a telecom customer is likely to churn using logistic regression.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("customerChurn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

# Encode categorical variables
def preprocess_data(df):
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]
    scaler = StandardScaler()
    X[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(X[['MonthlyCharges', 'TotalCharges']])
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns

(X_train, X_test, y_train, y_test), feature_names = preprocess_data(df)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Sidebar - User Input
st.sidebar.header("📥 Input Customer Details")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    Partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
    Dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
    TotalCharges = st.sidebar.slider("Total Charges", 0.0, 10000.0, 2500.0)

    # Create a DataFrame with one row for model prediction
    data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'gender_Male': 1 if gender == 'Male' else 0,
        'SeniorCitizen': SeniorCitizen,
        'Partner_Yes': 1 if Partner == 'Yes' else 0,
        'Dependents_Yes': 1 if Dependents == 'Yes' else 0,
        'PhoneService_Yes': 1 if PhoneService == 'Yes' else 0,
        'MultipleLines_No phone service': 1 if MultipleLines == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if MultipleLines == 'Yes' else 0,
        'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
        'InternetService_No': 1 if InternetService == 'No' else 0,
        'OnlineSecurity_No': 1 if OnlineSecurity == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if OnlineSecurity == 'No internet service' else 0,
        'OnlineBackup_No': 1 if OnlineBackup == 'No' else 0,
        'OnlineBackup_No internet service': 1 if OnlineBackup == 'No internet service' else 0,
        'DeviceProtection_No': 1 if DeviceProtection == 'No' else 0,
        'DeviceProtection_No internet service': 1 if DeviceProtection == 'No internet service' else 0,
        'TechSupport_No': 1 if TechSupport == 'No' else 0,
        'TechSupport_No internet service': 1 if TechSupport == 'No internet service' else 0,
        'StreamingTV_No': 1 if StreamingTV == 'No' else 0,
        'StreamingTV_No internet service': 1 if StreamingTV == 'No internet service' else 0,
        'StreamingMovies_No': 1 if StreamingMovies == 'No' else 0,
        'StreamingMovies_No internet service': 1 if StreamingMovies == 'No internet service' else 0,
        'Contract_One year': 1 if Contract == 'One year' else 0,
        'Contract_Two year': 1 if Contract == 'Two year' else 0,
        'PaperlessBilling_Yes': 1 if PaperlessBilling == 'Yes' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0,
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Match feature columns to training set
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0  # add missing columns as 0

input_df = input_df[feature_names]  # reorder columns

# Make prediction
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Show result
st.subheader("🔍 Prediction Result:")
if prediction == 1:
    st.error(f"⚠️ This customer is **likely to churn**. (Risk score: {probability:.2f})")
else:
    st.success(f"✅ This customer is **likely to stay**. (Risk score: {probability:.2f})")

# Show accuracy (optional)
if st.checkbox("Show model accuracy on test data"):
    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    st.write(f"Model Accuracy: **{acc:.2%}**")

