import joblib
import pandas as pd
import numpy as np

# Load model and encoders
try:
    model = joblib.load('loan_approval_model.joblib')
    encoders = joblib.load('label_encoders.joblib')
    print("✅ Model and encoders loaded.")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# Features expected by the model
FEATURES = [
    'person_age', 'person_gender', 'person_education', 'person_income', 
    'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent', 
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
    'credit_score', 'previous_loan_defaults_on_file'
]

# Sample data
data = {
    'person_age': 30,
    'person_gender': 'female',
    'person_education': 'Master',
    'person_income': 60000,
    'person_emp_exp': 5,
    'person_home_ownership': 'RENT',
    'loan_amnt': 10000,
    'loan_intent': 'EDUCATION',
    'loan_int_rate': 11.5,
    'loan_percent_income': 10000 / 60000,
    'cb_person_cred_hist_length': 3,
    'credit_score': 650,
    'previous_loan_defaults_on_file': 'No'
}

df = pd.DataFrame([data])

# Encode
for col, le in encoders.items():
    if col in df.columns:
        print(f"Encoding {col}...")
        df[col] = le.transform(df[col])

# Reorder
df = df[FEATURES]

# Predict
try:
    prob = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]
    print(f"✅ Prediction successful! Prob: {prob:.4f}, Decision: {prediction}")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
