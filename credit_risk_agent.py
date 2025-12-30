import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Load Data
try:
    df = pd.read_csv('loan_data.csv')
    print("Columns in dataset:", df.columns.tolist())
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Define features and target
target = 'loan_status'
features = [
    'person_age', 'person_gender', 'person_education', 'person_income', 
    'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent', 
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
    'credit_score', 'previous_loan_defaults_on_file'
]

# Check if features exist
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    print(f"Warning: Missing columns {missing_cols}")
    # Adjust features or error out
    features = [c for c in features if c in df.columns]

X = df[features]
y = df[target]

# Preprocessing
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# HistGradientBoostingClassifier handles NaNs natively, but OrdinalEncoder needs help with NaNs in categorical if any
# However, for simplicity and robustness:
# We will use OrdinalEncoder for categoricals. 
# Categorical features in HGBC must be encoded as integers (0 to n_categories - 1).

# Pipeline for categorical features
# We treat unknown categories as a new category
cat_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)

# Pipeline for numerical features (Pass through or simple imputation? HGBC handles NaNs, so pass through is fine)
# But strictly speaking, sklearn's OrdinalEncoder might complain if we don't handle it precisely.
# Let's just use the native support of HGBC for categorical features using 'categorical_features' param?
# HGBC requires categorical features to be integer-encoded.

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ],
    verbose_feature_names_out=False
).set_output(transform="pandas")

# Model
# Prompt asks for: learning_rate=0.05, max_depth=10
model = HistGradientBoostingClassifier(
    learning_rate=0.05, 
    max_depth=10, 
    categorical_features='from_dtype', # This will use the categorical dtype from pandas if set, or we can rely on preprocessor output
    random_state=42
)

# Since we use OrdinalEncoder, the columns become float/int. We need to tell HGBC which are categorical?
# Update: Recent sklearn versions: if we pass dataframe with category dtypes, it handles it.
# Alternative: Use the pipeline output. OrdinalEncoder returns numbers. HGBC treats them as continuous unless specified.
# Let's map the categorical columns indices after transformation.
# Actually, simpler approach: Use the Pipeline with the model.

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# We need to specify categorical features for HGBC if we want it to treat them as such.
# But with OrdinalEncoder, they are just integers.
# Let's try fitting without explicit categorical_features mask first inside pipeline, 
# usually it performs well enough, or we can tune it.
# The prompt says "HistGradientBoostingClassifier (Scikit-Learn)", implies default or specific params.
# I will proceed with fitting.

print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
acc = accuracy_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"ROC-AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Prediction Function for the verification / usage
def predict_loan_decision(client_data):
    """
    client_data: dict containing feature values
    """
    # Create DataFrame
    input_df = pd.DataFrame([client_data])
    
    # Validate rules
    if input_df['person_age'].iloc[0] > 100 or input_df['person_age'].iloc[0] < 18:
        return "Erreur: Âge invalide."
    if input_df['person_income'].iloc[0] < 0:
        return "Erreur: Revenu négatif."
        
    # Predict
    prob = pipeline.predict_proba(input_df)[0][1]
    decision = 1 if prob >= 0.5 else 0 # Default threshold, can be tuned
    status = "APPROUVÉ" if decision == 1 else "REFUSÉ"
    
    # Analysis
    analysis = []
    # Logic for analysis (mocked based on heuristics or feature importance)
    # Ideally use SHAP, but for this task simple heuristics:
    if input_df['loan_percent_income'].iloc[0] > 0.4:
        analysis.append("Risque élevé : Ratio dette/revenu trop important (> 40%).")
    else:
        analysis.append("Point fort : Ratio dette/revenu raisonnable.")
        
    if input_df['previous_loan_defaults_on_file'].iloc[0] == 'Yes': # Assuming Yes/No
        analysis.append("Point faible : Historique de défaut de paiement.")
    
    # Return formatted info
    return {
        "Statut": status,
        "Confiance": f"{prob:.1%}" if status == "APPROUVÉ" else f"{(1-prob):.1%}", # Confidence in the decision
        "Analyse": analysis
    }

# Example Usage
if __name__ == "__main__":
    import sys
    # If args provided, could parse them, but for now just run training and print one example
    example_client = {
        'person_age': 25, 
        'person_gender': 'male',
        'person_education': 'Bachelor',
        'person_income': 50000, 
        'person_emp_exp': 2,
        'person_home_ownership': 'RENT', 
        'loan_amnt': 8000, 
        'loan_intent': 'EDUCATION', 
        'loan_int_rate': 11.0, 
        'loan_percent_income': 0.16, 
        'cb_person_cred_hist_length': 3,
        'credit_score': 650,
        'previous_loan_defaults_on_file': 'No'
    }
    print("\nExample Prediction:")
    print(predict_loan_decision(example_client))
