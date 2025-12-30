import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- Feature Configuration ---
FEATURES = [
    'person_age', 'person_gender', 'person_education', 'person_income', 
    'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent', 
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
    'credit_score', 'previous_loan_defaults_on_file'
]

import joblib
import sys

# --- UI Layout & Styling ---
st.set_page_config(page_title="Credit Risk AI", page_icon="💳", layout="wide")
st.write("### 🚀 Initialisation de l'application...") # Debugging point

# Custom CSS for Modern "Premium" Look
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1 {
        color: #1E3A8A; /* Dark Blue */
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #2563EB;
        font-weight: bold;
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #2563EB 0%, #1E40AF 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        background: linear-gradient(90deg, #1E40AF 0%, #1E3A8A 100%);
        color: white;
    }

    /* Cards/Containers */
    .st-emotion-cache-1y4p8pa {
        padding: 2rem;
        border-radius: 15px;
        background-color: #F3F4F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Error boxes */
    .stSuccess, .stError {
        border-radius: 10px;
        padding: 1rem;
    }
    
</style>
""", unsafe_allow_html=True)

# Debugging Info (Sidebar)
with st.sidebar.expander("🛠️ Debug Info (Environment)", expanded=True):
    st.write(f"Python Version: {sys.version}")
    st.write(f"Joblib Version: {joblib.__version__}")
    try:
        import xgboost
        st.write(f"XGBoost Version: {xgboost.__version__}")
    except ImportError:
        st.write("XGBoost NOT FOUND")
    try:
        import sklearn
        st.write(f"Scikit-Learn Version: {sklearn.__version__}")
    except ImportError:
        st.write("Scikit-Learn NOT FOUND")

# --- Model Logic (Cached) ---
@st.cache_resource
def load_and_train_model():
    # Load Pre-trained Model and Encoders
    try:
        model = joblib.load('loan_approval_model.joblib')
        encoders = joblib.load('label_encoders.joblib')
        return model, encoders
    except Exception as e:
        st.error(f"Erreur de chargement du modèle ou des encodeurs: {e}")
        return None, None

# --- UI Layout & Styling ---
# Header
col_head1, col_head2, col_head3 = st.columns([1, 6, 1])
with col_head2:
    st.title("💳 Credit Risk AI Assistant")
    st.markdown("<p style='text-align: center; color: #6B7280; font-size: 1.1rem;'>Analysez les demandes de prêt avec la puissance du Machine Learning</p>", unsafe_allow_html=True)
    st.markdown("---")

# Initialize Model
pipeline, encoders = load_and_train_model()

if pipeline:
    # Main Container
    with st.container():
        # Layout: 2 Columns for Inputs
        col_left, col_right = st.columns(2, gap="large")
        
        with col_left:
            st.markdown("### 👤 Informations Client")
            with st.container(border=True):
                age = st.slider("Âge (ans)", 18, 100, 30)
                gender = st.selectbox("Genre", ["female", "male"])
                education = st.selectbox("Éducation", ["High School", "Bachelor", "Associate", "Master", "Doctorate"])
                income = st.number_input("Revenu Annuel (€)", min_value=0, value=60000, step=1000, format="%d")
                emp_exp = st.slider("Années d'expérience pro", 0, 50, 5)
                home_ownership = st.selectbox("Type de Résidence", ["RENT", "OWN", "MORTGAGE", "OTHER"])

        with col_right:
            st.markdown("### 💰 Détails du Prêt")
            with st.container(border=True):
                amount = st.number_input("Montant Demandé (€)", min_value=500, value=10000, step=500, format="%d")
                intent = st.selectbox("Motif du Prêt", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
                credit_score = st.number_input("Score de Crédit", min_value=300, max_value=850, value=650)
                int_rate = st.slider("Taux d'intérêt (%)", 5.0, 25.0, 11.5, step=0.1)

        # Additional Risk Info
        with st.expander("📂 Historique de Crédit (Optionnel)", expanded=False):
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                default_file_disp = st.radio("Défaut de paiement passé ?", ["Non", "Oui"], horizontal=True)
                default_file = 'Yes' if default_file_disp == "Oui" else 'No'
            with col_ex2:
                cred_hist = st.number_input("Ancienneté historique (années)", 0, 50, 3)

    # Computed Metrics
    percent_income = amount / income if income > 0 else 0
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("� LANCER L'ANALYSE DU RISQUE"):
            # Prepare Data    
            client_data = pd.DataFrame([{
                'person_age': age,
                'person_gender': gender,
                'person_education': education,
                'person_income': income,
                'person_emp_exp': emp_exp,
                'person_home_ownership': home_ownership,
                'loan_amnt': amount,
                'loan_intent': intent,
                'loan_int_rate': int_rate,
                'loan_percent_income': percent_income,
                'cb_person_cred_hist_length': cred_hist,
                'credit_score': credit_score,
                'previous_loan_defaults_on_file': default_file
            }])

            # Processing
            with st.spinner('Analyse par l\'IA en cours...'):
                try:
                    # Encoding
                    for col, le in encoders.items():
                        if col in client_data.columns:
                            # Use [[col]] to keep 2D for OrdinalEncoder
                            client_data[col] = le.transform(client_data[[col]]).flatten()
                    
                    # Ensure correct column order
                    client_data = client_data[FEATURES]
                    
                    prob_default = pipeline.predict_proba(client_data)[0][1] # Prob(1)
                    decision = pipeline.predict(client_data)[0]
                    prob_repay = 1 - prob_default
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")
                    st.stop()

            # Display Results
            st.markdown("---")
            
            # Logic: 0 = Approved, 1 = Refused
            if decision == 0:
                st.markdown(f"""
                <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #10B981;">
                    <h2 style="color: #065F46; margin:0;">✅ DOSSIER APPROUVÉ</h2>
                    <p style="color: #047857; font-size: 1.2rem; margin-top: 10px;">Confiance de l'IA : <strong>{prob_repay:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                 st.markdown(f"""
                <div style="background-color: #FEE2E2; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #EF4444;">
                    <h2 style="color: #991B1B; margin:0;">🚫 RISQUE ÉLEVÉ - REFUS RECOMMANDÉ</h2>
                    <p style="color: #B91C1C; font-size: 1.2rem; margin-top: 10px;">Probabilité de défaut : <strong>{prob_default:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)

            # Details
            st.markdown("### � Analyse Détaillée")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Ratio Dette/Revenu", f"{percent_income:.1%}", delta="-2%" if percent_income < 0.2 else "Alert" if percent_income > 0.4 else None, delta_color="inverse")
            with col_res2:
                 st.metric("Taux d'intérêt", f"{int_rate}%")
            with col_res3:
                 st.metric("Score de Crédit", credit_score)

            # Justification Text (Styled)
            st.info("💡 **Facteurs Clés** : " + 
                   ("Le ratio dette/revenu est critique. " if percent_income > 0.4 else "Endettement maîtrisé. ") +
                   ("Historique de paiement négatif détecté. " if default_file == 'Yes' else "Historique bancaire sain. ")
            )
else:
    st.error("Erreur critique : Le modèle n'a pas pu être chargé.")
