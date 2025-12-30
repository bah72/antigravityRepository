import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, classification_report
from sklearn.impute import SimpleImputer

# Set style
sns.set(style="whitegrid")

def get_base64_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64

# Load Data
try:
    df = pd.read_csv('loan_data.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Setup
target = 'loan_status'
features = [
    'person_age', 'person_gender', 'person_education', 'person_income', 
    'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent', 
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
    'credit_score', 'previous_loan_defaults_on_file'
]
features = [col for col in features if col in df.columns]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Model 1: HistGradientBoosting (Our Champion) ---
# Preprocessing for HGB (Ordinal for Cats)
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

hgb_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)
hgb_model = Pipeline([
    ('preprocessor', hgb_preprocessor),
    ('classifier', HistGradientBoostingClassifier(learning_rate=0.05, max_depth=10, random_state=42))
])

# --- Model 2: Logistic Regression (Benchmark) ---
# Preprocessing for LR (OneHot for Cats, Scaling for Num, Imputation)
lr_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols)
    ]
)
lr_model = Pipeline([
    ('preprocessor', lr_preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Train
print("Training HGB...")
hgb_model.fit(X_train, y_train)
print("Training LR...")
lr_model.fit(X_train, y_train)

# Evaluate
models = {'HistGradientBoosting': hgb_model, 'LogisticRegression': lr_model}
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'auc': auc, 'acc': acc, 'y_pred': y_pred, 'y_proba': y_proba}

best_model_name = max(results, key=lambda k: results[k]['auc'])
print(f"Best Model: {best_model_name}")

# --- Plots ---

# 1. ROC Curve Comparison
fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.3f})")
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_title('Comparaison Courbe ROC')
ax_roc.set_xlabel('Taux Faux Positifs')
ax_roc.set_ylabel('Taux Vrais Positifs')
ax_roc.legend()
img_roc = get_base64_image(fig_roc)
plt.close(fig_roc)

# 2. Confusion Matrix (Best Model)
best_res = results[best_model_name]
cm = confusion_matrix(y_test, best_res['y_pred'])
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_title(f'Matrice de Confusion ({best_model_name})')
ax_cm.set_xlabel('Prédit')
ax_cm.set_ylabel('Réel')
img_cm = get_base64_image(fig_cm)
plt.close(fig_cm)

# 3. Feature Importance (HGB specific via Permutation Importance or native if available)
# HGB doesn't have feature_importances_ attribute directly on the pipeline object easily accessible without permutation importance 
# strictly speaking, but let's use Permutation Importance for robustness for the best model.
from sklearn.inspection import permutation_importance

print("Calculating Feature Importance...")
# Use a subset of test data for speed if needed, but test set is likely fine
perm_importance = permutation_importance(models[best_model_name], X_test, y_test, n_repeats=5, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
ax_fi.barh(np.array(features)[sorted_idx], perm_importance.importances_mean[sorted_idx])
ax_fi.set_title(f'Importance des Variables ({best_model_name})')
img_fi = get_base64_image(fig_fi)
plt.close(fig_fi)


# --- HTML Generation ---

html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Risque Crédit</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f9; color: #333; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #2c3e50; }}
        .metric-box {{ display: inline-block; padding: 15px; background: #e8f4f8; border-radius: 5px; margin-right: 20px; min-width: 150px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        .chart {{ margin: 30px 0; text-align: center; }}
        img {{ max-width: 100%; border: 1px solid #eee; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        .winner {{ font-weight: bold; color: green; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Rapport de Performance du Modèle Crédit</h1>
        <p>Généré automatiquement par l'Agent Risque Crédit.</p>
        
        <h2>Comparaison des Modèles</h2>
        <table>
            <thead>
                <tr>
                    <th>Modèle</th>
                    <th>ROC-AUC</th>
                    <th>Précision (Accuracy)</th>
                </tr>
            </thead>
            <tbody>
                {"".join([f"<tr class='{'winner' if name==best_model_name else ''}'><td>{name} {'(Gagnant)' if name==best_model_name else ''}</td><td>{res['auc']:.4f}</td><td>{res['acc']:.4f}</td></tr>" for name, res in results.items()])}
            </tbody>
        </table>

        <h2>Performance du Meilleur Modèle ({best_model_name})</h2>
        <div>
            <div class="metric-box">
                <div class="metric-value">{results[best_model_name]['auc']:.2%}</div>
                <div class="metric-label">ROC-AUC</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{results[best_model_name]['acc']:.2%}</div>
                <div class="metric-label">Précision Globale</div>
            </div>
        </div>

        <div class="chart">
            <h3>Courbe ROC : Comparaison</h3>
            <img src="data:image/png;base64,{img_roc}" alt="ROC Curve">
        </div>

        <div class="chart">
            <h3>Matrice de Confusion</h3>
            <p>Visualisation des erreurs de classification sur l'ensemble de test.</p>
            <img src="data:image/png;base64,{img_cm}" alt="Confusion Matrix">
        </div>

        <div class="chart">
            <h3>Importance des Variables</h3>
            <p>Facteurs ayant le plus d'impact sur la décision du modèle.</p>
            <img src="data:image/png;base64,{img_fi}" alt="Feature Importance">
        </div>
        
        <h2>Analyse de Cas Précédent</h2>
        <p>Le client test (30 ans, 60k revenu, RENT) a été classé comme <strong>REFUSÉ</strong> malgré un ratio dette/revenu sain. L'importance des variables ci-dessus suggère que des facteurs comme le taux d'intérêt (loan_int_rate) ou le montant du prêt (loan_amnt) combinés au type de logement (person_home_ownership) jouent un rôle crucial.</p>
    </div>
</body>
</html>
"""

with open('credit_risk_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Report generated: credit_risk_report.html")
