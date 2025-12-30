# Projet Assistant Risque Crédit

Ce projet contient un ensemble d'outils pour l'analyse du risque crédit utilisant un modèle **HistGradientBoostingClassifier**.

## 📂 Fichiers Clés

- **`app.py`** : Une application web interactive (Streamlit) pour les assistants bancaires. Permet la saisie et l'évaluation en temps réel.
- **`credit_risk_agent.py`** : Le script principal contenant la logique du modèle et les fonctions de prédiction.
- **`generate_html_report.py`** : Script générant un rapport HTML complet (`credit_risk_report.html`) comparant les modèles et visualisant les performances.
- **`loan_data.csv`** : Les données utilisées pour l'entraînement.

## 🚀 Utilisation

### 1. Application Bancaire (Interface Web)
Pour lancer l'assistant interactif :
```bash
streamlit run app.py
```
Cela ouvrira votre navigateur avec le formulaire de saisie.

### 2. Rapport de Performance
Pour regénérer le rapport d'analyse (Courbes ROC, Matrices de confusion) :
```bash
python generate_html_report.py
```
Le résultat sera dans `credit_risk_report.html`.

### 3. Prédiction en Ligne de Commande
Pour tester rapidement un profil via Python :
```bash
python predict_runner.py
```

## 🛠️ Prérequis
Les librairies suivantes sont nécessaires :
```bash
pip install pandas scikit-learn matplotlib seaborn streamlit
```
