"""
Heart Disease Prediction — Flask Web Application
=================================================
Backend API that trains the ML model on startup,
then serves predictions via POST /predict
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# ── App setup ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Global model state ─────────────────────────────────────
model      = None
scaler     = None
feature_cols = None
model_name = None
model_acc  = None
model_auc  = None
all_results = []

# ============================================================
# TRAIN MODEL ON STARTUP
# ============================================================
def train_model():
    global model, scaler, feature_cols, model_name, model_acc, model_auc, all_results

    columns = [
        'age','sex','cp','trestbps','chol','fbs',
        'restecg','thalach','exang','oldpeak','slope',
        'ca','thal','target'
    ]

    # Load dataset
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )
    try:
        df = pd.read_csv(url, names=columns, na_values='?')
        print(f"[✓] Dataset loaded from UCI ({df.shape[0]} rows)")
    except Exception:
        np.random.seed(42)
        n = 303
        df = pd.DataFrame({
            'age':      np.random.randint(29, 77, n).astype(float),
            'sex':      np.random.randint(0, 2, n).astype(float),
            'cp':       np.random.randint(0, 4, n).astype(float),
            'trestbps': np.random.randint(94, 200, n).astype(float),
            'chol':     np.random.randint(126, 564, n).astype(float),
            'fbs':      np.random.randint(0, 2, n).astype(float),
            'restecg':  np.random.randint(0, 3, n).astype(float),
            'thalach':  np.random.randint(71, 202, n).astype(float),
            'exang':    np.random.randint(0, 2, n).astype(float),
            'oldpeak':  np.round(np.random.uniform(0, 6.2, n), 1),
            'slope':    np.random.randint(0, 3, n).astype(float),
            'ca':       np.random.randint(0, 4, n).astype(float),
            'thal':     np.random.choice([3.0, 6.0, 7.0], n),
            'target':   np.random.randint(0, 2, n),
        })
        print("[!] Using synthetic demo dataset")

    df['target'] = (df['target'] > 0).astype(int)
    df.dropna(inplace=True)

    X = df.drop('target', axis=1)
    y = df['target']
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'Logistic Regression':  LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':        DecisionTreeClassifier(random_state=42),
        'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM':                  SVC(probability=True, random_state=42),
        'K-Nearest Neighbors':  KNeighborsClassifier(),
    }

    best_auc  = -1
    best_mdl  = None
    best_name = None
    results   = []

    for name, mdl in models.items():
        mdl.fit(X_train_sc, y_train)
        y_pred = mdl.predict(X_test_sc)
        y_prob = mdl.predict_proba(X_test_sc)[:, 1]
        acc = round(accuracy_score(y_test, y_pred) * 100, 2)
        auc = round(roc_auc_score(y_test, y_prob) * 100, 2)
        results.append({'name': name, 'accuracy': acc, 'auc': auc})
        print(f"  {name:25s} | Acc: {acc:.1f}% | AUC: {auc:.1f}%")
        if auc > best_auc:
            best_auc  = auc
            best_mdl  = mdl
            best_name = name

    model      = best_mdl
    model_name = best_name
    model_acc  = next(r['accuracy'] for r in results if r['name'] == best_name)
    model_auc  = best_auc
    all_results = results
    print(f"\n[✓] Best model: {model_name} (AUC {model_auc:.1f}%)")


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/model-info')
def model_info():
    """Return info about the trained model."""
    return jsonify({
        'model_name': model_name,
        'accuracy':   model_acc,
        'auc':        model_auc,
        'all_results': all_results,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Receive patient data as JSON, return prediction.

    Expected JSON keys (13 clinical features):
      age, sex, cp, trestbps, chol, fbs, restecg,
      thalach, exang, oldpeak, slope, ca, thal
    """
    try:
        data = request.get_json(force=True)

        # Build input dataframe in the correct feature order
        row = {col: float(data[col]) for col in feature_cols}
        df_in = pd.DataFrame([row], columns=feature_cols)
        df_sc = scaler.transform(df_in)

        prob  = float(model.predict_proba(df_sc)[0][1])
        label = 1 if prob >= 0.5 else 0

        return jsonify({
            'prediction': label,
            'probability': round(prob * 100, 1),
            'risk_level': 'HIGH RISK' if label == 1 else 'LOW RISK',
            'model_used': model_name,
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  Heart Disease Prediction — Web App")
    print("=" * 55)
    print("\n[*] Training models, please wait...\n")
    train_model()
    print(f"\n[✓] Server starting at http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
