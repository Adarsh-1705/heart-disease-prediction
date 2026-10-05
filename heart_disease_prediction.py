# ============================================================
# Heart Disease Prediction using Machine Learning
# ============================================================
# Dataset: Cleveland Heart Disease Dataset (UCI Repository)
# Features: 13 clinical features -> Target: Disease (0/1)
# ============================================================

import os
import sys

# Force UTF-8 output on Windows to handle box-drawing characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Output directory ───────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Set plot style ─────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("  Heart Disease Prediction — ML Project")
print("=" * 60)

# Column names for the Cleveland dataset
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

# Load from UCI or local CSV
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
try:
    df = pd.read_csv(url, names=columns, na_values='?')
    print(f"\n[✓] Dataset loaded from UCI ({df.shape[0]} rows, {df.shape[1]} cols)")
except Exception:
    # Fallback: generate a realistic synthetic dataset for demo
    np.random.seed(42)
    n = 303
    df = pd.DataFrame({
        'age':      np.random.randint(29, 77, n),
        'sex':      np.random.randint(0, 2, n),
        'cp':       np.random.randint(0, 4, n),
        'trestbps': np.random.randint(94, 200, n),
        'chol':     np.random.randint(126, 564, n),
        'fbs':      np.random.randint(0, 2, n),
        'restecg':  np.random.randint(0, 3, n),
        'thalach':  np.random.randint(71, 202, n),
        'exang':    np.random.randint(0, 2, n),
        'oldpeak':  np.round(np.random.uniform(0, 6.2, n), 1),
        'slope':    np.random.randint(0, 3, n),
        'ca':       np.random.randint(0, 4, n),
        'thal':     np.random.choice([3, 6, 7], n),
        'target':   np.random.randint(0, 2, n),
    })
    print(f"\n[!] Using synthetic demo dataset ({n} rows)")

# Binarize target (0 = no disease, 1 = disease)
df['target'] = (df['target'] > 0).astype(int)

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n── 2. EXPLORATORY DATA ANALYSIS ──────────────────────")
print(df.head())
print(f"\nShape : {df.shape}")
print(f"Nulls :\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")

# Drop missing values (Cleveland dataset has a few '?' rows)
df.dropna(inplace=True)
print(f"\nShape after dropping nulls: {df.shape}")

# ── Plot 1: Target distribution ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
counts = df['target'].value_counts()
ax.bar(['No Disease (0)', 'Disease (1)'], counts.values,
       color=['steelblue', 'tomato'], edgecolor='white')
ax.set_title('Target Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(counts.values):
    ax.text(i, v + 2, str(v), ha='center', fontweight='bold')

# ── Plot 2: Correlation heatmap ────────────────────────────
ax2 = axes[1]
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax2, cmap='coolwarm',
            annot=False, linewidths=0.5)
ax2.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_eda_overview.png'), bbox_inches='tight')
plt.close()
print("\n[✓] Saved: 01_eda_overview.png")

# ── Plot 3: Key feature distributions ─────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
features = ['age', 'chol', 'trestbps', 'thalach', 'oldpeak', 'cp']
titles    = ['Age', 'Cholesterol', 'Resting BP', 'Max Heart Rate', 'ST Depression', 'Chest Pain Type']

for ax, feat, title in zip(axes.flat, features, titles):
    for val, color, label in zip([0, 1], ['steelblue', 'tomato'], ['No Disease', 'Disease']):
        ax.hist(df[df['target'] == val][feat], bins=20,
                alpha=0.6, color=color, label=label, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)

plt.suptitle('Feature Distributions by Target Class', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_feature_distributions.png'), bbox_inches='tight')
plt.close()
print("[✓] Saved: 02_feature_distributions.png")

# ============================================================
# 3. PREPROCESSING
# ============================================================
print("\n── 3. PREPROCESSING ───────────────────────────────────")

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train size : {X_train.shape[0]}")
print(f"Test  size : {X_test.shape[0]}")

# ============================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================
print("\n── 4. MODEL TRAINING ──────────────────────────────────")

models = {
    'Logistic Regression':    LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':          DecisionTreeClassifier(random_state=42),
    'Random Forest':          RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':      GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM':                    SVC(probability=True, random_state=42),
    'K-Nearest Neighbors':    KNeighborsClassifier(),
}

results = []
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    cv_sc  = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy').mean()

    results.append({'Model': name, 'Accuracy': acc, 'AUC-ROC': auc, 'CV Accuracy': cv_sc})
    print(f"  {name:25s} | Acc: {acc:.4f} | AUC: {auc:.4f} | CV: {cv_sc:.4f}")

results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)

# ── Plot 4: Model comparison ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results_df))
w = 0.25
ax.bar(x - w, results_df['Accuracy'],   width=w, label='Accuracy',    color='steelblue')
ax.bar(x,     results_df['AUC-ROC'],    width=w, label='AUC-ROC',     color='tomato')
ax.bar(x + w, results_df['CV Accuracy'],width=w, label='CV Accuracy', color='mediumseagreen')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax.set_ylim(0.5, 1.05)
ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_model_comparison.png'), bbox_inches='tight')
plt.close()
print("\n[✓] Saved: 03_model_comparison.png")

# ============================================================
# 5. BEST MODEL — DETAILED ANALYSIS
# ============================================================
best_name  = results_df.iloc[0]['Model']
best_model = models[best_name]
print(f"\n── 5. BEST MODEL: {best_name} ──────────────────────")

y_pred_best = best_model.predict(X_test_sc)
y_prob_best = best_model.predict_proba(X_test_sc)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['No Disease', 'Disease']))

# ── Plot 5: Confusion matrix + ROC ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[0].set_title(f'Confusion Matrix\n{best_name}', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
auc_val = roc_auc_score(y_test, y_prob_best)
axes[1].plot(fpr, tpr, color='tomato', lw=2, label=f'AUC = {auc_val:.4f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title(f'ROC Curve — {best_name}', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_best_model_analysis.png'), bbox_inches='tight')
plt.close()
print("[✓] Saved: 04_best_model_analysis.png")

# ── Plot 6: Feature importance (if tree-based) ─────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(f'Feature Importance — {best_name}', fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_feature_importance.png'), bbox_inches='tight')
    plt.close()
    print("[✓] Saved: 05_feature_importance.png")

# ============================================================
# 6. HYPERPARAMETER TUNING (Random Forest)
# ============================================================
print("\n── 6. HYPERPARAMETER TUNING (Random Forest) ───────────")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [None, 5, 10],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid.fit(X_train_sc, y_train)
print(f"Best Params : {grid.best_params_}")
print(f"Best CV Acc : {grid.best_score_:.4f}")
tuned_acc = accuracy_score(y_test, grid.best_estimator_.predict(X_test_sc))
print(f"Test Acc    : {tuned_acc:.4f}")

# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================
def predict_heart_disease(patient_data: dict) -> str:
    """
    Predict heart disease risk for a new patient.

    Parameters (keys):
      age, sex, cp, trestbps, chol, fbs, restecg,
      thalach, exang, oldpeak, slope, ca, thal

    Returns:
      Risk string with probability.
    """
    df_in = pd.DataFrame([patient_data], columns=X.columns)
    df_sc  = scaler.transform(df_in)
    prob   = best_model.predict_proba(df_sc)[0][1]
    label  = "HIGH RISK 🔴" if prob >= 0.5 else "LOW RISK  🟢"
    return f"Prediction: {label} | Probability: {prob:.2%}"


# Example patient
sample = {
    'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 132,
    'chol': 342, 'fbs': 0, 'restecg': 1, 'thalach': 166,
    'exang': 0, 'oldpeak': 1.4, 'slope': 2, 'ca': 0, 'thal': 2
}
print(f"\n── Sample Prediction ─────────────────────────────────")
print(f"Patient : {sample}")
print(predict_heart_disease(sample))

print("\n" + "=" * 60)
print(f"\n  All outputs saved to: {OUTPUT_DIR}")
print("=" * 60)
