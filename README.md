# Heart Disease Prediction — Machine Learning Project

## Overview
Predicts whether a patient has heart disease using 13 clinical features from the **Cleveland Heart Disease Dataset** (UCI ML Repository).

---

## Dataset Features

| Feature     | Description                              | Type        |
|-------------|------------------------------------------|-------------|
| age         | Age in years                             | Numeric     |
| sex         | Sex (1=male, 0=female)                   | Categorical |
| cp          | Chest pain type (0–3)                    | Categorical |
| trestbps    | Resting blood pressure (mm Hg)           | Numeric     |
| chol        | Serum cholesterol (mg/dl)                | Numeric     |
| fbs         | Fasting blood sugar > 120 mg/dl (1/0)   | Categorical |
| restecg     | Resting ECG results (0–2)                | Categorical |
| thalach     | Max heart rate achieved                  | Numeric     |
| exang       | Exercise-induced angina (1/0)            | Categorical |
| oldpeak     | ST depression induced by exercise        | Numeric     |
| slope       | Slope of peak exercise ST segment (0–2) | Categorical |
| ca          | Number of major vessels (0–3)            | Numeric     |
| thal        | Thalassemia type (3/6/7)                 | Categorical |
| **target**  | **Heart disease (1=yes, 0=no)**          | **Label**   |

---

## Project Structure

```
heart_disease_prediction.py   ← Main ML script
README.md                     ← This file
outputs/
  01_eda_overview.png          ← Target distribution + correlation heatmap
  02_feature_distributions.png ← Feature histograms by class
  03_model_comparison.png      ← Accuracy / AUC / CV comparison bar chart
  04_best_model_analysis.png   ← Confusion matrix + ROC curve
  05_feature_importance.png    ← Feature importance (tree-based models)
```

---

## Models Used

1. **Logistic Regression** — Baseline linear classifier
2. **Decision Tree** — Interpretable tree-based model
3. **Random Forest** — Ensemble of decision trees
4. **Gradient Boosting** — Sequential boosting ensemble
5. **SVM** — Support vector machine with RBF kernel
6. **K-Nearest Neighbors** — Distance-based classifier

---

## Pipeline

```
Raw Data
   ↓
EDA (distributions, correlations)
   ↓
Preprocessing (drop nulls, train/test split 80/20, StandardScaler)
   ↓
Train 6 Models
   ↓
Evaluate (Accuracy, AUC-ROC, 5-Fold CV)
   ↓
Best Model Selected
   ↓
Hyperparameter Tuning (GridSearchCV)
   ↓
Prediction Function (new patient data)
```

---

## How to Run

```bash
# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn

# Run the project
python heart_disease_prediction.py
```

---

## Expected Performance (Real UCI Dataset)

| Model               | Accuracy | AUC-ROC |
|---------------------|----------|---------|
| Logistic Regression | ~85%     | ~0.91   |
| Random Forest       | ~87%     | ~0.93   |
| Gradient Boosting   | ~88%     | ~0.94   |
| SVM                 | ~84%     | ~0.90   |

> **Note:** The script auto-downloads the Cleveland dataset from UCI.  
> If the URL is unavailable it falls back to a synthetic demo dataset.

---

## Predicting a New Patient

```python
sample = {
    'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 132,
    'chol': 342, 'fbs': 0, 'restecg': 1, 'thalach': 166,
    'exang': 0, 'oldpeak': 1.4, 'slope': 2, 'ca': 0, 'thal': 2
}
print(predict_heart_disease(sample))
# → Prediction: LOW RISK 🟢 | Probability: 21.90%
```

---

## Tech Stack
- **Python 3.x**
- **scikit-learn** — ML models, preprocessing, evaluation
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualization
