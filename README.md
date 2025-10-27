# 🏆 World Cup Match Analysis — Regression Models

## 📘 Overview
This project explores **FIFA World Cup match data** using three types of regression models:

| Notebook | Method | Goal |
|-----------|---------|------|
| `world_cup_linear_regression.ipynb` | Linear Regression | Predict total goals in a match using simple linear relationships. |
| `world_cup_MLR_analysis.ipynb` | Multiple Linear Regression | Model total goals using multiple numeric and categorical predictors. |
| `world_cup_logistic_regression.ipynb` | Logistic Regression | Predict whether the **home team wins** a match (binary classification). |

The dataset used is:  
`WorldCupMatches_cleaned.csv`

---

## ⚙️ Requirements
All notebooks are written in Python and require the following libraries:

```bash
pip install pandas numpy matplotlib scikit-learn seaborn joblib nbformat
```

---

## 📂 Files
```
├── WorldCupMatches_cleaned.csv          # Dataset
├── world_cup_linear_regression.ipynb    # Simple Linear Regression
├── world_cup_MLR_analysis.ipynb         # Multiple Linear Regression
└── world_cup_logistic_regression.ipynb  # Logistic Regression
```

---

## 🧩 1. Linear Regression
**File:** `world_cup_linear_regression.ipynb`  
**Objective:**  
Predict **Total Goals** in a match using a simple linear regression model.

**Key Steps:**
- Feature engineering: `TotalGoals`, `GoalDiff`, `is_knockout`
- Model: `LinearRegression()` from scikit-learn
- Metrics: MAE, RMSE, R², CV R²
- Visualization: Actual vs Predicted total goals

---

## 📊 2. Multiple Linear Regression
**File:** `world_cup_MLR_analysis.ipynb`  
**Objective:**  
Use multiple predictors (year, attendance, match stage, etc.) to estimate **Total Goals**.

**Key Steps:**
- Numeric + Categorical feature handling via `ColumnTransformer`
- Pipeline: Imputation → Scaling → One-Hot Encoding → Linear Regression
- Evaluation: MAE, RMSE, R², CV R²
- Diagnostics: residual plots, coefficient tables
- Interpretation: which features most strongly affect total goals

---

## ⚽ 3. Logistic Regression
**File:** `world_cup_logistic_regression.ipynb`  
**Objective:**  
Predict if the **home team wins (1)** or **does not win (0)** using logistic regression.

**Key Steps:**
- Binary target creation: `HomeWin = (home_team_goals > away_team_goals)`
- Model pipeline with scaling and encoding
- Evaluation Metrics:
  - Accuracy, Precision, Recall, F1-score
  - ROC–AUC, Precision–Recall AUC
- Visuals: confusion matrix, ROC curve, PR curve
- Coefficient and **odds ratio** analysis

---

## 🧠 Key Learning Outcomes
- Compare **linear vs multiple linear** vs **logistic** regression methods.  
- Learn how numeric and categorical features affect model accuracy.  
- Understand regression assumptions and model diagnostics.  
- Apply machine learning pipelines for robust preprocessing.  

---

## 🚀 How to Run
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Load any of the `.ipynb` files.
3. Run all cells to reproduce results.

Or use VS Code / Google Colab for faster execution.

---

## 📈 Possible Extensions
- Add **Ridge/Lasso Regression** to test regularization effects.
- Explore **multiclass logistic regression** (HomeWin/Draw/AwayWin).
- Include **feature importance** and **permutation analysis**.
- Train/test split across **different World Cup years**.

## 🧑‍💻 Author
**Adnan Altimeemy**  
Data Scientist  
Educational Data Science & Machine Learning Enthusiast

---

## 📄 License
This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this work with attribution.

