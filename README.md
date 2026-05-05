# ML Assignment – Online Retail II UCI Dataset
### Supervised Learning for Revenue Prediction | All Seven Research Questions

---

## Overview

This notebook implements a complete supervised machine learning pipeline for **revenue prediction** on the Online Retail II UCI Dataset. It covers the full workflow from data loading and preprocessing through model comparison, feature importance analysis, robustness testing, and a final model recommendation — structured around seven research questions (RQ1–RQ7).

| Attribute | Detail |
|---|---|
| **Dataset** | Online Retail II UCI Dataset |
| **File** | `Online_Retail_II_Cleaned (1).xlsx` |
| **Rows** | 10,000 transactions |
| **Columns** | 20 features |
| **Target Variable** | `Revenue` (continuous — regression task) |
| **Domain** | Retail analytics, product performance, customer behaviour prediction |
| **Random Seed** | 42 (all experiments) |

---

## Repository Structure

```
ML_Assignment_All_RQs (2).ipynb   ← Main notebook (all 7 RQs)
Online_Retail_II_Cleaned.xlsx     ← Input dataset (required)
README.md                         ← This file
```

---

## Requirements

Install dependencies before running:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap openpyxl
```

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisations |
| `scikit-learn` | Preprocessing, model training, evaluation |
| `xgboost` | Gradient boosted trees |
| `shap` | Model explainability (SHAP values) |
| `openpyxl` | Reading `.xlsx` dataset file |

> **Note:** The notebook was developed for Google Colab. The dataset path is set to `/content/Online_Retail_II_Cleaned (1).xlsx`. Update this path if running locally.

---

## Dataset Features

| Group | Features |
|---|---|
| **Transaction** | `Quantity`, `UnitPrice`, `Discount` |
| **Temporal** | `Year`, `Month`, `Day`, `Hour` |
| **Categorical** | `Country`, `Region`, `ProductCategory`, `OrderType`, `PaymentMode` |
| **Customer** | `ReturnFlag` |
| **Drop cols** | `InvoiceNo`, `InvoiceDate`, `CustomerID`, `StockCode`, `Description`, `Revenue` (target) |

**Engineered features** (created in RQ4 preprocessing experiments):
- `DiscountedPrice` = `UnitPrice × (1 − Discount/100)`
- `DiscountImpact` = `Discount × Quantity`
- `PriceQtyRatio` = `UnitPrice / (Quantity + 1)`

---

## Notebook Structure & Research Questions

### Global Setup
Imports all libraries, loads the dataset, defines feature groups, builds a reusable preprocessing pipeline (median imputation → StandardScaler for numeric; mode imputation → OneHotEncoder for categorical), and sets an 80/20 stratified train-test split.

---

### RQ1 — Baseline Performance
> *How effectively can baseline models predict revenue?*

Trains four baseline regressors: **Linear Regression**, **Decision Tree**, **k-NN**, and **Ridge Regression** on the full preprocessed feature set.

**Output:** `Table I` — MAE, RMSE, R² for each baseline model.  
**Figure 1** — Grouped bar chart comparing all three metrics across baseline models.

---

### RQ2 — Model Comparison
> *Which model achieves the best predictive performance?*

Trains five candidate models: **Linear Regression**, **Random Forest**, **Gradient Boosting**, **XGBoost**, and **SVR**. Results sorted by R² descending.

**Output:** `Table II` — Full comparison (MAE, RMSE, R²) sorted by R².  
**Figure 2** — Dual horizontal bar chart: (a) RMSE ranked lowest-to-best, (b) R² ranked highest-to-best.

---

### RQ3 — Feature Group Ablation
> *Which groups of features contribute most to prediction quality?*

Trains Random Forest on five feature subsets: Temporal Only, Categorical Only, Transaction Only, Transaction + Temporal, and Full Feature Set.

**Output:** `Table III` — MAE, RMSE, R² per feature group.  
**Figure 3** — Side-by-side horizontal bar charts: (a) RMSE, (b) R² by feature group.

---

### RQ4 — Effect of Preprocessing Strategies
> *How do different preprocessing pipelines affect model performance?*

Trains Random Forest under five preprocessing configurations: raw numerics only, label encoding, one-hot encoding, full pipeline with scaling, and full pipeline with engineered features.

**Output:** `Table IV` — MAE, RMSE, R² per preprocessing strategy.  
**Figure 4** — Bar + line chart showing RMSE and R² across preprocessing stages.

---

### RQ5 — Feature Importance & Interpretability
> *Which features drive revenue prediction, and what do they mean?*

Uses the best model (Random Forest, full pipeline) to extract Gini-based feature importances for the top 12 features. Also generates a SHAP summary plot on a 300-sample subset of the training data.

**Output:** `Table V` — Top-12 features ranked by importance score.  
**Figure 5** — Horizontal importance bar chart (YlOrRd colormap).  
**Figure 5b** — SHAP beeswarm summary plot.

---

### RQ6 — Robustness & Generalisation
> *How stable is the best model under different conditions?*

Evaluates Random Forest (full pipeline) across six scenarios:

| Scenario | Description |
|---|---|
| 80/20 Split | Standard holdout |
| 70/30 Split | Larger test set |
| 5-Fold CV | Cross-validated R² (mean ± std) |
| 10-Fold CV | Cross-validated R² (mean ± std) |
| 10% Noise | Gaussian noise added to numeric features |
| 20% Missingness | 20% of all values randomly set to NaN |

**Output:** `Table VI` — MAE, RMSE, R², Stability label per scenario.  
**Figure 6** — Three-panel chart: (a) R² bar, (b) RMSE bar, (c) CV boxplot for 5-fold vs 10-fold.

---

### RQ7 — Final Recommendation
> *Which model is most practically useful for deployment?*

Scores four models — **Linear Regression**, **Random Forest**, **Gradient Boosting**, **XGBoost** — across seven criteria (1–5 scale): Predictive Accuracy, Interpretability, Robustness, Nonlinear Modelling, Computational Efficiency, Deployment Suitability, Managerial Usefulness.

**Output:** `Table VII` — Decision matrix heatmap scores.  
**Figure 7** — (a) Decision matrix heatmap (RdYlGn), (b) Radar chart comparing all four models.

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | Mean Absolute Error | Average absolute deviation in revenue units |
| **RMSE** | Root Mean Squared Error | Penalises large errors more heavily |
| **R²** | Coefficient of Determination | Proportion of variance explained (1.0 = perfect) |

---

## How to Run

**Google Colab (recommended):**
1. Upload `Online_Retail_II_Cleaned (1).xlsx` to `/content/`
2. Open the notebook in Colab
3. Run all cells top-to-bottom (`Runtime → Run all`)

**Local Jupyter:**
1. Update the dataset path in the load cell:
   ```python
   df = pd.read_excel('path/to/Online_Retail_II_Cleaned.xlsx')
   ```
2. Run: `jupyter notebook "ML_Assignment_All_RQs (2).ipynb"`

---

## Key Results Summary

| RQ | Finding |
|---|---|
| RQ1 | All baselines capture revenue patterns; Decision Tree leads among baselines |
| RQ2 | Ensemble models (Random Forest, Gradient Boosting, XGBoost) substantially outperform linear baselines |
| RQ3 | Full feature set yields the best R²; transaction features alone are the strongest single group |
| RQ4 | Full pipeline with feature engineering achieves the highest performance |
| RQ5 | `Quantity`, `UnitPrice`, and `Discount`-derived features are the top revenue predictors |
| RQ6 | Random Forest degrades gradually under noise/missingness — robust and reliable |
| RQ7 | **Random Forest** is the recommended model: best balance of accuracy, robustness, and deployment suitability |

---

## Notes

- All experiments use `random_state=42` for reproducibility.
- SHAP computation samples 300 training instances to keep runtime manageable.
- The preprocessing pipeline is defined once as `make_preprocessor()` and reused consistently across all RQs to avoid data leakage.
- Cross-validation in RQ6 is computed on the full dataset (`X`, `y`) using `cross_val_score`.
