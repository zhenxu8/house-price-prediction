# House Prices Prediction – Kaggle Project
Data science project applying EDA, missing value imputation, rare category grouping, and leakage-free pipelines to predict house prices. Compares linear models, Random Forest, Gradient Boosting, XGBoost, and LightGBM with cross-validated RMSE optimization.

**Author:** Zhen Xu  
**Project Type:** Supervised Machine Learning (Regression)  
**Dataset:** Kaggle – *House Prices: Advanced Regression Techniques*  
**Tech Stack:** Python · Pandas · NumPy · scikit-learn · XGBoost · LightGBM  

---

## Project Overview

This project develops an end-to-end machine learning pipeline to predict residential property sale prices using structured tabular data from Ames, Iowa.

The objective is to model the relationship between **79 explanatory variables** and the target:

**Target Variable:** `SalePrice` (continuous)

The project focuses on:

- Robust preprocessing  
- Domain-aware missing value treatment  
- Feature engineering  
- Model comparison  
- Hyperparameter optimization  
- Generalization performance  

---

## Dataset Description

The dataset includes features describing:

- Property size (e.g., `LotArea`, `GrLivArea`)  
- Construction details (`YearBuilt`, `OverallQual`)  
- Location (`Neighborhood`)  
- Quality and condition ratings  
- Basement and garage attributes  
- Amenities and structural features  

**Provided datasets:**

- **`train.csv`** – Features + target (`SalePrice`)  
- **`test.csv`** – Features only  

---

## Evaluation Metric

Kaggle evaluates submissions using:

**Root Mean Squared Log Error (RMSLE)**

A log transformation is applied because it:

- Reduces target skewness  
- Stabilizes variance  
- Penalizes relative prediction errors  
- Limits sensitivity to extreme outliers  

---

## Workflow Summary

1. Exploratory Data Analysis (EDA)  
2. Target distribution & skewness assessment  
3. Log transformation (`log1p`)  
4. Missing value treatment  
5. Rare category grouping  
6. Categorical encoding  
7. Feature scaling  
8. Pipeline construction  
9. Model training  
10. Cross-validation  
11. Hyperparameter tuning  
12. Final prediction & submission  

---

## Exploratory Data Analysis

### Target Variable

- Detected right-skewed distribution  
- Confirmed via histogram, skewness statistic, and Q-Q plot  
- Applied log transformation for modeling stability  

### Feature Relationships

- Scatterplots revealed linear and nonlinear patterns  
- Correlation analysis identified strong predictors  
- ANOVA screening evaluated categorical feature relevance  

---

## Data Preprocessing Strategy

### Missing Value Handling (Domain-Aware)

Different imputation strategies were applied based on feature semantics:

| Scenario | Strategy |
|----------|----------|
| Missing indicates absence (garage/basement/etc.) | Impute `"None"` / `0` |
| Skewed numerical feature (`LotFrontage`) | Median imputation |
| Rare minimal missing rows (`Electrical`) | Row removal |

---

### Rare Category Grouping

**Problem:**

- Sparse one-hot encoded features  
- Increased overfitting risk  
- Noisy statistical distributions  

**Solution:**

- Grouped low-frequency categories into `"Other"`  
- Reduced sparsity and improved model stability  

---

### Feature Selection

Removed low-information or highly imbalanced features:

- `Street`  
- `Utilities`  
- `CentralAir`  
- `Id`  

---

### Leakage-Free Pipeline

Implemented using:

- `Pipeline`  
- `ColumnTransformer`  
- Custom transformers  

**Benefits:**

- Prevents data leakage  
- Ensures reproducibility  
- Applies consistent transformations across CV folds  

---

## Models Evaluated

### Linear Baselines

- Ridge Regression  
- Lasso Regression  
- ElasticNet  

**Purpose:**

- Interpretability  
- Regularization under multicollinearity  
- Benchmarking  

---

### Tree-Based Models

- Random Forest  
- Gradient Boosting  
- XGBoost  
- LightGBM  

**Purpose:**

- Capture nonlinear relationships  
- Learn feature interactions  
- Improve predictive accuracy  

---

## Hyperparameter Optimization

### Stage 1: RandomizedSearchCV

- Efficient exploration of large parameter space  
- Lower computational cost than exhaustive grid search  

---

### Stage 2: GridSearchCV Refinement

- Local fine-tuning around promising configurations  
- Improved performance stability  

---

## Validation Strategy

Used **5-fold Cross-Validation** on:

`log(SalePrice)`

**Reported:**

- Mean CV RMSE (log scale)  
- Standard deviation  

**Rationale:**

- Aligns with Kaggle evaluation metric  
- Provides reliable generalization estimate  

---

## Final Output

- Trained optimized boosting model  
- Generated predictions on test set  
- Created Kaggle submission file  

---

## Key Takeaways

- Built an end-to-end ML pipeline  
- Applied domain-aware data cleaning  
- Reduced categorical sparsity via rare label grouping  
- Addressed skewed targets using log transformation  
- Compared linear and ensemble models  
- Performed two-stage hyperparameter tuning  
- Evaluated performance using cross-validated RMSE  

---

## Skills Demonstrated

- Data Cleaning & Imputation  
- Exploratory Data Analysis  
- Feature Engineering  
- Categorical Encoding  
- Feature Scaling  
- Model Selection  
- Cross-Validation  
- Hyperparameter Tuning  
- Bias–Variance Trade-off  
- Pipeline Design  

---

## Competition Reference

Kaggle – House Prices: Advanced Regression Techniques  
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
