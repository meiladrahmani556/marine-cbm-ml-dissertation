# 🛠 Marine Gas Turbine Condition-Based Monitoring (CBM)  
## Machine Learning Dissertation Project

---

## 📌 Project Overview

This project develops a complete machine learning pipeline to predict gas turbine degradation in marine propulsion systems using the Condition-Based Monitoring (CBM) dataset.

The workflow follows established predictive maintenance practices:
1. Data acquisition
2. Exploratory data analysis
3. Data cleaning & preprocessing
4. Baseline modelling
5. Model optimisation & validation
6. Model interpretation
7. Learning curves & robustness analysis

The objective is not only predictive performance, but also interpretability and engineering insight.

---

## 📊 Dataset Description

The dataset is sourced from Kaggle:

**Condition-Based Monitoring in Marine System**  
URL: https://www.kaggle.com/datasets/kunalnehete/condition-based-monitoring-cbm-in-marine-system

It contains sensor measurements (pressures, temperatures, revolutions, torques, fuel flow, etc.) recorded from a marine gas turbine system under various operational conditions.  
The target variable modelled is:

➡ **GT Compressor decay state coefficient**

---

## 📥 Dataset Acquisition

Dataset download is handled in **Notebook 01** using the Kaggle API:

- Upload `kaggle.json` in Colab
- Authenticate Kaggle CLI
- Download raw CSV
- Save to `data/raw`

Errors encountered:
- Kaggle API not configured → fixed by uploading and placing `kaggle.json` correctly.
- Empty `data/raw` list → resolved with proper extraction and file move logic.

---

## 📌 Project Structure

Marine - cbm - ml - dissertation/

│

├── data/

│ ├── raw/ # Raw dataset
│ ├── processed/ # Processed ML inputs (generated in notebooks)

│

├── models/ # Saved model artifacts

├── notebooks/ # All Jupyter notebooks

├── requirements.txt # Required Python libraries

└── README.md # Project documentation

The following notebooks have been completed, with documented errors and resolution strategies:

---


---

## 🚧 Pipeline Status

### 🟢 Completed

#### **Notebook 01 — Data Acquisition**
- Kaggle API setup
- Dataset download
- Raw file extraction
- Folder structure creation  
*Errors & Fixes:*
- Needed to upload and position `kaggle.json`
- Empty folder before extraction resolved

---

#### **Notebook 02 — Exploratory Data Analysis**
- Dataset overview
- Summary statistics
- Missing value inspection
- Feature distributions
- Correlation heatmap
- Target vs feature analysis  
*Errors & Fixes:*
- All numeric columns read as objects initially → fixed using correct separators and conversion
- Inconsistent formatting → cast to numeric with `errors='coerce'`

---

#### **Notebook 03 — Data Cleaning & Preprocessing**
- Numeric casting
- Duplicate removal
- Missing values handling
- Train/test splitting
- Feature scaling (StandardScaler)  
*Errors & Fixes:*
- Column name whitespace → removed via `.str.strip()`
- Colab persistence issues → preprocessing replicated in later notebooks

---

#### **Notebook 04 — Baseline Modelling**
- Linear Regression
- Random Forest Regressor  
*Errors & Fixes:*
- Missing processed files → replaced by on-the-fly preprocessing in the notebook
- Column indexing KeyError → fixed by cleaning column names

---

#### **Notebook 05 — Model Optimisation & Validation**
- Cross-validation
- GridSearch hyperparameter tuning
- Training vs testing performance
- Overfitting check  
*Highlights:*
- Tuned R² ≈ **0.9982**
- CV R² ≈ **0.9963**
- Training & testing results confirm generalisation

---

#### **Notebook 06 — Model Interpretation & Engineering Analysis**
- Feature importance
- Correlation with target
- Residual distribution
- Actual vs Predicted plot  
*Insights:*
- Top features make engineering sense (pressure & temperature)
- Low risk of leakage
- Stable residuals

---

#### **Notebook 07 — Learning Curves & Robustness Analysis**
- Learning curve evaluation
- Model complexity (R² vs n_estimators)
- Feature removal robustness test  
*Findings:*
- Training & validation curves converge → low bias and variance
- Minimal performance drop after removing top 2 features  
  ➤ R² reduced from **0.9982** to **0.9978**  
  → Model is robust and not overly reliant on single features

---

## 🛠 Python Packages Used

- `pandas`, `numpy` → data manipulation  
- `scikit-learn` → preprocessing, modelling, grid search, learning curves  
- `matplotlib`, `seaborn` → visualisation  
- `joblib` → model saving  
- `google.colab.files` → file upload

---

## 📚 Code References & Reuse

External resources consulted:
- Scikit-learn official docs: https://scikit-learn.org/
- Kaggle API guide: https://www.kaggle.com/docs/api
- Learning curve technique: sklearn.model_selection.learning_curve

All code was adapted for this project; no full external scripts were copied.

---

## 📊 Training Curves & Metrics

Training curves and complexity analyses are provided in **Notebook 07** including:

- Learning curve with R²
- Model complexity plot (effect of forest size)
- Robustness plot excluding top features

These fulfil the requirement for performance curves.

---

## ⚠ Known Limitations

- Models trained on tabular snapshots rather than time series streams.
- No domain-specific feature engineering was applied (e.g., thermodynamic derived variables).
- Hyperparameter search limited to practical ranges due to execution time.

---


---

## 🚧 Pipeline Status

### 🟢 Completed

#### **Notebook 01 — Data Acquisition**
- Kaggle API setup
- Dataset download
- Raw file extraction
- Folder structure creation  
*Errors & Fixes:*
- Needed to upload and position `kaggle.json`
- Empty folder before extraction resolved

---

#### **Notebook 02 — Exploratory Data Analysis**
- Dataset overview
- Summary statistics
- Missing value inspection
- Feature distributions
- Correlation heatmap
- Target vs feature analysis  
*Errors & Fixes:*
- All numeric columns read as objects initially → fixed using correct separators and conversion
- Inconsistent formatting → cast to numeric with `errors='coerce'`

---

#### **Notebook 03 — Data Cleaning & Preprocessing**
- Numeric casting
- Duplicate removal
- Missing values handling
- Train/test splitting
- Feature scaling (StandardScaler)  
*Errors & Fixes:*
- Column name whitespace → removed via `.str.strip()`
- Colab persistence issues → preprocessing replicated in later notebooks

---

#### **Notebook 04 — Baseline Modelling**
- Linear Regression
- Random Forest Regressor  
*Errors & Fixes:*
- Missing processed files → replaced by on-the-fly preprocessing in the notebook
- Column indexing KeyError → fixed by cleaning column names

---

#### **Notebook 05 — Model Optimisation & Validation**
- Cross-validation
- GridSearch hyperparameter tuning
- Training vs testing performance
- Overfitting check  
*Highlights:*
- Tuned R² ≈ **0.9982**
- CV R² ≈ **0.9963**
- Training & testing results confirm generalisation

---

#### **Notebook 06 — Model Interpretation & Engineering Analysis**
- Feature importance
- Correlation with target
- Residual distribution
- Actual vs Predicted plot  
*Insights:*
- Top features make engineering sense (pressure & temperature)
- Low risk of leakage
- Stable residuals

---

#### **Notebook 07 — Learning Curves & Robustness Analysis**
- Learning curve evaluation
- Model complexity (R² vs n_estimators)
- Feature removal robustness test  
*Findings:*
- Training & validation curves converge → low bias and variance
- Minimal performance drop after removing top 2 features  
  ➤ R² reduced from **0.9982** to **0.9978**  
  → Model is robust and not overly reliant on single features

---

## 🛠 Python Packages Used

- `pandas`, `numpy` → data manipulation  
- `scikit-learn` → preprocessing, modelling, grid search, learning curves  
- `matplotlib`, `seaborn` → visualisation  
- `joblib` → model saving  
- `google.colab.files` → file upload

---

## 📚 Code References & Reuse

External resources consulted:
- Scikit-learn official docs: https://scikit-learn.org/
- Kaggle API guide: https://www.kaggle.com/docs/api
- Learning curve technique: sklearn.model_selection.learning_curve

All code was adapted for this project; no full external scripts were copied.

---

## 📊 Training Curves & Metrics

Training curves and complexity analyses are provided in **Notebook 07** including:

- Learning curve with R²
- Model complexity plot (effect of forest size)
- Robustness plot excluding top features

These fulfil the requirement for performance curves.

---

## ⚠ Known Limitations

- Models trained on tabular snapshots rather than time series streams.
- No domain-specific feature engineering was applied (e.g., thermodynamic derived variables).
- Hyperparameter search limited to practical ranges due to execution time.

---


---

## 🚧 Pipeline Status

### 🟢 Completed

#### **Notebook 01 — Data Acquisition**
- Kaggle API setup
- Dataset download
- Raw file extraction
- Folder structure creation  
*Errors & Fixes:*
- Needed to upload and position `kaggle.json`
- Empty folder before extraction resolved

---

#### **Notebook 02 — Exploratory Data Analysis**
- Dataset overview
- Summary statistics
- Missing value inspection
- Feature distributions
- Correlation heatmap
- Target vs feature analysis  
*Errors & Fixes:*
- All numeric columns read as objects initially → fixed using correct separators and conversion
- Inconsistent formatting → cast to numeric with `errors='coerce'`

---

#### **Notebook 03 — Data Cleaning & Preprocessing**
- Numeric casting
- Duplicate removal
- Missing values handling
- Train/test splitting
- Feature scaling (StandardScaler)  
*Errors & Fixes:*
- Column name whitespace → removed via `.str.strip()`
- Colab persistence issues → preprocessing replicated in later notebooks

---

#### **Notebook 04 — Baseline Modelling**
- Linear Regression
- Random Forest Regressor  
*Errors & Fixes:*
- Missing processed files → replaced by on-the-fly preprocessing in the notebook
- Column indexing KeyError → fixed by cleaning column names

---

#### **Notebook 05 — Model Optimisation & Validation**
- Cross-validation
- GridSearch hyperparameter tuning
- Training vs testing performance
- Overfitting check  
*Highlights:*
- Tuned R² ≈ **0.9982**
- CV R² ≈ **0.9963**
- Training & testing results confirm generalisation

---

#### **Notebook 06 — Model Interpretation & Engineering Analysis**
- Feature importance
- Correlation with target
- Residual distribution
- Actual vs Predicted plot  
*Insights:*
- Top features make engineering sense (pressure & temperature)
- Low risk of leakage
- Stable residuals

---

#### **Notebook 07 — Learning Curves & Robustness Analysis**
- Learning curve evaluation
- Model complexity (R² vs n_estimators)
- Feature removal robustness test  
*Findings:*
- Training & validation curves converge → low bias and variance
- Minimal performance drop after removing top 2 features  
  ➤ R² reduced from **0.9982** to **0.9978**  
  → Model is robust and not overly reliant on single features

---

## 🛠 Python Packages Used

- `pandas`, `numpy` → data manipulation  
- `scikit-learn` → preprocessing, modelling, grid search, learning curves  
- `matplotlib`, `seaborn` → visualisation  
- `joblib` → model saving  
- `google.colab.files` → file upload

---

## 📚 Code References & Reuse

External resources consulted:
- Scikit-learn official docs: https://scikit-learn.org/
- Kaggle API guide: https://www.kaggle.com/docs/api
- Learning curve technique: sklearn.model_selection.learning_curve

All code was adapted for this project; no full external scripts were copied.

---

## 📊 Training Curves & Metrics

Training curves and complexity analyses are provided in **Notebook 07** including:

- Learning curve with R²
- Model complexity plot (effect of forest size)
- Robustness plot excluding top features

These fulfil the requirement for performance curves.

---

## ⚠ Known Limitations

- Models trained on tabular snapshots rather than time series streams.
- No domain-specific feature engineering was applied (e.g., thermodynamic derived variables).
- Hyperparameter search limited to practical ranges due to execution time.

---


---

## 🚧 Pipeline Status

### 🟢 Completed

#### **Notebook 01 — Data Acquisition**
- Kaggle API setup
- Dataset download
- Raw file extraction
- Folder structure creation  
*Errors & Fixes:*
- Needed to upload and position `kaggle.json`
- Empty folder before extraction resolved

---

#### **Notebook 02 — Exploratory Data Analysis**
- Dataset overview
- Summary statistics
- Missing value inspection
- Feature distributions
- Correlation heatmap
- Target vs feature analysis  
*Errors & Fixes:*
- All numeric columns read as objects initially → fixed using correct separators and conversion
- Inconsistent formatting → cast to numeric with `errors='coerce'`

---

#### **Notebook 03 — Data Cleaning & Preprocessing**
- Numeric casting
- Duplicate removal
- Missing values handling
- Train/test splitting
- Feature scaling (StandardScaler)  
*Errors & Fixes:*
- Column name whitespace → removed via `.str.strip()`
- Colab persistence issues → preprocessing replicated in later notebooks

---

#### **Notebook 04 — Baseline Modelling**
- Linear Regression
- Random Forest Regressor  
*Errors & Fixes:*
- Missing processed files → replaced by on-the-fly preprocessing in the notebook
- Column indexing KeyError → fixed by cleaning column names

---

#### **Notebook 05 — Model Optimisation & Validation**
- Cross-validation
- GridSearch hyperparameter tuning
- Training vs testing performance
- Overfitting check  
*Highlights:*
- Tuned R² ≈ **0.9982**
- CV R² ≈ **0.9963**
- Training & testing results confirm generalisation

---

#### **Notebook 06 — Model Interpretation & Engineering Analysis**
- Feature importance
- Correlation with target
- Residual distribution
- Actual vs Predicted plot  
*Insights:*
- Top features make engineering sense (pressure & temperature)
- Low risk of leakage
- Stable residuals

---

#### **Notebook 07 — Learning Curves & Robustness Analysis**
- Learning curve evaluation
- Model complexity (R² vs n_estimators)
- Feature removal robustness test  
*Findings:*
- Training & validation curves converge → low bias and variance
- Minimal performance drop after removing top 2 features  
  ➤ R² reduced from **0.9982** to **0.9978**  
  → Model is robust and not overly reliant on single features

---

## 🛠 Python Packages Used

- `pandas`, `numpy` → data manipulation  
- `scikit-learn` → preprocessing, modelling, grid search, learning curves  
- `matplotlib`, `seaborn` → visualisation  
- `joblib` → model saving  
- `google.colab.files` → file upload

---

## 📚 Code References & Reuse

External resources consulted:
- Scikit-learn official docs: https://scikit-learn.org/
- Kaggle API guide: https://www.kaggle.com/docs/api
- Learning curve technique: sklearn.model_selection.learning_curve

All code was adapted for this project; no full external scripts were copied.

---

## 📊 Training Curves & Metrics

Training curves and complexity analyses are provided in **Notebook 07** including:

- Learning curve with R²
- Model complexity plot (effect of forest size)
- Robustness plot excluding top features

These fulfil the requirement for performance curves.

---

## ⚠ Known Limitations

- Models trained on tabular snapshots rather than time series streams.
- No domain-specific feature engineering was applied (e.g., thermodynamic derived variables).
- Hyperparameter search limited to practical ranges due to execution time.

---

## 📌 Project Status & Academic Context

This repository documents the complete development of a machine learning pipeline for predicting marine gas turbine degradation using condition-based monitoring data.

The project demonstrates:

- A structured end-to-end ML workflow
- Reproducible preprocessing and model training
- Hyperparameter optimisation with cross-validation
- Performance evaluation using R² metrics
- Learning curve analysis to assess bias and variance
- Robustness testing via feature removal

The focus of this work is numerical modelling and regression analysis rather than deep learning or image-based approaches, in alignment with supervisory guidance.

All modelling decisions, encountered errors, and corrective steps are documented within the respective notebooks to ensure transparency and reproducibility.

---

## 📍 Author

Meilad Rahmani — Marine Engineering MSc  
2026

---

## 📍 Author

Meilad Rahmani — Marine Engineering MSc  
2026
---

## 📍 Author

Meilad Rahmani — Marine Engineering MSc  
2026
---

## 📍 Author

Meilad Rahmani — Mechanical Engineering 
2026







 
