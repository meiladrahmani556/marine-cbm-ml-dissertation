# 🛠 Marine Gas Turbine Condition-Based Monitoring (CBM)
## Machine Learning — BEng Final Year Project

**Student:** Meilad Rahmani | **ID:** UP2071176
**Course:** BEng (Hons) Mechanical Engineering | **Course Code:** M30031
**Supervisor:** Dr Sergey Yakalov | **University:** University of Portsmouth | **Year:** 2025–26

---
---

## 🚀 Quick Start

**To run any notebook:**
1. Go to the `JupyterNotebook/` folder in this repo
2. Click the notebook you want to open
3. Click **Open in Colab** at the top of the notebook
4. Click **Runtime → Run all**
5. Upload `Conditional_Base_Monitoring in Marine_System.csv` when prompted
6. All cells will run automatically from top to bottom

**Dataset download:**
- Download the CSV from [Kaggle](https://www.kaggle.com/datasets/kunalnehete/condition-based-monitoring-cbm-in-marine-system)
- Filename: `Conditional_Base_Monitoring in Marine_System.csv`

**Note:** Notebook 01 requires a `kaggle.json` API key for automated download. All other notebooks only need the CSV file.

---


## 📌 Project Overview

This project develops a complete machine learning pipeline to predict gas turbine compressor degradation in marine propulsion systems using the Condition-Based Monitoring (CBM) dataset from Kaggle.

The focus is on **numerical regression modelling** using tabular sensor data — in alignment with supervisory guidance to pursue a numerical ML approach rather than image-based or CNN methods.

The pipeline covers:
1. Dataset acquisition
2. Exploratory data analysis (EDA)
3. Dataset cleaning and preprocessing
4. Train / validation / test splitting
5. Baseline modelling (Linear Regression)
6. Advanced modelling and optimisation (Random Forest, XGBoost)
7. Model performance assessment — training curves and metrics
8. Model interpretation and engineering analysis
9. Robustness testing

The objective is predictive accuracy, interpretability, and engineering insight into which sensor readings drive compressor degradation.

**Project status:** All 8 notebooks complete and verified one-click executable. Final dissertation report in progress — submission deadline 13 May 2026.

---

## 📊 Dataset Description

**Source:** Kaggle — Condition-Based Monitoring in Marine System
**URL:** https://www.kaggle.com/datasets/kunalnehete/condition-based-monitoring-cbm-in-marine-system

The dataset contains approximately **11,900 observations** of sensor measurements recorded from a simulated naval gas turbine propulsion system under various operational conditions.

**Features include:**
- GT shaft torque (GTT) [kN m]
- GT rate of revolutions (GTn) [rpm]
- Gas Generator rate of revolutions (GGn) [rpm]
- Starboard and Port Propeller Torque [kN]
- HP Turbine exit temperature (T48) [°C]
- GT Compressor inlet and outlet air temperature (T1, T2) [°C]
- HP Turbine exit pressure (P48) [bar]
- GT Compressor inlet and outlet air pressure (P1, P2) [bar]
- GT exhaust gas pressure (Pexh) [bar]
- Turbine Injection Control (TIC) [%]
- Fuel flow (mf) [kg/s]

**Target variable:**
> ➡ **GT Compressor decay state coefficient** — a continuous value between 0.95 and 1.00 representing the degree of compressor degradation.

**Acknowledged limitation:** The dataset is simulation-based rather than drawn from a live operational system. Generalisation to real naval vessels requires further validation with live sensor data.

---

## 📥 Dataset Acquisition

Dataset download is handled in **Notebook 01** using the Kaggle API.

**Steps to reproduce:**
1. Open Notebook 01 in Google Colab
2. Upload your `kaggle.json` API key when prompted
3. The notebook authenticates the Kaggle CLI automatically
4. Downloads and extracts the raw CSV to `data/raw/`

**Errors encountered and resolved:**
- Kaggle API not configured → fixed by uploading and correctly placing `kaggle.json`
- Empty `data/raw/` folder → resolved with correct extraction and file move logic
- Column names read with whitespace → fixed with `.str.strip()`

---

## 📁 Project Structure

```
marine-cbm-ml-dissertation/
│
├── JupyterNotebook/
│   ├── 01_dataset_acquisition.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_cleaning___preprocessing.ipynb
│   ├── 04_baseline_model_development.ipynb
│   ├── 05_model_optimisation___validation.ipynb
│   ├── 06_model_interpretation___engineering_analysis.ipynb
│   ├── 07_learning_curves___robustness_analysis.ipynb
│   └── 08_XGBoost_Model.ipynb
│
├── data/
│   ├── raw/         # Raw dataset (downloaded via Kaggle API)
│   └── processed/   # Processed ML inputs (generated in notebooks)
│
├── models/          # Saved model artifacts (.pkl files)
├── requirements.txt # Required Python libraries
└── README.md        # This file
```

All notebooks are **one-click executable** on Google Colab. Each notebook begins by mounting Google Drive and loading the dataset directly — no dependency on output from previous notebooks.

**How to run any notebook:**
1. Open the notebook in Google Colab
2. Run all cells from top to bottom
3. Upload the CSV file (`Conditional_Base_Monitoring in Marine_System.csv`) when the upload dialog appears
4. All preprocessing, training, and evaluation runs automatically

---

## 🚧 Pipeline — Notebook by Notebook

### ✅ Notebook 01 — Data Acquisition
- Kaggle API authentication and setup
- Dataset download via Kaggle CLI
- Raw CSV extraction
- Folder structure creation

*Errors & Fixes:*
- Required upload and correct placement of `kaggle.json`
- Empty folder before extraction resolved with explicit file move logic

---

### ✅ Notebook 02 — Exploratory Data Analysis (EDA)
- Dataset overview and shape inspection
- Summary statistics for all features
- Missing value inspection
- Feature distribution plots (histograms)
- Correlation heatmap across all sensor variables
- Target variable distribution and range analysis

*Errors & Fixes:*
- All numeric columns initially read as objects → fixed using correct CSV separator (`;`) and decimal (`,`) settings
- Inconsistent formatting → cast to numeric with `pd.to_numeric(errors='coerce')`

---

### ✅ Notebook 03 — Data Cleaning & Preprocessing
- Column name whitespace removal
- Numeric type casting for all columns
- Duplicate row removal
- Missing value handling
- Feature scaling using `StandardScaler`
- Train / test split (80/20, `random_state=42`)

*Errors & Fixes:*
- Column name whitespace caused KeyErrors → removed via `.str.strip()`
- Google Colab file persistence issues → preprocessing steps replicated at the top of each subsequent notebook to ensure one-click executability

---

### ✅ Notebook 04 — Baseline Model Development
- Linear Regression as the performance baseline
- Random Forest Regressor initial implementation
- Evaluation using R², MAE, and RMSE on the test set
- Actual vs Predicted plots for both models

*Errors & Fixes:*
- Missing processed files from previous sessions → resolved by reproducing preprocessing in-notebook
- Column indexing KeyError → fixed by stripping column name whitespace before splitting

**Baseline Results:**

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.9014 | 0.00335 | 0.00468 |
| Random Forest (default) | 0.994 | 0.0007 | 0.0010 |

---

### ✅ Notebook 05 — Model Optimisation & Validation
- GridSearchCV hyperparameter tuning for Random Forest
- 5-fold cross-validation
- Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- Training vs testing performance comparison
- Overfitting check

**Optimised Random Forest Results:**

| Metric | Score |
|---|---|
| Test R² | **0.9982** |
| Cross-Validation R² | **0.9963** |
| MAE | 0.000399 |
| RMSE | 0.000633 |

---

### ✅ Notebook 06 — Model Interpretation & Engineering Analysis
- Feature importance ranking from Random Forest
- Correlation of features with the target variable
- Residual distribution plot
- Actual vs Predicted scatter plot

*Key findings:*
- Top features (compressor outlet temperature T2, HP turbine exit pressure P48) make strong engineering sense
- Stable, symmetric residual distribution — no systematic bias
- Low risk of data leakage

---

### ✅ Notebook 07 — Learning Curves & Robustness Analysis
- Learning curve — R² vs training set size
- Model complexity analysis — R² vs number of estimators
- Feature removal robustness test (removing top 2 features)

*Key findings:*
- Training and validation curves converge → low bias, low variance, no significant overfitting
- After removing top 2 features: R² reduced from **0.9982 → 0.9978**
  → Model is robust and not over-reliant on individual sensor readings

---

### ✅ Notebook 08 — XGBoost Model & Full Model Comparison
- XGBoost Regressor implemented as a third model
- GridSearchCV hyperparameter tuning for XGBoost
  - Parameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
  - Best parameters found: `colsample_bytree=0.8`, `learning_rate=0.05`, `max_depth=7`, `n_estimators=300`, `subsample=0.8`
  - Best cross-validation R²: **0.9967**
- 5-fold cross-validation — mean R² = **0.9967** (scores: 0.9966, 0.9969, 0.9967, 0.9964, 0.9967)
- Residual distribution and Actual vs Predicted plots
- Feature importance analysis for XGBoost
- Learning curve for XGBoost
- **Full three-model comparison** — Linear Regression vs Random Forest vs XGBoost

**Full Model Comparison — Test Set Results:**

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.9014 | 0.00335 | 0.00468 |
| Random Forest (tuned) ⭐ | **0.9982** | **0.000399** | **0.000633** |
| XGBoost (tuned) | 0.9971 | 0.000603 | 0.000805 |

**Random Forest is selected as the primary model** based on superior test set performance and validated generalisation through cross-validation.

---

## 📈 Model Performance & Training Curves

Training curves are provided in **Notebooks 07 and 08** and include:

- Learning curve (R² vs training set size) for all three models
- Model complexity plot (effect of number of estimators on Random Forest)
- Robustness plot excluding top features
- Actual vs Predicted scatter plots for all models
- Residual distribution plots for all models

All three models are compared side by side to demonstrate iterative model development and justify the final model selection.

---

## 🛠 Python Packages Used

| Package | Role |
|---|---|
| `pandas` | Data loading, manipulation, and cleaning |
| `numpy` | Numerical operations and array handling |
| `scikit-learn` | Preprocessing, modelling, GridSearchCV, cross-validation, learning curves |
| `xgboost` | XGBoost regression model |
| `matplotlib` | Plotting and visualisation |
| `seaborn` | Statistical visualisation (heatmaps, distributions) |
| `joblib` | Saving and loading trained model files |
| `google.colab.drive` | Google Drive mounting for file persistence |
| `google.colab.files` | File upload (Kaggle API key) |

---

## 📚 Code References & Reuse

External resources consulted during development:

- **Scikit-learn documentation** — preprocessing, model selection, learning curves: https://scikit-learn.org/
- **XGBoost documentation** — XGBRegressor API and hyperparameters: https://xgboost.readthedocs.io/
- **Kaggle API guide** — dataset download and authentication: https://www.kaggle.com/docs/api
- **Learning curve technique** — `sklearn.model_selection.learning_curve`
- **GridSearchCV** — `sklearn.model_selection.GridSearchCV`

All code was written and adapted specifically for this project. No external scripts were copied wholesale. All sources are referenced inline in the relevant notebooks.

---

## ⚠ Known Limitations & Bugs

- Models are trained on tabular sensor snapshots rather than time-series data streams — temporal degradation patterns are not captured
- No domain-specific thermodynamic feature engineering has been applied
- Hyperparameter search is limited to practical ranges due to Google Colab execution time constraints
- Dataset is simulation-based — results may not directly generalise to live naval gas turbines without further validation
- Google Colab sessions reset between runs — all notebooks reproduce preprocessing steps at the top to compensate
- All notebooks have been verified one-click executable on a fresh Colab runtime as of May 2026

**Known bugs:** No unresolved bugs at time of submission. All previously encountered errors (Kaggle API, column name whitespace, file persistence) have been resolved and documented within the relevant notebooks.

---

## 📌 Conclusions

- **Random Forest achieves R² = 0.9982** on the test set — compressor degradation can be predicted with very high accuracy from sensor data alone
- **Performance hierarchy confirmed:** Random Forest (0.9982) > XGBoost (0.9971) > Linear Regression (0.9014)
- **Key diagnostic features:** Compressor outlet temperature (T2) and HP Turbine exit pressure (P48) are the dominant predictors — consistent with thermodynamic engineering expectations
- **XGBoost best parameters:** `colsample_bytree=0.8`, `learning_rate=0.05`, `max_depth=7`, `n_estimators=300`, `subsample=0.8`
- The ML pipeline is robust, generalisable, and interpretable — providing actionable insight for condition-based maintenance in marine engineering

---

## 🔭 Future Work

- Validate the model on real operational sensor data from live naval gas turbines
- Explore LSTM and time-series sequence models to capture temporal degradation patterns
- Apply thermodynamic feature engineering to improve physical interpretability
- Extend the pipeline to simultaneously predict turbine and compressor health
- Deploy the best model as a **Streamlit web app** for real-time degradation prediction from user-input sensor readings

---

## 📍 Author

**Meilad Rahmani**
BEng (Hons) Mechanical Engineering | UP2071176 | M30031
University of Portsmouth | 2025–26
Supervisor: Dr Sergey Yakalov
