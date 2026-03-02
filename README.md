# 🛠 Marine Gas Turbine Condition-Based Monitoring (CBM)  
## Machine Learning Dissertation Project

---

## 📌 Project Overview

This project implements a complete machine learning pipeline to predict gas turbine degradation in a marine propulsion system using the **Condition-Based Monitoring (CBM) dataset**.

The pipeline follows standard practices in engineering analytics and predictive maintenance:

1. Data acquisition  
2. Exploratory data analysis (EDA)  
3. Data cleaning and preprocessing  
4. Baseline modelling  
5. Model optimisation and validation  
6. Model interpretation and engineering analysis

The goal is to build and rigorously evaluate regression models that can accurately estimate turbine decay behaviour, with strong reproducibility and academic accountability.

---

## 🗂 Repository Structure
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


---

## 📌 Pipeline Status

The following notebooks have been completed, with documented errors and resolution strategies:

---

### 🟢 Notebook 01 — Data Acquisition  
**Purpose:** Download and prepare the marine CBM dataset.

**Challenges/Errors:**
- **Kaggle CLI not configured in Colab:** Required uploading `kaggle.json` and setting permissions.
- **No CSV after Kaggle download:** Occurred due to missing or incorrectly placed API credentials. Fixed by proper API key upload and CLI setup.
- **Empty `data/raw` folder:** Resolved by verifying extracted files and ensuring the dataset was downloaded and unzipped in Colab.

---

### 🟢 Notebook 02 — Exploratory Data Analysis (EDA)  
**Purpose:** Understand dataset structure and patterns.

**Challenges/Errors:**
- **All columns read as objects:** Caused by invalid CSV parsing due to formatting. Resolved by correctly handling separators and parsing numeric formats.
- **Data types incorrect after load:** Solved by converting columns to numeric using `errors='coerce'`.
  
EDA steps included:
- Statistical summaries
- Missing value inspection
- Correlation heatmap
- Individual feature vs target scatter plots

---

### 🟢 Notebook 03 — Data Cleaning & Preprocessing  
**Purpose:** Transform raw data into model-ready datasets.

**Challenges/Errors:**
- **Column names contained whitespace/hidden characters:** Fix applied using `.str.strip()` to clean headers.
- **No saved processed files in Notebook 04 due to Colab runtime reset:** Led to moving split logic into each notebook for reproducibility.

Cleaning steps included:
- Numeric conversion
- Duplicate removal
- Missing value handling
- Train/test split
- Feature scaling with StandardScaler

---

### 🟢 Notebook 04 — Baseline Model Development  
**Purpose:** Train baseline regression models.

**Models:**
- Linear Regression
- Random Forest Regressor

**Challenges/Errors:**
- **Missing processed CSVs in Notebook 04:** Resolved by recreating preprocessing steps directly in the notebook to ensure independence from Notebook 03.
- **KeyError on dropping target column:** Caused by trailing whitespace in column names. Fixed using `.str.strip()`.

Evaluation included:
- MAE, RMSE, R² metrics
- Residual analysis
- Feature importance estimation
- Engineering interpretation of model behaviour

---

### 🟢 Notebook 05 — Model Optimisation & Validation  
**Purpose:** Improve model performance and robustness.

Key steps:
- Cross-validation (5-fold R²)
- Hyperparameter tuning via GridSearch
- Comparison between baseline and tuned RF
- Training vs test performance analysis
- Overfitting assessment

**Results:**
- Baseline R²: ~0.9982  
- Tuned R²: ~0.9982  
- Best CV R²: ~0.9963  
- Training vs test R² demonstrates good generalisation

---

### 🟢 Notebook 06 — Model Interpretation & Engineering Analysis  
**Purpose:** Analyse model behaviour and feature influence.

Insights included:
- Top feature importance values
- Correlation with target analysis
- Residual distribution
- Actual vs predicted scatter plot
- Engineering interpretation of results

**Feature importance highlights:**
1. GT Compressor outlet air temperature (T2)  
2. HP Turbine exit pressure (P48)  
3. GT exhaust gas pressure  
4. GT shaft torque (GTT)  
5. Gas Generator revolutions (GGn)

These features make *physical sense* as predictors of degradation state.

---

## 🧪 Dataset Acquisition

The dataset is sourced from Kaggle:
- **Condition-Based Monitoring in Marine System**  
- URL: https://www.kaggle.com/datasets/kunalnehete/condition-based-monitoring-cbm-in-marine-system

Download is performed via Kaggle API in Notebook 01.

---

## 🧠 Modelling Summary

### Baseline Models
- Linear Regression – interpretable baseline
- Random Forest – captures non-linear dynamics

### Optimised Models
- Random Forest with tuned hyperparameters
- Grid search used for systematic optimisation

### Evaluation Metrics
- **R²:** Goodness of fit
- **RMSE:** Error magnitude
- **Cross-Validation:** Model stability
- **Residual Analysis:** Error distribution

---

## 🛠 Tools & Libraries

- Python 3.x  
- Pandas  
- NumPy  
- Scikit-Learn  
- Matplotlib  
- Seaborn  
- Jupyter Notebook / Google Colab

---

## ▶ Running the Project

1. Clone repository  
2. Install dependencies:


pip install -r requirements.txt
`bash
3. Run notebook in order:

01 → 02 → 03 → 04 → 05 → 06
Note: Some notebooks require Kaggle API credentials.

## 📌 Notes
- Colab session storage is temporary → mounting Google Drive is recommended for persistence.
- All notebooks are designed to be one-click executable from start to end.

## 📍 Future Work
- Test with additional models (GBR/LightGBM/XGBoost)
- Deploy predictive interface (Streamlit/Flask)
- Extended robustness testing (feature removal / subset analysis)

## 📍 Author

Meilad Rahmani
2026










 
