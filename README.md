# 🚢 Marine Condition-Based Monitoring (CBM)
## Predictive Modelling of Marine Gas Turbine Degradation Using Machine Learning

---

## 📌 Project Overview

This project develops and evaluates multiple machine learning models to predict the **degradation coefficient** of a marine gas turbine propulsion system using multivariate numerical sensor data.

The objective is to design and document a complete end-to-end machine learning pipeline, including:

- Dataset acquisition  
- Exploratory Data Analysis (EDA)  
- Data cleaning and preprocessing  
- Train / validation / test splitting  
- Model development and optimisation  
- Performance comparison and evaluation  

This repository serves as the technical implementation of the dissertation project.

# 🔎 Engineering Understanding

Marine propulsion systems operate under complex thermodynamic and mechanical conditions. Monitoring degradation is critical for:

- Preventive maintenance  
- Failure avoidance  
- Operational efficiency  
- Cost reduction  

Condition-Based Monitoring (CBM) enables maintenance decisions based on system condition rather than fixed schedules. Accurate prediction of degradation supports more efficient and reliable marine operations.

# 📊 Dataset Information

**Dataset Name:** Condition-Based Monitoring (CBM) in Marine System  
**Source:** Kaggle  
**Dataset URL:**  
https://www.kaggle.com/datasets/kunalnehete/condition-based-monitoring-cbm-in-marine-system  

## Dataset Description

The dataset contains multivariate numerical sensor measurements collected from a marine gas turbine system operating under varying conditions.

Each row represents a system state described by multiple operational features.

### 🎯 Target Variable

- `Degradation Coefficient` (continuous regression target)

### 📥 Example Dataset Preview

| Sensor_1 | Sensor_2 | Sensor_3 | ... | Degradation_Coefficient |
|----------|----------|----------|-----|--------------------------|
| 12.45    | 450.32   | 0.0034   | ... | 0.987                    |
| 11.98    | 447.10   | 0.0029   | ... | 0.973                    |
| 13.12    | 452.78   | 0.0038   | ... | 0.991                    |
| 12.67    | 449.33   | 0.0031   | ... | 0.982                    |
| 11.55    | 444.21   | 0.0027   | ... | 0.965                    |

*(Replace with actual dataset preview from your notebook.)*

# ⚙️ Dataset Acquisition

The dataset is downloaded using the Kaggle API.

#Step 1 - Install Kaggle

pip install kaggle 

#step 2 - Add API Key 

Place kaggle.json inside:
~/.kaggle/

#Step 3 – Download Dataset

kaggle datasets download -d kunalnehete/condition-based-monitoring-cbm-in-marine-system

# Step 4 – Extract Files 

Extract contents into:
data/raw/

---

# 📈 Exploratory Data Analysis (EDA)

EDA was conducted to understand:

- Feature distributions  
- Correlations between variables  
- Multicollinearity  
- Outliers  
- Feature–target relationships  

Visualisations include:

- Histograms  
- Correlation heatmap  
- Boxplots  
- Target distribution plots  

Notebook:

`02_eda.ipynb`

# 🧹 Data Cleaning & Preprocessing

Preprocessing steps include:

- Handling missing values  
- Feature scaling (StandardScaler)  
- Outlier inspection  
- Data normalisation for neural networks  
- Saving processed datasets  

Processed data is stored in:

Data/processed/

Notebook:

`03_preprocessing_and_split.ipynb`

# 🔀 Train / Validation / Test Split

The dataset was divided into:

- 70% Training  
- 15% Validation  
- 15% Test  

Measures taken:

- Fixed random seed for reproducibility  
- No data leakage  
- Scalers fitted only on training data  

# 🤖 Model Development

Three progressively optimised models were developed and compared.

## 🔹 Model 2 – Random Forest Regressor

Purpose:

- Capture nonlinear feature interactions  
- Improve predictive accuracy  

Hyperparameters tuned:

- Number of estimators  
- Maximum depth  
- Minimum samples split  

Notebook:

`05_model_2_random_forest.ipynb`

## 🔹 Model 3 – Neural Network (MLP Regressor)

Architecture:

- Fully connected dense layers  
- ReLU activation  
- Dropout regularisation  
- Adam optimiser  

Optimisation techniques:

- Hyperparameter tuning  
- Learning rate scheduling  
- Early stopping  
- Architecture experimentation  

Notebook:

`06_model_3_neural_network.ipynb`

Training curves are saved in:

# 📊 Model Performance Comparison

| Model               | MSE    | RMSE   | R² Score |
|---------------------|--------|--------|----------|
| Linear Regression   | 0.0025 | 0.050  | 0.82     |
| Random Forest       | 0.0014 | 0.037  | 0.91     |
| Neural Network      | 0.0011 | 0.033  | 0.94     |

*(Replace example values with final results.)*

# 📉 Model Evaluation

Final evaluation includes:

- Test set performance  
- Residual analysis  
- Prediction vs Actual plots  
- Random sample prediction  

Notebook:

`07_model_evaluation.ipynb`

# 📦 Python Packages Used

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- tensorflow / keras  
- scipy  
- kaggle  

All dependencies are listed in:

`requirements.txt`

# 🏗 Repository Structure

marine-cbm-ml-dissertation/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
├── models/
├── results/
├── requirements.txt
└── README.md

# 📌 Conclusions

This project demonstrates:

- A complete end-to-end machine learning workflow  
- Comparative model analysis  
- Iterative optimisation  
- Application of machine learning to marine engineering systems  

The optimised neural network achieved the strongest generalisation performance.

---

# 🚀 Future Work

- Time-series modelling using LSTM  
- Advanced hyperparameter optimisation  
- Feature importance analysis using SHAP  
- Real-time monitoring simulation  
- Streamlit deployment for interactive demonstration  
