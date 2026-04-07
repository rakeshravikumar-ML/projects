# Telecom Customer Churn Prediction

An end-to-end applied machine learning project for predicting telecom customer churn using structured customer data.  
This project demonstrates a practical ML workflow including data cleaning, exploratory data analysis (EDA), preprocessing, class imbalance handling, model training, hyperparameter tuning, evaluation, and a simple Streamlit-based prediction interface.

---

## Project Overview

Customer churn prediction is a common business problem where the goal is to identify customers who are likely to stop using a service.  
This project builds a binary classification pipeline to support churn risk prediction using telecom customer account and service-related features.

### Key Highlights
- End-to-end supervised machine learning workflow
- Exploratory data analysis and feature understanding
- Class imbalance handling using **SMOTE / SMOTEENN**
- Model training and evaluation using **Scikit-learn**
- Hyperparameter optimization with **Optuna**
- Lightweight **Streamlit** app for interactive predictions

---

## Repository Structure

    telecom-customer-churn-prediction/
    │
    ├── app/
    │   └── streamlit_app.py
    │
    ├── data/
    │   └── customer_churn_sample.csv
    │
    ├── docs/
    │   └── deployment_notes.md
    │
    ├── models/
    │   └── ada_boost_churn_model.pkl
    │
    ├── notebooks/
    │   ├── 01_churn_eda.ipynb
    │   └── 02_model_building.ipynb
    │
    └── README.md

---

## Workflow Summary

### 1. Exploratory Data Analysis
- Reviewed dataset structure and feature types
- Checked target distribution for churn imbalance
- Explored patterns across customer attributes and service usage
- Identified preprocessing needs and feature preparation steps

### 2. Data Preparation
- Cleaned and prepared structured tabular data
- Applied feature engineering / encoding as needed
- Addressed class imbalance using resampling techniques such as **SMOTE** and **SMOTEENN**

### 3. Model Development
- Trained classification models for churn prediction
- Evaluated performance using standard classification metrics
- Used **Optuna** for hyperparameter tuning on selected models

### 4. Inference Interface
- Built a simple **Streamlit** application to demonstrate business-facing churn predictions

---

## Tools & Technologies

- **Python**
- **Pandas / NumPy**
- **Scikit-learn**
- **Matplotlib / Seaborn** *(if used in notebooks)*
- **SMOTE / SMOTEENN**
- **Optuna**
- **Streamlit**

---

## Files Included

### Notebooks
- **01_churn_eda.ipynb**  
  Exploratory data analysis, target review, feature understanding, and early observations.

- **02_model_building.ipynb**  
  Data preprocessing, model training, resampling, tuning, evaluation, and model selection.

### App
- **streamlit_app.py**  
  Streamlit interface for running churn predictions using the saved trained model.

### Model
- **ada_boost_churn_model.pkl**  
  Serialized trained model used by the Streamlit app.

### Data
- **customer_churn_sample.csv**  
  Sample dataset included for demonstration and reproducibility.

### Docs
- **deployment_notes.md**  
  Optional notes related to deployment experiments or environment setup.

---

## How to Run the Streamlit App

1. Clone the repository
2. Navigate to this project folder
3. Install dependencies
4. Run the Streamlit app

    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

> Note: Ensure the model file path in the Streamlit app matches the current folder structure.

---

## Project Purpose

This project is part of my applied machine learning portfolio and is designed to demonstrate:

- Practical business-oriented ML problem solving
- Structured experimentation and model evaluation
- Handling imbalanced classification problems
- Converting notebook work into a lightweight user-facing application

---

## Notes

- This project is intended as a portfolio demonstration of an end-to-end applied ML workflow.
- The included app is designed for demonstration and recruiter review.
- Some deployment approaches were explored during development, but this repository is primarily focused on the reproducible project workflow and code.

---

## Author

**Rakesh Ravikumar**  
Junior Machine Learning / AI Systems Candidate
