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

```text
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
