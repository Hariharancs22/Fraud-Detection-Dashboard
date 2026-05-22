# Real-Time Fraud Detection System with Explainable AI & Live Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://fraud-detection-dashboard-hariharancs.streamlit.app/)

**Live Interactive Dashboard:** [Click here to view the Streamlit App](https://fraud-detection-dashboard-hariharancs.streamlit.app/) or 
https://fraud-detection-dashboard-hariharancs.streamlit.app/

Executive Summary
An end-to-end machine learning and data engineering pipeline designed to identify, segment, and explain fraudulent financial transactions in real time. Facing a severe **96.5% to 3.5% class imbalance**, this architecture leverages **SMOTE** for data synthesis, an advanced **XGBoost Classifier** optimized via 5-fold cross-validation, and **SHAP (SHapley Additive exPlanations)** to provide transparent, case-by-case model interpretability. 

The final system shifts business operations from reactive damage control to proactive prevention, delivering a **58.80% fraud dollar recovery rate** and an estimated **$10.29 Million in annual savings** if scaled to a mid-sized financial institution.

---

📊 Quick Performance & Insights Glance

1. The Modeling Journey
We systematically pushed the model's boundaries through rigorous hyperparameter tuning and custom threshold calibration, maximizing our F1-score.
*(Note: View the `model_comparison.png` file in this repository to see the performance chart!)*

2. Global Explanations (Why the Model Flags Fraud)
Using SHAP values, we broke open the "black box" of XGBoost to map exactly which behavioral markers signal a fraudulent attack vector.
*(Note: View the `shap_summary.png` file in this repository to see the feature importance!)*

---

📁 Repository Structure
* `analysis.ipynb` - Core data science pipeline (Data processing, SMOTE, Model tuning)
* `dashboard/` - Contains the live Streamlit application (`app.py`), the pickled XGBoost model, and dashboard data.
* `requirements.txt` - Python deployment dependencies.
* `README.md` - Project documentation.
* `Project_Summary.docx` - The complete business report.
