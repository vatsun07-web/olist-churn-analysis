# Olist Customer Retention Risk Analysis

**Portfolio Project 4 | CRISP-DM | Binary Churn Classification**
🔴 Live Dashboard: https://vatsun07-web-olist-churn-analysis.streamlit.app

## Business Problem
Build a churn classifier to identify Olist customers unlikely to make a repeat purchase within 90 days, enabling proactive retention targeting.

## Dataset
Brazilian E-Commerce Public Dataset by Olist — 9 tables, ~1.4M rows.

## Pipeline
| Notebook | Phase | Description |
|---|---|---|
| NB00 | Business Understanding | Problem framing, success criteria |
| NB01 | Data Understanding | Profiling all 9 tables |
| NB02 | Data Preparation | Joins, churn label, master table |
| NB03 | EDA | Churn drivers, distributions |
| NB04 | Feature Engineering | 27 engineered features |
| NB05 | Modelling | LR, RF, LightGBM + Optuna tuning |
| NB06 | Evaluation | SHAP, risk segmentation, threshold analysis |

## Key Results
- **Best model:** Tuned Logistic Regression — ROC-AUC 0.626
- **Top churn driver:** Product category (mean |SHAP| = 0.317)
- **Churn rate:** 97.82% — structural ceiling explained in NB06
- **Class imbalance handled** via class_weight='balanced'

## Stack
Python · pandas · scikit-learn · LightGBM · Optuna · SHAP · Parquet
