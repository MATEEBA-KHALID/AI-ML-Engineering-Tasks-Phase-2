# Customer Churn Prediction Pipeline

## Objective
Build a production-ready machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset.

## Dataset
Telco Customer Churn Dataset

## Features
- Data preprocessing with Pipeline API
- Missing value handling
- Feature scaling
- One-Hot Encoding
- Logistic Regression
- Random Forest
- Hyperparameter tuning using GridSearchCV
- Model evaluation
- Export pipeline using Joblib

## Models Used
1. Logistic Regression
2. Random Forest Classifier

## Evaluation Metrics
- Accuracy
- F1 Score
- Confusion Matrix
- Classification Report

## Run

```bash
pip install -r requirements.txt
python churn_pipeline.py
```

## Output

```text
customer_churn_pipeline.pkl
```

Reusable trained pipeline for future predictions.
