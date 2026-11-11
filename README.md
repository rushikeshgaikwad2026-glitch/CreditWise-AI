# CreditWise AI – Intelligent Loan Approval & Credit Risk Assessment

CreditWise AI is an interactive machine learning decision-support application designed to estimate loan default risk and support data-driven credit portfolio analysis.

The application demonstrates machine learning, business analytics, risk assessment, explainable AI, and interactive dashboard development.

## Key Features

1. **Executive Dashboard** – Overview of the credit portfolio, default trends, and key risk indicators.
2. **Applicant Risk Assessment** – Estimate an applicant's probability of default using a calibrated machine learning model.
3. **Portfolio Analytics** – Explore portfolio composition, loan characteristics, and risk patterns.
4. **Explainable AI** – Explore model explanations and factors contributing to risk estimates.
5. **Batch Assessment** – Assess multiple applicant records in one workflow.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- Joblib
- Pillow

## Machine Learning Approach

The application uses a calibrated Random Forest classification model trained on a credit risk dataset.

The deployed candidate model excludes `loan_grade` to avoid relying on a potentially problematic underwriting-related feature.

The model estimates the probability of loan default. It does not make an actual lending approval or rejection decision.

## Dataset

Source: [Credit Risk Dataset – Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

The dataset contains historical loan and borrower information. Monetary values are represented in USD.

## Run Locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Run the application:

   `streamlit run app.py`

## Model and Data

The application requires the calibrated model stored at:

`models/credit_risk_rf_optimized_calibrated_no_loan_grade.joblib`

It also uses the processed portfolio dataset:

`data/processed/credit_risk_cleaned.csv`

## Important Limitations and Responsible Use

- This is an educational and portfolio demonstration project, not a production lending system.
- Estimated default probability is not a guarantee of future borrower behaviour.
- Model outputs should not be used as the sole basis for lending decisions.
- Historical data may contain bias and may not represent future applicants.
- SHAP feature attributions describe model behaviour and do not establish causation.
- Descriptive subgroup analysis is not a formal legal fairness assessment.
- The deployed model and explanatory visualizations may represent different stages of the prediction pipeline.

## Project Purpose

This project demonstrates the practical application of machine learning and analytics to a business problem, translating credit risk data into interpretable insights through an interactive application.
