# 💳 CreditWise AI
### Intelligent Loan Default Risk Assessment & Credit Portfolio Analytics

<p align="center">
  <strong>Turning Credit Data into Smarter Risk Insights.</strong>
</p>

<p align="center">
  An interactive Machine Learning application that estimates loan default probability, evaluates applicant risk, and transforms credit portfolio data into meaningful business insights.
</p>

<p align="center">
  ![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
  ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-2E8B57?style=for-the-badge)
  ![Status](https://img.shields.io/badge/Status-Portfolio%20Project-6C63FF?style=for-the-badge)
</p>

---

## 🎯 Project Overview

Financial institutions process thousands of loan applications while balancing business growth, credit quality, and financial risk.

**CreditWise AI** demonstrates how machine learning and business analytics can help understand borrower risk and explore portfolio-level lending patterns.

The application combines predictive modelling, interactive dashboards, and explainable AI to provide a practical view of credit risk.

It is designed as a decision-support and analytical demonstration—not an automated loan approval system.

## ✨ Key Features

| Module | What it delivers |
|---|---|
| 📊 **Executive Dashboard** | High-level portfolio overview, default trends, and key risk indicators. |
| 👤 **Applicant Risk Assessment** | Estimates an individual applicant's probability of default using a calibrated ML model. |
| 📈 **Portfolio Analytics** | Explores borrower characteristics, loan distribution, and portfolio risk patterns. |
| 🧠 **Explainable AI** | Helps interpret model behaviour and understand factors contributing to risk estimates. |
| 📁 **Batch Assessment** | Supports risk assessment of multiple applicant records through a single workflow. |

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Application | Streamlit |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualisation | Plotly |
| Model Persistence | Joblib |
| Image Processing | Pillow |

## 🤖 Machine Learning Approach

The project uses a **Calibrated Random Forest Classification Model** to estimate the probability of loan default.

The selected deployment candidate excludes `loan_grade` to avoid relying on a potentially problematic underwriting-related feature.

**Model workflow:**

1. Data preparation and preprocessing
2. Exploratory data analysis
3. Feature engineering and selection
4. Model training and evaluation
5. Probability calibration
6. Integration into the Streamlit application

The output is an estimated default probability. It should not be interpreted as a guaranteed outcome or an actual lending decision.

## 📊 Business Value

CreditWise AI connects technical machine learning concepts with practical business questions:

- **Risk Identification:** How can potentially high-risk applications be identified for further review?
- **Portfolio Monitoring:** What patterns exist across loan amounts, interest rates, borrower characteristics, and default outcomes?
- **Data-Driven Insights:** How can historical data support better-informed business discussions?
- **Model Transparency:** How can model predictions be explored and interpreted?
- **Operational Efficiency:** How can multiple applicant records be assessed through a consistent analytical workflow?

## 📂 Project Structure

```text
CreditWise-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│       └── credit_risk_cleaned.csv
│
├── models/
│   └── credit_risk_rf_optimized_calibrated_no_loan_grade.joblib
│
├── notebooks/
│   └── Model Development & Analysis
│
├── reports/
│   └── Model Evaluation & Analysis
│
└── src/
    ├── data_loader.py
    ├── model_service.py
    └── Application Modules
```

## 🚀 Run the Project Locally

Follow these steps to run CreditWise AI on your machine.

**1. Clone the repository**

```bash
git clone https://github.com/rushikeshgaikwad2026-glitch/CreditWise-AI.git
```

**2. Navigate to the project directory**

```bash
cd CreditWise-AI
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate the environment (Windows PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Launch the application**

```bash
streamlit run app.py
```

The application will open in your browser, normally at `http://localhost:8501`.

**Required project assets:**
- Processed dataset: `data/processed/credit_risk_cleaned.csv`
- Calibrated model: `models/credit_risk_rf_optimized_calibrated_no_loan_grade.joblib`

## 📚 Dataset

**Source:** [Credit Risk Dataset – Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

The dataset contains historical borrower and loan information used for credit risk analysis and model development.

**Currency:** Monetary values are represented in USD.

## 🛡️ Responsible AI & Limitations

CreditWise AI is an educational and portfolio demonstration project.

- Model predictions represent estimated default probabilities, not guaranteed borrower outcomes.
- Predictions must not be used as the sole basis for lending decisions.
- Historical data can contain biases and may not represent future applicants.
- SHAP feature attributions explain model behaviour; they do not establish causation.
- Descriptive subgroup analysis is not a formal legal fairness assessment.
- The deployed prediction model and explanatory visualisations may represent different stages of the modelling pipeline.

## 💡 Skills Demonstrated

**Business & Analytical Skills**
- Business problem understanding
- Credit risk analysis
- Data-driven decision support
- Portfolio performance analysis
- Translating analytical findings into business insights

**Technical Skills**
- Python programming
- Data cleaning and preprocessing
- Exploratory data analysis
- Machine learning model development
- Probability calibration
- Explainable AI
- Interactive dashboard development
- Streamlit application deployment

## 👨‍💻 About This Project

CreditWise AI is part of my hands-on portfolio demonstrating the practical application of **Machine Learning, Business Analytics, and Data-Driven Decision Making** to a real-world financial services use case.

My objective was to go beyond building a predictive model and develop an interactive application that connects data science with business understanding.

---

<p align="center">
  <strong>Built with Python, Machine Learning & Business Thinking.</strong>
  <br>
  <em>Turning Data into Decisions. Making Analytics Meaningful.</em>
</p>