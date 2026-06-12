# hr-attrition-analysis
# 👥 IBM HR Employee Attrition Analysis

> End-to-end machine learning project predicting and explaining
> employee attrition using the IBM HR Analytics dataset.

## 🔗 Live Dashboard
[View Dashboard →](https://hr-attrition-onikeyek.streamlit.app/)

## 📌 Project Overview
This project analyses 1,470 employee records to identify
attrition drivers, build a predictive model, and deliver
actionable business recommendations worth $9.62M in
potential annual savings.

## 🔬 Key Findings
- **Overtime** is the #1 attrition driver — employees working
  overtime leave at **3x the rate** of others (30.5% vs 10.4%)
- **Stock options** are an underutilised retention lever —
  employees with zero options leave at 24.4% vs 9.9%
- **Sales department** has the highest attrition at 20.6%
- **18-25 age group** at 34.8% — more than double the average

## 🤖 Model Performance
| Model | ROC-AUC | F1 Score |
|-------|---------|----------|
| Logistic Regression | 0.787 | 0.459 |
| Random Forest | 0.797 | 0.380 |
| **XGBoost ★** | **0.801** | **0.511** |

## 💰 Business Impact
| Intervention | Est. Annual Saving |
|---|---|
| Overtime Control | $2.23M |
| Salary Review | $1.90M |
| Career Development | $2.34M |
| Stock Option Expansion | $3.15M |
| **Total** | **$9.62M** |

## 🛠 Tech Stack
- **Language:** Python 3.12
- **ML:** Scikit-learn, XGBoost, SHAP
- **Preprocessing:** SMOTE (imbalanced-learn)
- **Dashboard:** Streamlit + Plotly
- **Environment:** JupyterLab

## 📁 Project Structure
```
hr-attrition/
├── data/
│   └── ibm_hr_attrition.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_explainability.ipynb
├── dashboard/
│   └── app.py
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
├── report/
│   ├── recommendations.png
│   └── action_plan.png
├── requirements.txt
└── README.md
```

## 🚀 Run Locally
```bash
git clone https://github.com/onikeyek/hr-attrition-analysis
cd hr-attrition-analysis
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## 📊 Dataset
IBM HR Analytics Employee Attrition Dataset
- 1,470 employees
- 35 features
- 16.1% attrition rate
- Source: Kaggle
