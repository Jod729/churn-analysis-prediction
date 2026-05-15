# ChurnAI Enterprise: Customer Intelligence Platform

## Overview
ChurnAI Enterprise is a production-grade machine learning platform designed to identify high-risk customer segments and predict churn with high precision. By leveraging advanced ensemble learning and automated feature engineering, this platform provides actionable intelligence to protect recurring revenue and optimize retention strategies.

## Features
- **Predictive Intelligence**: High-sensitivity CatBoost engine optimized for Recall and F1-Score.
- **Enterprise Dashboard**: Professional SaaS-inspired interface for executive-level business overview.
- **Automated Risk Profiling**: Real-time risk categorization (Stable to Critical) based on individual customer parameters.
- **Leakage-Free Pipeline**: Industry-standard ML architecture with stratified cross-validation and in-fold SMOTE imbalance handling.
- **Tactical Recommendations**: Automated retention suggestions driven by customer-specific risk factors.

## Machine Learning Engineering
The platform is built on a robust ML foundation designed for real-world business deployment:

### 1. Ensemble Pipeline
The core engine utilizes a **CatBoost Classifier**, a high-performance gradient-boosting ensemble optimized for categorical data handling and robust generalization. The model was tuned using **Bayesian Optimization** (Optuna).

### 2. Feature Engineering & Selection
Beyond raw data, the system engineers custom behavioral and financial indicators:
- **Monthly-to-Total Ratio**: Detection of sudden billing velocity changes.
- **Tenure-Monthly Interaction**: Capturing the combined effect of loyalty and spend.
- **Service Density**: Quantifying the customer's overall service ecosystem.

### 3. Production Training Pipeline (`advanced_ml.py`)
The repository includes a standalone, optimized training script (`advanced_ml.py`) that encapsulates:
- Automated preprocessing and feature engineering.
- Stratified 5-fold cross-validation for stability verification.
- Decision boundary optimization via Precision-Recall curves.
- Automated artifact versioning and persistence.

## Tech Stack
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, CatBoost, imbalanced-learn
- **UI Framework**: Streamlit
- **Persistence**: Joblib
- **Visualization**: Matplotlib, Seaborn

## Model Performance
The platform is optimized for **Recall** to minimize missed churn opportunities while maintaining high precision.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 78.0% |
| **Recall** | 72.7% |
| **F1-Score** | 0.64 |
| **ROC-AUC** | 0.84 |

## Dashboard Preview

### Executive Overview
![Overview](assets/screenshots/overview.png)

### Behavioral Analytics
![Analytics](assets/screenshots/analytics.png)

### Engine Intelligence
![Intelligence](assets/screenshots/intelligence.png)

### Prediction Lab
![Prediction](assets/screenshots/prediction.png)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Jod729/churn-analysis-prediction.git
cd churn-analysis-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Platform
```bash
# Start the Streamlit Dashboard
streamlit run app.py

# (Optional) Re-run the Advanced ML Training Pipeline
python advanced_ml.py
```

## Project Architecture
```text
├── app.py                      # Main production dashboard
├── advanced_ml.py              # Automated training & optimization pipeline
├── utils/
│   ├── helpers.py              # Centralized feature engineering
│   └── inference.py            # High-performance prediction engine
├── models/
│   └── churn_model_v3.joblib   # Persisted production artifacts
├── data/
│   └── Telco-Customer-Churn.csv # Customer intelligence dataset
└── assets/screenshots/         # Platform UI previews
```

## Business Insights
- **Contract Volatility**: Month-to-month contracts are identified as the strongest predictor of churn risk.
- **Service Synergy**: Absence of Tech Support and Online Security correlates highly with customer attrition.
- **Stability Threshold**: Customers exceeding 12 months of tenure show significantly higher loyalty scores.

## Future Improvements
- Multi-region churn trend analysis.
- Automated email triggers for high-risk customer outreach.
- SHAP-based local explainability for individual predictions.

## Author
Developed by **Jod729**.
