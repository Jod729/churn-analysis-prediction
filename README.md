# ChurnAI Enterprise: Customer Retention Intelligence Platform

![Project Status](https://img.shields.io/badge/Status-Production--Ready-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![CatBoost](https://img.shields.io/badge/Model-CatBoost-yellow)

## 📌 Project Overview
**ChurnAI Enterprise** is a production-grade machine learning platform designed to identify high-risk customer segments and predict churn with high precision. Churn—the rate at which customers stop doing business with an entity—is a critical business metric where early detection can save millions in recurring revenue. This platform provides actionable intelligence, risk profiling, and tactical retention strategies through a state-of-the-art predictive engine.

## 🚀 Key Features
- **High-Performance Backend**: Leverages an optimized CatBoost ensemble trained via Bayesian optimization (Optuna).
- **Leakage-Free Pipeline**: Implements stratified cross-validation with SMOTE applied strictly within-fold to ensure robust generalization.
- **Dynamic Risk Profiling**: Real-time customer risk assessment with categorical probability classification (Stable to Critical).
- **Premium Analytics Dashboard**: A modern, SaaS-inspired interface for executive-level business intelligence.
- **Actionable Insights**: Automated retention recommendations based on individual customer risk drivers.
- **Production-Ready Architecture**: Modular code structure with persistent model artifacts and centralized feature engineering.

## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost
- **Imbalance Handling**: Imbalanced-learn (SMOTE)
- **Deployment & UI**: Streamlit
- **Persistence**: Joblib
- **Visualization**: Matplotlib, Seaborn

## 📊 Model Performance
The engine is optimized for **Recall** and **F1-Score** to prioritize the detection of at-risk customers (minimizing False Negatives).

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 78.0% |
| **Recall (Sensitivity)** | **72.7%** |
| **F1-Score** | **0.64** |
| **ROC-AUC** | 0.84 |
| **Optimal Threshold** | 0.6961 |

## 📂 Project Structure
```text
├── app.py                      # Main production dashboard
├── requirements.txt            # Dependency configuration
├── README.md                   # Platform documentation
├── data/
│   └── Telco-Customer-Churn.csv # Customer intelligence dataset
├── models/
│   └── churn_model_v3.joblib   # Persisted model artifacts
├── plots/
│   └── (visualizations)        # Performance and analytical charts
└── utils/
    ├── helpers.py              # Centralized feature engineering
    └── inference.py            # High-performance prediction engine
```

## 📸 Dashboard Preview
*(Add your dashboard screenshots here to showcase the premium UI)*

## ⚙️ Installation & Usage

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
streamlit run app.py
```

## 💡 Business Intelligence
The platform identifies critical churn drivers such as:
- **Contract Structure**: Month-to-month contracts are identified as the primary risk factor.
- **Service Gaps**: Absence of technical support and online security significantly increases churn velocity.
- **Tenure Velocity**: New customers (tenure < 12 months) are flagged as a high-instability segment requiring immediate engagement.

## 🛠️ Future Roadmap
- Integration with live CRM data streams.
- Deep SHAP-based local explanations for every prediction.
- Automated A/B test tracking for retention interventions.

## 👤 Author
Developed by **Jod729** - Focused on bridging the gap between advanced ML research and production-grade business intelligence.
