import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import modularized backend logic
from utils.inference import load_artifacts, run_inference

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="ChurnAI Enterprise | State-of-the-Art Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ASSET LOADING (CACHED)
# ==========================================
@st.cache_resource
def get_artifacts():
    return load_artifacts()

@st.cache_data
def get_raw_data():
    path = "data/Telco-Customer-Churn.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

artifacts = get_artifacts()
raw_df = get_raw_data()

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary: #2563eb;
        --slate-50: #f8fafc;
        --slate-100: #f1f5f9;
        --slate-200: #e2e8f0;
        --slate-800: #1e293b;
        --slate-900: #0f172a;
    }

    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: var(--slate-50); }
    
    .hero-panel {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        padding: 3.5rem;
        border-radius: 1.5rem;
        color: white;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .hero-title { font-size: 3rem; font-weight: 800; letter-spacing: -0.04em; margin-bottom: 0.75rem; line-height: 1.1; }
    .hero-tagline { font-size: 1.1rem; color: #94a3b8; max-width: 650px; line-height: 1.6; }
    
    .hero-info-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 2rem;
        margin-top: 2.5rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    .hero-info-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.12em; }
    .hero-info-value { font-size: 1.25rem; font-weight: 700; color: #f1f5f9; }

    .section-header { font-size: 1.6rem; font-weight: 800; color: var(--slate-900); margin: 3rem 0 1.5rem 0; display: flex; align-items: center; gap: 1rem; }
    .section-header::after { content: ""; height: 2px; flex: 1; background: linear-gradient(to right, var(--slate-200), transparent); }
    
    .metric-card { background: white; padding: 1.75rem; border-radius: 1.25rem; border: 1px solid var(--slate-200); box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition: all 0.3s ease; }
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 20px -5px rgba(0,0,0,0.08); border-color: var(--primary); }
    .metric-label { font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-value { font-size: 2.25rem; font-weight: 800; color: var(--slate-900); margin: 0.5rem 0; }
    
    .plot-box { background: white; padding: 2rem; border-radius: 1.5rem; border: 1px solid var(--slate-200); margin-bottom: 2rem; }
    .plot-title { font-size: 1.1rem; font-weight: 700; color: var(--slate-800); margin-bottom: 1.5rem; }

    .risk-badge { padding: 2.5rem; border-radius: 1.75rem; text-align: center; margin: 2rem 0; border: 1px solid transparent; }
    .risk-label { font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.2em; opacity: 0.8; margin-bottom: 0.75rem; }
    .risk-level { font-size: 3.5rem; font-weight: 900; letter-spacing: -0.04em; margin-bottom: 0.5rem; }
    .risk-prob { font-size: 1.25rem; font-weight: 600; }

    .risk-critical { background: #fff1f2; color: #9f1239; border-color: #fecdd3; }
    .risk-high { background: #fff7ed; color: #c2410c; border-color: #ffedd5; }
    .risk-moderate { background: #fefce8; color: #a16207; border-color: #fef9c3; }
    .risk-stable { background: #f0fdf4; color: #15803d; border-color: #dcfce7; }
    
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid var(--slate-100); }
    .nav-brand { font-size: 1.5rem; font-weight: 900; color: var(--slate-900); margin-bottom: 3rem; letter-spacing: -0.05em; }
    .nav-brand span { color: var(--primary); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown('<div class="nav-brand">CHURNAI<span>PRO</span></div>', unsafe_allow_html=True)
    nav_selection = st.radio("Navigation", ["Executive Overview", "Behavioral Analytics", "Engine Intelligence", "Prediction Lab"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### System Health")
    st.markdown('<div style="font-size:0.85rem; color:#64748b;"><span style="color:#22c55e;">●</span> Engine Status: <b>ACTIVE</b></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.85rem; color:#64748b;"><span style="color:#22c55e;">●</span> Deployment: <b>STABLE</b></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("© 2026 Enterprise Release v3.4.2")

# ==========================================
# HERO PANEL
# ==========================================
st.markdown(f"""
    <div class="hero-panel">
        <div class="hero-title">Predictive Revenue Safeguard</div>
        <div class="hero-tagline">Advanced gradient-boosted intelligence for early churn detection. Secure your customer base with high-sensitivity predictive modeling.</div>
        <div class="hero-info-grid">
            <div class="hero-info-item"><div class="hero-info-label">Detection Recall</div><div class="hero-info-value">{artifacts['metrics_snapshot']['recall']:.1%}</div></div>
            <div class="hero-info-item"><div class="hero-info-label">Stability AUC</div><div class="hero-info-value">{artifacts['metrics_snapshot']['auc']:.2f}</div></div>
            <div class="hero-info-item"><div class="hero-info-label">Database Size</div><div class="hero-info-value">{len(raw_df):,}</div></div>
            <div class="hero-info-item"><div class="hero-info-label">Environment</div><div class="hero-info-value" style="color:#60a5fa;">Production</div></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# SECTIONS
# ==========================================

if nav_selection == "Executive Overview":
    st.markdown('<div class="section-header">Executive Dashboard</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    metrics = [("Model Accuracy", f"{artifacts['metrics_snapshot']['auc']:.1%}"), ("Churn Recall", f"{artifacts['metrics_snapshot']['recall']:.1%}"), ("F1 Precision", f"{artifacts['metrics_snapshot']['f1']:.2f}"), ("Avg Churn Rate", f"{(raw_df['Churn'] == 'Yes').mean() if not raw_df.empty else 0:.1%}")]
    for i, (l, v) in enumerate(metrics):
        with cols[i]: st.markdown(f'<div class="metric-card"><div class="metric-label">{l}</div><div class="metric-value">{v}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Customer Insight Feed</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(20), use_container_width=True)

elif nav_selection == "Behavioral Analytics":
    st.markdown('<div class="section-header">Analytical Deep Dive</div>', unsafe_allow_html=True)
    if not raw_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="plot-box"><div class="plot-title">Class Distribution</div>', unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(8, 4.5)); sns.countplot(x='Churn', data=raw_df, palette=['#e2e8f0', '#2563eb'], ax=ax1); sns.despine(); st.pyplot(fig1)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="plot-box"><div class="plot-title">Contract Mix vs. Risk</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(8, 4.5)); sns.countplot(x='Contract', hue='Churn', data=raw_df, palette=['#cbd5e1', '#2563eb'], ax=ax2); sns.despine(); st.pyplot(fig2)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="plot-box"><div class="plot-title">Tenure Risk Density Analysis</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(16, 4.5)); sns.kdeplot(data=raw_df, x='tenure', hue='Churn', fill=True, palette=['#64748b', '#2563eb'], ax=ax3); sns.despine(); st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)

elif nav_selection == "Engine Intelligence":
    st.markdown('<div class="section-header">Engine Logic & Performance</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("#### Logic Summary\n- **Algorithm**: CatBoost Ensemble\n- **Handling**: In-pipeline SMOTE\n- **Optimization**: Bayesian Search (40 trials)\n- **Features**: 19 Base + 6 Engineered")
        try:
            model = artifacts['pipeline'].named_steps['clf']; pre = artifacts['pipeline'].named_steps['pre']; f_names = pre.get_feature_names_out(); f_imp = model.feature_importances_
            imp_df = pd.DataFrame({'Feature': f_names, 'Importance': f_imp}).sort_values('Importance', ascending=False).head(10)
            st.markdown("#### Primary Churn Determinants")
            fig_i, ax_i = plt.subplots(figsize=(10, 6)); sns.barplot(x="Importance", y="Feature", data=imp_df, palette="Blues_r", ax=ax_i); sns.despine(); st.pyplot(fig_i)
        except: st.info("Loading determinants...")
    with col2:
        st.markdown("#### Performance Validation")
        if os.path.exists("plots/ensemble_confusion_matrix.png"): st.image("plots/ensemble_confusion_matrix.png", use_container_width=True)
        if os.path.exists("plots/pr_curve.png"): st.image("plots/pr_curve.png", use_container_width=True)

elif nav_selection == "Prediction Lab":
    st.markdown('<div class="section-header">Risk Profiling Laboratory</div>', unsafe_allow_html=True)
    with st.container():
        i1, i2, i3 = st.columns(3)
        with i1:
            st.markdown("#### Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"]); senior = st.selectbox("Senior Citizen", ["No", "Yes"]); partner = st.selectbox("Has Partner", ["Yes", "No"]); dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        with i2:
            st.markdown("#### Subscription")
            tenure = st.slider("Tenure (Months)", 0, 72, 12); monthly = st.number_input("Monthly Charges ($)", 18.0, 130.0, 70.0); total = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0); contract = st.selectbox("Contract Structure", ["Month-to-month", "One year", "Two year"])
        with i3:
            st.markdown("#### Infrastructure")
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]); payment = st.selectbox("Payment Gateway", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]); paperless = st.selectbox("Paperless Billing", ["Yes", "No"]); phone = st.selectbox("Phone Service", ["Yes", "No"])
        with st.expander("Granular Service Parameters"):
            a1, a2, a3 = st.columns(3)
            with a1: multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"]); security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            with a2: backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"]); protect = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            with a3: support = st.selectbox("Technical Support", ["No", "Yes", "No internet service"]); tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

    if st.button("RUN RISK DIAGNOSTIC", use_container_width=True):
        input_data = pd.DataFrame([{
            'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0, 'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet, 'OnlineSecurity': security, 'OnlineBackup': backup,
            'DeviceProtection': protect, 'TechSupport': support, 'StreamingTV': tv, 'StreamingMovies': "No",
            'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total
        }])
        
        result = run_inference(input_data, artifacts)
        
        st.markdown(f"""
            <div class="risk-badge {result['style']}">
                <div class="risk-label">Diagnostic Outcome</div>
                <div class="risk-level">{result['category']}</div>
                <div class="risk-prob">Churn Probability: {result['probability']:.1%}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Revenue Protection Strategy</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("#### Primary Risk Drivers")
            if contract == "Month-to-month": st.error("○ Month-to-month billing structure is highly volatile.")
            if internet == "Fiber optic": st.warning("○ Fiber optic segments demonstrate increased churn velocity.")
            if tenure < 12: st.warning("○ Customer is within the critical 12-month stabilization period.")
            if support == "No": st.info("○ Absence of technical support correlates with lower engagement.")
        with r2:
            st.markdown("#### Tactical Recommendations")
            if result['probability'] >= result['threshold']:
                st.write("- **Proactive Outreach**: Immediate success-manager intervention.")
                st.write("- **Contract Locking**: Propose 12-month loyalty incentives.")
                st.write("- **Service Bundling**: Complimentary Tech-Support trial.")
            else:
                st.write("- **Continuous Engagement**: Enroll in digital loyalty program.")
                st.write("- **Up-selling**: Target for 'Premium Security' expansion.")

st.markdown("---")
st.markdown(f'<div style="text-align: center; color: #94a3b8; padding: 2rem 0; font-size: 0.85rem;">© {datetime.now().year} ChurnAI Enterprise Intelligence | v3.4.2 Production Release</div>', unsafe_allow_html=True)