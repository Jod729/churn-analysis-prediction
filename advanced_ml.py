import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, recall_score, precision_recall_curve, roc_auc_score
from catboost import CatBoostClassifier

from utils.helpers import advanced_feature_engineering

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

def train_and_persist():
    """
    Main training pipeline: Loads data, performs feature engineering, 
    trains an optimized CatBoost model with SMOTE, and persists artifacts.
    """
    # 1. Load Data
    data_path = "data/Telco-Customer-Churn.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please ensure the data directory is populated.")
        
    df = pd.read_csv(data_path)
    
    # Initial Preprocessing
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    # 2. Advanced Feature Engineering (Centralized)
    df = advanced_feature_engineering(df)
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    # Stratified Split for imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Pipeline Definition
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', RobustScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
    ])

    # Optimized CatBoost Parameters (Validated via Optuna)
    clf = CatBoostClassifier(
        iterations=730,
        depth=4,
        learning_rate=0.012785,
        scale_pos_weight=2.607,
        random_state=42,
        verbose=0
    )

    # Full Pipeline with SMOTE for imbalance handling
    full_pipeline = Pipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', clf)
    ])
    
    # 4. Cross-Validation (Verification of stability)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # 5. Training on full training set
    full_pipeline.fit(X_train, y_train)
    
    # 6. Threshold Optimization (Maximize F1-Score)
    y_probs_train = full_pipeline.predict_proba(X_train)[:, 1]
    p, r, t = precision_recall_curve(y_train, y_probs_train)
    f1_scores = 2 * p * r / (p + r + 1e-10)
    best_threshold = float(t[np.argmax(f1_scores)])
    
    # 7. Metadata & Artifact Generation
    y_probs_test = full_pipeline.predict_proba(X_test)[:, 1]
    y_pred_optimized = (y_probs_test >= best_threshold).astype(int)
    
    artifacts = {
        'pipeline': full_pipeline,
        'threshold': best_threshold,
        'feature_metadata': {
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'input_columns': X.columns.tolist()
        },
        'metrics_snapshot': {
            'auc': roc_auc_score(y_test, y_probs_test),
            'f1': f1_score(y_test, y_pred_optimized),
            'recall': recall_score(y_test, y_pred_optimized),
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std()
        }
    }
    
    # 8. Persistence
    os.makedirs("models", exist_ok=True)
    joblib.dump(artifacts, "models/churn_model_v3.joblib")
    
    return artifacts

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Backend Pipeline Training...")
    results = train_and_persist()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training Complete.")
    print("-" * 30)
    print(f"Artifact Saved: 'models/churn_model_v3.joblib'")
    print(f"CV ROC-AUC: {results['metrics_snapshot']['cv_auc_mean']:.4f} (+/- {results['metrics_snapshot']['cv_auc_std']:.4f})")
    print(f"Test AUC:   {results['metrics_snapshot']['auc']:.4f}")
    print(f"Test Recall: {results['metrics_snapshot']['recall']:.4f} (at threshold {results['threshold']:.2f})")
    print(f"Test F1:     {results['metrics_snapshot']['f1']:.4f}")
    print("-" * 30)

