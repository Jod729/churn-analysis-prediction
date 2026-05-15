import joblib
import pandas as pd
import os
from .helpers import advanced_feature_engineering

def load_artifacts(model_path="models/churn_model_v3.joblib"):
    """Loads model artifacts from the specified path."""
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

def run_inference(input_df, artifacts):
    """
    Executes the full inference pipeline: Feature Engineering -> Alignment -> Prediction.
    """
    # 1. Feature Engineering (Centralized)
    processed_df = advanced_feature_engineering(input_df)
    
    # 2. Column Alignment
    # Ensures input matches the exact column order and presence expected by the pipeline
    input_cols = artifacts['feature_metadata']['input_columns']
    aligned_df = processed_df[input_cols]
    
    # 3. Probability Prediction
    probability = artifacts['pipeline'].predict_proba(aligned_df)[0][1]
    
    # 4. Risk Classification based on threshold
    threshold = artifacts['threshold']
    
    if probability >= 0.75:
        category, style = "CRITICAL", "risk-critical"
    elif probability >= threshold:
        category, style = "HIGH RISK", "risk-high"
    elif probability >= 0.25:
        category, style = "MODERATE", "risk-moderate"
    else:
        category, style = "STABLE", "risk-stable"
        
    return {
        'probability': probability,
        'category': category,
        'style': style,
        'threshold': threshold
    }
