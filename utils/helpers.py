import pandas as pd

def advanced_feature_engineering(df):
    """
    Consistent feature engineering for training and inference.
    """
    df = df.copy()
    
    # Financial Ratios
    df["Monthly_to_Total_Ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    df["Tenure_Monthly"] = df["tenure"] * df["MonthlyCharges"]
    
    # Service Count (Binary count of all services)
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    # Handle both training (full dataset) and inference (single row)
    df["Services_Count"] = (df[service_cols].isin(['Yes', 'Fiber optic', 'DSL'])).sum(axis=1)
    
    # Behavioral Flags
    df["Is_MTM"] = (df["Contract"] == "Month-to-month").astype(int)
    df["No_Security"] = (df["OnlineSecurity"] == "No").astype(int)
    df["Fiber_User"] = (df["InternetService"] == "Fiber optic").astype(int)
    
    return df
