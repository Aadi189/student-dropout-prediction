# src/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    """
    Loads data, corrects column names, engineers academic features,
    filters out 'Enrolled' rows, and maps Target to binary.
    """
    # 1. Load and initial clean
    df = pd.read_csv(filepath)
    df.rename(columns={'Nacionality': 'Nationality'}, inplace=True)
    
    # 2. Feature Engineering (Academic Indicators)
    df["pass_eff_1st"] = np.divide(
        df["Curricular units 1st sem (approved)"], df["Curricular units 1st sem (enrolled)"],
        out=np.zeros_like(df["Curricular units 1st sem (approved)"], dtype=float),
        where=df["Curricular units 1st sem (enrolled)"] != 0
    )
    df["pass_eff_2nd"] = np.divide(
        df["Curricular units 2nd sem (approved)"], df["Curricular units 2nd sem (enrolled)"],
        out=np.zeros_like(df["Curricular units 2nd sem (approved)"], dtype=float),
        where=df["Curricular units 2nd sem (enrolled)"] != 0
    )
    df["pass_eff_avg"] = (df["pass_eff_1st"] + df["pass_eff_2nd"]) / 2

    df["credit_util_1st"] = np.divide(
        df["Curricular units 1st sem (credited)"], df["Curricular units 1st sem (enrolled)"],
        out=np.zeros_like(df["Curricular units 1st sem (credited)"], dtype=float),
        where=df["Curricular units 1st sem (enrolled)"] != 0
    )
    df["credit_util_2nd"] = np.divide(
        df["Curricular units 2nd sem (credited)"], df["Curricular units 2nd sem (enrolled)"],
        out=np.zeros_like(df["Curricular units 2nd sem (credited)"], dtype=float),
        where=df["Curricular units 2nd sem (enrolled)"] != 0
    )
    df["credit_util_avg"] = (df["credit_util_1st"] + df["credit_util_2nd"]) / 2

    df["eval_rate_1st"] = np.divide(
        df["Curricular units 1st sem (evaluations)"], df["Curricular units 1st sem (enrolled)"],
        out=np.zeros_like(df["Curricular units 1st sem (evaluations)"], dtype=float),
        where=df["Curricular units 1st sem (enrolled)"] != 0
    )
    df["eval_rate_2nd"] = np.divide(
        df["Curricular units 2nd sem (evaluations)"], df["Curricular units 2nd sem (enrolled)"],
        out=np.zeros_like(df["Curricular units 2nd sem (evaluations)"], dtype=float),
        where=df["Curricular units 2nd sem (enrolled)"] != 0
    )
    df["eval_rate_avg"] = (df["eval_rate_1st"] + df["eval_rate_2nd"]) / 2

    df["improvement"] = df["Curricular units 2nd sem (approved)"] - df["Curricular units 1st sem (approved)"]

    # Drop temporary columns used for math
    df.drop(columns=["pass_eff_1st", "pass_eff_2nd", "credit_util_1st", "credit_util_2nd", "eval_rate_1st", "eval_rate_2nd"], inplace=True, errors='ignore')
    df.replace([float("inf"), -float("inf")], 0, inplace=True)
    df.fillna(0, inplace=True)

    # 3. Filter Target and map to Binary (Dropout: 1, Graduate: 0)
    df = df[df['Target'] != 'Enrolled'].copy()
    df['target_binary'] = df['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)
    
    # 4. Separate features and labels
    X = df.drop(columns=['Target', 'target_binary'])
    y = df['target_binary']
    
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
