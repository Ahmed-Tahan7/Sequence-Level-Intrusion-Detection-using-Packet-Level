import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_pipeline(files):
    """
    Full preprocessing pipeline:
    - Load CSVs
    - Convert numeric columns
    - Scale numeric columns
    - Encode categorical columns
    Returns:
        df: preprocessed DataFrame
        scaler: fitted StandardScaler for numeric features
        numeric_cols: list of numeric columns
        all_columns: final list of columns (numeric + encoded categorical)
    """
    # 1️⃣ Load CSVs
    frames = [pd.read_csv(f, encoding='latin1', dtype=str, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # 2️⃣ Convert numeric columns
    numeric_cols = []
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            numeric_cols.append(col)
        except:
            continue

    # 3️⃣ Scale numeric columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0))

    # 4️⃣ Encode categorical columns
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    all_columns = numeric_cols + categorical_cols
    df = df[all_columns]  # ensure consistent order

    return df, scaler, numeric_cols, all_columns