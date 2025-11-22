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
    """
    # 1️⃣ Load CSVs as strings to avoid mixed-type issues
    frames = [pd.read_csv(f, encoding='latin1', dtype=str, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # 2️⃣ Convert numeric columns to float
    numeric_cols = df.columns[df.apply(lambda col: col.str.replace('.', '', 1).str.isnumeric().all())].tolist()
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 3️⃣ Scale numeric columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 4️⃣ Encode categorical columns
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    return df, scaler
