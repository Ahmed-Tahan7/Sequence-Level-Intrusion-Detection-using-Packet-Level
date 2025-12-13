import pandas as pd
import numpy as np

# Load your training CSVs
files = [
    r"data/UNSW-NB15_3.csv",
    r"data/UNSW-NB15_4.csv"
]

df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

# Drop unwanted columns like in preprocessing
drop_cols = ['id','srcip','dstip','stime','ltime','attack_cat']
df_features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Drop label
if "label" in df_features.columns:
    df_features = df_features.drop(columns=["label"])

# Save column order
training_columns = df_features.columns.to_numpy()
np.save("models/training_columns.npy", training_columns)
print("Saved training columns → models/training_columns.npy")
