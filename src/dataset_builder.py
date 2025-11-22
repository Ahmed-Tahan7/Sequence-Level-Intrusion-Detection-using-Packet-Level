import os
import glob
from typing import List, Optional, Tuple, Generator

import numpy as np
import pandas as pd
import tensorflow as tf


def find_csv_files(data_dir: str) -> List[str]:
    """Return sorted list of CSV files in data_dir"""
    pattern = os.path.join(data_dir, "*.csv")
    return sorted(glob.glob(pattern))


def load_and_concatenate_csv(files: List[str], nrows: Optional[int] = None) -> pd.DataFrame:
    """Load multiple CSVs safely and concatenate into a single DataFrame"""
    dfs = []
    for f in files:
        print(f"Loading {f} ...")
        dfs.append(pd.read_csv(f, encoding='latin1', low_memory=False, nrows=nrows))
    df = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    # Strip column names of BOM / extra spaces
    df.columns = [c.strip() for c in df.columns]
    return df


def sequence_generator(X: np.ndarray, y: np.ndarray, seq_len: int, stride: int = 1) -> Generator:
    """
    Yields sequences one by one instead of building a giant array
    X: (num_samples, num_features)
    y: (num_samples,)
    """
    n_samples, n_features = X.shape
    for start in range(0, n_samples - seq_len + 1, stride):
        X_seq = X[start:start + seq_len]
        y_seq = 1 if np.any(y[start:start + seq_len] == 1) else 0
        yield X_seq.astype(np.float32), np.array(y_seq, dtype=np.int32)


def build_tf_dataset(X: np.ndarray, y: np.ndarray, seq_len: int, stride: int = 1,
                     batch_size: int = 64) -> tf.data.Dataset:
    """
    Returns a tf.data.Dataset that loads sequences lazily in batches
    """
    output_signature = (
        tf.TensorSpec(shape=(seq_len, X.shape[1]), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    ds = tf.data.Dataset.from_generator(
        lambda: sequence_generator(X, y, seq_len, stride),
        output_signature=output_signature
    )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def prepare_dataset(data_dir: str,
                    seq_len: int = 20,
                    stride: int = 1,
                    nrows: Optional[int] = None,
                    batch_size: int = 64) -> Tuple[tf.data.Dataset, pd.DataFrame]:
    """
    High-level helper to load CSVs, preprocess, and return a lazy TF dataset
    along with the raw DataFrame.
    """
    files = find_csv_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    df = load_and_concatenate_csv(files, nrows=nrows)
    print("Columns in dataset:", df.columns)

    # Detect label column: either named 'label' or take last column
    if 'label' not in df.columns:
        print("No 'label' column found. Using last column as label.")
        df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

    # Convert label to numeric safely
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    # Binary labels (0=normal, 1=attack)
    y_flows = (df['label'] != 0).astype(int).values

    # Remove non-feature columns
    drop_cols = [c for c in ['id', 'srcip', 'dstip', 'stime', 'ltime', 'attack_cat'] if c in df.columns]
    df_features = df.drop(columns=drop_cols, errors='ignore')

    # Convert all features to numeric
    df_features = df_features.apply(pd.to_numeric, errors='coerce').fillna(0)
    X = df_features.values.astype(np.float32)

    print(f"Feature matrix shape: {X.shape}")

    # Build lazy TensorFlow dataset
    ds = build_tf_dataset(X, y_flows, seq_len=seq_len, stride=stride, batch_size=batch_size)

    return ds, df