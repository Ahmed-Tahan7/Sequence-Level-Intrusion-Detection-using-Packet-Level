import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from main import build_sequences

SEQ_LEN = 20
STRIDE = 10
BATCH_SIZE = 64

def preprocess_test_csv(csv_path, training_columns, scaler):
    df = pd.read_csv(csv_path)
    # Add missing columns
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[training_columns]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    df_scaled = scaler.transform(df)
    return df_scaled.astype(np.float32)

def run_inference(model_path, threshold_path, csv_path, training_columns_path, scaler_path, dynamic_threshold=True):
    print(f"Loading model: {model_path}")
    model = load_model(model_path, compile=False)

    # Load training artifacts
    training_columns = np.load(training_columns_path, allow_pickle=True)
    scaler = joblib.load(scaler_path)

    # Preprocess test data
    X_test = preprocess_test_csv(csv_path, training_columns, scaler)
    dummy_y = np.zeros(len(X_test), dtype=np.int32)
    X_seq, _ = build_sequences(X_test, dummy_y)

    print("Running inference...")
    X_pred = model.predict(X_seq, batch_size=BATCH_SIZE)
    errors = np.mean((X_seq - X_pred) ** 2, axis=(1,2))

    # Decide threshold
    if dynamic_threshold:
        print(f"Error stats: min={errors.min():.5f}, max={errors.max():.5f}, mean={errors.mean():.5f}")
        threshold = np.percentile(errors, 90)  # adjust percentile
        print(f"Dynamic threshold (90th percentile of test errors): {threshold:.5f}")
    else:
        threshold = np.load(threshold_path) if os.path.exists(threshold_path) else 0.1
        print(f"Using loaded threshold: {threshold:.5f}")

    y_pred = (errors > threshold).astype(int)

    # Plot error distribution
    plt.figure(figsize=(10,6))
    plt.hist(errors, bins=100, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold={threshold:.5f}')
    plt.xlabel("Reconstruction error")
    plt.ylabel("Count")
    plt.title("Sequence Error Distribution")
    plt.legend()
    plt.show()

    print(f"Predictions completed. Total sequences: {len(y_pred)}")
    return y_pred, errors, threshold

if __name__ == "__main__":
    model_path = "models/cnn_lstm_autoencoder.keras"
    threshold_path = "models/threshold.npy"
    training_columns_path = "models/training_columns.npy"
    scaler_path = "models/scaler.pkl"
    csv_path = "extra_data/UNSW_NB15_testing-set.csv"

    # Set dynamic_threshold=True to use test-data-based threshold
    y_pred, errors, threshold = run_inference(model_path, threshold_path, csv_path, training_columns_path, scaler_path, dynamic_threshold=True)

    print(f"\nUsing threshold: {threshold:.5f}\n")
    for i in range(min(20, len(y_pred))):
        print(f"Sequence {i}: Predicted={y_pred[i]}, Error={errors[i]:.5f}")
