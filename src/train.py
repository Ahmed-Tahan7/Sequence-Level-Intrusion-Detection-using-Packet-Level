import numpy as np
import tensorflow as tf
import os
from src.model import build_cnn_lstm_autoencoder
from src.dataset_builder import build_tf_dataset

# ---------------------------------------------------------
# Autoencoder Training
# ---------------------------------------------------------
def train_autoencoder(train_ds: tf.data.Dataset,
                      seq_len: int,
                      n_features: int,
                      epochs: int = 20,
                      latent_dim: int = 64):
    """
    Train CNN-LSTM Autoencoder using a memory-efficient tf.data.Dataset.
    """
    model = build_cnn_lstm_autoencoder(seq_len, n_features, latent_dim)

    history = model.fit(
        train_ds,
        epochs=epochs
    )

    return model, history


# ---------------------------------------------------------
# Pipeline Wrapper
# ---------------------------------------------------------
def train_pipeline(
        preprocessed_df,
        y_flows,
        seq_len=20,
        stride=1,
        batch_size=64,
        epochs=20):

    """
    Preprocessed df → sequences → train autoencoder only on NORMAL traffic → save model + threshold.
    """

    # -----------------------------------------------------
    # Extract features and labels
    # -----------------------------------------------------
    X = preprocessed_df.drop(columns=["label"], errors="ignore").values.astype(np.float32)
    y = y_flows.astype(np.int32)

    print(f"Full feature matrix shape: {X.shape}, Labels shape: {y.shape}")

    # -----------------------------------------------------
    # Filter NORMAL records (label=0) for training
    # -----------------------------------------------------
    normal_mask = (y == 0)
    X_normal = X[normal_mask]
    y_normal = y[normal_mask]

    print(f"Normal flows used for training: {X_normal.shape[0]} / {X.shape[0]}")

    # -----------------------------------------------------
    # Build TF datasets
    # -----------------------------------------------------
    train_ds = build_tf_dataset(
        X_normal, y_normal,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size
    )

    full_ds = build_tf_dataset(
        X, y,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size
    )

    # -----------------------------------------------------
    # Train model
    # -----------------------------------------------------
    n_features = X.shape[1]

    model, history = train_autoencoder(
        train_ds=train_ds,
        seq_len=seq_len,
        n_features=n_features,
        epochs=epochs
    )

    # -----------------------------------------------------
    # Compute reconstruction errors for all sequences
    # -----------------------------------------------------
    print("Computing threshold...")

    all_errors = []
    for batch_x, _ in full_ds:
        recon = model.predict(batch_x, verbose=0)
        batch_err = np.mean((batch_x.numpy() - recon)**2, axis=(1, 2))
        all_errors.append(batch_err)

    all_errors = np.concatenate(all_errors)

    threshold = np.percentile(all_errors, 95)
    print(f"Computed threshold: {threshold}")

    # -----------------------------------------------------
    # Save model + threshold
    # -----------------------------------------------------
    os.makedirs("models", exist_ok=True)

    model.save("models/autoencoder.h5")
    print("Model saved → models/autoencoder.h5")

    np.save("models/threshold.npy", threshold)
    print("Threshold saved → models/threshold.npy")

    return model, history, full_ds, threshold
