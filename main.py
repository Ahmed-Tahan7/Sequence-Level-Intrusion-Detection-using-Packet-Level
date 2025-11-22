import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ------------------------- Config -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

SEQ_LEN = 20
STRIDE = 10
BATCH_SIZE = 64
EPOCHS = 20
LATENT_DIM = 64

# ------------------------- Dataset -------------------------
def load_csvs(data_dir):
    """Load all CSVs and concatenate into a single DataFrame."""
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
    df_list = []
    for f in all_files:
        print(f"Loading {f} ...")
        df_list.append(pd.read_csv(f))
    df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    return df

def preprocess_df(df):
    """Keep numeric features, scale them, and extract labels."""
    drop_cols = ['id','srcip','dstip','stime','ltime','attack_cat']
    df_features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df_features = df_features.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Label handling
    if "label" in df.columns:
        label_col = "label"
    else:
        label_col = df_features.columns[-1]   # fallback

    y_flows = df_features[label_col].astype(int).values
    df_features = df_features.drop(columns=[label_col])

    # Scale features
    scaler = StandardScaler()
    df_features_scaled = scaler.fit_transform(df_features)

    return df_features_scaled.astype(np.float32), y_flows

def build_sequences(X, y, seq_len=SEQ_LEN, stride=STRIDE):
    """Convert feature matrix to overlapping sequences."""
    n_samples = (len(X) - seq_len) // stride + 1
    X_seq = np.zeros((n_samples, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        start = i * stride
        end = start + seq_len
        X_seq[i] = X[start:end]
        y_seq[i] = int(np.any(y[start:end] != 0))  # sequence is anomaly if any sample is anomaly
    return X_seq, y_seq

def build_tf_dataset(X_seq, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((X_seq, X_seq))  # autoencoder target = input
    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ------------------------- Model -------------------------
def build_cnn_lstm_autoencoder(seq_len, n_features, latent_dim=LATENT_DIM):
    inputs = layers.Input(shape=(seq_len, n_features))
    # Encoder
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    encoded = layers.LSTM(latent_dim)(x)
    # Decoder
    x = layers.RepeatVector(seq_len)(encoded)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    outputs = layers.Conv1D(n_features, kernel_size=3, padding='same', activation='linear')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# ------------------------- Train & Evaluate -------------------------
def train_autoencoder(X_seq, y_seq, seq_len=SEQ_LEN, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train only on NORMAL sequences."""
    normal_mask = (y_seq == 0)
    X_train = X_seq[normal_mask]
    if X_train.shape[0] == 0:
        raise ValueError("No normal sequences found for training!")
    print(f"Training dataset shape (only normals): {X_train.shape}")
    model = build_cnn_lstm_autoencoder(seq_len, X_train.shape[2])
    train_ds = build_tf_dataset(X_train)
    history = model.fit(train_ds, epochs=epochs)
    return model, history

def evaluate_model(model, X_seq, y_seq):
    print("Running model predictions...")
    X_pred = model.predict(X_seq, batch_size=BATCH_SIZE)
    errors = np.mean(np.square(X_seq - X_pred), axis=(1, 2))
    threshold = np.percentile(errors[y_seq == 0], 95)
    print(f"Computed threshold = {threshold}")
    y_pred = (errors > threshold).astype(int)
    accuracy = np.mean(y_pred == y_seq)
    print(f"Evaluation accuracy: {accuracy * 100:.2f}%")
    return errors, y_pred, threshold

def plot_error_distribution(errors, threshold):
    plt.hist(errors, bins=100, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--')
    plt.xlabel("Reconstruction error")
    plt.ylabel("Count")
    plt.show()

# ------------------------- Main -------------------------
def main():
    print("Loading dataset...")
    df = load_csvs(DATA_DIR)

    print("Preprocessing...")
    X, y = preprocess_df(df)

    print("Building sequences...")
    X_seq, y_seq = build_sequences(X, y)

    print("Training autoencoder...")
    model, history = train_autoencoder(X_seq, y_seq)

    print("Evaluating model...")
    errors, y_pred, threshold = evaluate_model(model, X_seq, y_seq)
    plot_error_distribution(errors, threshold)

    # Save model + threshold in TensorFlow SavedModel format ✅
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_TF_PATH = os.path.join(MODEL_DIR, "cnn_lstm_autoencoder_tf")
    print("Saving TensorFlow model...")
    model.save(MODEL_TF_PATH)
    threshold_path = os.path.join(MODEL_DIR, "threshold.npy")
    np.save(threshold_path, threshold)
    print(f"TensorFlow model saved → {MODEL_TF_PATH}")
    print(f"Threshold saved → {threshold_path}")

if __name__ == "__main__":
    main()