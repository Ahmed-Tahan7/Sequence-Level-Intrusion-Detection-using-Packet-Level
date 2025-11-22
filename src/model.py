import tensorflow as tf
from tensorflow.keras import layers, models
# -------------------------------------------------------------
# 1D-CNN + LSTM Autoencoder for Sequence-Level Intrusion Detection
# -------------------------------------------------------------

def build_cnn_lstm_autoencoder(seq_len: int, n_features: int, latent_dim: int = 64):
    """
    Build a hybrid 1D-CNN + LSTM autoencoder.

    Input shape: (seq_len, n_features)
    Output: reconstructed sequence with same shape
    """

    inputs = layers.Input(shape=(seq_len, n_features))

    # ----------------------
    # Encoder
    # ----------------------
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # LSTM Encoder
    x = layers.LSTM(128, return_sequences=True)(x)
    encoded = layers.LSTM(latent_dim, return_sequences=False, name="latent_vector")(x)

    # ----------------------
    # Decoder
    # ----------------------
    x = layers.RepeatVector(seq_len // 2)(encoded)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)

    x = layers.Conv1DTranspose(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv1DTranspose(filters=64, kernel_size=3, padding="same", activation="relu")(x)

    # Final reconstruction
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = models.Model(inputs, outputs, name="cnn_lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")

    return model


# -------------------------------------------------------------
# Helper function to summarize model
# -------------------------------------------------------------

def print_model_summary(model):
    print("\n===== MODEL SUMMARY =====")
    model.summary()
    print("==========================\n")
