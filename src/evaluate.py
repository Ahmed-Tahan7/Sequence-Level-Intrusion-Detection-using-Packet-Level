import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# from src.model import build_cnn_lstm_autoencoder
# from src.dataset_builder import build_sequences_from_array, create_sequence_labels


def compute_reconstruction_errors(model, X_sequences):
    """
    Compute MSE reconstruction error for each sequence.
    """
    recon = model.predict(X_sequences, verbose=0)
    errors = np.mean(np.square(X_sequences - recon), axis=(1, 2))
    return errors


def choose_threshold(errors, factor: float = 1.0):
    """
    Choose anomaly threshold = mean + factor * std
    """
    return errors.mean() + factor * errors.std()


def evaluate_model(model, X_seq, y_seq, threshold=None):
    """
    Compute errors, generate predictions, and print evaluation metrics.
    """
    errors = compute_reconstruction_errors(model, X_seq)

    if threshold is None:
        threshold = choose_threshold(errors)

    y_pred = (errors > threshold).astype(int)

    print("\n===== Evaluation Metrics =====")
    print(classification_report(y_seq, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_seq, y_pred))

    try:
        auc = roc_auc_score(y_seq, errors)
        print(f"ROC-AUC: {auc:.4f}")
    except:
        print("ROC-AUC could not be computed.")

    return errors, y_pred, threshold


def plot_error_distribution(errors, threshold):
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=50)
    plt.axvline(threshold, color='red', linestyle='--', label='threshold')
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()