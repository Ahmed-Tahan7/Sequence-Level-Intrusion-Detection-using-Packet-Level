# live_analyzer.py
"""
Live packet-to-sequence analyzer that uses a trained CNN-LSTM autoencoder
to detect anomalous sequences of network flows.

Usage (Windows):
  - Run as Administrator:  python live_analyzer.py --iface "Ethernet"

Usage (Linux/macOS):
  - Run as root: sudo python3 live_analyzer.py --iface eth0

Notes:
  - Requires scapy (packet capture), tensorflow (model), numpy, pandas, joblib (optional).
  - Model and threshold must be in ./models/ (cnn_lstm_autoencoder.h5, threshold.npy).
  - If you saved a scaler (models/scaler.pkl) it will be loaded and used.
"""

import os
import time
import argparse
import collections
import threading
import numpy as np
import pandas as pd

# packet capture
from scapy.all import sniff, IP, TCP, UDP, Raw

# ML
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib  # optional (scaler)

# ----------------- Config -----------------
SEQ_LEN = 20            # sequence length expected by model
STRIDE = 1              # stride for sliding sequences
FLOW_TIMEOUT = 60.0     # seconds to consider a flow finished if no packets (eviction)
REPORT_EVERY = 1.0      # seconds between checks

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_lstm_autoencoder.h5")
THRESH_PATH = os.path.join(MODEL_DIR, "threshold.npy")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")  # optional

# Feature names (simple, compatible with training pipeline which used numeric features)
# IMPORTANT: adapt these to exactly match the features you trained your model on.
FEATURE_COLUMNS = [
    "pkt_count", "byte_count", "duration", "avg_pkt_size",
    "min_pkt_size", "max_pkt_size", "src_port", "dst_port",
    "proto_num", "tcp_flags"   # tcp_flags is an int summary; UDP will be 0
]

# ----------------- Utilities -----------------
def now_ts():
    return time.time()

# Convert protocol name to number (to keep numeric features)
PROTO_MAP = {"tcp": 6, "udp": 17, "icmp": 1}

def proto_to_num(p):
    try:
        return int(p)
    except Exception:
        return PROTO_MAP.get(str(p).lower(), 0)

def summarize_tcp_flags(packet):
    """Return an integer representing flags set in TCP."""
    if not packet.haslayer(TCP):
        return 0
    flags = packet[TCP].flags
    # flags is a string like 'S' or 'PA' in scapy; map to bits
    mapping = {"F":1, "S":2, "R":4, "P":8, "A":16, "U":32, "E":64, "C":128}
    total = 0
    for ch in str(flags):
        total |= mapping.get(ch, 0)
    return total

# ----------------- Flow store -----------------
class Flow:
    def __init__(self, srcip, dstip, sport, dport, proto):
        self.srcip = srcip
        self.dstip = dstip
        self.sport = sport
        self.dport = dport
        self.proto = proto
        self.start = now_ts()
        self.last = self.start
        self.pkt_count = 0
        self.byte_count = 0
        self.sizes = []
        self.tcp_flags_accum = 0

    def add_packet(self, pkt_len, ts, tcp_flags=0):
        self.pkt_count += 1
        self.byte_count += pkt_len
        self.sizes.append(pkt_len)
        self.tcp_flags_accum |= tcp_flags
        self.last = ts

    def to_feature_vector(self):
        duration = max(0.0, self.last - self.start)
        avg_size = np.mean(self.sizes) if self.sizes else 0.0
        min_size = float(np.min(self.sizes)) if self.sizes else 0.0
        max_size = float(np.max(self.sizes)) if self.sizes else 0.0
        proto_num = proto_to_num(self.proto)
        return np.array([
            self.pkt_count,
            self.byte_count,
            duration,
            avg_size,
            min_size,
            max_size,
            int(self.sport) if self.sport is not None else 0,
            int(self.dport) if self.dport is not None else 0,
            proto_num,
            int(self.tcp_flags_accum)
        ], dtype=np.float32)

# ----------------- Flow Manager -----------------
class FlowManager:
    def __init__(self, timeout=FLOW_TIMEOUT):
        self.flows = {}  # key -> Flow
        self.lock = threading.Lock()
        self.timeout = timeout
        self.recent_flow_vectors = collections.deque(maxlen=200_000)  # store recent finished flows as vectors

    def _key_from_pkt(self, pkt):
        # 5-tuple key: srcip, dstip, sport, dport, proto
        if IP not in pkt:
            return None
        ip = pkt[IP]
        src = ip.src
        dst = ip.dst
        proto = ip.proto
        sport = None
        dport = None
        if pkt.haslayer(TCP):
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        return (src, dst, sport, dport, proto)

    def add_packet(self, pkt):
        key = self._key_from_pkt(pkt)
        if key is None:
            return
        ts = now_ts()
        pkt_len = len(pkt)
        tcp_flags = summarize_tcp_flags(pkt)

        with self.lock:
            if key not in self.flows:
                self.flows[key] = Flow(*key)
            flow = self.flows[key]
            flow.add_packet(pkt_len, ts, tcp_flags)

    def evict_old_flows(self):
        """Move timed out flows to recent_flow_vectors for sequence building."""
        now = now_ts()
        moved = 0
        with self.lock:
            keys = list(self.flows.keys())
            for k in keys:
                f = self.flows[k]
                if now - f.last > self.timeout:
                    vec = f.to_feature_vector()
                    self.recent_flow_vectors.append(vec)
                    del self.flows[k]
                    moved += 1
        return moved

    def get_recent_vectors(self):
        """Return numpy array of recent finished flow vectors."""
        with self.lock:
            if not self.recent_flow_vectors:
                return np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32)
            arr = np.stack(list(self.recent_flow_vectors), axis=0)
            return arr

# ----------------- Model helper -----------------
def load_detection_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    threshold = None
    if os.path.exists(THRESH_PATH):
        threshold = np.load(THRESH_PATH)
    scaler = None
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print("Loaded scaler from", SCALER_PATH)
        except Exception as e:
            print("Warning: failed to load scaler:", e)
            scaler = None
    print("Model and threshold loaded.")
    return model, threshold, scaler

# ----------------- Sequence builder -----------------
def build_sliding_sequences_from_vectors(vectors, seq_len=SEQ_LEN, stride=STRIDE):
    """Given recent flow-level vectors (N, d) build sliding sequences (M, seq_len, d)."""
    n = vectors.shape[0]
    if n < seq_len:
        return np.zeros((0, seq_len, vectors.shape[1]), dtype=np.float32)
    count = 1 + (n - seq_len) // stride
    seqs = np.zeros((count, seq_len, vectors.shape[1]), dtype=np.float32)
    idx = 0
    for start in range(0, n - seq_len + 1, stride):
        seqs[idx] = vectors[start:start + seq_len]
        idx += 1
    return seqs

# ----------------- Packet sniffing thread -----------------
def start_sniffing(iface, flow_mgr, stop_event):
    def pkt_callback(pkt):
        flow_mgr.add_packet(pkt)
    print(f"Starting sniff on iface={iface} ... (press Ctrl+C to stop)")
    sniff(iface=iface, prn=pkt_callback, store=False, stop_filter=lambda x: stop_event.is_set())

# ----------------- Main live loop -----------------
def run_live_analyzer(iface):
    # Load model + threshold + scaler
    model, threshold, scaler = load_detection_model()

    flow_mgr = FlowManager(timeout=FLOW_TIMEOUT)
    stop_event = threading.Event()

    # Start sniffing thread
    t = threading.Thread(target=start_sniffing, args=(iface, flow_mgr, stop_event), daemon=True)
    t.start()

    try:
        last_checked = now_ts()
        while True:
            time.sleep(REPORT_EVERY)
            moved = flow_mgr.evict_old_flows()
            if moved:
                print(f"Evicted {moved} finished flows -> added to buffer (size now {len(flow_mgr.recent_flow_vectors)})")
            # build sequences from recent finished flows
            vectors = flow_mgr.get_recent_vectors()
            if vectors.shape[0] < SEQ_LEN:
                # not enough flows yet
                continue

            seqs = build_sliding_sequences_from_vectors(vectors, seq_len=SEQ_LEN, stride=STRIDE)
            # optionally scale with scaler
            ns, s_len, d = seqs.shape
            flat = seqs.reshape((ns * s_len, d))
            if scaler is not None:
                # scaler expects shape (n_samples, n_features)
                flat = scaler.transform(flat)
            else:
                # If no scaler, normalize per-feature by simple standardization on the fly
                # (mean/std across the window) — crude but better than nothing
                mu = flat.mean(axis=0, keepdims=True)
                sigma = flat.std(axis=0, keepdims=True) + 1e-6
                flat = (flat - mu) / sigma

            seqs_scaled = flat.reshape((ns, s_len, d))

            # Predict reconstructions. Model input shape: (batch, seq_len, features)
            recon = model.predict(seqs_scaled, batch_size=64, verbose=0)
            errs = np.mean((seqs_scaled - recon) ** 2, axis=(1,2))

            # Use loaded threshold if available; otherwise compute on the fly from the lower half of errors
            if threshold is None:
                # choose median + multiplier (ad-hoc)
                thr = np.percentile(errs, 95)
            else:
                thr = float(threshold)

            # Report anomalies
            anomaly_indices = np.where(errs > thr)[0]
            if anomaly_indices.size:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[{ts}] Detected {anomaly_indices.size} anomalous sequences (threshold={thr:.6g}).")
                # Print brief info for each anomaly: last flow vector in sequence
                for i in anomaly_indices:
                    seq = seqs_scaled[i]
                    last_flow = seq[-1]  # numeric vector of last flow in sequence
                    print(f"  - seq_idx={i}, error={errs[i]:.6g}, last_flow=[{', '.join(f'{v:.3g}' for v in last_flow[:6])} ...]")
            # else no anomalies
    except KeyboardInterrupt:
        print("Stopping sniffing...")
        stop_event.set()
        t.join(timeout=2)
        print("Stopped.")
    except Exception as e:
        print("Error in live analyzer:", e)
        stop_event.set()
        t.join(timeout=2)


# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live packet analyzer -> CNN-LSTM autoencoder anomaly detection")
    parser.add_argument("--iface", type=str, default=None, help="Network interface to sniff (required)")
    args = parser.parse_args()

    if args.iface is None:
        print("You must specify an interface: e.g. --iface eth0 (or 'Ethernet' on Windows)")
        raise SystemExit(1)

    run_live_analyzer(args.iface)