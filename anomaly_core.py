"""
Core logic for the Network Anomaly Detection project.

Pipeline:  flow records -> feature matrix -> ML model -> anomaly score
           -> alert (score above threshold) -> human-readable reason

Two models are supported:
  * Isolation Forest  - unsupervised, trained on NORMAL traffic only
  * Random Forest     - supervised, needs labelled data

Everything here is plain pandas / numpy / scikit-learn so it can be tested
without the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
NUMERIC = [
    "duration",          # flow duration (seconds)
    "fwd_packets",       # packets source -> destination
    "bwd_packets",       # packets destination -> source
    "src_bytes",         # bytes source -> destination
    "dst_bytes",         # bytes destination -> source
    "pkt_rate",          # packets per second
    "bytes_per_packet",  # average packet size
    "unique_dst_ports",  # distinct destination ports touched by the source
    "conn_rate",         # connections per second from the source
    "syn_ratio",         # share of SYN packets (0..1)
]
PROTOCOLS = ["TCP", "UDP", "ICMP"]
META = ["timestamp", "src_ip", "dst_port", "protocol"]
SCHEMA = META + NUMERIC
ATTACK_TYPES = ["DoS", "PortScan", "BruteForce"]

NICE = {
    "duration": "flow duration",
    "fwd_packets": "forward packets",
    "bwd_packets": "backward packets",
    "src_bytes": "source bytes",
    "dst_bytes": "destination bytes",
    "pkt_rate": "packet rate",
    "bytes_per_packet": "bytes per packet",
    "unique_dst_ports": "distinct ports probed",
    "conn_rate": "connection rate",
    "syn_ratio": "SYN ratio",
}


# --------------------------------------------------------------------------
# Synthetic traffic generator
# --------------------------------------------------------------------------
def _finish(rng, proto, duration, fwd, bwd, src_b, dst_b, ports, conn, syn, dport, stealth_p=0.0):
    """Apply optional 'low and slow' stealth, then compute derived columns."""
    n = len(fwd)
    if stealth_p > 0:
        s = rng.random(n) < stealth_p
        k = int(s.sum())
        if k:
            fwd[s] = np.maximum(1, fwd[s] / rng.uniform(5, 12, k))
            bwd[s] = np.maximum(0, bwd[s] / rng.uniform(5, 12, k))
            src_b[s] = src_b[s] / rng.uniform(5, 12, k)
            dst_b[s] = dst_b[s] / rng.uniform(5, 12, k)
            conn[s] = conn[s] / rng.uniform(5, 12, k)
            ports[s] = np.maximum(1, ports[s] / rng.uniform(3, 8, k))
            duration[s] = duration[s] * rng.uniform(2, 6, k)
    fwd = np.maximum(1, np.round(fwd))
    bwd = np.maximum(0, np.round(bwd))
    total_pk = fwd + bwd
    return pd.DataFrame(
        {
            "protocol": proto,
            "dst_port": dport.astype(int),
            "duration": duration,
            "fwd_packets": fwd,
            "bwd_packets": bwd,
            "src_bytes": np.maximum(0, src_b),
            "dst_bytes": np.maximum(0, dst_b),
            "pkt_rate": total_pk / np.maximum(duration, 1e-3),
            "bytes_per_packet": (np.maximum(0, src_b) + np.maximum(0, dst_b)) / total_pk,
            "unique_dst_ports": np.round(np.maximum(1, ports)),
            "conn_rate": conn,
            "syn_ratio": np.clip(syn, 0, 1),
        }
    )


def _normal(rng, n):
    proto = rng.choice(PROTOCOLS, n, p=[0.80, 0.18, 0.02])
    duration = np.clip(rng.lognormal(-0.3, 1.0, n), 0.001, 300)
    fwd = np.maximum(1, rng.lognormal(2.2, 0.7, n)).round()
    bwd = (fwd * rng.uniform(0.6, 1.4, n)).round()
    src_b = fwd * np.clip(rng.normal(500, 180, n), 40, 1400)
    dst_b = bwd * np.clip(rng.normal(900, 350, n), 40, 1500)
    ports = rng.poisson(0.3, n) + 1.0
    conn = rng.lognormal(0.4, 0.6, n)
    syn = rng.beta(1, 8, n)
    dport = rng.choice([80, 443, 53, 22, 25, 8080], n, p=[0.30, 0.40, 0.15, 0.07, 0.04, 0.04])
    # ~2 % benign bulk transfers (backups, big downloads): unusual but harmless.
    bulk = rng.random(n) < 0.02
    k = int(bulk.sum())
    if k:
        fwd[bulk] = rng.lognormal(6.2, 0.4, k).round()
        bwd[bulk] = (fwd[bulk] * rng.uniform(0.3, 0.6, k)).round()
        src_b[bulk] = fwd[bulk] * rng.normal(1200, 100, k)
        dst_b[bulk] = bwd[bulk] * rng.normal(60, 10, k)
        duration[bulk] = rng.lognormal(3.5, 0.4, k)
    return _finish(rng, proto, duration, fwd, bwd, src_b, dst_b, ports, conn, syn, dport)


def _dos(rng, n):
    proto = rng.choice(PROTOCOLS, n, p=[0.60, 0.30, 0.10])
    duration = np.clip(rng.lognormal(-0.5, 0.6, n), 0.001, 60)
    fwd = rng.lognormal(7.0, 0.6, n)
    bwd = fwd * rng.uniform(0.0, 0.05, n)
    src_b = fwd * np.clip(rng.normal(70, 25, n), 20, 200)
    dst_b = bwd * 60
    ports = np.ones(n)
    conn = rng.lognormal(4.0, 0.5, n)
    syn = rng.beta(8, 2, n)
    dport = rng.choice([80, 443], n)
    return _finish(rng, proto, duration, fwd, bwd, src_b, dst_b, ports, conn, syn, dport, stealth_p=0.10)


def _portscan(rng, n):
    proto = np.array(["TCP"] * n)
    duration = np.clip(rng.lognormal(-1.5, 0.5, n), 0.001, 20)
    fwd = rng.lognormal(3.0, 0.5, n)
    bwd = fwd * rng.uniform(0.0, 0.2, n)
    src_b = fwd * np.clip(rng.normal(44, 6, n), 20, 80)
    dst_b = bwd * np.clip(rng.normal(40, 6, n), 20, 80)
    ports = rng.lognormal(4.2, 0.4, n)
    conn = rng.lognormal(3.0, 0.5, n)
    syn = rng.beta(9, 1, n)
    dport = rng.integers(1, 65535, n)
    return _finish(rng, proto, duration, fwd, bwd, src_b, dst_b, ports, conn, syn, dport, stealth_p=0.15)


def _bruteforce(rng, n):
    proto = np.array(["TCP"] * n)
    duration = np.clip(rng.lognormal(1.5, 0.4, n), 0.01, 120)
    fwd = rng.lognormal(3.2, 0.3, n)
    bwd = fwd * rng.uniform(0.8, 1.2, n)
    src_b = fwd * np.clip(rng.normal(150, 40, n), 40, 400)
    dst_b = bwd * np.clip(rng.normal(180, 40, n), 40, 400)
    ports = np.ones(n)
    conn = rng.lognormal(2.3, 0.4, n)
    syn = rng.beta(3, 5, n)
    dport = rng.choice([22, 21, 3389], n, p=[0.6, 0.2, 0.2])
    return _finish(rng, proto, duration, fwd, bwd, src_b, dst_b, ports, conn, syn, dport, stealth_p=0.15)


_ATTACK_FUNCS = {"DoS": _dos, "PortScan": _portscan, "BruteForce": _bruteforce}


def generate_flows(n=8000, attack_ratio=0.12, seed=42, attack_types=None, start=None):
    """Create a labelled synthetic flow table (label 0 = normal, 1 = attack)."""
    rng = np.random.default_rng(seed)
    attack_types = list(attack_types) if attack_types else list(ATTACK_TYPES)
    n_att = int(round(n * attack_ratio))
    n_norm = n - n_att

    parts = [_normal(rng, n_norm).assign(label=0, attack_type="Normal")]
    if n_att > 0:
        split = np.array_split(np.arange(n_att), len(attack_types))
        for name, idx in zip(attack_types, split):
            if len(idx):
                parts.append(_ATTACK_FUNCS[name](rng, len(idx)).assign(label=1, attack_type=name))
    df = pd.concat(parts, ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

    start = pd.Timestamp.now().floor("s") if start is None else pd.Timestamp(start)
    df["timestamp"] = start + pd.to_timedelta(np.sort(rng.uniform(0, max(n, 1), len(df))), unit="s")
    df["timestamp"] = df["timestamp"].dt.floor("s")
    ips = [
        f"192.168.{a}.{b}" if lab == 0 else f"185.220.{a}.{b}"
        for lab, a, b in zip(df["label"], rng.integers(0, 4, len(df)), rng.integers(2, 250, len(df)))
    ]
    df["src_ip"] = ips
    return df[SCHEMA + ["label", "attack_type"]]


# --------------------------------------------------------------------------
# Loading real data (best effort converter for CIC-IDS2017-style CSV files)
# --------------------------------------------------------------------------
def from_cicids(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a CIC-IDS2017 style flow table into this project's schema.
    Best effort: CIC-IDS2017 has no per-source port or connection counters,
    so 'unique_dst_ports' is set to 1 and 'conn_rate' reuses the flow packet rate.
    """
    d = raw.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    need = ["flow duration", "total fwd packets", "total backward packets",
            "total length of fwd packets", "total length of bwd packets"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise ValueError(f"Not a CIC-IDS2017 style file, missing columns: {missing}")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
    dur = np.maximum(d["flow duration"].astype(float) / 1e6, 1e-6)
    fwd = d["total fwd packets"].astype(float)
    bwd = d["total backward packets"].astype(float)
    sb = d["total length of fwd packets"].astype(float)
    db = d["total length of bwd packets"].astype(float)
    tot = np.maximum(fwd + bwd, 1)
    proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
    proto = d["protocol"].map(proto_map).fillna("TCP") if "protocol" in d.columns else "TCP"
    syn = d["syn flag count"].astype(float) / tot if "syn flag count" in d.columns else 0.0
    out = pd.DataFrame(
        {
            "timestamp": pd.Timestamp.now().floor("s") + pd.to_timedelta(np.arange(len(d)), unit="s"),
            "src_ip": "unknown",
            "dst_port": d["destination port"].astype(int) if "destination port" in d.columns else 0,
            "protocol": proto.values if hasattr(proto, "values") else proto,
            "duration": dur.values,
            "fwd_packets": fwd.values,
            "bwd_packets": bwd.values,
            "src_bytes": sb.values,
            "dst_bytes": db.values,
            "pkt_rate": (tot / dur).values,
            "bytes_per_packet": ((sb + db) / tot).values,
            "unique_dst_ports": 1.0,
            "conn_rate": (tot / dur).values,
            "syn_ratio": np.clip(syn.values if hasattr(syn, "values") else syn, 0, 1),
        }
    )
    if "label" in d.columns:
        lab = d["label"].astype(str).str.strip()
        out["attack_type"] = lab.where(lab.str.lower() != "benign", "Normal").values
        out["label"] = (lab.str.lower() != "benign").astype(int).values
    return out.reset_index(drop=True)


def load_uploaded(raw: pd.DataFrame, cic_format: bool = False) -> pd.DataFrame:
    """Validate an uploaded CSV and return a frame in project schema."""
    if cic_format:
        df = from_cicids(raw)
    else:
        df = raw.copy()
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in NUMERIC + ["protocol"] if c not in df.columns]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.Timestamp.now().floor("s") + pd.to_timedelta(np.arange(len(df)), unit="s")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "src_ip" not in df.columns:
            df["src_ip"] = "unknown"
        if "dst_port" not in df.columns:
            df["dst_port"] = 0
        if "label" in df.columns:
            lab = df["label"]
            if lab.dtype == object:
                df["label"] = (~lab.astype(str).str.strip().str.lower().isin(["0", "normal", "benign"])).astype(int)
            else:
                df["label"] = (lab.astype(float) > 0).astype(int)
            if "attack_type" not in df.columns:
                df["attack_type"] = np.where(df["label"] == 1, "Attack", "Normal")
    df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=NUMERIC).reset_index(drop=True)
    df["protocol"] = df["protocol"].astype(str).str.upper()
    if len(df) < 200:
        raise ValueError("Need at least 200 valid rows.")
    return df


def split_data(df: pd.DataFrame, test_size=0.3, seed=42):
    strat = df["label"] if "label" in df.columns and df["label"].nunique() > 1 else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)


# --------------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------------
class Detector:
    """Isolation Forest (unsupervised) or Random Forest (supervised) detector."""

    ISO = "Isolation Forest"
    RF = "Random Forest"

    def __init__(self, kind=ISO, n_estimators=150, seed=42):
        if kind not in (self.ISO, self.RF):
            raise ValueError("kind must be 'Isolation Forest' or 'Random Forest'")
        self.kind, self.n_estimators, self.seed = kind, n_estimators, seed
        self.scaler = StandardScaler()
        self.model = None
        self.columns: list[str] = []
        self.train_scores: np.ndarray | None = None
        self.ref_median: pd.Series | None = None
        self.ref_scale: pd.Series | None = None

    # -- features ---------------------------------------------------------
    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        x = np.log1p(df[NUMERIC].astype(float).clip(lower=0))
        proto = df["protocol"].astype(str).str.upper()
        for p in PROTOCOLS:
            x[f"proto_{p}"] = (proto == p).astype(float).values
        return x

    # -- training ---------------------------------------------------------
    def fit(self, df: pd.DataFrame, y=None):
        X = self._matrix(df)
        self.columns = list(X.columns)
        y = None if y is None else np.asarray(y)
        if self.kind == self.ISO:
            normal = X if y is None else X[y == 0]
            if len(normal) < 50:
                raise ValueError("Not enough normal flows to learn from.")
            self.scaler.fit(normal)
            self.model = IsolationForest(
                n_estimators=self.n_estimators, max_samples=256, random_state=self.seed, n_jobs=-1
            ).fit(self.scaler.transform(normal))
            ref = np.log1p(df[NUMERIC].astype(float).clip(lower=0))
            ref = ref if y is None else ref[y == 0]
        else:
            if y is None or len(np.unique(y)) < 2:
                raise ValueError("Random Forest needs labelled data with both classes.")
            self.scaler.fit(X)
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators, class_weight="balanced", random_state=self.seed, n_jobs=-1
            ).fit(self.scaler.transform(X), y)
            ref = np.log1p(df[NUMERIC].astype(float).clip(lower=0))[y == 0]
        self.ref_median = ref.median()
        mad = (ref - self.ref_median).abs().median() * 1.4826
        self.ref_scale = mad.where(mad > 1e-6, ref.std().replace(0, 1.0)).fillna(1.0)
        # reference score distribution of normal training traffic
        norm_df = df if y is None else df[y == 0]
        self.train_scores = self.score(norm_df)
        return self

    # -- scoring ----------------------------------------------------------
    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Higher = more anomalous (0..1 for Random Forest)."""
        Xs = self.scaler.transform(self._matrix(df)[self.columns])
        if self.kind == self.ISO:
            return -self.model.score_samples(Xs)
        return self.model.predict_proba(Xs)[:, 1]

    def threshold(self, setting: float) -> float:
        """
        Isolation Forest: `setting` is the percentile of NORMAL training scores
        (0.97 -> about 3 % of normal flows would raise an alert).
        Random Forest: `setting` is the probability cut-off.
        """
        if self.kind == self.ISO:
            return float(np.quantile(self.train_scores, setting))
        return float(setting)

    # -- explanations -----------------------------------------------------
    def reasons(self, df: pd.DataFrame, top: int = 3) -> list[str]:
        """Short text listing the features that deviate most from normal traffic."""
        if len(df) == 0:
            return []
        lg = np.log1p(df[NUMERIC].astype(float).clip(lower=0))
        z = (lg - self.ref_median) / self.ref_scale
        out = []
        raw = df[NUMERIC].astype(float).reset_index(drop=True)
        zv = z.reset_index(drop=True)
        for i in range(len(zv)):
            row = zv.iloc[i].abs().sort_values(ascending=False).head(top).index
            bits = []
            for c in row:
                direction = "high" if zv.iloc[i][c] > 0 else "low"
                bits.append(f"{NICE[c]} {direction} ({_fmt(raw.iloc[i][c])})")
            out.append("; ".join(bits))
        return out


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.2f}"


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(y_true, score, pred) -> dict:
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, score) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "accuracy": accuracy_score(y_true, pred),
        "auc": auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def roc_points(y_true, score):
    fpr, tpr, _ = roc_curve(y_true, score)
    return fpr, tpr
