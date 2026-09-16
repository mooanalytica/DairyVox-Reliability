
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

EPS = 1e-12


# ----------------------------------------------------------------- metrics
def expected_calibration_error(y_true, confidence, prediction, n_bins=10):
    """Top-label expected calibration error over equal-width confidence bins."""
    y_true = np.asarray(y_true)
    confidence = np.asarray(confidence)
    prediction = np.asarray(prediction)
    correct = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def brier_binary(y_true, p1):
    y_true = np.asarray(y_true, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    return float(np.mean((p1 - y_true) ** 2))


def brier_multiclass(y_true, proba, n_classes):
    y_true = np.asarray(y_true)
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def log_loss_safe(y_true, proba, n_classes):
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba), EPS, 1.0)
    onehot = np.eye(n_classes)[y_true]
    return float(-np.mean(np.sum(onehot * np.log(proba), axis=1)))


# ----------------------------------------------------------------- binary calibrators
def _fit_temperature_binary(p1_train, y_train):
    z = np.log(np.clip(p1_train, EPS, 1 - EPS) / np.clip(1 - p1_train, EPS, 1 - EPS))

    def nll(t):
        p = 1.0 / (1.0 + np.exp(-z / t))
        p = np.clip(p, EPS, 1 - EPS)
        return -np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return res.x


def calibrate_binary(p1_train, y_train, p1_apply, method):
    """Return calibrated positive-class probabilities for the apply split."""
    if method == "raw":
        return np.asarray(p1_apply)
    if method == "platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(np.asarray(p1_train).reshape(-1, 1), y_train)
        return lr.predict_proba(np.asarray(p1_apply).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(np.asarray(p1_train), y_train)
        return iso.predict(np.asarray(p1_apply))
    if method == "temperature":
        t = _fit_temperature_binary(np.asarray(p1_train), np.asarray(y_train))
        z = np.log(np.clip(p1_apply, EPS, 1 - EPS) / np.clip(1 - np.asarray(p1_apply), EPS, 1 - EPS))
        return 1.0 / (1.0 + np.exp(-z / t))
    raise ValueError(method)


# ----------------------------------------------------------------- multiclass calibrators
def _fit_temperature_multiclass(proba_train, y_train):
    logits = np.log(np.clip(proba_train, EPS, 1.0))

    def nll(t):
        scaled = logits / t
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled)
        p = p / p.sum(axis=1, keepdims=True)
        p = np.clip(p, EPS, 1.0)
        return -np.mean(np.log(p[np.arange(len(y_train)), y_train]))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return res.x


def calibrate_multiclass(proba_train, y_train, proba_apply, method, n_classes):
    proba_train = np.asarray(proba_train)
    proba_apply = np.asarray(proba_apply)
    if method == "raw":
        return proba_apply
    if method == "temperature":
        t = _fit_temperature_multiclass(proba_train, np.asarray(y_train))
        logits = np.log(np.clip(proba_apply, EPS, 1.0)) / t
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)
    if method in ("ovr_platt", "ovr_isotonic"):
        out = np.zeros_like(proba_apply)
        for k in range(n_classes):
            yk = (np.asarray(y_train) == k).astype(int)
            if method == "ovr_platt":
                out[:, k] = calibrate_binary(proba_train[:, k], yk, proba_apply[:, k], "platt")
            else:
                out[:, k] = calibrate_binary(proba_train[:, k], yk, proba_apply[:, k], "isotonic")
        row = out.sum(axis=1, keepdims=True)
        return out / np.clip(row, EPS, None)
    raise ValueError(method)


# ----------------------------------------------------------------- group-aware application
def group_aware_binary(farms, p1, y, method):
    """Calibrate each farm's positive probabilities using the other farms."""
    farms = np.asarray(farms)
    p1 = np.asarray(p1, dtype=float)
    y = np.asarray(y, dtype=int)
    out = np.zeros_like(p1)
    for f in np.unique(farms):
        tr = farms != f
        ap = farms == f
        out[ap] = calibrate_binary(p1[tr], y[tr], p1[ap], method)
    return out


def group_aware_multiclass(farms, proba, y, method, n_classes):
    farms = np.asarray(farms)
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    out = np.zeros_like(proba)
    for f in np.unique(farms):
        tr = farms != f
        ap = farms == f
        out[ap] = calibrate_multiclass(proba[tr], y[tr], proba[ap], method, n_classes)
    return out


BINARY_METHODS = ["raw", "platt", "isotonic", "temperature"]
MULTICLASS_METHODS = ["raw", "temperature", "ovr_platt", "ovr_isotonic"]


# ----------------------------------------------------------------- selective prediction
def selective_prediction(y_true, y_pred, confidence, coverages, average="macro"):
    """Retained score and achieved abstention across coverage targets. At each
    target the least-confident clips are withheld down to the coverage level."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence)
    order = np.argsort(-confidence)
    n = len(y_true)
    rows = []
    for cov in coverages:
        k = int(round(cov * n))
        k = max(k, 1)
        keep = order[:k]
        rows.append({
            "coverage_target": cov,
            "coverage_achieved": k / n,
            "abstention_rate": 1.0 - k / n,
            "retained_macro_f1": float(f1_score(y_true[keep], y_pred[keep], average=average, zero_division=0)),
            "n_retained": k,
        })
    return rows
