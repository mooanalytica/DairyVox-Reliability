from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

EPS = 1e-12


def expected_calibration_error(y_true, confidence, prediction, n_bins=10):
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


def _fit_temperature_binary(p1_train, y_train):
    z = np.log(np.clip(p1_train, EPS, 1 - EPS) / np.clip(1 - p1_train, EPS, 1 - EPS))

    def nll(t):
        p = 1.0 / (1.0 + np.exp(-z / t))
        p = np.clip(p, EPS, 1 - EPS)
        return -np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return res.x


def calibrate_binary(p1_train, y_train, p1_apply, method):
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


def group_aware_binary(farms, p1, y, method):
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


def selective_prediction(y_true, y_pred, confidence, coverages, average="macro"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence)
    order = np.argsort(-confidence, kind="stable")
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


def composite_pick(rows, keys=("ece", "brier", "logloss")):
    ranked = {k: sorted(range(len(rows)), key=lambda i: rows[i][k]) for k in keys}
    score = [0] * len(rows)
    for k in keys:
        for r, idx in enumerate(ranked[k]):
            score[idx] += r
    return int(np.argmin(score))


def probability_quality(y, P, task, n_classes):
    return {"ece": expected_calibration_error(y, P.max(1), P.argmax(1)),
            "brier": brier_binary(y, P[:, 1]) if task == "binary" else brier_multiclass(y, P, n_classes),
            "logloss": log_loss_safe(y, P, n_classes)}


def _calibrate(P_fit, y_fit, P_apply, method, task, n_classes):
    if task == "binary":
        p1 = calibrate_binary(P_fit[:, 1], y_fit, P_apply[:, 1], method)
        return np.c_[1 - p1, p1]
    return calibrate_multiclass(P_fit, y_fit, P_apply, method, n_classes)


def _group_aware(farms, P, y, method, task, n_classes):
    if task == "binary":
        p1 = group_aware_binary(farms, P[:, 1], y, method)
        return np.c_[1 - p1, p1]
    return group_aware_multiclass(farms, P, y, method, n_classes)


def nested_fold_reliability(inner, outer, task, n_classes, coverage=0.80):
    methods = BINARY_METHODS if task == "binary" else MULTICLASS_METHODS
    Pi, yi, fi = inner["P"], inner["y"], inner["farms"]
    cands = []
    for m in methods:
        CPi = _group_aware(fi, Pi, yi, m, task, n_classes)
        cands.append({"method": m, "CP": CPi, **probability_quality(yi, CPi, task, n_classes)})
    best = cands[composite_pick(cands)]

    def r80(P):
        return next(r for r in selective_prediction(yi, P.argmax(1), P.max(1), [coverage]))["retained_macro_f1"]

    inner_raw80, inner_cal80 = r80(Pi), r80(best["CP"])
    score = "calibrated" if inner_cal80 > inner_raw80 else "raw"

    CPo = _calibrate(Pi, yi, outer["P"], best["method"], task, n_classes)
    Po = CPo if score == "calibrated" else outer["P"]
    inner_conf = (best["CP"] if score == "calibrated" else Pi).max(1)
    threshold = float(np.quantile(inner_conf, 1.0 - coverage))
    return {"method": best["method"], "score": score, "calibrated_P": CPo, "selected_P": Po,
            "inner_candidates": [{k: v for k, v in c.items() if k != "CP"} for c in cands],
            "inner_retained_raw": inner_raw80, "inner_retained_calibrated": inner_cal80,
            "threshold": threshold}


def deferred_mask(confidence, coverage=0.80):
    confidence = np.asarray(confidence)
    n = len(confidence)
    order = np.argsort(-confidence, kind="stable")
    mask = np.ones(n, dtype=bool)
    mask[order[:max(int(round(coverage * n)), 1)]] = False
    return mask


def standardise_predictions(df, target_col, held_col, inner_col=None):
    out = pd.DataFrame({"target": df[target_col], "held": df[held_col].astype(str).str.extract(r"(\d)")[0],
                        "_record_key": df["_record_key"], "y": df["y_true"].astype(int),
                        "task": df["task_type"]})
    if inner_col is not None:
        out["inner_val_farm"] = df[inner_col].astype(str).str.extract(r"(\d)")[0]
    for c in df.columns:
        if c.startswith("proba__"):
            out[c] = df[c]
    return out


def prediction_matrix(sub):
    if sub["task"].iloc[0] == "binary":
        p1 = sub["proba__1"].to_numpy()
        return np.c_[1 - p1, p1], ["0", "1"]
    cols = sorted(c for c in sub.columns if c.startswith("proba__") and c != "proba__1" and sub[c].notna().any())
    return sub[cols].to_numpy(), [c[len("proba__"):] for c in cols]


def nested_reliability(outer, inner, target, coverage=0.80):
    o = outer[outer.target == target]
    i = inner[inner.target == target]
    task = o["task"].iloc[0]
    parts, folds = [], []
    for held in sorted(o["held"].unique()):
        oo, ii = o[o.held == held].reset_index(drop=True), i[i.held == held].reset_index(drop=True)
        Po, classes = prediction_matrix(oo)
        Pi, classes_i = prediction_matrix(ii)
        assert classes == classes_i
        res = nested_fold_reliability({"P": Pi, "y": ii["y"].to_numpy(), "farms": ii["inner_val_farm"].to_numpy()},
                                      {"P": Po}, task, len(classes), coverage)
        best = next(c for c in res["inner_candidates"] if c["method"] == res["method"])
        folds.append({"target": target, "heldout_farm": held, "selected_method": res["method"],
                      "score_version": res["score"], "n_inner_clips": len(ii),
                      "learned_threshold": res["threshold"],
                      "inner_ece_raw": res["inner_candidates"][0]["ece"], "inner_ece_selected": best["ece"],
                      "inner_retained_macro_f1_raw": res["inner_retained_raw"],
                      "inner_retained_macro_f1_calibrated": res["inner_retained_calibrated"]})
        parts.append({"clip_ids": oo["_record_key"].to_numpy(), "farms": np.full(len(oo), int(held)),
                      "y": oo["y"].to_numpy(), "raw_P": Po, "cal_P": res["calibrated_P"], "sel_P": res["selected_P"],
                      "threshold": np.full(len(oo), res["threshold"])})
    pooled = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    pooled.update(task=task, classes=classes, folds=pd.DataFrame(folds))
    return pooled


def pooled_selection(outer, target, coverage=0.80):
    o = outer[outer.target == target].reset_index(drop=True)
    P, classes = prediction_matrix(o)
    task, y, farms, n = o["task"].iloc[0], o["y"].to_numpy(), o["held"].to_numpy(), len(classes)
    methods = BINARY_METHODS if task == "binary" else MULTICLASS_METHODS
    cands = []
    for m in methods:
        CP = _group_aware(farms, P, y, m, task, n)
        cands.append({"method": m, "CP": CP, **probability_quality(y, CP, task, n)})
    best = cands[composite_pick(cands)]
    r80 = {tag: next(r for r in selective_prediction(y, CP.argmax(1), CP.max(1), [coverage]))
           for tag, CP in (("raw", P), ("calibrated", best["CP"]))}
    tag = max(r80, key=lambda t: r80[t]["retained_macro_f1"])
    return {"target": target, "selected_method": best["method"], "ece_raw": cands[0]["ece"],
            "ece_selected": best["ece"], "score_version": tag,
            "retained_macro_f1_80cov": r80[tag]["retained_macro_f1"]}


def summarise_model(outer, inner, targets, coverages, coverage=0.80):
    store, calib, select, curves, folds, pooled_rows = {}, [], [], [], [], []
    for target in targets:
        r = nested_reliability(outer, inner, target, coverage)
        store[target] = r
        y, n = r["y"], len(r["classes"])
        q_raw = probability_quality(y, r["raw_P"], r["task"], n)
        q_cal = probability_quality(y, r["cal_P"], r["task"], n)
        f = r["folds"]
        calib.append({"target": target,
                      "selected_method": ";".join(f"{h}:{m}" for h, m in zip(f.heldout_farm, f.selected_method)),
                      "ece_raw": q_raw["ece"], "ece_selected": q_cal["ece"],
                      "brier_raw": q_raw["brier"], "brier_selected": q_cal["brier"],
                      "logloss_raw": q_raw["logloss"], "logloss_selected": q_cal["logloss"]})
        for tag, P in (("raw", r["raw_P"]), ("calibrated", r["cal_P"]), ("selected", r["sel_P"])):
            for row in selective_prediction(y, P.argmax(1), P.max(1), coverages):
                curves.append({"target": target, "score": tag, **row})
        sP = r["sel_P"]
        base = f1_score(y, sP.argmax(1), average="macro", zero_division=0)
        at = next(x for x in selective_prediction(y, sP.argmax(1), sP.max(1), [coverage]))
        kept = sP.max(1) >= r["threshold"]
        select.append({"target": target,
                       "score_version": ";".join(f"{h}:{s}" for h, s in zip(f.heldout_farm, f.score_version)),
                       "baseline_macro_f1": base, "retained_macro_f1_80cov": at["retained_macro_f1"],
                       "selective_gain": at["retained_macro_f1"] - base,
                       "abstention_rate": at["abstention_rate"],
                       "learned_threshold_coverage": float(kept.mean()),
                       "learned_threshold_retained_macro_f1":
                           f1_score(y[kept], sP.argmax(1)[kept], average="macro", zero_division=0),
                       "learned_threshold_coverage_by_farm": ";".join(
                           f"{h}:{kept[r['farms'] == h].mean():.3f}" for h in sorted(set(r["farms"].tolist())))})
        folds.append(f)
        pooled_rows.append(pooled_selection(outer, target, coverage))
    return (store, pd.DataFrame(calib), pd.DataFrame(select), pd.DataFrame(curves),
            pd.concat(folds, ignore_index=True), pd.DataFrame(pooled_rows))
