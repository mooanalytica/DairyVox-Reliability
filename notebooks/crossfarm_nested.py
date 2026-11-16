from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

K_VALUES = [32, 48, 64, 80, 96]
KSTUDY_SEEDS = [7, 19, 31, 43, 55]
KMEANS_BATCH = 2048
KMEANS_MAX_ITER = 200
KMEANS_N_INIT = 10
REASSIGNMENT_RATIO = 0.01
PCA_DIM = 128
SILHOUETTE_MAX_SAMPLES = 3000
FINAL_SEED = 31
SCORECARD_PCA_SEED = 31
KSTUDY_PCA_SEED = 0

SCORECARD_WEIGHTS = {
    "segmentation_improvement": 0.30,
    "fragment_control": 0.20,
    "clustering_quality": 0.20,
    "sequence_richness": 0.20,
    "confound_control": 0.10,
}
SCORECARD_PENALTIES = {
    "no_single_segment_improvement": 0.10,
    "fragment_explosion": 0.15,
    "confound_worsened": 0.10,
}
FRAGMENT_EXPLOSION_ABS_PCT = 25.0
FRAGMENT_EXPLOSION_DELTA_PCT = 7.0
CONFOUND_WORSENED_DELTA_PCT = 5.0

AUDIT_FIELDS = ["farm", "barn_zone", "microphone", "mic_placement_context"]
LOFO_AUDIT_FIELDS = ["barn_zone", "microphone", "mic_placement_context"]
AUDIT_FLAG_THRESHOLD = 0.80
AUDIT_HIGH_THRESHOLD = 0.90
AUDIT_MIN_SEGMENTS = 8

VOCAB_PCA_SEED = 31
STRUCTURAL_CONTEXT_COLS = ["barn_zone", "microphone", "mic_placement_context"]
FULL_CONTEXT_COLS = ["farm", "barn_zone", "microphone", "mic_placement_context", "day", "date"]
AUDIO_FAMILIES = ["unigram_props", "tfidf_1to3", "transition_probs"]
VOCAB_VARIANTS = ["full", "high_bias_filtered"]
CONTEXT_VARIANTS = ["audio_only", "plus_structural_context", "plus_full_context"]
CONTINUOUS_FAMILIES = ["pooled_mean", "pooled_mean_std"]
REPRESENTATIONS = ["units", "continuous"]
TARGET_SPECS = [
    {"target_name": "proxy_functional_v1", "task_type": "multiclass"},
    {"target_name": "is_nonvocal", "task_type": "binary"},
    {"target_name": "is_distress_related", "task_type": "binary"},
    {"target_name": "is_resource_related", "task_type": "binary"},
]
FARMS = [1, 2, 3]

SEGMENTATIONS = {
    "baseline": OUT / "02_segments",
    "candidate_A": OUT / "02b_segment_sweep" / "candidate_A",
    "candidate_B": OUT / "02b_segment_sweep" / "candidate_B",
    "candidate_C": OUT / "02b_segment_sweep" / "candidate_C",
}
SEGMENT_VECTORS = {
    "baseline": OUT / "03_segment_vectors" / "segment_vectors.npz",
    "candidate_A": OUT / "02b_segment_sweep" / "candidate_A" / "03_segment_vectors" / "segment_vectors.npz",
    "candidate_B": OUT / "02b_segment_sweep" / "candidate_B" / "03_segment_vectors" / "segment_vectors.npz",
    "candidate_C": OUT / "02b_segment_sweep" / "candidate_C" / "03_segment_vectors" / "segment_vectors.npz",
}
LAYER2_VECTORS = dict(SEGMENT_VECTORS, candidate_C=OUT / "03_segment_vectors_refined" / "segment_vectors.npz")
CANDIDATES = ["candidate_A", "candidate_B", "candidate_C"]


def load_clip_table():
    man = pd.read_csv(OUT / "00_manifest" / "manifest.csv")
    man["farm_num"] = pd.to_numeric(man["farm"], errors="coerce")
    pmap = pd.read_csv(OUT / "11_layer2_proxy_labels" / "proxy_label_mapping_table.csv")
    main2proxy = dict(zip(pmap["main_label_raw"], pmap["proxy_functional_v1"]))
    clips = man[["clip_id", "farm", "farm_num", "barn_zone", "microphone", "mic_placement_context",
                 "day", "date", "main_category_final"]].copy()
    clips["proxy_functional_v1"] = clips["main_category_final"].map(main2proxy)
    clips["is_nonvocal"] = (clips["proxy_functional_v1"] == "physiological_nonvocal").astype(int)
    clips["is_resource_related"] = (clips["proxy_functional_v1"] == "ingestion_resource").astype(int)
    clips["is_distress_related"] = (clips["proxy_functional_v1"] == "stress_discomfort").astype(int)
    return clips


def load_segmentation(name):
    root = SEGMENTATIONS[name]
    segments = pd.read_csv(root / "segments.csv")
    summary = pd.read_csv(root / "segment_summary.csv")
    vec = np.load(SEGMENT_VECTORS[name], allow_pickle=False)
    segments = segments.set_index(segments["segment_id"].astype(str)).loc[vec["segment_ids"].astype(str)]
    return {
        "segments": segments.reset_index(drop=True),
        "summary": summary,
        "vectors": vec["vectors"].astype(np.float32),
    }


def load_layer2_vectors(name):
    vec = np.load(LAYER2_VECTORS[name], allow_pickle=True)
    return {
        "vectors": vec["vectors"].astype(np.float32),
        "clip_ids": vec["clip_ids"].astype(str),
        "segment_index": vec["segment_index"].astype(int),
    }


def _cluster_matrix(vectors, pca_seed):
    scaled = StandardScaler().fit_transform(vectors)
    n_comp = min(PCA_DIM, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=pca_seed)
    return pca.fit_transform(scaled).astype(np.float32)


def _kmeans(X, k, seed):
    model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=KMEANS_BATCH,
                            max_iter=KMEANS_MAX_ITER, n_init=KMEANS_N_INIT,
                            reassignment_ratio=REASSIGNMENT_RATIO)
    labels = model.fit_predict(X)
    return model, labels


def _size_stats(labels, k):
    counts = np.bincount(labels, minlength=k)
    nonzero = counts[counts > 0]
    p = nonzero / nonzero.sum()
    ent = -np.sum(p * np.log(p + 1e-12))
    return {
        "min_cluster_size": int(nonzero.min()),
        "normalized_cluster_entropy": float(ent / np.log(len(nonzero))) if len(nonzero) > 1 else 0.0,
        "singleton_clusters": int(np.sum(nonzero == 1)),
    }


def _silhouette(X, labels, seed):
    if len(labels) > SILHOUETTE_MAX_SAMPLES:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(labels), size=SILHOUETTE_MAX_SAMPLES, replace=False))
        return float(silhouette_score(X[idx], labels[idx]))
    return float(silhouette_score(X, labels))


def kstudy(X):
    rows = []
    for k in K_VALUES:
        runs, per_seed = [], []
        for seed in KSTUDY_SEEDS:
            _, labels = _kmeans(X, k, seed)
            runs.append(labels)
            per_seed.append({
                "silhouette": _silhouette(X, labels, seed),
                "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
                "davies_bouldin": float(davies_bouldin_score(X, labels)),
                **_size_stats(labels, k),
            })
        aris = [adjusted_rand_score(runs[i], runs[j])
                for i in range(len(runs)) for j in range(i + 1, len(runs))]
        s = pd.DataFrame(per_seed)
        rows.append({
            "k": k,
            "n_segments": int(X.shape[0]),
            "mean_pairwise_ari": float(np.mean(aris)),
            "std_pairwise_ari": float(np.std(aris)),
            "mean_silhouette": float(s["silhouette"].mean()),
            "mean_calinski_harabasz": float(s["calinski_harabasz"].mean()),
            "mean_davies_bouldin": float(s["davies_bouldin"].mean()),
            "mean_min_cluster_size": float(s["min_cluster_size"].mean()),
            "mean_normalized_cluster_entropy": float(s["normalized_cluster_entropy"].mean()),
            "mean_singleton_clusters": float(s["singleton_clusters"].mean()),
        })
    return score_kstudy(pd.DataFrame(rows))


def _minmax(s, higher_is_better=True):
    s = s.astype(float)
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    base = pd.Series(1.0, index=s.index) if (pd.isna(lo) or pd.isna(hi) or hi == lo) else (s - lo) / (hi - lo)
    return base if higher_is_better else 1.0 - base


def score_kstudy(df):
    df = df.copy()
    score = (0.40 * _minmax(df["mean_pairwise_ari"]).fillna(0.0)
             + 0.20 * _minmax(df["mean_silhouette"]).fillna(0.0)
             + 0.15 * _minmax(df["mean_calinski_harabasz"]).fillna(0.0)
             + 0.10 * _minmax(df["mean_davies_bouldin"], higher_is_better=False).fillna(0.0)
             + 0.10 * _minmax(df["mean_normalized_cluster_entropy"]).fillna(0.0)
             + 0.05 * _minmax(df["mean_min_cluster_size"]).fillna(0.0))
    df["selection_score"] = score - np.where(df["mean_singleton_clusters"].fillna(0) > 0, 0.05, 0.0)
    return df


def select_k(scored):
    return int(scored.loc[scored["selection_score"].idxmax(), "k"])


def segmentation_stats(seg, clip_set):
    summ = seg["summary"][seg["summary"]["clip_id"].isin(clip_set)]
    segs = seg["segments"][seg["segments"]["clip_id"].isin(clip_set)]
    return {
        "single_segment_clip_pct": float((summ["final_segment_count"] == 1).mean() * 100.0),
        "tiny_segments_lt_015_pct": float((segs["duration_sec"] < 0.15).mean() * 100.0),
    }


def sequence_richness(seq_by_clip):
    rows = []
    for seq in seq_by_clip.values():
        n = len(seq)
        transitions = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
        counts = pd.Series(seq).value_counts(normalize=True)
        ent = float(-(counts * np.log(counts + 1e-12)).sum())
        max_ent = float(np.log(len(counts))) if len(counts) > 1 else 0.0
        rows.append({
            "n_unique_units": len(set(seq)),
            "transition_rate": transitions / max(n - 1, 1),
            "token_entropy_normalized": ent / max_ent if max_ent > 0 else 0.0,
        })
    df = pd.DataFrame(rows)
    return {
        "mean_unique_units_per_clip": float(df["n_unique_units"].mean()),
        "mean_transition_rate": float(df["transition_rate"].mean()),
        "mean_token_entropy_normalized": float(df["token_entropy_normalized"].mean()),
    }


def confound_summary(units, seg_clip, clips, fields=AUDIT_FIELDS):
    meta = clips.set_index("clip_id")
    df = pd.DataFrame({"unit": units, "clip_id": seg_clip})
    seg_count = df.groupby("unit").size()
    flagged_pct, high_pct, median_dom = [], [], []
    for field in fields:
        vals = df["clip_id"].map(meta[field])
        vals = vals.where(~(vals.isna() | (vals.astype(str).str.strip() == "")))
        dom = df.assign(v=vals).dropna(subset=["v"]).groupby(["unit", "v"]).size().groupby("unit").max()
        prop = (dom.reindex(seg_count.index) / seg_count)
        flag = (seg_count >= AUDIT_MIN_SEGMENTS) & (prop >= AUDIT_FLAG_THRESHOLD)
        high = (seg_count >= AUDIT_MIN_SEGMENTS) & (prop >= AUDIT_HIGH_THRESHOLD)
        flagged_pct.append(float(flag.mean() * 100.0))
        high_pct.append(float(high.mean() * 100.0))
        median_dom.append(float(prop.median()))
    return {
        "mean_flagged_unit_pct": float(np.mean(flagged_pct)),
        "mean_high_unit_pct": float(np.mean(high_pct)),
        "mean_median_dominant_context_proportion": float(np.mean(median_dom)),
    }


def evaluate_segmentation(name, seg, clip_set, clips, pca_seed):
    keep = seg["segments"]["clip_id"].isin(clip_set).to_numpy()
    segments = seg["segments"][keep].reset_index(drop=True)
    X = _cluster_matrix(seg["vectors"][keep], pca_seed)
    scored = kstudy(X)
    best = scored.loc[scored["selection_score"].idxmax()]
    k = int(best["k"])
    model, _ = _kmeans(X, k, FINAL_SEED)
    units = model.predict(X)
    order = np.lexsort((segments["segment_index"].to_numpy(), segments["clip_id"].to_numpy()))
    seq_by_clip = {}
    for i in order:
        seq_by_clip.setdefault(segments["clip_id"].iat[i], []).append(int(units[i]))
    row = {
        "segmentation": name,
        **segmentation_stats(seg, clip_set),
        "best_k": k,
        "best_mean_pairwise_ari": float(best["mean_pairwise_ari"]),
        "best_selection_score": float(best["selection_score"]),
        **sequence_richness(seq_by_clip),
        **confound_summary(units, segments["clip_id"].to_numpy(), clips),
    }
    return row, scored.assign(segmentation=name)


def scorecard(candidate_rows, baseline_row):
    df = pd.DataFrame(candidate_rows)
    df["single_segment_improvement_pct"] = baseline_row["single_segment_clip_pct"] - df["single_segment_clip_pct"]
    df["clustering_quality_value"] = 0.60 * df["best_mean_pairwise_ari"] + 0.40 * df["best_selection_score"]
    df["sequence_richness_value"] = (0.35 * df["mean_unique_units_per_clip"]
                                     + 0.35 * df["mean_token_entropy_normalized"]
                                     + 0.30 * df["mean_transition_rate"])
    df["confound_control_value"] = (0.60 * df["mean_flagged_unit_pct"]
                                    + 0.40 * df["mean_median_dominant_context_proportion"] * 100.0)
    w = SCORECARD_WEIGHTS
    df["raw_weighted_score"] = (
        w["segmentation_improvement"] * _minmax(df["single_segment_improvement_pct"])
        + w["fragment_control"] * _minmax(df["tiny_segments_lt_015_pct"], higher_is_better=False)
        + w["clustering_quality"] * _minmax(df["clustering_quality_value"])
        + w["sequence_richness"] * _minmax(df["sequence_richness_value"])
        + w["confound_control"] * _minmax(df["confound_control_value"], higher_is_better=False))
    p = SCORECARD_PENALTIES
    df["penalty"] = (
        np.where(df["single_segment_improvement_pct"] <= 0, p["no_single_segment_improvement"], 0.0)
        + np.where((df["tiny_segments_lt_015_pct"] > FRAGMENT_EXPLOSION_ABS_PCT)
                   | ((df["tiny_segments_lt_015_pct"] - baseline_row["tiny_segments_lt_015_pct"])
                      > FRAGMENT_EXPLOSION_DELTA_PCT), p["fragment_explosion"], 0.0)
        + np.where((df["mean_flagged_unit_pct"] - baseline_row["mean_flagged_unit_pct"])
                   > CONFOUND_WORSENED_DELTA_PCT, p["confound_worsened"], 0.0))
    df["final_selection_score"] = df["raw_weighted_score"] - df["penalty"]
    return df.sort_values(["final_selection_score", "best_mean_pairwise_ari", "single_segment_improvement_pct"],
                          ascending=False).reset_index(drop=True)


def choose_layer1_settings(clip_set, clips, segs, log=print):
    baseline_row, baseline_k = evaluate_segmentation("baseline", segs["baseline"], clip_set, clips, KSTUDY_PCA_SEED)
    cand_rows, cand_k = [], [baseline_k]
    for name in CANDIDATES:
        row, ks = evaluate_segmentation(name, segs[name], clip_set, clips, SCORECARD_PCA_SEED)
        cand_rows.append(row)
        cand_k.append(ks)
        log(f"  {name}: best K={row['best_k']} ARI={row['best_mean_pairwise_ari']:.4f} "
            f"flagged={row['mean_flagged_unit_pct']:.2f}%")
    card = scorecard(cand_rows, baseline_row)
    chosen = card.iloc[0]["segmentation"]
    vec = load_layer2_vectors(chosen)
    keep = np.isin(vec["clip_ids"], list(clip_set))
    k_table = kstudy(_cluster_matrix(vec["vectors"][keep], KSTUDY_PCA_SEED))
    k = select_k(k_table)
    log(f"  selected segmentation={chosen}, K={k}")
    return {
        "segmentation": chosen,
        "k": k,
        "scorecard": card,
        "baseline_row": baseline_row,
        "candidate_kstudies": pd.concat(cand_k, ignore_index=True),
        "kstudy": k_table,
    }


def project_segments(vectors, train_mask):
    train = vectors[train_mask]
    scaler = StandardScaler().fit(train)
    pca = PCA(n_components=PCA_DIM, random_state=VOCAB_PCA_SEED).fit(scaler.transform(train))
    return pca.transform(scaler.transform(vectors)), pca.transform(scaler.transform(train))


def fit_vocabulary(vectors, train_mask, k):
    projected, train = project_segments(vectors, train_mask)
    km = MiniBatchKMeans(n_clusters=k, random_state=FINAL_SEED, batch_size=KMEANS_BATCH,
                         max_iter=KMEANS_MAX_ITER, n_init=KMEANS_N_INIT).fit(train)
    return km.predict(projected)


def token_sequences(tokens, seg_clip, seg_index):
    order = np.lexsort((seg_index, seg_clip))
    seqs = {}
    for i in order:
        seqs.setdefault(seg_clip[i], []).append(int(tokens[i]))
    return seqs


def retained_units(tokens, seg_clip, train_clip_set, clips, k, fields):
    ctx = clips.set_index("clip_id")[fields]
    mask = np.array([c in train_clip_set for c in seg_clip])
    df = pd.DataFrame({"unit": tokens[mask], "clip_id": seg_clip[mask]})
    for f in fields:
        df[f] = df["clip_id"].map(ctx[f])
    dropped = set()
    for unit, grp in df.groupby("unit"):
        if len(grp) < AUDIT_MIN_SEGMENTS:
            continue
        for f in fields:
            if grp[f].value_counts(normalize=True).iloc[0] >= AUDIT_HIGH_THRESHOLD:
                dropped.add(int(unit))
                break
    return set(range(k)) - dropped


def audio_matrix(family, variant, seqs, train_clips, all_clips, kept, k):
    kept = kept if variant == "high_bias_filtered" else set(range(k))

    def filt(seq):
        return [t for t in seq if t in kept]

    if family == "unigram_props":
        def vec(seq):
            toks = filt(seq)
            v = np.zeros(k, dtype=np.float32)
            for t in toks:
                v[t] += 1.0
            return v / max(len(toks), 1)
        return np.vstack([vec(seqs[c]) for c in all_clips])
    if family == "transition_probs":
        def vec(seq):
            toks = filt(seq)
            m = np.zeros((k, k), dtype=np.float32)
            for a, b in zip(toks[:-1], toks[1:]):
                m[a, b] += 1.0
            tot = m.sum(axis=1, keepdims=True)
            return np.divide(m, tot, out=np.zeros_like(m), where=tot > 0).ravel()
        return np.vstack([vec(seqs[c]) for c in all_clips])
    if family == "tfidf_1to3":
        text = {c: " ".join("u%02d" % t for t in filt(seqs[c])) for c in all_clips}
        vz = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"u\d\d", lowercase=False)
        vz.fit([text[c] for c in train_clips])
        return vz.transform([text[c] for c in all_clips]).toarray().astype(np.float32)
    raise ValueError(family)


def context_matrix(variant, clips, train_clips, all_clips):
    if variant == "audio_only":
        return None
    cols = STRUCTURAL_CONTEXT_COLS if variant == "plus_structural_context" else FULL_CONTEXT_COLS
    frame = clips.set_index("clip_id")[cols].astype(str)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(frame.loc[train_clips])
    return enc.transform(frame.loc[all_clips]).astype(np.float32)


def bundle_keys():
    return [(f, v, c) for f in AUDIO_FAMILIES for v in VOCAB_VARIANTS for c in CONTEXT_VARIANTS]


def bundle_name(key):
    return f"{key[0]}_{key[1]}__{key[2]}"


def build_bundles(vec, train_mask, k, train_clips, all_clips, clips, audit_fields):
    tokens = fit_vocabulary(vec["vectors"], train_mask, k)
    seqs = token_sequences(tokens, vec["clip_ids"], vec["segment_index"])
    kept = retained_units(tokens, vec["clip_ids"], set(train_clips), clips, k, audit_fields)
    out = {}
    for key in bundle_keys():
        fam, var, ctx = key
        audio = audio_matrix(fam, var, seqs, train_clips, all_clips, kept, k)
        cm = context_matrix(ctx, clips, train_clips, all_clips)
        out[key] = audio if cm is None else np.hstack([audio, cm])
    empty = {c for c in all_clips if not any(t in kept for t in seqs[c])}
    return out, {"n_kept": len(kept), "empty_clips": empty}


def continuous_keys():
    return [(f, "continuous", c) for f in CONTINUOUS_FAMILIES for c in CONTEXT_VARIANTS]


def pooled_matrix(family, projected, seg_clip, train_clips, all_clips):
    rows = {}
    for c in all_clips:
        z = projected[seg_clip == c]
        rows[c] = z.mean(axis=0) if family == "pooled_mean" else np.r_[z.mean(axis=0), z.std(axis=0)]
    X = np.vstack([rows[c] for c in all_clips]).astype(np.float32)
    scaler = StandardScaler().fit(np.vstack([rows[c] for c in train_clips]))
    return scaler.transform(X).astype(np.float32)


def build_continuous_bundles(vec, train_mask, train_clips, all_clips, clips):
    projected, _ = project_segments(vec["vectors"], train_mask)
    out = {}
    for key in continuous_keys():
        fam, _, ctx = key
        audio = pooled_matrix(fam, projected, vec["clip_ids"], train_clips, all_clips)
        cm = context_matrix(ctx, clips, train_clips, all_clips)
        out[key] = audio if cm is None else np.hstack([audio, cm])
    return out


def representation_keys(representation):
    return bundle_keys() if representation == "units" else continuous_keys()


def fold_features(representation, vec, train_mask, k, train_clips, all_clips, clips, audit_fields):
    if representation == "units":
        return build_bundles(vec, train_mask, k, train_clips, all_clips, clips, audit_fields)
    return build_continuous_bundles(vec, train_mask, train_clips, all_clips, clips), None


def make_model(task):
    if task == "multiclass":
        return LogisticRegression(max_iter=4000, solver="saga", class_weight="balanced", random_state=42)
    return LogisticRegression(max_iter=4000, solver="liblinear", class_weight="balanced", random_state=42)


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def run_lofo(clips, fold_settings, nested_inner=True, representation="units", vocab_variants=None, log=print):
    known = clips[clips["farm_num"].isin(FARMS)].copy()
    known["farm_num"] = known["farm_num"].astype(int)
    farm_of = dict(zip(known["clip_id"], known["farm_num"]))
    labels = {s["target_name"]: dict(zip(known["clip_id"], known[s["target_name"]])) for s in TARGET_SPECS}
    known_ctx = known.assign(farm=known["farm_num"])

    pred_records, selection_records, fold_info = [], [], []
    for held in FARMS:
        st = fold_settings[held]
        vec = load_layer2_vectors(st["segmentation"])
        keep = np.isin(vec["clip_ids"], known["clip_id"].to_numpy())
        vec = {k_: v[keep] for k_, v in vec.items()}
        seg_train = np.array([farm_of[c] != held for c in vec["clip_ids"]])
        clip_order = list(dict.fromkeys(vec["clip_ids"][np.lexsort((vec["segment_index"], vec["clip_ids"]))]))
        train_clips = [c for c in clip_order if farm_of[c] != held]
        test_clips = [c for c in clip_order if farm_of[c] == held]
        ordered = train_clips + test_clips
        outer, filt = fold_features(representation, vec, seg_train, st["k"], train_clips, ordered, known_ctx,
                                    LOFO_AUDIT_FIELDS)
        row_of = {c: i for i, c in enumerate(ordered)}
        tr_rows = [row_of[c] for c in train_clips]
        te_rows = [row_of[c] for c in test_clips]

        inner_farms = [f for f in FARMS if f != held]
        inner_sets = {}
        for val_farm in inner_farms:
            fit_farm = [f for f in inner_farms if f != val_farm][0]
            fit_clips = [c for c in train_clips if farm_of[c] == fit_farm]
            val_clips = [c for c in train_clips if farm_of[c] == val_farm]
            if nested_inner:
                inner_order = fit_clips + val_clips
                inner_mask = np.array([farm_of[c] == fit_farm for c in vec["clip_ids"]])
                mats, _ = fold_features(representation, vec, inner_mask, st["k"], fit_clips, inner_order,
                                        known_ctx, LOFO_AUDIT_FIELDS)
                r = {c: i for i, c in enumerate(inner_order)}
                inner_sets[val_farm] = (mats, [r[c] for c in fit_clips], [r[c] for c in val_clips],
                                        fit_clips, val_clips)
            else:
                inner_sets[val_farm] = (outer, [row_of[c] for c in fit_clips], [row_of[c] for c in val_clips],
                                        fit_clips, val_clips)
        fold_info.append({"heldout_farm": held, "representation": representation,
                          "segmentation": st["segmentation"], "k": st["k"],
                          "n_units_retained_filtered": filt["n_kept"] if filt else None,
                          "n_train_clips_without_retained_unit":
                              len(filt["empty_clips"] & set(train_clips)) if filt else None,
                          "n_test_clips_without_retained_unit":
                              len(filt["empty_clips"] & set(test_clips)) if filt else None})
        keys = [k_ for k_ in representation_keys(representation)
                if vocab_variants is None or k_[1] in vocab_variants]

        for spec in TARGET_SPECS:
            target, task = spec["target_name"], spec["task_type"]
            y_tr_raw = np.array([labels[target][c] for c in train_clips])
            y_te_raw = np.array([labels[target][c] for c in test_clips])
            if task == "multiclass":
                enc = LabelEncoder().fit(y_tr_raw)
                y_tr, y_te, classes = enc.transform(y_tr_raw), enc.transform(y_te_raw), enc.classes_.tolist()
                yfun = lambda cs: enc.transform([labels[target][c] for c in cs])
            else:
                y_tr, y_te, classes = y_tr_raw.astype(int), y_te_raw.astype(int), [0, 1]
                yfun = lambda cs: np.array([labels[target][c] for c in cs]).astype(int)

            best_key, best_score = None, -1.0
            for key in keys:
                scores = []
                for val_farm in sorted(inner_sets):
                    mats, fit_rows, val_rows, fit_clips, val_clips = inner_sets[val_farm]
                    model = make_model(task).fit(mats[key][fit_rows], yfun(fit_clips))
                    scores.append(macro_f1(yfun(val_clips), model.predict(mats[key][val_rows])))
                mean_score = float(np.mean(scores))
                selection_records.append({"heldout_farm": str(held), "target_name": target,
                                          "bundle_name": bundle_name(key), "inner_macro_f1": mean_score})
                if mean_score > best_score:
                    best_score, best_key = mean_score, key

            X_tr, X_te = outer[best_key][tr_rows], outer[best_key][te_rows]
            model = make_model(task).fit(X_tr, y_tr)
            y_pred, proba = model.predict(X_te), model.predict_proba(X_te)
            y_dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te)
            for i, cid in enumerate(test_clips):
                rec = {"_record_key": cid, "target_name": target, "task_type": task,
                       "model_name": "logreg_balanced", "heldout_farm": str(held),
                       "segmentation": st["segmentation"], "k": st["k"],
                       "bundle_name": bundle_name(best_key),
                       "audio_feature_name": f"{best_key[0]}_{best_key[1]}", "variant": best_key[2],
                       "y_true": int(y_te[i]), "y_pred": int(y_pred[i]), "y_dummy": int(y_dummy[i])}
                if task == "multiclass":
                    rec["y_true_label"] = classes[y_te[i]]
                    rec["y_pred_label"] = classes[y_pred[i]]
                    for j, name in enumerate(classes):
                        rec[f"proba__{name}"] = float(proba[i, j])
                else:
                    rec["proba__1"] = float(proba[i, 1])
                pred_records.append(rec)
            log(f"farm={held} target={target} selected={bundle_name(best_key)} inner_macro_f1={best_score:.3f}")

    return pd.DataFrame(pred_records), pd.DataFrame(selection_records), pd.DataFrame(fold_info)


def inner_predictions(clips, fold_settings, pred_df, representation="units", log=print):
    known = clips[clips["farm_num"].isin(FARMS)].copy()
    known["farm_num"] = known["farm_num"].astype(int)
    farm_of = dict(zip(known["clip_id"], known["farm_num"]))
    labels = {s["target_name"]: dict(zip(known["clip_id"], known[s["target_name"]])) for s in TARGET_SPECS}
    known_ctx = known.assign(farm=known["farm_num"])
    key_of = {bundle_name(k): k for k in representation_keys(representation)}

    records, checks = [], []
    for held in FARMS:
        st = fold_settings[held]
        vec = load_layer2_vectors(st["segmentation"])
        keep = np.isin(vec["clip_ids"], known["clip_id"].to_numpy())
        vec = {k_: v[keep] for k_, v in vec.items()}
        clip_order = list(dict.fromkeys(vec["clip_ids"][np.lexsort((vec["segment_index"], vec["clip_ids"]))]))
        train_clips = [c for c in clip_order if farm_of[c] != held]
        fold_pred = pred_df[pred_df["heldout_farm"].astype(str) == str(held)]
        inner_farms = [f for f in FARMS if f != held]
        for val_farm in inner_farms:
            fit_farm = [f for f in inner_farms if f != val_farm][0]
            fit_clips = [c for c in train_clips if farm_of[c] == fit_farm]
            val_clips = [c for c in train_clips if farm_of[c] == val_farm]
            inner_mask = np.array([farm_of[c] == fit_farm for c in vec["clip_ids"]])
            mats, _ = fold_features(representation, vec, inner_mask, st["k"], fit_clips, fit_clips + val_clips,
                                    known_ctx, LOFO_AUDIT_FIELDS)
            nf = len(fit_clips)
            for spec in TARGET_SPECS:
                target, task = spec["target_name"], spec["task_type"]
                sub = fold_pred[fold_pred["target_name"] == target]
                bname = sub["bundle_name"].iloc[0]
                X = mats[key_of[bname]]
                y_fit = [labels[target][c] for c in fit_clips]
                y_val = [labels[target][c] for c in val_clips]
                if task == "multiclass":
                    classes = sorted(c[len("proba__"):] for c in sub.columns
                                     if c.startswith("proba__") and c != "proba__1" and sub[c].notna().any())
                    model = make_model(task).fit(X[:nf], np.asarray(y_fit))
                    P = model.predict_proba(X[nf:])
                    full = np.zeros((len(val_clips), len(classes)))
                    for j, cname in enumerate(model.classes_):
                        full[:, classes.index(cname)] = P[:, j]
                    y_pred = np.asarray(model.predict(X[nf:]))
                    y_true = np.asarray(y_val)
                else:
                    model = make_model(task).fit(X[:nf], np.asarray(y_fit).astype(int))
                    p1 = model.predict_proba(X[nf:])[:, 1]
                    y_pred = model.predict(X[nf:]).astype(int)
                    y_true = np.asarray(y_val).astype(int)
                checks.append({"heldout_farm": str(held), "target_name": target, "bundle_name": bname,
                               "inner_val_farm": val_farm, "macro_f1": macro_f1(y_true, y_pred)})
                for i, cid in enumerate(val_clips):
                    rec = {"_record_key": cid, "target_name": target, "task_type": task,
                           "heldout_farm": str(held), "inner_val_farm": str(val_farm),
                           "bundle_name": bname}
                    if task == "multiclass":
                        rec["y_true_label"] = y_val[i]
                        rec["y_true"] = classes.index(y_val[i])
                        for j, cname in enumerate(classes):
                            rec[f"proba__{cname}"] = float(full[i, j])
                    else:
                        rec["y_true"] = int(y_true[i])
                        rec["proba__1"] = float(p1[i])
                    records.append(rec)
        log(f"inner predictions for held-out farm {held} done")
    check = (pd.DataFrame(checks).groupby(["heldout_farm", "target_name", "bundle_name"], as_index=False)
             ["macro_f1"].mean().rename(columns={"macro_f1": "inner_macro_f1_recomputed"}))
    return pd.DataFrame(records), check


def summarise(pred_df):
    rows, perfarm = [], []
    for spec in TARGET_SPECS:
        t = spec["target_name"]
        sub = pred_df[pred_df["target_name"] == t]
        for name, col in (("logreg_balanced", "y_pred"), ("dummy_most_frequent", "y_dummy")):
            yt, yp = sub["y_true"], sub[col]
            rows.append({"target_name": t, "task_type": spec["task_type"], "model_name": name,
                         "n_samples": int(len(sub)), "macro_f1": macro_f1(yt, yp),
                         "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
                         "accuracy": float(accuracy_score(yt, yp)),
                         "macro_precision": float(precision_score(yt, yp, average="macro", zero_division=0)),
                         "macro_recall": float(recall_score(yt, yp, average="macro", zero_division=0))})
        for farm, pf in sub.groupby("heldout_farm"):
            perfarm.append({"target_name": t, "heldout_farm": farm, "n_samples": int(len(pf)),
                            "macro_f1": macro_f1(pf["y_true"], pf["y_pred"]),
                            "balanced_accuracy": float(balanced_accuracy_score(pf["y_true"], pf["y_pred"]))})
    return pd.DataFrame(rows), pd.DataFrame(perfarm)
