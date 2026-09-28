
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent

K = 48
PCA_DIM = 128
CLUSTER_SEED = 31
CLUSTER_BATCH = 2048
CLUSTER_MAX_ITER = 200

BIAS_FIELDS = ["farm", "barn_zone", "microphone", "mic_placement_context"]
BIAS_DOMINANT_THRESHOLD = 0.90
BIAS_MIN_SEGMENTS = 8

STRUCTURAL_CONTEXT_COLS = ["barn_zone", "microphone", "mic_placement_context"]
FULL_CONTEXT_COLS = ["farm", "barn_zone", "microphone", "mic_placement_context", "day", "date"]

AUDIO_FAMILIES = ["unigram_props", "tfidf_1to3", "transition_probs"]
VOCAB_VARIANTS = ["full", "high_bias_filtered"]
CONTEXT_VARIANTS = ["audio_only", "plus_structural_context", "plus_full_context"]


def load_corpus():
    """Return per-segment embeddings and a clip-level metadata/label frame."""
    seg = np.load(ROOT / "output/03_segment_vectors_refined/segment_vectors.npz",
                  allow_pickle=True)
    seg_vectors = seg["vectors"].astype(np.float32)
    seg_clip = seg["clip_ids"].astype(str)
    seg_index = seg["segment_index"].astype(int)

    man = pd.read_csv(ROOT / "output/00_manifest/manifest.csv")
    man["farm"] = pd.to_numeric(man["farm"], errors="coerce")
    pmap = pd.read_csv(ROOT / "output/11_layer2_proxy_labels/proxy_label_mapping_table.csv")
    main2proxy = dict(zip(pmap["main_label_raw"], pmap["proxy_functional_v1"]))

    cols = ["clip_id", "farm", "barn_zone", "microphone", "mic_placement_context",
            "day", "date", "main_category_final"]
    clips = man[cols].copy()
    clips["proxy_functional_v1"] = clips["main_category_final"].map(main2proxy)
    clips["is_nonvocal"] = (clips["proxy_functional_v1"] == "physiological_nonvocal").astype(int)
    clips["is_resource_related"] = (clips["proxy_functional_v1"] == "ingestion_resource").astype(int)
    clips["is_distress_related"] = (clips["proxy_functional_v1"] == "stress_discomfort").astype(int)
    clips = clips[clips["farm"].isin([1, 2, 3])].reset_index(drop=True)
    clips["farm"] = clips["farm"].astype(int)

    keep = np.isin(seg_clip, clips["clip_id"].to_numpy())
    return {
        "seg_vectors": seg_vectors[keep],
        "seg_clip": seg_clip[keep],
        "seg_index": seg_index[keep],
        "clips": clips,
    }


def fit_fold_vocabulary(seg_vectors, train_mask):
    """Fit standardization, PCA, and K-means on training segments; return a
    function that maps any segment matrix to fold-local token indices."""
    train_vectors = seg_vectors[train_mask]
    scaler = StandardScaler().fit(train_vectors)
    pca = PCA(n_components=PCA_DIM, random_state=CLUSTER_SEED).fit(scaler.transform(train_vectors))
    kmeans = MiniBatchKMeans(
        n_clusters=K, random_state=CLUSTER_SEED, batch_size=CLUSTER_BATCH,
        max_iter=CLUSTER_MAX_ITER, n_init=10,
    ).fit(pca.transform(scaler.transform(train_vectors)))

    def assign(vectors):
        return kmeans.predict(pca.transform(scaler.transform(vectors)))

    return assign


def build_token_sequences(token_ids, seg_clip, seg_index):
    """Return {clip_id: [unit_index, ...]} ordered by segment index."""
    order = np.lexsort((seg_index, seg_clip))
    sequences: dict[str, list[int]] = {}
    for i in order:
        sequences.setdefault(seg_clip[i], []).append(int(token_ids[i]))
    return sequences


def audit_filtered_units(token_ids, seg_clip, train_clip_set, clips):
    """Return the set of unit indices retained in the high-bias-filtered
    vocabulary, using a context-bias audit over training clips only."""
    ctx = clips.set_index("clip_id")[BIAS_FIELDS]
    train_mask = np.array([c in train_clip_set for c in seg_clip])
    df = pd.DataFrame({"unit": token_ids[train_mask], "clip_id": seg_clip[train_mask]})
    for field in BIAS_FIELDS:
        df[field] = df["clip_id"].map(ctx[field])

    dropped = set()
    for unit, grp in df.groupby("unit"):
        if len(grp) < BIAS_MIN_SEGMENTS:
            continue
        for field in BIAS_FIELDS:
            dominant = grp[field].value_counts(normalize=True).iloc[0]
            if dominant >= BIAS_DOMINANT_THRESHOLD:
                dropped.add(int(unit))
                break
    return set(range(K)) - dropped


def _sequence_text(seq, retained):
    return " ".join("u%02d" % t for t in seq if t in retained)


def _filtered_seq(seq, retained):
    return [t for t in seq if t in retained]


def build_audio_matrix(family, variant, sequences, train_clips, all_clips, retained_units):
    """Return an aligned dense audio-feature matrix for the given family and
    vocabulary variant, ordered as all_clips. Vectorizers are fit on train_clips."""
    retained = retained_units if variant == "high_bias_filtered" else set(range(K))

    if family == "unigram_props":
        def vec(seq):
            toks = _filtered_seq(seq, retained)
            v = np.zeros(K, dtype=np.float32)
            for t in toks:
                v[t] += 1.0
            return v / max(len(toks), 1)
        return np.vstack([vec(sequences[c]) for c in all_clips])

    if family == "transition_probs":
        def vec(seq):
            toks = _filtered_seq(seq, retained)
            m = np.zeros((K, K), dtype=np.float32)
            for a, b in zip(toks[:-1], toks[1:]):
                m[a, b] += 1.0
            row_totals = m.sum(axis=1, keepdims=True)
            return np.divide(m, row_totals, out=np.zeros_like(m), where=row_totals > 0).ravel()
        return np.vstack([vec(sequences[c]) for c in all_clips])

    if family == "tfidf_1to3":
        text = {c: _sequence_text(sequences[c], retained) for c in all_clips}
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"u\d\d", lowercase=False)
        vectorizer.fit([text[c] for c in train_clips])
        return vectorizer.transform([text[c] for c in all_clips]).toarray().astype(np.float32)

    raise ValueError(family)


def build_context_matrix(variant, clips, train_clips, all_clips):
    """Return an aligned one-hot context matrix for the given context variant,
    or None for audio_only. Encoder is fit on train_clips."""
    if variant == "audio_only":
        return None
    cols = STRUCTURAL_CONTEXT_COLS if variant == "plus_structural_context" else FULL_CONTEXT_COLS
    frame = clips.set_index("clip_id")[cols].astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(frame.loc[train_clips])
    return encoder.transform(frame.loc[all_clips]).astype(np.float32)
