# Reliability-Aware Cross-Farm Evaluation of Dairy Cow Vocalization Models

Code for the analysis in our study of dairy cow vocalizations recorded on
commercial farms. The pipeline has three parts: discovering acoustic units
without labels, testing whether those units carry proxy-state signal across
farms, and checking how far the resulting predictions can be trusted.

## Pipeline

**Layer 1 (acoustic foundation).** Frame embeddings from a frozen wav2vec 2.0
encoder, change-point segmentation in the embedding space, PCA, then K-means to
build a discrete acoustic-unit vocabulary. Each clip becomes an ordered sequence
of unit tokens.

**Layer 2 (proxy-state inference).** Token-sequence features (unit composition,
TF-IDF 1-3 grams, transition probabilities) for four proxy targets, evaluated
three ways: audio-only, audio plus context, and leave-one-farm-out (LOFO).
Within-cohort cross-validation is grouped by recording session. LOFO is fully
nested: in each fold the segmentation candidate and the vocabulary size K are
chosen on the training farms, the vocabulary, context-bias audit, TF-IDF
weighting and context encoding are fitted on the training farms, and the feature
bundle is chosen by an inner leave-one-farm-out loop that rebuilds the features
from a single training farm.

**Layer 3 (reliability).** For each held-out farm, the calibrator and the score
used for abstention (raw or calibrated) are chosen from out-of-farm predictions
inside the training farms. Selective prediction at fixed coverage targets, a
confidence threshold learned on the training farms, and a model-behaviour
dictionary that assigns each unit a within-corpus reliability tier.

**Representation comparison.** Pooled continuous segment embeddings (the same
pipeline without the discretisation into units) and a log-mel CNN are evaluated
with the same folds, nested selection and Layer 3 procedure. Differences are
tested with a farm-stratified session-cluster bootstrap and Holm adjustment.

## Layout

```
notebooks/        pipeline notebooks
  crossfarm_nested.py    fully nested leave-one-farm-out pipeline
  layer2_foldwise.py     fold-local vocabulary and token features (notebook 13)
  layer3_reliability.py  calibration, selective prediction and nested selection
  cnn_baseline.ipynb     log-mel CNN baseline
configs/config.yaml   paths and parameters
docker/               runtime image
```

## Running it

Everything runs inside the Docker image (PyTorch, torchaudio, scikit-learn):

```bash
docker build -t cowvocal -f docker/Dockerfile .
docker run --rm --gpus all -v "$(pwd)":/workspace -w /workspace cowvocal \
  jupyter nbconvert --to notebook --execute --inplace notebooks/00_build_manifest.ipynb
```

Run the notebooks in this order:

1. Layer 1: `00`, `01`, `02`, `02b`, `02c`, `03`, `04`, `05`, `06`
2. Layer 2: `08`, `09`, `10`, `11`, `12`, `13b_layer2_context_ablation`,
   `13n`, `13b_layer2_actual_farm_holdout_ablation`, `14`
3. CNN baseline: `cnn_baseline`
4. Layer 3 and comparison: `16n`, `13c`, `17n`, `18n`

Notebooks `13` and `16` are the earlier leave-one-farm-out and reliability
analyses without the nested segmentation, vocabulary-size and calibrator choices;
the nested versions are `13n` and `16n`. Run `13` after `12` and `16` after `13`.

Random seeds are fixed in the notebook and module code (clustering, models, CNN),
and the CNN uses deterministic GPU settings, so runs are reproducible.

## Results and outputs

Notebook outputs are not committed and the `output/` directory is generated
locally, so the repository stays code-only. Reported results and figures are in
the paper and its supplementary material; reproducing them requires the Zenodo
dataset (see below).

## Data

The audio and metadata are on Zenodo under restricted access
(DOI: 10.5281/zenodo.17764250) because of farm confidentiality; access is through
a data access agreement. This repository is code only. Put the audio in
`data/extracted/` and the metadata in `data/metadata/` to run from scratch;
outputs are written to `output/`.
