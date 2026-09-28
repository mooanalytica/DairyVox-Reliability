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
three ways: audio-only, audio plus context, and leave-one-farm-out (LOFO). For
LOFO the vocabulary is rebuilt inside each fold on the training farms only, and
the feature bundle is chosen with an inner leave-one-farm-out loop. Within-cohort
cross-validation is grouped by recording session.

**Layer 3 (reliability).** Group-aware probability calibration, selective
prediction at a coverage target, and a model-behaviour dictionary that assigns
each unit a within-corpus reliability tier.

## Layout

```
notebooks/        pipeline notebooks, numbered in run order (00-16; 13b after 13)
  layer2_foldwise.py     fold-local vocabulary + token features
  layer3_reliability.py  calibration and selective prediction
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

Run the notebooks in order. Random seeds are fixed in the notebook and module code
(clustering, models, CNN), and the CNN uses deterministic GPU settings, so runs are
reproducible.

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
