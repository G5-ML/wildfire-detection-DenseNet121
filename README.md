# Wildfire image classification with TensorFlow DenseNet

This project is an end-to-end, reproducible binary image-classification system for the supplied
`wildfire` / `nowildfire` satellite-image archives. It trains an ImageNet-initialized DenseNet121,
fine-tunes it conservatively, evaluates it with exact scikit-learn metrics, produces Grad-CAM
explanations, tracks experiments with MLflow, versions pipeline stages with DVC, and exposes the
saved model through FastAPI.

The supplied data is kept in its original split to prevent leakage:

| Split | nowildfire | wildfire | Total |
|---|---:|---:|---:|
| train | 14,500 | 15,750 | 30,250 |
| valid | 2,820 | 3,480 | 6,300 |
| test | 2,820 | 3,480 | 6,300 |

Label `0` is always `nowildfire`; label `1` is always `wildfire`.

## Architecture and safeguards

The network uses DenseNet121 without its ImageNet classifier, global average pooling, batch
normalization, a GELU dense head, dropout, and a sigmoid output. Training has two phases:

1. Learn only the new classification head.
2. Unfreeze the final DenseNet layers at a 100x smaller learning rate while keeping all backbone
   BatchNorm layers frozen.

Regularization and reliability measures include realistic image augmentation, L2 regularization,
dropout, label smoothing, AdamW weight decay, gradient clipping, balanced class weights, early
stopping, `ReduceLROnPlateau`, deterministic seeds, `BackupAndRestore`, best-weight saving, and
periodic weight checkpoints. Mixed precision is enabled only when a GPU is present.

## Setup

Use Python 3.10-3.12. Python 3.11 is the tested target for TensorFlow 2.16-2.18.

PowerShell:

```powershell
cd wildfire_project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,serve]"
```

Linux/macOS:

```bash
cd wildfire_project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,serve]'
```

DenseNet ImageNet weights are downloaded by Keras on the first training run. For an offline smoke
test, set `model.imagenet_weights: null` in `params.yaml`.

## Run the pipeline

All paths and hyperparameters live in `params.yaml`.

```bash
wildfire-ml --config params.yaml prepare
wildfire-ml --config params.yaml train
wildfire-ml --config params.yaml evaluate
```

Or run the reproducible DVC graph:

```bash
dvc repro
dvc metrics show
dvc plots show
```

Preparation safely rejects ZIP path traversal and symbolic links, validates both classes, checks
image readability, writes `data/processed/manifest.csv`, and is idempotent. If an archive changes
after partial extraction, rerun `prepare --force`.

Training writes:

- `artifacts/models/best.weights.h5` — best validation PR-AUC weights.
- `artifacts/models/checkpoints/epoch_XXX.weights.h5` — periodic recovery snapshots.
- `artifacts/models/final.keras` and `final.weights.h5` — deployable model and weights.
- `artifacts/logs/training.csv`, TensorBoard events, and JSON history.
- `mlruns/` — local MLflow parameters, metrics, training artifacts, and serialized model.

Open the tracking interfaces with:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
tensorboard --logdir artifacts/logs/tensorboard
```

## Evaluation

The validation split selects the probability threshold that maximizes F-beta (beta defaults to 2,
favoring wildfire recall). That frozen threshold is then applied once to the untouched test split.
Set `evaluation.threshold` to a number in `params.yaml` to use a fixed operational threshold.

The implementation computes:

- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, and `fbeta_score`;
- `confusion_matrix`;
- `roc_auc_score`, ROC `auc`, and full `roc_curve` coordinates;
- `average_precision_score`, trapezoidal PR-AUC, and full `precision_recall_curve` coordinates;
- `brier_score_loss` for probability calibration quality.

Results are in `artifacts/reports/metrics.json`; `dvc_metrics.json` contains the scalar DVC view.
Validation and test subdirectories also contain raw prediction CSVs, curve JSON, confusion matrices,
ROC plots, and precision-recall plots.

## Classification and Grad-CAM

Classify one image and save an explanation overlay:

```bash
wildfire-ml --config params.yaml predict \
  data/processed/test/wildfire/example.jpg \
  --gradcam-output artifacts/reports/example_gradcam.png
```

The Grad-CAM color map is derived from the final 7x7 DenseNet feature tensor. Warm colors indicate
regions that most increased the score for the explained class. It is a local sensitivity map, not a
segmentation mask or proof of causal reasoning. Check maps for reliance on smoke/fire rather than
watermarks, borders, acquisition artifacts, or geography.

The complete interactive walkthrough is
[`notebooks/wildfire_classification_gradcam.ipynb`](notebooks/wildfire_classification_gradcam.ipynb).
It prepares data, displays samples, optionally trains, loads evaluation results, classifies test
images, and visualizes Grad-CAM side-by-side.

## Serve the model

After training:

```bash
uvicorn wildfire_ml.api:app --host 0.0.0.0 --port 8000
curl -X POST -F "file=@data/processed/test/wildfire/example.jpg" http://localhost:8000/predict
```

`POST /explain` also returns a base64 PNG Grad-CAM overlay. Uploads are capped at 10 MiB. For a
containerized API and MLflow server:

```bash
docker compose up --build
```

The API is at `http://localhost:8000/docs`; MLflow is at `http://localhost:5000`.

When `WILDFIRE_PREDICTION_LOG` is set, the API appends JSONL telemetry containing timestamp, image
SHA-256, score, decision, threshold, and model version; image bytes and names are not retained. The
Compose setup writes it to `monitoring/predictions.jsonl`. Compare it with the held-out reference:

```bash
wildfire-ml --config params.yaml monitor monitoring/predictions.jsonl
```

The report at `artifacts/monitoring/drift_report.json` includes population stability index (PSI),
score mean, alert-rate shift, and low-confidence rate. Fewer than 100 production events are marked
`insufficient_data`; PSI at or above 0.2 is a review warning, not an automatic retraining trigger.

## MLOps layout

```text
datasets/*.zip ──prepare──> data/processed + manifest
                              │
                              └──train──> checkpoints + final.keras + MLflow
                                             │
                     valid ──threshold────────┤
                     test  ──evaluate─────────┴──> metrics + plots
                                                        │
                                              CLI / notebook / API
```

- DVC records data, parameters, artifacts, metrics, and the stage dependency graph.
- MLflow records comparable experiment runs and model artifacts.
- GitHub Actions lints, tests, and builds the serving container.
- Docker packages the same saved `.keras` model used by the notebook and CLI.
- Optional privacy-conscious telemetry and PSI reporting screen for post-deployment score drift.
- Unit tests cover archive safety, label invariants, exact metrics, model shape, fine-tuning policy,
  and Grad-CAM output.

Run local quality checks with:

```bash
ruff check src tests
pytest
```

## Production notes

Before operational use, calibrate and approve the alert threshold against the cost of missed fires
and false alarms; assess performance by geography, season, sensor, cloud/smoke conditions, and time;
monitor input and confidence drift; and require human review. Grad-CAM and high validation scores do
not make this a stand-alone emergency decision system.
