# Model card: Wildfire DenseNet121

## Intended use

Binary screening of images into `nowildfire` (0) and `wildfire` (1), with Grad-CAM as a debugging
aid. The model is not independently suitable for dispatch, evacuation, or other safety-critical
decisions.

## Training data

The repository contains supplied train/validation/test ZIPs. Data preparation produces a manifest
with the path, class, byte size, and ZIP CRC for every image. The original geographic-looking file
names are not parsed into model features. Confirm independently that nearby coordinates, repeat
captures, and acquisition dates do not cross splits.

## Model and training

ImageNet DenseNet121 transfer learning at 224x224 RGB, first with a frozen backbone and then partial
fine-tuning. See `params.yaml` for the complete reproducible configuration.

## Evaluation policy

The operating threshold is selected on validation F2 and frozen before test evaluation. Primary
operational metrics should emphasize recall and PR-AUC because missed fires may be more costly than
false positives. Accuracy alone is not sufficient.

## Known risks

- Geographic, seasonal, sensor, and weather distribution shift.
- Shortcut learning from borders, compression, clouds, labels, or collection artifacts.
- Poorly calibrated confidence under shift.
- Grad-CAM shows local sensitivity and can be plausible while the decision remains unreliable.

Record the final dataset version, MLflow run ID, test metrics, approved threshold, owners, and review
date here before deployment.
