# Fabric Defect Classification

## Project overview
This project compares two Roboflow-hosted DINOv3 image-classification models (an original version and an augmented version) with a locally trained ResNet18 model for classifying fabric images into:

- `defect_free`
- `minor_defect`

The held-out evaluation contains 509 images: 138 `defect_free` and 371 `minor_defect`.

## Main results

| Model | Accuracy | Macro F1 | Correct / 509 | Inference time per image |
|---|---:|---:|---:|---:|
| Original DINOv3 (v1) | 41.65% | 40.71% | 212 | 1,343.43 ms* |
| Augmented DINOv3 (v2) | 60.51% | 60.24% | 308 | 1,235.48 ms* |
| ResNet18 (CPU) | **82.32%** | **79.49%** | **419** | **111.16 ms** |

`*` DINOv3 values are end-to-end API request times and include network communication. ResNet18 was timed locally on CPU. These are not strictly like-for-like latency measurements.

See [`results/model_comparison.csv`](results/model_comparison.csv) and [`report/fabric_defect_classification_report.pdf`](report/fabric_defect_classification_report.pdf).

## Dataset and method
- Roboflow dataset: 5,000 images, 70/20/10 split, 224 × 224 resizing.
- Roboflow v2 augmentation: horizontal flip and rotation from -10° to +10°, augmentation multiplier 3×.
- Local ResNet18 folders reported 3,553 training, 1,015 validation, and 509 test images (5,077 total).
- The 509-image test set was used for the reported full-set comparison.

**Dataset consistency note:** The local folder counts total 5,077, while the Roboflow dataset version contains 5,000 images. Therefore, the local ResNet18 training/validation split is not confirmed to be exactly the same as the Roboflow training/validation split. This should be disclosed when presenting the comparison.

## ResNet18 evaluation
Confusion matrix (rows = actual class; columns = predicted class; class order: `defect_free`, `minor_defect`):

| Actual / Predicted | defect_free | minor_defect |
|---|---:|---:|
| defect_free | 115 | 23 |
| minor_defect | 67 | 304 |

ResNet18 inference on 509 images took 56.581 seconds, or 111.16 ms per image, on CPU.

## Training time
The ResNet18 training duration was not recorded. No training-time value is claimed in this report.

## Failure analysis
- ResNet18 correctly identified 115 of 138 defect-free images and 304 of 371 defective images.
- It missed 67 defective images (predicted them as defect-free).
- It falsely flagged 23 defect-free images as defective.
- Possible contributors include subtle defects, low contrast, lighting differences, fabric texture variation, and class imbalance. These are plausible explanations, not confirmed causes from a documented visual review.

## Reproducibility
The project used Python 3.11.9 and PyTorch 2.15.0+cpu in the recorded local environment. The ResNet18 model weights are not included in this starter package; add `resnet18_fabric.pth` only if you are permitted to publish the trained weights. Do not commit API keys or credentials.

## Repository structure
```text
fabric_defect_project/
├── README.md
├── requirements.txt
├── .gitignore
├── results/
│   └── model_comparison.csv
└── report/
    └── fabric_defect_classification_report.pdf
```
## Roboflow Project

**Public Roboflow Models:**  
[View Fabric Detection Models](https://app.roboflow.com/prashansas-workspace/fabric-detection-yhlkp/models)

The project contains two trained DINOv3 classification models:
- Original model (without augmentation)
- Augmented model (horizontal flip and rotation)
