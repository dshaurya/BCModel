from __future__ import annotations

from pathlib import Path
from collections import Counter
import csv
import json

import numpy as np
from PIL import Image
from ultralytics import YOLO

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        accuracy_score,
    )
except Exception as e:
    raise SystemExit("Please install scikit-learn: pip install scikit-learn") from e

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit("Please install matplotlib: pip install matplotlib") from e


BASE = Path(".")
DATASET_DIR = BASE / "Blood_Cancer_TestingData"
DETECT_MODEL = BASE / "detectorbc.pt"
CLASSIFY_MODEL = BASE / "classifymodel.pt"
OUTPUT_DIR = BASE / "evaluation_output"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def safe_crop(img: Image.Image, box, pad=6):
    w, h = img.size
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img.crop((x1, y1, x2, y2))


def find_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def infer_true_label(path: Path, dataset_root: Path):
    rel = path.relative_to(dataset_root)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return "unknown"


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def main():
    if not DATASET_DIR.exists():
        raise SystemExit(f"Dataset folder not found: {DATASET_DIR.resolve()}")
    if not DETECT_MODEL.exists():
        raise SystemExit(f"Detector model not found: {DETECT_MODEL.resolve()}")
    if not CLASSIFY_MODEL.exists():
        raise SystemExit(f"Classifier model not found: {CLASSIFY_MODEL.resolve()}")

    detector = YOLO(str(DETECT_MODEL))
    classifier = YOLO(str(CLASSIFY_MODEL))

    image_paths = find_images(DATASET_DIR)
    if not image_paths:
        raise SystemExit("No images found in dataset folder.")

    rows = []
    y_true = []
    y_pred = []
    missed_detection_count = 0
    multi_detection_count = 0

    for image_path in image_paths:
        true_label = infer_true_label(image_path, DATASET_DIR)
        img = Image.open(image_path).convert("RGB")

        det_res = detector.predict(source=img, verbose=False)[0]

        pred_label = "no_detection"
        det_count = 0
        best_det_conf = None
        cls_conf = None

        if det_res.boxes is not None and len(det_res.boxes) > 0:
            det_count = len(det_res.boxes)

            if det_count > 1:
                multi_detection_count += 1

            confs = det_res.boxes.conf.cpu().numpy()
            boxes = det_res.boxes.xyxy.cpu().numpy()
            best_idx = int(np.argmax(confs))

            best_det_conf = float(confs[best_idx])
            crop = safe_crop(img, boxes[best_idx])

            cls_res = classifier.predict(source=crop, verbose=False)[0]
            top1 = int(cls_res.probs.top1)
            pred_label = cls_res.names.get(top1, str(top1))
            cls_conf = float(cls_res.probs.top1conf.cpu())
        else:
            missed_detection_count += 1

        y_true.append(normalize_name(true_label))
        y_pred.append(normalize_name(pred_label))

        rows.append(
            {
                "image_path": str(image_path),
                "true_label": true_label,
                "pred_label": pred_label,
                "det_count": det_count,
                "best_det_conf": best_det_conf,
                "cls_conf": cls_conf,
            }
        )

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )

    with open(OUTPUT_DIR / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "num_images": len(image_paths),
        "num_labels": len(labels),
        "labels": labels,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "missed_detection_count": missed_detection_count,
        "multi_detection_count": multi_detection_count,
        "true_label_distribution": dict(Counter(y_true)),
        "pred_label_distribution": dict(Counter(y_pred)),
        "classification_report": report,
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig_w = max(8, len(labels) * 0.9)
    fig_h = max(6, len(labels) * 0.75)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Two-Stage Pipeline Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Done. Images: {len(image_paths)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Missed detections: {missed_detection_count}")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
