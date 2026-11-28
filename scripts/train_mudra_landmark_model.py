"""Train a bounded mudra classifier from the MIT-licensed Hugging Face dataset.

MediaPipe extracts 21 hand landmarks from each training image. The same
wrist-relative, palm-normalised representation is used by the runtime
``left_hand_rel`` / ``right_hand_rel`` frames, so the saved model can annotate
the adavu catalogue without re-running image inference.

This is a hand-shape prior. It is not trained on paired music and expert
Bharatanatyam choreography, so it cannot guarantee a correct composition for an
arbitrary song.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402

from app.motion.mudra import landmark_features, mirror_features  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "hf_datasets" / "Samarth0710__bharatanatyam-mudra-dataset" / "data"
DEFAULT_MODEL = ROOT / "models" / "mudra_landmark_classifier.joblib"
DEFAULT_META = ROOT / "models" / "mudra_landmark_classifier.json"
DEFAULT_HAND_MODEL = ROOT / "data" / "models" / "mediapipe" / "hand_landmarker.task"
DATASET_ID = "Samarth0710/bharatanatyam-mudra-dataset"
REVISION = "b3df2f983ef58d25e72d6191d1c4bddbbbbe96ab"
LICENSE = "MIT"


def _load_rows(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parquet in sorted(data_dir.glob("*.parquet")):
        table = pq.read_table(parquet, columns=["image", "label", "gesture_type", "label_id"])
        for row in table.to_pylist():
            if row.get("gesture_type") != "single_hand":
                continue
            image = row.get("image") or {}
            blob = image.get("bytes") if isinstance(image, dict) else None
            if not blob:
                continue
            rows.append({"bytes": blob, "label": str(row["label"]), "label_id": int(row["label_id"])})
    return rows


def _select_balanced(
    rows: list[dict[str, Any]], max_per_class: int, min_support: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    class_report: dict[str, int] = {}
    for label in sorted(by_label):
        items = by_label[label]
        class_report[label] = len(items)
        if len(items) < min_support:
            continue
        take = min(max_per_class, len(items))
        chosen = rng.permutation(len(items))[:take]
        selected.extend(items[int(i)] for i in chosen)
    return selected, class_report


def _extract_features(
    selected: list[dict[str, Any]], hand_model: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.40,
        min_hand_presence_confidence=0.40,
    )
    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[int] = []
    failures = {"no_hand": 0, "bad_landmarks": 0}
    started = time.time()

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        for image_id, row in enumerate(selected):
            try:
                pil = Image.open(io.BytesIO(row["bytes"])).convert("RGB")
                pil.thumbnail((256, 256))
                array = np.asarray(pil)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=array))
            except Exception:
                failures["bad_landmarks"] += 1
                continue
            if not result.hand_landmarks:
                failures["no_hand"] += 1
                continue
            lm = result.hand_landmarks[0]
            points = np.array([[p.x, p.y, p.z] for p in lm], dtype=float)
            feat = landmark_features(points)
            if feat is None:
                failures["bad_landmarks"] += 1
                continue
            features.append(feat)
            labels.append(row["label"])
            groups.append(image_id)
            features.append(mirror_features(feat))
            labels.append(row["label"])
            groups.append(image_id)
            if (image_id + 1) % 500 == 0:
                print(f"  landmarks {image_id + 1}/{len(selected)} · {time.time() - started:.1f}s", flush=True)

    if not features:
        raise RuntimeError("No hand landmarks were extracted; check the MediaPipe model and dataset images.")
    return np.stack(features), np.asarray(labels), np.asarray(groups), failures


def _dataset_provenance() -> dict[str, Any]:
    manifest_path = ROOT / "data" / "hf_datasets" / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return manifest.get("datasets", {}).get("mudra", {})
        except (OSError, ValueError):
            pass
    return {"repo_id": DATASET_ID, "revision": REVISION, "license": LICENSE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META)
    parser.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_MODEL)
    parser.add_argument("--max-per-class", type=int, default=180)
    parser.add_argument("--min-support", type=int, default=40)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.hand_model.is_file():
        raise SystemExit(f"Missing MediaPipe hand model: {args.hand_model}")
    if not list(args.data_dir.glob("*.parquet")):
        raise SystemExit(f"No parquet files found in {args.data_dir}")

    print(f"Reading single-hand rows from {args.data_dir}", flush=True)
    rows = _load_rows(args.data_dir)
    selected, class_report = _select_balanced(rows, args.max_per_class, args.min_support, args.seed)
    print(f"Selected {len(selected)} images from {len(class_report)} source classes "
          f"({sum(1 for n in class_report.values() if n >= args.min_support)} meet min support)", flush=True)

    X, y, groups, failures = _extract_features(selected, args.hand_model)
    print(f"Feature matrix {X.shape}; extraction failures {failures}", flush=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=2.0, solver="lbfgs")),
    ])
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    classes = sorted(set(y))
    metrics = {
        "accuracy": round(float(accuracy_score(y[test_idx], pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y[test_idx], pred)), 4),
        "macro_f1": round(float(f1_score(y[test_idx], pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y[test_idx], pred, average="weighted", zero_division=0)), 4),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_images": int(len(set(groups[train_idx]))),
        "test_images": int(len(set(groups[test_idx]))),
        "grouped_by_image": True,
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, args.model_out)
    meta = {
        "model_id": "nritya-mudra-landmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "single_hand_mudra_classification",
        "dataset": _dataset_provenance(),
        "license": LICENSE,
        "class_names": classes,
        "class_source_counts": class_report,
        "feature_description": (
            "21 wrist-relative (x,y) landmarks normalised by wrist-to-index-MCP palm length; "
            "original and x-mirrored views used for training and inference"
        ),
        "selection": {
            "max_per_class": args.max_per_class,
            "min_support": args.min_support,
            "selected_images": len(selected),
            "extraction_failures": failures,
        },
        "metrics": metrics,
        "scope": "bounded hand-shape prior for catalogue annotation, not music-to-choreography supervision",
    }
    args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved {args.model_out} and {args.meta_out}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
