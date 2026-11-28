"""Trained mudra landmark classifier and catalogue annotation helpers.

The model is trained on the MIT-licensed Hugging Face Bharatanatyam Mudra
Dataset by ``scripts/train_mudra_landmark_model.py``. It consumes the same
wrist-relative 21-point hand landmarks that the motion extractor stores in
``left_hand_rel`` / ``right_hand_rel``, so no image or extra inference is
needed at choreography time.

This is a bounded hand-shape prior, not a music-to-expert-choreography model.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = ROOT / "models" / "mudra_landmark_classifier.joblib"
DEFAULT_META_PATH = ROOT / "models" / "mudra_landmark_classifier.json"
DEFAULT_ANNOTATION_PATH = ROOT / "data" / "catalogue" / "mudra_annotations.json"


def landmark_features(points: Any) -> np.ndarray | None:
    """Return orientation-independent 21x2 palm-normalised landmark features."""
    if points is None:
        return None
    arr = np.asarray(points, dtype=float)
    if arr.shape != (21, 3) or not np.isfinite(arr).all():
        return None
    xy = arr[:, :2]
    palm = float(np.linalg.norm(xy[9] - xy[0]))
    if palm < 1e-6:
        return None
    return ((xy - xy[0]) / palm).astype(np.float32).reshape(-1)


def mirror_features(features: np.ndarray) -> np.ndarray:
    """Mirror x coordinates so left/right camera views share one vocabulary."""
    arr = np.asarray(features, dtype=np.float32).reshape(21, 2).copy()
    arr[:, 0] *= -1.0
    return arr.reshape(-1)


def load_metadata(path: Path = DEFAULT_META_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class MudraClassifier:
    """Lazy wrapper around the saved sklearn pipeline."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        meta_path: Path = DEFAULT_META_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.metadata = load_metadata(self.meta_path)
        self.model: Any = None
        self.class_names: list[str] = list(self.metadata.get("class_names", []))
        self.load_error: str | None = None
        if self.model_path.is_file():
            try:
                import joblib
                self.model = joblib.load(self.model_path)
                if not self.class_names:
                    self.class_names = [str(c) for c in getattr(self.model, "classes_", [])]
            except Exception as exc:  # pragma: no cover - corrupted local model
                self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None and bool(self.class_names)

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_id", "unversioned"))

    def predict(self, points: Any) -> dict[str, Any] | None:
        """Classify one wrist-relative landmark set, or None when unusable."""
        if not self.available:
            return None
        features = landmark_features(points)
        if features is None:
            return None
        probs = np.asarray(self.model.predict_proba([features])[0], dtype=float)
        mirrored = np.asarray(self.model.predict_proba([mirror_features(features)])[0], dtype=float)
        combined = np.maximum(probs, mirrored)
        idx = int(np.argmax(combined))
        return {
            "label": self.class_names[idx],
            "confidence": round(float(combined[idx]), 4),
            "basis": "trained-mudra-landmark-classifier",
            "model": self.version,
        }


def load_annotations(root: Path = ROOT, path: Path | None = None) -> dict[str, Any]:
    """Load locally generated mudra annotations, or an empty mapping."""
    p = Path(path) if path else Path(root) / DEFAULT_ANNOTATION_PATH
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get("units", {}) if isinstance(data, dict) else {}


def annotation_summary(root: Path = ROOT, path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else Path(root) / DEFAULT_ANNOTATION_PATH
    anns = load_annotations(root, p)
    return {
        "path": str(p.relative_to(root)).replace("\\", "/") if p.is_relative_to(root) else str(p),
        "available": p.is_file(),
        "unit_count": len(anns),
        "label_count": len({label for a in anns.values() for label in a.get("labels", [])}),
    }


def annotate_catalogue(
    root: Path,
    catalogue: dict[str, Any],
    classifier: MudraClassifier,
    out_path: Path | None = None,
    max_frames_per_unit: int = 8,
) -> dict[str, Any]:
    """Annotate catalogue units by classifying real detected hand landmarks."""
    if not classifier.available:
        raise RuntimeError("Mudra classifier is not available; train it first.")
    p = Path(out_path) if out_path else Path(root) / DEFAULT_ANNOTATION_PATH
    annotations: dict[str, Any] = {}
    motion_cache: dict[str, dict[str, Any]] = {}

    for unit in catalogue.get("units", []):
        motion_rel = unit.get("motion_path")
        if not motion_rel:
            continue
        mpath = Path(root) / motion_rel
        if not mpath.is_file():
            continue
        if str(mpath) not in motion_cache:
            motion_cache[str(mpath)] = json.loads(mpath.read_text(encoding="utf-8"))
        frames = motion_cache[str(mpath)].get("frames", [])
        start = int(unit.get("start_frame", 0))
        end = min(int(unit.get("end_frame", len(frames) - 1)), len(frames) - 1)
        if end <= start:
            continue
        sample_idx = np.linspace(start, end, min(max_frames_per_unit, end - start + 1), dtype=int)

        sides: dict[str, Any] = {}
        for side, key in (("left", "left_hand_rel"), ("right", "right_hand_rel")):
            preds = [classifier.predict(frames[int(i)].get(key)) for i in sample_idx]
            valid = [p for p in preds if p]
            if not valid:
                continue
            counts = Counter(p["label"] for p in valid)
            label, votes = counts.most_common(1)[0]
            confidence = float(np.mean([p["confidence"] for p in valid if p["label"] == label]))
            sides[side] = {
                "label": label,
                "confidence": round(confidence, 4),
                "support": int(votes),
                "detections": len(valid),
            }

        labels = sorted({s["label"] for s in sides.values()})
        if not labels:
            continue
        annotations[str(unit["unit_id"])] = {
            **sides,
            "labels": labels,
            "model": classifier.version,
        }

    payload = {
        "version": 1,
        "model": classifier.version,
        "model_metadata": classifier.metadata,
        "unit_count": len(annotations),
        "units": annotations,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return annotations
