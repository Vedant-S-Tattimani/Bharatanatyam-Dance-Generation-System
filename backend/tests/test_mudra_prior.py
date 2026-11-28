"""Focused tests for the bounded, trained mudra hand-shape prior."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.motion.mudra import (  # noqa: E402
    MudraClassifier,
    annotate_catalogue,
    landmark_features,
    mirror_features,
)
from app.planner.choreographer import _select_unit_sequence  # noqa: E402


class _ConstantProbabilityModel:
    def __init__(self, proba: list[float]) -> None:
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.proba, (len(X), 1))


class _StubClassifier:
    available = True
    version = "stub-v1"
    metadata: dict[str, object] = {}

    def predict(self, points: object) -> dict[str, object] | None:
        return None if points is None else {"label": "TestMudra", "confidence": 0.75}


def _valid_points() -> np.ndarray:
    points = np.zeros((21, 3), dtype=float)
    points[:, 0] = np.linspace(0.0, 0.2, 21)
    points[9, :2] = [0.1, 0.0]
    return points


def test_landmark_features_are_palm_normalized_and_wrist_anchored() -> None:
    features = landmark_features(_valid_points())
    assert features is not None
    assert features.shape == (42,)
    assert features.dtype == np.float32
    assert np.allclose(features[:2], 0.0)
    assert np.isclose(features[18], 1.0)  # index 9 x / palm length
    mirrored = mirror_features(features)
    assert np.allclose(mirrored[0::2], -features[0::2])
    assert np.allclose(mirrored[1::2], features[1::2])


@pytest.mark.parametrize(
    "points",
    [
        None,
        np.zeros((20, 3)),
        np.full((21, 3), np.nan),
        np.zeros((21, 3)),
    ],
)
def test_landmark_features_reject_unusable_points(points: object) -> None:
    assert landmark_features(points) is None


def test_classifier_without_local_model_is_inert(tmp_path: Path) -> None:
    classifier = MudraClassifier(tmp_path / "missing.joblib", tmp_path / "missing.json")
    assert classifier.available is False
    assert classifier.version == "unversioned"
    assert classifier.predict(_valid_points()) is None


def test_classifier_loads_metadata_and_uses_mirrored_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps({"model_id": "unit-test-v1", "class_names": ["A", "B"]}),
        encoding="utf-8",
    )
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"stub")
    monkeypatch.setattr(joblib, "load", lambda _: _ConstantProbabilityModel([0.2, 0.8]))

    classifier = MudraClassifier(model_file, meta)
    assert classifier.available is True
    assert classifier.version == "unit-test-v1"
    prediction = classifier.predict(_valid_points())
    assert prediction == {
        "label": "B",
        "confidence": 0.8,
        "basis": "trained-mudra-landmark-classifier",
        "model": "unit-test-v1",
    }
    assert classifier.predict(None) is None


def test_annotate_catalogue_uses_existing_landmarks_only(tmp_path: Path) -> None:
    motion_path = tmp_path / "data" / "motion" / "clip.json"
    motion_path.parent.mkdir(parents=True)
    frames = [
        {
            "t": i / 30.0,
            "pose_world": [[0.0, 0.0, 0.0]] * 33,
            "left_hand_rel": _valid_points().tolist(),
            "right_hand_rel": None,
        }
        for i in range(4)
    ]
    motion_path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    catalogue = {
        "units": [
            {
                "unit_id": "u1",
                "motion_path": "data/motion/clip.json",
                "start_frame": 0,
                "end_frame": 3,
            }
        ]
    }
    out = tmp_path / "data" / "catalogue" / "mudra_annotations.json"

    annotations = annotate_catalogue(tmp_path, catalogue, _StubClassifier(), out_path=out)

    assert annotations["u1"]["labels"] == ["TestMudra"]
    assert annotations["u1"]["left"] == {
        "label": "TestMudra",
        "confidence": 0.75,
        "support": 4,
        "detections": 4,
    }
    assert "right" not in annotations["u1"]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["unit_count"] == 1
    assert payload["model"] == "stub-v1"


def _unit(unit_id: str, bpm: float) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "adavu": "Nattadavu",
        "source_stem": unit_id,
        "confirmed": True,
        "beat_times_s": [round(i * 0.5, 4) for i in range(9)],
        "bpm": bpm,
        "start_pose": [[0.0, 0.0, 0.0]] * 12,
        "end_pose": [[0.0, 0.0, 0.0]] * 12,
        "start_left_hand": None,
        "start_right_hand": None,
        "end_left_hand": None,
        "end_right_hand": None,
        "left_hand_ratio": 1.0,
        "right_hand_ratio": 1.0,
    }


def test_mudra_annotations_change_phrase_scoring() -> None:
    beats = [
        {"time_s": round(i * 0.5, 4), "is_sam": i == 0, "tala_pos": (i % 8) + 1}
        for i in range(9)
    ]
    units = [_unit("a", 120.0), _unit("b", 100.0)]

    unweighted = _select_unit_sequence(units, beats, song_bpm=120.0, seed=7)
    assert unweighted[0]["unit"]["unit_id"] == "a"

    annotations = {
        "b": {
            "labels": ["TestMudra"],
            "left": {"confidence": 1.0},
            "right": {"confidence": 1.0},
        }
    }
    weighted = _select_unit_sequence(
        units,
        beats,
        song_bpm=120.0,
        seed=7,
        mudra_annotations=annotations,
        mudra_weight=2.0,
    )
    assert weighted[0]["unit"]["unit_id"] == "b"
    assert weighted[0]["mudra_labels"] == ["TestMudra"]
    assert weighted[0]["mudra_confidence"] == 1.0
