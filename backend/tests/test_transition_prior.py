"""Focused tests for the bounded recorded-join transition prior."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.motion.transition_prior import (  # noqa: E402
    FEATURE_NAMES,
    TransitionPrior,
    build_boundary_summary,
    join_features,
    pose_endpoint_distance,
)
from app.planner.choreographer import _select_unit_sequence  # noqa: E402


class _ConstantProbabilityModel:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(np.array([0.8, 0.2], dtype=float), (len(X), 1))


def _frames(with_right_hand: bool = False) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for i in range(8):
        pose = [[0.0, 0.0, 0.0] for _ in range(33)]
        pose[11] = [round(i * 0.01, 4), 0.0, 0.0]
        left = np.zeros((21, 3), dtype=float)
        left[:, 0] = np.linspace(0.0, 0.2, 21) + i * 0.001
        left[9, :2] = [0.1 + i * 0.001, 0.0]
        frame: dict[str, object] = {
            "t": round(i / 30.0, 5),
            "pose_world": pose,
            "travel_x": round(i * 0.01, 4),
            "foot_l_y": 0.02 if i % 2 else 0.05,
            "foot_r_y": 0.04,
            "left_hand_rel": left.tolist(),
            "right_hand_rel": None,
        }
        if with_right_hand:
            frame["right_hand_rel"] = left.tolist()
        frames.append(frame)
    return frames


def _unit(unit_id: str) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "adavu": "Nattadavu",
        "source_stem": unit_id,
        "confirmed": True,
        "beat_times_s": [round(i * 0.5, 4) for i in range(9)],
        "bpm": 120.0,
        "duration_s": 4.0,
        "start_frame": 0,
        "end_frame": 7,
        "start_pose": [[0.0, 0.0, 0.0]] * 12,
        "end_pose": [[0.0, 0.0, 0.0]] * 12,
        "start_left_hand": None,
        "start_right_hand": None,
        "end_left_hand": None,
        "end_right_hand": None,
        "left_hand_ratio": 1.0,
        "right_hand_ratio": 1.0,
    }


def test_boundary_summary_is_serialisable_and_uses_genuine_hands_only() -> None:
    summary = build_boundary_summary(_unit("u1"), _frames())

    assert len(summary["vel_start"]) == 12
    assert len(summary["vel_end"]) == 12
    assert summary["left_hand_start"] is not None
    assert summary["left_hand_end"] is not None
    assert summary["right_hand_start"] is None
    assert summary["right_hand_end"] is None
    assert json.loads(json.dumps(summary))["unit_id"] == "u1"


def test_join_features_have_fixed_shape_and_honest_hand_availability() -> None:
    a = build_boundary_summary(_unit("a"), _frames())
    b = build_boundary_summary(_unit("b"), _frames())
    features = join_features(a, _unit("a"), b, _unit("b"))

    assert features is not None
    assert features.shape == (len(FEATURE_NAMES),)
    assert features[7] == 1.0  # left hand available
    assert features[10] == 0.0  # right hand genuinely missing
    assert pose_endpoint_distance(a, a) > 0.0  # this synthetic unit starts and ends at different x
    assert pose_endpoint_distance(a, b) > 0.0
    assert np.isfinite(pose_endpoint_distance(a, b))
    assert join_features(a, _unit("a"), {}, _unit("b")) is None


def test_transition_prior_without_local_artifacts_is_inert(tmp_path: Path) -> None:
    prior = TransitionPrior(
        tmp_path / "missing.joblib",
        tmp_path / "missing.json",
        tmp_path / "missing_summaries.json",
    )

    assert prior.available is False
    assert prior.version == "unversioned"
    assert prior.status()["available"] is False
    assert prior.score_many(_unit("a"), [_unit("b")]) == [None]


def test_transition_prior_scores_with_metadata_positive_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps({
            "model_id": "unit-test-join-v1",
            "class_names": ["0", "1"],
            "positive_index": 1,
        }),
        encoding="utf-8",
    )
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"stub")
    monkeypatch.setattr(joblib, "load", lambda _: _ConstantProbabilityModel())
    summary_file = tmp_path / "summaries.json"
    summary_file.write_text(
        json.dumps({
            "units": {
                "a": build_boundary_summary(_unit("a"), _frames()),
                "b": build_boundary_summary(_unit("b"), _frames()),
            }
        }),
        encoding="utf-8",
    )

    prior = TransitionPrior(model_file, meta, summary_file)
    assert prior.available is True
    assert prior.version == "unit-test-join-v1"
    assert prior.score_many(_unit("a"), [_unit("b"), _unit("a")]) == [0.2, 0.2]


class _StubJoinPrior:
    available = True
    version = "stub-join-v1"

    def score_many(self, from_unit: dict[str, object], candidates: list[dict[str, object]]) -> list[float]:
        return [1.0 if u["unit_id"] == "b" else 0.0 for u in candidates]


def test_recorded_join_prior_changes_later_phrase_choice() -> None:
    beats = [
        {"time_s": round(i * 0.5, 4), "is_sam": i == 0, "tala_pos": (i % 8) + 1}
        for i in range(17)
    ]
    units = [_unit("a"), _unit("b")]
    units[0]["bpm"] = 120.0
    units[1]["bpm"] = 100.0

    unweighted = _select_unit_sequence(units, beats, song_bpm=120.0, seed=7)
    assert [s["unit"]["unit_id"] for s in unweighted] == ["a", "a"]
    assert all(s["transition_score"] is None for s in unweighted)

    weighted = _select_unit_sequence(
        units,
        beats,
        song_bpm=120.0,
        seed=7,
        transition_prior=_StubJoinPrior(),
        transition_weight=5.0,
    )
    assert weighted[0]["unit"]["unit_id"] == "a"
    assert weighted[1]["unit"]["unit_id"] == "b"
    assert weighted[0]["transition_score"] is None
    assert weighted[1]["transition_score"] == 1.0
