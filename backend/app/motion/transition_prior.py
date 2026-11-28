"""Recorded-join transition prior for the choreography planner.

The model is trained by ``scripts/train_transition_prior.py`` from *recorded*
consecutive adavu phrase units in the local catalogue. It scores how well the
motion flow at the end of one recorded unit continues into the start of another.

This is deliberately complementary to the planner's existing endpoint pose and
hand-distance costs: endpoint distance is **not** a model feature, because every
recorded adjacent pair shares an endpoint frame and using that distance would
only relearn a tautology. The prior uses short boundary windows (velocity,
travel, foot height, and genuine hand flow) plus tempo/duration compatibility.

Scope: this is a bounded join-ordering prior over recorded motion. It is not
expert choreography supervision, does not synthesize frames or hand poses, and
makes no claim that a generated composition is the correct choreography for a
song.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = ROOT / "models" / "transition_prior.joblib"
DEFAULT_META_PATH = ROOT / "models" / "transition_prior.json"
DEFAULT_SUMMARY_PATH = ROOT / "data" / "catalogue" / "transition_features.json"

POSE_MATCH_INDICES = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
HAND_MATCH_INDICES = (0, 4, 8, 12, 16, 20)
DEFAULT_WINDOW = 6

FEATURE_NAMES: tuple[str, ...] = (
    "pose_velocity_rms",
    "travel_velocity_delta",
    "foot_height_delta",
    "bpm_log2_ratio",
    "duration_delta",
    "left_hand_endpoint_rms",
    "left_hand_velocity_rms",
    "left_hand_available",
    "right_hand_endpoint_rms",
    "right_hand_velocity_rms",
    "right_hand_available",
)


def _compact_pose(frame: dict[str, Any]) -> np.ndarray | None:
    """Return the 12x3 compact pose signature used by the catalogue."""
    arr = np.asarray(frame.get("pose_world"), dtype=float)
    if arr.shape != (33, 3) or not np.isfinite(arr).all():
        return None
    return arr[list(POSE_MATCH_INDICES)].copy()


def _compact_hand(frame: dict[str, Any], key: str) -> np.ndarray | None:
    """Return the 6-point hand signature when genuine 21-point landmarks exist."""
    arr = np.asarray(frame.get(key), dtype=float)
    if arr.shape != (21, 3) or not np.isfinite(arr).all():
        return None
    return arr[list(HAND_MATCH_INDICES)].copy()


def _mean_velocity(frames: list[dict[str, Any]], key: str) -> np.ndarray | None:
    extractor = _compact_pose if key == "pose_world" else lambda fr: _compact_hand(fr, key)
    vals = [extractor(fr) for fr in frames]
    if len(vals) < 2 or any(v is None or v.shape != vals[0].shape for v in vals):
        return None
    stacked = np.stack([v for v in vals if v is not None], axis=0)
    if not np.isfinite(stacked).all():
        return None
    return np.mean(np.diff(stacked, axis=0), axis=0)


def _hand_endpoint_velocity(
    frames: list[dict[str, Any]], key: str
) -> dict[str, list[list[float]]] | None:
    """Return genuine hand endpoint and time-normalised velocity, or None."""
    samples: list[tuple[float, np.ndarray]] = []
    for fr in frames:
        points = _compact_hand(fr, key)
        if points is None:
            continue
        t = float(fr.get("t"))
        if not math.isfinite(t):
            continue
        samples.append((t, points))
    if len(samples) < 2:
        return None
    times = np.asarray([s[0] for s in samples], dtype=float)
    points = np.stack([s[1] for s in samples], axis=0)
    dt = np.diff(times)
    usable = dt > 1e-6
    if not usable.any() or not np.isfinite(points).all():
        return None
    velocity = np.mean(np.diff(points, axis=0)[usable] / dt[usable, None, None], axis=0)
    return {
        "points": np.round(points[-1], 5).tolist(),
        "velocity": np.round(velocity, 5).tolist(),
    }


def build_boundary_summary(
    unit: dict[str, Any],
    frames: list[dict[str, Any]],
    window: int = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Build a JSON-serialisable summary of a unit's recorded head/tail motion."""
    start = int(unit["start_frame"])
    end = min(int(unit["end_frame"]), len(frames) - 1)
    if end <= start or not frames:
        raise ValueError(f"Unit {unit.get('unit_id')} has no usable frame window")

    win = max(2, int(window))
    head = frames[start : min(end + 1, start + win)]
    tail = frames[max(start, end - win + 1) : end + 1]

    pose_start = _compact_pose(head[0])
    pose_end = _compact_pose(tail[-1])
    if pose_start is None or pose_end is None:
        raise ValueError(f"Unit {unit.get('unit_id')} has invalid compact pose endpoints")

    vel_start = _mean_velocity(head, "pose_world")
    vel_end = _mean_velocity(tail, "pose_world")
    zero_pose = np.zeros((len(POSE_MATCH_INDICES), 3), dtype=float)

    head_tx = [float(fr.get("travel_x") or 0.0) for fr in head]
    tail_tx = [float(fr.get("travel_x") or 0.0) for fr in tail]
    return {
        "unit_id": str(unit["unit_id"]),
        "source_stem": str(unit.get("source_stem", "")),
        "pose_start": np.round(pose_start, 5).tolist(),
        "pose_end": np.round(pose_end, 5).tolist(),
        "vel_start": np.round(vel_start if vel_start is not None else zero_pose, 5).tolist(),
        "vel_end": np.round(vel_end if vel_end is not None else zero_pose, 5).tolist(),
        "travel_v_start": round(float(np.mean(np.diff(head_tx))) if len(head_tx) > 1 else 0.0, 5),
        "travel_v_end": round(float(np.mean(np.diff(tail_tx))) if len(tail_tx) > 1 else 0.0, 5),
        "foot_start": [
            round(float(head[0].get("foot_l_y") or 0.0), 5),
            round(float(head[0].get("foot_r_y") or 0.0), 5),
        ],
        "foot_end": [
            round(float(tail[-1].get("foot_l_y") or 0.0), 5),
            round(float(tail[-1].get("foot_r_y") or 0.0), 5),
        ],
        "left_hand_start": _hand_endpoint_velocity(head, "left_hand_rel"),
        "left_hand_end": _hand_endpoint_velocity(tail, "left_hand_rel"),
        "right_hand_start": _hand_endpoint_velocity(head, "right_hand_rel"),
        "right_hand_end": _hand_endpoint_velocity(tail, "right_hand_rel"),
    }


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def join_features(
    from_summary: dict[str, Any],
    from_unit: dict[str, Any],
    to_summary: dict[str, Any],
    to_unit: dict[str, Any],
) -> np.ndarray | None:
    """Compute the 11 recorded-join features. Endpoint pose distance is excluded."""
    try:
        vel_from = np.asarray(from_summary["vel_end"], dtype=float)
        vel_to = np.asarray(to_summary["vel_start"], dtype=float)
        foot_from = np.asarray(from_summary["foot_end"], dtype=float)
        foot_to = np.asarray(to_summary["foot_start"], dtype=float)
        bpm_from = float(from_unit.get("bpm") or from_summary.get("bpm") or 90.0)
        bpm_to = float(to_unit.get("bpm") or to_summary.get("bpm") or 90.0)
        dur_from = float(from_unit.get("duration_s") or from_summary.get("duration") or 0.0)
        dur_to = float(to_unit.get("duration_s") or to_summary.get("duration") or 0.0)
        values = [
            _rms(vel_from, vel_to),
            abs(float(from_summary["travel_v_end"]) - float(to_summary["travel_v_start"])),
            float(np.mean(np.abs(foot_from - foot_to))),
            abs(math.log2(max(bpm_from, 30.0) / max(bpm_to, 30.0))),
            abs(dur_from - dur_to),
        ]
        for side in ("left", "right"):
            a = from_summary.get(f"{side}_hand_end")
            b = to_summary.get(f"{side}_hand_start")
            if not a or not b:
                values.extend([0.15, 0.15, 0.0])
            else:
                values.extend([
                    _rms(np.asarray(a["points"], dtype=float), np.asarray(b["points"], dtype=float)),
                    _rms(np.asarray(a["velocity"], dtype=float), np.asarray(b["velocity"], dtype=float)),
                    1.0,
                ])
        features = np.asarray(values, dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
        return None
    return features


def pose_endpoint_distance(from_summary: dict[str, Any], to_summary: dict[str, Any]) -> float | None:
    """Endpoint RMS used only for dataset mining and baseline reporting."""
    try:
        a = np.asarray(from_summary["pose_end"], dtype=float)
        b = np.asarray(to_summary["pose_start"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return _rms(a, b)


def load_metadata(path: Path = DEFAULT_META_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_summaries(path: Path = DEFAULT_SUMMARY_PATH) -> dict[str, Any]:
    """Load the locally generated per-unit boundary summaries, or an empty mapping."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    units = payload.get("units", {})
    return units if isinstance(units, dict) else {}


class TransitionPrior:
    """Lazy wrapper around the trained join model and its boundary-summary artifact."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        meta_path: Path = DEFAULT_META_PATH,
        summary_path: Path = DEFAULT_SUMMARY_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.summary_path = Path(summary_path)
        self.metadata = load_metadata(self.meta_path)
        self.summaries = load_summaries(self.summary_path)
        self.class_names: list[str] = [str(x) for x in self.metadata.get("class_names", [])]
        self.positive_index: int = int(self.metadata.get("positive_index", 1))
        self.model: Any = None
        self.load_error: str | None = None
        if self.model_path.is_file():
            try:
                import joblib

                self.model = joblib.load(self.model_path)
            except Exception as exc:  # pragma: no cover - corrupted local artifact
                self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def model_available(self) -> bool:
        return self.model is not None

    @property
    def summaries_available(self) -> bool:
        return bool(self.summaries)

    @property
    def available(self) -> bool:
        return self.model_available and self.summaries_available

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_id", "unversioned"))

    def status(self) -> dict[str, Any]:
        """API-facing status with honest availability split and held-out metrics."""
        return {
            "available": self.available,
            "version": self.version if self.available else None,
            "load_error": self.load_error,
            "model_available": self.model_available,
            "summaries_available": self.summaries_available,
            "summary_unit_count": len(self.summaries),
            "summary_path": str(self.summary_path),
            "metadata": self.metadata or None,
        }

    def score_many(
        self,
        from_unit: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[float | None]:
        """Return natural-join probability for each candidate, or None when unscorable."""
        if not self.available:
            return [None] * len(candidates)
        from_summary = self.summaries.get(str(from_unit.get("unit_id")))
        if not from_summary:
            return [None] * len(candidates)

        rows: list[np.ndarray] = []
        positions: list[int] = []
        for pos, candidate in enumerate(candidates):
            to_summary = self.summaries.get(str(candidate.get("unit_id")))
            if not to_summary:
                continue
            features = join_features(from_summary, from_unit, to_summary, candidate)
            if features is not None:
                rows.append(features)
                positions.append(pos)
        if not rows:
            return [None] * len(candidates)

        scores: list[float | None] = [None] * len(candidates)
        try:
            proba = np.asarray(self.model.predict_proba(np.stack(rows)), dtype=float)
        except Exception:  # pragma: no cover - incompatible local artifact
            return scores
        if proba.ndim != 2 or self.positive_index >= proba.shape[1]:
            return scores
        for pos, row_idx in enumerate(positions):
            scores[row_idx] = round(float(proba[row_idx, self.positive_index]), 4)
        return scores
