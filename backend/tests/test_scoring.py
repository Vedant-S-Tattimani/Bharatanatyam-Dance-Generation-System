"""Unit tests for scripts/score_foot_strikes.py (detector precision/recall/signed timing vs beat error)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from score_foot_strikes import evaluate_clip_labels  # noqa: E402


def test_evaluate_clip_labels_reports_detector_and_beat_errors_separately():
    gt_data = {
        "stem": "TestClip",
        "fps": 30.0,
        "true_strikes": [
            {"frame": 30, "time_s": 1.000, "side": "left"},
            {"frame": 60, "time_s": 2.000, "side": "right"},
            {"frame": 90, "time_s": 3.000, "side": "left"},
        ],
        "true_beats_s": [1.000, 2.000, 3.000],
    }
    motion_data = {
        "fps": 30.0,
        "foot_strikes": [
            # Matched TP 1: 30ms early (-30ms)
            {"frame": 29, "time_s": 0.970, "side": "left"},
            # Matched TP 2: 40ms early (-40ms)
            {"frame": 59, "time_s": 1.960, "side": "right"},
            # False positive INSIDE the labeled [1.0..3.0s] span: still pose jitter on right at t=2.5s
            {"frame": 75, "time_s": 2.500, "side": "right"},
            # Missing FN: true strike at f=90 left missed
        ],
    }
    audio_data = {
        "beats": [
            {"time_s": 1.020},
            {"time_s": 2.020},
            {"time_s": 3.020},
        ]
    }

    res = evaluate_clip_labels(gt_data, motion_data, audio_data, tol_ms=100.0)
    det = res["detector_vs_true_strikes"]
    assert det["tp"] == 2 and det["fp"] == 1 and det["fn"] == 1
    assert abs(det["precision"] - (2 / 3)) < 1e-3
    assert abs(det["recall"] - (2 / 3)) < 1e-3
    assert det["timing_error"]["median_signed_ms"] == -35.0
    assert det["timing_error"]["early_pct"] == 100.0

    bt = res["beat_tracker_vs_true_beats"]
    assert bt["precision"] == 1.0 and bt["recall"] == 1.0
    assert bt["error_ms"]["median_ms"] == 20.0

    ts = res["true_strikes_vs_true_beats"]
    assert ts["median_abs_ms"] == 0.0
    assert ts["beat_period_ms"] == 1000.0
    assert ts["same_tempo_random_baseline_ms"] == 250.0


def test_scorer_ignores_unlabeled_clip_regions():
    gt_data = {
        "stem": "PartialClip",
        "fps": 30.0,
        "labeled_range_s": [10.0, 20.0],
        "true_strikes": [
            {"frame": 330, "time_s": 11.000, "side": "left"},
            {"frame": 360, "time_s": 12.000, "side": "right"},
        ],
        "true_beats_s": [11.000, 12.000],
    }
    motion_data = {
        "fps": 30.0,
        "foot_strikes": [
            {"frame": 30, "time_s": 1.000, "side": "left"},    # Unlabeled early intro -> ignored!
            {"frame": 330, "time_s": 11.000, "side": "left"},  # Inside [10, 20]
            {"frame": 360, "time_s": 12.000, "side": "right"}, # Inside [10, 20]
            {"frame": 900, "time_s": 30.000, "side": "left"},  # Unlabeled late part -> ignored!
        ],
    }
    audio_data = {
        "beats": [
            {"time_s": 1.000},   # Unlabeled intro -> ignored!
            {"time_s": 11.010},
            {"time_s": 12.010},
            {"time_s": 35.000},  # Unlabeled outro -> ignored!
        ]
    }
    res = evaluate_clip_labels(gt_data, motion_data, audio_data, tol_ms=100.0)
    assert res["detector_vs_true_strikes"]["ignored_outside_range"] == 2
    assert res["detector_vs_true_strikes"]["fp"] == 0
    assert res["detector_vs_true_strikes"]["precision"] == 1.0
    assert res["detector_vs_true_strikes"]["recall"] == 1.0
    assert res["beat_tracker_vs_true_beats"]["ignored_outside_range"] == 2
    assert res["beat_tracker_vs_true_beats"]["precision"] == 1.0
    assert res["beat_tracker_vs_true_beats"]["recall"] == 1.0

    # Also verify automatic [min..max] fallback range when labeled_range_s is omitted
    gt_auto = {**gt_data, "labeled_range_s": None}
    res_auto = evaluate_clip_labels(gt_auto, motion_data, audio_data, tol_ms=100.0)
    assert res_auto["detector_vs_true_strikes"]["ignored_outside_range"] == 2
    assert res_auto["detector_vs_true_strikes"]["precision"] == 1.0
    assert res_auto["beat_tracker_vs_true_beats"]["ignored_outside_range"] == 2
    assert res_auto["beat_tracker_vs_true_beats"]["precision"] == 1.0

