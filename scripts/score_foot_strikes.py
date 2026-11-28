"""
score_foot_strikes.py – Score detected foot strikes and detected beats separately
against human-labeled ground-truth JSONs (data/labels/<stem>_gt.json).

Reports three separate groups:
  1. Foot-Strike Detector vs User True Strikes (precision, recall, F1, signed timing error)
  2. Beat Tracker vs User True Beats (precision, recall, signed & abs ms error)
  3. User True Strikes vs User True Beats (true synchronization vs same-tempo chance)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.audio.analyzer import compute_beat_error_ms  # noqa: E402


def _bipartite_greedy_match(
    preds: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    tol_s: float,
    match_side: bool = True,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], float]], list[dict[str, Any]], list[dict[str, Any]]]:
    """One-to-one closest-time match within tol_s (optionally enforcing matching L/R side)."""
    candidates: list[tuple[float, int, int]] = []
    for i, p in enumerate(preds):
        for j, g in enumerate(truths):
            if match_side and p.get("side") != g.get("side"):
                continue
            dt = abs(float(p["time_s"]) - float(g["time_s"]))
            if dt <= tol_s:
                candidates.append((dt, i, j))
    candidates.sort(key=lambda x: x[0])

    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for _, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        signed_ms = (float(preds[i]["time_s"]) - float(truths[j]["time_s"])) * 1000.0
        matches.append((preds[i], truths[j], signed_ms))

    fps = [p for idx, p in enumerate(preds) if idx not in used_p]
    fns = [g for idx, g in enumerate(truths) if idx not in used_g]
    return matches, fps, fns


def _resolve_eval_window(
    explicit_range: list[float] | tuple[float, float] | None,
    item_times_s: list[float],
    pad_s: float,
) -> tuple[float, float] | None:
    """
    Determine evaluation [start_s, end_s] so unlabeled clip regions are never penalized.
    Uses explicit labeled_range_s when supplied; otherwise derives [min(true)-pad, max(true)+pad].
    """
    if explicit_range and len(explicit_range) == 2 and explicit_range[0] is not None and explicit_range[1] is not None:
        return (min(float(explicit_range[0]), float(explicit_range[1])), max(float(explicit_range[0]), float(explicit_range[1])))
    if item_times_s:
        return (max(0.0, float(min(item_times_s)) - pad_s), float(max(item_times_s)) + pad_s)
    return None


def evaluate_clip_labels(
    gt_data: dict[str, Any],
    motion_data: dict[str, Any],
    audio_data: dict[str, Any],
    tol_ms: float = 100.0,
) -> dict[str, Any]:
    """Evaluate detector strikes, detected beats, and true strike-to-beat sync strictly inside the labeled time range."""
    fps = float(motion_data.get("fps") or gt_data.get("fps") or 30.0)
    tol_s = tol_ms / 1000.0
    explicit_range = gt_data.get("labeled_range_s")

    true_strikes_all = [
        {
            "frame": int(st["frame"]),
            "time_s": float(st.get("time_s", st["frame"] / fps)),
            "side": st["side"],
        }
        for st in gt_data.get("true_strikes", [])
    ]
    strike_win = _resolve_eval_window(explicit_range, [s["time_s"] for s in true_strikes_all], pad_s=tol_s)
    if strike_win is not None:
        s0, s1 = strike_win
        true_strikes = [s for s in true_strikes_all if s0 - 1e-6 <= s["time_s"] <= s1 + 1e-6]
        det_strikes_all = motion_data.get("foot_strikes", [])
        det_strikes = [
            {"frame": int(st["frame"]), "time_s": float(st["time_s"]), "side": st["side"]}
            for st in det_strikes_all
            if s0 - 1e-6 <= float(st["time_s"]) <= s1 + 1e-6
        ]
        det_strikes_ignored = len(det_strikes_all) - len(det_strikes)
    else:
        true_strikes = []
        det_strikes = []
        det_strikes_ignored = len(motion_data.get("foot_strikes", []))

    matches, fp_list, fn_list = _bipartite_greedy_match(det_strikes, true_strikes, tol_s=tol_s, match_side=True)
    tp = len(matches)
    fp = len(fp_list)
    fn = len(fn_list)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None

    if matches:
        signed_errs = np.array([m[2] for m in matches], dtype=float)
        det_timing = {
            "median_signed_ms": round(float(np.median(signed_errs)), 2),
            "mean_signed_ms": round(float(np.mean(signed_errs)), 2),
            "median_abs_ms": round(float(np.median(np.abs(signed_errs))), 2),
            "median_signed_frames": round(float(np.median(signed_errs) / 1000.0 * fps), 2),
            "early_pct": round(float(np.mean(signed_errs < 0) * 100.0), 1),
            "late_pct": round(float(np.mean(signed_errs > 0) * 100.0), 1),
        }
    else:
        det_timing = {
            "median_signed_ms": None,
            "mean_signed_ms": None,
            "median_abs_ms": None,
            "median_signed_frames": None,
            "early_pct": None,
            "late_pct": None,
        }

    # 2. Beat Tracker vs True Beats (evaluated strictly within labeled beat range)
    true_beats_all = [float(b) for b in gt_data.get("true_beats_s", [])]
    beat_win = _resolve_eval_window(explicit_range, true_beats_all, pad_s=tol_s)
    if beat_win is not None:
        b0, b1 = beat_win
        true_beats = [t for t in true_beats_all if b0 - 1e-6 <= t <= b1 + 1e-6]
        det_beats_all = [float(b["time_s"]) for b in audio_data.get("beats", [])]
        det_beats = [t for t in det_beats_all if b0 - 1e-6 <= t <= b1 + 1e-6]
        det_beats_ignored = len(det_beats_all) - len(det_beats)
    else:
        true_beats = []
        det_beats = []
        det_beats_ignored = len(audio_data.get("beats", []))

    b_matches, b_fps, b_fns = _bipartite_greedy_match(
        [{"time_s": t} for t in det_beats],
        [{"time_s": t} for t in true_beats],
        tol_s=tol_s,
        match_side=False,
    )
    btp, bfp, bfn = len(b_matches), len(b_fps), len(b_fns)
    beat_eval = {
        "eval_range_s": [round(beat_win[0], 4), round(beat_win[1], 4)] if beat_win else None,
        "ignored_outside_range": det_beats_ignored,
        "true_beat_count": len(true_beats),
        "detected_beat_count": len(det_beats),
        "precision": round(btp / (btp + bfp), 4) if (btp + bfp) else None,
        "recall": round(btp / (btp + bfn), 4) if (btp + bfn) else None,
        "error_ms": compute_beat_error_ms(det_beats, true_beats),
    }

    # 3. True Dancer Strikes vs True Beats vs same-tempo chance
    if true_strikes and true_beats:
        bt_arr = np.asarray(sorted(true_beats), dtype=float)
        st_arr = np.asarray([s["time_s"] for s in true_strikes], dtype=float)
        true_signed = np.array([(t - bt_arr[int(np.argmin(np.abs(bt_arr - t)))]) * 1000.0 for t in st_arr], dtype=float)
        period_ms = float(np.median(np.diff(bt_arr)) * 1000.0) if len(bt_arr) >= 2 else None
        true_sync = {
            "beat_period_ms": round(period_ms, 2) if period_ms else None,
            "same_tempo_random_baseline_ms": round(period_ms * 0.25, 2) if period_ms else None,
            "median_abs_ms": round(float(np.median(np.abs(true_signed))), 2),
            "median_signed_ms": round(float(np.median(true_signed)), 2),
            "early_pct": round(float(np.mean(true_signed < 0) * 100.0), 1),
        }
    else:
        true_sync = {"beat_period_ms": None, "same_tempo_random_baseline_ms": None, "median_abs_ms": None, "median_signed_ms": None, "early_pct": None}

    return {
        "stem": gt_data.get("stem"),
        "tol_ms": tol_ms,
        "detector_vs_true_strikes": {
            "eval_range_s": [round(strike_win[0], 4), round(strike_win[1], 4)] if strike_win else None,
            "ignored_outside_range": det_strikes_ignored,
            "true_count": len(true_strikes),
            "detected_count": len(det_strikes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "timing_error": det_timing,
        },
        "beat_tracker_vs_true_beats": beat_eval,
        "true_strikes_vs_true_beats": true_sync,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--tol-ms", type=float, default=100.0)
    args = ap.parse_args()

    labels_dir = ROOT / args.labels_dir
    files = sorted(labels_dir.glob("*_gt.json"))
    if not files:
        print(f"No *_gt.json label files found in {labels_dir}. Use focus_6_labeler.html to mark labels first.", file=sys.stderr)
        sys.exit(1)

    results = []
    for lp in files:
        gt = json.loads(lp.read_text(encoding="utf-8"))
        stem = gt["stem"]
        m = json.loads((ROOT / "data" / "motion" / f"{stem}.json").read_text(encoding="utf-8"))
        a = json.loads((ROOT / "data" / "audio" / f"{stem}_analysis.json").read_text(encoding="utf-8"))
        results.append(evaluate_clip_labels(gt, m, a, tol_ms=args.tol_ms))

    out_path = ROOT / "data" / "diagnostics" / "human_label_scores.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

