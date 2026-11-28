"""
builder.py – Step 3: Build a provisional Bharatanatyam adavu phrase catalogue
from extracted Phase 1 motion JSONs and audio beat/tala analyses.

Quality gates (PROVISIONAL until user approves thresholds):
  - Excludes broken clips and unconfirmed alias clips (e.g., KorvaiAdavu, MandiAdavu2).
  - Marks ThaThaiThamAdavu units as confirmed=False (ThaThaiTham1-6 also carry KorvaiAdavu in adavu_name2).
  - Excludes clips with high monocular scale drift (torso_scale_cv > max_scale_cv).
  - Splits at uninterpolated >100ms gaps (never bridging long gaps).
  - Retains contiguous valid segments >= min_segment_s with per-segment L/R hand coverage >= min_hand_ratio.
  - Slices clean segments into musical avartanam / beat-aligned phrase units.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# Key joint subset for fast boundary pose matching (shoulders, elbows, wrists, hips, knees, ankles)
POSE_MATCH_INDICES = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
HAND_MATCH_INDICES = (0, 4, 8, 12, 16, 20)  # wrist + 5 fingertips
UNCONFIRMED_CATEGORIES = {"ThaThaiThamAdavu"}


def _is_unconfirmed_alias_clip(clip: dict[str, Any]) -> bool:
    """Return True when a clip only enters a canonical adavu through an unconfirmed alias."""
    alts = clip.get("adavu_alternates") or []
    if not alts:
        return False
    fname_norm = clip["filename"].lower().replace(" ", "")
    canon_norm = clip["adavu_canonical"].lower().replace(" ", "")
    for alt in alts:
        alt_stem = alt.lower().replace("adavu", "").replace(" ", "")
        if alt_stem and fname_norm.startswith(alt_stem) and not fname_norm.startswith(canon_norm[:6]):
            return True
    return False


def _pose_signature(frame_rec: dict[str, Any]) -> list[list[float]]:
    """Extract compact 12-joint 3D pose signature for transition distance scoring."""
    pw = frame_rec.get("pose_world")
    if not pw:
        return []
    return [[round(float(x), 4) for x in pw[j]] for j in POSE_MATCH_INDICES]


def _hand_signature(frame_rec: dict[str, Any], key: str) -> list[list[float]] | None:
    """Extract compact 6-point hand signature when hand landmarks were detected, else None."""
    pts = frame_rec.get(key)
    if not pts:
        return None
    return [[round(float(x), 4) for x in pts[j]] for j in HAND_MATCH_INDICES]


def extract_clean_segments(
    frames: list[dict[str, Any]],
    fps: float,
    min_segment_s: float = 2.0,
    min_hand_ratio: float = 0.50,
) -> list[dict[str, Any]]:
    """Split frames at any invalid/long-gap frames and keep clean segments meeting duration and hand thresholds."""
    min_frames = max(2, int(round(min_segment_s * fps)))
    segments: list[dict[str, Any]] = []
    i = 0
    n = len(frames)

    while i < n:
        if not frames[i].get("valid"):
            i += 1
            continue
        start = i
        while i < n and frames[i].get("valid"):
            i += 1
        end = i - 1
        length = end - start + 1
        if length < min_frames:
            continue

        seg_frames = frames[start : end + 1]
        lh_ratio = float(np.mean([fr.get("left_hand_rel") is not None for fr in seg_frames]))
        rh_ratio = float(np.mean([fr.get("right_hand_rel") is not None for fr in seg_frames]))
        if lh_ratio < min_hand_ratio or rh_ratio < min_hand_ratio:
            continue

        segments.append({
            "start_frame": start,
            "end_frame": end,
            "duration_s": round(length / fps, 3),
            "left_hand_ratio": round(lh_ratio, 4),
            "right_hand_ratio": round(rh_ratio, 4),
        })
    return segments


def _collect_unit_strikes(
    all_strikes: list[dict[str, Any]],
    s_idx: int,
    e_idx: int,
) -> list[dict[str, Any]]:
    """Return foot-strike events falling inside [s_idx, e_idx]."""
    return [
        {"frame": int(st["frame"]), "time_s": float(st["time_s"]), "side": st["side"]}
        for st in all_strikes
        if s_idx <= int(st["frame"]) <= e_idx
    ]


def _slice_segment_into_units(
    seg: dict[str, Any],
    frames: list[dict[str, Any]],
    all_strikes: list[dict[str, Any]],
    beat_times_s: list[float],
    beat_source: str,
    fps: float,
    bpm: float,
    tala_beats: int,
    adavu: str,
    confirmed: bool,
    stem: str,
    motion_rel_path: str,
    target_phrase_beats: int = 8,
) -> list[dict[str, Any]]:
    """Slice a clean valid segment into whole-phrase beat-aligned units (typically 8-beat Adi avartanams)."""
    t_start = float(frames[seg["start_frame"]]["t"])
    t_end = float(frames[seg["end_frame"]]["t"])
    in_beats = [float(b) for b in beat_times_s if t_start - 0.04 <= b <= t_end + 0.04]

    if len(in_beats) < max(4, target_phrase_beats // 2 + 1):
        return []

    units: list[dict[str, Any]] = []
    stride = max(4, target_phrase_beats)
    idx = 0
    while idx + stride - 1 < len(in_beats):
        win_beats = in_beats[idx : idx + stride]
        s_idx = max(seg["start_frame"], int(round(win_beats[0] * fps)))
        e_idx = min(seg["end_frame"], int(round(win_beats[-1] * fps)))
        if idx + stride < len(in_beats):
            next_end = int(round(in_beats[idx + stride] * fps))
            if next_end <= seg["end_frame"]:
                win_beats = in_beats[idx : idx + stride + 1]
                e_idx = next_end

        if e_idx > s_idx + 10:
            sub_frames = frames[s_idx : e_idx + 1]
            lh = float(np.mean([fr.get("left_hand_rel") is not None for fr in sub_frames]))
            rh = float(np.mean([fr.get("right_hand_rel") is not None for fr in sub_frames]))
            units.append({
                "unit_id": f"{stem}_s{s_idx}_e{e_idx}",
                "adavu": adavu,
                "confirmed": confirmed,
                "source_stem": stem,
                "motion_path": motion_rel_path,
                "fps": fps,
                "bpm": round(bpm, 2),
                "tala_beats": tala_beats,
                "beat_source": beat_source,
                "beat_count": len(win_beats),
                "beat_times_s": [round(float(b), 4) for b in win_beats],
                "start_frame": s_idx,
                "end_frame": e_idx,
                "duration_s": round((e_idx - s_idx) / fps, 3),
                "left_hand_ratio": round(lh, 4),
                "right_hand_ratio": round(rh, 4),
                "start_pose": _pose_signature(frames[s_idx]),
                "end_pose": _pose_signature(frames[e_idx]),
                "start_left_hand": _hand_signature(frames[s_idx], "left_hand_rel"),
                "start_right_hand": _hand_signature(frames[s_idx], "right_hand_rel"),
                "end_left_hand": _hand_signature(frames[e_idx], "left_hand_rel"),
                "end_right_hand": _hand_signature(frames[e_idx], "right_hand_rel"),
                "start_travel_x": float(frames[s_idx].get("travel_x") or 0.0),
                "end_travel_x": float(frames[e_idx].get("travel_x") or 0.0),
                "foot_strikes": _collect_unit_strikes(all_strikes, s_idx, e_idx),
            })
        idx += stride
    return units


def _process_clip_for_catalogue(task: dict[str, Any]) -> dict[str, Any]:
    """Worker that validates one clip against provisional quality rules and extracts phrase units."""
    clip = task["clip"]
    root = Path(task["root"])
    stem = Path(clip["filename"]).stem
    mpath = root / "data" / "motion" / f"{stem}.json"
    apath = root / "data" / "audio" / f"{stem}_analysis.json"
    rpath = root / "data" / "audio" / f"{stem}_beats_review.json"

    if not (mpath.is_file() and apath.is_file()):
        return {"status": "missing_data", "stem": stem, "adavu": clip["adavu_canonical"], "units": []}

    motion = json.loads(mpath.read_text(encoding="utf-8"))
    analysis = json.loads(apath.read_text(encoding="utf-8"))

    beat_times = [float(b["time_s"]) for b in analysis["beats"]]
    beat_source = "unverified"
    if rpath.is_file():
        review = json.loads(rpath.read_text(encoding="utf-8"))
        if review.get("verified_beats_s"):
            beat_times = [float(t) for t in review["verified_beats_s"]]
            beat_source = "verified"

    torso_cv = float(motion.get("depth_travel_reliability", {}).get("torso_scale_cv", 0.0))
    if torso_cv > task["max_scale_cv"]:
        return {
            "status": "excluded_scale_drift",
            "stem": stem,
            "adavu": clip["adavu_canonical"],
            "torso_scale_cv": torso_cv,
            "units": [],
        }

    fps = float(motion["fps"])
    bpm = float(analysis.get("bpm_median") or 90.0)
    tala_beats = int(analysis.get("tala_beats") or 8)
    frames = motion["frames"]

    segments = extract_clean_segments(
        frames, fps=fps, min_segment_s=task["min_segment_s"], min_hand_ratio=task["min_hand_ratio"]
    )
    if not segments:
        return {
            "status": "excluded_no_clean_segments",
            "stem": stem,
            "adavu": clip["adavu_canonical"],
            "units": [],
        }

    adavu = clip["adavu_canonical"]
    confirmed = adavu not in UNCONFIRMED_CATEGORIES
    motion_rel = str(mpath.relative_to(root)).replace("\\", "/")
    all_strikes = motion.get("foot_strikes", [])
    units: list[dict[str, Any]] = []
    for seg in segments:
        units.extend(
            _slice_segment_into_units(
                seg=seg,
                frames=frames,
                all_strikes=all_strikes,
                beat_times_s=beat_times,
                beat_source=beat_source,
                fps=fps,
                bpm=bpm,
                tala_beats=tala_beats,
                adavu=adavu,
                confirmed=confirmed,
                stem=stem,
                motion_rel_path=motion_rel,
                target_phrase_beats=task["target_phrase_beats"],
            )
        )

    if not units:
        return {
            "status": "excluded_short_segments_lt_phrase_beats",
            "stem": stem,
            "adavu": adavu,
            "units": [],
        }

    return {
        "status": "included",
        "stem": stem,
        "adavu": adavu,
        "confirmed": confirmed,
        "clean_segments": len(segments),
        "clean_dur_s": round(sum(s["duration_s"] for s in segments), 2),
        "units": units,
    }


def build_catalogue(
    root: Path,
    out_path: Path | None = None,
    allowed_categories: set[str] | None = None,
    min_segment_s: float = 2.0,
    min_hand_ratio: float = 0.50,
    max_scale_cv: float = 0.12,
    target_phrase_beats: int = 8,
    workers: int | None = None,
) -> dict[str, Any]:
    """
    Build the provisional Bharatanatyam adavu phrase catalogue from extracted motion and audio files.
    """
    inv_path = root / "docs" / "inventory_report.json"
    report = json.loads(inv_path.read_text(encoding="utf-8"))

    tasks: list[dict[str, Any]] = []
    excluded_aliases: list[str] = []
    for clip in report["clips"]:
        if clip.get("broken"):
            continue
        if allowed_categories and clip["adavu_canonical"] not in allowed_categories:
            continue
        if _is_unconfirmed_alias_clip(clip):
            excluded_aliases.append(clip["filename"])
            continue
        stem = Path(clip["filename"]).stem
        if (root / "data" / "motion" / f"{stem}.json").is_file():
            tasks.append({
                "root": str(root),
                "clip": clip,
                "min_segment_s": min_segment_s,
                "min_hand_ratio": min_hand_ratio,
                "max_scale_cv": max_scale_cv,
                "target_phrase_beats": target_phrase_beats,
            })

    max_w = workers or min(8, max(2, (os.cpu_count() or 4) // 2))
    if max_w <= 1 or len(tasks) <= 2:
        results = [_process_clip_for_catalogue(t) for t in tasks]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as ex:
            results = list(ex.map(_process_clip_for_catalogue, tasks))

    all_units: list[dict[str, Any]] = []
    per_adavu: dict[str, dict[str, Any]] = {}
    excluded_clips: list[dict[str, Any]] = []

    for r in results:
        adavu = r["adavu"]
        bucket = per_adavu.setdefault(
            adavu,
            {
                "confirmed": adavu not in UNCONFIRMED_CATEGORIES,
                "clips_evaluated": 0,
                "clips_included": 0,
                "clean_segments": 0,
                "phrase_units": 0,
                "clean_dur_s": 0.0,
            },
        )
        bucket["clips_evaluated"] += 1
        if r["status"] == "included" and r["units"]:
            bucket["clips_included"] += 1
            bucket["clean_segments"] += r["clean_segments"]
            bucket["phrase_units"] += len(r["units"])
            bucket["clean_dur_s"] = round(bucket["clean_dur_s"] + r["clean_dur_s"], 2)
            all_units.extend(r["units"])
        else:
            excluded_clips.append({"stem": r["stem"], "adavu": adavu, "reason": r["status"]})

    catalogue = {
        "version": "1.1",
        "quality_rules_status": "PROVISIONAL (pending user approval on previews and beat verification)",
        "quality_rules": {
            "min_segment_s": min_segment_s,
            "min_hand_ratio": min_hand_ratio,
            "max_scale_cv": max_scale_cv,
            "target_phrase_beats": target_phrase_beats,
        },
        "unconfirmed_categories_note": {
            "ThaThaiThamAdavu": "Marked confirmed=false because ThaThaiTham1-6 carry unconfirmed KorvaiAdavu alias in adavu_name2.",
        },
        "adavu_categories": sorted(per_adavu.keys()),
        "confirmed_adavu_categories": sorted(k for k in per_adavu if k not in UNCONFIRMED_CATEGORIES),
        "total_units": len(all_units),
        "confirmed_units": sum(1 for u in all_units if u["confirmed"]),
        "unconfirmed_units": sum(1 for u in all_units if not u["confirmed"]),
        "per_adavu_summary": per_adavu,
        "excluded_unconfirmed_aliases": sorted(excluded_aliases),
        "excluded_clips": excluded_clips,
        "units": all_units,
    }

    target_out = out_path or (root / "data" / "catalogue" / "catalogue.json")
    target_out.parent.mkdir(parents=True, exist_ok=True)
    target_out.write_text(json.dumps(catalogue, indent=2), encoding="utf-8")
    return catalogue


def load_catalogue(catalogue_path: Path) -> dict[str, Any]:
    """Load an existing catalogue JSON from disk."""
    return json.loads(catalogue_path.read_text(encoding="utf-8"))
