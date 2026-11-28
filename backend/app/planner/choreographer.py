"""
choreographer.py – Step 4: Beat-synchronized Bharatanatyam choreography planner.

Maps clean adavu phrase units from the catalogue onto an analyzed song's Adi Tala
avartanam beat grid on a strictly constant-FPS timeline, optimizes tempo/pose/hand
continuity + adavu diversity across joins, blends boundaries without inventing
missing hand poses, and measures:
  - Frame quantization check: timeline events vs detected/verified beats (<= 25ms median, <= 60ms max)
  - Planned foot strikes vs song beats (against same-tempo chance, labeled verified vs unverified)
  - Separate transition jumps: body, hands (real detections only), and stance foot sliding
  - Optional bounded mudra prior from a trained landmark classifier (never an expert-choreography label)
  - Optional bounded recorded-join prior from trained boundary-flow features (join ordering only)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.audio.analyzer import compute_beat_error_ms
from app.motion.extractor import L_AN, L_FOOT, R_AN, R_FOOT, compare_foot_strikes_to_beats

# Coarse expected energy of each adavu family, used only as a small selection bias
# so vigorous passages favour jumps/footwork and quiet passages favour lyrical units.
ADAVU_ENERGY: dict[str, float] = {
    "DhalankuAdavu": 0.50,
    "KarthariAdavu": 0.70,
    "KattiAdavu": 0.60,
    "KudithuMettuAdavu": 0.90,
    "MandiAdavu": 0.65,
    "Nattadavu": 0.45,
    "ParavalAdavu": 0.70,
    "SarukkalAdavu": 0.55,
    "ThaThaiThaHaAdavu": 0.80,
    "ThaThaiThamAdavu": 0.75,
    "ThahathaJamTharithaAdavu": 0.70,
    "ThaiThaiThaThamAdavu": 0.70,
    "ThaiyaThaihiAdavu": 0.60,
    "Thattadavu": 0.55,
    "ThattimettuAdavu": 0.60,
    "TheerumanamAdavu": 0.35,
    "UthplavanaAdavu": 0.95,
    "UtsangaAdavu": 0.40,
}


def _energy_at(energy_curve: list[float] | None, fraction: float) -> float | None:
    """Sample a normalised 0..1 song-energy curve at a relative position."""
    if not energy_curve:
        return None
    idx = int(np.clip(round(fraction * (len(energy_curve) - 1)), 0, len(energy_curve) - 1))
    return float(energy_curve[idx])


def _pose_distance(sig_a: list[list[float]], sig_b: list[list[float]]) -> float:
    """Compute RMS 3D distance between two compact 12-joint pose signatures."""
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.5
    a = np.asarray(sig_a, dtype=float)
    b = np.asarray(sig_b, dtype=float)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def _hand_pair_distance(
    end_l: list[list[float]] | None,
    end_r: list[list[float]] | None,
    start_l: list[list[float]] | None,
    start_r: list[list[float]] | None,
) -> float:
    """Compute boundary hand-shape mismatch when genuine hand detections exist on both sides."""
    dists = []
    for a, b in ((end_l, start_l), (end_r, start_r)):
        if a and b and len(a) == len(b):
            aa = np.asarray(a, dtype=float)
            bb = np.asarray(b, dtype=float)
            dists.append(float(np.sqrt(np.mean(np.sum((aa - bb) ** 2, axis=-1)))))
    return float(np.mean(dists)) if dists else 0.15


def _select_unit_sequence(
    units: list[dict[str, Any]],
    song_beats: list[dict[str, Any]],
    song_bpm: float,
    allowed_adavus: set[str] | None = None,
    include_unconfirmed: bool = False,
    exclude_stems: set[str] | None = None,
    energy_curve: list[float] | None = None,
    song_duration_s: float = 0.0,
    seed: int = 42,
    mudra_annotations: dict[str, Any] | None = None,
    mudra_weight: float = 0.35,
    transition_prior: Any | None = None,
    transition_weight: float = 0.35,
) -> list[dict[str, Any]]:
    """Greedily schedule phrase units over song beat windows optimizing tempo, pose, hand continuity, and variety."""
    excluded_norm = {s.lower() for s in (exclude_stems or set()) if s}
    pool = [
        u
        for u in units
        if (include_unconfirmed or u.get("confirmed", True))
        and (not allowed_adavus or u["adavu"] in allowed_adavus)
        and u["source_stem"].lower() not in excluded_norm
        and len(u.get("beat_times_s", [])) >= 2
    ]
    if not pool or len(song_beats) < 2:
        return []

    rng = np.random.default_rng(seed)
    beat_times = [float(b["time_s"]) for b in song_beats]
    n_beats = len(beat_times)
    sam_idx = next((i for i, b in enumerate(song_beats) if b.get("is_sam")), 0)

    schedule: list[dict[str, Any]] = []
    cur = sam_idx if 0 < sam_idx < 4 else 0
    idx_queue: list[tuple[int, int]] = [(0, cur)] if cur > 0 else []

    recent_adavus: list[str] = []
    recent_stems: list[str] = []
    prev_unit: dict[str, Any] | None = None
    prev_end_pose: list[list[float]] = []
    prev_end_lh: list[list[float]] | None = None
    prev_end_rh: list[list[float]] | None = None
    prev_mudra_labels: set[str] = set()

    while idx_queue or cur < n_beats - 1:
        if idx_queue:
            w_start, w_end = idx_queue.pop(0)
        else:
            w_start = cur
            w_end = min(n_beats - 1, cur + 8)
            if w_end <= w_start:
                break
        n_intervals = w_end - w_start
        target_energy = _energy_at(
            energy_curve,
            beat_times[w_start] / max(song_duration_s, beat_times[-1], 1e-6),
        )
        best_score = float("inf")
        best_unit: dict[str, Any] | None = None
        best_cand_i = -1
        transition_scores: list[float | None] = [None] * len(pool)
        if (
            prev_unit is not None
            and transition_prior is not None
            and getattr(transition_prior, "available", False)
        ):
            transition_scores = transition_prior.score_many(prev_unit, pool)

        for cand_i, u in enumerate(pool):
            u_intervals = len(u["beat_times_s"]) - 1
            length_pen = 0.0 if u_intervals >= n_intervals else (n_intervals - u_intervals) * 1.2
            tempo_pen = abs(math.log2(max(song_bpm, 30.0) / max(float(u["bpm"]), 30.0))) * 1.5
            pose_pen = _pose_distance(prev_end_pose, u["start_pose"]) * 1.2 if prev_end_pose else 0.0
            hand_pen = (
                _hand_pair_distance(prev_end_lh, prev_end_rh, u.get("start_left_hand"), u.get("start_right_hand")) * 1.4
                if prev_end_pose
                else 0.0
            )
            hand_bonus = -0.35 * (float(u["left_hand_ratio"]) + float(u["right_hand_ratio"]))
            repeat_pen = 0.0
            if recent_adavus and u["adavu"] == recent_adavus[-1]:
                repeat_pen += 0.45
            if u["source_stem"] in recent_stems[-3:]:
                repeat_pen += 0.30
            energy_pen = (
                abs(ADAVU_ENERGY.get(u["adavu"], 0.5) - target_energy) * 0.9
                if target_energy is not None else 0.0
            )
            mudra_term = 0.0
            if mudra_annotations:
                ann = mudra_annotations.get(str(u["unit_id"]), {})
                labels = [str(x) for x in ann.get("labels", []) if x]
                if labels:
                    confidence = max(
                        float(ann.get(side, {}).get("confidence", 0.0)) for side in ("left", "right")
                    )
                    mudra_term -= mudra_weight * 0.25 * confidence
                    if prev_mudra_labels:
                        mudra_term += mudra_weight * 0.20 * len(prev_mudra_labels.intersection(labels))
            join_score = transition_scores[cand_i] if cand_i < len(transition_scores) else None
            transition_term = (
                transition_weight * (1.0 - float(join_score)) if join_score is not None else 0.0
            )
            jitter = float(rng.uniform(0.0, 0.03))
            score = (
                length_pen + tempo_pen + pose_pen + hand_pen + hand_bonus
                + repeat_pen + energy_pen + mudra_term + transition_term + jitter
            )
            if score < best_score:
                best_score = score
                best_unit = u
                best_cand_i = cand_i

        assert best_unit is not None
        used_intervals = min(n_intervals, len(best_unit["beat_times_s"]) - 1)
        target_beats = beat_times[w_start : w_start + used_intervals + 1]
        src_beats = best_unit["beat_times_s"][: used_intervals + 1]
        best_ann = (mudra_annotations or {}).get(str(best_unit["unit_id"]), {})
        best_mudra_labels = [str(x) for x in best_ann.get("labels", []) if x]
        best_mudra_confidence = max(
            (float(best_ann.get(side, {}).get("confidence", 0.0)) for side in ("left", "right")),
            default=0.0,
        )
        best_join_score = (
            transition_scores[best_cand_i]
            if 0 <= best_cand_i < len(transition_scores) else None
        )
        schedule.append({
            "unit": best_unit,
            "song_beat_start_idx": w_start,
            "song_beat_end_idx": w_start + used_intervals,
            "target_beats_s": target_beats,
            "src_beats_s": src_beats,
            "target_energy": round(target_energy, 4) if target_energy is not None else None,
            "mudra_labels": best_mudra_labels,
            "mudra_confidence": round(best_mudra_confidence, 4),
            "transition_score": round(float(best_join_score), 4) if best_join_score is not None else None,
        })
        if w_start >= cur:
            cur = w_start + used_intervals
        elif w_start + used_intervals < w_end:
            idx_queue.insert(0, (w_start + used_intervals, w_end))

        prev_unit = best_unit
        prev_end_pose = best_unit["end_pose"]
        prev_end_lh = best_unit.get("end_left_hand")
        prev_end_rh = best_unit.get("end_right_hand")
        recent_adavus.append(best_unit["adavu"])
        recent_stems.append(best_unit["source_stem"])
        prev_mudra_labels = set(best_mudra_labels)

    return schedule


def _sample_unit_at_times(
    unit_frames: list[dict[str, Any]],
    unit_t0: float,
    src_beats_s: list[float],
    target_beats_s: list[float],
    out_times: np.ndarray,
    travel_offset: float,
) -> list[dict[str, Any]]:
    """Sample a phrase unit on explicit constant-FPS timestamps without inventing missing hand poses."""
    if len(out_times) == 0:
        return []

    src_rel_times = np.array([float(fr["t"]) - unit_t0 for fr in unit_frames], dtype=float)
    sb = np.asarray(src_beats_s, dtype=float) - unit_t0
    tb = np.asarray(target_beats_s, dtype=float)
    mapped_src_rel = np.interp(out_times, tb, sb)

    src_pose = np.array([fr["pose_world"] for fr in unit_frames], dtype=float)
    src_tx = np.array([float(fr.get("travel_x") or 0.0) for fr in unit_frames], dtype=float)
    src_jy = np.array([float(fr.get("jump_y") or 0.0) for fr in unit_frames], dtype=float)
    src_fl = np.array([float(fr.get("foot_l_y") or 0.0) for fr in unit_frames], dtype=float)
    src_fr = np.array([float(fr.get("foot_r_y") or 0.0) for fr in unit_frames], dtype=float)

    n_src = len(unit_frames)
    src_f_pos = np.interp(mapped_src_rel, src_rel_times, np.arange(n_src, dtype=float))

    sampled: list[dict[str, Any]] = []
    for k, out_t in enumerate(out_times):
        pos = float(np.clip(src_f_pos[k], 0.0, n_src - 1.0))
        i0 = int(math.floor(pos))
        i1 = min(n_src - 1, i0 + 1)
        alpha = pos - i0

        pose = (1.0 - alpha) * src_pose[i0] + alpha * src_pose[i1]
        tx = float((1.0 - alpha) * src_tx[i0] + alpha * src_tx[i1]) + travel_offset
        jy = float((1.0 - alpha) * src_jy[i0] + alpha * src_jy[i1])
        fl = float((1.0 - alpha) * src_fl[i0] + alpha * src_fl[i1])
        fr_y = float((1.0 - alpha) * src_fr[i0] + alpha * src_fr[i1])

        lh0, lh1 = unit_frames[i0].get("left_hand_rel"), unit_frames[i1].get("left_hand_rel")
        rh0, rh1 = unit_frames[i0].get("right_hand_rel"), unit_frames[i1].get("right_hand_rel")

        if lh0 is not None and lh1 is not None:
            lh_out = np.round((1.0 - alpha) * np.asarray(lh0, dtype=float) + alpha * np.asarray(lh1, dtype=float), 4).tolist()
        else:
            lh_out = unit_frames[i0 if alpha < 0.5 else i1].get("left_hand_rel")

        if rh0 is not None and rh1 is not None:
            rh_out = np.round((1.0 - alpha) * np.asarray(rh0, dtype=float) + alpha * np.asarray(rh1, dtype=float), 4).tolist()
        else:
            rh_out = unit_frames[i0 if alpha < 0.5 else i1].get("right_hand_rel")

        sampled.append({
            "t": round(float(out_t), 4),
            "pose_world": np.round(pose, 4).tolist(),
            "travel_x": round(tx, 4),
            "jump_y": round(jy, 4),
            "foot_l_y": round(fl, 4),
            "foot_r_y": round(fr_y, 4),
            "left_hand_rel": lh_out,
            "right_hand_rel": rh_out,
        })
    return sampled


def _blend_boundary_symmetric(
    frames: list[dict[str, Any]],
    join_idx: int,
    half_win: int = 5,
) -> None:
    """
    Smoothly blend body pose, travel_x, and real hand detections across join_idx-1 -> join_idx.
    Uses a symmetric midpoint transition so neither tail nor head suffers a step jump,
    and NEVER invents missing hand poses (None remains None everywhere).
    """
    n = len(frames)
    left_len = min(half_win, join_idx)
    right_len = min(half_win, n - join_idx)
    if left_len < 2 or right_len < 2:
        return

    f_prev = frames[join_idx - 1]
    f_next = frames[join_idx]
    p_prev = np.asarray(f_prev["pose_world"], dtype=float)
    p_next = np.asarray(f_next["pose_world"], dtype=float)
    p_mid = 0.5 * (p_prev + p_next)

    tx_prev = float(f_prev["travel_x"])
    tx_next = float(f_next["travel_x"])
    tx_mid = 0.5 * (tx_prev + tx_next)

    # Blend left side (tail approaching join -> midpoint)
    for k in range(left_len):
        idx = join_idx - left_len + k
        w = 0.5 * (1.0 - math.cos(math.pi * (k + 1) / (left_len + 1))) * 0.5
        # Smooth ramp up to 0.5 at join_idx-1
        alpha = (k + 1) / (left_len + 1)
        ease = 0.5 * (1.0 - math.cos(math.pi * alpha)) * 0.5
        cur_p = np.asarray(frames[idx]["pose_world"], dtype=float)
        frames[idx]["pose_world"] = np.round((1.0 - ease * 2) * cur_p + (ease * 2) * p_mid, 4).tolist()
        frames[idx]["travel_x"] = round((1.0 - ease * 2) * float(frames[idx]["travel_x"]) + (ease * 2) * tx_mid, 4)
        _ = w

    # Blend right side (midpoint -> head departing join)
    for k in range(right_len):
        idx = join_idx + k
        alpha = (right_len - k) / (right_len + 1)
        ease = 0.5 * (1.0 - math.cos(math.pi * alpha))
        cur_p = np.asarray(frames[idx]["pose_world"], dtype=float)
        frames[idx]["pose_world"] = np.round((1.0 - ease) * cur_p + ease * p_mid, 4).tolist()
        frames[idx]["travel_x"] = round((1.0 - ease) * float(frames[idx]["travel_x"]) + ease * tx_mid, 4)

    # Symmetric hand crossfade ONLY over detected pairs (never invent hands!)
    for key in ("left_hand_rel", "right_hand_rel"):
        hp0 = f_prev.get(key)
        hn0 = f_next.get(key)
        if hp0 is None or hn0 is None:
            continue
        h_mid = 0.5 * (np.asarray(hp0, dtype=float) + np.asarray(hn0, dtype=float))
        for k in range(left_len):
            idx = join_idx - left_len + k
            cur_h = frames[idx].get(key)
            if cur_h is None:
                continue
            alpha = (k + 1) / (left_len + 1)
            ease = 0.5 * (1.0 - math.cos(math.pi * alpha))
            frames[idx][key] = np.round((1.0 - ease) * np.asarray(cur_h, dtype=float) + ease * h_mid, 4).tolist()
        for k in range(right_len):
            idx = join_idx + k
            cur_h = frames[idx].get(key)
            if cur_h is None:
                continue
            alpha = (right_len - k) / (right_len + 1)
            ease = 0.5 * (1.0 - math.cos(math.pi * alpha))
            frames[idx][key] = np.round((1.0 - ease) * np.asarray(cur_h, dtype=float) + ease * h_mid, 4).tolist()


def _compute_transition_metrics(
    frames: list[dict[str, Any]],
    boundary_indices: list[int],
) -> dict[str, Any]:
    """Compute separate transition jump metrics for body, real hands, and stance foot sliding."""
    body_jumps: list[float] = []
    hand_jumps: list[float] = []
    foot_slides: list[float] = []

    for idx in boundary_indices:
        if idx <= 0 or idx >= len(frames):
            continue
        f0, f1 = frames[idx - 1], frames[idx]
        p0 = np.asarray(f0["pose_world"], dtype=float)
        p1 = np.asarray(f1["pose_world"], dtype=float)
        body_jumps.append(float(np.sqrt(np.mean(np.sum((p1 - p0) ** 2, axis=-1)))))

        # Foot sliding across the stance/lowest grounded foot (X+Z plane including travel_x)
        fl0, fr0 = float(f0.get("foot_l_y", 0.0)), float(f0.get("foot_r_y", 0.0))
        fl1, fr1 = float(f1.get("foot_l_y", 0.0)), float(f1.get("foot_r_y", 0.0))
        slides = []
        if fl0 <= 0.12 and fl1 <= 0.12:
            dx_a = (p1[L_AN, 0] + float(f1["travel_x"])) - (p0[L_AN, 0] + float(f0["travel_x"]))
            dz_a = p1[L_AN, 2] - p0[L_AN, 2]
            dx_f = (p1[L_FOOT, 0] + float(f1["travel_x"])) - (p0[L_FOOT, 0] + float(f0["travel_x"]))
            dz_f = p1[L_FOOT, 2] - p0[L_FOOT, 2]
            slides.append(0.5 * (math.hypot(dx_a, dz_a) + math.hypot(dx_f, dz_f)))
        if fr0 <= 0.12 and fr1 <= 0.12:
            dx_a = (p1[R_AN, 0] + float(f1["travel_x"])) - (p0[R_AN, 0] + float(f0["travel_x"]))
            dz_a = p1[R_AN, 2] - p0[R_AN, 2]
            dx_f = (p1[R_FOOT, 0] + float(f1["travel_x"])) - (p0[R_FOOT, 0] + float(f0["travel_x"]))
            dz_f = p1[R_FOOT, 2] - p0[R_FOOT, 2]
            slides.append(0.5 * (math.hypot(dx_a, dz_a) + math.hypot(dx_f, dz_f)))
        if not slides:
            # Fallback to lower foot across join when both slightly above baseline threshold
            if min(fl0, fl1) <= min(fr0, fr1):
                dx = (p1[L_AN, 0] + float(f1["travel_x"])) - (p0[L_AN, 0] + float(f0["travel_x"]))
                dz = p1[L_AN, 2] - p0[L_AN, 2]
            else:
                dx = (p1[R_AN, 0] + float(f1["travel_x"])) - (p0[R_AN, 0] + float(f0["travel_x"]))
                dz = p1[R_AN, 2] - p0[R_AN, 2]
            slides.append(math.hypot(dx, dz))
        foot_slides.append(float(max(slides)))

        for key in ("left_hand_rel", "right_hand_rel"):
            h0, h1 = f0.get(key), f1.get(key)
            if h0 is not None and h1 is not None:
                ha0 = np.asarray(h0, dtype=float)
                ha1 = np.asarray(h1, dtype=float)
                hand_jumps.append(float(np.sqrt(np.mean(np.sum((ha1 - ha0) ** 2, axis=-1)))))

    def _stats(vals: list[float]) -> dict[str, float | int | None]:
        if not vals:
            return {"count": 0, "median_torso": None, "max_torso": None}
        return {
            "count": len(vals),
            "median_torso": round(float(np.median(vals)), 4),
            "max_torso": round(float(np.max(vals)), 4),
        }

    return {
        "transition_count": len(boundary_indices),
        "body": _stats(body_jumps),
        "hands": _stats(hand_jumps),
        "foot_sliding": _stats(foot_slides),
    }


def _map_planned_foot_strikes(
    schedule: list[dict[str, Any]],
    motion_cache: dict[str, dict[str, Any]],
    song_beats_s: list[float],
    beat_source: str,
    bpm: float,
    fps: float,
) -> dict[str, Any]:
    """Map planned source foot strikes into the generated timeline and compare to song beats at the same tempo."""
    mapped_strikes: list[dict[str, Any]] = []
    seen_times: list[float] = []

    for item in schedule:
        u = item["unit"]
        # Read strikes from unit metadata or freshly loaded motion JSON
        unit_strikes = u.get("foot_strikes")
        if unit_strikes is None:
            m_data = motion_cache.get(u["motion_path"], {})
            unit_strikes = [
                st for st in m_data.get("foot_strikes", [])
                if u["start_frame"] <= int(st["frame"]) <= u["end_frame"]
            ]
        sb = np.asarray(item["src_beats_s"], dtype=float)
        tb = np.asarray(item["target_beats_s"], dtype=float)
        for st in unit_strikes:
            st_t = float(st["time_s"])
            if st_t < sb[0] - 0.05 or st_t > sb[-1] + 0.05:
                continue
            out_t = float(np.interp(st_t, sb, tb))
            if any(abs(out_t - prev_t) < 0.12 for prev_t in seen_times):
                continue
            seen_times.append(out_t)
            mapped_strikes.append({
                "frame": int(round((out_t - schedule[0]["target_beats_s"][0]) * fps)),
                "time_s": round(out_t, 4),
                "side": st["side"],
                "adavu": u["adavu"],
            })

    comp = compare_foot_strikes_to_beats(mapped_strikes, song_beats_s, fps=fps)
    period_ms = round(60000.0 / bpm, 2) if bpm > 0 else None
    rand_ms = round(period_ms * 0.25, 2) if period_ms else None
    frac_period = round(comp["median_ms"] / period_ms, 3) if (comp["median_ms"] is not None and period_ms) else None
    return {
        **comp,
        "beat_source": beat_source,
        "beat_period_ms": period_ms,
        "err_frac_of_period": frac_period,
        "same_tempo_random_baseline_ms": rand_ms,
    }


def plan_choreography(
    audio_analysis: dict[str, Any],
    catalogue: dict[str, Any],
    root: Path,
    fps: float = 30.0,
    allowed_adavus: set[str] | None = None,
    include_unconfirmed: bool = False,
    exclude_stems: set[str] | None = None,
    blend_frames: int = 10,
    seed: int = 42,
    mudra_annotations: dict[str, Any] | None = None,
    mudra_weight: float = 0.35,
    transition_prior: Any | None = None,
    transition_weight: float = 0.35,
) -> dict[str, Any]:
    """
    Generate a constant-FPS, beat-synchronized choreography timeline for an analyzed song.
    """
    song_beats = audio_analysis.get("beats", [])
    verified_beats_list = audio_analysis.get("verified_beats_s")
    if verified_beats_list:
        eval_beat_times = [float(t) for t in verified_beats_list]
        beat_source = "verified"
    else:
        eval_beat_times = [float(b["time_s"]) for b in song_beats]
        beat_source = "unverified"

    song_bpm = float(audio_analysis.get("bpm_median") or 90.0)
    units = catalogue.get("units", [])

    schedule = _select_unit_sequence(
        units=units,
        song_beats=song_beats,
        song_bpm=song_bpm,
        allowed_adavus=allowed_adavus,
        include_unconfirmed=include_unconfirmed,
        exclude_stems=exclude_stems,
        energy_curve=audio_analysis.get("energy_curve"),
        song_duration_s=float(audio_analysis.get("duration_s") or 0.0),
        seed=seed,
        mudra_annotations=mudra_annotations,
        mudra_weight=mudra_weight,
        transition_prior=transition_prior,
        transition_weight=transition_weight,
    )
    if not schedule:
        raise RuntimeError("No matching phrase units available in catalogue for the song beat grid.")

    # Single uniform constant-FPS grid spanning [first scheduled beat, last scheduled beat]
    t_grid_start = float(schedule[0]["target_beats_s"][0])
    t_grid_end = float(schedule[-1]["target_beats_s"][-1])
    n_grid = max(2, int(round((t_grid_end - t_grid_start) * fps)) + 1)
    grid_times = t_grid_start + np.arange(n_grid, dtype=float) / fps

    motion_cache: dict[str, dict[str, Any]] = {}
    all_frames: list[dict[str, Any]] = []
    boundary_indices: list[int] = []
    timeline_events: list[dict[str, Any]] = []
    prev_travel_end = 0.0
    cursor_idx = 0

    for item_i, item in enumerate(schedule):
        unit = item["unit"]
        mpath = root / unit["motion_path"]
        if unit["motion_path"] not in motion_cache:
            motion_cache[unit["motion_path"]] = json.loads(mpath.read_text(encoding="utf-8"))
        full_frames = motion_cache[unit["motion_path"]]["frames"]
        u_frames = full_frames[unit["start_frame"] : unit["end_frame"] + 1]
        u_t0 = float(u_frames[0]["t"])

        win_end_t = float(item["target_beats_s"][-1])
        if item_i == len(schedule) - 1:
            end_idx = n_grid
        else:
            end_idx = min(n_grid, int(np.searchsorted(grid_times, win_end_t, side="left")))
        if end_idx <= cursor_idx:
            continue

        slice_times = grid_times[cursor_idx:end_idx]
        next_start_tx = float(u_frames[0].get("travel_x") or 0.0)
        offset = max(-1.5, min(1.5, prev_travel_end)) - next_start_tx

        sampled = _sample_unit_at_times(
            unit_frames=u_frames,
            unit_t0=u_t0,
            src_beats_s=item["src_beats_s"],
            target_beats_s=item["target_beats_s"],
            out_times=slice_times,
            travel_offset=offset,
        )
        for fr in sampled:
            fr["adavu"] = unit["adavu"]
            fr["source_stem"] = unit["source_stem"]
            fr["confirmed"] = unit.get("confirmed", True)

        if all_frames and sampled:
            boundary_indices.append(len(all_frames))
        all_frames.extend(sampled)
        prev_travel_end = float(all_frames[-1]["travel_x"]) if all_frames else 0.0
        cursor_idx = end_idx

        for b_idx, bt in enumerate(item["target_beats_s"][:-1]):
            timeline_events.append({
                "time_s": round(float(bt), 4),
                "unit_id": unit["unit_id"],
                "adavu": unit["adavu"],
                "confirmed": unit.get("confirmed", True),
                "source_stem": unit["source_stem"],
                "is_phrase_start": (b_idx == 0),
                "song_beat_index": item["song_beat_start_idx"] + b_idx,
                "mudra_labels": item["mudra_labels"],
                "mudra_confidence": item["mudra_confidence"],
                "transition_score": item.get("transition_score"),
            })

    # Apply symmetric boundary smoothing after full constant-FPS frame assembly
    half_win = max(2, blend_frames // 2)
    for b_idx in boundary_indices:
        _blend_boundary_symmetric(all_frames, b_idx, half_win=half_win)

    for f_i, fr in enumerate(all_frames):
        fr["f"] = f_i

    quantized_event_times = [round(round((ev["time_s"] - t_grid_start) * fps) / fps + t_grid_start, 4) for ev in timeline_events]
    frame_quantization = compute_beat_error_ms(quantized_event_times, [float(b["time_s"]) for b in song_beats])
    transition_metrics = _compute_transition_metrics(all_frames, boundary_indices)
    planned_strikes = _map_planned_foot_strikes(
        schedule=schedule,
        motion_cache=motion_cache,
        song_beats_s=eval_beat_times,
        beat_source=beat_source,
        bpm=song_bpm,
        fps=fps,
    )

    duration_s = round((len(all_frames) - 1) / fps, 3) if len(all_frames) >= 2 else 0.0
    transition_scores = [
        float(s["transition_score"]) for s in schedule if s.get("transition_score") is not None
    ]
    transition_enabled = bool(
        transition_prior is not None and getattr(transition_prior, "available", False)
    )
    return {
        "song_bpm": round(song_bpm, 2),
        "tala": audio_analysis.get("tala", "Adi"),
        "separation_backend": audio_analysis.get("separation_backend"),
        "beat_source": beat_source,
        "affect": audio_analysis.get("affect"),
        "affect_timeline": [
            {
                "time_s": round(float(s["target_beats_s"][0]), 4),
                "target_energy": s["target_energy"],
                "adavu": s["unit"]["adavu"],
            }
            for s in schedule
        ],
        "include_unconfirmed": include_unconfirmed,
        "excluded_stems": sorted(exclude_stems or set()),
        "fps": fps,
        "duration_s": duration_s,
        "n_frames": len(all_frames),
        "phrase_count": len(schedule),
        "adavus_used": sorted({s["unit"]["adavu"] for s in schedule}),
        "source_stems_used": sorted({s["unit"]["source_stem"] for s in schedule}),
        "frame_quantization_check_ms": frame_quantization,
        "planned_foot_strikes_vs_beats": planned_strikes,
        "transition_metrics": transition_metrics,
        "mudra_planning": {
            "enabled": bool(mudra_annotations),
            "weight": round(float(mudra_weight), 4),
            "annotated_units": len(mudra_annotations or {}),
            "selected_labels": sorted({label for s in schedule for label in s["mudra_labels"]}),
            "basis": "trained-hand-shape-prior" if mudra_annotations else "disabled",
        },
        "transition_planning": {
            "enabled": transition_enabled,
            "weight": round(float(transition_weight), 4),
            "model": getattr(transition_prior, "version", None) if transition_enabled else None,
            "scored_joins": len(transition_scores),
            "mean_score": round(float(np.mean(transition_scores)), 4) if transition_scores else None,
            "basis": "trained-recorded-join-prior" if transition_enabled else "disabled",
            "scope": "bounded join ordering from recorded motion; not expert choreography",
        },
        "timeline_events": timeline_events,
        "frames": all_frames,
    }
