"""
extractor.py – Extract body pose and pose-guided hand landmarks from adavu videos.

Key invariants:
  - Facing normalization uses ONE constant yaw rotation per clip derived from
    opening valid frames (preserves turns, bends, jumps, and travel).
  - Body size is normalized by median torso length (shoulders-to-hips).
  - Interpolation gap limit is <= 100 ms converted to frames via floor(0.100 * fps).
    Longer gaps are flagged/excluded and NEVER interpolated.
  - Hand poses (21 landmarks per hand) are NEVER interpolated or invented.
  - Left/right hands are verified against pose wrists to prevent L/R swaps.
  - Reports explicit MediaPipe travel/depth reliability diagnostics and
    foot-strike movement events vs beats (ms and frames).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.signal import savgol_filter

# MediaPipe Pose landmark indices
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_PK, R_PK = 17, 18
L_IX, R_IX = 19, 20
L_HP, R_HP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

KEY_BODY_INDICES = (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HP, R_HP, L_KN, R_KN, L_AN, R_AN)
MIN_POSE_VIS = 0.5
MAX_INTERP_GAP_MS = 100.0
MAX_HAND_WRIST_DIST_TORSO = 0.35

DEFAULT_POSE_MODEL = Path(__file__).resolve().parents[3] / "data" / "models" / "mediapipe" / "pose_landmarker_full.task"
DEFAULT_HAND_MODEL = Path(__file__).resolve().parents[3] / "data" / "models" / "mediapipe" / "hand_landmarker.task"


def max_gap_frames_for_fps(fps: float, max_gap_ms: float = MAX_INTERP_GAP_MS) -> int:
    """Convert <= 100 ms gap ceiling into integer frame count using clip fps."""
    if fps <= 0:
        return 0
    return max(0, int(math.floor((max_gap_ms / 1000.0) * fps)))


def find_gaps_and_interpolate(
    values: np.ndarray,
    valid_mask: np.ndarray,
    max_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Interpolate ONLY short gaps (length <= max_gap_frames) bounded by valid frames.
    Longer gaps and leading/trailing invalid runs are NEVER interpolated (remain NaN).

    Args:
      values: shape (N, ...) float array
      valid_mask: shape (N,) bool array
      max_gap_frames: max gap length allowed for interpolation

    Returns:
      (out_values, out_valid_mask, flagged_long_gaps)
    """
    n = len(valid_mask)
    out = np.array(values, dtype=float, copy=True)
    out_valid = np.array(valid_mask, dtype=bool, copy=True)
    flagged: list[dict[str, Any]] = []

    i = 0
    while i < n:
        if out_valid[i]:
            i += 1
            continue
        start = i
        while i < n and not out_valid[i]:
            i += 1
        end = i - 1  # inclusive
        gap_len = end - start + 1
        has_left = start > 0 and out_valid[start - 1]
        has_right = i < n and out_valid[i]

        if has_left and has_right and gap_len <= max_gap_frames:
            v0 = out[start - 1]
            v1 = out[i]
            for k in range(1, gap_len + 1):
                alpha = k / (gap_len + 1)
                out[start + k - 1] = (1.0 - alpha) * v0 + alpha * v1
                out_valid[start + k - 1] = True
        else:
            out[start : end + 1] = np.nan
            reason = "edge_dropout" if not (has_left and has_right) else f"long_gap_gt_{max_gap_frames}_frames"
            flagged.append({
                "start_frame": int(start),
                "end_frame": int(end),
                "length_frames": int(gap_len),
                "reason": reason,
            })

    return out, out_valid, flagged


def smooth_contiguous_runs(
    values: np.ndarray,
    valid_mask: np.ndarray,
    window: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing strictly within each contiguous valid run."""
    out = np.array(values, dtype=float, copy=True)
    if window < 3 or window % 2 == 0:
        return out
    n = len(valid_mask)
    flat = out.reshape(n, -1)
    i = 0
    while i < n:
        if not valid_mask[i]:
            i += 1
            continue
        start = i
        while i < n and valid_mask[i]:
            i += 1
        run_len = i - start
        if run_len >= window:
            flat[start:i] = savgol_filter(flat[start:i], window_length=window, polyorder=polyorder, axis=0)
    return out


def compute_opening_yaw_rad(
    world_seq: np.ndarray,
    valid_mask: np.ndarray,
    opening_frames: int = 30,
) -> float:
    """
    Compute ONE yaw angle (radians) from the opening valid frames so the
    dancer's left-to-right shoulder/hip axis aligns with world +X (facing -Z).
    """
    valid_indices = np.where(valid_mask)[0][:opening_frames]
    if len(valid_indices) == 0:
        return 0.0

    vecs = []
    for idx in valid_indices:
        w = world_seq[idx]
        # Vector from anatomical right to anatomical left in XZ plane
        v_sh = w[L_SH] - w[R_SH]
        v_hp = w[L_HP] - w[R_HP]
        v = 0.65 * v_sh + 0.35 * v_hp
        vecs.append([v[0], v[2]])

    med_vec = np.median(np.asarray(vecs, dtype=float), axis=0)
    if np.linalg.norm(med_vec) < 1e-6:
        return 0.0
    # atan2(dz, dx): angle to rotate around Y so med_vec aligns with +X
    return float(math.atan2(med_vec[1], med_vec[0]))


def rotate_y(points: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Rotate (..., 3) XYZ points around Y axis by -yaw_rad so opening +yaw aligns to +X."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    out = np.array(points, dtype=float, copy=True)
    x = points[..., 0]
    z = points[..., 2]
    out[..., 0] = c * x + s * z
    out[..., 2] = -s * x + c * z
    return out


def verify_and_assign_hand(
    hand_img_pts: np.ndarray,
    expected_side: str,
    pose_img_pts: np.ndarray,
    torso_img_px: float,
    img_w: int,
    img_h: int,
) -> tuple[bool, str]:
    """
    Verify that a hand detected in a wrist crop actually belongs to the expected
    anatomical wrist (Left=15, Right=16) and is not swapped with the opposite wrist.
    """
    wrist_hand_px = hand_img_pts[0, :2] * np.array([img_w, img_h], dtype=float)
    pose_l_px = pose_img_pts[L_WR, :2] * np.array([img_w, img_h], dtype=float)
    pose_r_px = pose_img_pts[R_WR, :2] * np.array([img_w, img_h], dtype=float)

    d_l = float(np.linalg.norm(wrist_hand_px - pose_l_px) / max(torso_img_px, 1.0))
    d_r = float(np.linalg.norm(wrist_hand_px - pose_r_px) / max(torso_img_px, 1.0))

    if expected_side == "left":
        if d_l > MAX_HAND_WRIST_DIST_TORSO:
            return False, f"too_far_from_left_wrist ({d_l:.2f})"
        if d_l > d_r + 0.04:
            return False, f"closer_to_right_wrist (dL={d_l:.2f}, dR={d_r:.2f})"
        return True, "ok"
    else:
        if d_r > MAX_HAND_WRIST_DIST_TORSO:
            return False, f"too_far_from_right_wrist ({d_r:.2f})"
        if d_r > d_l + 0.04:
            return False, f"closer_to_left_wrist (dR={d_r:.2f}, dL={d_l:.2f})"
        return True, "ok"


def compute_rolling_floor_baseline(
    both_low_y: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
    window_s: float = 5.0,
    percentile: float = 92.0,
) -> np.ndarray:
    """
    Compute a time-varying rolling floor Y baseline (in pixels) over valid frames so
    forward/backward stage distance shifts do not strand foot height above floor contact.
    """
    n = len(both_low_y)
    baseline = np.full(n, np.nan, dtype=float)
    valid_finite = valid_mask & np.isfinite(both_low_y)
    if not np.any(valid_finite):
        return baseline

    global_fallback = float(np.nanpercentile(both_low_y[valid_finite], percentile))
    half_win = max(10, int(round(0.5 * window_s * max(fps, 1.0))))
    for i in range(n):
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        win_mask = valid_finite[lo:hi]
        if np.count_nonzero(win_mask) >= 5:
            baseline[i] = float(np.nanpercentile(both_low_y[lo:hi][win_mask], percentile))
        else:
            baseline[i] = global_fallback

    # Smooth the rolling baseline lightly with mirror-safe Savitzky-Golay (mode="nearest" avoids edge tilt)
    smooth_win = min(n if n % 2 == 1 else max(1, n - 1), max(5, int(round(1.5 * max(fps, 1.0))) | 1))
    if smooth_win >= 5 and smooth_win <= n:
        baseline = savgol_filter(baseline, window_length=smooth_win, polyorder=2, mode="nearest")
    return baseline


def detect_foot_strikes(
    foot_l_y: np.ndarray,
    foot_r_y: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
    min_lift_torso: float = 0.085,
    min_prominence_torso: float = 0.080,
    contact_thresh_torso: float = 0.055,
    min_drop_step_torso: float = 0.025,
    min_gap_s: float = 0.22,
    use_trough_timing: bool = True,
    use_crossed_leg_rejection: bool = True,
    use_adaptive_amplitude: bool = False,
    ankle_x_l_torso: np.ndarray | None = None,
    ankle_x_r_torso: np.ndarray | None = None,
    ankle_vis_l: np.ndarray | None = None,
    ankle_vis_r: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    Detect candidate foot-strike timestamps from vertical foot height curves
    (0 = rolling floor baseline, >0 = lifted).

    Optional boolean switches (all preserve baseline behavior when False):
      - use_trough_timing: snap from first threshold-entry minimum to true deepest contact trough within ~140ms.
      - use_crossed_leg_rejection: reject occluded crossed-leg false strikes when ankles overlap horizontally.
      - use_adaptive_amplitude: allow smaller real heel-drop amplitudes (0.058 torso) when landing settles cleanly.
    """
    n = len(valid_mask)
    if n < 5 or fps <= 0:
        return []

    eff_min_lift = 0.068 if use_adaptive_amplitude else min_lift_torso
    eff_min_prom = 0.065 if use_adaptive_amplitude else min_prominence_torso
    eff_min_drop = max(0.028, min_drop_step_torso) if use_adaptive_amplitude else min_drop_step_torso

    lookback = max(4, int(round(0.45 * fps)))
    vel_win = max(2, int(round(0.10 * fps)))
    trough_ahead = max(2, int(round(0.14 * fps)))
    min_sep_frames = max(2, int(round(min_gap_s * fps)))
    strikes: list[dict[str, Any]] = []

    for side, curve, other_curve, vis_arr in (
        ("left", foot_l_y, foot_r_y, ankle_vis_l),
        ("right", foot_r_y, foot_l_y, ankle_vis_r),
    ):
        last_strike = -min_sep_frames
        i = 2
        while i < n - 1:
            if not (valid_mask[i - 1] and valid_mask[i] and valid_mask[i + 1]):
                i += 1
                continue
            if i - last_strike < min_sep_frames:
                i += 1
                continue
            cur = curve[i]
            if np.isnan(cur) or cur > contact_thresh_torso:
                i += 1
                continue
            if not (cur <= curve[i - 1] and cur <= curve[i + 1] and curve[i - 1] > cur):
                i += 1
                continue

            commit_idx = i
            if use_trough_timing:
                hi = min(n - 1, i + trough_ahead + 1)
                best_idx = i
                best_val = float(cur)
                for j in range(i + 1, hi):
                    if not valid_mask[j] or np.isnan(curve[j]):
                        break
                    if curve[j] > contact_thresh_torso:
                        break
                    if curve[j] < best_val:
                        best_val = float(curve[j])
                        best_idx = j
                commit_idx = best_idx

            cur_commit = float(curve[commit_idx])
            if use_adaptive_amplitude:
                if cur_commit > 0.045:
                    i += 1
                    continue
                settle_end = min(n, commit_idx + 3)
                settle_vals = curve[commit_idx:settle_end]
                settle_valid = settle_vals[~np.isnan(settle_vals)]
                if len(settle_valid) == 0 or float(np.mean(settle_valid)) > contact_thresh_torso:
                    i += 1
                    continue

            if use_crossed_leg_rejection and ankle_x_l_torso is not None and ankle_x_r_torso is not None:
                xl = ankle_x_l_torso[commit_idx]
                xr = ankle_x_r_torso[commit_idx]
                other_h = other_curve[commit_idx]
                low_vis = vis_arr is not None and np.isfinite(vis_arr[commit_idx]) and float(vis_arr[commit_idx]) < 0.65
                if np.isfinite(xl) and np.isfinite(xr) and abs(float(xl) - float(xr)) < 0.14:
                    # Crossed-leg occlusion: reject when visibility is low or opposite foot is clearly lower on floor
                    if low_vis or (np.isfinite(other_h) and cur_commit > float(other_h) + 0.025):
                        i += 1
                        continue

            prev_win = curve[max(0, commit_idx - lookback) : commit_idx]
            prev_valid = prev_win[~np.isnan(prev_win)]
            if len(prev_valid) == 0:
                i += 1
                continue
            peak_h = float(np.max(prev_valid))
            if peak_h < eff_min_lift or (peak_h - cur_commit) < eff_min_prom:
                i += 1
                continue

            drop_win = curve[max(0, i - vel_win) : i]
            drop_valid = drop_win[~np.isnan(drop_win)]
            if len(drop_valid) == 0 or (float(np.max(drop_valid)) - float(cur)) < eff_min_drop:
                i += 1
                continue

            strikes.append({
                "frame": int(commit_idx),
                "time_s": round(float(commit_idx / fps), 4),
                "side": side,
                "height_torso": round(cur_commit, 4),
                "lift_prominence": round(float(peak_h - cur_commit), 4),
            })
            last_strike = commit_idx
            i = commit_idx + 1

    strikes.sort(key=lambda s: s["frame"])
    return strikes


def compare_foot_strikes_to_beats(
    strikes: list[dict[str, Any]],
    beat_times_s: list[float],
    fps: float,
) -> dict[str, Any]:
    """Measure offset between detected foot-strike movement events and beats in ms and frames (Measure 4c)."""
    if not strikes or not beat_times_s:
        return {
            "strike_count": len(strikes),
            "beat_count": len(beat_times_s),
            "median_ms": None,
            "mean_ms": None,
            "max_ms": None,
            "median_frames": None,
            "max_frames": None,
        }
    bt = np.asarray(beat_times_s, dtype=float)
    diffs_ms = np.array([float(np.min(np.abs(bt - s["time_s"])) * 1000.0) for s in strikes], dtype=float)
    diffs_frames = diffs_ms / 1000.0 * fps
    return {
        "strike_count": len(strikes),
        "beat_count": len(beat_times_s),
        "median_ms": round(float(np.median(diffs_ms)), 2),
        "mean_ms": round(float(np.mean(diffs_ms)), 2),
        "max_ms": round(float(np.max(diffs_ms)), 2),
        "median_frames": round(float(np.median(diffs_frames)), 2),
        "max_frames": round(float(np.max(diffs_frames)), 2),
    }


def extract_clip_motion(
    video_path: Path,
    beat_times_s: list[float] | None = None,
    pose_model: Path = DEFAULT_POSE_MODEL,
    hand_model: Path = DEFAULT_HAND_MODEL,
    max_seconds: float = 0.0,
) -> dict[str, Any]:
    """
    Extract, normalize, and quality-check body & hand motion from a single adavu video.
    """
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_gap_frames = max_gap_frames_for_fps(fps, MAX_INTERP_GAP_MS)

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )

    raw_img_lms: list[np.ndarray | None] = []
    raw_world_lms: list[np.ndarray | None] = []
    raw_valid: list[bool] = []
    raw_left_hands: list[np.ndarray | None] = []
    raw_right_hands: list[np.ndarray | None] = []
    swap_rejections = {"left": 0, "right": 0}
    overlapping_wrist_frames = 0
    torso_px_samples: list[float] = []

    frame_idx = 0
    last_ts_ms = -1

    with (
        mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm,
        mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm,
    ):
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t_s = frame_idx / fps
            if max_seconds > 0 and t_s > max_seconds:
                break

            ts_ms = max(last_ts_ms + 1, int(round(t_s * 1000.0)))
            last_ts_ms = ts_ms

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if width > 640:
                pose_h = max(2, int(round(height * (640.0 / width))))
                pose_rgb = cv2.resize(rgb, (640, pose_h), interpolation=cv2.INTER_AREA)
            else:
                pose_rgb = rgb
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=pose_rgb)
            pres = pose_lm.detect_for_video(mp_img, ts_ms)

            if pres.pose_landmarks and pres.pose_world_landmarks:
                plm = pres.pose_landmarks[0]
                wlm = pres.pose_world_landmarks[0]
                img_arr = np.array([[l.x, l.y, l.z, l.visibility or 0.0] for l in plm], dtype=float)
                wld_arr = np.array([[l.x, l.y, l.z] for l in wlm], dtype=float)

                # Core torso + limbs visibility check
                core_ok = all(img_arr[j, 3] >= MIN_POSE_VIS for j in (L_SH, R_SH, L_HP, R_HP, L_KN, R_KN, L_AN, R_AN))
                raw_img_lms.append(img_arr)
                raw_world_lms.append(wld_arr)
                raw_valid.append(bool(core_ok))

                sh_mid_px = (img_arr[L_SH, :2] + img_arr[R_SH, :2]) * 0.5 * np.array([width, height])
                hp_mid_px = (img_arr[L_HP, :2] + img_arr[R_HP, :2]) * 0.5 * np.array([width, height])
                torso_px = max(float(np.linalg.norm(hp_mid_px - sh_mid_px)), 40.0)
                if core_ok:
                    torso_px_samples.append(torso_px)

                # Check wrist overlap (e.g., Anjali / chest center)
                lw_px = img_arr[L_WR, :2] * np.array([width, height])
                rw_px = img_arr[R_WR, :2] * np.array([width, height])
                if float(np.linalg.norm(lw_px - rw_px) / torso_px) < 0.18:
                    overlapping_wrist_frames += 1

                # Pose-guided wrist crops for left and right hands (NEVER invent missing hands)
                crop_size = int(max(140, torso_px * 0.65))
                lh_out = None
                rh_out = None
                for side, wi, ii in (("left", L_WR, L_IX), ("right", R_WR, R_IX)):
                    if img_arr[wi, 3] < 0.4:
                        continue
                    cx = int(((img_arr[wi, 0] + img_arr[ii, 0]) * 0.5) * width)
                    cy = int(((img_arr[wi, 1] + img_arr[ii, 1]) * 0.5) * height)
                    x0, y0 = max(0, cx - crop_size // 2), max(0, cy - crop_size // 2)
                    x1, y1 = min(width, cx + crop_size // 2), min(height, cy + crop_size // 2)
                    if x1 - x0 < 40 or y1 - y0 < 40:
                        continue
                    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
                    hres = hand_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
                    if hres.hand_landmarks:
                        hl = hres.hand_landmarks[0]
                        # Map back to full-frame normalized [0, 1] coords
                        h_full = np.array([
                            [(pt.x * (x1 - x0) + x0) / width, (pt.y * (y1 - y0) + y0) / height, pt.z]
                            for pt in hl
                        ], dtype=float)
                        ok_side, _ = verify_and_assign_hand(h_full, side, img_arr, torso_px, width, height)
                        if ok_side:
                            if side == "left":
                                lh_out = h_full
                            else:
                                rh_out = h_full
                        else:
                            swap_rejections[side] += 1

                raw_left_hands.append(lh_out)
                raw_right_hands.append(rh_out)
            else:
                raw_img_lms.append(None)
                raw_world_lms.append(None)
                raw_valid.append(False)
                raw_left_hands.append(None)
                raw_right_hands.append(None)

            frame_idx += 1

    cap.release()
    n_frames = len(raw_valid)
    if n_frames == 0:
        raise RuntimeError(f"No frames read from {video_path}")

    valid_arr = np.asarray(raw_valid, dtype=bool)
    img_seq = np.full((n_frames, 33, 4), np.nan, dtype=float)
    world_seq = np.full((n_frames, 33, 3), np.nan, dtype=float)
    for i in range(n_frames):
        if raw_img_lms[i] is not None:
            img_seq[i] = raw_img_lms[i]
            world_seq[i] = raw_world_lms[i]

    # Interpolate ONLY short pose gaps <= max_gap_frames (<= 100 ms); flag longer gaps
    world_interp, valid_after_interp, flagged_long_gaps = find_gaps_and_interpolate(
        world_seq, valid_arr, max_gap_frames=max_gap_frames
    )
    img_interp, _, _ = find_gaps_and_interpolate(
        img_seq, valid_arr, max_gap_frames=max_gap_frames
    )

    # Normalize body size by median world torso length & image torso pixel height
    valid_idx = np.where(valid_after_interp)[0]
    if len(valid_idx) == 0:
        raise RuntimeError(f"No valid pose frames in {video_path}")

    sh_w = 0.5 * (world_interp[valid_idx, L_SH] + world_interp[valid_idx, R_SH])
    hp_w = 0.5 * (world_interp[valid_idx, L_HP] + world_interp[valid_idx, R_HP])
    torso_world_m = float(np.median(np.linalg.norm(sh_w - hp_w, axis=1))) or 0.5

    sh_px = 0.5 * (img_interp[valid_idx, L_SH, :2] + img_interp[valid_idx, R_SH, :2]) * np.array([width, height])
    hp_px = 0.5 * (img_interp[valid_idx, L_HP, :2] + img_interp[valid_idx, R_HP, :2]) * np.array([width, height])
    torso_img_px = float(np.median(np.linalg.norm(sh_px - hp_px, axis=1))) or 100.0

    # ONE constant facing rotation per clip from opening valid frames
    opening_yaw = compute_opening_yaw_rad(world_interp, valid_after_interp, opening_frames=min(30, len(valid_idx)))
    world_norm = np.full_like(world_interp, np.nan)
    world_norm[valid_idx] = rotate_y(world_interp[valid_idx], opening_yaw) / torso_world_m

    # Smooth jitter strictly inside contiguous valid runs (never across flagged long gaps)
    smooth_win = 7 if fps >= 45.0 else 5
    world_smooth = smooth_contiguous_runs(world_norm, valid_after_interp, window=smooth_win, polyorder=2)

    # Stage travel X (in torso units, relative to opening frame) and foot contact / jump height
    hip_x_px = 0.5 * (img_interp[:, L_HP, 0] + img_interp[:, R_HP, 0]) * width
    hip_x0 = float(hip_x_px[valid_idx[0]])
    travel_x = (hip_x_px - hip_x0) / torso_img_px
    travel_x = smooth_contiguous_runs(travel_x, valid_after_interp, window=smooth_win, polyorder=2)

    # Lower-foot Y in torso units (0 = 95th percentile floor baseline, >0 = jumped upward)
    l_foot_max_y_px = np.full(n_frames, np.nan, dtype=float)
    r_foot_max_y_px = np.full(n_frames, np.nan, dtype=float)
    l_stack = np.stack([img_interp[:, L_AN, 1], img_interp[:, L_HEEL, 1], img_interp[:, L_FOOT, 1]], axis=-1)
    r_stack = np.stack([img_interp[:, R_AN, 1], img_interp[:, R_HEEL, 1], img_interp[:, R_FOOT, 1]], axis=-1)
    l_has = np.any(np.isfinite(l_stack), axis=-1)
    r_has = np.any(np.isfinite(r_stack), axis=-1)
    l_foot_max_y_px[l_has] = np.nanmax(l_stack[l_has], axis=-1) * height
    r_foot_max_y_px[r_has] = np.nanmax(r_stack[r_has], axis=-1) * height
    both_stack = np.stack([l_foot_max_y_px, r_foot_max_y_px], axis=-1)
    both_low_y = np.full(n_frames, np.nan, dtype=float)
    b_has = np.any(np.isfinite(both_stack), axis=-1)
    both_low_y[b_has] = np.nanmax(both_stack[b_has], axis=-1)
    floor_baseline_px = compute_rolling_floor_baseline(both_low_y, valid_after_interp, fps=fps)

    foot_l_height = (floor_baseline_px - l_foot_max_y_px) / torso_img_px
    foot_r_height = (floor_baseline_px - r_foot_max_y_px) / torso_img_px
    jump_height = np.minimum(foot_l_height, foot_r_height)

    # Hand trajectories: NEVER interpolate missing detections; smooth only contiguous detected runs
    lh_seq = np.full((n_frames, 21, 3), np.nan, dtype=float)
    rh_seq = np.full((n_frames, 21, 3), np.nan, dtype=float)
    lh_mask = np.zeros(n_frames, dtype=bool)
    rh_mask = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        if raw_left_hands[i] is not None:
            lh_seq[i] = raw_left_hands[i]
            lh_mask[i] = True
        if raw_right_hands[i] is not None:
            rh_seq[i] = raw_right_hands[i]
            rh_mask[i] = True

    lh_smooth_img = smooth_contiguous_runs(lh_seq, lh_mask, window=smooth_win, polyorder=2)
    rh_smooth_img = smooth_contiguous_runs(rh_seq, rh_mask, window=smooth_win, polyorder=2)

    # Wrist-relative normalized 3D hand coords (in hand-palm scale or torso units)
    lh_rel = np.full_like(lh_smooth_img, np.nan)
    rh_rel = np.full_like(rh_smooth_img, np.nan)
    for mask, h_arr, rel_arr in ((lh_mask, lh_smooth_img, lh_rel), (rh_mask, rh_smooth_img, rh_rel)):
        idxs = np.where(mask)[0]
        for i in idxs:
            pts_px = h_arr[i, :, :2] * np.array([width, height])
            pts_px = (pts_px - pts_px[0]) / torso_img_px
            z_rel = (h_arr[i, :, 2] - h_arr[i, 0, 2]) * (width / torso_img_px)
            rel_arr[i] = np.column_stack([pts_px, z_rel])

    # Foot-strike curve tracks average of ankle & heel height above rolling floor baseline
    # so heel-drop adavus (e.g., Kudithu-Mettu) are detected even when toes stay near the floor.
    l_strike_y_px = np.full(n_frames, np.nan, dtype=float)
    r_strike_y_px = np.full(n_frames, np.nan, dtype=float)
    l_st_stack = np.stack([img_interp[:, L_AN, 1], img_interp[:, L_HEEL, 1]], axis=-1)
    r_st_stack = np.stack([img_interp[:, R_AN, 1], img_interp[:, R_HEEL, 1]], axis=-1)
    l_st_has = np.any(np.isfinite(l_st_stack), axis=-1)
    r_st_has = np.any(np.isfinite(r_st_stack), axis=-1)
    l_strike_y_px[l_st_has] = np.nanmean(l_st_stack[l_st_has], axis=-1) * height
    r_strike_y_px[r_st_has] = np.nanmean(r_st_stack[r_st_has], axis=-1) * height
    l_strike_base = compute_rolling_floor_baseline(l_strike_y_px, valid_after_interp, fps=fps)
    r_strike_base = compute_rolling_floor_baseline(r_strike_y_px, valid_after_interp, fps=fps)
    strike_l_height = (l_strike_base - l_strike_y_px) / torso_img_px
    strike_r_height = (r_strike_base - r_strike_y_px) / torso_img_px

    # Foot-strike movement events vs beats (Measure 4c)
    ankle_x_l_torso = (img_interp[:, L_AN, 0] * width) / torso_img_px
    ankle_x_r_torso = (img_interp[:, R_AN, 0] * width) / torso_img_px
    foot_strikes = detect_foot_strikes(
        strike_l_height,
        strike_r_height,
        valid_after_interp,
        fps=fps,
        ankle_x_l_torso=ankle_x_l_torso,
        ankle_x_r_torso=ankle_x_r_torso,
        ankle_vis_l=img_interp[:, L_AN, 3],
        ankle_vis_r=img_interp[:, R_AN, 3],
    )
    strike_vs_beats = compare_foot_strikes_to_beats(foot_strikes, beat_times_s or [], fps=fps)

    # Depth / travel reliability diagnostics
    torso_cv = float(np.std(torso_px_samples) / max(np.mean(torso_px_samples), 1.0)) if torso_px_samples else 0.0
    world_z_std = float(np.nanstd(world_smooth[valid_idx, :, 2]))
    depth_travel_report = {
        "torso_img_px_median": round(torso_img_px, 1),
        "torso_scale_cv": round(torso_cv, 4),
        "scale_drift_flag": bool(torso_cv > 0.10),
        "world_z_std_torso": round(world_z_std, 4),
        "limitations": [
            "Monocular RGB MediaPipe cannot measure absolute Z-depth or forward/backward stage travel reliably.",
            "Apparent torso pixel height variation (CV={:.3f}) shifts the 2D image-space floor baseline during forward/backward motion.".format(torso_cv),
            "World landmarks are hip-centered, so stage X travel and jump height are inferred from 2D image coordinates.",
            "Quick 180/360 turns cause self-occlusion of wrists, fingers, and ankles.",
        ],
    }

    # Serialize frames compactly
    frames_out = []
    for i in range(n_frames):
        v = bool(valid_after_interp[i])
        frames_out.append({
            "f": i,
            "t": round(i / fps, 4),
            "valid": v,
            "interpolated": bool(v and not valid_arr[i]),
            "travel_x": round(float(travel_x[i]), 4) if v and np.isfinite(travel_x[i]) else None,
            "jump_y": round(float(jump_height[i]), 4) if v and np.isfinite(jump_height[i]) else None,
            "foot_l_y": round(float(foot_l_height[i]), 4) if v and np.isfinite(foot_l_height[i]) else None,
            "foot_r_y": round(float(foot_r_height[i]), 4) if v and np.isfinite(foot_r_height[i]) else None,
            "pose_world": np.round(world_smooth[i], 4).tolist() if v else None,
            "pose_img": np.round(img_interp[i, :, :2], 4).tolist() if v else None,
            "left_hand_img": np.round(lh_smooth_img[i, :, :2], 4).tolist() if lh_mask[i] else None,
            "right_hand_img": np.round(rh_smooth_img[i, :, :2], 4).tolist() if rh_mask[i] else None,
            "left_hand_rel": np.round(lh_rel[i], 4).tolist() if lh_mask[i] else None,
            "right_hand_rel": np.round(rh_rel[i], 4).tolist() if rh_mask[i] else None,
        })

    return {
        "video": video_path.name,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "duration_s": round(n_frames / fps, 3),
        "max_interp_gap_ms": MAX_INTERP_GAP_MS,
        "max_interp_gap_frames": max_gap_frames,
        "opening_facing_yaw_deg": round(math.degrees(opening_yaw), 2),
        "torso_world_m": round(torso_world_m, 4),
        "torso_img_px": round(torso_img_px, 1),
        "raw_pose_valid_ratio": round(float(np.mean(valid_arr)), 4),
        "final_pose_valid_ratio": round(float(np.mean(valid_after_interp)), 4),
        "left_hand_detected_ratio": round(float(np.mean(lh_mask)), 4),
        "right_hand_detected_ratio": round(float(np.mean(rh_mask)), 4),
        "hand_swap_check": {
            "left_swap_rejections": swap_rejections["left"],
            "right_swap_rejections": swap_rejections["right"],
            "overlapping_wrist_frames": overlapping_wrist_frames,
        },
        "flagged_long_gaps": flagged_long_gaps,
        "depth_travel_reliability": depth_travel_report,
        "foot_strikes": foot_strikes,
        "foot_strikes_vs_beats": strike_vs_beats,
        "frames": frames_out,
    }

