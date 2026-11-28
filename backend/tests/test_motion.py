"""Unit tests for Phase 1 Step 2: motion extraction helpers, <=100ms gap rule, hand-swap check, and facing rotation."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.motion.extractor import (  # noqa: E402
    L_HP,
    L_SH,
    L_WR,
    R_HP,
    R_SH,
    R_WR,
    compare_foot_strikes_to_beats,
    compute_opening_yaw_rad,
    compute_rolling_floor_baseline,
    detect_foot_strikes,
    find_gaps_and_interpolate,
    max_gap_frames_for_fps,
    rotate_y,
    verify_and_assign_hand,
)


def test_max_gap_frames_converts_100ms_by_fps():
    assert max_gap_frames_for_fps(25.0) == 2
    assert max_gap_frames_for_fps(30.0) == 3
    assert max_gap_frames_for_fps(59.94) == 5


def test_never_interpolates_long_gaps():
    # 10 frames, short 2-frame gap at [2,3] (<=2), long 3-frame gap at [5,6,7] (>2)
    vals = np.arange(10, dtype=float)[:, None]
    valid = np.array([True, True, False, False, True, False, False, False, True, True])
    out, out_valid, flagged = find_gaps_and_interpolate(vals, valid, max_gap_frames=2)

    assert out_valid[2] and out_valid[3]
    assert np.isclose(out[2, 0], 2.0) and np.isclose(out[3, 0], 3.0)
    assert not out_valid[5] and not out_valid[6] and not out_valid[7]
    assert np.isnan(out[5, 0]) and np.isnan(out[6, 0]) and np.isnan(out[7, 0])
    assert len(flagged) == 1
    assert flagged[0]["start_frame"] == 5 and flagged[0]["end_frame"] == 7


def test_opening_yaw_uses_single_rotation_preserving_later_turns():
    # Create 40 frames: first 20 face 45 deg yaw, last 20 turn by another 45 deg (to 90 deg)
    world = np.zeros((40, 33, 3), dtype=float)
    yaw1 = math.radians(45.0)
    yaw2 = math.radians(90.0)
    for i in range(40):
        ang = yaw1 if i < 20 else yaw2
        world[i, L_SH] = [0.2 * math.cos(ang), 0.5, 0.2 * math.sin(ang)]
        world[i, R_SH] = [-0.2 * math.cos(ang), 0.5, -0.2 * math.sin(ang)]
        world[i, L_HP] = [0.15 * math.cos(ang), 0.0, 0.15 * math.sin(ang)]
        world[i, R_HP] = [-0.15 * math.cos(ang), 0.0, -0.15 * math.sin(ang)]
    valid = np.ones(40, dtype=bool)
    yaw_clip = compute_opening_yaw_rad(world, valid, opening_frames=20)
    assert math.isclose(math.degrees(yaw_clip), 45.0, abs_tol=1e-3)

    rot = rotate_y(world, yaw_clip)
    # Opening frames aligned to +X (z ~= 0)
    assert abs(rot[0, L_SH, 2]) < 1e-6
    # Later turn is PRESERVED (~45 deg residual yaw, not erased!)
    diff_later = rot[30, L_SH] - rot[30, R_SH]
    residual_deg = math.degrees(math.atan2(diff_later[2], diff_later[0]))
    assert math.isclose(abs(residual_deg), 45.0, abs_tol=1e-2)


def test_hand_swap_detection_rejects_swapped_wrist():
    pose_img = np.zeros((33, 4), dtype=float)
    pose_img[L_WR] = [0.7, 0.5, 0.0, 1.0]
    pose_img[R_WR] = [0.3, 0.5, 0.0, 1.0]

    # Hand near right wrist should fail when assigned to "left"
    swapped_hand = np.zeros((21, 3), dtype=float)
    swapped_hand[0] = [0.31, 0.5, 0.0]
    ok, _ = verify_and_assign_hand(swapped_hand, "left", pose_img, torso_img_px=200.0, img_w=1000, img_h=1000)
    assert ok is False

    good_hand = np.zeros((21, 3), dtype=float)
    good_hand[0] = [0.71, 0.5, 0.0]
    ok2, _ = verify_and_assign_hand(good_hand, "left", pose_img, torso_img_px=200.0, img_w=1000, img_h=1000)
    assert ok2 is True


def test_foot_strike_detection_and_beat_comparison():
    fps = 30.0
    n = 90  # 3 seconds
    valid = np.ones(n, dtype=bool)
    l_y = np.zeros(n, dtype=float)
    r_y = np.zeros(n, dtype=float)
    # Left foot lifts around frame 20 and lands at frame 30 (t = 1.0s)
    l_y[20:30] = np.linspace(0.02, 0.15, 10)
    l_y[28] = 0.12
    l_y[29] = 0.06
    l_y[30] = 0.01
    l_y[31] = 0.015

    # Legacy behavior fires on threshold crossing at frame 30; default trough timing snaps to frame 32 (y=0.0 floor).
    strikes_legacy = detect_foot_strikes(l_y, r_y, valid, fps=fps, use_trough_timing=False)
    assert len(strikes_legacy) == 1
    assert strikes_legacy[0]["frame"] == 30
    comp = compare_foot_strikes_to_beats(strikes_legacy, [1.02], fps=fps)
    assert comp["median_ms"] == 20.0
    assert abs(comp["median_frames"] - 0.6) < 0.05

    strikes_trough = detect_foot_strikes(l_y, r_y, valid, fps=fps, use_trough_timing=True)
    assert len(strikes_trough) == 1
    assert strikes_trough[0]["frame"] == 32


def test_render_side_by_side_preview_single_pass(tmp_path: Path):
    import cv2
    from app.motion.preview import render_side_by_side_preview

    src_vid = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(src_vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for _ in range(5):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    frames = [
        {"f": i, "t": i * 0.1, "valid": False, "pose_img": None, "left_hand_img": None, "right_hand_img": None}
        for i in range(5)
    ]
    motion = {"fps": 10.0, "width": 160, "height": 120, "frames": frames}
    out_mp4 = tmp_path / "out_preview.mp4"
    render_side_by_side_preview(src_vid, motion, out_mp4, panel_width=160)
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0


def test_rolling_floor_baseline_handles_camera_distance_shift():
    fps = 30.0
    n = 360  # 12 seconds: first 6s floor at y=800, last 6s dancer steps closer -> floor at y=860
    valid = np.ones(n, dtype=bool)
    l_px = np.full(n, 800.0)
    r_px = np.full(n, 800.0)
    l_px[180:] = 860.0
    r_px[180:] = 860.0

    # Real strike in first half around frame 45 and in second half around frame 300
    l_px[36:45] = np.linspace(800.0, 740.0, 9)  # lift up (smaller pixel y)
    l_px[44] = 770.0
    l_px[45] = 800.0
    r_px[290:300] = np.linspace(860.0, 800.0, 10)
    r_px[299] = 830.0
    r_px[300] = 860.0

    both = np.maximum(l_px, r_px)
    baseline = compute_rolling_floor_baseline(both, valid, fps=fps, window_s=3.0)
    torso_px = 200.0
    l_h = (baseline - l_px) / torso_px
    r_h = (baseline - r_px) / torso_px

    strikes = detect_foot_strikes(l_h, r_h, valid, fps=fps)
    frames = [s["frame"] for s in strikes]
    assert any(42 <= f <= 48 for f in frames), f"Missed strike in first half: {frames}"
    assert any(297 <= f <= 303 for f in frames), f"Missed strike in second half: {frames}"


def test_still_pose_jitter_produces_zero_false_foot_strikes():
    fps = 30.0
    n = 180
    valid = np.ones(n, dtype=bool)
    rng = np.random.default_rng(7)
    # Typical stationary MediaPipe ankle noise (~ +/-0.025 torso units, occasional 0.045 spike)
    l_y = np.clip(0.015 + rng.normal(0.0, 0.018, size=n), 0.0, 0.06)
    r_y = np.clip(0.015 + rng.normal(0.0, 0.018, size=n), 0.0, 0.06)
    strikes = detect_foot_strikes(l_y, r_y, valid, fps=fps)
    assert strikes == []


def test_trough_timing_switch_fixes_descent_shoulder():
    fps = 30.0
    n = 90
    valid = np.ones(n, dtype=bool)
    l_y = np.zeros(n, dtype=float)
    r_y = np.zeros(n, dtype=float)
    # Foot lifts to 0.16, enters contact_thresh at frame 28 (local shoulder 0.032 -> 0.035 at 29 -> true trough 0.002 at 31)
    l_y[18:27] = np.linspace(0.02, 0.16, 9)
    l_y[27] = 0.090
    l_y[28] = 0.032
    l_y[29] = 0.035  # brief hesitation on descent
    l_y[30] = 0.012
    l_y[31] = 0.002  # true bottom contact
    l_y[32] = 0.004

    old_s = detect_foot_strikes(l_y, r_y, valid, fps=fps, use_trough_timing=False)
    new_s = detect_foot_strikes(l_y, r_y, valid, fps=fps, use_trough_timing=True)
    assert old_s[0]["frame"] == 28
    assert new_s[0]["frame"] == 31


def test_crossed_leg_rejection_and_adaptive_amplitude_switches():
    fps = 30.0
    n = 90
    valid = np.ones(n, dtype=bool)
    l_y = np.zeros(n, dtype=float)
    r_y = np.zeros(n, dtype=float)
    xl = np.full(n, 0.30)
    xr = np.full(n, -0.30)
    vis_l = np.ones(n, dtype=float)
    vis_r = np.ones(n, dtype=float)

    # Synthetic crossed-leg false positive at frame 30 (L ankle crosses near R with low vis and sits higher)
    l_y[22:30] = np.linspace(0.14, 0.04, 8)
    l_y[30] = 0.035
    l_y[31] = 0.040
    xl[26:34] = -0.25  # within 0.05 torso of xr=-0.30
    vis_l[28:33] = 0.40

    # Synthetic low-amplitude real Kudithu-Mettu drop at frame 60 (lift 0.072, trough 0.004, settles)
    r_y[50:58] = np.linspace(0.01, 0.072, 8)
    r_y[58] = 0.045
    r_y[59] = 0.018
    r_y[60] = 0.000
    r_y[61:64] = 0.002

    base = detect_foot_strikes(
        l_y, r_y, valid, fps=fps, use_trough_timing=False, use_crossed_leg_rejection=False,
    )
    no_cross = detect_foot_strikes(
        l_y, r_y, valid, fps=fps, use_crossed_leg_rejection=True,
        ankle_x_l_torso=xl, ankle_x_r_torso=xr, ankle_vis_l=vis_l, ankle_vis_r=vis_r,
    )
    adapt = detect_foot_strikes(
        l_y, r_y, valid, fps=fps, use_crossed_leg_rejection=True, use_adaptive_amplitude=True,
        ankle_x_l_torso=xl, ankle_x_r_torso=xr, ankle_vis_l=vis_l, ankle_vis_r=vis_r,
    )
    assert any(s["frame"] == 30 and s["side"] == "left" for s in base)
    assert not any(s["side"] == "left" and 28 <= s["frame"] <= 34 for s in no_cross)
    assert any(s["frame"] == 60 and s["side"] == "right" for s in adapt)

