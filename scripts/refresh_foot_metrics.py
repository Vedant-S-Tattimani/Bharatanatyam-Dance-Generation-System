"""
refresh_foot_metrics.py – Recompute rolling floor baseline, foot heights, jump_y,
and still-pose-filtered foot strikes across extracted motion JSONs using the
functions in backend/app/motion/extractor.py.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.motion.extractor import (  # noqa: E402
    L_AN,
    L_FOOT,
    L_HEEL,
    R_AN,
    R_FOOT,
    R_HEEL,
    compare_foot_strikes_to_beats,
    compute_rolling_floor_baseline,
    detect_foot_strikes,
)


def _refresh_one(mpath: Path) -> dict:
    m = json.loads(mpath.read_text(encoding="utf-8"))
    stem = mpath.stem
    apath = ROOT / "data" / "audio" / f"{stem}_analysis.json"
    beat_times = []
    if apath.is_file():
        a = json.loads(apath.read_text(encoding="utf-8"))
        beat_times = [float(b["time_s"]) for b in a["beats"]]

    fps = float(m["fps"])
    width = int(m["width"])
    height = int(m["height"])
    torso_img_px = float(m["torso_img_px"])
    frames = m["frames"]
    n = len(frames)

    valid = np.array([bool(fr.get("valid")) for fr in frames], dtype=bool)
    l_foot_max_y_px = np.full(n, np.nan, dtype=float)
    r_foot_max_y_px = np.full(n, np.nan, dtype=float)
    l_strike_y_px = np.full(n, np.nan, dtype=float)
    r_strike_y_px = np.full(n, np.nan, dtype=float)
    xl_torso = np.full(n, np.nan, dtype=float)
    xr_torso = np.full(n, np.nan, dtype=float)

    for i, fr in enumerate(frames):
        pi = fr.get("pose_img")
        if not valid[i] or not pi:
            continue
        l_foot_max_y_px[i] = max(pi[L_AN][1], pi[L_HEEL][1], pi[L_FOOT][1]) * height
        r_foot_max_y_px[i] = max(pi[R_AN][1], pi[R_HEEL][1], pi[R_FOOT][1]) * height
        l_strike_y_px[i] = 0.5 * (pi[L_AN][1] + pi[L_HEEL][1]) * height
        r_strike_y_px[i] = 0.5 * (pi[R_AN][1] + pi[R_HEEL][1]) * height
        xl_torso[i] = (pi[L_AN][0] * width) / torso_img_px
        xr_torso[i] = (pi[R_AN][0] * width) / torso_img_px

    both = np.full(n, np.nan, dtype=float)
    for i in range(n):
        vals = [v for v in (l_foot_max_y_px[i], r_foot_max_y_px[i]) if np.isfinite(v)]
        if vals:
            both[i] = max(vals)

    floor_curve = compute_rolling_floor_baseline(both, valid, fps=fps)
    foot_l_h = (floor_curve - l_foot_max_y_px) / torso_img_px
    foot_r_h = (floor_curve - r_foot_max_y_px) / torso_img_px
    jump_h = np.minimum(foot_l_h, foot_r_h)

    l_st_base = compute_rolling_floor_baseline(l_strike_y_px, valid, fps=fps)
    r_st_base = compute_rolling_floor_baseline(r_strike_y_px, valid, fps=fps)
    strike_l_h = (l_st_base - l_strike_y_px) / torso_img_px
    strike_r_h = (r_st_base - r_strike_y_px) / torso_img_px

    old_count = len(m.get("foot_strikes", []))
    strikes = detect_foot_strikes(
        strike_l_h,
        strike_r_h,
        valid,
        fps=fps,
        ankle_x_l_torso=xl_torso,
        ankle_x_r_torso=xr_torso,
    )
    fsvb = compare_foot_strikes_to_beats(strikes, beat_times, fps=fps)

    for i, fr in enumerate(frames):
        v = bool(valid[i])
        fr["jump_y"] = round(float(jump_h[i]), 4) if v and np.isfinite(jump_h[i]) else None
        fr["foot_l_y"] = round(float(foot_l_h[i]), 4) if v and np.isfinite(foot_l_h[i]) else None
        fr["foot_r_y"] = round(float(foot_r_h[i]), 4) if v and np.isfinite(foot_r_h[i]) else None

    m["foot_strikes"] = strikes
    m["foot_strikes_vs_beats"] = fsvb
    mpath.write_text(json.dumps(m, separators=(",", ":")), encoding="utf-8")
    return {
        "stem": stem,
        "old_strikes": old_count,
        "new_strikes": len(strikes),
        "median_ms": fsvb.get("median_ms"),
    }


def main() -> None:
    paths = sorted((ROOT / "data" / "motion").glob("*.json"))
    workers = min(8, max(2, (os.cpu_count() or 4) // 2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_refresh_one, paths))
    print(f"Refreshed {len(rows)} motion JSONs.")
    for r in rows[:6]:
        print(" ", r)


if __name__ == "__main__":
    main()

