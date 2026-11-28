"""
preview.py – Render side-by-side preview videos (source frame | extracted body + hands skeleton)
muxed with the source clip's audio (or click-track audio).
"""

from __future__ import annotations

from functools import lru_cache
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

POSE_EDGES = (
    (11, 12), (11, 23), (12, 24), (23, 24),  # torso
    (11, 13), (13, 15),                       # left arm
    (12, 14), (14, 16),                       # right arm
    (23, 25), (25, 27), (27, 29), (29, 31),   # left leg & foot
    (24, 26), (26, 28), (28, 30), (30, 32),   # right leg & foot
    (0, 11), (0, 12),                         # head-to-shoulders
)

HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def _draw_skeleton_canvas(
    frame_rec: dict[str, Any],
    w: int,
    h: int,
    beat_info: str = "",
) -> np.ndarray:
    canvas = np.full((h, w, 3), 24, dtype=np.uint8)

    if frame_rec["valid"] and frame_rec["pose_img"]:
        pts = np.asarray(frame_rec["pose_img"], dtype=float)
        px = np.column_stack([pts[:, 0] * w, pts[:, 1] * h]).astype(int)
        for a, b in POSE_EDGES:
            cv2.line(canvas, tuple(px[a]), tuple(px[b]), (90, 220, 140), 3, cv2.LINE_AA)
        for idx in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32):
            cv2.circle(canvas, tuple(px[idx]), 5, (255, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "LOW CONFIDENCE / GAP (NOT INTERPOLATED)", (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)

    # Draw left & right hand landmarks when detected (never invented)
    for key, col in (("left_hand_img", (255, 190, 60)), ("right_hand_img", (80, 180, 255))):
        hpts = frame_rec.get(key)
        if hpts:
            hp = np.asarray(hpts, dtype=float)
            hpx = np.column_stack([hp[:, 0] * w, hp[:, 1] * h]).astype(int)
            for a, b in HAND_EDGES:
                cv2.line(canvas, tuple(hpx[a]), tuple(hpx[b]), col, 2, cv2.LINE_AA)
            for p in hpx:
                cv2.circle(canvas, tuple(p), 2, (255, 255, 255), -1, cv2.LINE_AA)

    hud = f"t={frame_rec['t']:.2f}s | L-hand:{'Y' if frame_rec.get('left_hand_img') else '-'} R-hand:{'Y' if frame_rec.get('right_hand_img') else '-'}"
    if beat_info:
        hud = f"{hud} | {beat_info}"
    cv2.putText(canvas, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    return canvas


@lru_cache(maxsize=4)
def _pick_h264_encoder(ffmpeg_bin: str) -> tuple[str, list[str]]:
    """Probe the fastest working H.264 encoder once per process (NVENC -> MF -> libx264)."""
    candidates: list[tuple[str, list[str]]] = [
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-pix_fmt", "yuv420p"]),
        ("h264_mf", ["-c:v", "h264_mf", "-hw_encoding", "1", "-rate_control", "quality", "-quality", "75", "-pix_fmt", "yuv420p"]),
        ("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "0", "-pix_fmt", "yuv420p"]),
    ]
    for name, flags in candidates:
        probe = subprocess.run(
            [
                ffmpeg_bin, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.1",
                *flags, "-f", "mp4", "-y", os.devnull,
            ],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode == 0:
            return name, flags
    return "libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "0", "-pix_fmt", "yuv420p"]


def render_side_by_side_preview(
    source_video: Path,
    motion: dict[str, Any],
    out_mp4: Path,
    audio_wav: Path | None = None,
    beats: list[dict[str, Any]] | None = None,
    panel_width: int = 640,
    max_seconds: float = 0.0,
) -> Path:
    """
    Render a side-by-side preview MP4 (source left | extracted skeleton right),
    muxing original audio or click-track WAV in a single hardware-accelerated ffmpeg pass when available.
    """
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source_video}")

    fps = float(motion["fps"])
    src_w, src_h = int(motion["width"]), int(motion["height"])
    panel_h = int(round(panel_width * (src_h / max(src_w, 1))))
    if panel_h % 2 == 1:
        panel_h += 1

    out_w, out_h = panel_width * 2, panel_h
    frames = motion["frames"]
    beats = beats or []
    beat_idx = 0

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg:
        audio_src = str(audio_wav) if audio_wav and audio_wav.is_file() else str(source_video)
        _, enc_flags = _pick_h264_encoder(ffmpeg)
        pipe_cmd = [
            ffmpeg, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}", "-r", f"{fps:.6f}",
            "-i", "pipe:0",
            "-i", audio_src,
            "-map", "0:v:0", "-map", "1:a:0?",
            *enc_flags,
            "-c:a", "aac", "-shortest", str(out_mp4),
        ]
        proc = subprocess.Popen(pipe_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert proc.stdin is not None
            for rec in frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                if max_seconds > 0 and rec["t"] > max_seconds:
                    break

                while beat_idx + 1 < len(beats) and beats[beat_idx + 1]["time_s"] <= rec["t"]:
                    beat_idx += 1
                beat_txt = ""
                if beats and beats[beat_idx]["time_s"] <= rec["t"] + 0.05:
                    b = beats[beat_idx]
                    beat_txt = f"Beat {b['tala_pos']}{'*SAM*' if b['is_sam'] else ''}"

                left_panel = cv2.resize(bgr, (panel_width, panel_h))
                right_panel = _draw_skeleton_canvas(rec, panel_width, panel_h, beat_info=beat_txt)
                combined = np.ascontiguousarray(np.hstack([left_panel, right_panel]))
                proc.stdin.write(combined.tobytes())
            proc.stdin.close()
            proc.wait(timeout=300)
            if proc.returncode == 0 and out_mp4.is_file() and out_mp4.stat().st_size > 0:
                cap.release()
                return out_mp4
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        beat_idx = 0

    with tempfile.TemporaryDirectory(prefix="nritya_prev_") as tmp_dir:
        raw_vid = Path(tmp_dir) / "silent.mp4" if ffmpeg else out_mp4
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_vid), fourcc, fps, (out_w, out_h))

        for rec in frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if max_seconds > 0 and rec["t"] > max_seconds:
                break

            while beat_idx + 1 < len(beats) and beats[beat_idx + 1]["time_s"] <= rec["t"]:
                beat_idx += 1
            beat_txt = ""
            if beats and beats[beat_idx]["time_s"] <= rec["t"] + 0.05:
                b = beats[beat_idx]
                beat_txt = f"Beat {b['tala_pos']}{'*SAM*' if b['is_sam'] else ''}"

            left_panel = cv2.resize(bgr, (panel_width, panel_h))
            right_panel = _draw_skeleton_canvas(rec, panel_width, panel_h, beat_info=beat_txt)
            combined = np.hstack([left_panel, right_panel])
            writer.write(combined)

        writer.release()
        cap.release()

        if ffmpeg:
            audio_src = str(audio_wav) if audio_wav and audio_wav.is_file() else str(source_video)
            cpu_cmd = [
                ffmpeg, "-y", "-v", "error",
                "-i", str(raw_vid), "-i", audio_src,
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "0", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(out_mp4),
            ]
            res2 = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=300)
            if res2.returncode != 0:
                shutil.copy(raw_vid, out_mp4)

    return out_mp4


def _draw_choreo_frame(
    frame_rec: dict[str, Any],
    w: int,
    h: int,
    t_s: float,
    beat_txt: str = "",
    waiting: bool = False,
) -> np.ndarray:
    """Render normalized 3D choreography frame onto a 2D stick-figure stage canvas."""
    canvas = np.full((h, w, 3), 20, dtype=np.uint8)
    floor_y = int(round(h * 0.84))
    cv2.line(canvas, (40, floor_y), (w - 40, floor_y), (70, 75, 85), 2, cv2.LINE_AA)

    pw = np.asarray(frame_rec["pose_world"], dtype=float)
    tx = float(frame_rec.get("travel_x", 0.0))
    jy = float(frame_rec.get("jump_y", 0.0))
    scale = h * 0.24

    # Ground lowest foot onto floor line and lift only during positive jump_y
    foot_y_max = float(np.max(pw[[27, 28, 29, 30, 31, 32], 1]))
    cx0 = w * 0.5 + tx * scale
    cy0 = floor_y - foot_y_max * scale - jy * scale
    px = np.column_stack([cx0 + pw[:, 0] * scale, cy0 + pw[:, 1] * scale]).astype(int)

    for a, b in POSE_EDGES:
        cv2.line(canvas, tuple(px[a]), tuple(px[b]), (90, 225, 145), 3, cv2.LINE_AA)
    for idx in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32):
        cv2.circle(canvas, tuple(px[idx]), 5, (255, 255, 255), -1, cv2.LINE_AA)

    # Render genuine detected hands attached to pose wrists (never invent missing hands)
    for key, wrist_idx, col in (
        ("left_hand_rel", 15, (255, 190, 60)),
        ("right_hand_rel", 16, (80, 180, 255)),
    ):
        h_rel = frame_rec.get(key)
        if h_rel is None:
            continue
        hr = np.asarray(h_rel, dtype=float)
        wx, wy = float(px[wrist_idx, 0]), float(px[wrist_idx, 1])
        hpx = np.column_stack([wx + hr[:, 0] * scale, wy + hr[:, 1] * scale]).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(canvas, tuple(hpx[a]), tuple(hpx[b]), col, 2, cv2.LINE_AA)
        for p in hpx:
            cv2.circle(canvas, tuple(p), 2, (255, 255, 255), -1, cv2.LINE_AA)

    adavu = frame_rec.get("adavu", "-")
    conf_tag = "" if frame_rec.get("confirmed", True) else " [unconfirmed]"
    hud = f"t={t_s:.2f}s | {adavu}{conf_tag}"
    if waiting:
        hud += " | waiting for beat 1"
    if beat_txt:
        hud += f" | {beat_txt}"
    cv2.putText(canvas, hud, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2, cv2.LINE_AA)
    return canvas


def render_choreography_video(
    choreo: dict[str, Any],
    audio_source: Path,
    out_mp4: Path,
    beats: list[dict[str, Any]] | None = None,
    width: int = 960,
    height: int = 540,
    max_seconds: float = 0.0,
) -> Path:
    """Render a planned choreography JSON into a stick-figure MP4 muxed with song audio from t=0."""
    frames = choreo["frames"]
    fps = float(choreo["fps"])
    if not frames:
        raise ValueError("Choreography has zero frames")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    t0 = float(frames[0]["t"])
    lead_frames = max(0, int(round(t0 * fps)))
    beats = beats or []
    beat_idx = 0

    _, enc_flags = _pick_h264_encoder(ffmpeg)
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        "-i", str(audio_source),
        "-map", "0:v:0", "-map", "1:a:0?",
        *enc_flags,
        "-c:a", "aac", "-shortest", str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for i in range(lead_frames):
            img = _draw_choreo_frame(frames[0], width, height, i / fps, waiting=True)
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
        for rec in frames:
            t_cur = float(rec["t"])
            if max_seconds > 0 and (t_cur - t0) > max_seconds:
                break
            while beat_idx + 1 < len(beats) and beats[beat_idx + 1]["time_s"] <= t_cur:
                beat_idx += 1
            beat_txt = ""
            if beats and beats[beat_idx]["time_s"] <= t_cur + 0.05:
                b = beats[beat_idx]
                beat_txt = f"Beat {b['tala_pos']}{'*SAM*' if b['is_sam'] else ''}"
            img = _draw_choreo_frame(rec, width, height, t_cur, beat_txt=beat_txt)
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
        proc.stdin.close()
        proc.wait(timeout=300)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)
        raise

    if proc.returncode != 0:
        err = (proc.stderr.read() if proc.stderr else b"").decode(errors="ignore")
        raise RuntimeError(f"ffmpeg failed rendering choreography preview: {err}")
    return out_mp4

