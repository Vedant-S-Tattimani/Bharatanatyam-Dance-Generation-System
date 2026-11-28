"""
run_pilot_phase1.py – Phase 1 Steps 1 & 2 on the 4 approved pilot adavus:
  Nattadavu, ThaThaiThamAdavu (excluding unconfirmed KorvaiAdavu alias clips),
  KudithuMettuAdavu, and ThaiyaThaihiAdavu.

Runs in strict order:
  Stage 1: Audio analysis + Demucs/HPSS separation + click tracks + beat review JSONs
  Stage 2: Motion extraction + normalized body/hands + side-by-side preview MP4s
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

PILOT_CATEGORIES = {
    "Nattadavu",
    "ThaThaiThamAdavu",
    "KudithuMettuAdavu",
    "ThaiyaThaihiAdavu",
}


def is_unconfirmed_alias_clip(clip: dict) -> bool:
    """Exclude clips merged into a canonical category through an unconfirmed alias."""
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


def select_pilot_clips(report_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (selected_clips, excluded_alias_clips) from inventory_report.json."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = []
    excluded_aliases = []
    for c in report["clips"]:
        if c["adavu_canonical"] not in PILOT_CATEGORIES or c.get("broken"):
            continue
        vis = c.get("visibility", {})
        if not (vis.get("full_body_visible") and vis.get("hands_mudra_ready")):
            continue
        if is_unconfirmed_alias_clip(c):
            excluded_aliases.append(c)
            continue
        selected.append(c)
    return selected, excluded_aliases


def _audio_worker(task: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    if str(ROOT / "backend") not in sys.path:
        sys.path.insert(0, str(ROOT / "backend"))

    import torch
    torch.set_num_threads(2)

    from app.audio.analyzer import analyze_audio, write_beat_review_file, write_click_track

    clip = task["clip"]
    vpath = Path(task["video_path"])
    stem = vpath.stem
    audio_dir = ROOT / "data" / "audio"
    analysis_json = audio_dir / f"{stem}_analysis.json"
    click_wav = audio_dir / f"{stem}_clicks.wav"
    review_json = audio_dir / f"{stem}_beats_review.json"

    if task["resume"] and analysis_json.is_file() and click_wav.is_file() and review_json.is_file():
        analysis = json.loads(analysis_json.read_text(encoding="utf-8"))
        cached = True
    else:
        analysis, y_mix, sr = analyze_audio(
            vpath,
            tala_name=task["tala"],
            use_demucs=not task["no_demucs"],
        )
        analysis_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        write_click_track(y_mix, sr, analysis["beats"], click_wav)
        write_beat_review_file(analysis, review_json)
        cached = False

    print(
        f"[Audio {task['idx']}/{task['total']}] {clip['adavu_canonical']} :: {clip['filename']} "
        f"({'cached' if cached else analysis['separation_backend']}, bpm={analysis['bpm_median']}, "
        f"beats={analysis['beat_count']}, sam={analysis['first_sam_index']})",
        file=sys.stderr,
        flush=True,
    )
    return {"filename": clip["filename"], "analysis": analysis, "clicks_wav": str(click_wav.relative_to(ROOT))}


def _motion_worker(task: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if str(ROOT / "backend") not in sys.path:
        sys.path.insert(0, str(ROOT / "backend"))

    from app.motion.extractor import extract_clip_motion
    from app.motion.preview import render_side_by_side_preview

    clip = task["clip"]
    vpath = Path(task["video_path"])
    stem = vpath.stem
    audio_dir = ROOT / "data" / "audio"
    motion_dir = ROOT / "data" / "motion"
    preview_dir = ROOT / "data" / "previews"

    analysis = json.loads((audio_dir / f"{stem}_analysis.json").read_text(encoding="utf-8"))
    click_wav = audio_dir / f"{stem}_clicks.wav"
    motion_json = motion_dir / f"{stem}.json"

    if task["resume"] and not task.get("refresh_motion") and motion_json.is_file():
        motion = json.loads(motion_json.read_text(encoding="utf-8"))
        cached = True
    else:
        beat_times = [b["time_s"] for b in analysis["beats"]]
        motion = extract_clip_motion(
            vpath,
            beat_times_s=beat_times,
            max_seconds=task["max_extract_seconds"],
        )
        motion["adavu_canonical"] = clip["adavu_canonical"]
        motion_json.write_text(json.dumps(motion, separators=(",", ":")), encoding="utf-8")
        cached = False

    preview_path = preview_dir / f"{stem}_preview.mp4"
    click_preview_path = preview_dir / f"{stem}_clicks_preview.mp4"
    made_preview = None
    made_click_preview = None

    if task["render_preview"]:
        if not (task["resume"] and preview_path.is_file()):
            render_side_by_side_preview(
                vpath, motion, preview_path, audio_wav=None,
                beats=analysis["beats"], max_seconds=task["preview_seconds"],
            )
        made_preview = str(preview_path.relative_to(ROOT))

    if task["render_click_preview"]:
        if not (task["resume"] and click_preview_path.is_file()):
            render_side_by_side_preview(
                vpath, motion, click_preview_path, audio_wav=click_wav,
                beats=analysis["beats"], max_seconds=task["preview_seconds"],
            )
        made_click_preview = str(click_preview_path.relative_to(ROOT))

    fsvb = motion["foot_strikes_vs_beats"]
    print(
        f"[Motion {task['idx']}/{task['total']}] {clip['adavu_canonical']} :: {clip['filename']} "
        f"({'cached' if cached else 'extracted'}) | pose={motion['final_pose_valid_ratio']:.1%} "
        f"L={motion['left_hand_detected_ratio']:.1%} R={motion['right_hand_detected_ratio']:.1%} "
        f"gaps={len(motion['flagged_long_gaps'])} strikes={fsvb['strike_count']} ({fsvb['median_ms']}ms)",
        file=sys.stderr,
        flush=True,
    )

    return {
        "adavu": clip["adavu_canonical"],
        "filename": clip["filename"],
        "fps": motion["fps"],
        "duration_s": motion["duration_s"],
        "separation_backend": analysis["separation_backend"],
        "separation_detail": analysis["separation_detail"],
        "bpm_median": analysis["bpm_median"],
        "beat_count": analysis["beat_count"],
        "first_sam_index": analysis["first_sam_index"],
        "max_interp_gap_frames": motion["max_interp_gap_frames"],
        "raw_pose_valid_ratio": motion["raw_pose_valid_ratio"],
        "final_pose_valid_ratio": motion["final_pose_valid_ratio"],
        "left_hand_detected_ratio": motion["left_hand_detected_ratio"],
        "right_hand_detected_ratio": motion["right_hand_detected_ratio"],
        "hand_swap_check": motion["hand_swap_check"],
        "flagged_long_gaps_count": len(motion["flagged_long_gaps"]),
        "opening_facing_yaw_deg": motion["opening_facing_yaw_deg"],
        "torso_scale_cv": motion["depth_travel_reliability"]["torso_scale_cv"],
        "scale_drift_flag": motion["depth_travel_reliability"]["scale_drift_flag"],
        "foot_strikes_vs_beats": fsvb,
        "preview_mp4": made_preview,
        "clicks_preview_mp4": made_click_preview,
        "clicks_wav": str(click_wav.relative_to(ROOT)),
    }


def _default_workers() -> tuple[int, int]:
    """Scale audio (GPU/CPU) and motion (CPU) workers to the host hardware."""
    cpus = os.cpu_count() or 8
    has_cuda = False
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except ImportError:
        has_cuda = False
    audio_w = 2 if has_cuda else max(2, cpus // 4)
    motion_w = min(12, max(2, cpus - 2))
    return audio_w, motion_w


def _configure_d_drive_scratch() -> None:
    """Keep PyTorch caches and temp files off low-space C: when running on D:."""
    if ROOT.drive.upper() == "D:":
        tmp_dir = ROOT / "data" / ".tmp"
        torch_dir = ROOT / "models" / "weights" / "torch_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        torch_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(torch_dir))
        os.environ["TMP"] = str(tmp_dir)
        os.environ["TEMP"] = str(tmp_dir)


def main():
    _configure_d_drive_scratch()
    default_audio_w, default_motion_w = _default_workers()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", default="docs/inventory_report.json")
    ap.add_argument("--data-dir", default="data/adavus")
    ap.add_argument("--tala", default="adi")
    ap.add_argument("--no-demucs", action="store_true", help="Skip Demucs and force HPSS fallback")
    ap.add_argument("--audio-workers", type=int, default=default_audio_w, help="Parallel workers for Stage 1 Audio/Demucs")
    ap.add_argument("--motion-workers", type=int, default=default_motion_w, help="Parallel workers for Stage 2 MediaPipe extraction")
    ap.add_argument("--resume", action="store_true", help="Reuse existing outputs when present")
    ap.add_argument("--refresh-motion", action="store_true", help="Re-run Stage 2 motion extraction even when JSON exists")
    ap.add_argument("--max-clips", type=int, default=0, help="Limit total pilot clips processed (0 = all)")
    ap.add_argument("--preview-clips", type=int, default=16, help="Number of clips to render side-by-side preview MP4s (0 = all)")
    ap.add_argument("--click-preview-clips", type=int, default=8, help="Number of clips to render click-audio preview MP4s")
    ap.add_argument("--preview-seconds", type=float, default=0.0, help="Limit preview duration in seconds (0 = full)")
    ap.add_argument("--max-extract-seconds", type=float, default=0.0, help="Limit extraction duration per clip (0 = full)")
    args = ap.parse_args()

    selected, excluded_aliases = select_pilot_clips(ROOT / args.inventory)

    # Interleave categories BEFORE slicing max_clips so subsets cover all 4 pilot adavus evenly
    by_cat: dict[str, list[dict]] = {}
    for c in selected:
        by_cat.setdefault(c["adavu_canonical"], []).append(c)
    ordered: list[dict] = []
    while any(by_cat.values()):
        for k in sorted(by_cat.keys()):
            if by_cat[k]:
                ordered.append(by_cat[k].pop(0))

    if args.max_clips > 0:
        ordered = ordered[:args.max_clips]

    (ROOT / "data" / "audio").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "motion").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "previews").mkdir(parents=True, exist_ok=True)

    print(f"Stage 1 (Audio) + Stage 2 (Motion) on {len(ordered)} interleaved pilot clips (resume={args.resume})", file=sys.stderr)
    print(f"Excluded {len(excluded_aliases)} unconfirmed alias clips.", file=sys.stderr)

    tasks = []
    for idx, clip in enumerate(ordered, 1):
        vpath = ROOT / args.data_dir / clip["relative_path"]
        tasks.append({
            "idx": idx,
            "total": len(ordered),
            "clip": clip,
            "video_path": str(vpath),
            "tala": args.tala,
            "no_demucs": args.no_demucs,
            "resume": args.resume,
            "refresh_motion": args.refresh_motion,
            "max_extract_seconds": args.max_extract_seconds,
            "preview_seconds": args.preview_seconds,
            "render_preview": (args.preview_clips == 0 or idx <= args.preview_clips),
            "render_click_preview": (idx <= args.click_preview_clips),
        })

    # Stage 1: Audio first
    if args.audio_workers <= 1:
        for t in tasks:
            _audio_worker(t)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.audio_workers) as ex:
            list(ex.map(_audio_worker, tasks))

    # Stage 2: Motion extraction second
    rows = []
    if args.motion_workers <= 1:
        for t in tasks:
            rows.append(_motion_worker(t))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.motion_workers) as ex:
            for res in ex.map(_motion_worker, tasks):
                rows.append(res)

    out_summary = {
        "pilot_categories": sorted(PILOT_CATEGORIES),
        "selected_count": len(ordered),
        "excluded_unconfirmed_alias_clips": [
            {"filename": c["filename"], "adavu_canonical": c["adavu_canonical"], "alternates": c["adavu_alternates"]}
            for c in excluded_aliases
        ],
        "clips": rows,
    }
    (ROOT / "data" / "pilot_summary.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    print(f"Pilot summary written to data/pilot_summary.json ({len(rows)} clips)", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
