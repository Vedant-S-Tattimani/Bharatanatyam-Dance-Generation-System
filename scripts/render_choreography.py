"""
render_choreography.py – Analyze a song audio/video file, plan synchronized
Bharatanatyam choreography (excluding any matching dataset source clip), and
render a stick-figure MP4 with audio under data/previews/choreography/.

Usage:
  python scripts/render_choreography.py path/to/song1.mp3 path/to/song2.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.audio.analyzer import analyze_audio  # noqa: E402
from app.catalogue.builder import build_catalogue, load_catalogue  # noqa: E402
from app.motion.preview import render_choreography_video  # noqa: E402
from app.planner.choreographer import plan_choreography  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("songs", nargs="+", help="One or more audio/video files")
    ap.add_argument("--tala", default="adi")
    ap.add_argument("--no-demucs", action="store_true")
    ap.add_argument("--include-unconfirmed", action="store_true", help="Allow ThaThaiTham unconfirmed units")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out-dir", default="data/previews/choreography")
    args = ap.parse_args()

    cat_path = ROOT / "data" / "catalogue" / "catalogue.json"
    cat = load_catalogue(cat_path) if cat_path.is_file() else build_catalogue(ROOT, out_path=cat_path)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for song_arg in args.songs:
        spath = Path(song_arg)
        if not spath.is_file():
            raise FileNotFoundError(f"Song file not found: {spath}")
        analysis, _, _ = analyze_audio(spath, tala_name=args.tala, use_demucs=not args.no_demucs)
        choreo = plan_choreography(
            audio_analysis=analysis,
            catalogue=cat,
            root=ROOT,
            fps=args.fps,
            include_unconfirmed=args.include_unconfirmed,
            exclude_stems={spath.stem},
        )
        suffix = "with_unconfirmed" if args.include_unconfirmed else "confirmed_only"
        out_mp4 = out_dir / f"{spath.stem}_{suffix}.mp4"
        render_choreography_video(choreo, spath, out_mp4, beats=analysis["beats"])
        reports.append({
            "song": spath.name,
            "out_mp4": str(out_mp4.relative_to(ROOT)).replace("\\", "/"),
            "song_bpm": choreo["song_bpm"],
            "separation_backend": choreo["separation_backend"],
            "include_unconfirmed": choreo["include_unconfirmed"],
            "excluded_stems": choreo["excluded_stems"],
            "adavus_used": choreo["adavus_used"],
            "source_stems_used": choreo["source_stems_used"],
            "duration_s": choreo["duration_s"],
            "n_frames": choreo["n_frames"],
            "frame_quantization_check_ms": choreo["frame_quantization_check_ms"],
            "planned_foot_strikes_vs_beats": choreo["planned_foot_strikes_vs_beats"],
            "transition_metrics": choreo["transition_metrics"],
        })
        print(f"Rendered {out_mp4} ({choreo['n_frames']} frames, {choreo['duration_s']}s)", file=sys.stderr)

    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

