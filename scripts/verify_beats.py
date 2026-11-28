"""
verify_beats.py – Apply user-verified beat/sam/tempo corrections to a clip's
*_beats_review.json, regenerate its *_clicks.wav, and report Measure 4a
(detected beats vs verified beats in ms).

Usage:
  # Inspect or re-score an edited review JSON:
  python scripts/verify_beats.py data/audio/Nattadavu1_beats_review.json

  # Or pass corrections directly on the command line:
  python scripts/verify_beats.py data/audio/Nattadavu1_beats_review.json --sam-beat 2 --tempo-scale 0.5 --offset-ms 15
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.audio.analyzer import (  # noqa: E402
    SR,
    apply_tempo_scale,
    compute_beat_error_ms,
    write_click_track,
)
import librosa  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review_json", help="Path to *_beats_review.json")
    ap.add_argument("--sam-beat", type=int, default=None, help="Verified first sam beat index (0..tala_beats-1)")
    ap.add_argument("--tempo-scale", type=float, default=None, help="Tempo scale (e.g. 0.5 for half, 2.0 for double)")
    ap.add_argument("--offset-ms", type=float, default=None, help="Shift all beats by milliseconds")
    ap.add_argument("--source-video", default=None, help="Optional source video/audio path to regenerate click WAV")
    args = ap.parse_args()

    rpath = Path(args.review_json)
    review = json.loads(rpath.read_text(encoding="utf-8"))

    if args.sam_beat is not None:
        review["verified_first_sam_index"] = args.sam_beat
    if args.tempo_scale is not None:
        review["verified_tempo_scale"] = args.tempo_scale
    if args.offset_ms is not None:
        review["verified_offset_ms"] = args.offset_ms

    detected_beats = review["detected_beats_s"]
    tala_beats = int(review["tala_beats"])

    if review.get("verified_beats_s"):
        verified_beats = [float(t) for t in review["verified_beats_s"]]
    else:
        scale = float(review.get("verified_tempo_scale") or 1.0)
        offset = float(review.get("verified_offset_ms") or 0.0)
        verified_beats = [round(float(t), 4) for t in apply_tempo_scale(detected_beats, tempo_scale=scale, offset_ms=offset)]
        review["verified_beats_s"] = verified_beats

    sam = int(review["verified_first_sam_index"]) if review.get("verified_first_sam_index") is not None else int(review["detected_first_sam_index"])
    err_4a = compute_beat_error_ms(detected_beats, verified_beats)
    review["measure_4a_detected_vs_verified_ms"] = err_4a
    rpath.write_text(json.dumps(review, indent=2), encoding="utf-8")

    print(f"Updated {rpath}")
    print(f"Measure 4a (detected vs verified beats): {json.dumps(err_4a)}")

    src = Path(args.source_video) if args.source_video else (ROOT / "data" / "adavus" / "videos" / review["source_file"])
    if src.is_file():
        y, sr = librosa.load(str(src), sr=SR, mono=True)
        beats_dicts = [
            {"time_s": t, "is_sam": ((i - sam) % tala_beats) == 0}
            for i, t in enumerate(verified_beats)
        ]
        click_out = rpath.with_name(rpath.name.replace("_beats_review.json", "_verified_clicks.wav"))
        write_click_track(y, sr, beats_dicts, click_out)
        print(f"Wrote verified click track: {click_out}")


if __name__ == "__main__":
    main()

