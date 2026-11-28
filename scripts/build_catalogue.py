"""CLI wrapper to build the Step 3 adavu motion catalogue under data/catalogue/."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.catalogue.builder import build_catalogue  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/catalogue/catalogue.json")
    ap.add_argument("--min-segment-s", type=float, default=2.0)
    ap.add_argument("--min-hand-ratio", type=float, default=0.50)
    ap.add_argument("--max-scale-cv", type=float, default=0.12)
    ap.add_argument("--phrase-beats", type=int, default=8)
    args = ap.parse_args()

    cat = build_catalogue(
        ROOT,
        out_path=ROOT / args.out,
        min_segment_s=args.min_segment_s,
        min_hand_ratio=args.min_hand_ratio,
        max_scale_cv=args.max_scale_cv,
        target_phrase_beats=args.phrase_beats,
    )
    summary = {
        "total_units": cat["total_units"],
        "adavu_categories": cat["adavu_categories"],
        "per_adavu_summary": cat["per_adavu_summary"],
        "excluded_clips": cat["excluded_clips"],
        "excluded_unconfirmed_aliases_count": len(cat["excluded_unconfirmed_aliases"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

