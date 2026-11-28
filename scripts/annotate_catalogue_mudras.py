"""Annotate catalogue phrase units with the locally trained mudra hand-shape prior.

The classifier reads the genuine ``left_hand_rel`` / ``right_hand_rel`` landmarks
already stored in the extracted motion JSONs. No image inference and no new
landmark generation happens at choreography time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.motion.mudra import MudraClassifier, annotate_catalogue  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=None,
        help="Defaults to <root>/data/catalogue/catalogue.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to <root>/data/catalogue/mudra_annotations.json",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--max-frames-per-unit", type=int, default=8)
    args = parser.parse_args()

    root = args.root.resolve()
    catalogue_path = args.catalogue or root / "data" / "catalogue" / "catalogue.json"
    model_path = args.model or root / "models" / "mudra_landmark_classifier.joblib"
    meta_path = args.meta or root / "models" / "mudra_landmark_classifier.json"
    out_path = args.out or root / "data" / "catalogue" / "mudra_annotations.json"

    if not catalogue_path.is_file():
        raise SystemExit(f"Catalogue not found: {catalogue_path}")
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))

    classifier = MudraClassifier(model_path, meta_path)
    if not classifier.available:
        raise SystemExit(
            "Mudra classifier unavailable. "
            f"Run scripts/train_mudra_landmark_model.py first. {classifier.load_error or ''}"
        )

    annotations = annotate_catalogue(
        root,
        catalogue,
        classifier,
        out_path=out_path,
        max_frames_per_unit=args.max_frames_per_unit,
    )
    labels = sorted({label for ann in annotations.values() for label in ann.get("labels", [])})
    print(f"Annotated {len(annotations)} catalogue units -> {out_path}")
    print(f"Model: {classifier.version} · distinct labels: {len(labels)} · {', '.join(labels[:8])}")


if __name__ == "__main__":
    main()
