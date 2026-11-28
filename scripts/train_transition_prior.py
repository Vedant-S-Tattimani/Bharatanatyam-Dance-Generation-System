"""Train a bounded recorded-join transition prior for the choreography planner.

Positive examples are consecutive phrase units recorded in the same source clip.
Hard negatives are non-adjacent pairs whose endpoint poses are already close, so
the model learns complementary boundary-flow cues instead of rediscovering the
catalogue's shared-endpoint construction. Splits are grouped by source clip.

Scope: this is a join-ordering prior over recorded Bharatanatyam adavu motion.
It is not paired music-to-expert-choreography supervision, invents no frames,
and cannot claim that an arbitrary song receives the "correct" choreography.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.motion.transition_prior import (  # noqa: E402
    FEATURE_NAMES,
    build_boundary_summary,
    join_features,
    pose_endpoint_distance,
)

DEFAULT_CATALOGUE = ROOT / "data" / "catalogue" / "catalogue.json"
DEFAULT_MODEL = ROOT / "models" / "transition_prior.joblib"
DEFAULT_META = ROOT / "models" / "transition_prior.json"
DEFAULT_SUMMARY = ROOT / "data" / "catalogue" / "transition_features.json"
MODEL_ID = "nritya-recorded-join-v1"


Example = tuple[np.ndarray, int, str, str, float]


def _load_inputs(catalogue_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    units = catalogue.get("units", [])
    if not units:
        raise SystemExit(f"Catalogue has zero units: {catalogue_path}")

    motion_cache: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for unit in units:
        rel = str(unit["motion_path"])
        if rel not in motion_cache:
            mpath = ROOT / rel
            if not mpath.is_file():
                raise SystemExit(f"Missing motion JSON: {mpath}")
            motion_cache[rel] = json.loads(mpath.read_text(encoding="utf-8"))
        summaries[str(unit["unit_id"])] = build_boundary_summary(
            unit, motion_cache[rel].get("frames", [])
        )
    return catalogue, summaries


def _build_examples(
    units: list[dict[str, Any]],
    summaries: dict[str, Any],
    *,
    max_pose_distance: float,
    negatives_per_positive: int,
    seed: int,
) -> tuple[list[Example], list[Example], dict[str, int]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        by_source.setdefault(str(unit["source_stem"]), []).append(unit)
    sources = sorted(by_source)
    rng = np.random.default_rng(seed)

    positives: list[Example] = []
    same_source_negatives: list[Example] = []

    def make_example(from_unit: dict[str, Any], to_unit: dict[str, Any], label: int) -> Example | None:
        uid_from, uid_to = str(from_unit["unit_id"]), str(to_unit["unit_id"])
        summary_from, summary_to = summaries.get(uid_from), summaries.get(uid_to)
        if not summary_from or not summary_to:
            return None
        features = join_features(summary_from, from_unit, summary_to, to_unit)
        distance = pose_endpoint_distance(summary_from, summary_to)
        if features is None or distance is None:
            return None
        return (
            features,
            label,
            str(from_unit["source_stem"]),
            str(to_unit["source_stem"]),
            distance,
        )

    for stem in sources:
        ordered = sorted(by_source[stem], key=lambda u: (int(u["start_frame"]), int(u["end_frame"])))
        for prev, nxt in zip(ordered, ordered[1:]):
            gap = int(nxt["start_frame"]) - int(prev["end_frame"])
            if 0 <= gap <= 1:
                example = make_example(prev, nxt, 1)
                if example is not None:
                    positives.append(example)
        for i, prev in enumerate(ordered):
            for nxt in ordered[i + 1 :]:
                if int(nxt["start_frame"]) - int(prev["end_frame"]) <= 1:
                    continue
                example = make_example(prev, nxt, 0)
                if example is not None and example[4] <= max_pose_distance:
                    same_source_negatives.append(example)

    cross_source_negatives: list[Example] = []
    target_cross = max(200, len(positives) * negatives_per_positive * 5)
    for _ in range(target_cross):
        source_from, source_to = rng.choice(sources, 2, replace=False)
        from_unit = by_source[str(source_from)][int(rng.integers(len(by_source[str(source_from)])))]
        to_unit = by_source[str(source_to)][int(rng.integers(len(by_source[str(source_to)])))]
        example = make_example(from_unit, to_unit, 0)
        if example is not None and example[4] <= max_pose_distance:
            cross_source_negatives.append(example)

    counts = {
        "sources": len(sources),
        "positives": len(positives),
        "same_source_hard_negatives": len(same_source_negatives),
        "cross_source_hard_negatives": len(cross_source_negatives),
    }
    return positives, same_source_negatives + cross_source_negatives, counts


def _split_sources(sources: list[str], test_size: float, seed: int) -> tuple[set[str], set[str]]:
    if len(sources) < 2:
        raise SystemExit("Need at least two source clips for a source-grouped split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(
        splitter.split(np.arange(len(sources)), groups=np.asarray(sources))
    )
    return {sources[int(i)] for i in train_idx}, {sources[int(i)] for i in test_idx}


def _in_split(example: Example, sources: set[str]) -> bool:
    return example[2] in sources and example[3] in sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--max-pose-distance", type=float, default=0.25)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.catalogue.is_file():
        raise SystemExit(f"Missing catalogue: {args.catalogue}")
    if args.max_pose_distance <= 0 or args.negatives_per_positive < 1:
        raise SystemExit("Expect positive --max-pose-distance and --negatives-per-positive >= 1")

    catalogue, summaries = _load_inputs(args.catalogue)
    units = catalogue.get("units", [])
    sources = sorted({str(u["source_stem"]) for u in units})
    positives, negatives, counts = _build_examples(
        units,
        summaries,
        max_pose_distance=args.max_pose_distance,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    print(f"Built {counts['positives']} recorded joins and "
          f"{counts['same_source_hard_negatives'] + counts['cross_source_hard_negatives']} hard negatives "
          f"from {counts['sources']} source clips", flush=True)

    train_sources, test_sources = _split_sources(sources, args.test_size, args.seed)
    rng = np.random.default_rng(args.seed)
    train_pos = [e for e in positives if _in_split(e, train_sources)]
    test_pos = [e for e in positives if _in_split(e, test_sources)]
    train_neg = [e for e in negatives if _in_split(e, train_sources)]
    test_neg = [e for e in negatives if _in_split(e, test_sources)]
    if not train_pos or not test_pos or not train_neg or not test_neg:
        raise SystemExit("Source-grouped split left an empty train/test class; adjust --test-size")

    rng.shuffle(train_neg)
    rng.shuffle(test_neg)
    train_neg = train_neg[: len(train_pos) * args.negatives_per_positive]
    test_neg = test_neg[: len(test_pos) * args.negatives_per_positive]
    train = train_pos + train_neg
    test = test_pos + test_neg

    X_train = np.stack([e[0] for e in train])
    y_train = np.asarray([e[1] for e in train])
    X_test = np.stack([e[0] for e in test])
    y_test = np.asarray([e[1] for e in test])

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    classes = np.asarray(model.named_steps["clf"].classes_)
    if 1 not in classes:
        raise SystemExit("Trained model has no positive class")
    positive_index = int(np.where(classes == 1)[0][0])
    proba = model.predict_proba(X_test)[:, positive_index]
    thresholded = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, thresholded)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, thresholded)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "average_precision": round(float(average_precision_score(y_test, proba)), 4),
        "chance_rate": round(float(y_test.mean()), 4),
        "endpoint_distance_baseline_roc_auc": round(
            float(roc_auc_score(y_test, [-e[4] for e in test])), 4
        ),
    }
    split_info = {
        "grouped_by_source_clip": True,
        "test_size": args.test_size,
        "seed": args.seed,
        "train_sources": sorted(train_sources),
        "test_sources": sorted(test_sources),
        "train_samples": len(train),
        "test_samples": len(test),
        "train_positives": len(train_pos),
        "train_negatives": len(train_neg),
        "test_positives": len(test_pos),
        "test_negatives": len(test_neg),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, args.model_out)
    metadata: dict[str, Any] = {
        "model_id": MODEL_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "recorded_join_flow_classification",
        "class_names": [str(c) for c in classes],
        "positive_index": positive_index,
        "feature_names": list(FEATURE_NAMES),
        "feature_description": (
            "Boundary-window pose velocity, travel velocity, foot height, tempo/duration "
            "compatibility, and genuine hand endpoint/velocity continuity. Endpoint pose "
            "distance is intentionally excluded because recorded adjacent units share a frame."
        ),
        "dataset": {
            "catalogue": str(args.catalogue.relative_to(ROOT)).replace("\\", "/"),
            "catalogue_version": catalogue.get("version"),
            "quality_rules_status": catalogue.get("quality_rules_status"),
            "unit_count": len(units),
            "source_count": len(sources),
            "motion_file_count": len({str(u["motion_path"]) for u in units}),
        },
        "pair_mining": {
            "window": args.window,
            "max_pose_distance": args.max_pose_distance,
            "negatives_per_positive": args.negatives_per_positive,
            **counts,
        },
        "split": split_info,
        "metrics": metrics,
        "scope": (
            "Bounded join-ordering prior from recorded motion; not expert choreography supervision, "
            "no frame or hand synthesis, and no claim of correct choreography for arbitrary songs"
        ),
    }
    args.meta_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_payload = {
        "version": 1,
        "model_id": MODEL_ID,
        "created_at": metadata["created_at"],
        "window": args.window,
        "feature_names": list(FEATURE_NAMES),
        "source_count": len(sources),
        "unit_count": len(summaries),
        "units": summaries,
    }
    args.summary_out.write_text(json.dumps(summary_payload, separators=(",", ":")), encoding="utf-8")

    print(f"Saved {args.model_out}, {args.meta_out}, and {args.summary_out}", flush=True)
    print(json.dumps({"split": split_info, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
