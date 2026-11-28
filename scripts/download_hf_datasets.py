"""Download licensed Hugging Face datasets used by Nritya's bounded training step.

All bytes land under gitignored ``data/hf_datasets/`` on the same drive as the
repository. A local manifest records the pinned revision and license so a model
can always be traced back to its source.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "data" / "hf_datasets"

DATASETS = {
    "mudra": {
        "repo_id": "Samarth0710/bharatanatyam-mudra-dataset",
        "revision": "b3df2f983ef58d25e72d6191d1c4bddbbbbe96ab",
        "license": "MIT",
        "local_dir": "Samarth0710__bharatanatyam-mudra-dataset",
        "allow_patterns": ["README.md", "data/*.parquet"],
        "purpose": "Single- and double-hand Bharatanatyam mudra images for landmark classifier training.",
    },
}


def _summarise(local_dir: Path) -> dict[str, int]:
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    return {"file_count": len(files), "total_bytes": sum(p.stat().st_size for p in files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(DATASETS), help="Download just one configured dataset")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest.setdefault("datasets", {})

    selected = [args.only] if args.only else sorted(DATASETS)
    for key in selected:
        cfg = DATASETS[key]
        local_dir = args.dest / cfg["local_dir"]
        print(f"Downloading {cfg['repo_id']} @ {cfg['revision'][:12]} -> {local_dir}", flush=True)
        path = Path(snapshot_download(
            repo_id=cfg["repo_id"],
            repo_type="dataset",
            revision=cfg["revision"],
            local_dir=str(local_dir),
            allow_patterns=cfg["allow_patterns"],
        ))
        manifest["datasets"][key] = {
            **cfg,
            "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            **_summarise(path),
        }
        print(f"  {manifest['datasets'][key]['file_count']} files, "
              f"{manifest['datasets'][key]['total_bytes']:,} bytes", flush=True)

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
