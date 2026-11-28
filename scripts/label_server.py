"""
label_server.py – Local web server for marking ground-truth foot-strike frames
and beats on the 6 focus clips without modifying frontend/.

Usage:
  python scripts/label_server.py --port 8765
Then open: http://127.0.0.1:8765/focus_6_labeler.html
Saved files go to data/labels/<stem>_gt.json (and verified beats sync to *_beats_review.json).
"""

from __future__ import annotations

import argparse
import functools
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREVIEWS_DIR = ROOT / "data" / "previews"
LABELS_DIR = ROOT / "data" / "labels"
SIX_FOCUS_STEMS = [
    "Nattadavu2.8",
    "Nattadavu3.6",
    "Nattadavu2.1",
    "KudithuMettuAdavu1",
    "ThaThaiTham1",
    "ThaiyaThaihi1",
]


def _load_clip_payload(stem: str) -> dict:
    if stem not in SIX_FOCUS_STEMS:
        raise KeyError(stem)
    m = json.loads((ROOT / "data" / "motion" / f"{stem}.json").read_text(encoding="utf-8"))
    a = json.loads((ROOT / "data" / "audio" / f"{stem}_analysis.json").read_text(encoding="utf-8"))
    gt_path = LABELS_DIR / f"{stem}_gt.json"
    existing = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.is_file() else None
    return {
        "stem": stem,
        "fps": float(m["fps"]),
        "duration_s": float(m["duration_s"]),
        "preview_song": f"{stem}_preview.mp4",
        "preview_clicks": f"{stem}_clicks_preview.mp4",
        "detected_strikes": m.get("foot_strikes", []),
        "detected_beats_s": [float(b["time_s"]) for b in a.get("beats", [])],
        "true_strikes": existing.get("true_strikes", []) if existing else [],
        "true_beats_s": existing.get("true_beats_s", []) if existing else [],
        "sam_beats_s": existing.get("sam_beats_s", []) if existing else [],
        "labeled_range_s": existing.get("labeled_range_s") if existing else None,
        "notes": existing.get("notes", "") if existing else "",
    }


def _save_clip_payload(stem: str, body: dict) -> Path:
    if stem not in SIX_FOCUS_STEMS:
        raise KeyError(stem)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT / "data" / "motion" / f"{stem}.json").read_text(encoding="utf-8"))
    fps = float(m["fps"])

    true_strikes = sorted(
        [
            {
                "frame": int(st["frame"]),
                "time_s": round(float(st["frame"]) / fps, 4),
                "side": str(st["side"]),
            }
            for st in body.get("true_strikes", [])
            if st.get("side") in ("left", "right")
        ],
        key=lambda x: (x["frame"], x["side"]),
    )
    true_beats = sorted(round(float(t), 4) for t in body.get("true_beats_s", []))
    sam_beats = sorted(round(float(t), 4) for t in body.get("sam_beats_s", []))

    raw_range = body.get("labeled_range_s")
    labeled_range = (
        [round(float(raw_range[0]), 4), round(float(raw_range[1]), 4)]
        if isinstance(raw_range, list) and len(raw_range) == 2 and raw_range[0] is not None and raw_range[1] is not None
        else None
    )

    gt_obj = {
        "stem": stem,
        "fps": fps,
        "labeled_range_s": labeled_range,
        "true_strikes": true_strikes,
        "true_beats_s": true_beats,
        "sam_beats_s": sam_beats,
        "notes": str(body.get("notes", "")),
    }
    gt_path = LABELS_DIR / f"{stem}_gt.json"
    gt_path.write_text(json.dumps(gt_obj, indent=2), encoding="utf-8")

    # Sync user-marked true beats into *_beats_review.json when provided
    rpath = ROOT / "data" / "audio" / f"{stem}_beats_review.json"
    if rpath.is_file() and true_beats:
        rev = json.loads(rpath.read_text(encoding="utf-8"))
        rev["verified_beats_s"] = true_beats
        rpath.write_text(json.dumps(rev, indent=2), encoding="utf-8")
    return gt_path


class LabelHandler(SimpleHTTPRequestHandler):
    """Static preview handler + /api/gt/<stem> persistence endpoints."""

    def _send_json(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/api/gt":
            self._send_json(200, {"stems": SIX_FOCUS_STEMS})
            return
        if self.path.startswith("/api/gt/"):
            stem = self.path.split("/api/gt/", 1)[1].strip("/")
            try:
                self._send_json(200, _load_clip_payload(stem))
            except KeyError:
                self._send_json(404, {"error": f"Unknown stem: {stem}"})
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith("/api/gt/"):
            stem = self.path.split("/api/gt/", 1)[1].strip("/")
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            try:
                saved = _save_clip_payload(stem, body)
                self._send_json(200, {"saved": str(saved.relative_to(ROOT)).replace("\\", "/")})
            except KeyError:
                self._send_json(404, {"error": f"Unknown stem: {stem}"})
            return
        self._send_json(404, {"error": "Not found"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    handler = functools.partial(LabelHandler, directory=str(PREVIEWS_DIR))
    srv = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving focus_6_labeler at http://{args.host}:{args.port}/focus_6_labeler.html")
    srv.serve_forever()


if __name__ == "__main__":
    main()

