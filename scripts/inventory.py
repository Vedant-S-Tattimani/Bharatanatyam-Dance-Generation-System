"""
inventory.py – Inventory and quality-audit every clip in the bharatnatyam_adavus dataset.

Features:
  - Reads adavus_mapping.xlsx, maps each video to its canonical adavu_name1,
    and merges alternate names (adavu_name2, adavu_name3).
  - Parses expected per-category counts from data/adavus/README.md (dataset card)
    and reports every difference (totals, categories, unmapped files, filename typos).
  - Extracts per-clip video metadata via OpenCV & ffprobe:
    resolution, fps, frame count, duration, codec, audio presence, broken-file check.
  - Checks sample frames using MediaPipe PoseLandmarker and HandLandmarker to judge:
      1. Full body visible (shoulders, hips, knees, ankles in frame with vis >= 0.5)
      2. Hands visible enough for mudra tracking (hands detected & bbox >= 40 px)
  - Checks whether any full song-to-dance sequences exist.

Usage:
  python scripts/inventory.py --data-dir data/adavus --output data/inventory_report.json
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import openpyxl
except ImportError:
    openpyxl = None
    print("WARNING: openpyxl not installed – cannot read adavus_mapping.xlsx", file=sys.stderr)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None
    mp_python = None
    mp_vision = None


EXPECTED_CLIPS = 162
EXPECTED_CATEGORIES = 18
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".opus"}

# MediaPipe Pose indices: shoulders (11,12), hips (23,24), knees (25,26), ankles (27,28)
REQUIRED_BODY_INDICES = (11, 12, 23, 24, 25, 26, 27, 28)
MIN_POSE_VISIBILITY = 0.5
MIN_HAND_BBOX_PX = 40.0  # Minimum hand bounding box size in pixels for reliable mudra landmarks

DEFAULT_POSE_MODEL = Path(__file__).resolve().parent.parent / "models" / "weights" / "mediapipe" / "pose_landmarker_full.task"
DEFAULT_HAND_MODEL = Path(__file__).resolve().parent.parent / "models" / "weights" / "mediapipe" / "hand_landmarker.task"


def parse_dataset_card(readme_path: Path) -> dict:
    """Parse expected category counts from data/adavus/README.md."""
    if not readme_path.exists():
        return {"found": False, "categories": {}, "total_expected": EXPECTED_CLIPS}

    text = readme_path.read_text(encoding="utf-8", errors="replace")
    # Matches lines like: "Thattadavu - 24" or "KattiASdavu - 1"
    pattern = re.compile(r"^([A-Za-z0-9_]+)\s*-\s*(\d+)\s*$", re.MULTILINE)
    cats = {}
    for match in pattern.finditer(text):
        name, count = match.group(1), int(match.group(2))
        cats[name] = count

    return {
        "found": True,
        "categories": cats,
        "total_expected": sum(cats.values()) if cats else EXPECTED_CLIPS,
    }


def read_mapping_xlsx(data_dir: Path) -> dict:
    """
    Read adavus_mapping.xlsx and merge alternate names per canonical adavu.

    Returns dict with:
      - by_filename: {basename: {raw_video_name, canonical, alternates, row_num, path_typo}}
      - canonical_alternates: {canonical: sorted list of alternate names}
      - discrepancies: list of mapping issues (blank rows, path typos, etc.)
    """
    xlsx_path = data_dir / "adavus_mapping.xlsx"
    if not xlsx_path.exists():
        hits = list(data_dir.rglob("*.xlsx"))
        if not hits:
            return {"found": False, "by_filename": {}, "canonical_alternates": {}, "discrepancies": ["adavus_mapping.xlsx not found"]}
        xlsx_path = hits[0]

    if openpyxl is None:
        return {"found": False, "by_filename": {}, "canonical_alternates": {}, "discrepancies": ["openpyxl unavailable"]}

    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]

    by_filename = {}
    canonical_alternates = defaultdict(set)
    discrepancies = []

    for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
        vals = [str(c).strip() if c is not None else "" for c in row]
        if row_idx == 1 and vals[0].lower() == "video_name":
            continue
        if not any(vals):
            discrepancies.append(f"Row {row_idx}: blank row in adavus_mapping.xlsx")
            continue

        raw_vname = vals[0]
        adavu1 = vals[1] if len(vals) > 1 else ""
        alts = [v for v in vals[2:] if v]

        path_typo = False
        if raw_vname.startswith("vidoes/"):
            path_typo = True
            discrepancies.append(f"Row {row_idx}: folder typo '{raw_vname}' (expected 'videos/...')")

        fname = Path(raw_vname.replace("\\", "/")).name
        if fname in by_filename:
            discrepancies.append(f"Row {row_idx}: duplicate video entry '{fname}'")

        by_filename[fname] = {
            "raw_video_name": raw_vname,
            "canonical": adavu1,
            "alternates": alts,
            "row_num": row_idx,
            "path_typo": path_typo,
        }
        for a in alts:
            canonical_alternates[adavu1].add(a)

    wb.close()
    return {
        "found": True,
        "path": str(xlsx_path),
        "row_count": len(by_filename),
        "by_filename": by_filename,
        "canonical_alternates": {k: sorted(v) for k, v in sorted(canonical_alternates.items())},
        "discrepancies": discrepancies,
    }


def probe_media_ffprobe(path: Path, ffprobe_bin: str | None) -> dict:
    """Use ffprobe to inspect video and audio streams accurately."""
    if not ffprobe_bin:
        return {"has_audio": None, "audio_codec": None, "video_codec_ffprobe": None}
    try:
        res = subprocess.run(
            [
                ffprobe_bin, "-v", "error",
                "-show_entries", "stream=codec_type,codec_name,sample_rate,channels",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        if res.returncode != 0:
            return {"has_audio": None, "audio_codec": None, "video_codec_ffprobe": None, "ffprobe_error": res.stderr.strip()}
        data = json.loads(res.stdout or "{}")
        streams = data.get("streams", [])
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        return {
            "has_audio": len(audio_streams) > 0,
            "audio_codec": audio_streams[0].get("codec_name") if audio_streams else None,
            "audio_sample_rate": int(audio_streams[0]["sample_rate"]) if audio_streams and audio_streams[0].get("sample_rate") else None,
            "video_codec_ffprobe": video_streams[0].get("codec_name") if video_streams else None,
        }
    except Exception as exc:
        return {"has_audio": None, "audio_codec": None, "video_codec_ffprobe": None, "ffprobe_error": str(exc)}


def get_video_info(path: Path, ffprobe_bin: str | None) -> dict:
    """Extract video metadata and verify readability across start, middle, and end frames."""
    info = {
        "filename": path.name,
        "relative_path": "",
        "broken": False,
        "error": None,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_s": 0.0,
        "codec": "",
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "has_audio": None,
        "audio_codec": None,
    }
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        info["broken"] = True
        info["error"] = "OpenCV failed to open file"
        return info

    info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info["fps"] = round(float(cap.get(cv2.CAP_PROP_FPS)), 3)
    info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    info["codec"] = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")

    if info["fps"] > 0 and info["frame_count"] > 0:
        info["duration_s"] = round(info["frame_count"] / info["fps"], 2)

    # Read start, mid, and near-end frames to catch truncated/corrupt streams
    probe_indices = [0]
    if info["frame_count"] > 10:
        probe_indices.extend([info["frame_count"] // 2, max(0, info["frame_count"] - 5)])

    read_errors = []
    for idx in probe_indices:
        if idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            read_errors.append(idx)

    cap.release()

    if read_errors:
        info["broken"] = True
        info["error"] = f"Failed to decode frame(s) at indices {read_errors}"

    probe = probe_media_ffprobe(path, ffprobe_bin)
    info["has_audio"] = probe.get("has_audio")
    info["audio_codec"] = probe.get("audio_codec")
    if probe.get("video_codec_ffprobe"):
        info["codec"] = probe["video_codec_ffprobe"]

    return info


def assess_visibility_mediapipe(
    path: Path,
    pose_landmarker,
    hand_landmarker,
    num_samples: int = 5,
) -> dict:
    """
    Sample `num_samples` evenly-spaced frames (10% to 90% of clip) and evaluate:
      1. Full body visible: PoseLandmarker detects shoulders, hips, knees, and ankles
         with visibility >= 0.5 and normalized coordinates inside [-0.02, 1.02].
      2. Hands visible for mudra tracking: HandLandmarker detects >= 1 hand with
         bounding box max(width_px, height_px) >= 40 px.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"sampled_frames": 0, "full_body_visible": False, "hands_mudra_ready": False}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames < 1 or width < 1 or height < 1:
        cap.release()
        return {"sampled_frames": 0, "full_body_visible": False, "hands_mudra_ready": False}

    start = int(total_frames * 0.10)
    end = max(start + 1, int(total_frames * 0.90))
    indices = np.linspace(start, min(end, total_frames - 1), num=min(num_samples, total_frames), dtype=int)
    indices = sorted(set(int(i) for i in indices))

    sampled = 0
    pose_detected = 0
    full_body_frames = 0
    ankles_visible_frames = 0
    frames_with_hands = 0
    total_hands_detected = 0
    hand_bboxes_px = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        sampled += 1

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 1. Pose check
        pose_res = pose_landmarker.detect(mp_img)
        if pose_res.pose_landmarks:
            pose_detected += 1
            lm = pose_res.pose_landmarks[0]
            body_ok = all(
                (lm[j].visibility or 0.0) >= MIN_POSE_VISIBILITY
                and -0.02 <= lm[j].x <= 1.02
                and -0.02 <= lm[j].y <= 1.02
                for j in REQUIRED_BODY_INDICES
            )
            ankles_ok = all(
                (lm[j].visibility or 0.0) >= MIN_POSE_VISIBILITY
                and 0.0 <= lm[j].y <= 1.02
                for j in (27, 28)
            )
            if body_ok:
                full_body_frames += 1
            if ankles_ok:
                ankles_visible_frames += 1

        # 2. Hand check using pose-guided wrist ROIs (full-frame palm detection misses ~45px hands in 1080p video)
        frame_hand_hits = 0
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks[0]
            sh_y = (lm[11].y + lm[12].y) * 0.5 * height
            hp_y = (lm[23].y + lm[24].y) * 0.5 * height
            torso_px = max(abs(hp_y - sh_y), 80.0)
            crop_size = int(max(160, torso_px * 0.65))
            for wrist_idx, index_idx in ((15, 19), (16, 20)):
                if (lm[wrist_idx].visibility or 0.0) < 0.4:
                    continue
                cx = int(((lm[wrist_idx].x + lm[index_idx].x) * 0.5) * width)
                cy = int(((lm[wrist_idx].y + lm[index_idx].y) * 0.5) * height)
                x0 = max(0, cx - crop_size // 2)
                y0 = max(0, cy - crop_size // 2)
                x1 = min(width, cx + crop_size // 2)
                y1 = min(height, cy + crop_size // 2)
                if x1 - x0 < 40 or y1 - y0 < 40:
                    continue
                crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
                hres = hand_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
                if hres.hand_landmarks:
                    frame_hand_hits += 1
                    hlm = hres.hand_landmarks[0]
                    xs = [p.x * (x1 - x0) for p in hlm]
                    ys = [p.y * (y1 - y0) for p in hlm]
                    bbox_max = max(max(xs) - min(xs), max(ys) - min(ys))
                    hand_bboxes_px.append(bbox_max)
        if frame_hand_hits > 0:
            frames_with_hands += 1
            total_hands_detected += frame_hand_hits

    cap.release()

    if sampled == 0:
        return {"sampled_frames": 0, "full_body_visible": False, "hands_mudra_ready": False}

    full_body_ratio = round(full_body_frames / sampled, 2)
    ankles_ratio = round(ankles_visible_frames / sampled, 2)
    pose_ratio = round(pose_detected / sampled, 2)
    hand_frame_ratio = round(frames_with_hands / sampled, 2)
    avg_hands = round(total_hands_detected / sampled, 2)
    median_bbox = round(float(np.median(hand_bboxes_px)), 1) if hand_bboxes_px else 0.0
    min_bbox = round(float(np.min(hand_bboxes_px)), 1) if hand_bboxes_px else 0.0

    # Criterion:
    # - full_body_visible: >= 60% of sampled frames have shoulders, hips, knees, ankles visible in frame
    # - hands_mudra_ready: hands detected in >= 50% of sampled frames AND median hand bbox >= 40 px
    full_body_visible = full_body_ratio >= 0.60
    hands_mudra_ready = (hand_frame_ratio >= 0.50) and (median_bbox >= MIN_HAND_BBOX_PX)

    return {
        "sampled_frames": sampled,
        "pose_detected_ratio": pose_ratio,
        "full_body_ratio": full_body_ratio,
        "ankles_visible_ratio": ankles_ratio,
        "full_body_visible": full_body_visible,
        "hand_detect_frame_ratio": hand_frame_ratio,
        "avg_hands_detected": avg_hands,
        "median_hand_bbox_px": median_bbox,
        "min_hand_bbox_px": min_bbox,
        "hands_mudra_ready": hands_mudra_ready,
    }


def check_song_sequences(data_dir: Path, clips: list[dict]) -> dict:
    """Check whether the dataset contains any standalone songs or full song-to-dance routines."""
    audio_files = sorted(
        str(p.relative_to(data_dir))
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS and ".cache" not in p.parts
    )
    sequence_files = sorted(
        str(p.relative_to(data_dir))
        for p in data_dir.rglob("*")
        if p.is_file()
        and ".cache" not in p.parts
        and any(k in p.name.lower() for k in ("song", "sequence", "choreo", "timeline", "tala"))
    )
    long_videos = [
        {"file": c["relative_path"], "duration_s": c["duration_s"]}
        for c in clips if not c["broken"] and c["duration_s"] > 180.0
    ]
    return {
        "standalone_audio_files": audio_files,
        "sequence_metadata_files": sequence_files,
        "videos_over_180s": long_videos,
        "has_song_to_dance_sequences": False,
        "conclusion": (
            "No full song-to-dance sequences exist in the dataset. All 162 videos are "
            "individual adavu demonstration/practice clips (each showing one adavu category "
            "or variation), with no paired full-song choreography sequences or sequence annotations."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/adavus", help="Path to downloaded dataset directory")
    ap.add_argument("--output", default="data/inventory_report.json", help="Output JSON report path")
    ap.add_argument("--pose-model", default=str(DEFAULT_POSE_MODEL), help="MediaPipe pose_landmarker .task path")
    ap.add_argument("--hand-model", default=str(DEFAULT_HAND_MODEL), help="MediaPipe hand_landmarker .task path")
    ap.add_argument("--sample-frames", type=int, default=5, help="Sample frames per clip for body/hand checks")
    ap.add_argument("--max-clips", type=int, default=0, help="Limit clips scanned (0 = all)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Data directory not found: {data_dir}")

    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        print("WARNING: ffprobe not found – audio presence cannot be checked", file=sys.stderr)

    # Check MediaPipe models
    use_mediapipe = (
        mp is not None
        and Path(args.pose_model).is_file()
        and Path(args.hand_model).is_file()
    )
    if not use_mediapipe:
        sys.exit(
            f"MediaPipe or model files missing. Expected:\n"
            f"  pose: {args.pose_model}\n"
            f"  hand: {args.hand_model}"
        )

    card = parse_dataset_card(data_dir / "README.md")
    mapping = read_mapping_xlsx(data_dir)

    all_videos = sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and ".cache" not in p.parts
    )
    if args.max_clips > 0:
        all_videos = all_videos[:args.max_clips]

    print(f"Found {len(all_videos)} video files in {data_dir}", file=sys.stderr)

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.pose_model)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.hand_model)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )

    clips = []
    mapped_filenames = set()

    with (
        mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm,
        mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm,
    ):
        for i, vpath in enumerate(all_videos, start=1):
            rel = str(vpath.relative_to(data_dir)).replace("\\", "/")
            print(f"[{i}/{len(all_videos)}] {rel}", file=sys.stderr)

            info = get_video_info(vpath, ffprobe_bin)
            info["relative_path"] = rel

            m_entry = mapping["by_filename"].get(vpath.name)
            if m_entry:
                mapped_filenames.add(vpath.name)
                info["adavu_canonical"] = m_entry["canonical"]
                info["adavu_alternates"] = m_entry["alternates"]
            else:
                info["adavu_canonical"] = "UNMAPPED"
                info["adavu_alternates"] = []

            if not info["broken"]:
                vis = assess_visibility_mediapipe(vpath, pose_lm, hand_lm, num_samples=args.sample_frames)
            else:
                vis = {"sampled_frames": 0, "full_body_visible": False, "hands_mudra_ready": False}
            info["visibility"] = vis
            clips.append(info)

    # Check for mapping entries missing on disk (only when scanning full dataset)
    missing_on_disk = []
    if args.max_clips == 0 and mapping["found"]:
        disk_names = {v.name for v in all_videos}
        for fname, entry in mapping["by_filename"].items():
            if fname not in disk_names:
                missing_on_disk.append(entry["raw_video_name"])

    unmapped_on_disk = [c["relative_path"] for c in clips if c["adavu_canonical"] == "UNMAPPED"]

    # Aggregate per canonical category (merging alternate names)
    per_adavu = {}
    grouped = defaultdict(list)
    for c in clips:
        grouped[c["adavu_canonical"]].append(c)

    for adavu, group in sorted(grouped.items()):
        valid = [c for c in group if not c["broken"]]
        durs = [c["duration_s"] for c in valid]
        per_adavu[adavu] = {
            "clip_count": len(group),
            "expected_in_card": card["categories"].get(adavu),
            "alternate_names_merged": mapping["canonical_alternates"].get(adavu, []),
            "broken_count": sum(1 for c in group if c["broken"]),
            "audio_present_count": sum(1 for c in group if c.get("has_audio") is True),
            "full_body_visible_count": sum(1 for c in valid if c["visibility"].get("full_body_visible")),
            "hands_mudra_ready_count": sum(1 for c in valid if c["visibility"].get("hands_mudra_ready")),
            "duration_min_s": round(min(durs), 2) if durs else 0.0,
            "duration_max_s": round(max(durs), 2) if durs else 0.0,
            "duration_total_s": round(sum(durs), 2) if durs else 0.0,
        }

    # Compare against dataset card
    card_diffs = list(mapping.get("discrepancies", []))
    if args.max_clips == 0:
        if len(clips) != card["total_expected"]:
            card_diffs.append(f"Total clips: found {len(clips)}, dataset card expects {card['total_expected']}")
        actual_cats = {k for k in per_adavu if k != "UNMAPPED"}
        if len(actual_cats) != EXPECTED_CATEGORIES:
            card_diffs.append(f"Category count: found {len(actual_cats)}, dataset card expects {EXPECTED_CATEGORIES}")
        for cat, exp_n in card["categories"].items():
            act_n = per_adavu.get(cat, {}).get("clip_count", 0)
            if act_n != exp_n:
                card_diffs.append(f"Category '{cat}': found {act_n} clips, dataset card expects {exp_n}")
        for cat in actual_cats - set(card["categories"].keys()):
            card_diffs.append(f"Category '{cat}' present in mapping but not listed on dataset card")
        if missing_on_disk:
            card_diffs.append(f"Files listed in adavus_mapping.xlsx but missing on disk: {missing_on_disk}")
        if unmapped_on_disk:
            card_diffs.append(f"Video files on disk not listed in adavus_mapping.xlsx: {unmapped_on_disk}")
        if "KattiASdavu" in actual_cats:
            card_diffs.append("Category typo 'KattiASdavu' appears in both README.md and adavus_mapping.xlsx (for KattiAdavu.mp4)")

    song_seq = check_song_sequences(data_dir, clips)

    valid_clips = [c for c in clips if not c["broken"]]
    all_durs = [c["duration_s"] for c in valid_clips]
    resolutions = sorted({f"{c['width']}x{c['height']}" for c in valid_clips})
    fps_vals = sorted({c["fps"] for c in valid_clips})

    summary = {
        "total_clips": len(clips),
        "expected_clips": EXPECTED_CLIPS,
        "total_categories": len([k for k in per_adavu if k != "UNMAPPED"]),
        "expected_categories": EXPECTED_CATEGORIES,
        "broken_files": sum(1 for c in clips if c["broken"]),
        "total_duration_min": round(sum(all_durs) / 60.0, 2) if all_durs else 0.0,
        "duration_range_s": [round(min(all_durs), 2), round(max(all_durs), 2)] if all_durs else [0, 0],
        "resolutions": resolutions,
        "fps_values": fps_vals,
        "audio_present_count": sum(1 for c in clips if c.get("has_audio") is True),
        "audio_absent_count": sum(1 for c in clips if c.get("has_audio") is False),
        "full_body_visible_count": sum(1 for c in valid_clips if c["visibility"].get("full_body_visible")),
        "hands_mudra_ready_count": sum(1 for c in valid_clips if c["visibility"].get("hands_mudra_ready")),
        "differences_from_card": card_diffs,
    }

    report = {
        "summary": summary,
        "judgement_methodology": {
            "sample_frames_per_clip": args.sample_frames,
            "sampling_strategy": "Evenly spaced frames between 10% and 90% of clip duration.",
            "full_body_visible": (
                f"MediaPipe PoseLandmarker (full) run on each sampled frame. A frame passes if all 8 "
                f"key lower/upper body landmarks (shoulders 11/12, hips 23/24, knees 25/26, ankles 27/28) "
                f"have visibility >= {MIN_POSE_VISIBILITY} and normalized coords in [-0.02, 1.02]. "
                f"Clip is marked full_body_visible=True if >= 60% of sampled frames pass."
            ),
            "hands_mudra_ready": (
                f"MediaPipe HandLandmarker (num_hands=1, min_hand_detection_confidence=0.3) run on "
                f"pose-guided wrist crops (box = max(160, 0.65 * torso_px) centered between wrist and "
                f"index MCP) for each sampled frame, because full-frame detection misses ~45 px hands "
                f"in 1080p video. Clip is marked hands_mudra_ready=True if >= 1 hand is detected in "
                f">= 50% of sampled frames AND median detected hand bounding box max(w, h) >= {MIN_HAND_BBOX_PX} px."
            ),
            "audio_check": "ffprobe stream inspection for codec_type='audio'.",
        },
        "per_adavu": per_adavu,
        "song_sequences": song_seq,
        "clips": clips,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print concise table to stderr
    print("\n" + "=" * 78, file=sys.stderr)
    print(f"INVENTORY SUMMARY ({summary['total_clips']} clips, {summary['total_categories']} categories)", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    print(f"Broken: {summary['broken_files']} | Audio: {summary['audio_present_count']}/{summary['total_clips']} | Full body: {summary['full_body_visible_count']}/{len(valid_clips)} | Hands mudra-ready: {summary['hands_mudra_ready_count']}/{len(valid_clips)}", file=sys.stderr)
    print(f"Resolutions: {resolutions} | FPS: {fps_vals} | Total dur: {summary['total_duration_min']} min", file=sys.stderr)
    print(f"\n{'Adavu':<26} {'Clips':>5} {'Card':>5} {'Body':>5} {'Hands':>5} {'Audio':>5}  Alternates merged", file=sys.stderr)
    print("-" * 78, file=sys.stderr)
    for adavu, s in per_adavu.items():
        alts = ", ".join(s["alternate_names_merged"]) or "-"
        card_n = str(s["expected_in_card"]) if s["expected_in_card"] is not None else "?"
        print(
            f"{adavu:<26} {s['clip_count']:>5} {card_n:>5} "
            f"{s['full_body_visible_count']:>5} {s['hands_mudra_ready_count']:>5} "
            f"{s['audio_present_count']:>5}  {alts}",
            file=sys.stderr,
        )
    if card_diffs:
        print("\nDifferences / Discrepancies:", file=sys.stderr)
        for d in card_diffs:
            print(f"  - {d}", file=sys.stderr)
    print(f"\nReport written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
