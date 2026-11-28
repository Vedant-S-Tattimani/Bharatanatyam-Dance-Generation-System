"""
diagnose_pilot.py – Comprehensive validation, Demucs repair, missing preview rendering,
foot-strike timing diagnosis, 6-clip timeline plots, beat-verifier HTML, and quality-rule
filtering for the 38-clip Phase 1 pilot.

All outputs are written under data/ (gitignored).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.audio.analyzer import (  # noqa: E402
    analyze_audio,
    apply_tempo_scale,
    write_beat_review_file,
    write_click_track,
)
from app.motion.extractor import compare_foot_strikes_to_beats  # noqa: E402
from app.motion.preview import render_side_by_side_preview  # noqa: E402
from run_pilot_phase1 import select_pilot_clips  # noqa: E402

SIX_DIAG_STEMS = [
    "Nattadavu2.8",
    "Nattadavu3.6",
    "Nattadavu2.1",
    "KudithuMettuAdavu1",
    "ThaThaiTham1",
    "ThaiyaThaihi1",
]


def check_av_offset_ms(video_path: Path, ffprobe_bin: str | None) -> float | None:
    if not ffprobe_bin or not video_path.is_file():
        return None
    pr = subprocess.run(
        [ffprobe_bin, "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json", str(video_path)],
        capture_output=True, text=True, timeout=15,
    )
    if pr.returncode != 0:
        return None
    streams = json.loads(pr.stdout or "{}").get("streams", [])
    v_st = next((float(s.get("start_time", 0.0)) for s in streams if s.get("codec_type") == "video"), 0.0)
    a_st = next((float(s.get("start_time", 0.0)) for s in streams if s.get("codec_type") == "audio"), 0.0)
    return round((a_st - v_st) * 1000.0, 3)


def validate_all_outputs(selected: list[dict]) -> tuple[list[str], list[str]]:
    """Validate all 38 motion JSONs, audio JSONs, review JSONs, and click WAVs."""
    errors: list[str] = []
    hpss_stems: list[str] = []
    for c in selected:
        stem = Path(c["filename"]).stem
        ap = ROOT / "data" / "audio" / f"{stem}_analysis.json"
        rp = ROOT / "data" / "audio" / f"{stem}_beats_review.json"
        cw = ROOT / "data" / "audio" / f"{stem}_clicks.wav"
        mp = ROOT / "data" / "motion" / f"{stem}.json"

        for p in (ap, rp, cw, mp):
            if not p.is_file() or p.stat().st_size == 0:
                errors.append(f"Missing or empty: {p}")

        if ap.is_file():
            try:
                a = json.loads(ap.read_text(encoding="utf-8"))
                if a.get("beat_count", 0) <= 0 or len(a.get("beats", [])) != a.get("beat_count"):
                    errors.append(f"Beat count mismatch in {ap.name}")
                if a.get("separation_backend") != "demucs":
                    hpss_stems.append(stem)
            except Exception as exc:
                errors.append(f"Corrupt audio JSON {ap.name}: {exc}")

        if mp.is_file():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                frames = m.get("frames", [])
                if not frames or len(frames) != m.get("n_frames"):
                    errors.append(f"Motion frame count mismatch in {mp.name}")
                for fr in frames:
                    if fr.get("valid") and fr.get("pose_world") is None:
                        errors.append(f"Valid frame missing pose_world in {mp.name} frame {fr.get('f')}")
                        break
            except Exception as exc:
                errors.append(f"Corrupt motion JSON {mp.name}: {exc}")

    return errors, hpss_stems


def rerun_demucs_for_hpss(selected: list[dict], hpss_stems: list[str], timeout_s: int = 900) -> list[dict]:
    """Re-run Demucs one clip at a time with longer timeout on HPSS fallback clips."""
    by_stem = {Path(c["filename"]).stem: c for c in selected}
    results = []
    for stem in hpss_stems:
        clip = by_stem[stem]
        vpath = ROOT / "data" / "adavus" / clip["relative_path"]
        print(f"[Demucs retry] {stem} (timeout={timeout_s}s, single-clip)...", file=sys.stderr, flush=True)
        analysis, y_mix, sr = analyze_audio(vpath, tala_name="adi", use_demucs=True, demucs_timeout=timeout_s)

        audio_dir = ROOT / "data" / "audio"
        (audio_dir / f"{stem}_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        write_click_track(y_mix, sr, analysis["beats"], audio_dir / f"{stem}_clicks.wav")
        write_beat_review_file(analysis, audio_dir / f"{stem}_beats_review.json")

        # Refresh foot_strikes_vs_beats in motion JSON without re-running MediaPipe
        mpath = ROOT / "data" / "motion" / f"{stem}.json"
        motion = json.loads(mpath.read_text(encoding="utf-8"))
        beat_times = [b["time_s"] for b in analysis["beats"]]
        motion["foot_strikes_vs_beats"] = compare_foot_strikes_to_beats(
            motion.get("foot_strikes", []), beat_times, fps=float(motion["fps"])
        )
        mpath.write_text(json.dumps(motion, separators=(",", ":")), encoding="utf-8")

        # Invalidate old click preview if clicks changed so preview step re-renders it
        cprev = ROOT / "data" / "previews" / f"{stem}_clicks_preview.mp4"
        if cprev.is_file():
            cprev.unlink()

        results.append({
            "stem": stem,
            "separation_backend": analysis["separation_backend"],
            "separation_detail": analysis["separation_detail"],
            "bpm_median": analysis["bpm_median"],
            "beat_count": analysis["beat_count"],
        })
        print(f"  -> {stem}: backend={analysis['separation_backend']}, bpm={analysis['bpm_median']}, beats={analysis['beat_count']}", file=sys.stderr, flush=True)
    return results


def _render_preview_worker(task: dict) -> dict:
    if str(ROOT / "backend") not in sys.path:
        sys.path.insert(0, str(ROOT / "backend"))
    from app.motion.preview import render_side_by_side_preview

    vpath = Path(task["video_path"])
    stem = vpath.stem
    motion = json.loads((ROOT / "data" / "motion" / f"{stem}.json").read_text(encoding="utf-8"))
    analysis = json.loads((ROOT / "data" / "audio" / f"{stem}_analysis.json").read_text(encoding="utf-8"))
    click_wav = ROOT / "data" / "audio" / f"{stem}_clicks.wav"
    prev_mp4 = ROOT / "data" / "previews" / f"{stem}_preview.mp4"
    cprev_mp4 = ROOT / "data" / "previews" / f"{stem}_clicks_preview.mp4"

    did = []
    if not prev_mp4.is_file() or prev_mp4.stat().st_size == 0:
        render_side_by_side_preview(vpath, motion, prev_mp4, audio_wav=None, beats=analysis["beats"])
        did.append("preview")
    if not cprev_mp4.is_file() or cprev_mp4.stat().st_size == 0:
        render_side_by_side_preview(vpath, motion, cprev_mp4, audio_wav=click_wav, beats=analysis["beats"])
        did.append("clicks_preview")

    if did:
        print(f"[Preview {task['idx']}/{task['total']}] {stem}: rendered {', '.join(did)}", file=sys.stderr, flush=True)
    return {"stem": stem, "rendered": did}


def render_all_missing_previews(selected: list[dict], workers: int = 4) -> None:
    tasks = [
        {"idx": i, "total": len(selected), "video_path": str(ROOT / "data" / "adavus" / c["relative_path"])}
        for i, c in enumerate(selected, 1)
    ]
    if workers <= 1:
        for t in tasks:
            _render_preview_worker(t)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_render_preview_worker, tasks))


def _load_clip_beats(stem: str, analysis: dict) -> tuple[list[float], str]:
    """Use verified beats from *_beats_review.json when present, otherwise unverified detected beats."""
    rpath = ROOT / "data" / "audio" / f"{stem}_beats_review.json"
    if rpath.is_file():
        rev = json.loads(rpath.read_text(encoding="utf-8"))
        if rev.get("verified_beats_s"):
            return [float(t) for t in rev["verified_beats_s"]], "verified"
        scale = rev.get("verified_tempo_scale")
        offset = rev.get("verified_offset_ms")
        if (scale is not None and float(scale) != 1.0) or (offset is not None and float(offset) != 0.0):
            base = np.array([float(b["time_s"]) for b in analysis["beats"]], dtype=float)
            shifted = apply_tempo_scale(base, tempo_scale=float(scale or 1.0), offset_ms=float(offset or 0.0))
            return [float(t) for t in shifted], "verified"
    return [float(b["time_s"]) for b in analysis["beats"]], "unverified"


def signed_errors_and_random_baseline(
    strike_times: list[float],
    beat_times: list[float],
    rng: np.random.Generator,
) -> dict:
    """Compute signed nearest-beat errors (strike - beat) and same-tempo empirical random baseline."""
    if not strike_times or not beat_times:
        return {
            "median_signed_ms": None,
            "mean_signed_ms": None,
            "early_pct": None,
            "late_pct": None,
            "random_baseline_median_ms": None,
        }
    st = np.asarray(strike_times, dtype=float)
    bt = np.asarray(beat_times, dtype=float)
    signed_ms = []
    for t in st:
        idx = int(np.argmin(np.abs(bt - t)))
        signed_ms.append(float((t - bt[idx]) * 1000.0))
    signed_arr = np.asarray(signed_ms, dtype=float)

    # Empirical uniform-random baseline across the SAME beat track (200 Monte-Carlo trials)
    t0, t1 = float(bt[0]), float(bt[-1])
    rand_medians = []
    if t1 > t0:
        for _ in range(200):
            r_t = rng.uniform(t0, t1, size=len(st))
            dists = [float(np.min(np.abs(bt - rt)) * 1000.0) for rt in r_t]
            rand_medians.append(float(np.median(dists)))

    return {
        "median_signed_ms": round(float(np.median(signed_arr)), 2),
        "mean_signed_ms": round(float(np.mean(signed_arr)), 2),
        "early_pct": round(float(np.mean(signed_arr < 0) * 100.0), 1),
        "late_pct": round(float(np.mean(signed_arr > 0) * 100.0), 1),
        "random_baseline_median_ms": round(float(np.mean(rand_medians)), 2) if rand_medians else None,
    }


def extract_contiguous_clean_segments(motion: dict, min_dur_s: float = 2.0) -> list[dict]:
    """Split a clip's frames at any invalid/long-gap frames and return clean segments >= min_dur_s."""
    frames = motion["frames"]
    fps = float(motion["fps"])
    min_len = int(round(min_dur_s * fps))
    segs = []
    i = 0
    n = len(frames)
    while i < n:
        if not frames[i]["valid"]:
            i += 1
            continue
        s = i
        while i < n and frames[i]["valid"]:
            i += 1
        length = i - s
        if length >= min_len:
            seg_frames = frames[s:i]
            lh_ratio = float(np.mean([fr["left_hand_img"] is not None for fr in seg_frames]))
            rh_ratio = float(np.mean([fr["right_hand_img"] is not None for fr in seg_frames]))
            segs.append({
                "start_frame": s,
                "end_frame": i - 1,
                "duration_s": round(length / fps, 2),
                "left_hand_ratio": round(lh_ratio, 3),
                "right_hand_ratio": round(rh_ratio, 3),
            })
    return segs


def plot_six_timelines(diag_rows: list[dict], out_png: Path) -> None:
    """Plot beats and foot strikes on one timeline per clip for the 6 selected clips."""
    by_stem = {r["stem"]: r for r in diag_rows}
    fig, axes = plt.subplots(len(SIX_DIAG_STEMS), 1, figsize=(15, 13), sharex=False)
    for ax, stem in zip(axes, SIX_DIAG_STEMS):
        r = by_stem[stem]
        a = json.loads((ROOT / "data" / "audio" / f"{stem}_analysis.json").read_text(encoding="utf-8"))
        m = json.loads((ROOT / "data" / "motion" / f"{stem}.json").read_text(encoding="utf-8"))
        beats = np.array([b["time_s"] for b in a["beats"]], dtype=float)
        sam_beats = np.array([b["time_s"] for b in a["beats"] if b["is_sam"]], dtype=float)
        strikes = m.get("foot_strikes", [])
        l_strikes = np.array([s["time_s"] for s in strikes if s["side"] == "left"], dtype=float)
        r_strikes = np.array([s["time_s"] for s in strikes if s["side"] == "right"], dtype=float)

        # Show first 45 seconds so beats & strikes are crisp and readable
        t_max = min(45.0, float(m["duration_s"]))
        b_sub = beats[beats <= t_max]
        sam_sub = sam_beats[sam_beats <= t_max]
        l_sub = l_strikes[l_strikes <= t_max]
        r_sub = r_strikes[r_strikes <= t_max]

        for t in b_sub:
            ax.axvline(t, color="#9aa4b2", linewidth=0.9, alpha=0.6)
        for t in sam_sub:
            ax.axvline(t, color="#e63946", linewidth=1.6, alpha=0.85, linestyle="--")
        if len(l_sub):
            ax.scatter(l_sub, np.full_like(l_sub, 0.65), color="#1d3557", marker="v", s=55, label="Left strike" if stem == SIX_DIAG_STEMS[0] else "")
        if len(r_sub):
            ax.scatter(r_sub, np.full_like(r_sub, 0.35), color="#2a9d8f", marker="^", s=55, label="Right strike" if stem == SIX_DIAG_STEMS[0] else "")

        ax.set_xlim(0, t_max)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.35, 0.65])
        ax.set_yticklabels(["R", "L"], fontsize=8)
        frac_s = f"{r['err_frac_of_period']:.2f}T" if r["err_frac_of_period"] is not None else "n/a"
        rand_s = f"{r['rand_analytic_ms']:.0f}ms" if r["rand_analytic_ms"] is not None else "n/a"
        ax.set_title(
            f"{stem} ({r['adavu']}) | BPM={r['bpm_median']} (T={r['beat_period_ms']:.0f}ms) | "
            f"med_err={r['median_abs_ms']}ms ({frac_s}, rand~{rand_s}) | "
            f"signed={r['median_signed_ms']}ms | early={r['early_pct']}% | strikes={r['strike_count']}",
            fontsize=9.5, loc="left",
        )
        if stem == SIX_DIAG_STEMS[0]:
            ax.legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time (seconds, first 45s window)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=140)
    plt.close(fig)


def save_strike_check_montages(out_dir: Path) -> None:
    """Save 4-frame contact sheets of detected foot strikes vs raw frames for the 6 diagnostic clips."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in SIX_DIAG_STEMS:
        m = json.loads((ROOT / "data" / "motion" / f"{stem}.json").read_text(encoding="utf-8"))
        a = json.loads((ROOT / "data" / "audio" / f"{stem}_analysis.json").read_text(encoding="utf-8"))
        vpath = ROOT / "data" / "adavus" / "videos" / m["video"]
        beats = np.array([b["time_s"] for b in a["beats"]], dtype=float)
        strikes = m.get("foot_strikes", [])[:4]
        if not strikes:
            continue
        cap = cv2.VideoCapture(str(vpath))
        panels = []
        for s in strikes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame"])
            ok, bgr = cap.read()
            if not ok:
                continue
            bgr = cv2.resize(bgr, (480, 270))
            nearest = float(beats[np.argmin(np.abs(beats - s["time_s"]))]) if len(beats) else 0.0
            diff_ms = (s["time_s"] - nearest) * 1000.0
            cv2.rectangle(bgr, (0, 0), (480, 34), (15, 15, 15), -1)
            cv2.putText(
                bgr,
                f"{s['side']} f={s['frame']} t={s['time_s']:.2f}s (beat_dt={diff_ms:+.0f}ms)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            panels.append(bgr)
        cap.release()
        if panels:
            grid = np.hstack(panels)
            cv2.imwrite(str(out_dir / f"{stem}_strikes.png"), grid)


def build_beat_verifier_html(out_html: Path) -> None:
    """Write a zero-dependency local HTML tool for ear-checking & exporting beat corrections on the 6 clips."""
    clips_data = []
    for stem in SIX_DIAG_STEMS:
        a = json.loads((ROOT / "data" / "audio" / f"{stem}_analysis.json").read_text(encoding="utf-8"))
        clips_data.append({
            "stem": stem,
            "preview": f"{stem}_clicks_preview.mp4",
            "bpm": a["bpm_median"],
            "sam": a["first_sam_index"],
            "tala_beats": a["tala_beats"],
            "beats": [b["time_s"] for b in a["beats"]],
        })
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Nritya — 6-Clip Beat Verification Helper</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 20px; background: #111; color: #eee; }}
  select, input, button {{ padding: 6px 10px; margin: 4px; font-size: 14px; }}
  video {{ width: 100%; max-width: 1120px; background: #000; border-radius: 6px; display: block; margin: 12px 0; }}
  pre {{ background: #1d1d1d; padding: 12px; border-radius: 6px; overflow-x: auto; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
</style>
</head>
<body>
<h2>6-Clip Beat Verification Helper (with Click-Track Audio)</h2>
<div class="row">
  <label>Clip: <select id="clipSel"></select></label>
  <label>Tempo scale:
    <select id="scaleSel">
      <option value="1.0">1.0x (as detected)</option>
      <option value="0.5">0.5x (half tempo)</option>
      <option value="2.0">2.0x (double tempo)</option>
    </select>
  </label>
  <label>First Sam Beat (0-7): <input id="samInp" type="number" min="0" max="7" value="0" style="width:60px"></label>
  <label>Offset (ms): <input id="offInp" type="number" step="10" value="0" style="width:80px"></label>
  <button id="tapBtn">Tap Beat (Spacebar)</button>
  <button id="clearTapBtn">Clear Taps</button>
</div>
<video id="vid" controls></video>
<div><b>Tapped beats (<span id="tapCount">0</span>):</b> <span id="tapPreview">[]</span></div>
<h3>Run this command to save verification &amp; compute Measure 4a:</h3>
<pre id="cmdOut"></pre>
<script>
const CLIPS = {json.dumps(clips_data)};
const clipSel = document.getElementById('clipSel');
const scaleSel = document.getElementById('scaleSel');
const samInp = document.getElementById('samInp');
const offInp = document.getElementById('offInp');
const vid = document.getElementById('vid');
const cmdOut = document.getElementById('cmdOut');
const tapPreview = document.getElementById('tapPreview');
const tapCount = document.getElementById('tapCount');
let taps = [];

CLIPS.forEach((c, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `${{c.stem}} (BPM=${{c.bpm}}, sam=${{c.sam}})`;
  clipSel.appendChild(opt);
}});

function refresh() {{
  const c = CLIPS[clipSel.value];
  let cmd = `python scripts/verify_beats.py "data/audio/${{c.stem}}_beats_review.json" --sam-beat ${{samInp.value}} --tempo-scale ${{scaleSel.value}} --offset-ms ${{offInp.value}}`;
  if (taps.length >= 4) {{
    cmd += `\\n# Or paste tapped timestamps into verified_beats_s in data/audio/${{c.stem}}_beats_review.json`;
  }}
  cmdOut.textContent = cmd;
  tapCount.textContent = taps.length;
  tapPreview.textContent = JSON.stringify(taps);
}}

function loadClip() {{
  const c = CLIPS[clipSel.value];
  vid.src = c.preview;
  samInp.value = c.sam;
  scaleSel.value = "1.0";
  offInp.value = "0";
  taps = [];
  refresh();
}}

clipSel.onchange = loadClip;
scaleSel.onchange = refresh;
samInp.oninput = refresh;
offInp.oninput = refresh;
document.getElementById('tapBtn').onclick = () => {{
  taps.push(Number(vid.currentTime.toFixed(4)));
  refresh();
}};
document.getElementById('clearTapBtn').onclick = () => {{ taps = []; refresh(); }};
document.addEventListener('keydown', (e) => {{
  if (e.code === 'Space' && document.activeElement.tagName !== 'INPUT') {{
    e.preventDefault();
    taps.push(Number(vid.currentTime.toFixed(4)));
    refresh();
  }}
}});
loadClip();
</script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main():
    if ROOT.drive.upper() == "D:":
        tmp_dir = ROOT / "data" / ".tmp"
        torch_dir = ROOT / "models" / "weights" / "torch_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        torch_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(torch_dir))
        os.environ["TMP"] = str(tmp_dir)
        os.environ["TEMP"] = str(tmp_dir)
    default_prev_w = min(8, max(2, (os.cpu_count() or 8) // 2))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-demucs-retry", action="store_true")
    ap.add_argument("--skip-previews", action="store_true")
    ap.add_argument("--demucs-timeout", type=int, default=900)
    ap.add_argument("--preview-workers", type=int, default=default_prev_w)
    args = ap.parse_args()

    selected, excluded_aliases = select_pilot_clips(ROOT / "docs" / "inventory_report.json")
    print(f"Pilot clips: {len(selected)} | Excluded Korvai aliases: {len(excluded_aliases)}", file=sys.stderr)

    # 1. Validate all 38 motion and audio files
    errors, hpss_stems = validate_all_outputs(selected)
    print(f"Validation errors: {len(errors)} | HPSS fallback clips: {hpss_stems}", file=sys.stderr)
    if errors:
        for e in errors:
            print("  ERROR:", e, file=sys.stderr)
        sys.exit(1)

    # 2. Re-run Demucs one clip at a time on the 4 HPSS fallback clips
    demucs_retries = []
    if not args.skip_demucs_retry and hpss_stems:
        demucs_retries = rerun_demucs_for_hpss(selected, hpss_stems, timeout_s=args.demucs_timeout)

    # 3. Render song and click previews for all 38 clips
    if not args.skip_previews:
        render_all_missing_previews(selected, workers=args.preview_workers)

    # Verify all 76 previews exist and are non-empty
    preview_missing = []
    for c in selected:
        stem = Path(c["filename"]).stem
        for suf in ("_preview.mp4", "_clicks_preview.mp4"):
            p = ROOT / "data" / "previews" / f"{stem}{suf}"
            if not p.is_file() or p.stat().st_size == 0:
                preview_missing.append(str(p))

    # 4. Full diagnostics + rebuild 38-clip pilot_summary.json
    ffprobe_bin = shutil.which("ffprobe")
    rng = np.random.default_rng(42)
    diag_rows = []
    summary_rows = []
    clean_stats = defaultdict(lambda: {"total_clips": 0, "clips_strict_pass": 0, "clips_with_clean_segs": 0, "clean_segments": 0, "clean_dur_s": 0.0})

    for c in selected:
        stem = Path(c["filename"]).stem
        vpath = ROOT / "data" / "adavus" / c["relative_path"]
        a = json.loads((ROOT / "data" / "audio" / f"{stem}_analysis.json").read_text(encoding="utf-8"))
        m = json.loads((ROOT / "data" / "motion" / f"{stem}.json").read_text(encoding="utf-8"))

        beat_times, beat_source = _load_clip_beats(stem, a)
        strikes = m.get("foot_strikes", [])
        strike_times = [s["time_s"] for s in strikes]
        fps = float(m["fps"])
        bpm = float(a["bpm_median"])
        period_ms = 60000.0 / bpm if bpm > 0 else 0.0
        fsvb = compare_foot_strikes_to_beats(strikes, beat_times, fps=fps)
        med_ms = fsvb.get("median_ms")
        frac_period = round(med_ms / period_ms, 3) if (med_ms is not None and period_ms > 0) else None
        rand_analytic = round(period_ms * 0.25, 2) if period_ms > 0 else None

        sig = signed_errors_and_random_baseline(strike_times, beat_times, rng)
        av_ms = check_av_offset_ms(vpath, ffprobe_bin)

        # Same-tempo-scale comparison at 0.5x and 2.0x (each evaluated against its OWN chance baseline)
        half_bt = apply_tempo_scale(np.asarray(beat_times), tempo_scale=0.5).tolist()
        dbl_bt = apply_tempo_scale(np.asarray(beat_times), tempo_scale=2.0, duration_s=m["duration_s"]).tolist()
        half_med = compare_foot_strikes_to_beats(strikes, half_bt, fps)["median_ms"]
        dbl_med = compare_foot_strikes_to_beats(strikes, dbl_bt, fps)["median_ms"]
        half_period_ms = round(period_ms * 2.0, 2) if period_ms > 0 else None
        dbl_period_ms = round(period_ms * 0.5, 2) if period_ms > 0 else None
        half_rand_emp = signed_errors_and_random_baseline(strike_times, half_bt, rng)["random_baseline_median_ms"]
        dbl_rand_emp = signed_errors_and_random_baseline(strike_times, dbl_bt, rng)["random_baseline_median_ms"]

        # Quality-rule evaluation
        pose_ok = m["final_pose_valid_ratio"] >= 0.85
        hands_ok = m["left_hand_detected_ratio"] >= 0.50 and m["right_hand_detected_ratio"] >= 0.50
        drift_ok = m["depth_travel_reliability"]["torso_scale_cv"] <= 0.12
        strict_pass = pose_ok and hands_ok and drift_ok

        segs = extract_contiguous_clean_segments(m, min_dur_s=2.0)
        usable_segs = [s for s in segs if s["left_hand_ratio"] >= 0.50 and s["right_hand_ratio"] >= 0.50]

        adavu = c["adavu_canonical"]
        clean_stats[adavu]["total_clips"] += 1
        clean_stats[adavu]["clips_strict_pass"] += int(strict_pass)
        clean_stats[adavu]["clips_with_clean_segs"] += int(len(usable_segs) > 0 and drift_ok)
        if drift_ok:
            clean_stats[adavu]["clean_segments"] += len(usable_segs)
            clean_stats[adavu]["clean_dur_s"] = round(clean_stats[adavu]["clean_dur_s"] + sum(s["duration_s"] for s in usable_segs), 2)

        diag_rows.append({
            "adavu": adavu,
            "stem": stem,
            "filename": c["filename"],
            "fps": fps,
            "duration_s": m["duration_s"],
            "separation_backend": a["separation_backend"],
            "beat_source": beat_source,
            "bpm_median": bpm,
            "beat_period_ms": round(period_ms, 2),
            "strike_count": len(strikes),
            "median_abs_ms": med_ms,
            "median_frames": fsvb.get("median_frames"),
            "err_frac_of_period": frac_period,
            "rand_analytic_ms": rand_analytic,
            "rand_empirical_ms": sig["random_baseline_median_ms"],
            "median_signed_ms": sig["median_signed_ms"],
            "early_pct": sig["early_pct"],
            "late_pct": sig["late_pct"],
            "half_tempo_med_ms": half_med,
            "half_period_ms": half_period_ms,
            "half_rand_empirical_ms": half_rand_emp,
            "double_tempo_med_ms": dbl_med,
            "double_period_ms": dbl_period_ms,
            "double_rand_empirical_ms": dbl_rand_emp,
            "double_err_frac_of_period": round(dbl_med / dbl_period_ms, 3) if (dbl_med is not None and dbl_period_ms) else None,
            "av_offset_ms": av_ms,
            "pose_valid": m["final_pose_valid_ratio"],
            "l_hand": m["left_hand_detected_ratio"],
            "r_hand": m["right_hand_detected_ratio"],
            "long_gaps": len(m.get("flagged_long_gaps", [])),
            "torso_scale_cv": m["depth_travel_reliability"]["torso_scale_cv"],
            "strict_clip_pass": strict_pass,
            "usable_segments": len(usable_segs),
            "usable_dur_s": round(sum(s["duration_s"] for s in usable_segs), 2),
        })

        summary_rows.append({
            "adavu": adavu,
            "filename": c["filename"],
            "fps": m["fps"],
            "duration_s": m["duration_s"],
            "separation_backend": a["separation_backend"],
            "separation_detail": a["separation_detail"],
            "bpm_median": a["bpm_median"],
            "beat_count": a["beat_count"],
            "first_sam_index": a["first_sam_index"],
            "max_interp_gap_frames": m["max_interp_gap_frames"],
            "raw_pose_valid_ratio": m["raw_pose_valid_ratio"],
            "final_pose_valid_ratio": m["final_pose_valid_ratio"],
            "left_hand_detected_ratio": m["left_hand_detected_ratio"],
            "right_hand_detected_ratio": m["right_hand_detected_ratio"],
            "hand_swap_check": m["hand_swap_check"],
            "flagged_long_gaps_count": len(m["flagged_long_gaps"]),
            "opening_facing_yaw_deg": m["opening_facing_yaw_deg"],
            "torso_scale_cv": m["depth_travel_reliability"]["torso_scale_cv"],
            "scale_drift_flag": m["depth_travel_reliability"]["scale_drift_flag"],
            "foot_strikes_vs_beats": fsvb,
            "preview_mp4": f"data/previews/{stem}_preview.mp4",
            "clicks_preview_mp4": f"data/previews/{stem}_clicks_preview.mp4",
            "clicks_wav": f"data/audio/{stem}_clicks.wav",
        })

    # Rebuild full 38-clip pilot_summary.json
    pilot_summary = {
        "pilot_categories": sorted({c["adavu_canonical"] for c in selected}),
        "selected_count": len(summary_rows),
        "separation_backends": dict(Counter(r["separation_backend"] for r in summary_rows)),
        "alias_status": {
            "KorvaiAdavu": "UNCONFIRMED — excluded 10 Korvai-named clips (8 passed both checks); note ThaThaiTham1-6 also list KorvaiAdavu in adavu_name2.",
            "MandiAdavu2": "UNCONFIRMED — non-pilot clip mapped via scripts/mapping_overrides.json (confirmed_by_user=false).",
        },
        "excluded_unconfirmed_alias_clips": [
            {"filename": c["filename"], "adavu_canonical": c["adavu_canonical"], "alternates": c["adavu_alternates"]}
            for c in excluded_aliases
        ],
        "clips": summary_rows,
    }
    (ROOT / "data" / "pilot_summary.json").write_text(json.dumps(pilot_summary, indent=2), encoding="utf-8")

    # 5. Generate 6-clip timeline plot, strike contact sheets, and beat verifier HTML
    diag_dir = ROOT / "data" / "diagnostics"
    plot_six_timelines(diag_rows, diag_dir / "six_clips_beat_strike_timelines.png")
    save_strike_check_montages(diag_dir / "strike_frames")
    build_beat_verifier_html(ROOT / "data" / "previews" / "beat_verifier.html")

    diag_report = {
        "validated_clips": len(selected),
        "validation_errors": errors,
        "demucs_retries": demucs_retries,
        "preview_missing": preview_missing,
        "quality_rule_summary_per_adavu": dict(clean_stats),
        "clips": diag_rows,
    }
    (diag_dir / "foot_strike_diagnostics.json").write_text(json.dumps(diag_report, indent=2), encoding="utf-8")
    print(f"Diagnostics complete. Preview missing={len(preview_missing)}. Wrote {diag_dir / 'foot_strike_diagnostics.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()

