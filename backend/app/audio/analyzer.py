"""
analyzer.py – Audio feature extraction, Demucs/HPSS percussion separation,
beat tracking with half/double tempo & sam overrides, click-track generation,
and detected-vs-verified beat error measurement (Measure 4a).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .tala import get_tala

SR = 22050
HOP = 512


def _extract_audio_to_wav(media_path: Path, out_wav: Path, sr: int = SR) -> None:
    """Extract mono WAV from audio or video file using ffmpeg when needed."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-i", str(media_path),
            "-vn", "-ac", "1", "-ar", str(sr),
            "-f", "wav", str(out_wav),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and out_wav.is_file() and out_wav.stat().st_size > 0:
            return
    # Fallback: librosa load & write
    y, loaded_sr = librosa.load(str(media_path), sr=sr, mono=True)
    sf.write(str(out_wav), y, loaded_sr)


def separate_percussion(
    audio_path: Path,
    sr: int = SR,
    use_demucs: bool = True,
    demucs_timeout: int = 900,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """
    Separate full mix and percussive signal.

    Attempts Demucs (if installed and enabled) to isolate drums/percussion;
    otherwise falls back cleanly to librosa HPSS.

    Returns:
      (y_mix, y_perc, separation_backend, separation_detail)
      where separation_backend is explicitly "demucs" or "hpss_fallback".
    """
    with tempfile.TemporaryDirectory(prefix="nritya_aud_") as tmp_dir:
        tmp_wav = Path(tmp_dir) / "mix.wav"
        _extract_audio_to_wav(audio_path, tmp_wav, sr=sr)
        y_mix, _ = librosa.load(str(tmp_wav), sr=sr, mono=True)

        if use_demucs:
            try:
                import demucs  # noqa: F401
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                out_dir = Path(tmp_dir) / "demucs_out"
                env = os.environ.copy()
                root_path = Path(__file__).resolve().parents[3]
                if root_path.drive.upper() == "D:":
                    torch_cache = root_path / "models" / "weights" / "torch_cache"
                    torch_cache.mkdir(parents=True, exist_ok=True)
                    env.setdefault("TORCH_HOME", str(torch_cache))
                cmd = [
                    sys.executable, "-m", "demucs",
                    "--two-stems=drums",
                    "--shifts", "0",
                    "-n", "htdemucs",
                    "-d", device,
                    "-o", str(out_dir),
                    str(tmp_wav),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=demucs_timeout)
                drums_hits = list(out_dir.rglob("drums.wav"))
                if proc.returncode == 0 and drums_hits:
                    y_perc, _ = librosa.load(str(drums_hits[0]), sr=sr, mono=True)
                    return y_mix, y_perc, "demucs", f"Demucs (htdemucs --two-stems=drums --shifts 0, device={device}) separated percussion"
                reason = (proc.stderr or proc.stdout or "demucs exited non-zero").strip().splitlines()[-1]
                y_harm, y_perc = librosa.effects.hpss(y_mix)
                del y_harm
                return y_mix, y_perc, "hpss_fallback", f"Demucs failed ({reason}); used librosa.effects.hpss fallback"
            except ImportError:
                y_harm, y_perc = librosa.effects.hpss(y_mix)
                del y_harm
                return y_mix, y_perc, "hpss_fallback", "Demucs not installed; used librosa.effects.hpss fallback"
            except Exception as exc:
                y_harm, y_perc = librosa.effects.hpss(y_mix)
                del y_harm
                return y_mix, y_perc, "hpss_fallback", f"Demucs error ({exc}); used librosa.effects.hpss fallback"

    y_harm, y_perc = librosa.effects.hpss(y_mix)
    del y_harm
    return y_mix, y_perc, "hpss_fallback", "Demucs disabled by flag; used librosa.effects.hpss fallback"


def apply_tempo_scale(
    beat_times: np.ndarray,
    tempo_scale: float = 1.0,
    offset_ms: float = 0.0,
    duration_s: float | None = None,
) -> np.ndarray:
    """
    Adjust beat timestamps for half-tempo (0.5), double-tempo (2.0), or custom scaling,
    plus a user-supplied millisecond offset.

    - tempo_scale = 2.0 (double tempo): inserts midpoints between consecutive beats.
    - tempo_scale = 0.5 (half tempo): keeps every second beat.
    - Other positive scales: rescales beat intervals anchored at beat[0] (or median IBI / scale).
    """
    bt = np.asarray(beat_times, dtype=float)
    if len(bt) == 0:
        return bt

    if abs(tempo_scale - 1.0) < 1e-6:
        scaled = bt.copy()
    elif abs(tempo_scale - 2.0) < 1e-6 and len(bt) >= 2:
        mids = 0.5 * (bt[:-1] + bt[1:])
        scaled = np.empty(len(bt) * 2 - 1, dtype=float)
        scaled[0::2] = bt
        scaled[1::2] = mids
        last_dt = float(bt[-1] - bt[-2]) * 0.5
        if duration_s is None or bt[-1] + last_dt <= duration_s:
            scaled = np.append(scaled, bt[-1] + last_dt)
    elif abs(tempo_scale - 0.5) < 1e-6 and len(bt) >= 2:
        scaled = bt[::2].copy()
    else:
        if tempo_scale <= 0:
            raise ValueError("tempo_scale must be > 0")
        anchor = bt[0]
        scaled = anchor + (bt - anchor) / tempo_scale

    if offset_ms != 0.0:
        scaled = scaled + (offset_ms / 1000.0)
        scaled = scaled[scaled >= 0.0]

    if duration_s is not None:
        scaled = scaled[scaled <= duration_s + 1e-6]

    return scaled


def guess_sam_index(y_perc: np.ndarray, sr: int, beat_times: np.ndarray, tala_beats: int) -> int:
    """Estimate sam phase (0..tala_beats-1) using max percussive onset strength per phase."""
    if len(beat_times) == 0:
        return 0
    env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=HOP)
    frames = np.clip(librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP), 0, max(0, len(env) - 1))
    vals = np.array([float(env[max(0, f - 3):f + 4].max()) if len(env) else 0.0 for f in frames])
    scores = [float(vals[p::tala_beats].mean()) if len(vals[p::tala_beats]) else 0.0 for p in range(tala_beats)]
    return int(np.argmax(scores))


def estimate_beats_independent(
    y_perc: np.ndarray,
    sr: int = SR,
    hop: int = HOP,
) -> dict[str, Any]:
    """
    Second, independent beat/tempo estimator using Fourier Tempogram global dominant
    pulse + local onset-peak snapping (no dynamic programming beat_track).
    Used strictly as an independent cross-check; agreement does not imply truth.
    """
    if len(y_perc) < sr:
        return {"bpm": 0.0, "beats_s": []}
    env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop)
    if len(env) < 8:
        return {"bpm": 0.0, "beats_s": []}
    tg = librosa.feature.tempogram(onset_envelope=env, sr=sr, hop_length=hop, win_length=384)
    bpms = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=hop)
    mean_tg = tg.mean(axis=1)
    mask = (bpms >= 35.0) & (bpms <= 200.0)
    if not np.any(mask):
        return {"bpm": 0.0, "beats_s": []}
    sub_idx = np.where(mask)[0]
    best_idx = sub_idx[int(np.argmax(mean_tg[sub_idx]))]
    dom_bpm = float(bpms[best_idx])
    period_s = 60.0 / max(dom_bpm, 1.0)

    # Phase-lock uniform tempogram grid against onset_strength via circular phase sweep
    duration_s = len(y_perc) / sr
    n_phases = 24
    best_phase = 0.0
    best_score = -1.0
    t_env = np.arange(len(env)) * (hop / sr)
    for k in range(n_phases):
        ph = (k / n_phases) * period_s
        grid = np.arange(ph, duration_s, period_s)
        if len(grid) == 0:
            continue
        f_idx = np.clip(np.round(grid * sr / hop).astype(int), 0, len(env) - 1)
        sc = float(np.mean([float(env[max(0, fi - 2) : fi + 3].max()) for fi in f_idx]))
        if sc > best_score:
            best_score = sc
            best_phase = ph

    grid = np.arange(best_phase, duration_s, period_s)
    # Snap each grid tick to nearest local onset peak within +/- 15% of period
    snap_rad_f = max(1, int(round(0.15 * period_s * sr / hop)))
    snapped: list[float] = []
    for gt in grid:
        c = int(round(gt * sr / hop))
        lo, hi = max(0, c - snap_rad_f), min(len(env), c + snap_rad_f + 1)
        if hi > lo:
            pk = lo + int(np.argmax(env[lo:hi]))
            snapped.append(round(float(t_env[pk]), 4))
    return {"bpm": round(dom_bpm, 2), "beats_s": snapped}


def _apply_tempo_octave_check(
    raw_beats: np.ndarray,
    onset_env: np.ndarray,
    y_perc: np.ndarray,
    sr: int,
    hop: int = HOP,
) -> tuple[np.ndarray, str]:
    """
    Fix 4 (switch-gated): fold hallucinated 2x sub-beats down to 0.5x when one parity
    consistently sits in silence/weak resonance (low percussive RMS + onset strength)
    and resulting tempo remains >= 38 BPM.
    """
    if len(raw_beats) < 8 or len(onset_env) == 0:
        return raw_beats, "unchanged"
    est_bpm = 60.0 / max(float(np.median(np.diff(raw_beats))), 1e-4)
    if est_bpm * 0.5 < 38.0:
        return raw_beats, "unchanged"

    frames = np.clip(librosa.time_to_frames(raw_beats, sr=sr, hop_length=hop), 0, len(onset_env) - 1)
    rad = max(2, int(round(0.065 * sr / hop)))
    strengths = np.array([float(onset_env[max(0, int(f) - rad) : int(f) + rad + 1].max()) for f in frames])

    win_samps = max(1, int(round(0.065 * sr)))
    perc_peaks = np.array([
        float(np.max(np.abs(y_perc[max(0, int(t * sr) - win_samps) : min(len(y_perc), int(t * sr) + win_samps)])))
        if len(y_perc) else 0.0
        for t in raw_beats
    ])

    even_s, odd_s = strengths[0::2], strengths[1::2]
    even_p, odd_p = perc_peaks[0::2], perc_peaks[1::2]
    if len(even_s) < 3 or len(odd_s) < 3:
        return raw_beats, "unchanged"

    med_even, med_odd = float(np.median(even_s)), float(np.median(odd_s))
    hi_s, lo_s = max(med_even, med_odd), min(med_even, med_odd)

    p_even, p_odd = float(np.median(even_p)), float(np.median(odd_p))
    hi_p, lo_p = max(p_even, p_odd), min(p_even, p_odd)

    onset_silent = hi_s > 0 and lo_s < 0.42 * hi_s
    perc_silent = hi_p > 0 and lo_p < 0.28 * hi_p
    same_side = (med_even >= med_odd) == (p_even >= p_odd)
    if (onset_silent and perc_silent and same_side) or (hi_p > 0 and lo_p < 0.14 * hi_p):
        chosen = raw_beats[0::2] if p_even >= p_odd else raw_beats[1::2]
        return chosen, "folded_half_silent_parity"
    return raw_beats, "unchanged"


def analyze_audio(
    audio_path: Path,
    tala_name: str = "adi",
    use_demucs: bool = True,
    sam_beat_index: int | None = None,
    tempo_scale: float = 1.0,
    offset_ms: float = 0.0,
    sr: int = SR,
    demucs_timeout: int = 900,
    use_tempo_octave_check: bool = False,
) -> tuple[dict[str, Any], np.ndarray, int]:
    """
    Analyze audio/video track with librosa (+ Demucs or HPSS fallback).

    Returns:
      (analysis_dict, y_mix, sr)
    """
    tala = get_tala(tala_name)
    y_mix, y_perc, sep_backend, sep_detail = separate_percussion(
        audio_path, sr=sr, use_demucs=use_demucs, demucs_timeout=demucs_timeout
    )
    duration_s = float(len(y_mix) / sr) if sr > 0 else 0.0

    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=HOP)
    tempo_est, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP,
        units="frames",
    )
    raw_beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    octave_note = "disabled"
    if use_tempo_octave_check:
        raw_beats, octave_note = _apply_tempo_octave_check(raw_beats, onset_env, y_perc=y_perc, sr=sr, hop=HOP)

    # Apply tempo scaling (e.g. 0.5x half-tempo, 2.0x double-tempo) & offset
    beat_times = apply_tempo_scale(raw_beats, tempo_scale=tempo_scale, offset_ms=offset_ms, duration_s=duration_s)

    if len(beat_times) >= 2:
        ibis = np.diff(beat_times)
        local_bpms = 60.0 / np.clip(ibis, 1e-4, None)
        local_bpms = np.append(local_bpms, local_bpms[-1])
        bpm_median = float(np.median(local_bpms))
    else:
        raw_bpm = float(np.atleast_1d(tempo_est)[0]) if np.size(tempo_est) else 0.0
        bpm_median = raw_bpm * tempo_scale
        local_bpms = np.full(len(beat_times), bpm_median, dtype=float)

    if sam_beat_index is not None:
        first_sam = int(sam_beat_index) % tala.beats
        sam_source = "user_override"
    else:
        first_sam = guess_sam_index(y_perc, sr, beat_times, tala.beats)
        sam_source = "auto_onset_heuristic"

    # Beat positions 1..tala.beats (1 = sam)
    idx = np.arange(len(beat_times))
    tala_positions = ((idx - first_sam) % tala.beats) + 1 if len(beat_times) else np.array([], dtype=int)

    # Onsets and RMS energy curve
    perc_onsets = librosa.onset.onset_detect(y=y_perc, sr=sr, hop_length=HOP, units="time")
    rms_curve = librosa.feature.rms(y=y_mix, hop_length=HOP)[0]
    rms_max = float(rms_curve.max()) if len(rms_curve) and float(rms_curve.max()) > 0 else 1.0
    rms_norm = rms_curve / rms_max

    # Additive affect / intensity estimate. This is a transparent heuristic over
    # RMS loudness, onset density and tempo, NOT an emotion classifier.
    n_energy_bins = 32
    if len(rms_norm):
        bin_edges = np.linspace(0, len(rms_norm), n_energy_bins + 1).astype(int)
        energy_curve = [
            round(float(rms_norm[bin_edges[i] : bin_edges[i + 1]].mean()), 4)
            if bin_edges[i + 1] > bin_edges[i] else 0.0
            for i in range(n_energy_bins)
        ]
    else:
        energy_curve = [0.0] * n_energy_bins
    energy_mean = float(np.mean(energy_curve))
    energy_peak = float(np.max(energy_curve))
    onset_density_10s = round(len(perc_onsets) / max(duration_s, 1.0) * 10.0, 2)
    intensity_score = (
        0.55 * energy_mean
        + 0.25 * min(onset_density_10s / 12.0, 1.0)
        + 0.20 * min(max(bpm_median, 0.0) / 160.0, 1.0)
    )
    if intensity_score < 0.28:
        affect_label = "calm"
    elif intensity_score < 0.48:
        affect_label = "flowing"
    elif intensity_score < 0.68:
        affect_label = "bright"
    else:
        affect_label = "driving"
    affect = {
        "label": affect_label,
        "intensity_score": round(intensity_score, 4),
        "energy_mean": round(energy_mean, 4),
        "energy_peak": round(energy_peak, 4),
        "dynamic_range": round(energy_peak - energy_mean, 4),
        "onset_density_per_10s": onset_density_10s,
        "basis": "heuristic RMS loudness + onset density + tempo (not an emotion classifier)",
    }

    beat_records = []
    for i, t in enumerate(beat_times):
        t_next = float(beat_times[i + 1]) if i + 1 < len(beat_times) else min(duration_s, float(t) + (60.0 / max(bpm_median, 1.0)))
        f0 = int(librosa.time_to_frames(float(t), sr=sr, hop_length=HOP))
        f1 = max(f0 + 1, int(librosa.time_to_frames(t_next, sr=sr, hop_length=HOP)))
        f0 = min(f0, max(0, len(rms_norm) - 1))
        f1 = min(f1, len(rms_norm))
        energy = float(rms_norm[f0:f1].mean()) if len(rms_norm) else 0.0
        beat_records.append({
            "index": i,
            "time_s": round(float(t), 4),
            "tala_pos": int(tala_positions[i]),
            "is_sam": bool(tala_positions[i] == 1),
            "local_bpm": round(float(local_bpms[i]), 2),
            "energy": round(energy, 4),
        })

    analysis = {
        "source_file": audio_path.name,
        "duration_s": round(duration_s, 3),
        "sample_rate": sr,
        "tala": tala.name,
        "tala_beats": tala.beats,
        "separation_backend": sep_backend,
        "separation_detail": sep_detail,
        "tempo_octave_check": octave_note,
        "tempo_scale": tempo_scale,
        "offset_ms": offset_ms,
        "bpm_median": round(bpm_median, 2),
        "first_sam_index": first_sam,
        "sam_source": sam_source,
        "beat_count": len(beat_records),
        "onset_count": int(len(perc_onsets)),
        "perc_onsets_s": [round(float(t), 4) for t in perc_onsets],
        "energy_curve": energy_curve,
        "affect": affect,
        "beats": beat_records,
    }
    return analysis, y_mix, sr


def write_click_track(
    y_mix: np.ndarray,
    sr: int,
    beats: list[dict[str, Any]],
    out_wav: Path,
) -> Path:
    """Write a WAV with accented clicks on sam (1500 Hz) and regular clicks on other beats (900 Hz)."""
    sam_times = np.array([b["time_s"] for b in beats if b["is_sam"]], dtype=float)
    other_times = np.array([b["time_s"] for b in beats if not b["is_sam"]], dtype=float)

    n = len(y_mix)
    clicks_sam = (
        librosa.clicks(times=sam_times, sr=sr, length=n, click_freq=1500.0, click_duration=0.10)
        if len(sam_times) else np.zeros(n, dtype=float)
    )
    clicks_other = (
        librosa.clicks(times=other_times, sr=sr, length=n, click_freq=900.0, click_duration=0.05)
        if len(other_times) else np.zeros(n, dtype=float)
    )

    mixed = 0.75 * y_mix + 0.65 * clicks_sam + 0.45 * clicks_other
    peak = float(np.max(np.abs(mixed))) if n else 1.0
    if peak > 1.0:
        mixed = mixed / peak

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), mixed, sr)
    return out_wav


def write_beat_review_file(analysis: dict[str, Any], out_json: Path) -> Path:
    """
    Save an editable beat-review JSON file so a user can verify/correct beats by ear
    (by setting tempo_scale, first_sam_index, offset_ms, or explicit verified_beats_s).
    """
    review = {
        "instructions": (
            "Listen to the matching *_clicks.wav file. To correct beats/sam, edit "
            "'verified_tempo_scale' (e.g. 0.5 or 2.0), 'verified_first_sam_index' (0..tala_beats-1), "
            "'verified_offset_ms', or supply explicit timestamps in 'verified_beats_s', "
            "then run: python scripts/verify_beats.py <this_file>"
        ),
        "source_file": analysis["source_file"],
        "tala": analysis["tala"],
        "tala_beats": analysis["tala_beats"],
        "separation_backend": analysis["separation_backend"],
        "detected_bpm": analysis["bpm_median"],
        "detected_first_sam_index": analysis["first_sam_index"],
        "detected_beats_s": [b["time_s"] for b in analysis["beats"]],
        "verified_first_sam_index": None,
        "verified_tempo_scale": None,
        "verified_offset_ms": None,
        "verified_beats_s": None,
        "notes": "",
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(review, indent=2), encoding="utf-8")
    return out_json


def compute_beat_error_ms(
    predicted_times_s: list[float] | np.ndarray,
    reference_times_s: list[float] | np.ndarray,
) -> dict[str, float | int | None]:
    """
    Compute nearest-beat alignment error in milliseconds (Measure 4a / 4b).

    For each predicted/reference event, finds absolute distance to nearest counterpart.
    Returns median_ms, mean_ms, max_ms, p95_ms, count.
    """
    pred = np.asarray(predicted_times_s, dtype=float)
    ref = np.asarray(reference_times_s, dtype=float)
    if len(pred) == 0 or len(ref) == 0:
        return {"count": 0, "median_ms": None, "mean_ms": None, "p95_ms": None, "max_ms": None}

    diffs_ms = np.array([float(np.min(np.abs(ref - t)) * 1000.0) for t in pred], dtype=float)
    return {
        "count": int(len(diffs_ms)),
        "median_ms": round(float(np.median(diffs_ms)), 2),
        "mean_ms": round(float(np.mean(diffs_ms)), 2),
        "p95_ms": round(float(np.percentile(diffs_ms, 95)), 2),
        "max_ms": round(float(np.max(diffs_ms)), 2),
    }

