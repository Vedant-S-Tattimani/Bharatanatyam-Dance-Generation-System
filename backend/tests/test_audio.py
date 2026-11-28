"""Unit tests for Phase 1 Step 1: audio analysis, tala, tempo overrides, and beat errors."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.audio.analyzer import (  # noqa: E402
    SR,
    analyze_audio,
    apply_tempo_scale,
    compute_beat_error_ms,
    write_click_track,
)
from app.audio.tala import get_tala  # noqa: E402


def _make_click_song(tmp_path: Path, bpm: float = 120.0, duration_s: float = 8.0) -> tuple[Path, np.ndarray]:
    """Synthesize a clean percussive click track at an exact BPM."""
    sr = SR
    n = int(duration_s * sr)
    y = np.zeros(n, dtype=np.float32)
    step = 60.0 / bpm
    beat_times = np.arange(0.5, duration_s - 0.2, step)
    for idx, t in enumerate(beat_times):
        s = int(t * sr)
        burst_len = int(0.03 * sr)
        tt = np.arange(burst_len) / sr
        freq = 880.0 if idx % 8 == 0 else 440.0
        burst = 0.8 * np.sin(2 * np.pi * freq * tt) * np.exp(-tt * 60.0)
        end = min(n, s + burst_len)
        y[s:end] += burst[: end - s]
    wav_path = tmp_path / "synth_adi.wav"
    sf.write(str(wav_path), y, sr)
    return wav_path, beat_times


def test_tala_defaults_to_adi_8_beats():
    tala = get_tala("Adi Tala")
    assert tala.name == "Adi"
    assert tala.beats == 8
    assert sum(tala.angams) == 8


def test_apply_tempo_scale_half_and_double():
    beats = np.array([1.0, 2.0, 3.0, 4.0])
    half = apply_tempo_scale(beats, tempo_scale=0.5)
    assert np.allclose(half, [1.0, 3.0])

    double = apply_tempo_scale(beats, tempo_scale=2.0, duration_s=5.0)
    assert np.allclose(double, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])


def test_analyze_audio_hpss_fallback_and_overrides(tmp_path: Path):
    wav_path, gt_beats = _make_click_song(tmp_path, bpm=120.0, duration_s=8.0)
    analysis, y_mix, sr = analyze_audio(
        wav_path,
        tala_name="adi",
        use_demucs=False,
        sam_beat_index=2,
        tempo_scale=1.0,
    )
    assert analysis["separation_backend"] == "hpss_fallback"
    assert analysis["tala_beats"] == 8
    assert analysis["first_sam_index"] == 2
    assert analysis["sam_source"] == "user_override"
    assert 110.0 <= analysis["bpm_median"] <= 130.0
    assert analysis["beats"][2]["is_sam"] is True

    # Click track writes cleanly
    click_wav = tmp_path / "clicks.wav"
    write_click_track(y_mix, sr, analysis["beats"], click_wav)
    assert click_wav.is_file() and click_wav.stat().st_size > 0

    # Measure 4a error against ground-truth beats
    err = compute_beat_error_ms([b["time_s"] for b in analysis["beats"]], gt_beats)
    assert err["count"] > 0
    assert err["median_ms"] <= 25.0


def test_compute_beat_error_ms_exact_values():
    err = compute_beat_error_ms([1.000, 2.020, 3.050], [1.000, 2.000, 3.000])
    assert err["count"] == 3
    assert err["median_ms"] == 20.0
    assert err["max_ms"] == 50.0


def test_analyze_audio_deterministic_repeat(tmp_path: Path):
    wav_path, _ = _make_click_song(tmp_path, bpm=120.0, duration_s=8.0)
    a1, _, _ = analyze_audio(wav_path, tala_name="adi", use_demucs=False)
    a2, _, _ = analyze_audio(wav_path, tala_name="adi", use_demucs=False)
    assert a1["bpm_median"] == a2["bpm_median"]
    assert a1["beats"] == a2["beats"]


def test_tempo_octave_check_slow_sparse_and_fast_click_tracks(tmp_path: Path):
    # Slow 44 BPM sparse track with weak resonant sub-beats (0.12 amplitude) at 88 BPM:
    # beat_track doubles to ~88 BPM; octave check must fold silent parity back to ~44 BPM.
    sr = SR
    dur_slow = 24.0
    y_slow = np.zeros(int(dur_slow * sr), dtype=np.float32)
    step_half = 60.0 / 88.0
    for idx, t in enumerate(np.arange(0.8, dur_slow - 0.5, step_half)):
        s = int(t * sr)
        tt = np.arange(int(0.03 * sr)) / sr
        amp = 0.95 if idx % 2 == 0 else 0.12
        y_slow[s : s + len(tt)] += amp * np.sin(2 * np.pi * 660.0 * tt) * np.exp(-tt * 60.0)
    slow_wav = tmp_path / "slow_44_sub.wav"
    sf.write(str(slow_wav), y_slow, sr)

    a_slow_old, _, _ = analyze_audio(slow_wav, tala_name="adi", use_demucs=False, use_tempo_octave_check=False)
    a_slow_new, _, _ = analyze_audio(slow_wav, tala_name="adi", use_demucs=False, use_tempo_octave_check=True)
    assert a_slow_old["bpm_median"] > 70.0
    assert 40.0 <= a_slow_new["bpm_median"] <= 48.0
    assert a_slow_new["tempo_octave_check"] == "folded_half_silent_parity"

    # Fast 160 BPM track: octave check MUST NOT erroneously fold fast true beats in half
    fast_wav, gt_fast = _make_click_song(tmp_path, bpm=160.0, duration_s=10.0)
    a_fast_new, _, _ = analyze_audio(fast_wav, tala_name="adi", use_demucs=False, use_tempo_octave_check=True)
    assert 150.0 <= a_fast_new["bpm_median"] <= 170.0
    err_fast = compute_beat_error_ms([b["time_s"] for b in a_fast_new["beats"]], gt_fast)
    assert err_fast["median_ms"] <= 25.0

