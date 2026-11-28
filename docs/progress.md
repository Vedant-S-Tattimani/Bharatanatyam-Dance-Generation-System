# Nritya Phase 1 Progress & Decision Log

**Phase 1 Status:** `experimental, accuracy unverified` (NOT PASSED)  
**Phase 2 Website Status:** `implemented, browser-smoke-verified` (accuracy still unverified)  
**Phase 3 Model Training:** Not started  
**Labels Directory (`data/labels/`):** `0` files (no human labels created; **accuracy is unverified** across all real clips)

---

## 1. Switch-Gated Fixes & Ablation Decisions (Max 2 Tuning Rounds)

| Fix & Switch | Default | Synthetic Test Result | Real-Clip (38 clips, **unverified**) Result | Decision |
|---|---|---|---|---|
| **Fix 1: Strike trough timing** (`use_trough_timing`) in [`extractor.py`](file:///d:/dance%20clone/backend/app/motion/extractor.py) | `True` (kept ON) | **PASS** (`test_trough_timing_switch_fixes_descent_shoulder`, `test_foot_strike_detection_and_beat_comparison`) | Median nearest-beat error moved `0.2697 T` -> `0.2434 T`; signed median bias `-85.57 ms` -> `-60.80 ms`; clips `< 0.25 T` moved `15/38` -> `19/38`; intro (`<2.0s`) triggers stayed `5` | **Keep ON** |
| **Fix 2: Crossed-leg rejection** (`use_crossed_leg_rejection`) in [`extractor.py`](file:///d:/dance%20clone/backend/app/motion/extractor.py) | `True` (kept ON) | **PASS** (`test_crossed_leg_rejection_and_adaptive_amplitude_switches`) | Removed `26` occluded crossed-leg triggers (`2169 -> 2143` strikes); preserved `0.2434 T`, `-60.80 ms` signed bias, `19/38` clips `< 0.25 T`, `5` `<2.0s` intro triggers | **Keep ON** |
| **Fix 3: Adaptive low-amplitude thresholds** (`use_adaptive_amplitude`) in [`extractor.py`](file:///d:/dance%20clone/backend/app/motion/extractor.py) | `False` (left OFF) | **PASS** synthetic (`0.072` lift detected with Round 2 floor/prominence) | **Round 1** (`0.058 / 0.055`): `<2.0s` intro triggers doubled (`5 -> 11`) and median error worsened (`0.2434 -> 0.2488 T`). **Round 2** (`0.068 / 0.065`): median error `0.2404 T`, `21/38` `< 0.25 T`, **but** `<2.0s` intro triggers still worsened `5 -> 8` (`+60%`) | **Leave OFF** after 2 rounds |
| **Fix 4: Tempo octave parity check** (`use_tempo_octave_check`) in [`analyzer.py`](file:///d:/dance%20clone/backend/app/audio/analyzer.py) | `False` (turned OFF) | **PASS** (`test_tempo_octave_check_slow_sparse_and_fast_click_tracks`) | **Round 1**: mis-folded `Nattadavu2.6`, `Nattadavu2.8`, and `ThaThaiTham5`. **Round 2**: stopped folding `Nattadavu2.6`/`2.8`/`3.6` and improved `6` clips (`ThaiyaThaihi2`, `ThaThaiTham1..4`, `ThaThaiTham6`; Tempogram `±8%` agreement `22/38 -> 27/38`), **but** `ThaThaiTham5` still worsened (`0.1649 T -> 0.2729 T`) | **Turned OFF by default** (`ThaThaiTham5` got worse) |

### Important Statistical & Verification Caveat
- **Accuracy is unverified:** There are `0` human-verified ground-truth beat or foot-strike annotations in `data/labels/`. Every real-clip error number in this project is measured only against unverified automated outputs.
- **Being under `0.25 T` does not by itself show better than chance:** Even when median nearest-beat error falls below `0.25 T` (half the expected uniform midpoint under a period `T`), that does **not** prove genuine beat synchronization or beat-tracking accuracy. Nearest-beat distance omits false positives and missed strikes, depends on the unverified tracker's chosen tempo scale (where octave errors change `T`), can drop when extra detections or dense pulses bracket beats, and offers no accuracy proof without human-labeled beats and foot strikes.

### Final Refreshed 38-Clip Snapshot (**unverified**)
Measured on the final refreshed `data/motion/*.json` and `data/audio/*_analysis.json` (`use_trough_timing=True`, `use_crossed_leg_rejection=True`, `use_adaptive_amplitude=False`, `use_tempo_octave_check=False`):
- Baseline (all fixes OFF): `2205` strikes, `5` intro (`<2.0s`) strikes, median error `0.2697 T`, median signed bias `-85.57 ms`, `15/38` clips `< 0.25 T`.
- Final defaults (Fixes 1 & 2 ON; Fixes 3 & 4 OFF): `2143` strikes, `5` intro (`<2.0s`) strikes, median error `0.2434 T`, median signed bias `-60.48 ms`, `19/38` clips `< 0.25 T`.

### Independent Beat Cross-Check & Start-Shift Stability
- **Independent estimator (`estimate_beats_independent`, harmonic-free predominant local pulse / Fourier tempogram):** BPM agreement within `±8%` was `22/38` with `use_tempo_octave_check=False` (`27/38` in Fix 4 Round 2 experiment). Agreement between two unverified trackers indicates structural consistency only, **not** musical ground-truth proof.
- **Start-shift stability (`+120 ms`, `+250 ms`, `+500 ms` trim via `ffmpeg`, Demucs `--shifts 0`, with `use_tempo_octave_check=False` default):**
  - `ThaThaiTham1`: `86.13 BPM` across all shifts (`3.9 / 5.4 / 10.9 ms` drift).
  - `KudithuMettuAdavu1`: `92.29` at `+120/+250 ms`, `95.7` at `+500 ms` (`3.9 / 5.4 / 12.3 ms` drift).
  - `Nattadavu2.6`: Base `172.27 BPM` held at `+500 ms` (`12.3 ms` drift), **but** `librosa.beat_track` dropped to `86.13 BPM` on `+120 ms` and `+250 ms` crops.
  - `ThaiyaThaihi2`: `86.13 BPM` at `0 / +120 ms` (`3.9 ms` drift) and `83.35 BPM` at `+250 / +500 ms` (`5.4 / 10.9 ms` drift).

---

## 2. Catalogue, End-to-End Prototype & Verification

- **Catalogue (`data/catalogue/catalogue.json`):** Rebuilt via [`build_catalogue.py`](file:///d:/dance%20clone/scripts/build_catalogue.py) to **`459` total units** (`317` confirmed across `KudithuMettuAdavu` [62], `Nattadavu` [176], `ThaiyaThaihiAdavu` [79]; **`142` unconfirmed** `ThaThaiThamAdavu` with `confirmed: false`). Quality rules remain `PROVISIONAL`.
- **Tests (`backend/tests/`):** `.venv\Scripts\python.exe -m pytest backend/tests -q` -> **`24 passed`** (latest run `60.39s`) with `use_tempo_octave_check=False` default.

---

## 3. API Changes Summary (Full Contract in [`api_changes.md`](file:///d:/dance%20clone/docs/api_changes.md))

1. **Renamed metric:** `measure_4b_timeline_vs_beats_ms` -> `frame_quantization_check_ms` (measures 30 fps quantization, not foot-beat accuracy).
2. **Added `planned_foot_strikes_vs_beats`:** Same-tempo evaluation of mapped source foot strikes against beats (`beat_source`, `beat_period_ms`, `err_frac_of_period`, `same_tempo_random_baseline_ms`).
3. **Unconfirmed gate (`include_unconfirmed`):** `/api/catalogue` accepts `include_unconfirmed` and returns `confirmed_units`, `unconfirmed_units`, `confirmed_adavu_categories`, and `quality_rules_status`; `POST /api/choreography/generate` defaults `include_unconfirmed=false` (excludes `ThaThaiThamAdavu`).
4. **Source self-exclusion (`exclude_stems` / `excluded_stems`):** Auto-excludes uploaded file stem plus optional comma-separated stems from unit retrieval.
5. **Constant-FPS frame grid & honest hand metrics:** `frames` follow `t = t0 + f/fps`, missing hand poses stay `null` (never hallucinated), `transition_metrics` reports separate `body`, `hands`, and stance `foot_sliding`.
6. **New `audio_analysis.tempo_octave_check` field:** Reports `"disabled"` by default (or `"unchanged"` / `"folded_half_silent_parity"` when `use_tempo_octave_check=True`).
7. **`GET /api/health` catalogue total:** Refreshed catalogue holds `459` units (`317` confirmed + `142` unconfirmed).
8. **Static website:** FastAPI serves `frontend/index.html` at `/` and mounts `/assets`; health reports `frontend_available`.
9. **Affect / energy fields:** `audio_analysis.energy_curve`, `audio_analysis.affect`, `beats[].energy`, top-level `affect`, and `affect_timeline[]` are additive heuristics, not emotion classification or learned choreography.
10. **Retained media and export:** Uploaded source is retained per job; `POST /api/choreography/{job_id}/render` writes `preview.mp4`, exposed via `/api/media/{job_id}/preview.mp4`.
11. **`source_available`:** Both job detail and `GET /api/jobs` report retained-media presence, so older JSON-only jobs no longer trigger a probing 404.
12. **Demo and jobs endpoints:** `GET /api/demo-info`, `GET /api/demo-audio`, and `GET /api/jobs` support one-click demo reuse and recent-job reload.

---

## 4. Website Implementation & Browser Smoke Verification

Implemented a dependency-free static SPA in `frontend/` (`index.html`, `assets/styles.css`, `assets/app.js`) served directly by FastAPI. It uses Canvas 2D with a 3D projection from `pose_world` landmarks rather than a npm/Three.js build.

Observed browser checks on `http://127.0.0.1:8000/`:

- Health showed `GPU ready`; catalogue showed `459` units, `317` confirmed, `142` unconfirmed, `4` families; `PROVISIONAL` quality rules were displayed in the UI.
- Demo flow fetched `/api/demo-audio` as a Blob, uploaded it as `demo.mp3`, and produced job `5d69ad64eda8`: toast `Generated 7 phrases at 99.4 BPM.`, stats `99.4 BPM`, `BRIGHT`, `7` phrases, `883 frames @ 30 fps`, `00:29.4`, and rhythm summary `50 beats · sam at index 0 (auto_onset_heuristic) · separation: hpss_fallback · bright passage`.
- Playback check: `audio.currentTime` advanced `0` → `0.232319`; status moved `PERFORMING` → `READY` after pause; canvas sample hashes differed between `t=3.2` and `t=9.6`.
- JSON download completed as `nritya-5d69ad64eda8.json` (`1,698,913` bytes) with keys including `audio_analysis`, `affect`, `affect_timeline`, `frames`, and `timeline_events`.
- MP4 export completed in `16,660 ms`; modal computed `display: grid`, `position: fixed`, `zIndex: 60`; video source was `/api/media/5d69ad64eda8/preview.mp4`, `readyState 2`; no console errors observed.
- Upload flow: dropping `data/samples/demo.mp3` into the visible drop zone and generating produced job `957346365ea9`, a blob-backed audio source, the same `99.4 BPM` / `7` phrase stats, and `source_available: true` from `GET /api/choreography/957346365ea9`.
- Older JSON-only job `1c796c440e7c` loaded with `source_available: false`, `audioSrc: null`, `renderDisabled: true`, subtitle `JSON only (no retained audio)`, and no console errors.

The screenshots were captured but page pixels were not visually inspected by the model; UI verification above used DOM, computed CSS, media state, and console evidence.

---

## 5. Final Executable Verification

- `node --check frontend/assets/app.js` -> passed with no output.
- `.venv\Scripts\python.exe -m pytest backend/tests -q` -> **`24 passed`** (latest run `60.39s`).
- The API test module covers affect/energy fields, `affect_timeline`, static frontend serving, demo info/audio, jobs, retained source, MP4 render/preview, and `source_available`.

---

## 6. Honest Scope of the Complete Prototype

- The website performs retrieval, recombination, and time-warping of recorded adavu segments; it does **not** implement a trained generative dance model.
- `affect` labels (`calm`, `flowing`, `bright`, `driving`) are derived from RMS loudness, onset density, and tempo with a documented `basis`; they are not verified emotions or audience-perceived mood.
- No human labels exist (`data/labels/`: `0` files). All beat/strike accuracy statements remain unverified, including small nearest-beat errors.
- Catalogue quality remains `PROVISIONAL`; `ThaThaiThamAdavu` stays opt-in.
- Media, motion, jobs, previews, and the local demo excerpt are gitignored and stay on `D:`.
