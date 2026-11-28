# Nritya Phase 1 Progress & Decision Log

**Phase 1 Status:** `experimental, accuracy unverified` (NOT PASSED)  
**Phase 2 Website Status:** `implemented, desktop/mobile-width iframe DOM/console-verified` (pixels still visually unverified)\
**Phase 3 Model Training:** `implemented, bounded hand-shape + recorded-join priors` (not a choreography generator)\
**Phase 4 Visual Reference:** `implemented, static atlas inset + opt-in auto-rigged FBX sculpt in WebGL` (Canvas mannequin default; approximate recorded-pose retargeting, not verified choreography)\
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
- **Tests (`backend/tests/`):** `.venv\Scripts\python.exe -m pytest backend/tests -q` -> **`39 passed`** (latest run `61.96s`) with `use_tempo_octave_check=False` default. The added mudra and transition-prior tests cover feature normalization, missing-artifact inertness, metadata loading, landmark-only annotation, boundary summaries, honest hand availability, source-held-out training behavior, planner scoring, and missing/present FBX atlas/rigged-GLB API cases.

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
13. **FBX reference and auto-rig endpoints:** `GET /api/reference-model` reports optional atlas metadata plus `rigged` / `rig`, `GET /api/reference-model/atlas` serves the fixed PNG, `GET /api/reference-model/glb` serves the auto-rigged skinned GLB, and `GET /api/health` includes `reference_model`. The frontend draws the procedural mannequin by default and drives the rig from recorded poses only when `?rig=1` is present; missing artifacts leave generation unchanged.

---

## 4. Website Implementation & Browser Smoke Verification

Implemented a static SPA in `frontend/` (`index.html`, `assets/styles.css`, `assets/portfolio.css`, `assets/app.js`, `assets/dancer3d.js`) served directly by FastAPI, with no npm build step. The default Canvas 2D renderer projects `pose_world` landmarks into a procedural female mannequin; genuine `left_hand_rel` / `right_hand_rel` landmarks are still drawn only when detected. When `?rig=1` is present and the local rigged GLB is available, Three.js r160 (vendored under `frontend/assets/vendor/`) renders the supplied sculpt instead and rotates its deform bones from the same recorded joints; the rig is off by default because its inferred skin weights visibly distort the sculpt.

Observed browser checks on `http://127.0.0.1:8000/`:

- Health showed `GPU ready`; catalogue showed `459` units, `317` confirmed, `142` unconfirmed, `4` families; `PROVISIONAL` quality rules were displayed in the UI.
- Demo flow fetched `/api/demo-audio` as a Blob, uploaded it as `demo.mp3`, and produced job `5d69ad64eda8`: toast `Generated 7 phrases at 99.4 BPM.`, stats `99.4 BPM`, `BRIGHT`, `7` phrases, `883 frames @ 30 fps`, `00:29.4`, and rhythm summary `50 beats · sam at index 0 (auto_onset_heuristic) · separation: hpss_fallback · bright passage`.
- Playback check: `audio.currentTime` advanced `0` → `0.232319`; status moved `PERFORMING` → `READY` after pause; canvas sample hashes differed between `t=3.2` and `t=9.6`.
- JSON download completed as `nritya-5d69ad64eda8.json` (`1,698,913` bytes) with keys including `audio_analysis`, `affect`, `affect_timeline`, `frames`, and `timeline_events`.
- MP4 export completed in `16,660 ms`; modal computed `display: grid`, `position: fixed`, `zIndex: 60`; video source was `/api/media/5d69ad64eda8/preview.mp4`, `readyState 2`; no console errors observed.
- Upload flow: dropping `data/samples/demo.mp3` into the visible drop zone and generating produced job `957346365ea9`, a blob-backed audio source, the same `99.4 BPM` / `7` phrase stats, and `source_available: true` from `GET /api/choreography/957346365ea9`.
- Older JSON-only job `1c796c440e7c` loaded with `source_available: false`, `audioSrc: null`, `renderDisabled: true`, subtitle `JSON only (no retained audio)`, and no console errors.

The screenshots were captured but page pixels were not visually inspected by the model; UI verification above used DOM, computed CSS, media state, and console evidence. After the renderer swap, a DOM/console check confirmed the new canvas aria label and produced no console errors, but the desktop tab was not visible, so live canvas pixels remain visually unverified.

The current rewritten SPA was later checked in live `1440×900` and `390×844` iframe viewports (not physical devices). Both loaded with `motion-ready`, the `Source sculpt · static` button visible, and `aria-pressed` changing `true → false → true` on click. Desktop computed layout used a two-column `.studio-grid`, four-column `.how-grid`, visible `.topnav`, hidden `.menu-toggle`, and sticky process steps; mobile-width layout used single-column grids, hidden `.topnav`, visible `.menu-toggle`, and relative process steps. Both fetched `/api/reference-model` and `/api/reference-model/atlas` with `200`, and no console errors were observed. The desktop canvas produced a non-empty PNG data URL, but pixels were still not visually inspected. Deterministic renderer evidence was obtained separately via `node scripts/validate_renderer.js`: `VERDICT: PASS · 44 fills, 87 strokes, 20 ellipses, 32 mannequin fills, 42 hand-landmark strokes, 1 atlas draws`.

---

## 5. Phase 3: Bounded Mudra Hand-Shape Prior

- **Dataset:** MIT-licensed Hugging Face dataset [`Samarth0710/bharatanatyam-mudra-dataset`](https://huggingface.co/datasets/Samarth0710/bharatanatyam-mudra-dataset), pinned at revision `b3df2f983ef58d25e72d6191d1c4bddbbbbe96ab`. Downloaded locally to `data/hf_datasets/Samarth0710__bharatanatyam-mudra-dataset` via [`download_hf_datasets.py`](file:///d:/dance%20clone/scripts/download_hf_datasets.py); manifest recorded `9 files, 803,013,404 bytes`.
- **Training input:** [`train_mudra_landmark_model.py`](file:///d:/dance%20clone/scripts/train_mudra_landmark_model.py) selected `5040` images from `28` classes and extracted wrist-relative, palm-normalized 21-point xy landmarks from genuine MediaPipe hand detections (original + mirrored), producing a `(8810, 42)` feature matrix. Extraction failures were `{'no_hand': 635, 'bad_landmarks': 0}`.
- **Held-out evaluation:** `GroupShuffleSplit` grouped by image: `7048` train / `1762` test samples. Accuracy `0.7026`, balanced accuracy `0.7046`, macro F1 `0.6947`, weighted F1 `0.6968`. Artifacts: `models/mudra_landmark_classifier.joblib` and `models/mudra_landmark_classifier.json`, both gitignored and reproducible.
- **Catalogue annotation:** [`annotate_catalogue_mudras.py`](file:///d:/dance%20clone/scripts/annotate_catalogue_mudras.py) classified only existing detected `left_hand_rel` / `right_hand_rel` landmarks in the local motion JSONs. Output: `459` annotated units and `25` distinct mudra labels in gitignored `data/catalogue/mudra_annotations.json` (`116,849` bytes). No finger poses were invented.
- **Runtime integration:** `backend/app/motion/mudra.py` wraps the classifier; the planner applies a bounded hand-shape scoring term and emits `mudra_planning`, `mudra_labels`, and `mudra_confidence`. `GET /api/model-info` reports availability, version, metadata, metrics, and annotation coverage; `GET /api/health` includes a compact `mudra_model` summary. Missing local artifacts leave generation working with `mudra_planning.enabled: false`.
- **Honest scope:** This is a hand-shape prior over mudra images, not a paired music-to-expert-choreography model. No such paired corpus was found, so held-out choreography correctness for arbitrary songs is **not claimed** or measured here.

---

## 6. Phase 3b: Bounded Recorded-Join Transition Prior

- **Training input:** [`train_transition_prior.py`](file:///d:/dance%20clone/scripts/train_transition_prior.py) read the `459`-unit catalogue and its local motion JSONs, then built short head/tail boundary summaries using recorded pose velocity, travel velocity, foot height, genuine hand endpoint/velocity continuity, tempo, and duration.
- **Labels:** `404` positive joins were consecutive units from the same source clip (shared endpoint frame, gap `0` or `1`). `990` hard negatives were non-adjacent or cross-source pairs already within `0.25` pose-endpoint RMS. Endpoint distance itself was deliberately excluded from model features because the trivial baseline ROC AUC was `1.0`; the planner already scores endpoint pose and hand distance separately.
- **Held-out evaluation:** `GroupShuffleSplit` by source clip held out `7` of `32` sources. Training had `301` positives / `600` negatives; held-out had `103` positives / `206` negatives. Accuracy `0.9644`, balanced accuracy `0.9684`, ROC AUC `0.9962`, average precision `0.9930`, chance `0.3333`. These numbers measure recorded-adjacency separation, **not** expert choreography quality.
- **Artifacts:** `models/transition_prior.joblib`, `models/transition_prior.json`, and `data/catalogue/transition_features.json` (`459` per-unit summaries). All are local-only and gitignored; generation remains functional when they are absent.
- **Runtime integration:** `backend/app/motion/transition_prior.py` scores candidate joins in a batch from the saved summaries; `_select_unit_sequence` applies only a bounded `(1 - probability) * weight` cost and emits `transition_planning` plus per-event `transition_score`. `GET /api/model-info` and `GET /api/health` report availability/version/summary coverage without claiming choreography correctness.

---

## 7. Meeko-Style Frontend Effects, FBX Reference, and Auto-Rig

- **Reference reconnaissance:** [`meeko_effects_report.md`](file:///d:/dance%20clone/data/scratch/meeko_effects_report.md) recorded observed scroll/visual patterns from `meeko-template.framer.website` without copying code, copy, images, or assets. Frontend changes were additive to the existing Meeko-style rewrite, preserving the recorded-pose mannequin fallback and genuine hand landmarks.
- **Frontend effects:** hero word entrance, `reveal-scale` cards, `slide-accent` notes, sticky desktop process steps, compact scrolled header, and a `Source sculpt · static` stage toggle. All motion effects have reduced-motion opt-outs; the atlas inset can be hidden without changing generated choreography.
- **Renderer validator repair:** `scripts/validate_renderer.js` now stubs `querySelectorAll`, `querySelector`, `focus`, `matchMedia`, `innerWidth`/`innerHeight`, global and window `IntersectionObserver`, `FakeImage`, `drawImage`, and `fillText`. It asserts genuine-hand guards, procedural mannequin helpers, the reference endpoint, the rigged stage switch, the local module import map, and an atlas draw. Observed result: `VERDICT: PASS · 44 fills, 87 strokes, 20 ellipses, 32 mannequin fills, 42 hand-landmark strokes, 1 atlas draws`.
- **FBX conversion:** [`fbx_conversion_report.md`](file:///d:/dance%20clone/data/scratch/fbx_conversion_report.md) documents the supplied `4_Sculpt_19022025.fbx` (21,184,972 bytes) and portable Blender 4.5.0 conversion. The source has 54 meshes, 11 textureless materials, **0 armatures/bones/vertex groups, 0 actions, 0 images**; no source animation was fabricated.
- **Derived static artifacts:** a structurally validated static GLB (34,111,148 bytes, 0 animations/skins/images/textures) and a 2048×2048 RGBA atlas (4×4, 16 tiles of 512 px, row-major, transparent background). Frame `0` is the `0.0°` turntable angle and is the only frame drawn in the stage inset.
- **Local auto-rig:** `data/scratch/blender_autorig_fbx.py` (untracked/local-only) creates a 21-joint humanoid armature, runs ARMATURE_AUTO bone heat with a deterministic nearest-segment fallback for 18 meshes, and exports `bharatanatyam_dancer_rigged.glb` (62,614,544 bytes, 1 skin, 19 deform joints, 89 skinned primitives, 0 baked animations) plus `bharatanatyam_rig_manifest.json`. BezierCurve.002 was omitted by the exporter because it has no skin/primitive.
- **Runtime retargeting:** `frontend/assets/dancer3d.js` loads the GLB with vendored Three.js r160 GLTFLoader, normalizes GLTFLoader's bone-name sanitization (`upper_arm.L` -> `upper_armL`), and rotates the 19 deform bones from recorded `pose_world` joint directions. The nose stands in for the head joint, hips/chest/neck are synthesized from recorded landmarks, and no finger pose is invented. If the rig fails or is absent, the Canvas mannequin remains active.
- **API/UI integration:** `GET /api/reference-model` exposes atlas metadata, `rigged`, and the retarget manifest; `GET /api/reference-model/atlas` serves the fixed PNG; `GET /api/reference-model/glb` serves the rigged GLB; `GET /api/health` includes `reference_model`. The stage shows a `Rigged sculpt · recorded pose` badge only when `?rig=1` opts into the rig; the Canvas mannequin remains the default. The atlas inset is still labelled static reference art and is never used as choreography, pose, or hand data.

---

## 8. Final Executable Verification

- `node --check frontend/assets/app.js` and a module check of `frontend/assets/dancer3d.js` -> passed with no output.
- `node scripts/validate_renderer.js` -> `VERDICT: PASS · 44 fills, 87 strokes, 20 ellipses, 32 mannequin fills, 42 hand-landmark strokes, 1 atlas draws`.
- `.venv\Scripts\python.exe -m pytest backend/tests -q` -> **`39 passed`** (latest run `61.96s`).
- `GET http://127.0.0.1:8000/api/reference-model` -> `200` with `available: true`, `frame_count: 16`, `grid: {"columns":4,"rows":4}`, `tile_size: 512`, and `rigged: true`; `/api/reference-model/atlas` and `/api/reference-model/glb` returned `200` in the live browser (GLB `62,614,544` bytes in `7,972 ms`). Desktop/mobile-width iframe checks observed no console errors and a working reference toggle.
- After the default revert, `http://127.0.0.1:8000/` was checked with `motion-ready`, a `451×418` Canvas labelled `Animated Bharatanatyam dancer generated from recorded movement`, an empty `dancer3dMount`, a hidden rig badge, no GLB request, and no console errors.
- The opt-in URL `http://127.0.0.1:8000/?rig=1` was checked separately: the rig badge became visible, a `408×377` WebGL canvas appeared, the GLB returned `200` in `7,972 ms`, and no console errors were observed.
- Earlier opt-in rig check: `modulePresent: true`, `rigReady: true`, `mountActive: true`, `rigStatusVisible: true`, `19` bound deform bones, `408×377` WebGL canvas, no console errors. `snapshot()` quaternions changed between recorded frames `0` and `90` for `hips`, `spine`, `upper_arm.L`, `forearm.R`, `thigh.L`, and `shin.R`. Stage-crop screenshots measured `37.9%` of pixels changing by more than `12/255` between the two frames. The visible distortion led to the rig being made opt-in rather than default.
- The API tests cover affect/energy fields, `affect_timeline`, static frontend serving, demo info/audio, jobs, retained source, MP4 render/preview, `source_available`, mudra and recorded-join planner metadata, compact response fields, `/api/model-info`, and missing/present atlas/rigged-GLB artifacts.
- Mudra training and catalogue annotation were executed on the local dataset and are reported in section 5; their generated artifacts are local-only and gitignored.

---

## 9. Honest Scope of the Complete Prototype

- The website performs retrieval, recombination, and time-warping of recorded adavu segments; it does **not** implement a trained generative dance model.
- The default and fallback female mannequin is procedural Canvas 2D geometry over real projected landmarks. When opted in with `?rig=1`, the supplied FBX sculpt is auto-rigged in WebGL and driven by recorded body/wrist directions, but it is not a photorealistic character and its skin weights are approximate. Missing hand landmarks remain `null` and are not invented.
- The trained classifier ranks hand shapes among recorded poses; it is not an expert choreography model and does not guarantee correct choreography for every song.
- `affect` labels (`calm`, `flowing`, `bright`, `driving`) are derived from RMS loudness, onset density, and tempo with a documented `basis`; they are not verified emotions or audience-perceived mood.
- No human labels exist (`data/labels/`: `0` files). All beat/strike accuracy statements remain unverified, including small nearest-beat errors.
- Catalogue quality remains `PROVISIONAL`; `ThaThaiThamAdavu` stays opt-in.
- The FBX atlas is static reference art: its 16 frames are camera angles around one pose, not dance mudras or rhythm. The separately derived auto-rig follows recorded pose directions only in the opt-in WebGL stage and does not affect planned choreography, prove cultural correctness, or synthesize finger/hand landmarks. Backend MP4 export remains on the existing 2D renderer.
- Media, motion, jobs, previews, Hugging Face datasets, imported FBX/GLB/atlas artifacts, scratch reports, trained artifacts, and the local demo excerpt are gitignored and stay on `D:`.
