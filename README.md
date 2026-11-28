# Bharatanatyam Dance Generation System (Nritya)

Nritya is a runnable research prototype that listens to uploaded music, estimates its pulse/tala/energy, and composes a Bharatanatyam animation from recorded adavu motion. A FastAPI server exposes both the JSON API and a no-build browser studio with a vendored Three.js renderer for the optional rigged sculpt.

> **Status:** End-to-end prototype implemented and browser-smoke-tested. The default stage uses the proven procedural Canvas mannequin from real `pose_world` landmarks. A locally auto-rigged version of the supplied FBX sculpt is available only with `?rig=1` because its inferred skin weights distort the sculpt. Mudra and recorded-join priors are integrated. Musical accuracy is **unverified** (`0` human labels in `data/labels/`), and catalogue quality rules remain `PROVISIONAL`.

## What works today

- Upload an audio or video file, or use the optional local demo excerpt.
- Detect tempo, beats, sam/tala position, an energy contour, and a transparent intensity heuristic (`calm` / `flowing` / `bright` / `driving`). This is **not** an emotion classifier.
- Plan beat-locked 8-beat phrases from the local adavu catalogue, with a small energy-aware bias toward vigorous or lyrical units.
- Watch the procedural female mannequin, projected from real `pose_world` landmarks, perform the planned frames in sync with the source audio. Genuine `left_hand_rel` / `right_hand_rel` finger landmarks are drawn only when detected; missing hands are never invented.
- Optionally rank phrases with a locally trained mudra classifier that reads those same detected landmarks. It is a bounded hand-shape prior, not an expert choreography generator.
- Optionally rank phrase joins with a locally trained, source-held-out prior over recorded boundary-window motion. It is a bounded join-ordering cost, not expert choreography supervision, and it never synthesizes frames or hands.
- The `Source sculpt · static` toggle controls the frame-0 turntable inset. The atlas inset is reference art only, never choreography or hand data. A locally auto-rigged GLB can be inspected with `?rig=1`, but it is off by default because the inferred skin weights visibly distort the unweighted source sculpt.
- Inspect beats, energy, phrase timeline, mudra-prior status, recorded-join-prior status, and quality metrics; download the full choreography JSON.
- Export a standalone MP4 (first 45 seconds at 960×540 in the browser) when the source media was retained.
- Reload recent jobs; older JSON-only jobs are shown without a broken audio source or disabled-by-context export.

The dancer is assembled from retrieved and time-warped recorded adavu segments. It is **not** produced by a trained dance diffusion model, and no paired music-to-expert-Bharatanatyam choreography corpus was found. The trained model only selects among real recorded hand shapes.

## Project layout

- `backend/` — FastAPI app, audio analysis, catalogue builder, choreography planner, motion preview renderer, and tests.
- `frontend/` — Static SPA (`index.html`, `assets/styles.css`, `assets/portfolio.css`, `assets/app.js`, `assets/nritya-mark.svg`, `assets/dancer3d.js`) served directly by FastAPI at `/`. No npm build step; Three.js r160 and GLTFLoader are vendored under `assets/vendor/` with their MIT license.
- `scripts/` — Inventory, catalogue build, extraction, beat diagnostics, scoring, render helpers, licensed Hugging Face download, mudra training, catalogue annotation, recorded-join-prior training, and deterministic renderer validation.
- `data/imported_models/bharathanatyam-dancer/derived/` — Local-only static GLB, 4×4 transparent turntable atlas, auto-rigged skinned GLB, and retarget manifest derived from the supplied FBX. Gitignored.
- `data/` — Local dataset, MediaPipe models, extracted motion, catalogue, boundary summaries, jobs, annotations, sample audio, imported model source, and scratch reports. Gitignored.
- `models/` — Locally trained mudra classifier and recorded-join prior with metadata. Generated files are gitignored and reproducible from `scripts/`.
- `docs/` — Inventory report, API contract notes, and progress/decision log.
- `backend/requirements-phase0.txt` — Minimal inventory/test dependencies.
- `backend/requirements-phase1.txt` — Full runtime dependencies (FastAPI, librosa, Demucs, MediaPipe, PyTorch, pytest, etc.).

## Setup

1. Ensure Python 3.12+ and `ffmpeg` / `ffprobe` are installed and on `PATH`.
2. Create and activate the virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r backend\requirements-phase1.txt
   ```
   For inventory-only work, `backend\requirements-phase0.txt` is sufficient.
3. Copy `.env.example` to `.env` if needed, and keep `.env` out of git.
4. Download the dataset locally into `data/adavus/` (gitignored):
   ```powershell
   hf download vibhuti16/bharatnatyam_adavus --repo-type dataset --revision 2c4f81e260114f9015215c0f13188225c8533ca9 --local-dir data/adavus
   ```
   If the `hf` CLI is unavailable, use `huggingface_hub.snapshot_download` with `repo_type="dataset"` and the same revision.
5. Download the two official MediaPipe `.task` models into `data/models/mediapipe/` (gitignored):
   - Pose: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`
   - Hand: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`

   ```powershell
   New-Item -ItemType Directory -Force -Path "data/models/mediapipe" | Out-Null
   curl.exe -sSL -o "data/models/mediapipe/pose_landmarker_full.task" "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
   curl.exe -sSL -o "data/models/mediapipe/hand_landmarker.task" "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
   ```
6. Run the inventory before checks that read `docs/inventory_report.json`, then build the catalogue:
   ```powershell
   python scripts\inventory.py --data-dir "data/adavus" --overrides "scripts/mapping_overrides.json" --output "docs/inventory_report.json"
   python scripts\build_catalogue.py --out "data/catalogue/catalogue.json"
   pytest backend\tests -q
   ```
   The current local catalogue contains `459` units: `317` confirmed and `142` unconfirmed `ThaThaiThamAdavu`. The unconfirmed family is opt-in in the UI/API.
7. Optional: download the MIT-licensed mudra dataset, train the bounded hand-shape prior, and annotate the catalogue. All artifacts stay under gitignored `data/` and `models/` on `D:`.
   ```powershell
   .venv\Scripts\python.exe scripts\download_hf_datasets.py --only mudra
   .venv\Scripts\python.exe scripts\train_mudra_landmark_model.py --max-per-class 180
   .venv\Scripts\python.exe scripts\annotate_catalogue_mudras.py
   ```
   If these mudra files are absent, generation still works and `mudra_planning.enabled` is `false`; `/api/model-info` reports the missing artifacts.
8. Train the bounded recorded-join prior from the local catalogue and motion JSONs:
   ```powershell
   .venv\Scripts\python.exe scripts\train_transition_prior.py
   ```
   It writes `models/transition_prior.joblib`, `models/transition_prior.json`, and gitignored `data/catalogue/transition_features.json`. If any are absent, generation still works and `transition_planning.enabled` is `false`.
9. Optional: if the supplied FBX-derived assets are present in `data/imported_models/bharathanatyam-dancer/derived/`, the server exposes `/api/reference-model`, `/api/reference-model/atlas`, and `/api/reference-model/glb`. The studio shows the labelled static frame-0 inset. The auto-rigged dancer is opt-in via `?rig=1` and is off by default because its inferred skin weights visibly distort the sculpt. The source FBX has no armature, actions, or textures; the rig was added locally with `data/scratch/blender_autorig_fbx.py` and is approximate. See `data/scratch/fbx_conversion_report.md`. Missing artifacts simply fall back to the procedural mannequin.
10. Start the server:
   ```powershell
   .venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
   ```
11. Open <http://127.0.0.1:8000/>. The API docs are at <http://127.0.0.1:8000/docs>.

### Demo clip

`GET /api/demo-info` reports whether `data/samples/demo.mp3` is present. The UI fetches that file through `GET /api/demo-audio` and uploads it as a real multipart file named `demo.mp3`; its stem is auto-excluded from retrieval so the planner cannot reuse its own source adavu motion. This excerpt is local-only and gitignored, so a fresh clone may need to supply its own file or simply use an upload.

## Using the website

1. In **Source**, drop/choose an audio or video file, or click **Use 30-second demo excerpt**.
2. In **Rhythm & mapping**, choose tala, tempo interpretation, sam index, offset, frame rate, seed, Demucs separation, unconfirmed adavus, and adavu filters.
3. Click **Generate choreography**. Progress steps report upload, analysis, planning, and frame preparation.
4. Use the transport bar to play, pause, seek, change speed, or loop. The stage follows `audio.currentTime`.
5. The procedural mannequin remains the default stage. Add `?rig=1` to inspect the locally auto-rigged sculpt, which shows the **Rigged sculpt · recorded pose** badge when available; it is off by default because of visible skin-weight distortion. Click **Source sculpt · static** to hide or show the labelled FBX atlas inset; it never changes the generated choreography.
6. Review **Overview**, **Rhythm**, **Quality**, and **Phrase timeline** tabs. **JSON** downloads the full job; **Export MP4** renders and opens the result in a modal.

If a saved job has no retained source media, its subtitle reads `JSON only (no retained audio)`, its audio source is `null`, and MP4 export is disabled.

## API summary

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Status, CUDA/GPU, catalogue summary, frontend and demo availability |
| `GET` | `/api/catalogue` | Catalogue summary; `include_units`, `include_unconfirmed` |
| `POST` | `/api/catalogue/rebuild` | Rebuild the catalogue from local extracted motion |
| `POST` | `/api/choreography/generate` | Upload media and generate a saved choreography job |
| `GET` | `/api/choreography/{job_id}` | Full saved job, including `audio_analysis`, `affect`, `affect_timeline`, frames, and `source_available` |
| `GET` | `/api/jobs` | Recent jobs, newest first, with `source_available` |
| `GET` | `/api/model-info` | Mudra and recorded-join prior availability, metadata, held-out metrics, and annotation coverage |
| `GET` | `/api/reference-model` | FBX atlas metadata plus `rigged` / `rig` manifest (`available: false` when artifacts are absent) |
| `GET` | `/api/reference-model/atlas` | Transparent 4×4 turntable atlas PNG bytes (`404` when absent) |
| `GET` | `/api/reference-model/glb` | Auto-rigged skinned GLB bytes (`404` when absent) |
| `GET` | `/api/demo-info` | Demo excerpt metadata and availability |
| `GET` | `/api/demo-audio` | Demo MP3 bytes |
| `POST` | `/api/choreography/{job_id}/render` | Render retained media to MP4 (`max_seconds`, `width`, `height`) |
| `GET` | `/api/media/{job_id}/source` | Retained uploaded source media |
| `GET` | `/api/media/{job_id}/preview.mp4` | Rendered MP4 |
| `GET` | `/` | Static browser studio |
| `GET` | `/docs` | FastAPI OpenAPI UI |

`POST /api/choreography/generate` returns a compact response (job metadata, metrics, `mudra_planning`, `transition_planning`, timeline events, and frames); timeline events carry `mudra_labels`, `mudra_confidence`, and `transition_score`. Callers fetch `GET /api/choreography/{job_id}` for `audio_analysis` and affect fields. `source_available` and MP4 export depend on `data/jobs/{job_id}/source.*` being retained. Full field details are in [`docs/api_changes.md`](docs/api_changes.md).

## Verification evidence

Observed in this workspace after the final implementation edits:

- Pre-work push was verified at `50de383` on `main` / `origin/main`; the mannequin, dataset, and training work below remains uncommitted.
- `node --check frontend/assets/app.js` and a module check of `frontend/assets/dancer3d.js` — passed, no output.
- `node scripts/validate_renderer.js` — `VERDICT: PASS · 44 fills, 87 strokes, 20 ellipses, 32 mannequin fills, 42 hand-landmark strokes, 1 atlas draws`.
- `.venv\Scripts\python.exe -m pytest backend/tests -q` — `39 passed` (latest run `61.96s`), including missing/present atlas and rigged-GLB API cases.
- `GET http://127.0.0.1:8000/api/reference-model` — `200` with `available: true`, `frame_count: 16`, `grid: {"columns":4,"rows":4}`, `tile_size: 512`, and `rigged: true`; `/api/reference-model/atlas` and `/api/reference-model/glb` both returned `200` in the live browser (GLB `62,614,544` bytes, `7,972 ms`).
- Mudra training used `5040` images from `28` classes, produced a `(8810, 42)` feature matrix, and recorded held-out grouped-by-image metrics: accuracy `0.7026`, balanced accuracy `0.7046`, macro F1 `0.6947`, weighted F1 `0.6968`.
- Recorded-join training used `404` recorded adjacent joins and `990` hard negatives from `32` source clips, with a source-grouped split (`301/600` train, `103/206` held out) over `11` boundary-flow features. Held-out metrics: accuracy `0.9644`, balanced accuracy `0.9684`, ROC AUC `0.9962`, average precision `0.9930`, chance `0.3333`. The separately reported endpoint-distance baseline ROC AUC was `1.0`, which is why endpoint distance is excluded from model features.
- Catalogue annotation produced `459` annotated units with `25` distinct mudra labels from existing detected landmarks.
- Earlier browser flows (before the renderer swap) produced jobs `5d69ad64eda8` and `957346365ea9`, with JSON download and MP4 export in `16,660 ms`.
- The rewritten SPA was checked in live `1440×900` and `390×844` iframe viewports (not physical devices): both loaded with `motion-ready`, the reference button visible with text `Source sculpt · static`, and `aria-pressed` changing `true → false → true` on click. Desktop computed layout had a two-column `.studio-grid`, four-column `.how-grid`, visible `.topnav`, hidden `.menu-toggle`, and sticky process steps; mobile-width layout had single-column grids, hidden `.topnav`, visible `.menu-toggle`, and relative process steps. No console errors were observed.
- After reverting the default stage, `http://127.0.0.1:8000/` was checked with `motion-ready`, a `451×418` Canvas labelled `Animated Bharatanatyam dancer generated from recorded movement`, an empty `dancer3dMount`, a hidden rig badge, no `/api/reference-model/glb` request, and no console errors. The metadata and atlas endpoints returned `200`.
- The opt-in URL `http://127.0.0.1:8000/?rig=1` was checked separately: the rig badge became visible, a `408×377` WebGL canvas appeared in `dancer3dMount`, the GLB returned `200` in `7,972 ms`, and no console errors were observed.
- An earlier opt-in rig verification observed `modulePresent: true`, `rigReady: true`, `mountActive: true`, `rigStatusVisible: true`, a `408×377` WebGL canvas, `19` bound deform bones, and no console errors. `snapshot()` quaternions changed between recorded frames `0` and `90` for all six sampled bones (`hips`, `spine`, `upper_arm.L`, `forearm.R`, `thigh.L`, `shin.R`). Screenshot pixel analysis of the stage crop measured `37.9%` of pixels changing by more than `12/255` between those two frames, but the visible distortion led to the rig being made opt-in rather than default.

## Accuracy and licensing caveats

- There are `0` human-verified beat or foot-strike labels. Small nearest-beat errors do **not** prove musical accuracy; see [`docs/progress.md`](docs/progress.md).
- `affect` and `energy_curve` are additive heuristics over RMS loudness, onset density, and tempo. They are documented as heuristics in the API response (`basis`).
- The catalogue's quality rules are `PROVISIONAL`; `ThaThaiThamAdavu` is unconfirmed and disabled by default.
- No paired music-to-expert Bharatanatyam choreography dataset was found. The mudra model is a hand-shape prior and the recorded-join model is a join-ordering prior over local motion; neither guarantees "the right choreography for every song" and no held-out choreography correctness claim is made.
- Recorded-join metrics measure separation of recorded adjacency from hard negatives, not aesthetic or cultural correctness. The model excludes endpoint pose distance because recorded adjacent units share a frame; it does not synthesize poses or hand landmarks.
- The supplied dancer FBX is a static, textureless sculpt with no armature, weights, actions, or keyframes. The locally auto-rigged GLB can be inspected in WebGL with `?rig=1` and follows recorded body/wrist directions, but its bone heat/nearest-segment weights are approximate, its bind pose is not a neutral T-pose, and finger poses are not synthesized. The default studio stage is the procedural Canvas mannequin; backend MP4 export also uses the existing 2D renderer. The atlas is still a 16-angle camera turntable of one pose, not Bharatanatyam motion. The source archive did not include a license or attribution file, so all FBX-derived files remain local and gitignored.
- Dataset: [`vibhuti16/bharatnatyam_adavus`](https://huggingface.co/datasets/vibhuti16/bharatnatyam_adavus), CC BY-NC-ND 4.0, non-commercial research use only. Media, motion, jobs, and exports remain local.
- Mudra training dataset: [`Samarth0710/bharatanatyam-mudra-dataset`](https://huggingface.co/datasets/Samarth0710/bharatanatyam-mudra-dataset), MIT, pinned at `b3df2f983ef58d25e72d6191d1c4bddbbbbe96ab`. See [`CREDITS.md`](CREDITS.md).
