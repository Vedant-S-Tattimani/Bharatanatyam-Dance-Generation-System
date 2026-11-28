# Bharatanatyam Dance Generation System (Nritya)

Nritya is a runnable research prototype that listens to uploaded music, estimates its pulse/tala/energy, and composes a Bharatanatyam animation from recorded adavu motion. A FastAPI server exposes both the JSON API and a dependency-free browser studio.

> **Status:** End-to-end prototype implemented and browser-smoke-tested. Musical accuracy is **unverified** (`0` human labels in `data/labels/`), and catalogue quality rules remain `PROVISIONAL`.

## What works today

- Upload an audio or video file, or use the optional local demo excerpt.
- Detect tempo, beats, sam/tala position, an energy contour, and a transparent intensity heuristic (`calm` / `flowing` / `bright` / `driving`). This is **not** an emotion classifier.
- Plan beat-locked 8-beat phrases from the local adavu catalogue, with a small energy-aware bias toward vigorous or lyrical units.
- Watch a Canvas 2D, 3D-projected dancer perform the planned frames in sync with the source audio.
- Inspect beats, energy, phrase timeline, and quality metrics; download the full choreography JSON.
- Export a standalone MP4 (first 45 seconds at 960×540 in the browser) when the source media was retained.
- Reload recent jobs; older JSON-only jobs are shown without a broken audio source or disabled-by-context export.

The dancer is assembled from retrieved and time-warped recorded adavu segments. It is **not** produced by a trained dance diffusion model.

## Project layout

- `backend/` — FastAPI app, audio analysis, catalogue builder, choreography planner, motion preview renderer, and tests.
- `frontend/` — Static SPA (`index.html`, `assets/styles.css`, `assets/app.js`) served directly by FastAPI at `/`. No npm build step and no Three.js dependency.
- `scripts/` — Inventory, catalogue build, extraction, beat diagnostics, scoring, and render helpers.
- `data/` — Local dataset, MediaPipe models, extracted motion, catalogue, jobs, and sample audio. Gitignored.
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
7. Start the server:
   ```powershell
   .venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
   ```
8. Open <http://127.0.0.1:8000/>. The API docs are at <http://127.0.0.1:8000/docs>.

### Demo clip

`GET /api/demo-info` reports whether `data/samples/demo.mp3` is present. The UI fetches that file through `GET /api/demo-audio` and uploads it as a real multipart file named `demo.mp3`; its stem is auto-excluded from retrieval so the planner cannot reuse its own source adavu motion. This excerpt is local-only and gitignored, so a fresh clone may need to supply its own file or simply use an upload.

## Using the website

1. In **Source**, drop/choose an audio or video file, or click **Use 30-second demo excerpt**.
2. In **Rhythm & mapping**, choose tala, tempo interpretation, sam index, offset, frame rate, seed, Demucs separation, unconfirmed adavus, and adavu filters.
3. Click **Generate choreography**. Progress steps report upload, analysis, planning, and frame preparation.
4. Use the transport bar to play, pause, seek, change speed, or loop. The canvas follows `audio.currentTime`.
5. Review **Overview**, **Rhythm**, **Quality**, and **Phrase timeline** tabs. **JSON** downloads the full job; **Export MP4** renders and opens the result in a modal.

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
| `GET` | `/api/demo-info` | Demo excerpt metadata and availability |
| `GET` | `/api/demo-audio` | Demo MP3 bytes |
| `POST` | `/api/choreography/{job_id}/render` | Render retained media to MP4 (`max_seconds`, `width`, `height`) |
| `GET` | `/api/media/{job_id}/source` | Retained uploaded source media |
| `GET` | `/api/media/{job_id}/preview.mp4` | Rendered MP4 |
| `GET` | `/` | Static browser studio |
| `GET` | `/docs` | FastAPI OpenAPI UI |

`POST /api/choreography/generate` returns a compact response (job metadata, metrics, timeline events, and frames); callers fetch `GET /api/choreography/{job_id}` for `audio_analysis` and affect fields. `source_available` and MP4 export depend on `data/jobs/{job_id}/source.*` being retained. Full field details are in [`docs/api_changes.md`](docs/api_changes.md).

## Verification evidence

Observed in this workspace after the final implementation edits:

- `node --check frontend/assets/app.js` — passed, no output.
- `.venv\Scripts\python.exe -m pytest backend/tests -q` — `24 passed` (latest run `60.39s`).
- Browser demo generation produced job `5d69ad64eda8`: `99.4 BPM`, `bright`, `7` phrases, `883 frames @ 30 fps`, `50 beats`, and `hpss_fallback` separation.
- Browser upload flow produced job `957346365ea9` from a dropped `demo.mp3`: toast `Generated 7 phrases at 99.4 BPM.`, blob-backed audio, and `source_available: true`.
- Browser playback advanced audio time and changed canvas frame hashes; JSON download completed; MP4 export completed in `16,660 ms` with a playable modal video; older JSON-only jobs loaded without console errors.

## Accuracy and licensing caveats

- There are `0` human-verified beat or foot-strike labels. Small nearest-beat errors do **not** prove musical accuracy; see [`docs/progress.md`](docs/progress.md).
- `affect` and `energy_curve` are additive heuristics over RMS loudness, onset density, and tempo. They are documented as heuristics in the API response (`basis`).
- The catalogue's quality rules are `PROVISIONAL`; `ThaThaiThamAdavu` is unconfirmed and disabled by default.
- Dataset: [`vibhuti16/bharatnatyam_adavus`](https://huggingface.co/datasets/vibhuti16/bharatnatyam_adavus), CC BY-NC-ND 4.0, non-commercial research use only. Media, motion, jobs, and exports remain local.
