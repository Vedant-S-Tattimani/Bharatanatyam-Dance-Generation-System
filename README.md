# Bharatanatyam Dance Generation System (Nritya)

An AI-powered system that analyses an uploaded song and generates a synchronized 3D Bharatanatyam dancer performing adavus in time with the music.

> **Status:** Phase 0 (Setup & Data Inventory) complete — awaiting Phase 1 approval.

## Project Layout

- `backend/` — FastAPI server, AI/choreography modules, and tests (`backend/tests/`)
- `frontend/` — Next.js + Three.js 3D dancer interface
- `scripts/` — Dataset inventory, frame inspection, mapping overrides, and extraction tools
- `data/` — Local dataset (`data/adavus/`), MediaPipe models (`data/models/`), sample frames, and extracted motion (kept out of git)
- `docs/` — Inventory reports and project notes

## Setup (Phase 0)

1. Ensure Python 3.12+ and `ffmpeg` / `ffprobe` are installed.
2. Create and activate the virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r backend/requirements-phase0.txt
   ```
3. Copy `.env.example` to `.env` if needed (keep `.env` out of git).
4. Download the dataset locally into `data/adavus/` (gitignored):
   ```powershell
   hf download vibhuti16/bharatnatyam_adavus --repo-type dataset --revision 2c4f81e260114f9015215c0f13188225c8533ca9 --local-dir data/adavus
   ```
   *(If the `hf` CLI is unavailable, use `huggingface_hub.snapshot_download` with `repo_type="dataset"` and `revision="2c4f81e260114f9015215c0f13188225c8533ca9"`.)*
5. Download the two official MediaPipe `.task` model files into `data/models/mediapipe/` (gitignored):
   - **Pose Landmarker (full, float16):**
     `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`
     → `data/models/mediapipe/pose_landmarker_full.task`
   - **Hand Landmarker (float16):**
     `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
     → `data/models/mediapipe/hand_landmarker.task`
   ```powershell
   New-Item -ItemType Directory -Force -Path "data/models/mediapipe" | Out-Null
   curl.exe -sSL -o "data/models/mediapipe/pose_landmarker_full.task" "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
   curl.exe -sSL -o "data/models/mediapipe/hand_landmarker.task" "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
   ```
6. Run the inventory **before** any tests or checks that read `docs/inventory_report.json`:
   ```powershell
   python scripts/inventory.py --data-dir "data/adavus" --overrides "scripts/mapping_overrides.json" --output "docs/inventory_report.json"
   pytest backend/tests/test_inventory.py -v
   ```
