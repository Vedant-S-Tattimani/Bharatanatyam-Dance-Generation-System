# Nritya — AI-Powered Bharatanatyam Dance Generation System

An AI-powered system that analyses an uploaded song and generates a synchronized 3D Bharatanatyam dancer performing adavus in time with the music.

> **Status:** Phase 0 (Setup & Data Inventory) in progress.

## Setup (Phase 0)

1. Ensure Python 3.12+ and `ffmpeg` / `ffprobe` are installed.
2. Create and activate the virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements-phase0.txt
   ```
3. Copy `.env.example` to `.env` if needed (keep `.env` out of git).
4. Download the dataset (`vibhuti16/bharatnatyam_adavus`, commit `2c4f81e260114f9015215c0f13188225c8533ca9`) locally into `data/adavus/` (gitignored).
5. Run tests and inventory:
   ```powershell
   pytest tests/test_inventory.py -v
   python scripts/inventory.py --data-dir "data/adavus" --output "data/inventory_report.json"
   ```

## Credits & Asset Provenance

- **Bharatanatyam Adavus Dataset:** `vibhuti16/bharatnatyam_adavus` curated by **Vibhuti Shah**, licensed under **CC BY-NC-ND 4.0** (non-commercial use only; videos, extracted motion, and trained weights kept local and gitignored). Commit hash: `2c4f81e260114f9015215c0f13188225c8533ca9`.
- **Reference Pipeline (Read-Only):** [`O-tech477/Indian-Classical-Dance-Step-Generation-From-Song`](https://github.com/O-tech477/Indian-Classical-Dance-Step-Generation-From-Song) — no license file found; kept outside this repository for reference only.
- **3D Dancer Reference:** [`divycoders/NrityaVaani`](https://github.com/divycoders/NrityaVaani) — MIT License (code). Third-party 3D GLB assets inside NrityaVaani carry separate Sketchfab metadata and are treated as unconfirmed until verified on their source pages.

