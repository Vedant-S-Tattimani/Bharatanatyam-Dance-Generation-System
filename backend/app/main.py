"""
main.py – Step 5: FastAPI service for Nritya Bharatanatyam choreography generation.

Endpoints:
  GET  /api/health                  – Health, CUDA GPU status, and loaded catalogue summary
  GET  /api/catalogue               – List adavu categories, phrase units, and quality rules
  POST /api/catalogue/rebuild       – Rebuild data/catalogue/catalogue.json from extracted motion
  POST /api/choreography/generate   – Analyze uploaded song audio/video & return synchronized choreography
  GET  /api/choreography/{job_id}   – Fetch a previously generated choreography JSON by job ID
  POST /api/choreography/{job_id}/render – Render a saved job into an MP4 animation
  GET  /api/jobs                    – List recent generations
  GET  /api/demo-audio              – Bundled demo excerpt (when present)
  GET  /                            – Nritya web studio (when frontend is installed)
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import torch

from app.audio.analyzer import analyze_audio
from app.catalogue.builder import build_catalogue, load_catalogue
from app.motion.preview import render_choreography_video
from app.planner.choreographer import plan_choreography

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOGUE_PATH = ROOT / "data" / "catalogue" / "catalogue.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "output"
DEFAULT_TMP_DIR = ROOT / "data" / ".tmp"


def _ensure_d_drive_scratch() -> None:
    """Keep scratch files and model caches on D: when the repo lives on D:."""
    if ROOT.drive.upper() == "D:":
        DEFAULT_TMP_DIR.mkdir(parents=True, exist_ok=True)
        torch_dir = ROOT / "models" / "weights" / "torch_cache"
        torch_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(torch_dir))
        os.environ["TMP"] = str(DEFAULT_TMP_DIR)
        os.environ["TEMP"] = str(DEFAULT_TMP_DIR)


def create_app(
    root: Path = ROOT,
    catalogue_path: Path | None = None,
    output_dir: Path | None = None,
) -> FastAPI:
    """Build and configure the FastAPI application instance."""
    _ensure_d_drive_scratch()
    cat_path = catalogue_path or (root / "data" / "catalogue" / "catalogue.json")
    out_dir = output_dir or (root / "data" / "output")
    tmp_dir = root / "data" / ".tmp"
    frontend_dir = root / "frontend"
    assets_dir = frontend_dir / "assets"
    demo_audio_path = root / "data" / "samples" / "demo.mp3"

    app = FastAPI(title="Nritya Bharatanatyam Choreography API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _get_catalogue() -> dict[str, Any]:
        if cat_path.is_file():
            return load_catalogue(cat_path)
        return build_catalogue(root, out_path=cat_path)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        cat_summary: dict[str, Any] = {"loaded": cat_path.is_file(), "total_units": 0, "adavu_categories": []}
        if cat_path.is_file():
            c = load_catalogue(cat_path)
            cat_summary = {
                "loaded": True,
                "total_units": c.get("total_units", 0),
                "adavu_categories": c.get("adavu_categories", []),
            }
        return {
            "status": "ok",
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "catalogue": cat_summary,
            "frontend_available": (frontend_dir / "index.html").is_file(),
            "demo_audio_available": demo_audio_path.is_file(),
        }

    @app.get("/api/catalogue")
    def get_catalogue_summary(include_units: bool = False, include_unconfirmed: bool = True) -> dict[str, Any]:
        c = _get_catalogue()
        if not include_unconfirmed:
            units = [u for u in c.get("units", []) if u.get("confirmed", True)]
            filtered = {**c, "units": units, "total_units": len(units)}
            return filtered if include_units else {k: v for k, v in filtered.items() if k != "units"}
        if include_units:
            return c
        return {k: v for k, v in c.items() if k != "units"}

    @app.post("/api/catalogue/rebuild")
    def rebuild_catalogue_endpoint(
        min_segment_s: float = 2.0,
        min_hand_ratio: float = 0.50,
        max_scale_cv: float = 0.12,
        phrase_beats: int = 8,
    ) -> dict[str, Any]:
        c = build_catalogue(
            root,
            out_path=cat_path,
            min_segment_s=min_segment_s,
            min_hand_ratio=min_hand_ratio,
            max_scale_cv=max_scale_cv,
            target_phrase_beats=phrase_beats,
        )
        return {k: v for k, v in c.items() if k != "units"}

    @app.post("/api/choreography/generate")
    async def generate_choreography(
        file: UploadFile = File(...),
        tala: str = Form("adi"),
        use_demucs: bool = Form(True),
        sam_beat_index: int | None = Form(None),
        tempo_scale: float = Form(1.0),
        offset_ms: float = Form(0.0),
        fps: float = Form(30.0),
        allowed_adavus: str = Form(""),
        include_unconfirmed: bool = Form(False),
        exclude_stems: str = Form(""),
        seed: int = Form(42),
    ) -> dict[str, Any]:
        cat = _get_catalogue()
        if not cat.get("units"):
            raise HTTPException(status_code=400, detail="Catalogue has zero phrase units.")

        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        upload_stem = Path(file.filename or "upload.wav").stem
        suffix = Path(file.filename or "upload.wav").suffix or ".wav"
        tmp_media = tmp_dir / f"{job_id}{suffix}"
        job_dir = out_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        source_media = job_dir / f"source{suffix}"

        # Always exclude matching source clip stem so dataset test songs never reuse their own motion
        excluded_set = {s.strip() for s in exclude_stems.split(",") if s.strip()}
        if upload_stem:
            excluded_set.add(upload_stem)

        try:
            with tmp_media.open("wb") as f_out:
                shutil.copyfileobj(file.file, f_out)
            shutil.copyfile(tmp_media, source_media)
            analysis, _, _ = analyze_audio(
                tmp_media,
                tala_name=tala,
                use_demucs=use_demucs,
                sam_beat_index=sam_beat_index,
                tempo_scale=tempo_scale,
                offset_ms=offset_ms,
            )
            allowed_set = {a.strip() for a in allowed_adavus.split(",") if a.strip()} or None
            choreo = plan_choreography(
                audio_analysis=analysis,
                catalogue=cat,
                root=root,
                fps=fps,
                allowed_adavus=allowed_set,
                include_unconfirmed=include_unconfirmed,
                exclude_stems=excluded_set,
                seed=seed,
            )
        finally:
            tmp_media.unlink(missing_ok=True)

        result = {"job_id": job_id, "audio_analysis": analysis, **choreo}
        (out_dir / f"{job_id}.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
        return {
            "job_id": job_id,
            "song_bpm": result["song_bpm"],
            "tala": result["tala"],
            "separation_backend": result["separation_backend"],
            "beat_source": result["beat_source"],
            "include_unconfirmed": result["include_unconfirmed"],
            "excluded_stems": result["excluded_stems"],
            "fps": result["fps"],
            "duration_s": result["duration_s"],
            "n_frames": result["n_frames"],
            "phrase_count": result["phrase_count"],
            "adavus_used": result["adavus_used"],
            "source_stems_used": result["source_stems_used"],
            "frame_quantization_check_ms": result["frame_quantization_check_ms"],
            "planned_foot_strikes_vs_beats": result["planned_foot_strikes_vs_beats"],
            "transition_metrics": result["transition_metrics"],
            "timeline_events": result["timeline_events"],
            "frames": result["frames"],
        }

    @app.get("/api/choreography/{job_id}")
    def get_choreography_job(job_id: str) -> dict[str, Any]:
        if not job_id.isalnum():
            raise HTTPException(status_code=400, detail="Invalid job_id")
        job_file = out_dir / f"{job_id}.json"
        if not job_file.is_file():
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        data = json.loads(job_file.read_text(encoding="utf-8"))
        data["source_available"] = bool(sorted((out_dir / job_id).glob("source.*")))
        return data

    @app.get("/api/demo-info")
    def demo_info() -> dict[str, Any]:
        """Describe the optional bundled demo audio excerpt."""
        return {
            "available": demo_audio_path.is_file(),
            "filename": demo_audio_path.name if demo_audio_path.is_file() else None,
            "stem": demo_audio_path.stem if demo_audio_path.is_file() else None,
            "description": "Short excerpt from the local adavu dataset, for trying the studio without an upload.",
        }

    @app.get("/api/demo-audio", include_in_schema=False)
    def demo_audio() -> FileResponse:
        if not demo_audio_path.is_file():
            raise HTTPException(status_code=404, detail="No bundled demo audio available")
        return FileResponse(demo_audio_path, media_type="audio/mpeg", filename=demo_audio_path.name)

    @app.get("/api/jobs")
    def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
        """List previously generated choreography jobs, newest first."""
        if not out_dir.is_dir():
            return []
        recent = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        jobs: list[dict[str, Any]] = []
        for job_file in recent[: max(1, min(int(limit), 100))]:
            try:
                data = json.loads(job_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            jobs.append({
                "job_id": data.get("job_id") or job_file.stem,
                "source_file": (data.get("audio_analysis") or {}).get("source_file"),
                "song_bpm": data.get("song_bpm"),
                "tala": data.get("tala"),
                "duration_s": data.get("duration_s"),
                "phrase_count": data.get("phrase_count"),
                "adavus_used": data.get("adavus_used", []),
                "source_available": bool(sorted((out_dir / job_file.stem).glob("source.*"))),
            })
        return jobs

    @app.post("/api/choreography/{job_id}/render")
    def render_choreography_job(
        job_id: str,
        max_seconds: float = 60.0,
        width: int = 960,
        height: int = 540,
    ) -> dict[str, Any]:
        """Render a saved job into an MP4 animation muxed with its source audio."""
        if not job_id.isalnum():
            raise HTTPException(status_code=400, detail="Invalid job_id")
        job_file = out_dir / f"{job_id}.json"
        if not job_file.is_file():
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        sources = sorted((out_dir / job_id).glob("source.*"))
        if not sources:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' has no retained source media")
        job = json.loads(job_file.read_text(encoding="utf-8"))
        video_path = out_dir / job_id / "preview.mp4"
        beats = (job.get("audio_analysis") or {}).get("beats")
        render_choreography_video(
            job,
            sources[0],
            video_path,
            beats=beats,
            width=int(width),
            height=int(height),
            max_seconds=float(max_seconds),
        )
        return {
            "job_id": job_id,
            "video_url": f"/api/media/{job_id}/preview.mp4",
            "duration_s": round(float(min(max_seconds, job.get("duration_s", max_seconds))), 2)
            if max_seconds > 0 else job.get("duration_s"),
        }

    @app.get("/api/media/{job_id}/source", include_in_schema=False)
    def get_source_media(job_id: str) -> FileResponse:
        if not job_id.isalnum():
            raise HTTPException(status_code=400, detail="Invalid job_id")
        sources = sorted((out_dir / job_id).glob("source.*"))
        if not sources:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' has no retained source media")
        media_type = mimetypes.guess_type(sources[0].name)[0] or "application/octet-stream"
        return FileResponse(sources[0], media_type=media_type, filename=sources[0].name)

    @app.get("/api/media/{job_id}/preview.mp4", include_in_schema=False)
    def get_preview_video(job_id: str) -> FileResponse:
        if not job_id.isalnum():
            raise HTTPException(status_code=400, detail="Invalid job_id")
        video_path = out_dir / job_id / "preview.mp4"
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail=f"No preview for job '{job_id}'")
        return FileResponse(video_path, media_type="video/mp4", filename=f"nritya-{job_id}.mp4")

    # Static frontend is registered last so API routes always take precedence.
    if (frontend_dir / "index.html").is_file():

        @app.get("/", include_in_schema=False)
        def frontend_index() -> FileResponse:
            return FileResponse(frontend_dir / "index.html")

        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    else:

        @app.get("/", include_in_schema=False)
        def frontend_missing() -> dict[str, str]:
            return {"message": "Frontend not installed; API is available at /api/health and /docs."}

    return app


app = create_app()
