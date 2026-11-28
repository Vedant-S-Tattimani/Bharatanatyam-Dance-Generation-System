"""Integration and unit tests for Steps 3-5: Catalogue, Planner, Renderer, and FastAPI endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.audio.analyzer import SR, analyze_audio  # noqa: E402
from app.catalogue.builder import build_catalogue, extract_clean_segments  # noqa: E402
from app.main import create_app  # noqa: E402
from app.motion.preview import render_choreography_video  # noqa: E402
from app.planner.choreographer import plan_choreography  # noqa: E402


def _make_synthetic_project(tmp_path: Path) -> Path:
    """Create minimal synthetic inventory, motion, and audio fixtures on disk."""
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "data" / "motion").mkdir(parents=True)
    (tmp_path / "data" / "audio").mkdir(parents=True)

    fps = 30.0
    n_frames = 180  # 6 seconds
    beat_times = [round(i * 0.5, 4) for i in range(12)]

    def _make_pose_frames(pose_shift: float, keep_hands: bool = True, break_mid: bool = False) -> list[dict]:
        frames = []
        for f in range(n_frames):
            valid = not (break_mid and 80 <= f <= 95)
            pw = np.zeros((33, 3), dtype=float)
            pw[:, 0] = np.linspace(-0.2, 0.2, 33) + pose_shift
            pw[:, 1] = np.linspace(-0.5, 0.5, 33) + 0.05 * np.sin(f * 0.2)
            hand = np.zeros((21, 3), dtype=float).tolist() if (keep_hands and f % 3 != 0) else None
            frames.append({
                "f": f,
                "t": round(f / fps, 4),
                "valid": valid,
                "interpolated": False,
                "travel_x": round(0.1 * np.sin(f * 0.05), 4),
                "jump_y": 0.01,
                "foot_l_y": 0.02,
                "foot_r_y": 0.02,
                "pose_world": np.round(pw, 4).tolist() if valid else None,
                "pose_img": None,
                "left_hand_img": None,
                "right_hand_img": None,
                "left_hand_rel": hand if valid else None,
                "right_hand_rel": hand if valid else None,
            })
        return frames

    clips_inv = [
        {"filename": "Nattadavu1.mp4", "adavu_canonical": "Nattadavu", "adavu_alternates": [], "broken": False},
        {"filename": "ThaiyaThaihi1.mp4", "adavu_canonical": "ThaiyaThaihiAdavu", "adavu_alternates": [], "broken": False},
        {"filename": "ThaThaiTham1.mp4", "adavu_canonical": "ThaThaiThamAdavu", "adavu_alternates": ["KorvaiAdavu"], "broken": False},
        {"filename": "KorvaiAdavu1.mp4", "adavu_canonical": "ThaThaiThamAdavu", "adavu_alternates": ["KorvaiAdavu"], "broken": False},
        {"filename": "KudithuMettuAdavu3.mp4", "adavu_canonical": "KudithuMettuAdavu", "adavu_alternates": [], "broken": False},
    ]
    (tmp_path / "docs" / "inventory_report.json").write_text(json.dumps({"clips": clips_inv}), encoding="utf-8")
    (tmp_path / "frontend" / "assets").mkdir(parents=True)
    (tmp_path / "frontend" / "index.html").write_text("<html>Nritya</html>", encoding="utf-8")
    (tmp_path / "frontend" / "assets" / "app.js").write_text("window.ok = true;", encoding="utf-8")
    (tmp_path / "data" / "samples").mkdir(parents=True)
    (tmp_path / "data" / "samples" / "demo.mp3").write_bytes(b"demo-fixture")

    specs = [
        ("Nattadavu1", 0.0, True, False, 0.04),
        ("ThaiyaThaihi1", 0.04, True, False, 0.04),
        ("ThaThaiTham1", 0.08, True, False, 0.05),
        ("KorvaiAdavu1", 0.0, True, False, 0.04),  # Excluded as unconfirmed Korvai alias
        ("KudithuMettuAdavu3", 0.0, False, False, 0.05),  # Excluded: 0% hands
    ]
    for stem, shift, hands, brk, cv in specs:
        motion = {
            "fps": fps,
            "n_frames": n_frames,
            "duration_s": 6.0,
            "depth_travel_reliability": {"torso_scale_cv": cv},
            "foot_strikes": [
                {"frame": f_st, "time_s": round(f_st / fps, 4), "side": "left" if k % 2 == 0 else "right"}
                for k, f_st in enumerate(range(15, n_frames, 30))
            ],
            "frames": _make_pose_frames(shift, keep_hands=hands, break_mid=brk),
        }
        (tmp_path / "data" / "motion" / f"{stem}.json").write_text(json.dumps(motion), encoding="utf-8")
        analysis = {
            "bpm_median": 120.0,
            "tala": "Adi",
            "tala_beats": 8,
            "beats": [{"time_s": t, "tala_pos": (i % 8) + 1, "is_sam": (i % 8 == 0)} for i, t in enumerate(beat_times)],
        }
        (tmp_path / "data" / "audio" / f"{stem}_analysis.json").write_text(json.dumps(analysis), encoding="utf-8")

    return tmp_path


def test_extract_clean_segments_splits_gaps_and_checks_hands():
    frames = []
    for i in range(150):  # 5s @ 30fps
        valid = not (60 <= i <= 75)
        has_hand = i < 60
        frames.append({
            "f": i,
            "valid": valid,
            "left_hand_rel": [[0.0, 0.0, 0.0]] * 21 if has_hand else None,
            "right_hand_rel": [[0.0, 0.0, 0.0]] * 21 if has_hand else None,
        })
    segs = extract_clean_segments(frames, fps=30.0, min_segment_s=2.0, min_hand_ratio=0.50)
    assert len(segs) == 1
    assert segs[0]["start_frame"] == 0 and segs[0]["end_frame"] == 59


def test_catalogue_planner_api_and_render_end_to_end(tmp_path: Path):
    proj = _make_synthetic_project(tmp_path)
    cat = build_catalogue(proj, target_phrase_beats=4, workers=1)

    assert "KorvaiAdavu1.mp4" in cat["excluded_unconfirmed_aliases"]
    assert cat["confirmed_units"] > 0 and cat["unconfirmed_units"] > 0
    assert cat["per_adavu_summary"]["ThaThaiThamAdavu"]["confirmed"] is False
    assert all(u["source_stem"] != "KudithuMettuAdavu3" for u in cat["units"])

    # Synthetic 120 BPM WAV
    wav = tmp_path / "Nattadavu1.wav"
    sr = SR
    dur = 9.0
    t_arr = np.zeros(int(dur * sr), dtype=np.float32)
    for b_t in np.arange(0.5, dur - 0.2, 0.5):
        s = int(b_t * sr)
        tt = np.arange(int(0.03 * sr)) / sr
        t_arr[s : s + len(tt)] += 0.8 * np.sin(2 * np.pi * 440.0 * tt) * np.exp(-tt * 60.0)
    sf.write(str(wav), t_arr, sr)

    analysis, _, _ = analyze_audio(wav, tala_name="adi", use_demucs=False)
    assert len(analysis["energy_curve"]) == 32
    assert analysis["affect"]["label"] in {"calm", "flowing", "bright", "driving"}
    assert analysis["affect"]["basis"].startswith("heuristic")

    # 1. Test include_unconfirmed=False & exclude_stems={"Nattadavu1"}
    choreo_conf = plan_choreography(
        analysis, cat, root=proj, fps=30.0, include_unconfirmed=False, exclude_stems={"Nattadavu1"}
    )
    assert "ThaThaiThamAdavu" not in choreo_conf["adavus_used"]
    assert "Nattadavu1" not in choreo_conf["source_stems_used"]
    assert choreo_conf["frame_quantization_check_ms"]["median_ms"] <= 25.0
    assert choreo_conf["frame_quantization_check_ms"]["max_ms"] <= 60.0
    assert choreo_conf["planned_foot_strikes_vs_beats"]["strike_count"] > 0
    assert choreo_conf["transition_metrics"]["foot_sliding"]["count"] >= 0
    assert len(choreo_conf["affect_timeline"]) == choreo_conf["phrase_count"]
    # Strict constant-FPS spacing
    dts = [round(choreo_conf["frames"][i + 1]["t"] - choreo_conf["frames"][i]["t"], 4) for i in range(len(choreo_conf["frames"]) - 1)]
    assert all(abs(dt - 1.0 / 30.0) < 0.002 for dt in dts)
    assert abs(choreo_conf["duration_s"] - (choreo_conf["n_frames"] - 1) / 30.0) <= 0.002
    assert any(fr["left_hand_rel"] is None for fr in choreo_conf["frames"])

    # 2. Test include_unconfirmed=True enables ThaThaiThamAdavu
    choreo_unconf = plan_choreography(
        analysis, cat, root=proj, fps=30.0, include_unconfirmed=True, exclude_stems={"Nattadavu1"}
    )
    assert choreo_unconf["include_unconfirmed"] is True

    # 3. Render choreography MP4 with audio
    out_mp4 = tmp_path / "rendered_dance.mp4"
    render_choreography_video(choreo_conf, wav, out_mp4, beats=analysis["beats"], width=320, height=180)
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0

    # 4. Test FastAPI endpoints (auto-excludes uploaded Nattadavu1.wav stem!)
    app = create_app(root=proj)
    client = TestClient(app)
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["catalogue"]["total_units"] == cat["total_units"]

    r_cat = client.get("/api/catalogue?include_unconfirmed=false")
    assert r_cat.status_code == 200
    assert r_cat.json()["total_units"] == cat["confirmed_units"]

    with wav.open("rb") as fh:
        r_gen = client.post(
            "/api/choreography/generate",
            files={"file": ("Nattadavu1.wav", fh, "audio/wav")},
            data={"tala": "adi", "use_demucs": "false", "fps": "30", "include_unconfirmed": "false"},
        )
    assert r_gen.status_code == 200
    body = r_gen.json()
    assert "Nattadavu1" not in body["source_stems_used"]
    assert "ThaThaiThamAdavu" not in body["adavus_used"]
    assert body["frame_quantization_check_ms"]["median_ms"] <= 25.0

    r_get = client.get(f"/api/choreography/{body['job_id']}")
    assert r_get.status_code == 200
    assert r_get.json()["audio_analysis"]["affect"]["label"] in {"calm", "flowing", "bright", "driving"}
    assert r_get.json()["source_available"] is True
    assert client.get("/api/choreography/..bad").status_code == 400

    # 5. Website, demo, jobs, retained media, and export endpoints
    assert r_health.json()["frontend_available"] is True
    assert "<html>" in client.get("/").text
    assert "window.ok" in client.get("/assets/app.js").text
    r_demo_info = client.get("/api/demo-info")
    assert r_demo_info.status_code == 200 and r_demo_info.json()["available"] is True
    assert client.get("/api/demo-audio").content == b"demo-fixture"
    jobs_summary = client.get("/api/jobs").json()
    job_ids = [j["job_id"] for j in jobs_summary]
    assert body["job_id"] in job_ids
    assert next(j for j in jobs_summary if j["job_id"] == body["job_id"])["source_available"] is True
    assert client.get(f"/api/media/{body['job_id']}/source").status_code == 200

    orphan = proj / "data" / "output" / "orphan123.json"
    orphan.write_text(json.dumps({"job_id": "orphan123", "song_bpm": 90.0}), encoding="utf-8")
    assert client.get("/api/choreography/orphan123").json()["source_available"] is False
    assert next(j for j in client.get("/api/jobs").json() if j["job_id"] == "orphan123")["source_available"] is False
    r_render = client.post(
        f"/api/choreography/{body['job_id']}/render?max_seconds=2&width=160&height=90"
    )
    assert r_render.status_code == 200
    assert r_render.json()["video_url"] == f"/api/media/{body['job_id']}/preview.mp4"
    r_preview = client.get(r_render.json()["video_url"])
    assert r_preview.status_code == 200 and len(r_preview.content) > 0
