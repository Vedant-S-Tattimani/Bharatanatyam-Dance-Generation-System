# Backend API Field & Contract Changes (for `frontend/`)

Last updated: Phase 2 website/retained-media revision (`backend/app/main.py`, `backend/app/audio/analyzer.py`, `backend/app/planner/choreographer.py`, `backend/app/motion/preview.py`)

---

## Summary of Changes Since Initial Step 5 API

1. **`measure_4b_timeline_vs_beats_ms` renamed to `frame_quantization_check_ms`**
   - Reason: It verifies timeline event snapping against the `30 fps` grid (frame quantization), not musical synchronization of dancer footwork.
2. **Added `planned_foot_strikes_vs_beats`**
   - Compares mapped source foot strikes inside scheduled phrase units against song beats at the same tempo, with explicit `beat_source` (`"verified"` vs `"unverified"`).
3. **`ThaThaiThamAdavu` marked unconfirmed & switchable (`include_unconfirmed`)**
   - `ThaThaiThamAdavu` phrases carry `"confirmed": false` because `ThaThaiTham1–6.mp4` list `KorvaiAdavu` in `adavu_name2`.
   - `POST /api/choreography/generate` defaults to `include_unconfirmed=false` (excludes `ThaThaiThamAdavu` unless toggled on).
4. **Automatic & explicit source-clip exclusion (`exclude_stems` / `excluded_stems`)**
   - `POST /api/choreography/generate` automatically adds the uploaded file's stem to `exclude_stems` (plus any comma-separated stems passed in `exclude_stems`), preventing self-retrieval when testing with a dataset clip.
5. **Constant-FPS frame spacing fixed**
   - `frames` now lie strictly on `t = t0 + f / fps` with zero skipped boundary steps, so `duration_s == (n_frames - 1) / fps`.
6. **Updated `transition_metrics`**
   - Reports separate median and max (`torso` units) for `body`, `hands` (real detected pairs only, never invented), and recalculated stance `foot_sliding`.

7. **Static website served by FastAPI**
    - `GET /` returns `frontend/index.html` and `/assets` is mounted when the frontend is installed. `GET /api/health` now includes `frontend_available`.
8. **Additive affect / energy analysis (`NEW`)**
    - `audio_analysis` now includes `energy_curve` (32 normalised RMS bins) and `affect` (`label`, `intensity_score`, `energy_mean`, `energy_peak`, `dynamic_range`, `onset_density_per_10s`, `basis`). Each `beats[]` record also has `energy`.
    - The `basis` string explicitly says this is a heuristic, **not** an emotion classifier.
9. **Energy-aware planning (`NEW`)**
    - `POST /api/choreography/generate` results include top-level `affect` and `affect_timeline[]` (`time_s`, `target_energy`, `adavu`). Planner selection uses a bounded `ADAVU_ENERGY` bias; this is a transparent heuristic, not learned choreography.
10. **Retained source media and MP4 export (`NEW`)**
    - Uploaded bytes are retained at `data/jobs/{job_id}/source.*`. `POST /api/choreography/{job_id}/render` renders `preview.mp4`; `GET /api/media/{job_id}/source` and `GET /api/media/{job_id}/preview.mp4` expose the files.
    - `render` accepts `max_seconds`, `width`, `height` and returns `video_url` plus `duration_s`. The browser exports the first `45` seconds at `960x540`; the endpoint defaults to `60` seconds at `960x540`.
11. **`source_available` (`NEW`)**
    - `GET /api/choreography/{job_id}` and every item from `GET /api/jobs` report whether retained `source.*` media exists. Clients should use this field instead of probing `/api/media/{job_id}/source` and logging a 404 for older JSON-only jobs.
12. **Local demo endpoints (`NEW`)**
    - `GET /api/demo-info` and `GET /api/demo-audio` expose the optional local `data/samples/demo.mp3`. The file is gitignored; `health.demo_audio_available` reports its presence.
13. **Job listing (`NEW`)**
    - `GET /api/jobs?limit=20` returns recent saved jobs with `source_file`, tempo, tala, duration, phrase count, adavus used, and `source_available`.
14. **Compact generate response**
    - `POST /api/choreography/generate` intentionally omits `audio_analysis`, `affect`, and `affect_timeline`; fetch `GET /api/choreography/{job_id}` for those fields.

---

## Endpoint Reference

### `GET /api/health`
Response:
```json
{
  "status": "ok",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
  "catalogue": {
    "loaded": true,
    "total_units": 459,
    "adavu_categories": ["KudithuMettuAdavu", "Nattadavu", "ThaThaiThamAdavu", "ThaiyaThaihiAdavu"]
  },
  "frontend_available": true,
  "demo_audio_available": true
}
```

### `GET /api/catalogue`
Query parameters:
- `include_units: bool = false`
- `include_unconfirmed: bool = true` *(NEW — pass `false` to inspect confirmed-only units/counts)*

Response fields:
- `version`: `"1.1"`
- `quality_rules_status`: `"PROVISIONAL (pending user approval on previews and beat verification)"`
- `quality_rules`: `{ min_segment_s, min_hand_ratio, max_scale_cv, target_phrase_beats }`
- `unconfirmed_categories_note`: `{ "ThaThaiThamAdavu": "..." }`
- `adavu_categories`: all indexed categories (`string[]`)
- `confirmed_adavu_categories`: confirmed categories only (`string[]`)
- `total_units`, `confirmed_units`, `unconfirmed_units` (`int`)
- `per_adavu_summary`: keyed by adavu `{ confirmed, clips_evaluated, clips_included, clean_segments, phrase_units, clean_dur_s }`
- `excluded_unconfirmed_aliases`, `excluded_clips`
- `units` *(only when `include_units=true`)*: each unit has `confirmed: bool`, `beat_source: "verified" | "unverified"`, `foot_strikes`, `start_pose`, `end_pose`, `start_left_hand`, `start_right_hand`, `end_left_hand`, `end_right_hand`.

### `POST /api/catalogue/rebuild`
Query parameters: `min_segment_s=2.0`, `min_hand_ratio=0.50`, `max_scale_cv=0.12`, `phrase_beats=8`. Returns the updated `/api/catalogue` summary without `units`.

### `POST /api/choreography/generate` (`multipart/form-data`)
Form fields:
- `file` *(required)*: audio or video file
- `tala: string = "adi"`
- `use_demucs: bool = true` *(deterministic `--shifts 0`)*
- `sam_beat_index: int | null = null`
- `tempo_scale: float = 1.0`
- `offset_ms: float = 0.0`
- `fps: float = 30.0`
- `allowed_adavus: string = ""` *(comma-separated filter)*
- `include_unconfirmed: bool = false` *(NEW — opt-in for `ThaThaiThamAdavu`)*
- `exclude_stems: string = ""` *(NEW — comma-separated stems; upload stem is also auto-excluded)*
- `seed: int = 42`

Response fields (compact — `audio_analysis`, `affect`, and `affect_timeline` are **not** included here; fetch `GET /api/choreography/{job_id}` for them):
- Top-level:
  - `job_id: string`
  - `song_bpm: float`
  - `tala: string`
  - `separation_backend: "demucs" | "hpss_fallback"`
  - `beat_source: "verified" | "unverified"` *(NEW)*
  - `include_unconfirmed: bool` *(NEW)*
  - `excluded_stems: string[]` *(NEW)*
  - `fps: float`
  - `duration_s: float` *(matches `(n_frames - 1) / fps`)*
  - `n_frames: int`
  - `phrase_count: int`
  - `adavus_used: string[]`
  - `source_stems_used: string[]` *(NEW)*
- `frame_quantization_check_ms` *(RENAMED from `measure_4b_timeline_vs_beats_ms`)*:
  `{ count, median_ms, mean_ms, p95_ms, max_ms }`
- `planned_foot_strikes_vs_beats` *(NEW)*:
  `{ strike_count, beat_count, median_ms, mean_ms, max_ms, median_frames, max_frames, beat_source, beat_period_ms, err_frac_of_period, same_tempo_random_baseline_ms }`
- `transition_metrics`:
  `{ transition_count, body: {count, median_torso, max_torso}, hands: {count, median_torso, max_torso}, foot_sliding: {count, median_torso, max_torso} }`
- `timeline_events[]`:
  `{ time_s, unit_id, adavu, confirmed, source_stem, is_phrase_start, song_beat_index }`
- `frames[]`:
  `{ f, t, adavu, source_stem, confirmed, pose_world (33x3), travel_x, jump_y, foot_l_y, foot_r_y, left_hand_rel (21x3 | null), right_hand_rel (21x3 | null) }`
  > **Important:** `left_hand_rel` and `right_hand_rel` are `null` whenever no genuine hand detection existed in the source frame. Do **not** synthesize finger poses for `null` frames in the 3D frontend.

### `GET /api/choreography/{job_id}`
- Rejects non-alphanumeric `job_id` with HTTP `400`; returns HTTP `404` if missing; otherwise returns the full saved job JSON.
- Adds `audio_analysis`:
  - Existing fields: `source_file`, `duration_s`, `sample_rate`, `tala`, `tala_beats`, `separation_backend`, `separation_detail`, `tempo_octave_check` (`"unchanged" | "folded_half_silent_parity" | "disabled"`), `tempo_scale`, `offset_ms`, `bpm_median`, `first_sam_index`, `sam_source`, `beat_count`, `onset_count`, `perc_onsets_s`, `beats[]`.
  - `energy_curve`: `float[32]`, normalised RMS energy bins across the song.
  - `affect`: `{ label: "calm" | "flowing" | "bright" | "driving", intensity_score, energy_mean, energy_peak, dynamic_range, onset_density_per_10s, basis }`. `basis` states that this is a heuristic, not an emotion classifier.
  - Each `beats[]` record also carries `energy`.
- Adds top-level `affect` (same object) and `affect_timeline[]`: `{ time_s, target_energy, adavu }`, one entry per scheduled phrase.
- Adds `source_available: bool`, computed from retained `data/jobs/{job_id}/source.*`. Use this instead of probing `GET /api/media/{job_id}/source`.

### `GET /api/jobs`
Query: `limit: int = 20` (clamped to `1..100`). Returns newest-first `dict[]`:
```json
{
  "job_id": "string",
  "source_file": "string | null",
  "song_bpm": 99.4,
  "tala": "Adi",
  "duration_s": 29.4,
  "phrase_count": 7,
  "adavus_used": ["KudithuMettuAdavu"],
  "source_available": true
}
```

### `GET /api/demo-info`
Returns `{ available, filename, stem, description }` for the optional local `data/samples/demo.mp3`. The file is gitignored; `available: false` is normal on a fresh clone.

### `GET /api/demo-audio`
Returns the demo MP3 bytes (`audio/mpeg`), or HTTP `404` when no demo file exists. The browser fetches this as a Blob and uploads it with filename `demo.mp3`.

### `POST /api/choreography/{job_id}/render`
Query parameters: `max_seconds: float = 60.0`, `width: int = 960`, `height: int = 540`. Requires retained source media; otherwise HTTP `404`. Returns:
```json
{
  "job_id": "string",
  "video_url": "/api/media/{job_id}/preview.mp4",
  "duration_s": 29.4
}
```
The browser currently requests `max_seconds=45`, `width=960`, `height=540`.

### `GET /api/media/{job_id}/source`
Returns the retained uploaded media, or HTTP `404` if absent. Prefer `source_available` checks before calling.

### `GET /api/media/{job_id}/preview.mp4`
Returns a previously rendered MP4, or HTTP `404` if none exists.

### `GET /`
Serves the static browser studio when `frontend/index.html` exists; otherwise returns a short JSON message pointing to `/api/health` and `/docs`. Static assets are served from `/assets`.

