"""
save_sample_frames.py – Save annotated sample frames for visual inspection of
passing/failing full-body and mudra-ready checks, plus MandiAdavu2.mp4.

Outputs PNGs into data/sample_frames/ (gitignored).
"""

import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ROOT = Path(__file__).resolve().parent.parent
POSE_MODEL = ROOT / "data" / "models" / "mediapipe" / "pose_landmarker_full.task"
HAND_MODEL = ROOT / "data" / "models" / "mediapipe" / "hand_landmarker.task"
DATA_DIR = ROOT / "data" / "adavus"
OUT_DIR = ROOT / "data" / "sample_frames"

SAMPLES = [
    ("pass_both_KorvaiAdavu2.1.mp4", "videos/Korvai Adavu2.1.mp4", "PASS BOTH (ThaThaiThamAdavu)"),
    ("pass_both_Nattadavu1.mp4", "videos/Nattadavu1.mp4", "PASS BOTH (Nattadavu)"),
    ("fail_body_only_MandiAdavu1.1.mp4", "videos/Mandi Adavu1.1.mp4", "FAIL BODY ONLY (MandiAdavu)"),
    ("fail_body_only_Karthariadavu1.mp4", "videos/Karthariadavu1.mp4", "FAIL BODY ONLY (KarthariAdavu)"),
    ("fail_mudra_only_Thattadavu1.4.mp4", "videos/Thattadavu1.4.mp4", "FAIL MUDRA ONLY (Thattadavu)"),
    ("fail_mudra_only_MandiAdavu1.mp4", "videos/MandiAdavu1.mp4", "FAIL MUDRA ONLY (bbox<40px, MandiAdavu)"),
    ("fail_both_Thattadavu1.2.mp4", "videos/Thattadavu1.2.mp4", "FAIL BOTH (Thattadavu)"),
    ("fail_both_UthsangaAdavu.mp4", "videos/Uthsanga Adavu.mp4", "FAIL BOTH (UtsangaAdavu)"),
    ("confirm_label_MandiAdavu2.mp4", "videos/MandiAdavu2.mp4", "CHECK LABEL (MandiAdavu2.mp4)"),
]

KEY_BODY = (11, 12, 23, 24, 25, 26, 27, 28)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )

    saved = []
    with (
        mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm,
        mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm,
    ):
        for out_name, rel_path, label in SAMPLES:
            vpath = DATA_DIR / rel_path
            if not vpath.is_file():
                print(f"MISSING: {vpath}")
                continue
            cap = cv2.VideoCapture(str(vpath))
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Pick middle frame (50% of clip)
            mid_idx = fc // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
            ok, bgr = cap.read()
            cap.release()
            if not ok or bgr is None:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pres = pose_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            annot = bgr.copy()

            body_ok = False
            hand_info = []
            if pres.pose_landmarks:
                lm = pres.pose_landmarks[0]
                body_ok = all(
                    (lm[j].visibility or 0.0) >= 0.5 and -0.02 <= lm[j].x <= 1.02 and -0.02 <= lm[j].y <= 1.02
                    for j in KEY_BODY
                )
                for j in KEY_BODY:
                    px, py = int(lm[j].x * w), int(lm[j].y * h)
                    vis = lm[j].visibility or 0.0
                    col = (0, 255, 0) if vis >= 0.5 and 0 <= px <= w and 0 <= py <= h else (0, 0, 255)
                    cv2.circle(annot, (px, py), 8, col, -1)

                sh_y = (lm[11].y + lm[12].y) * 0.5 * h
                hp_y = (lm[23].y + lm[24].y) * 0.5 * h
                torso = max(abs(hp_y - sh_y), 80.0)
                csize = int(max(160, torso * 0.65))
                for wi, ii, side in ((15, 19, "L"), (16, 20, "R")):
                    if (lm[wi].visibility or 0.0) < 0.4:
                        continue
                    cx = int(((lm[wi].x + lm[ii].x) * 0.5) * w)
                    cy = int(((lm[wi].y + lm[ii].y) * 0.5) * h)
                    x0, y0 = max(0, cx - csize // 2), max(0, cy - csize // 2)
                    x1, y1 = min(w, cx + csize // 2), min(h, cy + csize // 2)
                    cv2.rectangle(annot, (x0, y0), (x1, y1), (255, 200, 0), 2)
                    if x1 - x0 >= 40 and y1 - y0 >= 40:
                        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
                        hres = hand_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
                        if hres.hand_landmarks:
                            hl = hres.hand_landmarks[0]
                            xs = [p.x * (x1 - x0) + x0 for p in hl]
                            ys = [p.y * (y1 - y0) + y0 for p in hl]
                            bpx = max(max(xs) - min(xs), max(ys) - min(ys))
                            hand_info.append(f"{side}:{bpx:.0f}px")
                            for px, py in zip(xs, ys):
                                cv2.circle(annot, (int(px), int(py)), 3, (0, 255, 255), -1)

            # Banner at top
            cv2.rectangle(annot, (0, 0), (w, 60), (20, 20, 20), -1)
            txt = f"{label} | frame {mid_idx}/{fc} ({w}x{h}) | body_frame_ok={body_ok} | hands={','.join(hand_info) or 'none'}"
            cv2.putText(annot, txt, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # Downscale to 960px wide for compact preview
            scale = 960 / w
            preview = cv2.resize(annot, (960, int(h * scale)))
            out_path = OUT_DIR / f"{out_name}.png"
            cv2.imwrite(str(out_path), preview)
            saved.append(str(out_path))
            print(f"Saved {out_path.name}: {txt}")

    print(f"Total saved: {len(saved)}")


if __name__ == "__main__":
    main()

