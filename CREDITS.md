# Credits & Dataset License

## Bharatanatyam Adavus Dataset
- **Dataset:** `vibhuti16/bharatnatyam_adavus`
- **Curated by:** Vibhuti Shah
- **Pinned Revision:** `2c4f81e260114f9015215c0f13188225c8533ca9`
- **License:** **CC BY-NC-ND 4.0** (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International)
- **Usage & Storage Constraints:** Non-commercial research/educational use only. Source videos, extracted pose/hand motion JSONs, audio files, previews, catalogue artifacts, and model weights are kept strictly local and gitignored.

## Bharatanatyam Mudra Hand-Shape Dataset
- **Dataset:** [`Samarth0710/bharatanatyam-mudra-dataset`](https://huggingface.co/datasets/Samarth0710/bharatanatyam-mudra-dataset)
- **Pinned Revision:** `b3df2f983ef58d25e72d6191d1c4bddbbbbe96ab`
- **License:** **MIT**
- **Local Storage:** Downloaded only under gitignored `data/hf_datasets/Samarth0710__bharatanatyam-mudra-dataset`; the download manifest is `data/hf_datasets/manifest.json` (`9 files, 803,013,404 bytes`). Nothing from this dataset is committed.
- **Derived Model:** `scripts/train_mudra_landmark_model.py` extracts genuine MediaPipe 21-point hand landmarks and trains a bounded mudra hand-shape classifier. The generated `models/mudra_landmark_classifier.joblib` and `models/mudra_landmark_classifier.json` are gitignored local artifacts, reproducible from the pinned dataset and script.
- **Scope:** The model is a hand-shape prior for ranking recorded adavu poses. It is not a paired music-to-expert-choreography dataset and does not support a claim of correct choreography for every song.

## Supplied Bharatanatyam Dancer FBX
- **Source:** local archive `D:\downloads\bharathanatyam-dancer.zip`, containing `4_Sculpt_19022025.fbx` (21,184,972 bytes).
- **Provenance:** no license, author, or attribution file was present in the supplied archive. Do not treat this as a licensed redistribution; the original FBX and all derived files are kept local and gitignored.
- **Derived outputs:** `data/imported_models/bharathanatyam-dancer/derived/bharatanatyam_dancer.glb`, `bharatanatyam_turntable_atlas_4x4_512.png`, `bharatanatyam_dancer_rigged.glb`, and `bharatanatyam_rig_manifest.json`, produced with portable Blender 4.5.0. Details and measured limitations are in `data/scratch/fbx_conversion_report.md`; the local-only auto-rig script is `data/scratch/blender_autorig_fbx.py`.
- **Scope:** the source is a static, textureless sculpt with no armature, weights, actions, or keyframes. The atlas is a 16-angle camera turntable of one pose and remains a labelled static reference inset. The rigged GLB is locally auto-rigged and, when the studio is opened with `?rig=1`, driven by recorded adavu `pose_world` body/wrist directions in the browser; its skin weights and bind pose are approximate, and finger poses are not synthesized. The default stage remains the procedural Canvas mannequin. Neither artifact is choreography supervision or proof of correct Bharatanatyam for arbitrary music.

## Bundled Browser Library

- **Three.js r160:** `frontend/assets/vendor/three.module.js`, `frontend/assets/vendor/loaders/GLTFLoader.js`, and `frontend/assets/vendor/utils/BufferGeometryUtils.js`, used only for the optional rigged-sculpt renderer. License: MIT, copied verbatim to `frontend/assets/vendor/THREE_LICENSE.txt` from <https://github.com/mrdoob/three.js/blob/r160/LICENSE>. No npm build step or network CDN is required at runtime.

