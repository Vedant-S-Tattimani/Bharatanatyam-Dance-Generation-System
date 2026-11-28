/* Rigged FBX sculpt renderer, driven by recorded pose_world landmarks. */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

let renderer = null;
let scene = null;
let camera = null;
let root = null;
let armature = null;
let manifest = null;
let mount = null;
let ready = false;
const bind = new Map();
const restDirections = new Map();
const restHip = new THREE.Vector3();
const tmpParentQuat = new THREE.Quaternion();
const tmpDelta = new THREE.Quaternion();
const tmpDesired = new THREE.Quaternion();

function fromBlender(values) {
  return new THREE.Vector3(values[0], values[2], -values[1]);
}

// GLTFLoader removes dots from Object3D names: "upper_arm.L" -> "upper_armL".
// Keep a few variants so older/other exporters with "_" suffixes also work.
function nodeNameKeys(name) {
  const value = String(name);
  const spaced = value.replace(/\s/g, '_');
  return [...new Set([
    value,
    spaced.replace(/\./g, ''),
    spaced.replace(/\./g, '_'),
    spaced.replace(/[\[\]:]/g, '_').replace(/\./g, ''),
  ])];
}

function finiteVec(values) {
  if (!Array.isArray(values) || values.length < 3) return null;
  const x = Number(values[0]);
  const y = Number(values[1]);
  const z = Number(values[2]);
  if (![x, y, z].every(Number.isFinite)) return null;
  return [x, y, z];
}

function mid(a, b) {
  return a.clone().add(b).multiplyScalar(0.5);
}

function buildTargets(pose) {
  if (!manifest || !Array.isArray(pose) || pose.length < 33) return null;
  const idx = manifest.joint_indices || {};
  const raw = {};
  for (const [name, i] of Object.entries(idx)) {
    raw[name] = finiteVec(pose[i]);
    if (!raw[name]) return null;
  }
  const headY = raw.nose[1];
  const footY = (raw.left_foot[1] + raw.right_foot[1]) * 0.5;
  const poseHeight = Math.abs(footY - headY);
  if (poseHeight < 0.1) return null;
  const scale = Number(manifest.model_height_m || 1.48) / poseHeight;
  const lift = footY * scale;
  const depth = -(Number(manifest.depth_offset_m) || 0);
  const conv = (p) => new THREE.Vector3(p[0] * scale, -p[1] * scale + lift, -p[2] * scale + depth);

  const points = {};
  for (const [name, p] of Object.entries(raw)) points[name] = conv(p);
  points.hip_center = mid(points.left_hip, points.right_hip);
  points.chest = mid(points.left_shoulder, points.right_shoulder);
  // MediaPipe nose is the closest available head proxy in the recorded poses.
  points.head = points.nose;
  points.neck = points.chest.clone().lerp(points.head, 0.38);
  return points;
}

function updateFrame(frame) {
  if (!ready || !frame || !Array.isArray(frame.pose_world)) return;
  const points = buildTargets(frame.pose_world);
  if (!points) return;

  armature.position.copy(points.hip_center).sub(restHip);
  armature.updateMatrixWorld(true);

  for (const name of manifest.bone_names) {
    const entry = bind.get(name);
    const restDir = restDirections.get(name);
    const spec = (manifest.bone_targets || {})[name];
    if (!entry || !restDir || !spec) continue;
    const from = points[spec.from];
    const to = points[spec.to];
    if (!from || !to) continue;
    const targetDir = to.clone().sub(from);
    if (targetDir.lengthSq() < 1e-10) continue;
    targetDir.normalize();

    tmpDelta.setFromUnitVectors(restDir, targetDir);
    tmpDesired.copy(tmpDelta).multiply(entry.quat);
    const bone = entry.bone;
    if (bone.parent && bone.parent.isBone) {
      bone.parent.getWorldQuaternion(tmpParentQuat);
      bone.quaternion.copy(tmpParentQuat.invert().multiply(tmpDesired));
    } else {
      bone.quaternion.copy(tmpDesired);
    }
    bone.updateMatrixWorld(true);
  }
  renderer.render(scene, camera);
}

function resize(width, height) {
  if (!renderer || !camera) return;
  const w = Math.max(1, Math.round(width));
  const h = Math.max(1, Math.round(height));
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.position.z = 2.35 + Math.max(0, 1.05 - camera.aspect) * 0.75;
  camera.lookAt(0, 0.78, 0);
  camera.updateProjectionMatrix();
  if (ready) renderer.render(scene, camera);
}

async function load(options) {
  const { mount: mountEl, url, manifest: loadedManifest } = options || {};
  if (!mountEl || !url || !loadedManifest) throw new Error('Missing rig load options');
  if (!window.WebGLRenderingContext) throw new Error('WebGL is not available');
  mount = mountEl;
  manifest = loadedManifest;

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
  renderer.setClearColor(0x0c101d, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  mount.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50);
  camera.position.set(0, 0.86, 2.35);
  camera.lookAt(0, 0.78, 0);

  scene.add(new THREE.HemisphereLight(0xfff0d0, 0x1b1230, 2.2));
  const key = new THREE.DirectionalLight(0xffd7a0, 2.4);
  key.position.set(1.6, 2.4, 2.2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x7fd7c0, 1.1);
  fill.position.set(-1.7, 1.4, 1.2);
  scene.add(fill);

  const gltf = await new GLTFLoader().loadAsync(url);
  root = gltf.scene;
  scene.add(root);
  root.updateMatrixWorld(true);

  const bonesByName = new Map();
  root.traverse((obj) => {
    if (!obj.isBone) return;
    for (const key of nodeNameKeys(obj.name)) bonesByName.set(key, obj);
  });
  armature = bonesByName.get('root') || null;
  if (!armature || !armature.isBone) {
    for (const obj of bonesByName.values()) {
      if (obj.isBone) { armature = obj; break; }
    }
  }
  if (!armature) throw new Error('Rig armature was not found in GLB');

  for (const name of manifest.bone_names || []) {
    const bone = nodeNameKeys(name).map((key) => bonesByName.get(key)).find(Boolean);
    if (!bone || !bone.isBone) continue;
    bind.set(name, {
      bone,
      quat: bone.getWorldQuaternion(new THREE.Quaternion()).clone(),
    });
    const rest = manifest.rest_direction && manifest.rest_direction[name];
    if (Array.isArray(rest)) restDirections.set(name, fromBlender(rest).normalize());
  }
  if (!bind.size || restDirections.size < 10) {
    const missing = (manifest.bone_names || []).filter((name) => !bind.has(name));
    throw new Error(
      `Rig manifest does not match GLB bones (bound=${bind.size}, rest=${restDirections.size}, missing=${missing.join(',') || 'none'})`
    );
  }
  const hip = manifest.rest_positions && manifest.rest_positions.hips;
  if (!hip || !Array.isArray(hip.head)) throw new Error('Rig manifest is missing rest hips');
  restHip.copy(fromBlender(hip.head));

  ready = true;
  mount.classList.add('is-active');
  renderer.render(scene, camera);
  return { boneCount: bind.size };
}

function snapshot(names) {
  if (!ready || !armature) return null;
  const requested = Array.isArray(names) && names.length ? names : (manifest?.bone_names || []);
  const bones = {};
  for (const name of requested) {
    const entry = bind.get(name);
    if (!entry) continue;
    const q = entry.bone.quaternion;
    bones[name] = [q.x, q.y, q.z, q.w].map((v) => Number(v.toFixed(6)));
  }
  return {
    ready,
    boneCount: bind.size,
    armature: {
      x: Number(armature.position.x.toFixed(6)),
      y: Number(armature.position.y.toFixed(6)),
      z: Number(armature.position.z.toFixed(6)),
    },
    bones,
  };
}

function setVisible(active) {
  if (mount) mount.style.visibility = active ? 'visible' : 'hidden';
}

function dispose() {
  ready = false;
  bind.clear();
  restDirections.clear();
  if (mount) {
    mount.classList.remove('is-active');
    mount.innerHTML = '';
  }
  if (renderer) {
    renderer.dispose();
    renderer.domElement.remove();
  }
  renderer = null;
  scene = null;
  camera = null;
  root = null;
  armature = null;
  manifest = null;
  mount = null;
}

window.NrityaDancer3D = {
  load,
  update: updateFrame,
  resize,
  snapshot,
  setVisible,
  dispose,
  get ready() { return ready; },
};
