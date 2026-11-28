/* Deterministic canvas-operation check for the dependency-free dancer renderer. */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const srcPath = path.resolve(__dirname, '..', 'frontend', 'assets', 'app.js');
const source = fs.readFileSync(srcPath, 'utf8');

assert(!source.includes('POSE_EDGES'), 'stick-figure pose edges are still present');
assert(!source.includes('KEY_JOINTS'), 'stick-figure joint list is still present');
for (const fn of ['drawTaperedLimb', 'drawSkirt', 'drawBlouse', 'drawHead', 'drawBangles']) {
  assert(source.includes(`function ${fn}(`), `missing procedural mannequin helper: ${fn}`);
}
assert(
  source.includes('if (!rel || !Array.isArray(rel) || rel.length < 21 || !pts[wristIdx]) continue;'),
  'genuine-hand landmark guard is missing'
);
assert(source.includes("'/api/reference-model'"), 'static FBX reference endpoint is missing');
assert(source.includes('function drawSourceSculpt('), 'static FBX atlas inset is missing');
assert(source.includes('state.dancer3DActive'), 'rigged sculpt stage switching is missing');
assert(source.includes('if (info.rigged && rigOptIn)'), 'opt-in rigged sculpt loading is missing');
assert(source.includes("rig=1"), 'rigged sculpt opt-in flag is missing');
const indexSource = fs.readFileSync(path.resolve(__dirname, '..', 'frontend', 'index.html'), 'utf8');
assert(indexSource.includes('type="importmap"'), 'Three.js import map is missing');
assert(indexSource.includes('assets/dancer3d.js'), 'rigged sculpt module is missing');
assert(indexSource.includes('id="dancer3dMount"'), 'rigged sculpt mount is missing');
const rigSource = fs.readFileSync(path.resolve(__dirname, '..', 'frontend', 'assets', 'dancer3d.js'), 'utf8');
assert(rigSource.includes('renderer.render(scene, camera)'), 'rigged sculpt render call is missing');
assert(rigSource.includes('function nodeNameKeys('), 'GLTFLoader bone-name normalization is missing');

const ops = {
  counts: Object.create(null),
  handStrokes: 0,
  mannequinFills: 0,
};
const bump = (name) => { ops.counts[name] = (ops.counts[name] || 0) + 1; };

class Gradient {
  addColorStop() { bump('colorStop'); }
}

const ctxStateStack = [];
const ctx = {
  globalAlpha: 1,
  shadowBlur: 0,
  shadowColor: '',
  fillStyle: '',
  strokeStyle: '',
  lineWidth: 1,
  lineCap: '',
  lineJoin: '',

  setTransform() { bump('setTransform'); },
  fillRect() { bump('fillRect'); },
  save() { ctxStateStack.push({ globalAlpha: this.globalAlpha, shadowBlur: this.shadowBlur }); bump('save'); },
  restore() {
    const state = ctxStateStack.pop();
    if (state) {
      this.globalAlpha = state.globalAlpha;
      this.shadowBlur = state.shadowBlur;
    }
    bump('restore');
  },
  createLinearGradient() { bump('linearGradient'); return new Gradient(); },
  createRadialGradient() { bump('radialGradient'); return new Gradient(); },
  beginPath() { bump('beginPath'); },
  closePath() { bump('closePath'); },
  moveTo() { bump('moveTo'); },
  lineTo() { bump('lineTo'); },
  quadraticCurveTo() { bump('quadraticCurveTo'); },
  arc() { bump('arc'); },
  ellipse() { bump('ellipse'); },
  drawImage() { bump('drawImage'); },
  fillText() { bump('fillText'); },
  translate() { bump('translate'); },
  scale() { bump('scale'); },
  fill() {
    bump('fill');
    if (this.shadowBlur > 0) ops.mannequinFills += 1;
  },
  stroke() {
    bump('stroke');
    if (this.globalAlpha === 0.92) ops.handStrokes += 1;
  },
};

function classList() {
  const items = new Set();
  return {
    add(...names) { names.forEach((n) => items.add(n)); },
    remove(...names) { names.forEach((n) => items.delete(n)); },
    contains(name) { return items.has(name); },
    toggle(name, force) {
      const on = force === undefined ? !items.has(name) : Boolean(force);
      if (on) items.add(name); else items.delete(name);
      return on;
    },
  };
}

function makeElement(id = '') {
  const el = {
    id,
    children: [],
    style: {},
    dataset: {},
    classList: classList(),
    className: '',
    value: '',
    textContent: '',
    title: '',
    disabled: false,
    checked: false,
    src: '',
    currentTime: 0,
    paused: true,
    ended: false,
    duration: 0,
    playbackRate: 1,
    width: 0,
    height: 0,
    attributes: {},
    addEventListener() {},
    removeEventListener() {},
    appendChild(child) { this.children.push(child); return child; },
    append(...children) { children.forEach((c) => this.children.push(c)); },
    remove() {},
    click() {},
    load() {},
    play() { return Promise.resolve(); },
    pause() {},
    scrollIntoView() {},
    setAttribute(name, value) { this.attributes[name] = value; },
    getAttribute(name) { return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null; },
    focus() {},
    querySelector(selector) { return makeElement(selector); },
    querySelectorAll(selector) { return selector === 'a' ? [makeElement('a')] : []; },
    closest() { return null; },
    getBoundingClientRect() { return { width: 866, height: 486 }; },
    getContext() { return ctx; },
  };
  Object.defineProperty(el, 'innerHTML', {
    get() { return ''; },
    set() { el.children.length = 0; },
  });
  return el;
}

const elements = new Map();
const document = {
  body: makeElement('body'),
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  },
  createElement(tag) { return makeElement(tag); },
  createTextNode(text) {
    const el = makeElement('#text');
    el.textContent = text;
    return el;
  },
  querySelector(selector) { return makeElement(selector); },
  querySelectorAll() { return []; },
  addEventListener() {},
};

class FakeImage {
  constructor() {
    this.complete = false;
    this.naturalWidth = 0;
    this.naturalHeight = 0;
    this._src = '';
    this.onload = null;
    this.onerror = null;
  }

  set src(value) {
    this._src = value;
    this.naturalWidth = 2048;
    this.naturalHeight = 2048;
    this.complete = true;
    if (this.onload) this.onload();
  }

  get src() { return this._src; }
}

class FakeIntersectionObserver {
  constructor(callback) { this.callback = callback; }
  observe() {}
  unobserve() {}
  disconnect() {}
}

const window = {
  devicePixelRatio: 1,
  innerWidth: 1440,
  innerHeight: 900,
  addEventListener() {},
  matchMedia() { return { matches: false, addEventListener() {}, removeEventListener() {} }; },
  IntersectionObserver: FakeIntersectionObserver,
  ResizeObserver: undefined,
};

let tickFn = null;
function requestAnimationFrame(cb) {
  tickFn = cb;
  return 1;
}

async function fakeFetch(url = '') {
  const payload = String(url).startsWith('/api/reference-model')
    ? {
        available: true,
        atlas_url: '/api/reference-model/atlas',
        frame_count: 16,
        grid: { columns: 4, rows: 4 },
        tile_size: 512,
      }
    : {};
  return {
    ok: true,
    status: 200,
    json: async () => payload,
    text: async () => '',
    blob: async () => ({}),
  };
}

const sandbox = {
  console,
  document,
  window,
  Image: FakeImage,
  IntersectionObserver: FakeIntersectionObserver,
  requestAnimationFrame,
  setTimeout() { return 1; },
  clearTimeout() {},
  fetch: fakeFetch,
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: srcPath });
assert(typeof tickFn === 'function', 'boot did not schedule the animation tick');
tickFn(1000);

const c = ops.counts;
assert((c.setTransform || 0) >= 1, 'canvas was not resized');
assert((c.fillRect || 0) >= 1, 'stage background was not drawn');
assert((c.linearGradient || 0) >= 3, 'stage gradients were not drawn');
assert((c.radialGradient || 0) >= 3, 'face and floor gradients were not drawn');
assert(ops.mannequinFills >= 12, `mannequin geometry too sparse: ${ops.mannequinFills} shaded fills`);
assert((c.ellipse || 0) >= 15, `mannequin curves too sparse: ${c.ellipse || 0} ellipses`);
assert(ops.handStrokes >= 42, `genuine hand landmarks were not drawn: ${ops.handStrokes} strokes`);
assert((c.drawImage || 0) >= 1, 'FBX-derived static sculpt atlas was not drawn');

console.log(
  `VERDICT: PASS · ${c.fill || 0} fills, ${c.stroke || 0} strokes, ` +
  `${c.ellipse || 0} ellipses, ${ops.mannequinFills} mannequin fills, ` +
  `${ops.handStrokes} hand-landmark strokes, ${c.drawImage || 0} atlas draws`
);
