/* Nritya studio — no build step, no external dependencies. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const POSE_EDGES = [
    [11, 12], [11, 23], [12, 24], [23, 24],
    [11, 13], [13, 15], [12, 14], [14, 16],
    [23, 25], [25, 27], [27, 29], [29, 31],
    [24, 26], [26, 28], [28, 30], [30, 32],
    [0, 11], [0, 12],
  ];
  const HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
  ];
  const KEY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

  // A real resting Nattadavu frame from the local extracted motion catalogue.
  const IDLE_FRAME = {
    pose_world: [[0.094,-1.23,-0.644],[0.113,-1.302,-0.605],[0.114,-1.303,-0.603],[0.114,-1.304,-0.603],[0.047,-1.299,-0.607],[0.047,-1.3,-0.609],[0.048,-1.303,-0.606],[0.208,-1.266,-0.374],[-0.087,-1.27,-0.406],[0.137,-1.179,-0.561],[0.048,-1.176,-0.566],[0.399,-0.955,-0.266],[-0.298,-1.003,-0.239],[0.899,-0.933,-0.299],[-0.804,-0.918,-0.135],[1.365,-0.983,-0.476],[-1.231,-0.893,-0.271],[1.47,-0.982,-0.503],[-1.334,-0.89,-0.265],[1.477,-1.026,-0.566],[-1.339,-0.935,-0.345],[1.381,-0.995,-0.511],[-1.248,-0.896,-0.309],[0.201,-0.009,-0.054],[-0.202,0.009,0.057],[0.52,0.697,-0.241],[-0.146,0.774,0.262],[1.119,1.207,-0.193],[-0.043,1.414,0.646],[1.209,1.269,-0.204],[-0.047,1.502,0.666],[1.431,1.288,-0.41],[-0.233,1.575,0.472]],
    left_hand_rel: [[0,0,0],[0,-0.07,-0.35],[0.043,-0.132,-0.459],[0.096,-0.155,-0.541],[0.144,-0.161,-0.586],[0.138,-0.124,-0.066],[0.2,-0.152,-0.236],[0.238,-0.164,-0.406],[0.268,-0.17,-0.528],[0.158,-0.095,-0.011],[0.22,-0.129,-0.109],[0.259,-0.145,-0.206],[0.287,-0.154,-0.277],[0.167,-0.06,-0.026],[0.222,-0.094,-0.154],[0.253,-0.111,-0.239],[0.275,-0.123,-0.282],[0.167,-0.018,-0.087],[0.217,-0.041,-0.213],[0.247,-0.06,-0.279],[0.272,-0.076,-0.31]],
    right_hand_rel: [[0,0,0],[-0.009,-0.035,-0.493],[-0.053,-0.091,-0.638],[-0.103,-0.128,-0.682],[-0.142,-0.155,-0.705],[-0.062,-0.091,-0.584],[-0.096,-0.143,-0.898],[-0.12,-0.182,-1.113],[-0.144,-0.218,-1.246],[-0.079,-0.09,-0.403],[-0.11,-0.147,-0.667],[-0.136,-0.193,-0.818],[-0.159,-0.238,-0.908],[-0.089,-0.087,-0.257],[-0.124,-0.137,-0.506],[-0.153,-0.17,-0.653],[-0.177,-0.199,-0.713],[-0.094,-0.083,-0.149],[-0.124,-0.117,-0.376],[-0.149,-0.138,-0.511],[-0.172,-0.153,-0.578]],
    travel_x: 0, jump_y: 0, foot_l_y: 0, foot_r_y: 0,
    adavu: 'resting pose',
  };

  const state = {
    file: null,
    useDemo: false,
    job: null,
    full: null,
    frames: [],
    beats: [],
    talaBeats: 8,
    tempo: 1,
    allowed: new Set(),
    catalogue: null,
    generating: false,
    objectUrl: null,
    demoBlob: null,
    sourceAvailable: true,
    progressTimers: [],
    lastBeatIndex: -1,
  };

  const canvas = $('dancerCanvas');
  const ctx = canvas.getContext('2d');
  let canvasW = 0;
  let canvasH = 0;

  /* ---------- small helpers ---------- */

  function toast(message, isError = false) {
    const t = $('toast');
    t.textContent = message;
    t.classList.toggle('error', isError);
    t.classList.add('show');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => t.classList.remove('show'), 3800);
  }

  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error((await res.text()) || `Request failed (${res.status})`);
    return res.json();
  }

  async function errorMessage(res) {
    try {
      const body = await res.json();
      return body.detail || JSON.stringify(body);
    } catch (_) {
      return (await res.text()) || `Request failed (${res.status})`;
    }
  }

  async function getDemoBlob() {
    if (!state.demoBlob) {
      const res = await fetch('/api/demo-audio');
      if (!res.ok) throw new Error(await errorMessage(res));
      state.demoBlob = await res.blob();
    }
    return state.demoBlob;
  }

  function fmtTime(sec) {
    const s = Math.max(0, Number(sec) || 0);
    const m = Math.floor(s / 60);
    const r = s - m * 60;
    return `${String(m).padStart(2, '0')}:${r.toFixed(1).padStart(4, '0')}`;
  }

  function fmtMs(ms) {
    return ms === null || ms === undefined ? '—' : `${Number(ms).toFixed(0)} ms`;
  }

  function fmtStat(value, digits = 1) {
    return value === null || value === undefined ? '—' : Number(value).toFixed(digits);
  }

  /* ---------- canvas stage ---------- */

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvasW = Math.max(1, rect.width);
    canvasH = Math.max(1, rect.height);
    canvas.width = Math.round(canvasW * dpr);
    canvas.height = Math.round(canvasH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function drawStage(frame, timeSec, idlePhase) {
    const w = canvasW;
    const h = canvasH;
    if (!w || !h) return;
    const floorY = h * 0.84;

    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, '#12162b');
    bg.addColorStop(0.55, '#0b0e1a');
    bg.addColorStop(1, '#070912');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // Temple arch silhouette.
    ctx.save();
    ctx.strokeStyle = 'rgba(216,162,74,0.13)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(w * 0.17, floorY);
    ctx.lineTo(w * 0.17, h * 0.25);
    ctx.quadraticCurveTo(w * 0.5, h * 0.02, w * 0.83, h * 0.25);
    ctx.lineTo(w * 0.83, floorY);
    ctx.stroke();
    ctx.strokeStyle = 'rgba(98,199,176,0.07)';
    ctx.beginPath();
    ctx.moveTo(w * 0.22, floorY);
    ctx.lineTo(w * 0.22, h * 0.29);
    ctx.quadraticCurveTo(w * 0.5, h * 0.09, w * 0.78, h * 0.29);
    ctx.lineTo(w * 0.78, floorY);
    ctx.stroke();
    ctx.restore();

    // Spotlight cone.
    const cone = ctx.createLinearGradient(0, 0, 0, floorY);
    cone.addColorStop(0, 'rgba(216,162,74,0.10)');
    cone.addColorStop(1, 'rgba(216,162,74,0)');
    ctx.fillStyle = cone;
    ctx.beginPath();
    ctx.moveTo(w * 0.5, 0);
    ctx.lineTo(w * 0.18, floorY);
    ctx.lineTo(w * 0.82, floorY);
    ctx.closePath();
    ctx.fill();

    // Floor and reflection pool.
    const pool = ctx.createRadialGradient(w * 0.5, floorY, 2, w * 0.5, floorY, w * 0.36);
    pool.addColorStop(0, 'rgba(216,162,74,0.18)');
    pool.addColorStop(0.55, 'rgba(98,199,176,0.05)');
    pool.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.save();
    ctx.translate(w * 0.5, floorY);
    ctx.scale(1, 0.28);
    ctx.fillStyle = pool;
    ctx.beginPath();
    ctx.arc(0, 0, w * 0.36, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    ctx.strokeStyle = 'rgba(216,162,74,0.18)';
    ctx.beginPath();
    ctx.moveTo(w * 0.12, floorY);
    ctx.lineTo(w * 0.88, floorY);
    ctx.stroke();

    const pose = frame.pose_world || IDLE_FRAME.pose_world;
    const scale = Math.min(h * 0.232, w * 0.168);
    const footY = Math.max(...[27, 28, 29, 30, 31, 32].map((i) => (pose[i] && pose[i][1]) || 0));
    const sway = idlePhase ? Math.sin(idlePhase) * w * 0.004 : 0;
    const cx = w * 0.5 + ((frame.travel_x || 0) * scale) + sway;
    const cy = floorY - footY * scale - (frame.jump_y || 0) * scale;

    const project = (p) => {
      const z = clamp(p[2] || 0, -1.5, 1.5);
      const k = 1 + z * 0.04;
      return [cx + p[0] * scale * k, cy + p[1] * scale * k];
    };
    const pts = pose.map(project);

    // Beat pulse ring on the floor.
    const beatPulse = beatPulseAt(timeSec);
    if (beatPulse > 0) {
      ctx.save();
      ctx.globalAlpha = beatPulse * 0.5;
      ctx.strokeStyle = '#e6785b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(cx, floorY, scale * (0.35 + (1 - beatPulse) * 0.55), scale * (0.1 + (1 - beatPulse) * 0.16), 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // Body limbs, far-to-near for a little depth.
    const edges = POSE_EDGES.map(([a, b]) => ({ a, b, z: ((pose[a] && pose[a][2]) || 0) + ((pose[b] && pose[b][2]) || 0) }));
    edges.sort((p, q) => q.z - p.z);
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'rgba(216,162,74,0.45)';
    ctx.shadowBlur = scale * 0.09;
    for (const e of edges) {
      const a = pts[e.a];
      const b = pts[e.b];
      if (!a || !b) continue;
      const isCore = e.a === 11 || e.a === 12 || e.a === 23 || e.a === 24;
      ctx.strokeStyle = isCore ? '#f0c987' : '#d8a24a';
      ctx.globalAlpha = clamp(0.82 + e.z * 0.04, 0.55, 1);
      ctx.lineWidth = scale * (isCore ? 0.058 : 0.044);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
    for (const i of KEY_JOINTS) {
      const p = pts[i];
      if (!p) continue;
      ctx.fillStyle = '#f7f1e5';
      ctx.beginPath();
      ctx.arc(p[0], p[1], scale * 0.024, 0, Math.PI * 2);
      ctx.fill();
    }

    // Head with a simple crown/bindi detail.
    const head = pts[0];
    if (head) {
      const r = scale * 0.108;
      const face = ctx.createRadialGradient(head[0] - r * 0.35, head[1] - r * 0.4, r * 0.1, head[0], head[1], r);
      face.addColorStop(0, '#f6dda8');
      face.addColorStop(1, '#b8792f');
      ctx.fillStyle = face;
      ctx.beginPath();
      ctx.arc(head[0], head[1], r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(98,199,176,0.75)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(head[0], head[1], r * 1.13, Math.PI * 1.08, Math.PI * 1.92);
      ctx.stroke();
      ctx.fillStyle = '#e6785b';
      ctx.beginPath();
      ctx.arc(head[0], head[1] - r * 0.1, r * 0.13, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // Genuine hand landmarks only — never invent missing ones.
    for (const [key, wristIdx, color] of [
      ['left_hand_rel', 15, '#f0c987'],
      ['right_hand_rel', 16, '#62c7b0'],
    ]) {
      const rel = frame[key];
      if (!rel || !Array.isArray(rel) || rel.length < 21 || !pts[wristIdx]) continue;
      const hp = rel.map((p) => [pts[wristIdx][0] + (p[0] || 0) * scale, pts[wristIdx][1] + (p[1] || 0) * scale]);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = Math.max(1, scale * 0.013);
      ctx.globalAlpha = 0.92;
      ctx.lineCap = 'round';
      for (const [a, b] of HAND_EDGES) {
        ctx.beginPath();
        ctx.moveTo(hp[a][0], hp[a][1]);
        ctx.lineTo(hp[b][0], hp[b][1]);
        ctx.stroke();
      }
      for (const i of [4, 8, 12, 16, 20]) {
        ctx.beginPath();
        ctx.arc(hp[i][0], hp[i][1], Math.max(1.4, scale * 0.016), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // Foot strike marks from the frame's grounded feet.
    ctx.save();
    for (const [idx, flKey] of [[27, 'foot_l_y'], [28, 'foot_r_y']]) {
      const lift = frame[flKey];
      if (lift === undefined || lift === null || lift > 0.09) continue;
      const p = pts[idx];
      if (!p) continue;
      ctx.globalAlpha = 0.35;
      ctx.strokeStyle = '#e6785b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.ellipse(p[0], floorY, scale * 0.13, scale * 0.035, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  function beatPulseAt(t) {
    if (!state.beats.length) return 0;
    let last = null;
    for (const b of state.beats) {
      if (b.time_s <= t + 0.02) last = b;
      else break;
    }
    if (!last) return 0;
    const dt = t - last.time_s;
    return dt >= 0 && dt < 0.38 ? 1 - dt / 0.38 : 0;
  }

  function frameAtTime(t) {
    if (!state.frames.length) return { frame: IDLE_FRAME, waiting: false, idle: true };
    const t0 = Number(state.frames[0].t) || 0;
    const fps = Number((state.full && state.full.fps) || (state.job && state.job.fps) || 30) || 30;
    if (t < t0 - 0.001) return { frame: state.frames[0], waiting: true, idle: false };
    const idx = clamp(Math.round((t - t0) * fps), 0, state.frames.length - 1);
    return { frame: state.frames[idx], waiting: false, idle: false };
  }

  function currentBeatIndex(t) {
    if (!state.beats.length) return -1;
    let idx = -1;
    for (let i = 0; i < state.beats.length; i++) {
      if (state.beats[i].time_s <= t + 0.03) idx = i;
      else break;
    }
    return idx;
  }

  function updateHUD(t, info) {
    const frame = info.frame;
    if (info.waiting) {
      $('stageAdavu').textContent = `${frame.adavu || '—'} · waiting for sam`;
    } else if (info.idle) {
      $('stageAdavu').textContent = frame.adavu || 'resting pose';
    } else {
      const conf = frame.confirmed === false ? ' · unconfirmed' : '';
      $('stageAdavu').textContent = `${frame.adavu || '—'}${conf}`;
    }
    $('stageTime').textContent = fmtTime(t);
    $('timeNow').textContent = fmtTime(t);
    const audio = $('audioEl');
    const duration = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : (state.job && state.job.duration_s) || 0;
    if (duration > 0) $('seekBar').value = String(clamp(t / duration, 0, 1) * 1000);

    const beatIdx = currentBeatIndex(t);
    if (beatIdx >= 0) {
      const b = state.beats[beatIdx];
      const dots = $('beatTrack').children;
      const pos = (Number(b.tala_pos) || 1) - 1;
      for (let i = 0; i < dots.length; i++) dots[i].classList.toggle('active', i === pos && t - b.time_s < 0.36);
      if (beatIdx !== state.lastBeatIndex && b.is_sam && t - b.time_s < 0.36) $('beatTrack').classList.add('sam-flash');
      state.lastBeatIndex = beatIdx;
    } else {
      for (const d of $('beatTrack').children) d.classList.remove('active');
    }
  }

  function heroWaveTick(ts) {
    const wave = $('heroWave');
    if (!wave.children.length) return;
    const n = wave.children.length;
    for (let i = 0; i < n; i++) {
      const v = 0.5 + 0.5 * Math.sin(ts / 620 + i * 0.42) * Math.sin(ts / 1730 + i * 0.13);
      wave.children[i].style.height = `${8 + v * 88}%`;
    }
  }

  function tick(ts) {
    requestAnimationFrame(tick);
    const audio = $('audioEl');
    const t = state.frames.length ? Number(audio.currentTime) || 0 : 0;
    const info = frameAtTime(t);
    drawStage(info.frame, t, info.idle ? ts / 1000 : 0);
    updateHUD(t, info);
    heroWaveTick(ts);
  }

  /* ---------- UI population ---------- */

  function buildHeroWave() {
    const wave = $('heroWave');
    for (let i = 0; i < 46; i++) wave.appendChild(document.createElement('i'));
  }

  function buildBeatTrack(n) {
    const track = $('beatTrack');
    track.innerHTML = '';
    for (let i = 0; i < clamp(n, 3, 16); i++) {
      const dot = document.createElement('span');
      dot.className = 'beat-dot' + (i === 0 ? ' sam' : '');
      track.appendChild(dot);
    }
  }

  function renderAdavuChips(items, container, onClick) {
    container.innerHTML = '';
    const list = Array.isArray(items) ? items : [];
    if (!list.length) {
      const span = document.createElement('span');
      span.className = 'muted';
      span.textContent = '—';
      container.appendChild(span);
      return;
    }
    for (const name of list) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip' + (state.allowed.has(name) ? ' selected' : '');
      chip.textContent = name.replace(/Adavu$/, '');
      chip.title = name;
      chip.addEventListener('click', () => {
        if (!onClick) return;
        if (state.allowed.has(name)) state.allowed.delete(name);
        else state.allowed.add(name);
        chip.classList.toggle('selected');
      });
      container.appendChild(chip);
    }
  }

  async function loadHealth() {
    try {
      const h = await getJSON('/api/health');
      const pill = $('healthPill');
      pill.classList.add('pill-ok');
      pill.innerHTML = '';
      const dot = document.createElement('i');
      dot.className = 'dot';
      pill.appendChild(dot);
      pill.appendChild(document.createTextNode(h.cuda_available ? 'GPU ready' : 'API ready'));
      pill.title = h.gpu_name ? `${h.gpu_name} · catalogue ${h.catalogue.total_units} units` : `Catalogue ${h.catalogue.total_units} units`;
    } catch (err) {
      const pill = $('healthPill');
      pill.classList.add('pill-bad');
      pill.textContent = 'API offline';
    }
  }

  async function loadCatalogue() {
    try {
      const cat = await getJSON('/api/catalogue');
      state.catalogue = cat;
      renderAdavuChips(cat.confirmed_adavu_categories || cat.adavu_categories || [], $('allowedChips'), true);
      const stats = $('catalogueStats');
      stats.innerHTML = '';
      const entries = [
        [cat.total_units ?? '—', 'phrase units'],
        [cat.confirmed_units ?? '—', 'confirmed'],
        [cat.unconfirmed_units ?? '—', 'unconfirmed (opt-in)'],
        [(cat.adavu_categories || []).length, 'adavu families'],
      ];
      for (const [value, label] of entries) {
        const d = document.createElement('div');
        d.className = 'cat-stat';
        const s = document.createElement('strong');
        s.textContent = String(value);
        const sp = document.createElement('span');
        sp.textContent = label;
        d.append(s, sp);
        stats.appendChild(d);
      }
      $('catalogueInfo').textContent = `Library: ${cat.total_units ?? '?'} units · ${(cat.adavu_categories || []).length} families · ${cat.quality_rules_status || 'provisional'}`;
    } catch (err) {
      $('catalogueInfo').textContent = `Catalogue unavailable: ${err.message}`;
    }
  }

  async function loadJobs() {
    try {
      const jobs = await getJSON('/api/jobs?limit=8');
      const host = $('recentJobs');
      host.innerHTML = '';
      if (!jobs.length) {
        const p = document.createElement('p');
        p.className = 'empty-line';
        p.textContent = 'No generations yet this session.';
        host.appendChild(p);
        return;
      }
      for (const job of jobs) {
        const item = document.createElement('div');
        item.className = 'recent-item';
        if (job.source_available === false) {
          item.classList.add('no-source');
          item.title = 'JSON only — source audio was not retained';
        } else {
          item.title = 'Load this generation';
        }
        const left = document.createElement('span');
        left.textContent = (job.source_file || job.job_id).slice(0, 26);
        const right = document.createElement('span');
        right.textContent = `${fmtStat(job.song_bpm, 0)} BPM · ${fmtStat(job.duration_s, 0)}s`;
        item.append(left, right);
        item.addEventListener('click', () => loadJob(job.job_id));
        host.appendChild(item);
      }
    } catch (_) { /* recent jobs are optional */ }
  }

  async function loadDemoInfo() {
    try {
      const info = await getJSON('/api/demo-info');
      if (!info.available) {
        $('demoBtn').disabled = true;
        $('demoStatus').textContent = 'No bundled demo clip found on this machine.';
        $('demoStatus').classList.add('error');
        $('heroDemoBtn').disabled = true;
      } else {
        $('demoStatus').textContent = '30-second adavu excerpt · stem auto-excluded from retrieval.';
      }
    } catch (_) {
      $('demoStatus').textContent = 'Demo availability unknown.';
    }
  }

  /* ---------- file handling ---------- */

  function setFile(file) {
    state.file = file;
    state.useDemo = false;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = URL.createObjectURL(file);
    $('audioEl').src = state.objectUrl;
    $('dropZone').classList.add('hidden');
    $('fileCard').classList.remove('hidden');
    $('fileName').textContent = file.name;
    $('fileMeta').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · local file`;
    $('timeTotal').textContent = fmtTime(0);
  }

  function setDemo() {
    state.file = null;
    state.useDemo = true;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
    $('audioEl').src = '/api/demo-audio';
    $('dropZone').classList.add('hidden');
    $('fileCard').classList.remove('hidden');
    $('fileName').textContent = 'demo.mp3';
    $('fileMeta').textContent = '30-second dataset excerpt · demo';
  }

  function clearFile() {
    state.file = null;
    state.useDemo = false;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
    $('audioEl').removeAttribute('src');
    $('audioEl').load();
    $('fileCard').classList.add('hidden');
    $('dropZone').classList.remove('hidden');
    $('fileInput').value = '';
  }

  /* ---------- generation ---------- */

  function setProgress(pct, text, stage) {
    $('progressPanel').classList.remove('hidden');
    $('progressBar').style.width = `${pct}%`;
    $('progressPct').textContent = `${pct}%`;
    $('progressText').textContent = text;
    $('stageLoadingText').textContent = text;
    const steps = $('progressSteps').children;
    for (let i = 0; i < steps.length; i++) steps[i].classList.toggle('done', i < stage);
  }

  function clearProgressTimers() {
    state.progressTimers.forEach(clearTimeout);
    state.progressTimers = [];
  }

  async function generate() {
    if (state.generating) return;
    if (!state.file && !state.useDemo) {
      toast('Choose an audio/video file or use the demo clip first.', true);
      return;
    }
    state.generating = true;
    const btn = $('generateBtn');
    btn.disabled = true;
    $('generateLabel').textContent = 'Generating…';
    $('stageLoading').classList.remove('hidden');
    $('stageStatus').textContent = 'COMPOSING';
    setProgress(6, 'Reading uploaded audio…', 0);

    clearProgressTimers();
    state.progressTimers.push(setTimeout(() => setProgress(22, 'Separating percussion…', 1), 900));
    state.progressTimers.push(setTimeout(() => setProgress(48, 'Tracking beats and tala grid…', 2), 4200));
    state.progressTimers.push(setTimeout(() => setProgress(72, 'Selecting and blending adavu phrases…', 3), 10000));
    state.progressTimers.push(setTimeout(() => setProgress(88, 'Preparing the dancer timeline…', 3), 18000));

    try {
      const fd = new FormData();
      if (state.file) {
        fd.append('file', state.file, state.file.name);
      } else {
        const demoBlob = await getDemoBlob();
        fd.append('file', demoBlob, 'demo.mp3');
      }
      fd.append('tala', $('talaSelect').value);
      fd.append('tempo_scale', String(state.tempo));
      fd.append('offset_ms', String(Number($('offsetMs').value) || 0));
      fd.append('fps', $('fpsInput').value);
      fd.append('seed', String(Number($('seedInput').value) || 42));
      fd.append('use_demucs', String($('demucsToggle').checked));
      fd.append('include_unconfirmed', String($('unconfirmedToggle').checked));
      fd.append('allowed_adavus', Array.from(state.allowed).join(','));
      const samRaw = $('samIndex').value.trim();
      if (samRaw !== '') fd.append('sam_beat_index', String(Number(samRaw)));

      const res = await fetch('/api/choreography/generate', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await errorMessage(res));
      const summary = await res.json();
      const full = await getJSON(`/api/choreography/${summary.job_id}`);
      clearProgressTimers();
      setProgress(100, 'Choreography ready', 4);
      applyJob(summary, full);
      toast(`Generated ${summary.phrase_count} phrases at ${fmtStat(summary.song_bpm, 1)} BPM.`);
    } catch (err) {
      clearProgressTimers();
      $('progressPanel').classList.add('hidden');
      toast(`Generation failed: ${err.message}`, true);
      $('stageStatus').textContent = state.frames.length ? 'READY' : 'IDLE';
    } finally {
      $('stageLoading').classList.add('hidden');
      btn.disabled = false;
      $('generateLabel').textContent = 'Generate choreography';
      state.generating = false;
    }
  }

  function applyJob(summary, full, options = {}) {
    state.job = summary;
    state.full = full;
    state.sourceAvailable = options.sourceAvailable !== false;
    state.frames = full.frames || summary.frames || [];
    const analysis = full.audio_analysis || {};
    state.beats = analysis.beats || [];
    state.talaBeats = Number(analysis.tala_beats) || 8;
    buildBeatTrack(state.talaBeats);

    // Replay from retained source media when present; older jobs may only have JSON.
    if (state.sourceAvailable && (state.useDemo || (!state.file && summary.job_id))) {
      $('audioEl').src = `/api/media/${summary.job_id}/source`;
    } else if (!state.sourceAvailable) {
      $('audioEl').removeAttribute('src');
    }
    $('audioEl').load();
    if (state.sourceAvailable) $('audioEl').currentTime = 0;
    $('renderBtn').disabled = !state.sourceAvailable;
    $('timeTotal').textContent = fmtTime(full.duration_s || summary.duration_s);
    $('stageStatus').textContent = 'READY';
    $('stageStatus').classList.remove('playing');
    $('jobId').textContent = `job ${summary.job_id}`;
    const affect = full.affect || analysis.affect || {};
    $('stageSubtitle').textContent = `${fmtStat(summary.song_bpm, 1)} BPM · ${affect.label || 'unknown'} energy · ${state.talaBeats}-beat ${analysis.tala || summary.tala || 'tala'} grid${state.sourceAvailable ? '' : ' · JSON only (no retained audio)'}`;

    $('statBpm').textContent = fmtStat(summary.song_bpm, 1);
    $('statAffect').textContent = (affect.label || '—').toUpperCase();
    $('statAffectBasis').textContent = `intensity ${fmtStat(affect.intensity_score, 2)} · heuristic`;
    $('statPhrases').textContent = String(summary.phrase_count ?? '—');
    $('statDuration').textContent = fmtTime(full.duration_s || summary.duration_s);
    $('statFrames').textContent = `${full.n_frames || summary.n_frames || 0} frames @ ${summary.fps || 30} fps`;

    renderAdavuChips(summary.adavus_used || [], $('adavuChips'), null);
    renderAdavuChips(summary.source_stems_used || [], $('sourceStems'), null);
    renderEnergy(analysis.energy_curve || [], full.affect_timeline || []);
    renderBeats(state.beats);
    renderMetrics(full);
    renderTimeline(full.timeline_events || []);

    const samIdx = analysis.first_sam_index;
    $('rhythmSummary').innerHTML = '';
    const summaryLine = document.createElement('div');
    summaryLine.textContent = `${state.beats.length} beats · sam at index ${samIdx ?? '?'} (${analysis.sam_source || 'auto'}) · separation: ${summary.separation_backend || analysis.separation_backend || '—'} · ${affect.label || '—'} passage`;
    $('rhythmSummary').appendChild(summaryLine);

    loadJobs();
  }

  function renderEnergy(curve, timeline) {
    const host = $('energyChart');
    host.innerHTML = '';
    if (!curve.length) {
      host.textContent = 'No energy curve available.';
      return;
    }
    for (let i = 0; i < curve.length; i++) {
      const bar = document.createElement('div');
      bar.className = 'energy-bar' + (Number(curve[i]) > 0.28 ? ' hot' : '');
      bar.style.height = `${clamp(Number(curve[i]) * 100 * 2.6, 3, 100)}%`;
      bar.title = `Section ${i + 1}: energy ${Number(curve[i]).toFixed(3)}`;
      if (i % 8 === 0) {
        const label = document.createElement('span');
        label.textContent = `${Math.round(i / curve.length * (state.job?.duration_s || 0))}s`;
        bar.appendChild(label);
      }
      host.appendChild(bar);
    }
    host.title = timeline.length
      ? timeline.map((e) => `${e.time_s}s → ${e.adavu} (target energy ${e.target_energy})`).join('\n')
      : '';
  }

  function renderBeats(beats) {
    const body = $('beatList');
    body.innerHTML = '';
    if (!beats.length) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 5;
      td.textContent = 'No beats detected.';
      tr.appendChild(td);
      body.appendChild(tr);
      return;
    }
    for (const b of beats) {
      const tr = document.createElement('tr');
      if (b.is_sam) tr.className = 'sam';
      const values = [
        String(b.index ?? ''),
        fmtTime(b.time_s),
        `${b.tala_pos ?? '—'}${b.is_sam ? ' · SAM' : ''}`,
        fmtStat(b.local_bpm, 1),
        fmtStat(b.energy, 3),
      ];
      values.forEach((v, i) => {
        const td = document.createElement('td');
        if (i === 4) td.className = 'energy-cell';
        td.textContent = v;
        tr.appendChild(td);
      });
      body.appendChild(tr);
    }
  }

  function renderMetrics(job) {
    const q = job.frame_quantization_check_ms || {};
    $('metricsQuant').textContent = `${fmtMs(q.median_ms)} / ${fmtMs(q.max_ms)}`;
    const st = job.planned_foot_strikes_vs_beats || {};
    $('metricsStrikes').textContent = `${fmtMs(st.median_ms)} (${st.strike_count ?? 0} strikes)`;
    $('metricsStrikesBasis').textContent = `${st.err_frac_of_period ?? '—'} of beat period · ${st.beat_source || 'unverified'}`;
    const t = job.transition_metrics || {};
    const stat = (key) => {
      const v = t[key] || {};
      return `${fmtStat(v.median_torso, 3)} / ${fmtStat(v.max_torso, 3)}`;
    };
    $('metricsBody').textContent = stat('body');
    $('metricsHands').textContent = stat('hands');
    $('metricsFoot').textContent = stat('foot_sliding');
  }

  function renderTimeline(events) {
    const host = $('timelineList');
    host.innerHTML = '';
    if (!events.length) {
      const p = document.createElement('p');
      p.className = 'empty-line';
      p.textContent = 'No phrase timeline yet.';
      host.appendChild(p);
      return;
    }
    for (const ev of events) {
      const row = document.createElement('div');
      row.className = 'timeline-item' + (ev.is_phrase_start ? ' phrase-start' : '');
      const time = document.createElement('span');
      time.className = 'timeline-time';
      time.textContent = fmtTime(ev.time_s);
      const mark = document.createElement('span');
      mark.className = 'timeline-mark';
      mark.textContent = ev.is_phrase_start ? '✦' : '·';
      const name = document.createElement('span');
      name.className = 'timeline-adavu';
      name.textContent = `${ev.adavu || '—'}${ev.confirmed === false ? ' (unconfirmed)' : ''}`;
      const src = document.createElement('span');
      src.className = 'timeline-src';
      src.textContent = ev.source_stem || '';
      row.append(time, mark, name, src);
      row.title = `${ev.unit_id || ''} · beat ${ev.song_beat_index ?? '?'}`;
      host.appendChild(row);
    }
  }

  async function loadJob(jobId) {
    try {
      const full = await getJSON(`/api/choreography/${jobId}`);
      const sourceAvailable = full.source_available !== false;
      const summary = {
        job_id: full.job_id || jobId,
        song_bpm: full.song_bpm,
        tala: full.tala,
        separation_backend: full.separation_backend,
        fps: full.fps,
        duration_s: full.duration_s,
        n_frames: full.n_frames,
        phrase_count: full.phrase_count,
        adavus_used: full.adavus_used,
        source_stems_used: full.source_stems_used,
      };
      state.file = null;
      state.useDemo = false;
      applyJob(summary, full, { sourceAvailable });
      toast(sourceAvailable
        ? `Loaded job ${jobId}.`
        : `Loaded job ${jobId} as JSON only; source audio was not retained.`);
    } catch (err) {
      toast(`Could not load job: ${err.message}`, true);
    }
  }

  /* ---------- transport ---------- */

  function togglePlay() {
    const audio = $('audioEl');
    if (!audio.src) {
      toast('Generate or load a choreography first.', true);
      return;
    }
    if (audio.paused) {
      audio.play().catch((err) => toast(`Playback blocked: ${err.message}`, true));
    } else {
      audio.pause();
    }
  }

  function syncPlayIcon() {
    const audio = $('audioEl');
    const playing = !audio.paused && !audio.ended;
    $('playIcon').innerHTML = playing
      ? '<path d="M7 5h4v14H7zm6 0h4v14h-4z"/>'
      : '<path d="M8 5.5v13l11-6.5z"/>';
    $('stageStatus').textContent = playing ? 'PERFORMING' : (state.frames.length ? 'READY' : 'IDLE');
    $('stageStatus').classList.toggle('playing', playing);
  }

  async function exportMp4() {
    if (!state.job) {
      toast('Generate a choreography before exporting.', true);
      return;
    }
    if (!state.sourceAvailable) {
      toast('This job has no retained source audio, so MP4 export is unavailable.', true);
      return;
    }
    const btn = $('renderBtn');
    const original = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Rendering…';
    toast('Rendering MP4 — this can take a minute for longer songs.');
    try {
      const res = await fetch(`/api/choreography/${state.job.job_id}/render?max_seconds=45&width=960&height=540`, { method: 'POST' });
      if (!res.ok) throw new Error(await errorMessage(res));
      const data = await res.json();
      showVideoModal(data.video_url);
    } catch (err) {
      toast(`Export failed: ${err.message}`, true);
    } finally {
      btn.disabled = false;
      btn.textContent = original;
    }
  }

  function showVideoModal(url) {
    const overlay = document.createElement('div');
    overlay.className = 'video-modal';
    overlay.innerHTML = `<div class="video-modal-inner"><div class="video-modal-head"><strong>Exported animation</strong><button type="button" aria-label="Close">×</button></div><video controls autoplay playsinline src="${url}"></video><p class="muted">First 45 seconds · rendered from the planned pose frames with the original audio.</p></div>`;
    overlay.querySelector('button').addEventListener('click', () => overlay.remove());
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    document.body.appendChild(overlay);
  }

  function downloadJson() {
    if (!state.full) {
      toast('Generate a choreography before downloading.', true);
      return;
    }
    const blob = new Blob([JSON.stringify(state.full)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nritya-${state.job.job_id}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ---------- event wiring ---------- */

  function bindEvents() {
    const drop = $('dropZone');
    drop.addEventListener('click', () => $('fileInput').click());
    drop.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('fileInput').click(); }
    });
    drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('dragover'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      drop.classList.remove('dragover');
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) setFile(file);
    });
    $('fileInput').addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
    });
    $('clearFile').addEventListener('click', clearFile);
    $('demoBtn').addEventListener('click', () => { setDemo(); toast('Demo clip loaded. Generate to see the dancer.'); });
    $('heroDemoBtn').addEventListener('click', () => {
      setDemo();
      $('studio').scrollIntoView({ behavior: 'smooth' });
      toast('Demo clip loaded in the studio.');
    });

    $('tempoGroup').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-tempo]');
      if (!b) return;
      state.tempo = Number(b.dataset.tempo);
      $('tempoCustom').value = String(state.tempo);
      for (const el of $('tempoGroup').children) el.classList.toggle('active', el === b);
    });
    $('tempoCustom').addEventListener('change', (e) => {
      const v = clamp(Number(e.target.value) || 1, 0.25, 4);
      state.tempo = v;
      e.target.value = String(v);
      for (const el of $('tempoGroup').children) el.classList.toggle('active', Number(el.dataset.tempo) === v);
    });
    $('talaSelect').addEventListener('change', () => {
      const beats = { adi: 8, rupaka: 6, misra_chapu: 7, khanda_chapu: 5, ata: 14 }[$('talaSelect').value] || 8;
      buildBeatTrack(beats);
    });

    $('generateBtn').addEventListener('click', generate);
    document.querySelectorAll('.tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach((t) => {
          t.classList.toggle('active', t === tab);
          t.setAttribute('aria-selected', String(t === tab));
        });
        document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${tab.dataset.tab}`));
      });
    });

    $('transportPlay').addEventListener('click', togglePlay);
    $('audioEl').addEventListener('play', syncPlayIcon);
    $('audioEl').addEventListener('pause', syncPlayIcon);
    $('audioEl').addEventListener('ended', () => {
      syncPlayIcon();
      if ($('loopToggle').getAttribute('aria-pressed') === 'true') {
        $('audioEl').currentTime = 0;
        $('audioEl').play().catch(() => {});
      }
    });
    $('audioEl').addEventListener('loadedmetadata', () => {
      $('timeTotal').textContent = fmtTime($('audioEl').duration || (state.job && state.job.duration_s) || 0);
    });
    $('seekBar').addEventListener('input', (e) => {
      const audio = $('audioEl');
      if (!Number.isFinite(audio.duration) || audio.duration <= 0) return;
      audio.currentTime = Number(e.target.value) / 1000 * audio.duration;
    });
    $('speedSelect').addEventListener('change', (e) => { $('audioEl').playbackRate = Number(e.target.value); });
    $('loopToggle').addEventListener('click', (e) => {
      const on = e.currentTarget.getAttribute('aria-pressed') === 'true';
      e.currentTarget.setAttribute('aria-pressed', String(!on));
    });
    $('renderBtn').addEventListener('click', exportMp4);
    $('downloadBtn').addEventListener('click', downloadJson);

    document.addEventListener('keydown', (e) => {
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
      if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
      if (e.code === 'ArrowRight' && $('audioEl').src) $('audioEl').currentTime += 2;
      if (e.code === 'ArrowLeft' && $('audioEl').src) $('audioEl').currentTime -= 2;
    });

    window.addEventListener('resize', resizeCanvas);
    if (window.ResizeObserver) new ResizeObserver(resizeCanvas).observe($('stageWrap'));
  }

  /* ---------- boot ---------- */

  buildHeroWave();
  buildBeatTrack(8);
  bindEvents();
  resizeCanvas();
  requestAnimationFrame(tick);
  loadHealth();
  loadCatalogue();
  loadJobs();
  loadDemoInfo();
})();
