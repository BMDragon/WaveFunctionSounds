'use strict';

const $ = id => document.getElementById(id);
const NAMES = ['C', 'C♯', 'D', 'E♭', 'E', 'F', 'F♯', 'G', 'A♭', 'A', 'B♭', 'B'];
const LOW = 48, HIGH = 84;      // keyboard range, C3..C6
const KEYMAP = {
  z: 48, s: 49, x: 50, d: 51, c: 52, v: 53, g: 54, b: 55, h: 56, n: 57, j: 58, m: 59,
  q: 60, 2: 61, w: 62, 3: 63, e: 64, r: 65, 5: 66, t: 67, 6: 68, y: 69, 7: 70, u: 71,
  i: 72, 9: 73, o: 74, 0: 75, p: 76,
};
const midiFreq = n => 440 * Math.pow(2, (n - 69) / 12);
const noteName = n => NAMES[n % 12] + (Math.floor(n / 12) - 1);
const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);

const state = { potential: 'infinite', data: null, notes: {}, focus: 57, requestId: 0 };

// ---------- controls ----------

function params() {
  return {
    potential: state.potential,
    mass: Math.pow(10, +$('mass').value),
    length: +$('length').value,
    depth: Math.pow(10, +$('depth').value),
    weighting: $('weighting').value,
    temperatureMode: document.querySelector('input[name=tmode]:checked').value,
    temperature: Math.pow(10, +$('temperature').value),
  };
}

function fmt(v, digits = 3) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString();
  return (+v.toPrecision(digits)).toString();
}

function refreshOutputs() {
  const p = params();
  $('massOut').textContent = `${fmt(p.mass)} mₑ`;
  $('lengthOut').textContent = `${fmt(p.length)} nm`;
  $('depthOut').textContent = `${fmt(p.depth)} eV`;
  $('temperatureOut').textContent = `${fmt(p.temperature)} K`;
  $('attackOut').textContent = `${$('attack').value} ms`;
  $('decayOut').textContent = `${(+$('decay').value).toFixed(1)} s`;
  $('releaseOut').textContent = `${(+$('release').value).toFixed(2)} s`;
  $('volumeOut').textContent = `${Math.round($('volume').value * 100)}%`;
  $('secondsOut').textContent = `${$('seconds').value} s`;
  $('lengthLabel').textContent = state.potential === 'oscillator' ? 'Ground-state width' : 'Well width';
  $('depthRow').hidden = state.potential !== 'finite';
  $('temperatureRows').hidden = p.weighting !== 'boltzmann';
  $('temperatureRow').hidden = p.temperatureMode !== 'fixed';
}

let updateTimer = null;
function scheduleUpdate(delay = 250) {
  clearTimeout(updateTimer);
  updateTimer = setTimeout(update, delay);
}

async function update() {
  const id = ++state.requestId;
  setStatus('Computing…');
  const body = { ...params(), focusFreq: midiFreq(state.focus), plotStates: 8,
                 freqs: range(LOW, HIGH + 1).map(midiFreq) };
  try {
    const res = await fetch('/api/model', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify(body) });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || res.statusText);
    if (id !== state.requestId) return;     // a newer request is on its way
    state.data = json;
    state.notes = {};
    range(LOW, HIGH + 1).forEach((n, i) => { state.notes[n] = json.notes[i]; });
    setStatus('');
    drawAll();
  } catch (error) {
    if (id === state.requestId) setStatus(`Error: ${error.message}`);
  }
}

function setStatus(text) { $('status').textContent = text; }

// ---------- audio ----------

let ctx = null, master = null, analyser = null, playerSource = null;
const voices = {};

function audio() {
  if (!ctx) {
    ctx = new AudioContext();
    master = ctx.createGain();
    master.gain.value = +$('volume').value;
    analyser = ctx.createAnalyser();
    analyser.fftSize = 8192;
    analyser.smoothingTimeConstant = 0;
    master.connect(analyser);
    analyser.connect(ctx.destination);
    requestAnimationFrame(drawSpectrogram);
    drawSpectrogramAxis();
  }
  if (ctx.state === 'suspended') ctx.resume();
  return ctx;
}

function periodicWave(note) {
  // PeriodicWave sums real[m] cos + imag[m] sin; our waveform is Re(sum c_m e^{i...}),
  // i.e. Re(c) cos - Im(c) sin. Scaled so every note peaks at most at 1.
  const size = Math.max(...note.m) + 1;
  const real = new Float32Array(size), imag = new Float32Array(size);
  let total = 0;
  note.m.forEach((m, i) => { total += Math.hypot(note.re[i], note.im[i]); });
  note.m.forEach((m, i) => { real[m] = note.re[i] / total; imag[m] = -note.im[i] / total; });
  return ctx.createPeriodicWave(real, imag, { disableNormalization: true });
}

function noteOn(n) {
  const note = state.notes[n];
  if (!note || !note.m.length) return;
  audio();
  if (voices[n]) releaseVoice(n, 0.01);
  const osc = ctx.createOscillator();
  osc.setPeriodicWave(periodicWave(note));
  osc.frequency.value = midiFreq(n);
  const gain = ctx.createGain();
  const now = ctx.currentTime, attack = +$('attack').value / 1000;
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(0.3, now + attack);
  gain.gain.setTargetAtTime(0, now + attack, +$('decay').value);   // exp(-t / decay)
  osc.connect(gain).connect(master);
  osc.start(now);
  voices[n] = { osc, gain, sustained: false };
  keyElement(n)?.classList.add('active');
  if (state.focus !== n) {
    state.focus = n;
    scheduleUpdate(150);
  }
}

function noteOff(n) {
  keyElement(n)?.classList.remove('active');
  const voice = voices[n];
  if (!voice) return;
  if ($('sustain').checked) voice.sustained = true;
  else releaseVoice(n);
}

function releaseVoice(n, release = +$('release').value) {
  const voice = voices[n];
  if (!voice) return;
  delete voices[n];
  const now = ctx.currentTime;
  voice.gain.gain.cancelScheduledValues(now);
  voice.gain.gain.setValueAtTime(voice.gain.gain.value, now);
  voice.gain.gain.setTargetAtTime(0, now, release);
  voice.osc.stop(now + 10 * release);
}

function releaseSustained() {
  Object.keys(voices).forEach(n => { if (voices[n].sustained) releaseVoice(+n); });
}

// ---------- keyboard ----------

const isBlack = n => [1, 3, 6, 8, 10].includes(n % 12);
const keyElement = n => document.querySelector(`.key[data-note="${n}"]`);

function buildKeyboard() {
  const board = $('keyboard');
  const whites = range(LOW, HIGH + 1).filter(n => !isBlack(n));
  const width = 100 / whites.length;
  const letters = Object.fromEntries(Object.entries(KEYMAP).map(([k, n]) => [n, k.toUpperCase()]));
  range(LOW, HIGH + 1).forEach(n => {
    const key = document.createElement('div');
    key.className = `key ${isBlack(n) ? 'black' : 'white'}`;
    key.dataset.note = n;
    const index = whites.filter(w => w < n).length;
    key.style.left = `${isBlack(n) ? index * width - width * 0.3 : index * width}%`;
    key.style.width = `${isBlack(n) ? width * 0.6 : width}%`;
    const label = document.createElement('span');
    label.textContent = (n % 12 === 0 ? noteName(n) + ' ' : '') + (letters[n] || '');
    key.appendChild(label);
    key.addEventListener('pointerdown', e => { e.preventDefault(); key.releasePointerCapture(e.pointerId); noteOn(n); });
    key.addEventListener('pointerenter', e => { if (e.buttons) noteOn(n); });
    key.addEventListener('pointerup', () => noteOff(n));
    key.addEventListener('pointerleave', () => noteOff(n));
    board.appendChild(key);
  });
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'range' && e.target.type !== 'checkbox' && e.target.type !== 'radio') return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code === 'Space') {
    e.preventDefault();
    if (e.repeat) return;
    $('sustain').checked = !$('sustain').checked;
    if (!$('sustain').checked) releaseSustained();
    return;
  }
  const n = KEYMAP[e.key.toLowerCase()];
  if (n !== undefined && !e.repeat) { e.preventDefault(); noteOn(n); }
});
document.addEventListener('keyup', e => {
  const n = KEYMAP[e.key.toLowerCase()];
  if (n !== undefined) noteOff(n);
});

// ---------- charts ----------

function colors() {
  const s = getComputedStyle(document.documentElement);
  const v = name => s.getPropertyValue(name).trim();
  return { text: v('--text'), muted: v('--muted'), grid: v('--grid'), accent: v('--accent'), panel: v('--panel') };
}
const LEVEL_COLORS = ['#e45756', '#f58518', '#d4a72c', '#54a24b', '#4c9be8', '#7b61d6', '#c45ab3', '#2bb5a8'];

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  canvas.dataset.h = canvas.dataset.h || canvas.getAttribute('height');
  const w = canvas.clientWidth, h = +canvas.dataset.h;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.height = `${h}px`;
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w, h };
}

function niceTicks(lo, hi, count = 5) {
  const step0 = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * (hi - lo); v += step) ticks.push(+v.toPrecision(12));
  return ticks;
}

function frame(canvas, xr, yr, { xlabel = '', ylabel = '', xticks, yticks, yfmt = fmt } = {}) {
  const { g, w, h } = setupCanvas(canvas);
  const css = colors();
  const L = 56, R = 12, T = 10, B = 34;
  const X = v => L + (v - xr[0]) / (xr[1] - xr[0]) * (w - L - R);
  const Y = v => T + (1 - (v - yr[0]) / (yr[1] - yr[0])) * (h - T - B);
  g.clearRect(0, 0, w, h);
  g.font = '11px system-ui, sans-serif';
  g.lineWidth = 1;
  g.strokeStyle = css.grid;
  g.fillStyle = css.muted;
  g.textAlign = 'right';
  g.textBaseline = 'middle';
  for (const t of yticks || niceTicks(yr[0], yr[1])) {
    g.beginPath(); g.moveTo(L, Y(t)); g.lineTo(w - R, Y(t)); g.stroke();
    g.fillText(yfmt(t), L - 6, Y(t));
  }
  g.textAlign = 'center';
  g.textBaseline = 'top';
  for (const t of xticks || niceTicks(xr[0], xr[1], Math.max(2, Math.floor((w - L - R) / 70)))) {
    g.fillText(fmt(t), X(t), h - B + 6);
  }
  g.textAlign = 'right';
  g.textBaseline = 'bottom';
  g.fillText(xlabel, w - R, h - 1);
  g.save();
  g.translate(12, T + (h - T - B) / 2);
  g.rotate(-Math.PI / 2);
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(ylabel, 0, 0);
  g.restore();
  const clip = () => { g.beginPath(); g.rect(L, T, w - L - R, h - T - B); g.clip(); };
  return { g, w, h, X, Y, L, R, T, B, css, clip };
}

function levelIndex(k) { return state.potential === 'infinite' ? k + 1 : k; }

function drawWavefunctions() {
  const { x, V, energies, psi } = state.data.plot;
  const amps = state.data.levels.amplitudes;
  const count = energies.length;
  const gaps = energies.slice(1).map((e, i) => e - energies[i]);
  const spacing = gaps.length ? Math.min(...gaps) : energies[0];
  const top = energies[count - 1] + 0.6 * (gaps[gaps.length - 1] ?? spacing);
  const finiteV = V.filter(v => v !== null);
  const bottom = Math.min(0, ...finiteV);
  const yr = [bottom - 0.06 * (top - bottom), top];
  const f = frame($('wavefunctions'), [x[0], x[x.length - 1]], yr, { xlabel: 'x (nm)', ylabel: 'energy (meV)' });
  const { g, X, Y, css } = f;
  g.save();
  f.clip();

  // potential, with vertical walls where it becomes infinite or jumps
  g.strokeStyle = css.text;
  g.lineWidth = 2;
  g.beginPath();
  let pen = false;
  x.forEach((xi, i) => {
    if (V[i] === null) { pen = false; return; }
    const y = Y(Math.min(V[i], top * 2));
    if (pen) g.lineTo(X(xi), y); else g.moveTo(X(xi), y);
    pen = true;
  });
  for (let i = 1; i < x.length; i++) {
    if ((V[i] === null) !== (V[i - 1] === null)) {
      const xi = X((x[i] + x[i - 1]) / 2);
      g.moveTo(xi, Y(V[i] ?? V[i - 1]));
      g.lineTo(xi, f.T);
    }
  }
  g.stroke();

  // levels and wavefunctions, each scaled to fit between its neighbours, opacity
  // following each level's weight
  const amax = Math.max(...amps.slice(0, count));
  for (let k = 0; k < count; k++) {
    const room = Math.min(gaps[k] ?? gaps[k - 1] ?? spacing, gaps[k - 1] ?? Infinity);
    const scale = 0.42 * room / Math.max(...psi[k].map(Math.abs));
    const color = LEVEL_COLORS[k % LEVEL_COLORS.length];
    g.globalAlpha = 0.3 + 0.7 * (amps[k] / amax);
    g.strokeStyle = color;
    g.lineWidth = 1;
    g.setLineDash([3, 4]);
    g.beginPath(); g.moveTo(X(x[0]), Y(energies[k])); g.lineTo(X(x[x.length - 1]), Y(energies[k])); g.stroke();
    g.setLineDash([]);
    g.lineWidth = 2;
    g.beginPath();
    psi[k].forEach((p, i) => {
      const px = X(x[i]), py = Y(energies[k] + scale * p);
      if (i) g.lineTo(px, py); else g.moveTo(px, py);
    });
    g.stroke();
    g.fillStyle = color;
    g.textAlign = 'left';
    g.textBaseline = 'bottom';
    g.font = '11px system-ui, sans-serif';
    g.fillText(`n=${levelIndex(k)}`, f.L + 4, Y(energies[k]) - 2);
  }
  g.globalAlpha = 1;
  g.restore();
}

function drawWeights() {
  const amps = state.data.levels.amplitudes;
  const p = amps.map(a => a * a);
  const total = p.reduce((s, v) => s + v, 0);
  const probs = p.map(v => v / total);
  const n0 = levelIndex(0);
  const f = frame($('weights'), [n0 - 0.6, n0 + probs.length - 0.4], [0, Math.max(...probs) * 1.1],
                  { xlabel: 'level n', ylabel: 'share of power' });
  const { g, X, Y, css } = f;
  const bw = Math.max(2, (X(1) - X(0)) * 0.7);
  probs.forEach((v, k) => {
    g.fillStyle = k < LEVEL_COLORS.length ? LEVEL_COLORS[k] : css.accent;
    g.fillRect(X(n0 + k) - bw / 2, Y(v), bw, Y(0) - Y(v));
  });
  const T = state.data.info.focusTemperature;
  const weighting = $('weighting').value;
  $('weightsSub').textContent = weighting === 'boltzmann'
    ? `at ${noteName(state.focus)}, T = ${fmt(T)} K` : `at ${noteName(state.focus)}`;
}

function waveform(note, phases) {
  return phases.map(ph => note.m.reduce((s, m, i) =>
    s + note.re[i] * Math.cos(2 * Math.PI * m * ph) - note.im[i] * Math.sin(2 * Math.PI * m * ph), 0));
}

function drawPeriod() {
  const note = state.data.focusNote;
  const phases = range(0, 801).map(i => i / 400);
  const y = waveform(note, phases);
  const peak = Math.max(...y.map(Math.abs)) || 1;
  const f = frame($('period'), [0, 2], [-1.1, 1.1], { xlabel: 'periods', ylabel: 'amplitude', yticks: [-1, 0, 1] });
  const { g, X, Y, css } = f;
  g.strokeStyle = css.accent;
  g.lineWidth = 2;
  g.beginPath();
  phases.forEach((ph, i) => { if (i) g.lineTo(X(ph), Y(y[i] / peak)); else g.moveTo(X(ph), Y(y[i] / peak)); });
  g.stroke();
  $('periodSub').textContent = `${noteName(state.focus)}, ${fmt(midiFreq(state.focus))} Hz`;
}

function drawSpectrum() {
  const note = state.data.focusNote;
  const mags = note.m.map((m, i) => Math.hypot(note.re[i], note.im[i]));
  const top = Math.max(...mags);
  const db = mags.map(v => 20 * Math.log10(v / top));
  const floor = -60;
  const shown = note.m.filter((m, i) => db[i] > floor);
  const maxM = Math.min(64, Math.max(8, ...shown));
  const f = frame($('spectrum'), [0.4, maxM + 0.6], [floor, 3], { xlabel: 'harmonic', ylabel: 'dB', yticks: [-60, -40, -20, 0] });
  const { g, X, Y, css } = f;
  const bw = Math.max(2, (X(1) - X(0)) * 0.7);
  g.fillStyle = css.accent;
  note.m.forEach((m, i) => {
    if (m > maxM || db[i] <= floor) return;
    g.fillRect(X(m) - bw / 2, Y(db[i]), bw, Y(floor) - Y(db[i]));
  });
}

function drawInfo() {
  const { info, warnings } = state.data;
  const rows = [];
  if (state.potential === 'oscillator') rows.push(`Level spacing ħω: <b>${fmt(info.hbarOmega_meV)} meV</b>`);
  rows.push(`Ground state: <b>${fmt(info.groundEnergy_meV)} meV</b> (${fmt(info.groundEnergy_K)} K)`);
  if (info.boundStates !== undefined) rows.push(`Bound states: <b>${info.boundStates}</b>, depth ${fmt(info.depth_K)} K`);
  rows.push(`Levels above 1% at ${noteName(state.focus)}: <b>${info.levelsAudible}</b>`);
  $('info').innerHTML = rows.map(r => `<div>${r}</div>`).join('') +
    warnings.map(w => `<div class="warning">${w}</div>`).join('');
}

function drawAll() {
  if (!state.data) return;
  drawWavefunctions();
  drawWeights();
  drawPeriod();
  drawSpectrum();
  drawInfo();
  drawSpectrogramAxis();
}

// ---------- spectrogram ----------

const SPEC_MIN = 40, SPEC_MAX = 12000, DB_MIN = -110, DB_MAX = -25;
const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
                 [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function inferno(t) {
  t = Math.min(1, Math.max(0, t)) * (INFERNO.length - 1);
  const i = Math.min(INFERNO.length - 2, Math.floor(t)), u = t - i;
  return INFERNO[i].map((c, k) => c + (INFERNO[i + 1][k] - c) * u);
}

let specBins = null;
function drawSpectrogram() {
  requestAnimationFrame(drawSpectrogram);
  const canvas = $('spectrogram');
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(260 * dpr);
  const g = canvas.getContext('2d');
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h; canvas.style.height = '260px';
    g.fillStyle = '#000004'; g.fillRect(0, 0, w, h);
  }
  if (!specBins || specBins.length !== analyser.frequencyBinCount) specBins = new Float32Array(analyser.frequencyBinCount);
  analyser.getFloatFrequencyData(specBins);
  const step = Math.max(1, Math.round(2 * dpr));
  g.drawImage(canvas, -step, 0);
  const column = g.createImageData(step, h);
  const binHz = ctx.sampleRate / analyser.fftSize;
  for (let y = 0; y < h; y++) {
    const freq = SPEC_MIN * Math.pow(SPEC_MAX / SPEC_MIN, 1 - y / (h - 1));
    const value = specBins[Math.min(specBins.length - 1, Math.round(freq / binHz))];
    const [r, gg, b] = inferno((value - DB_MIN) / (DB_MAX - DB_MIN));
    for (let x = 0; x < step; x++) {
      const o = (y * step + x) * 4;
      column.data[o] = r; column.data[o + 1] = gg; column.data[o + 2] = b; column.data[o + 3] = 255;
    }
  }
  g.putImageData(column, w - step, 0);
}

function drawSpectrogramAxis() {
  const { g, w, h } = setupCanvas($('specAxis'));
  g.clearRect(0, 0, w, h);
  g.font = '11px system-ui, sans-serif';
  g.textBaseline = 'middle';
  for (const f of [50, 100, 200, 500, 1000, 2000, 5000, 10000]) {
    const y = (1 - Math.log(f / SPEC_MIN) / Math.log(SPEC_MAX / SPEC_MIN)) * (h - 1);
    g.fillStyle = 'rgba(255,255,255,0.75)';
    g.fillText(f >= 1000 ? `${f / 1000} kHz` : `${f} Hz`, 6, y);
    g.fillStyle = 'rgba(255,255,255,0.2)';
    g.fillRect(56, y, 10, 1);
  }
  if (!ctx) {
    g.fillStyle = 'rgba(255,255,255,0.6)';
    g.textAlign = 'center';
    g.fillText('Play a note or render Un sospiro to start the spectrogram', w / 2, h / 2);
  }
}

// ---------- Un sospiro ----------

async function renderPiece() {
  const button = $('render');
  button.disabled = true;
  const started = performance.now();
  const tick = setInterval(() => {
    $('renderStatus').textContent = `Rendering… ${Math.round((performance.now() - started) / 1000)} s`;
  }, 500);
  try {
    audio();
    const body = { ...params(), seconds: +$('seconds').value, attack: +$('attack').value / 1000,
                   decay: +$('decay').value, release: +$('release').value };
    const res = await fetch('/api/render', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                             body: JSON.stringify(body) });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const url = URL.createObjectURL(await res.blob());
    const player = $('player');
    if (!playerSource) {
      playerSource = ctx.createMediaElementSource(player);
      playerSource.connect(analyser);
    }
    player.src = url;
    $('download').href = url;
    $('download').download = `unSospiro_${state.potential}.wav`;
    $('download').hidden = false;
    $('renderStatus').textContent = `Rendered in ${((performance.now() - started) / 1000).toFixed(1)} s`;
    drawSpectrogramAxis();
    player.play();
  } catch (error) {
    $('renderStatus').textContent = `Error: ${error.message}`;
  } finally {
    clearInterval(tick);
    button.disabled = false;
  }
}

// ---------- wiring ----------

document.querySelectorAll('#potential button').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('#potential button').forEach(b => b.classList.toggle('active', b === button));
    state.potential = button.dataset.value;
    refreshOutputs();
    scheduleUpdate(0);
  });
});
['mass', 'length', 'depth', 'temperature', 'weighting'].forEach(id => {
  $(id).addEventListener('input', () => { refreshOutputs(); scheduleUpdate(); });
});
document.querySelectorAll('input[name=tmode]').forEach(r => r.addEventListener('change', () => { refreshOutputs(); scheduleUpdate(0); }));
['attack', 'decay', 'release', 'seconds'].forEach(id => $(id).addEventListener('input', refreshOutputs));
$('volume').addEventListener('input', () => { refreshOutputs(); if (master) master.gain.value = +$('volume').value; });
$('sustain').addEventListener('change', () => { if (!$('sustain').checked) releaseSustained(); });
$('render').addEventListener('click', renderPiece);
window.addEventListener('resize', () => { clearTimeout(window.resizeTimer); window.resizeTimer = setTimeout(drawAll, 100); });

buildKeyboard();
refreshOutputs();
drawSpectrogramAxis();
update();
