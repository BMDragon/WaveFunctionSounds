'use strict';

const $ = id => document.getElementById(id);
const NAMES = ['C', 'C♯', 'D', 'E♭', 'E', 'F', 'F♯', 'G', 'A♭', 'A', 'B♭', 'B'];
const LOW = 36, HIGH = 84;      // keyboard range, C2..C6
const KEYMAP = {
  z: 48, s: 49, x: 50, d: 51, c: 52, v: 53, g: 54, b: 55, h: 56, n: 57, j: 58, m: 59,
  q: 60, 2: 61, w: 62, 3: 63, e: 64, r: 65, 5: 66, t: 67, 6: 68, y: 69, 7: 70, u: 71,
  i: 72, 9: 73, o: 74, 0: 75, p: 76,
};
// the same keys by physical position (e.code), so Shift or a keyboard layout can't change them
const CODEMAP = Object.fromEntries(Object.entries(KEYMAP).map(([k, n]) => [/\d/.test(k) ? `Digit${k}` : `Key${k.toUpperCase()}`, n]));
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
    barrierFraction: +$('barrierFraction').value,
    blochPhase: +$('blochPhase').value,
    ...($('sound').value !== 'states' ? { sound: $('sound').value } : {}),
    ...(state.potential === 'hydrogen' ? { Z: +$('charge').value, l: +$('ell').value } : {}),
    ...(state.potential === 'rectangle' ? { aspect: +$('aspect').value } : {}),
    ...(state.potential === 'custom' ? { shape: encodeShape(state.shape) } : {}),
    ...($('weighting').value === 'custom' ? { customWeights: state.weights } : {}),
    ...(state.potential === 'doublewell' && $('tunnelling').checked ? { tunnelling: true } : {}),
  };
}

// ---------- hand-set weights ----------

const WEIGHT_LEVELS = 16;
state.weights = range(0, WEIGHT_LEVELS).map(i => +Math.pow(0.7, i).toFixed(2));

function drawWeightBars() {
  const { g, w, h } = setupCanvas($('weightsCanvas'));
  const css = colors();
  g.clearRect(0, 0, w, h);
  const bw = w / WEIGHT_LEVELS;
  state.weights.forEach((v, i) => {
    g.fillStyle = i < LEVEL_COLORS.length ? LEVEL_COLORS[i] : css.accent;
    g.fillRect(i * bw + 2, (1 - v) * (h - 14), bw - 4, v * (h - 14));
    g.fillStyle = css.muted;
    g.font = '9px system-ui, sans-serif';
    g.textAlign = 'center';
    g.fillText(String(levelIndex(i)), i * bw + bw / 2, h - 3);
  });
}

function useWeights() {
  $('weighting').value = 'custom';
  refreshOutputs();
  scheduleUpdate();
  settingsChanged();
}

(() => {
  const canvas = $('weightsCanvas');
  const paint = e => {
    const r = canvas.getBoundingClientRect();
    const i = Math.floor((e.clientX - r.left) / r.width * WEIGHT_LEVELS);
    if (i < 0 || i >= WEIGHT_LEVELS) return;
    state.weights[i] = +Math.min(Math.max(1 - (e.clientY - r.top) / (r.height - 14), 0), 1).toFixed(3);
    drawWeightBars();
  };
  canvas.addEventListener('pointerdown', e => { canvas.setPointerCapture(e.pointerId); paint(e); });
  canvas.addEventListener('pointermove', e => { if (e.buttons) paint(e); });
  canvas.addEventListener('pointerup', useWeights);
  $('useWeights').addEventListener('click', useWeights);
  $('copyWeights').addEventListener('click', () => {
    const amps = state.data ? state.data.levels.amplitudes.slice(0, WEIGHT_LEVELS) : [];
    const top = Math.max(...amps, 1e-12);
    state.weights = range(0, WEIGHT_LEVELS).map(i => +((amps[i] || 0) / top).toFixed(3));
    drawWeightBars();
  });
  $('customize').addEventListener('toggle', drawWeightBars);
})();

// ---------- drawing a potential ----------

const SHAPE_POINTS = 64;
state.shape = range(0, SHAPE_POINTS).map(i => i < 32 ? 0.6 * (1 - i / 31) : 0.3 * (i - 32) / 31);
const encodeShape = shape => shape.map(v => Math.round(Math.min(Math.max(v, 0), 1) * 35).toString(36)).join('');
const decodeShape = text => [...text].map(c => parseInt(c, 36) / 35);

function drawShape() {
  const { g, w, h } = setupCanvas($('drawCanvas'));
  const css = colors();
  g.clearRect(0, 0, w, h);
  g.strokeStyle = css.grid;
  for (let i = 1; i < 4; i++) { g.beginPath(); g.moveTo(0, h * i / 4); g.lineTo(w, h * i / 4); g.stroke(); }
  g.fillStyle = css.accent + '33';
  g.strokeStyle = css.text;
  g.lineWidth = 2;
  g.beginPath();
  g.moveTo(0, h);
  state.shape.forEach((v, i) => g.lineTo(i / (SHAPE_POINTS - 1) * w, (1 - v) * (h - 4) + 2));
  g.lineTo(w, h);
  g.fill();
  g.beginPath();
  state.shape.forEach((v, i) => { const px = i / (SHAPE_POINTS - 1) * w, py = (1 - v) * (h - 4) + 2; if (i) g.lineTo(px, py); else g.moveTo(px, py); });
  g.stroke();
}

function useDrawing() {
  selectPotential('custom');
  refreshOutputs();
  scheduleUpdate();
  settingsChanged();
}

(() => {
  const canvas = $('drawCanvas');
  let last = null;
  const point = e => {
    const r = canvas.getBoundingClientRect();
    return [Math.round((e.clientX - r.left) / r.width * (SHAPE_POINTS - 1)), 1 - (e.clientY - r.top) / r.height];
  };
  const paint = e => {
    const [i, v] = point(e);
    const [i0, v0] = last || [i, v];
    const steps = Math.max(1, Math.abs(i - i0));
    for (let s = 0; s <= steps; s++) {        // fill in the points between two pointer events
      const j = Math.round(i0 + (i - i0) * s / steps);
      if (j >= 0 && j < SHAPE_POINTS) state.shape[j] = Math.min(Math.max(v0 + (v - v0) * s / steps, 0), 1);
    }
    last = [i, v];
    drawShape();
  };
  canvas.addEventListener('pointerdown', e => { canvas.setPointerCapture(e.pointerId); last = null; paint(e); });
  canvas.addEventListener('pointermove', e => { if (e.buttons) paint(e); });
  canvas.addEventListener('pointerup', () => { last = null; useDrawing(); });
  $('useDrawing').addEventListener('click', useDrawing);
  $('flatDrawing').addEventListener('click', () => { state.shape = state.shape.map(() => 0); drawShape(); useDrawing(); });
  $('customize').addEventListener('toggle', drawShape);
})();

// slider labels and a one-line description for each potential; no `depth` hides that slider
const POTENTIALS = {
  infinite: { length: 'Well width', hint: 'A particle in a box with infinitely high walls.' },
  finite: { length: 'Well width', depth: 'Well depth', hint: 'A box with walls of finite height: the states leak into the walls.' },
  oscillator: { length: 'Ground-state width', hint: 'V = ½mω²x²: evenly spaced levels.' },
  doublewell: { length: 'Well separation', depth: 'Barrier height',
                hint: 'Two wells side by side: below the barrier the levels pair up, split only by tunnelling.' },
  morse: { length: 'Range', depth: 'Dissociation energy',
           hint: 'A molecular bond: lopsided, with levels crowding together towards dissociation.' },
  lattice: { length: 'Lattice spacing', depth: 'Barrier height', lattice: true,
             hint: 'A Kronig–Penney crystal. Bloch phases other than 0 and π make the partials inharmonic.' },
  linear: { length: 'Width', depth: 'Height at the edges',
            hint: 'V = V₀|x| / (width/2): Airy-function states whose levels crowd together.' },
  circle: { length: 'Diameter',
            hint: 'A circular drum: a particle in a 2D disc. Each state is a partial at its wavenumber, so the partials are inharmonic (Bessel zeros). The plot shows each state along a diameter.' },
  rectangle: { length: 'Width', aspect: true,
               hint: 'A rectangular drum: a particle in a 2D box. Each state is a partial at its wavenumber, π√((a/W)² + (b/H)²), mostly inharmonic. The plot shows each state across the box.' },
  hydrogen: { atom: true,
              hint: 'A hydrogen-like atom: the radial states u = rR in the Coulomb potential plus the centrifugal ' +
                    'barrier for angular momentum ℓ, with levels −Z² Ry / n². The plot shows u(r).' },
  custom: { length: 'Width', depth: 'Height at the top',
            hint: 'Your drawn potential (Customize, below), with hard walls at its edges.' },
};

const attackSeconds = () => Math.pow(10, +$('attack').value) / 1000;     // the attack slider is logarithmic, 1 ms to 3 s
const decaySeconds = () => Math.pow(10, +$('decay').value);              // so is the decay slider, 0.03 s to 6 s

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
  $('attackOut').textContent = attackSeconds() < 1 ? `${Math.round(attackSeconds() * 1000)} ms` : `${attackSeconds().toFixed(2)} s`;
  $('sustainLevelOut').textContent = `${Math.round($('sustainLevel').value * 100)}%`;
  $('decayOut').textContent = `${fmt(decaySeconds(), 2)} s`;
  $('releaseOut').textContent = `${(+$('release').value).toFixed(2)} s`;
  $('volumeOut').textContent = `${Math.round($('volume').value * 100)}%`;
  $('secondsOut').textContent = `${$('seconds').value} s`;
  const kind = POTENTIALS[state.potential];
  $('lengthLabel').textContent = kind.length || '';
  $('depthLabel').textContent = kind.depth || '';
  $('depthRow').hidden = !kind.depth;
  $('barrierRow').hidden = $('blochRow').hidden = !kind.lattice;
  $('tunnellingRow').hidden = state.potential !== 'doublewell';
  $('aspectRow').hidden = !kind.aspect;
  $('lengthRow').hidden = !kind.length;
  $('chargeRow').hidden = $('ellRow').hidden = !kind.atom;
  $('chargeOut').textContent = $('charge').value;
  $('ellOut').textContent = $('ell').value;
  $('soundHint').textContent = {
    survival: 'The particle starts in the superposition of the states with these weights, and you hear the probability of ' +
              'still finding it in that state, |⟨ψ(0)|ψ(t)⟩|²: partials at the energy differences, with E₁ − E₀ at the key.',
    ground: 'Each state laid out at its real size on the ground state\'s width as the unit: the wider, higher states carry ' +
            'over into neighbouring units instead of being squeezed to fit, so each sounds near its own wavenumber.',
    states: 'Each state is a partial; the wavefunction, laid end to end, is one period of the sound.',
  }[$('sound').value];
  $('tunnellingRow').hidden ||= $('sound').value === 'survival';
  $('aspectOut').textContent = (+$('aspect').value).toFixed(2);
  $('potentialHint').textContent = kind.hint;
  $('barrierFractionOut').textContent = `${Math.round(p.barrierFraction * 100)}% of the spacing`;
  $('blochPhaseOut').textContent = `${p.blochPhase.toFixed(2)} π`;
  $('temperatureRows').hidden = p.weighting !== 'boltzmann';
  document.querySelector('#weighting option[value=custom]').hidden = p.weighting !== 'custom';
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

let additiveReady = null;      // the oscillator bank for partials at any frequencies

function additiveVoice(note, freq, total) {
  // Partials at any ratios (the 2D boxes' modes), summed by the oscillator bank in
  // additive.js; the node joins the voice as soon as the module has loaded
  const output = ctx.createGain();
  const source = { node: null, stopAt: Infinity,
                   stop(when) { this.stopAt = when; this.node?.port.postMessage(when); } };
  additiveReady.then(() => {
    if (ctx.currentTime >= source.stopAt) return;
    source.node = new AudioWorkletNode(ctx, 'additive', {
      numberOfInputs: 0, outputChannelCount: [1],
      processorOptions: { freqs: note.m.map(r => r * freq), re: note.re.map(v => v / total), im: note.im.map(v => v / total) } });
    source.node.port.postMessage(source.stopAt);
    source.node.connect(output);
  });
  return { output, sources: [source] };
}

function audio() {
  if (!ctx) {
    ctx = new AudioContext();
    additiveReady = ctx.audioWorklet.addModule('additive.js');
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

const amplitudeSum = note => note.m.reduce((sum, m, i) => sum + Math.hypot(note.re[i], note.im[i]), 0);

function periodicWave(note, total = amplitudeSum(note)) {
  // PeriodicWave sums real[m] cos + imag[m] sin; our waveform is Re(sum c_m e^{i...}),
  // i.e. Re(c) cos - Im(c) sin. Scaled by the sum of the amplitudes, so every note peaks at
  // most at 1.
  const size = Math.max(...note.m) + 1;
  const real = new Float32Array(size), imag = new Float32Array(size);
  note.m.forEach((m, i) => { real[m] = note.re[i] / total; imag[m] = -note.im[i] / total; });
  return ctx.createPeriodicWave(real, imag, { disableNormalization: true });
}

const isHarmonic = note => note.m.every(r => r >= 1 && Math.abs(r - Math.round(r)) < 1e-9);

function buildVoice(note, freq, start, total = amplitudeSum(note)) {
  // An audio node playing the note, and the sources to stop when it is released
  if (isHarmonic(note)) {
    const osc = ctx.createOscillator();
    osc.setPeriodicWave(periodicWave(note, total));
    osc.frequency.value = freq;
    osc.start(start);
    return { output: osc, sources: [osc] };
  }
  const offset = note.m[0] - Math.floor(note.m[0]);
  if (!note.m.every(r => Math.abs(r - offset - Math.round(r - offset)) < 1e-9)) return additiveVoice(note, freq, total);
  // Inharmonic partials at (n + offset) freq with integer n (the lattice's Bloch phase):
  // the waveform is Re(e^{2πi offset freq t} V(t)) with V(t) = sum c_n e^{2πi n freq t}
  // periodic, i.e. cos(2π offset freq t) Re V - sin(2π offset freq t) Im V. Re V and Im V
  // play as periodic waves, ring-modulated by a cosine and a sine at offset * freq.
  const ns = note.m.map(r => Math.round(r - offset));
  const size = Math.max(1, ...ns.map(Math.abs)) + 1;
  const reCos = new Float32Array(size), reSin = new Float32Array(size);
  const imCos = new Float32Array(size), imSin = new Float32Array(size);
  let dcRe = 0, dcIm = 0;
  ns.forEach((n, i) => {
    const re = note.re[i] / total, im = note.im[i] / total, k = Math.abs(n), s = Math.sign(n);
    if (n === 0) { dcRe += re; dcIm += im; return; }
    // c e^{inθ} = (re cos kθ - s im sin kθ) + i (s re sin kθ + im cos kθ) with k = |n|, s = sign(n)
    reCos[k] += re; reSin[k] -= s * im;
    imCos[k] += im; imSin[k] += s * re;
  });
  const sum = ctx.createGain();
  const sources = [];
  const wave = (cos, sin) => ctx.createPeriodicWave(cos, sin, { disableNormalization: true });
  const branch = (cos, sin, dc, carrierWave, sign) => {
    const mixer = ctx.createGain();
    mixer.gain.value = 0;                   // the carrier drives this gain: a ring modulator
    const osc = ctx.createOscillator();
    osc.setPeriodicWave(wave(cos, sin));
    osc.frequency.value = freq;
    osc.connect(mixer);
    const constant = ctx.createConstantSource();
    constant.offset.value = dc;
    constant.connect(mixer);
    const carrier = ctx.createOscillator();
    carrier.setPeriodicWave(carrierWave);
    carrier.frequency.value = offset * freq;
    carrier.connect(mixer.gain);
    const out = ctx.createGain();
    out.gain.value = sign;
    mixer.connect(out).connect(sum);
    sources.push(osc, constant, carrier);
  };
  branch(reCos, reSin, dcRe, wave(new Float32Array([0, 1]), new Float32Array([0, 0])), 1);    // cosine carrier
  branch(imCos, imSin, dcIm, wave(new Float32Array([0, 0]), new Float32Array([0, 1])), -1);   // sine carrier
  sources.forEach(source => source.start(start));
  return { output: sum, sources };
}

// notes outside the on-screen keyboard (a MIDI keyboard has 88 keys) are fetched when first played
const extraNotes = new Map();
let extraNotesKey = '';
const heldKeys = new Set();

function noteData(n) {
  const key = JSON.stringify(params());
  if (key !== extraNotesKey) { extraNotes.clear(); extraNotesKey = key; }
  if (!extraNotes.has(n)) {
    extraNotes.set(n, fetch('/api/notes', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify({ ...params(), freqs: [midiFreq(n)] }) })
      .then(res => res.json()).then(json => json.notes[0]).catch(() => { extraNotes.delete(n); return null; }));
  }
  return extraNotes.get(n);
}

function noteOn(n, velocity = 127) {
  heldKeys.add(n);
  if (state.notes[n]) startVoice(n, state.notes[n], velocity);
  else noteData(n).then(note => { if (note && heldKeys.has(n)) startVoice(n, note, velocity); });
}

function tunnellingVoice(note, freq, start) {
  // Each group of partials (a pair of states below the barrier, started in the left well)
  // sounds on the left with the probability of finding the particle in the left well,
  // P_L = (1 + cos 2π rate t) / 2, and on the right with 1 - P_L; groups without a rate
  // (states above the barrier) sound equally on both sides.
  const groups = note.groups.filter(group => group.m.length);
  const total = groups.reduce((sum, group) => sum + amplitudeSum(group), 0);
  const merger = ctx.createChannelMerger(2);
  const sources = [];
  const cosine = ctx.createPeriodicWave(new Float32Array([0, 1]), new Float32Array([0, 0]), { disableNormalization: true });
  for (const group of groups) {
    const voice = buildVoice(group, freq, start, total);
    sources.push(...voice.sources);
    let lfo = null;
    if (group.rate !== null) {
      lfo = ctx.createOscillator();
      lfo.setPeriodicWave(cosine);
      lfo.frequency.value = group.rate;
      sources.push(lfo);
      lfo.start(start);
    }
    [1, -1].forEach((sign, channel) => {
      const side = ctx.createGain();
      side.gain.value = 0.5;
      if (lfo) {
        const depth = ctx.createGain();
        depth.gain.value = 0.5 * sign;
        lfo.connect(depth).connect(side.gain);
      }
      voice.output.connect(side).connect(merger, 0, channel);
    });
  }
  return { output: merger, sources };
}

function startVoice(n, note, velocity) {
  if (!note.m.length) return;
  audio();
  if (voices[n]) releaseVoice(n, 0.01);
  const now = ctx.currentTime, attack = attackSeconds(), peak = 0.3 * velocity / 127;
  const { output, sources } = note.groups ? tunnellingVoice(note, midiFreq(n), now) : buildVoice(note, midiFreq(n), now);
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(peak, now + attack);
  gain.gain.setTargetAtTime(peak * +$('sustainLevel').value, now + attack, decaySeconds());   // decay towards the sustain level
  output.connect(gain).connect(master);
  voices[n] = { sources, gain, sustained: false };
  if (!+$('sustainLevel').value) {
    // with no sustain the note dies away even while held: stop it once it is below -100 dB
    sources.forEach(source => source.stop(now + attack + Math.log(1e5) * decaySeconds()));
  }
  keyElement(n)?.classList.add('active');
  if (state.focus !== n) {
    state.focus = n;
    focusChanged = true;
  }
}

// The charts follow the last note played, but redrawing them while keys are down delays the
// keys that follow, so they catch up once every key is up
let focusChanged = false;
function chartsCatchUp() {
  if (!focusChanged || heldKeys.size) return;
  clearTimeout(updateTimer);
  updateTimer = setTimeout(() => {
    if (heldKeys.size) return;                  // playing again: wait for the next release
    focusChanged = false;
    update();
  }, 150);
}

function noteOff(n) {
  heldKeys.delete(n);
  chartsCatchUp();
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
  voice.sources.forEach(source => source.stop(now + 10 * release));
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
    setPedal(!$('sustain').checked);
    return;
  }
  const n = CODEMAP[e.code];
  if (n !== undefined && !e.repeat) { e.preventDefault(); noteOn(n); }
});
document.addEventListener('keyup', e => {
  const n = CODEMAP[e.code];
  if (n !== undefined) noteOff(n);
});
// keys released while the page is not focused never send keyup: let go of them all
window.addEventListener('blur', () => [...heldKeys].forEach(noteOff));

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
  // gaps to the nearest distinct level: tunnelling pairs are nearly degenerate, so the two
  // states of a pair share a gap instead of squeezing each other flat
  const tolerance = count > 1 ? 0.02 * (energies[count - 1] - energies[0]) / (count - 1) : 0;
  const nearest = (e, side) => {
    const others = energies.filter(o => side * (o - e) > tolerance);
    return others.length ? Math.abs((side > 0 ? Math.min(...others) : Math.max(...others)) - e) : Infinity;
  };
  const below = energies.map(e => nearest(e, -1)), above = energies.map(e => nearest(e, 1));
  const distinct = below.concat(above).filter(Number.isFinite);
  const spacing = distinct.length ? Math.min(...distinct) : Math.abs(energies[0]) || 1;
  const top = energies[count - 1] + 0.6 * (Number.isFinite(below[count - 1]) ? below[count - 1] : spacing);
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
  const labels = [];      // one label per distinct level, e.g. "n=0,1" for a tunnelling pair
  for (let k = 0; k < count; k++) {
    const room = Number.isFinite(Math.min(below[k], above[k])) ? Math.min(below[k], above[k]) : spacing;
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
    if (k && energies[k] - energies[k - 1] <= tolerance) labels[labels.length - 1].text += `,${levelIndex(k)}`;
    else labels.push({ text: `n=${levelIndex(k)}`, energy: energies[k], color, alpha: g.globalAlpha });
  }
  g.textAlign = 'left';
  g.textBaseline = 'bottom';
  g.font = '11px system-ui, sans-serif';
  for (const label of labels) {
    g.globalAlpha = label.alpha;
    g.fillStyle = label.color;
    g.fillText(label.text, f.L + 4, Y(label.energy) - 2);
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
  const harmonic = isHarmonic(note);
  const cycles = harmonic ? 2 : 4;
  const phases = range(0, 400 * cycles + 1).map(i => i / 400);
  const y = waveform(note, phases);
  const peak = Math.max(...y.map(Math.abs)) || 1;
  const f = frame($('period'), [0, cycles], [-1.1, 1.1],
                  { xlabel: harmonic ? 'periods' : 'cycles of the key', ylabel: 'amplitude', yticks: [-1, 0, 1] });
  const { g, X, Y, css } = f;
  g.strokeStyle = css.accent;
  g.lineWidth = 2;
  g.beginPath();
  phases.forEach((ph, i) => { if (i) g.lineTo(X(ph), Y(y[i] / peak)); else g.moveTo(X(ph), Y(y[i] / peak)); });
  g.stroke();
  $('periodSub').textContent = `${noteName(state.focus)}, ${fmt(midiFreq(state.focus))} Hz` +
    (harmonic ? '' : ' — inharmonic, so it never repeats');
}

function drawSpectrum() {
  const note = state.data.focusNote;
  const mags = note.m.map((m, i) => Math.hypot(note.re[i], note.im[i]));
  const top = Math.max(...mags);
  const db = mags.map(v => 20 * Math.log10(v / top));
  const floor = -60;
  const position = note.m.map(Math.abs);          // partial frequency / key frequency
  const shown = position.filter((r, i) => db[i] > floor);
  const maxM = Math.min(64, Math.max(8, ...shown));
  const f = frame($('spectrum'), [0, maxM + 0.6], [floor, 3],
                  { xlabel: isHarmonic(note) ? 'harmonic' : 'frequency / key frequency', ylabel: 'dB', yticks: [-60, -40, -20, 0] });
  const { g, X, Y, css } = f;
  const bw = Math.max(2, (X(1) - X(0)) * 0.35);
  g.fillStyle = css.accent;
  position.forEach((r, i) => {
    if (r > maxM || db[i] <= floor) return;
    g.fillRect(X(r) - bw / 2, Y(db[i]), bw, Y(floor) - Y(db[i]));
  });
}

function drawInfo() {
  const { info, warnings } = state.data;
  const rows = [];
  if (state.potential === 'oscillator') rows.push(`Level spacing ħω: <b>${fmt(info.hbarOmega_meV)} meV</b>`);
  rows.push(`Ground state: <b>${fmt(info.groundEnergy_meV)} meV</b> (${fmt(info.groundEnergy_K)} K)`);
  if (info.boundStates !== undefined) rows.push(`Bound states: <b>${info.boundStates}</b>, depth ${fmt(info.depth_K)} K`);
  if (info.statesBelowBarrier !== undefined) rows.push(`States below the barrier: <b>${info.statesBelowBarrier}</b>` +
    (info.lowestSplitting_meV !== undefined ? `; lowest pair split by ${fmt(info.lowestSplitting_meV)} meV ` +
      `(${fmt(info.lowestSplitting_K)} K, so with tunnelling it crosses back and forth at ${fmt(info.lowestSplitting_K)} Hz)` : ''));
  if (info.bands !== undefined) rows.push(`Bands: <b>${info.bands}</b> (${info.bandsBelowBarrier} below the barrier), ` +
    `Bloch phase ${fmt(info.blochPhase)} π`);
  rows.push(`Levels above 1% at ${noteName(state.focus)}: <b>${info.levelsAudible}</b>`);
  if (info.excitation_meV !== undefined) rows.push(`First excitation E₁ − E₀: <b>${fmt(info.excitation_meV)} meV</b>, ` +
    `played at the key; ${info.survivalPartials} energy differences heard at ${noteName(state.focus)}`);
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
    const body = { ...params(), seconds: +$('seconds').value, attack: attackSeconds(),
                   decay: decaySeconds(), sustain: +$('sustainLevel').value, release: +$('release').value };
    if ($('piece').value === 'upload' && uploadedPiece) body.midi = uploadedPiece.data;
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
    const pieceName = $('piece').value === 'upload' && uploadedPiece ? uploadedPiece.name.replace(/\.midi?$/i, '') : 'unSospiro';
    $('download').download = `${pieceName}_${state.potential}.wav`;
    $('download').hidden = false;
    $('renderStatus').textContent = `Rendered in ${((performance.now() - started) / 1000).toFixed(1)} s`;
    drawSpectrogramAxis();
    player.play().catch(() => {});      // pausing before playback starts is fine
  } catch (error) {
    $('renderStatus').textContent = `Error: ${error.message}`;
  } finally {
    clearInterval(tick);
    button.disabled = false;
  }
}

// ---------- your own MIDI file ----------

let uploadedPiece = null;       // { name, data: base64 }

$('piece').addEventListener('change', () => {
  if ($('piece').value === 'upload') $('midiFile').click();
});
$('midiFile').addEventListener('change', async () => {
  const file = $('midiFile').files[0];
  if (!file) {
    if (!uploadedPiece) $('piece').value = 'builtin';
    return;
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  uploadedPiece = { name: file.name, data: btoa(binary) };
  $('pieceName').textContent = file.name;
  $('midiFile').value = '';
});

// ---------- recording what you play ----------

let recorder = null, recordedChunks = [], recording = false;

async function toggleRecording() {
  audio();
  if (!recorder) {
    await ctx.audioWorklet.addModule('recorder.js');
    recorder = new AudioWorkletNode(ctx, 'recorder');
    recorder.port.onmessage = event => { if (recording) recordedChunks.push(event.data); };
    analyser.connect(recorder);
    recorder.connect(ctx.destination);      // it outputs silence, but keeps the recorder running
  }
  recording = !recording;
  recorder.port.postMessage(recording);
  $('record').textContent = recording ? '■ Stop recording' : '● Record';
  $('record').classList.toggle('recording', recording);
  if (recording) {
    recordedChunks = [];
    $('recording').hidden = true;
    return;
  }
  const length = recordedChunks.reduce((n, chunk) => n + chunk[0].length, 0);
  const wav = encodeWav(recordedChunks, length, ctx.sampleRate);
  const link = $('recording');
  link.href = URL.createObjectURL(wav);
  link.download = `recording_${state.potential}.wav`;
  link.textContent = `Download recording (${(length / ctx.sampleRate).toFixed(1)} s)`;
  link.hidden = false;
}

function encodeWav(chunks, length, rate) {
  // 16-bit stereo PCM; each chunk is [left, right]
  const bytes = 4 * length;
  const buffer = new ArrayBuffer(44 + bytes);
  const view = new DataView(buffer);
  const text = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + bytes, true); text(8, 'WAVE');
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 2, true);
  view.setUint32(24, rate, true); view.setUint32(28, 4 * rate, true); view.setUint16(32, 4, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, bytes, true);
  let offset = 44;
  for (const [left, right] of chunks) {
    for (let i = 0; i < left.length; i++) {
      view.setInt16(offset, Math.max(-1, Math.min(1, left[i])) * 32767, true);
      view.setInt16(offset + 2, Math.max(-1, Math.min(1, right[i])) * 32767, true);
      offset += 4;
    }
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

// ---------- MIDI keyboard ----------

function setPedal(down) {
  $('sustain').checked = down;
  if (!down) releaseSustained();
}

function handleMidi(event) {
  // note on (0x9n, velocity > 0), note off (0x8n, or 0x9n with velocity 0), sustain pedal (CC 64)
  const [status, a, b] = event.data;
  const kind = status & 0xf0;
  if (kind === 0x90 && b > 0) noteOn(a, b);
  else if (kind === 0x80 || (kind === 0x90 && b === 0)) noteOff(a);
  else if (kind === 0xb0 && a === 64) setPedal(b >= 64);
}

async function connectMidi() {
  if (!navigator.requestMIDIAccess) {
    $('midiStatus').textContent = 'This browser has no Web MIDI; try Chrome or Edge.';
    return;
  }
  try {
    const access = await navigator.requestMIDIAccess();
    const attach = () => {
      const names = [];
      access.inputs.forEach(input => { input.onmidimessage = handleMidi; names.push(input.name); });
      $('midiStatus').textContent = names.length ? `Playing from ${names.join(', ')}` : 'No MIDI keyboard found: plug one in.';
    };
    access.onstatechange = attach;
    attach();
    $('midiConnect').hidden = true;
  } catch (error) {
    $('midiStatus').textContent = 'MIDI access was not allowed.';
  }
}

// ---------- presets and shareable links ----------

// Every setting, as short keys for the URL. Sliders on a log scale store the physical value.
const SETTINGS = {
  potential: ['p', () => state.potential, v => selectPotential(v)],
  mass: ['m', () => +Math.pow(10, +$('mass').value).toPrecision(3), v => { $('mass').value = Math.log10(v); }],
  length: ['L', () => +$('length').value, v => { $('length').value = v; }],
  depth: ['V', () => +Math.pow(10, +$('depth').value).toPrecision(3), v => { $('depth').value = Math.log10(v); }],
  barrierFraction: ['bw', () => +$('barrierFraction').value, v => { $('barrierFraction').value = v; }],
  blochPhase: ['phi', () => +$('blochPhase').value, v => { $('blochPhase').value = v; }],
  charge: ['Z', () => +$('charge').value, v => { $('charge').value = v; }],
  ell: ['l', () => +$('ell').value, v => { $('ell').value = v; }],
  aspect: ['ar', () => +$('aspect').value, v => { $('aspect').value = v; }],
  sound: ['snd', () => $('sound').value, v => { $('sound').value = ['survival', 'ground'].includes(v) ? v : 'states'; }],
  tunnelling: ['tu', () => +$('tunnelling').checked, v => { $('tunnelling').checked = !!+v; }],
  weighting: ['w', () => $('weighting').value, v => { $('weighting').value = v; }],
  temperatureMode: ['tm', () => document.querySelector('input[name=tmode]:checked').value,
                    v => { document.querySelector(`input[name=tmode][value=${v === 'fixed' ? 'fixed' : 'pitch'}]`).checked = true; }],
  temperature: ['T', () => +Math.pow(10, +$('temperature').value).toPrecision(4), v => { $('temperature').value = Math.log10(v); }],
  attack: ['a', () => +attackSeconds().toPrecision(3), v => { $('attack').value = Math.log10(v * 1000); }],
  decay: ['d', () => +decaySeconds().toPrecision(3), v => { $('decay').value = Math.log10(v); }],
  sustain: ['s', () => +$('sustainLevel').value, v => { $('sustainLevel').value = v; }],
  release: ['r', () => +$('release').value, v => { $('release').value = v; }],
  shape: ['v', () => (state.potential === 'custom' ? encodeShape(state.shape) : ''),
          v => { if (v) { state.shape = decodeShape(String(v)); drawShape(); } }],
  customWeights: ['cw', () => ($('weighting').value === 'custom' ? encodeShape(state.weights) : ''),
                  v => { if (v) { state.weights = decodeShape(String(v)); drawWeightBars(); } }],
};
let presets = [];

function selectPotential(value) {
  if (!POTENTIALS[value]) return;
  state.potential = value;
  document.querySelectorAll('#potential button').forEach(b => b.classList.toggle('active', b.dataset.value === value));
}

let ensembles = [];
const PRESET_ALIASES = { trombone: 'tenor_trombone', snare: 'snare_drum' };     // ids used by older links

async function loadPresets() {
  try {
    const data = await (await fetch('presets.json')).json();
    presets = data.presets;
    ensembles = data.ensembles || [];
  } catch (error) {
    presets = [];
  }
  // one menu: Default, then each ensemble's instruments (an instrument in several ensembles is
  // listed once, under the first), and "Custom sound" while the settings match no preset
  const menu = $('presets');
  const option = (id, name, parent = menu) => parent.appendChild(Object.assign(document.createElement('option'), { value: id, textContent: name }));
  option('', 'Custom sound').hidden = true;
  if (presets.some(p => p.id === 'default')) option('default', 'Default');
  const groups = ensembles.length ? ensembles : [{ name: 'Instruments', members: presets.map(p => p.id).filter(id => id !== 'default') }];
  const listed = new Set();
  groups.forEach(ensemble => {
    const members = ensemble.members.filter(id => !listed.has(id)).map(id => presets.find(p => p.id === id)).filter(Boolean);
    if (!members.length) return;
    const group = menu.appendChild(Object.assign(document.createElement('optgroup'), { label: ensemble.name }));
    members.forEach(p => { listed.add(p.id); option(p.id, p.name, group); });
  });
  menu.addEventListener('change', () => {
    const preset = presets.find(p => p.id === menu.value);
    if (preset) applySettings(preset, preset.id);
  });
  showPreset();
}

function applySettings(settings, presetId = null) {
  if (settings.tunnelling === undefined) settings = { ...settings, tunnelling: 0 };   // presets leave it off
  if (settings.sound === undefined) settings = { ...settings, sound: 'states' };      // and play the states
  for (const [name, [, , set]] of Object.entries(SETTINGS)) {
    if (settings[name] !== undefined && settings[name] !== null && settings[name] !== '') set(settings[name]);
  }
  presetId = PRESET_ALIASES[presetId] || presetId;
  state.preset = presets.some(p => p.id === presetId) ? presetId : null;
  refreshOutputs();
  showPreset();
  writeUrl();
  scheduleUpdate(0);
}

function settingsChanged() {
  // any manual change turns a preset into a custom sound
  state.preset = null;
  showPreset();
  writeUrl();
}

function showPreset() {
  const preset = presets.find(p => p.id === state.preset);
  $('presets').value = preset ? preset.id : '';
  document.querySelectorAll('.key').forEach(k => {
    const n = +k.dataset.note;
    k.classList.toggle('suggested', !!(preset && preset.keys && n >= preset.keys[0] && n <= preset.keys[1]));
  });
  $('presetHint').textContent = !preset ? 'Custom sound: pick a preset, or share this one with the link below.'
    : preset.match === undefined ? 'The starting sound: an infinite square well at T = pitch.'
    : `Matched to ${preset.notes > 1 ? preset.notes + ' recorded notes' : 'a recording'} of a real ` +
      `${preset.name.toLowerCase()} (${preset.match} dB away on average). ` +
      (preset.keys[1] > preset.keys[0] ? `Its range is ${noteName(preset.keys[0])}–${noteName(preset.keys[1])}`
                                       : `Best on ${noteName(preset.keys[0])}`) + ' (highlighted).';
}

function writeUrl() {
  const query = new URLSearchParams();
  if (state.preset) query.set('preset', state.preset);
  for (const [, [key, get]] of Object.entries(SETTINGS)) query.set(key, get());
  history.replaceState(null, '', '#' + query.toString());
}

function settingsFromUrl() {
  if (location.hash.length < 2) return null;
  const query = new URLSearchParams(location.hash.slice(1));
  const settings = { preset: query.get('preset') };
  for (const [name, [key]] of Object.entries(SETTINGS)) {
    if (!query.has(key)) continue;
    const raw = query.get(key);
    settings[name] = name === 'shape' || name === 'customWeights' || isNaN(+raw) || raw === '' ? raw : +raw;     // codes stay text
  }
  return settings;
}

$('copyLink').addEventListener('click', async () => {
  writeUrl();
  try {
    await navigator.clipboard.writeText(location.href);
    $('copyLink').textContent = 'Link copied';
  } catch (error) {
    window.prompt('Copy this link:', location.href);
  }
  setTimeout(() => { $('copyLink').textContent = 'Copy link to this sound'; }, 1500);
});

// ---------- wiring ----------

document.querySelectorAll('#potential button').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('#potential button').forEach(b => b.classList.toggle('active', b === button));
    state.potential = button.dataset.value;
    refreshOutputs();
    scheduleUpdate(0);
    settingsChanged();
  });
});
['mass', 'length', 'depth', 'temperature', 'weighting', 'barrierFraction', 'blochPhase', 'tunnelling', 'aspect', 'sound', 'charge', 'ell'].forEach(id => {
  $(id).addEventListener('input', () => { refreshOutputs(); scheduleUpdate(); settingsChanged(); });
});
document.querySelectorAll('input[name=tmode]').forEach(r => r.addEventListener('change', () => {
  refreshOutputs(); scheduleUpdate(0); settingsChanged();
}));
['attack', 'decay', 'sustainLevel', 'release'].forEach(id => $(id).addEventListener('input', () => { refreshOutputs(); settingsChanged(); }));
$('seconds').addEventListener('input', refreshOutputs);
$('volume').addEventListener('input', () => { refreshOutputs(); if (master) master.gain.value = +$('volume').value; });
$('sustain').addEventListener('change', () => { if (!$('sustain').checked) releaseSustained(); });
$('midiConnect').addEventListener('click', connectMidi);
$('record').addEventListener('click', toggleRecording);
$('render').addEventListener('click', renderPiece);
window.addEventListener('resize', () => { clearTimeout(window.resizeTimer); window.resizeTimer = setTimeout(drawAll, 100); });

buildKeyboard();
refreshOutputs();
drawSpectrogramAxis();
loadPresets().then(() => {
  const shared = settingsFromUrl();
  if (shared) applySettings(shared, shared.preset);
  else {
    state.preset = 'default';
    showPreset();
    update();
  }
});
