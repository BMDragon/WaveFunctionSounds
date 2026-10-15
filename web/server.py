import functools
import io
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import numpy as np
from scipy.io.wavfile import write

# Local web app for playing with the wavefunction sounds.
# Usage: python web/server.py [port], then open http://localhost:8000
repoRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repoRoot, 'backend'))

import infiniteSquareWell
import finiteSquareWell
import harmonicOscillator
import doubleWell
import morsePotential
import latticePotential
import linearPotential
import unSospiro

electronMass = 9.1093837e-31
eV = 1.602176634e-19
kB = 1.380649e-23
sampleRate = 44100
levelsShown = 30            # levels in the weights chart
lock = threading.Lock()     # the models cache coefficients lazily, so compute one request at a time

@functools.lru_cache(maxsize=16)
def model(potential, mass, length, depth, barrierFraction=0.2, blochPhase=0.8):
    # mass in electron masses, length in nm, depth (or barrier/height) in eV, Bloch phase in units of pi
    m, L, V = mass * electronMass, length * 1e-9, depth * eV
    if potential == 'infinite':
        return infiniteSquareWell.InfiniteSquareWell(m, L)
    if potential == 'finite':
        return finiteSquareWell.FiniteSquareWell(m, L, V)
    if potential == 'doublewell':
        return doubleWell.DoubleWell(m, L, V)
    if potential == 'morse':
        return morsePotential.MorsePotential(m, L, V)
    if potential == 'lattice':
        return latticePotential.LatticePotential(m, L, V, barrierFraction, blochPhase)
    if potential == 'linear':
        return linearPotential.LinearPotential(m, L, V)
    return harmonicOscillator.HarmonicOscillator(m, L)

def requestModel(req):
    lattice = req['potential'] == 'lattice'
    return model(req['potential'], float(req['mass']), float(req['length']), float(req.get('depth', 1.6)),
                 float(req.get('barrierFraction', 0.2)) if lattice else 0.2,
                 float(req.get('blochPhase', 0.8)) if lattice else 0.8)

def temperatureFor(req, freq):
    return float(req['temperature']) if req.get('temperatureMode') == 'fixed' else freq

def levelAmplitudes(m, temperature, weighting, count):
    # Amplitude of each of the first `count` levels, lowest first
    if isinstance(m, infiniteSquareWell.InfiniteSquareWell):
        return m.amplitudes(np.arange(1, count + 1), temperature, weighting)
    return m.amplitudes(temperature, weighting)[:count]

def describe(req):
    m = requestModel(req)
    weighting = req.get('weighting', 'boltzmann')
    focus = float(req['focusFreq'])
    focusT = temperatureFor(req, focus)

    plot = m.plotData(int(req.get('plotStates', 8)))
    count = levelsShown if isinstance(m, (infiniteSquareWell.InfiniteSquareWell, harmonicOscillator.HarmonicOscillator)) \
        else min(levelsShown, len(m.vn) if isinstance(m, finiteSquareWell.FiniteSquareWell) else len(m.eng))
    amps = levelAmplitudes(m, focusT, weighting, count)

    def note(f):
        harm, c = m.harmonicContent(f, sampleRate, temperatureFor(req, f), weighting)
        keep = np.abs(c) > 1e-6 * np.abs(c).max()
        return {'m': harm[keep].tolist(), 're': c[keep].real.tolist(), 'im': c[keep].imag.tolist()}

    energies = m.energies(count)
    allAmps = levelAmplitudes(m, focusT, weighting, 10**4)
    info = {'groundEnergy_meV': energies[0] / (eV * 1e-3), 'groundEnergy_K': energies[0] / kB,
            'levelsAudible': int((allAmps > 0.01 * allAmps.max()).sum()), 'focusTemperature': focusT}
    warnings = []
    if isinstance(m, finiteSquareWell.FiniteSquareWell):
        info['boundStates'] = len(m.vn)
        info['depth_K'] = m.potentialDiff / kB
    if isinstance(m, harmonicOscillator.HarmonicOscillator):
        info['hbarOmega_meV'] = m.hbarOmega / (eV * 1e-3)
        last = m.amplitudes(focusT, weighting)[-1]
        if weighting == 'boltzmann' and last > 1e-4:
            warnings.append(f'Only the lowest {harmonicOscillator.numStates} levels are included; '
                            f'the top one still has amplitude {last:.1e}, so the hottest notes are truncated.')
    if isinstance(m, doubleWell.DoubleWell):
        info['statesBelowBarrier'] = m.statesBelowBarrier()
        pairs = [(m.eng[2*k + 1] - m.eng[2*k]) * 1e3 for k in range(m.statesBelowBarrier() // 2)]
        if pairs:
            info['lowestSplitting_meV'] = pairs[0]
    if isinstance(m, morsePotential.MorsePotential):
        info['boundStates'] = len(m.eng)
        info['depth_K'] = m.D * eV / kB
    if isinstance(m, latticePotential.LatticePotential):
        info['bands'] = len(m.eng)
        info['bandsBelowBarrier'] = int((m.eng < m.barrier / eV).sum())
        info['blochPhase'] = m.blochPhase
    truncatable = (doubleWell.DoubleWell, linearPotential.LinearPotential, latticePotential.LatticePotential)
    if isinstance(m, truncatable) or (isinstance(m, morsePotential.MorsePotential) and len(m.eng) == m.maxStates):
        last = m.amplitudes(focusT, weighting)[-1]
        if weighting == 'boltzmann' and last > 1e-4:
            warnings.append(f'Only the lowest {len(m.eng)} levels are included; '
                            f'the top one still has amplitude {last:.1e}, so the hottest notes are truncated.')

    return {'plot': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in plot.items()
                     if k != 'psi'} | {'psi': [p.tolist() for p in plot['psi']]},
            'levels': {'energies_meV': (energies / (eV * 1e-3)).tolist(), 'amplitudes': amps.tolist()},
            'notes': [note(float(f)) for f in req.get('freqs', [])], 'focusNote': note(focus),
            'info': info, 'warnings': warnings}

def renderPiece(req):
    path = os.path.join(repoRoot, 'midi', 'unSospiro.mid')
    if not os.path.exists(path):
        raise FileNotFoundError('midi/unSospiro.mid not found')
    m = requestModel(req)
    weighting = req.get('weighting', 'boltzmann')
    seconds = min(max(float(req.get('seconds', 15)), 1), 120)
    try:
        notes = unSospiro.readNotes(path, seconds)
    except ImportError:
        raise FileNotFoundError('rendering needs the mido package: pip install mido')
    audio = unSospiro.render(
        notes, lambda f, sr: m.harmonicContent(f, sr, temperatureFor(req, f), weighting),
        nSeconds=seconds, attackTime=float(req.get('attack', 0.005)),
        decayTime=float(req.get('decay', 1.5)), releaseTime=float(req.get('release', 0.1)),
        sustainLevel=min(max(float(req.get('sustain', 0)), 0), 1))
    buffer = io.BytesIO()
    write(buffer, sampleRate, infiniteSquareWell.scale(audio))
    return buffer.getvalue()

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(repoRoot, 'web', 'static'), **kwargs)

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            with lock:
                if self.path == '/api/model':
                    body, kind = json.dumps(describe(req)).encode(), 'application/json'
                elif self.path == '/api/render':
                    body, kind = renderPiece(req), 'audio/wav'
                else:
                    self.send_error(404)
                    return
        except Exception as error:
            body = json.dumps({'error': f'{type(error).__name__}: {error}'}).encode()
            self.send_response(400 if isinstance(error, (KeyError, ValueError, FileNotFoundError)) else 500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if self.command == 'POST':
            sys.stderr.write(f'{self.command} {self.path} {args[1] if len(args) > 1 else ""}\n')

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    harmonicOscillator.harmonicContent(220)     # every oscillator shares these coefficients: compute them up front
    print(f'Serving on http://localhost:{port}  (Ctrl+C to stop)')
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
