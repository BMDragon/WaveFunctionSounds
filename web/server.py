import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)    # Ctrl+C just stops the server, without a traceback

import base64
import functools
import io
import json
import os
import sys
import threading
import traceback
import warnings
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
import customPotential
import box2D
import timeEvolution
import hydrogen
import unSospiro

electronMass = 9.1093837e-31
eV = 1.602176634e-19
kB = 1.380649e-23
sampleRate = 44100
levelsShown = 30            # levels in the weights chart
lock = threading.Lock()     # the models cache coefficients lazily, so compute one request at a time

@functools.lru_cache(maxsize=16)
def model(potential, mass, length, depth, barrierFraction=0.2, blochPhase=0.8, shape='', aspect=1.5, Z=1, l=0):
    # mass in electron masses, length in nm, depth (or barrier/height) in eV, Bloch phase in
    # units of pi, shape the code of a drawn potential, aspect the rectangle's height / width,
    # Z and l the hydrogen-like atom's nuclear charge and angular momentum
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
    if potential == 'hydrogen':
        return hydrogen.Hydrogen(m, Z, l)
    if potential == 'circle':
        return box2D.CircularBox(m, L)
    if potential == 'rectangle':
        return box2D.RectangularBox(m, L, aspect)
    if potential == 'custom':
        return customPotential.CustomPotential(m, L, V, customPotential.decodeShape(shape))
    return harmonicOscillator.HarmonicOscillator(m, L)

def requestModel(req):
    lattice, custom = req['potential'] == 'lattice', req['potential'] == 'custom'
    shape = str(req.get('shape', '')) if custom else ''
    if custom and (not 2 <= len(shape) <= 256 or any(c not in '0123456789abcdefghijklmnopqrstuvwxyz' for c in shape)):
        raise ValueError('a drawn potential needs a shape code of 2-256 characters 0-9 and a-z')
    return model(req['potential'], float(req['mass']), float(req['length']), float(req.get('depth', 1.6)),
                 float(req.get('barrierFraction', 0.2)) if lattice else 0.2,
                 float(req.get('blochPhase', 0.8)) if lattice else 0.8, shape,
                 min(max(float(req.get('aspect', 1.5)), 1), 4) if req['potential'] == 'rectangle' else 1.5,
                 *((min(max(int(req.get('Z', 1)), 1), 10), min(max(int(req.get('l', 0)), 0), 20))
                   if req['potential'] == 'hydrogen' else (1, 0)))

def weightingFor(req):
    # 'boltzmann', 'equal', 'inverse', or 'custom' with hand-set amplitudes in 'customWeights'
    weighting = req.get('weighting', 'boltzmann')
    if weighting != 'custom':
        return weighting
    weights = req.get('customWeights')
    if not isinstance(weights, list) or not 1 <= len(weights) <= 64 or \
            not all(isinstance(w, (int, float)) and 0 <= w <= 1 for w in weights) or not any(weights):
        raise ValueError('hand-set weights need 1-64 amplitudes between 0 and 1, not all zero')
    return tuple(float(w) for w in weights)

def temperatureFor(req, freq):
    return float(req['temperature']) if req.get('temperatureMode') == 'fixed' else freq

def levelAmplitudes(m, temperature, weighting, count):
    # Amplitude of each of the first `count` levels, lowest first
    if isinstance(m, infiniteSquareWell.InfiniteSquareWell):
        return m.amplitudes(np.arange(1, count + 1), temperature, weighting)
    return m.amplitudes(temperature, weighting)[:count]

liveSurvivalPartials = 400  # the strongest partials of a survival-probability note played live

def tunnelling(m, req):
    return bool(req.get('tunnelling')) and isinstance(m, doubleWell.DoubleWell) and not survival(req)

def survival(req):
    return req.get('sound') == 'survival'

def wrapping(req):
    # sound 'ground': the wavefunctions wrapped on the ground state's unit instead of each its own
    return 'ground' if req.get('sound') == 'ground' else 'own'

def stateCount(m):
    # how many states a model keeps
    if isinstance(m, infiniteSquareWell.InfiniteSquareWell):
        return box2D.Box2D.maxStates
    if isinstance(m, harmonicOscillator.HarmonicOscillator):
        return harmonicOscillator.numStates
    if isinstance(m, finiteSquareWell.FiniteSquareWell):
        return len(m.vn)
    return len(m.eng)

def content(m, req, f, sr, weighting):
    # the partials of a note: the wavefunctions' (harmonicContent, each state on its own unit or,
    # for sound = 'ground', on the ground state's) or, for sound = 'survival', the survival
    # probability's (see timeEvolution)
    T = temperatureFor(req, f)
    if survival(req):
        count = stateCount(m)
        return timeEvolution.survivalContent(m.energies(count), levelAmplitudes(m, T, weighting, count), f, sr)
    return m.harmonicContent(f, sr, T, weighting, wrapping(req))

def noteContent(m, req, f, weighting):
    # the partials of one note, dropping the negligible ones; with tunnelling, also the
    # groups of partials that move between the wells (see DoubleWell.tunnellingGroups)
    harm, c = content(m, req, f, sampleRate, weighting)
    if not len(c):
        return {'m': [], 're': [], 'im': []}
    keep = np.abs(c) > 1e-6 * np.abs(c).max()
    if survival(req) and keep.sum() > liveSurvivalPartials:
        keep = np.zeros(len(c), bool)
        keep[np.argsort(-np.abs(c), kind='stable')[:liveSurvivalPartials]] = True     # the lowest, among equals
    note = {'m': harm[keep].tolist(), 're': c[keep].real.tolist(), 'im': c[keep].imag.tolist(),
            'level': unSospiro.loudnessGain(harm, c, f)}     # scales the note to the common loudness
    if tunnelling(m, req):
        groups = m.tunnellingGroups(f, sampleRate, temperatureFor(req, f), weighting, wrapping(req))
        biggest = max(np.abs(gc).max() for _, gc, _ in groups)
        note['groups'] = []
        for gm, gc, rate in groups:
            keep = np.abs(gc) > 1e-6 * biggest
            note['groups'].append({'m': gm[keep].tolist(), 're': gc[keep].real.tolist(),
                                   'im': gc[keep].imag.tolist(), 'rate': rate})
        note['level'] = unSospiro.groupsGain(groups, f)
    return note

def notes(req):
    # just the notes for the requested pitches, for keys outside the on-screen keyboard
    m = requestModel(req)
    weighting = weightingFor(req)
    return {'notes': [noteContent(m, req, float(f), weighting) for f in req.get('freqs', [])]}

def describe(req):
    m = requestModel(req)
    weighting = weightingFor(req)
    focus = float(req['focusFreq'])
    focusT = temperatureFor(req, focus)

    plot = m.plotData(int(req.get('plotStates', 8)))
    count = levelsShown if isinstance(m, (infiniteSquareWell.InfiniteSquareWell, harmonicOscillator.HarmonicOscillator)) \
        else min(levelsShown, len(m.vn) if isinstance(m, finiteSquareWell.FiniteSquareWell) else len(m.eng))
    amps = levelAmplitudes(m, focusT, weighting, count)

    note = lambda f: noteContent(m, req, f, weighting)

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
            info['lowestSplitting_K'] = pairs[0] * 1e-3 * eV / kB     # = the tunnelling rate in Hz
    if isinstance(m, morsePotential.MorsePotential):
        info['boundStates'] = len(m.eng)
        info['depth_K'] = m.D * eV / kB
    if isinstance(m, latticePotential.LatticePotential):
        info['bands'] = len(m.eng)
        info['bandsBelowBarrier'] = int((m.eng < m.barrier / eV).sum())
        info['blochPhase'] = m.blochPhase
    if survival(req):
        harm, c = content(m, req, focus, sampleRate, weighting)
        heard = int((np.abs(c) > 1e-6 * np.abs(c).max()).sum()) if len(c) else 0
        gaps = m.energies(stateCount(m)) - m.energies(1)[0]
        nonzero = gaps[gaps > 1e-12 * gaps.max()] if gaps.max() > 0 else gaps[:0]
        if len(nonzero):
            info['excitation_meV'] = nonzero[0] / (eV * 1e-3)
        info['survivalPartials'] = heard
        if heard > liveSurvivalPartials:
            warnings.append(f'{heard} energy differences are heard at {round(focus)} Hz; the keyboard plays the '
                            f'{liveSurvivalPartials} strongest, and Render plays them all.')
        if not len(c):
            warnings.append('Only one level is weighted, so the survival probability stays at 1: no sound.')
    truncatable = (doubleWell.DoubleWell, linearPotential.LinearPotential, latticePotential.LatticePotential,
                   customPotential.CustomPotential, box2D.Box2D, hydrogen.Hydrogen)
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
    # the built-in Un sospiro, or a MIDI file sent as base64 in 'midi'
    if req.get('midi'):
        source = base64.b64decode(req['midi'])
        if len(source) > 5_000_000:
            raise ValueError('that MIDI file is too large (5 MB at most)')
    else:
        source = os.path.join(repoRoot, 'midi', 'unSospiro.mid')
        if not os.path.exists(source):
            raise FileNotFoundError('midi/unSospiro.mid not found')
    m = requestModel(req)
    weighting = weightingFor(req)
    seconds = min(max(float(req.get('seconds', 15)), 1), 120)
    try:
        notes = unSospiro.readNotes(source, seconds)
    except ImportError:
        raise FileNotFoundError('rendering needs the mido package: pip install mido')
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f'could not read that MIDI file ({type(error).__name__})')
    if tunnelling(m, req):
        renderer = unSospiro.renderStereo
        partials = lambda f, sr: m.tunnellingGroups(f, sr, temperatureFor(req, f), weighting, wrapping(req))
    else:
        renderer = unSospiro.render
        partials = lambda f, sr: content(m, req, f, sr, weighting)
    audio = renderer(
        notes, partials, nSeconds=seconds, attackTime=float(req.get('attack', 0.005)),
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
                elif self.path == '/api/notes':
                    body, kind = json.dumps(notes(req)).encode(), 'application/json'
                elif self.path == '/api/render':
                    body, kind = renderPiece(req), 'audio/wav'
                else:
                    self.send_error(404)
                    return
        except Exception as error:
            body = json.dumps({'error': f'{type(error).__name__}: {error}'}).encode()
            expected = isinstance(error, (KeyError, ValueError, FileNotFoundError))     # a bad request
            sys.stderr.write(f'Error in {self.path}: {type(error).__name__}: {error}\n' if expected
                             else f'Error in {self.path}:\n{traceback.format_exc()}')
            self.send_response(400 if expected else 500)
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

    # print nothing for requests that work, only errors
    def log_message(self, format, *args):
        pass

    def log_error(self, format, *args):
        if self.path != '/favicon.ico':                 # browsers ask for one; there is none
            sys.stderr.write(f'Error in {self.path}: {format % args}\n')

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    warnings.simplefilter('ignore', RuntimeWarning)     # numerical overflow in values that are then discarded
    harmonicOscillator.harmonicContent(220)     # every oscillator shares these coefficients: compute them up front
    print(f'Serving on http://localhost:{port}  (Ctrl+C to stop)')
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
