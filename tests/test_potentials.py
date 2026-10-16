import itertools
import math
import multiprocessing
import os
import sys
import time
import traceback

# one thread per worker process: the sweep runs many models in parallel
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import numpy as np
from scipy.optimize import brentq
from scipy.special import ai_zeros

# Robustness tests for every potential across the web app's slider ranges.
# Usage: python3 tests/test_potentials.py [--quick]   (TEST_WORKERS=n to limit the worker processes)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))
import infiniteSquareWell, finiteSquareWell, harmonicOscillator
import doubleWell, morsePotential, latticePotential, linearPotential
import numericalPotential as npot

ME, EV = npot.electronMass, npot.eV
QUICK = '--quick' in sys.argv
MASSES = (0.05, 1.0, 5.0) if QUICK else (0.05, 0.3, 1.0, 5.0)
LENGTHS = (1.0, 5.0, 20.0) if QUICK else (1.0, 2.5, 5.0, 10.0, 20.0)
DEPTHS = (0.05, 1.6, 5.0) if QUICK else (0.05, 0.3, 1.6, 5.0)
FREQS = (27.5, 130.81, 440.0, 1046.5, 4186.0)
WEIGHTINGS = (('boltzmann', None), ('boltzmann', 1.0), ('boltzmann', 20000.0), ('equal', None), ('inverse', None))

def build(potential, mass, length, depth, fraction=0.2, phase=0.8):
    m, L, V = mass * ME, length * 1e-9, depth * EV
    return {'infinite': lambda: infiniteSquareWell.InfiniteSquareWell(m, L),
            'finite': lambda: finiteSquareWell.FiniteSquareWell(m, L, V),
            'oscillator': lambda: harmonicOscillator.HarmonicOscillator(m, L),
            'doublewell': lambda: doubleWell.DoubleWell(m, L, V),
            'morse': lambda: morsePotential.MorsePotential(m, L, V),
            'linear': lambda: linearPotential.LinearPotential(m, L, V),
            'lattice': lambda: latticePotential.LatticePotential(m, L, V, fraction, phase)}[potential]()

def levelCount(model):
    if isinstance(model, infiniteSquareWell.InfiniteSquareWell):
        return 30
    if isinstance(model, harmonicOscillator.HarmonicOscillator):
        return harmonicOscillator.numStates
    if isinstance(model, finiteSquareWell.FiniteSquareWell):
        return len(model.vn)
    return len(model.eng)

def checkSound(model, problems):
    # harmonic content at extreme pitches, weightings and temperatures
    lattice = isinstance(model, latticePotential.LatticePotential)
    for f, (weighting, T) in itertools.product(FREQS, WEIGHTINGS):
        r, c = model.harmonicContent(f, 44100, T, weighting)
        tag = f'f={f} {weighting} T={T}'
        if len(r) == 0 or not np.all(np.isfinite(c)) or not np.any(np.abs(c) > 0):
            problems.append(f'{tag}: empty, zero or non-finite harmonic content'); continue
        if np.any(np.abs(r) * f >= 22050):
            problems.append(f'{tag}: partial at or above Nyquist')
        integer = np.allclose(r, np.round(r))
        if integer and np.any(np.asarray(r) < 1):
            problems.append(f'{tag}: harmonic number below 1')
        if not integer:
            offsets = np.asarray(r) - np.floor(r)
            if not lattice or np.ptp(offsets) > 1e-9:
                problems.append(f'{tag}: inharmonic partials without a single common offset')

def checkPlot(model, problems):
    p = model.plotData(8)
    n = len(p['x'])
    if not (len(p['psi']) >= 1 and all(len(s) == n for s in p['psi']) and len(p['V']) == n):
        problems.append('plotData: inconsistent lengths')
    values = [v for v in p['V'] if v is not None]
    if not (np.all(np.isfinite(p['x'])) and np.all(np.isfinite(values)) and np.all(np.isfinite(p['energies']))
            and all(np.all(np.isfinite(s)) for s in p['psi'])):
        problems.append('plotData: non-finite values')

def nodes(psi):
    # sign changes, ignoring the tiny tails
    big = psi[np.abs(psi) > 1e-3 * np.abs(psi).max()]
    return int(np.sum(np.diff(np.sign(big)) != 0))

def checkNumerical(model, problems):
    dx = model.x[1] - model.x[0]
    count = min(len(model.eng), 20)
    psi = model.psi[:, :count]
    overlap = psi.T @ psi * dx
    if np.abs(np.diag(overlap) - 1).max() > 1e-6:
        problems.append(f'normalization off by {np.abs(np.diag(overlap) - 1).max():.1e}')
    if np.abs(overlap - np.diag(np.diag(overlap))).max() > 1e-6:
        problems.append(f'orthogonality off by {np.abs(overlap - np.diag(np.diag(overlap))).max():.1e}')
    bad = [n for n in range(count) if nodes(model.psi[:, n]) != n]
    if bad:
        problems.append(f'node count wrong for states {bad[:5]}')
    if model.symmetric:
        for n in range(count):
            mirrored = model.psi[::-1, n] * model.parity[n]
            if np.abs(mirrored - model.psi[:, n]).max() > 1e-6 * np.abs(model.psi[:, n]).max():
                problems.append(f'state {n} does not have parity {model.parity[n]}'); break
    if np.any(model.eng >= model.boundLimit):
        problems.append('state above the bound limit kept')
    # every kept state must have decayed before the edges of the box
    edge = np.abs(model.psi[[0, -1], :]).max(axis=0) / np.abs(model.psi).max(axis=0)
    if edge.max() > 1e-4:
        problems.append(f'a state reaches the box edge (relative amplitude {edge.max():.1e}): box too small')

def checkExactStates(model, problems):
    # states from formulas (Morse, V-shaped): normalized, orthogonal, n nodes, negligible at
    # the edge of their extent, and solving the Schrodinger equation
    count = len(model.eng)
    picks = sorted({0, 1, 2, count // 2, count - 2, count - 1} & set(range(count)))
    lo, hi = min(model.extent(n)[0] for n in picks), max(model.extent(n)[1] for n in picks)
    x = np.linspace(lo, hi, 200001)
    dx = x[1] - x[0]
    psi = np.array([model.state(n, x) for n in picks])
    overlap = psi @ psi.T * dx
    if np.abs(np.diag(overlap) - 1).max() > 1e-6:
        problems.append(f'normalization off by {np.abs(np.diag(overlap) - 1).max():.1e}')
    if np.abs(overlap - np.diag(np.diag(overlap))).max() > 1e-6:
        problems.append(f'orthogonality off by {np.abs(overlap - np.diag(np.diag(overlap))).max():.1e}')
    for row, n in zip(psi, picks):
        if nodes(row) != n:
            problems.append(f'state {n} has {nodes(row)} nodes'); break
        left, right = model.extent(n)
        if np.abs(model.state(n, np.array([left + 1e-9, right]))).max() > 1e-6 * np.abs(row).max():
            problems.append(f'state {n} not negligible at the edge of its extent'); break
        allowed = (x > model.turningPoints(n)[0]) & (x < model.turningPoints(n)[1]) & (np.abs(x) > 50 * dx)
        lap = np.gradient(np.gradient(row, dx), dx)
        residual = (-model.kinetic * lap + (model.potential(x) - model.eng[n]) * row)[allowed][5:-5]
        if np.abs(residual).max() > 1e-3 * np.abs(model.eng[n] * row).max():
            problems.append(f'state {n} does not solve the Schrodinger equation'); break

def checkExact(potential, model, problems):
    if potential == 'morse':
        lam = model.r * math.sqrt(model.D / model.kinetic)
        n = np.arange(len(model.eng))
        exact = model.D * (1 - (1 - (n + 0.5) / lam)**2)
        spacing = np.diff(np.append(exact, model.D))
        # the most weakly bound states feel the finite box; compare the rest to their spacing
        good = n < max(len(n) - 3, 1)
        err = (np.abs(model.eng - exact) / spacing)[good].max()
        if err > 0.02:
            problems.append(f'Morse levels off by {err:.1%} of their spacing')
        if len(model.eng) < model.maxStates and len(model.eng) + model.unresolved != math.ceil(lam - 0.5):
            problems.append(f'{len(model.eng)} bound states (+{model.unresolved} unresolved), '
                            f'Morse formula gives {math.ceil(lam - 0.5)}')
        if model.unresolved:
            # only states extremely close to dissociation may be left out
            first = len(model.eng)
            gap = model.D * (1 - (first + 0.5) / lam)**2
            if gap > 1e-4 * model.D:
                problems.append(f'state {first} left out although it is bound by {gap / model.D:.1e} of D')
    if potential == 'linear':
        scale = (model.kinetic * model.slope**2)**(1/3)
        a, ap, _, _ = ai_zeros(model.maxStates)
        exact = np.sort(np.concatenate([-ap, -a]))[:len(model.eng)] * scale
        err = np.abs(model.eng / exact - 1).max()
        if err > 2e-3:
            problems.append(f'V-shaped levels off Airy zeros by {err:.1e}')
    if potential == 'finite':
        # every root solves its own equation, one per interval of pi/2 below u0, and the
        # wavefunctions are normalized and continuous with a continuous slope at the walls
        u0, vn = model.u0, model.vn
        if len(vn) != math.ceil(2 * u0 / math.pi):
            problems.append(f'{len(vn)} bound states, expected {math.ceil(2 * u0 / math.pi)}')
        for j, v in enumerate(vn):
            f = (math.sqrt(u0*u0 - v*v) * math.cos(v) - v * math.sin(v)) if j % 2 == 0 else \
                (math.sqrt(u0*u0 - v*v) * math.sin(v) + v * math.cos(v))
            if abs(f) > 1e-9 * u0 or not (j * math.pi / 2 <= v <= (j + 1) * math.pi / 2):
                problems.append(f'state {j}: v = {v:.6f} does not solve its equation in its interval'); break
        reach = 0.5 + 30 / model.alpha.min()                  # past every tail
        x = np.linspace(-reach, reach, 400001); dx = x[1] - x[0]
        for n in range(min(len(vn), 30)):
            psi = model.wavefunction(x, n)
            if abs((psi**2).sum() * dx - 1) > 1e-3:
                problems.append(f'state {n} normalization {(psi**2).sum() * dx:.4f}'); break
            for wall in (-0.5, 0.5):
                e = 1e-7
                inner, outer = model.wavefunction(np.array([wall * (1 - e)]), n)[0], model.wavefunction(np.array([wall * (1 + e)]), n)[0]
                slopeIn = (model.wavefunction(np.array([wall * (1 - e)]), n)[0] - model.wavefunction(np.array([wall * (1 - 2*e)]), n)[0])
                slopeOut = (model.wavefunction(np.array([wall * (1 + 2*e)]), n)[0] - model.wavefunction(np.array([wall * (1 + e)]), n)[0])
                scale = np.abs(psi).max()
                if abs(inner - outer) > 1e-4 * scale or abs(slopeIn - slopeOut) > 1e-3 * (abs(slopeIn) + abs(slopeOut) + 1e-12 * scale):
                    problems.append(f'state {n} not continuous at the wall x = {wall} L'); break
    if potential == 'lattice':
        checkLattice(model, problems)

def checkLattice(model, problems):
    # exact band energies: the Kronig-Penney equation holds at every band
    residual = np.abs(model.kronigPenney(np.sqrt(model.eng / model.kinetic))).max()
    if residual > 1e-8:
        problems.append(f'Kronig-Penney residual {residual:.1e}')
    if len(model.eng) != model.maxStates or np.any(np.diff(model.eng) < 0):
        problems.append(f'{len(model.eng)} bands found, or not ascending')
    # Bloch condition across the cell edge and the Schrodinger equation inside the well
    s = model.s * model.a; ds = s[1] - s[0]
    for n in (0, 1, 5, 20, 59):
        psi = model.psi[:, n]
        beyond = 2 * psi[-1] - psi[-2]                      # extrapolated to x = a/2
        jump = abs(beyond - np.exp(1j * model.phi) * psi[0]) / np.abs(psi).max()
        if jump > 0.05:
            problems.append(f'band {n}: Bloch condition off by {jump:.1e} across the cell edge'); break
        inside = np.abs(s[1:-1]) < model.w / 2 - 3 * ds
        if inside.sum() > 10 and model.eng[n] > 0:
            lap = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / ds**2
            k2 = model.eng[n] / model.kinetic
            resid = np.abs(lap[inside] + k2 * psi[1:-1][inside]).max() / (k2 * np.abs(psi).max())
            if resid > 1e-2 + (k2 * ds**2):
                problems.append(f'band {n}: Schrodinger equation off by {resid:.1e} in the well'); break
    # no missing bands: count them below the widest gap of the lowest 40 against a
    # finite-difference solve of the same cell
    from scipy.linalg import eigvalsh
    M = 1024
    a = model.a
    t = model.kinetic / (a / M)**2
    V = np.where(np.abs(-1/2 + np.arange(M) / M) > (1 - model.barrierFraction) / 2, model.V0, 0.0)
    H = np.diag(V + 2*t).astype(complex)
    i = np.arange(M - 1)
    H[i, i + 1] = H[i + 1, i] = -t
    H[M - 1, 0] = -t * np.exp(1j * model.phi); H[0, M - 1] = -t * np.exp(-1j * model.phi)
    reference = eigvalsh(H, subset_by_index=[0, 44])
    E = model.eng[:41]
    gap = int(np.argmax(np.diff(E) / E[1:]))
    cut = (E[gap] + E[gap + 1]) / 2
    if (reference < cut).sum() != gap + 1:
        problems.append(f'{gap + 1} bands below {cut:.3g} eV, finite-difference reference finds {(reference < cut).sum()}')

def checkConvergence(potential, model, args, problems):
    # solve again with twice the resolution and a bigger box: energies and sound must agree
    cls = type(model)
    fine = cls.__new__(cls)
    fine.__dict__.update({k: v for k, v in model.__dict__.items() if k not in ('x', 'V', 'eng', 'psi', 'parity', 'stateCoefficients')})
    fine.pointsPerWavelength = model.pointsPerWavelength * 2
    fine.maxPoints = model.maxPoints * 2
    box = model.box
    fine.box = lambda energy: tuple(1.4 * b for b in box(energy))
    fine.solve()
    fine.stateCoefficients = None
    count = min(len(model.eng), len(fine.eng), 30)
    spacing = np.diff(model.eng[:count + 1]).mean() if count > 1 else abs(model.eng[0]) + 1e-12
    err = np.abs(model.eng[:count] - fine.eng[:count]).max() / spacing
    if err > 0.02:
        problems.append(f'energies change by {err:.1%} of the level spacing on refinement')
    for f in (130.81, 1046.5):
        r1, c1 = model.harmonicContent(f)
        r2, c2 = fine.harmonicContent(f)
        n = min(len(c1), len(c2), 40)
        a1, a2 = np.abs(c1[:n]), np.abs(c2[:n])
        diff = np.abs(a1 - a2).max() / a2.max()
        if diff > 0.03:
            problems.append(f'f={f}: harmonic amplitudes change by {diff:.1%} on refinement')

def run(case):
    potential, mass, length, depth, fraction, phase = case
    problems, started = [], time.time()
    try:
        model = build(potential, mass, length, depth, fraction, phase)
    except ValueError as error:
        if potential == 'morse' and 'no bound states' in str(error):
            lam = length * math.sqrt(depth / (npot.hbar**2 / (2 * mass * ME) / EV * 1e18))
            return case, (['reported no bound states but the Morse formula gives some'] if lam > 0.5 else []), 0.0, 0.0, 'none bound'
        return case, [f'construction failed: {error}'], 0.0, 0.0, ''
    except Exception:
        return case, ['construction crashed: ' + traceback.format_exc().splitlines()[-1]], 0.0, 0.0, ''
    buildTime = time.time() - started
    try:
        t = time.time()
        model.harmonicContent(440.0)            # first call builds the coefficients
        coefficientTime = time.time() - t
        energies = model.energies(levelCount(model))
        if not (np.all(np.isfinite(energies)) and np.all(np.diff(energies) >= -1e-12 * np.abs(energies).max())):
            problems.append('energies not finite and ascending')
        checkSound(model, problems)
        checkPlot(model, problems)
        if isinstance(model, npot.NumericalPotential) and hasattr(model, 'psi'):     # solved on a grid
            checkNumerical(model, problems)
            checkConvergence(potential, model, case, problems)
        elif isinstance(model, npot.NumericalPotential):                             # exact states
            checkExactStates(model, problems)
        checkExact(potential, model, problems)
    except Exception:
        problems.append('check crashed: ' + traceback.format_exc().splitlines()[-1])
        coefficientTime = float('nan')
    return case, problems, buildTime, coefficientTime, f'{levelCount(model)} levels'

def cases():
    for potential in ('infinite', 'finite', 'oscillator', 'doublewell', 'morse', 'linear'):
        depths = DEPTHS if potential not in ('infinite', 'oscillator') else (1.6,)
        for mass, length, depth in itertools.product(MASSES, LENGTHS, depths):
            yield potential, mass, length, depth, 0.2, 0.8
    fractions, phases = ((0.05, 0.9), (0.0, 0.8, 1.0)) if QUICK else ((0.05, 0.2, 0.9), (0.0, 0.35, 0.8, 1.0))
    for mass, length, depth, fraction, phase in itertools.product(MASSES, LENGTHS, DEPTHS, fractions, phases):
        yield 'lattice', mass, length, depth, fraction, phase

def checkEndToEnd():
    # the web server's model and render handlers, and the Un sospiro renderer, for every
    # potential at its defaults and at slider extremes
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web'))
    import json
    import server, unSospiro
    problems = []
    freqs = [440 * 2**((n - 69) / 12) for n in range(48, 85)]
    base = dict(mass=1, length=5, depth=1.6, temperature=220, focusFreq=220, plotStates=8, freqs=freqs,
                weighting='boltzmann', temperatureMode='pitch', barrierFraction=0.2, blochPhase=0.8)
    notes = unSospiro.readNotes(os.path.join(server.repoRoot, 'midi', 'unSospiro.mid'), 2)
    for potential in ('infinite', 'finite', 'oscillator', 'doublewell', 'morse', 'lattice', 'linear', 'circle', 'rectangle',
                      'hydrogen'):
        for extra in ({}, dict(mass=5, length=20, depth=5), dict(mass=0.3, length=1, depth=0.3, temperatureMode='fixed',
                                                                   temperature=20000), dict(weighting='equal', blochPhase=1.0)):
            req = base | extra | dict(potential=potential)
            try:
                text = json.dumps(server.describe(req), allow_nan=False)
                if len(json.loads(text)['notes']) != len(freqs):
                    problems.append(f'{potential} {extra}: wrong number of notes')
            except Exception as error:
                problems.append(f'{potential} {extra}: describe failed: {type(error).__name__}: {error}')
        model = server.requestModel(base | dict(potential=potential))
        audio = unSospiro.render(notes, lambda f, sr: model.harmonicContent(f, sr), nSeconds=2)
        if not np.all(np.isfinite(audio)) or np.abs(audio).max() == 0:
            problems.append(f'{potential}: render produced silence or non-finite samples')
    # wrapping on the ground state's unit: the oscillator's Hermite functions are their own
    # Fourier transforms, so state n's copies (unit W = 2 oscillator lengths, period 2W) give
    # |c_m| = (1/2) sqrt(W) |1 + sign (-1)^m| sqrt(2 pi) |psi_n(pi m / W)| exactly
    import harmonicOscillator
    C = harmonicOscillator.computeCoefficients() if harmonicOscillator.stateCoefficients is None else harmonicOscillator.stateCoefficients
    kappa = np.pi * np.arange(1, 40) / 2
    worst = 0
    for n, psi in zip(range(60), harmonicOscillator.hermiteFunctions(kappa, 60)):
        sign = -1 if n % 2 == 0 else 1
        exact = 0.5 * math.sqrt(2) * np.abs(1 + sign * (-1.0)**np.arange(1, 40)) * math.sqrt(2 * math.pi) * np.abs(psi)
        worst = max(worst, np.abs(np.abs(C[n, 1:40]) - exact).max())
    if worst > 1e-6:
        problems.append(f'oscillator states wrapped on the ground state unit differ from their Fourier transforms by {worst:.1e}')
    # 2D boxes: partials at the exact wavenumber ratios, and the additive renderer they need
    from scipy.special import jn_zeros
    from scipy.integrate import trapezoid
    circle = server.requestModel(base | dict(potential='circle'))
    r, c = circle.harmonicContent(220, 44100, None, 'equal')
    if not np.allclose(r[:3], [1, jn_zeros(1, 1)[0] / jn_zeros(0, 1)[0], jn_zeros(2, 1)[0] / jn_zeros(0, 1)[0]]):
        problems.append(f'circle: first partials {r[:3]} are not the Bessel-zero ratios')
    if not np.allclose(np.abs(c[:2]), [1, 2]):
        problems.append('circle: the cos and sin modes of m = 1 do not add')
    square = server.requestModel(base | dict(potential='rectangle', aspect=1))
    r, c = square.harmonicContent(220, 44100, None, 'equal')
    if not np.allclose(r[:3], [1, math.sqrt(5 / 2), 2]):
        problems.append(f'square: first partials {r[:3]} are not sqrt((a^2 + b^2) / 2)')
    t = np.arange(20000) / 44100
    direct = sum((ci * np.exp(2j * np.pi * ri * 110 * t)).real for ri, ci in zip(r, c))
    if np.abs(unSospiro.partialWave(r, c, 110, t) - direct).max() > 1e-6 * np.abs(direct).max():
        problems.append('additive renderer disagrees with a direct sum of the partials')
    # hydrogen: E = -Z^2 Ry m / n^2, normalized states that solve the radial equation
    for Z, l in ((1, 0), (3, 4), (10, 20)):
        atom = server.requestModel(base | dict(potential='hydrogen', Z=Z, l=l))
        if not np.allclose(atom.eng[:3], -Z**2 * 13.605693122994 / np.arange(l + 1, l + 4)**2):
            problems.append(f'hydrogen Z={Z} l={l}: energies {atom.eng[:3]}')
        for i in (0, 20, 59):
            r = np.linspace(0, 4 * atom.turningPoints(i)[1] + 20 * atom.a, 200001)
            if abs(trapezoid(atom.u(i, r)**2, r) - 1) > 1e-6:
                problems.append(f'hydrogen Z={Z} l={l}: state {i} is not normalized')
        i = 2
        r = np.linspace(0.5 * atom.a / Z, 2 * atom.turningPoints(i)[1], 4000)
        u = atom.u(i, r)
        residual = -atom.kinetic * np.gradient(np.gradient(u, r), r) + (atom.potential(r) - atom.eng[i]) * u
        if np.abs(residual[5:-5]).max() > 0.02 * np.abs(atom.eng[i] * u).max():
            problems.append(f'hydrogen Z={Z} l={l}: state {i} does not solve the radial equation')
    # time evolution: the survival probability's partials are the energy differences over E_1 - E_0
    import timeEvolution
    r, c = timeEvolution.survivalContent([1, 4, 9], [1, 1, 1], 220)
    if not (np.allclose(r, [1, 5 / 3, 8 / 3]) and np.allclose(c, 2 / 9)):
        problems.append(f'survival probability of three equal levels: partials {r} {c}')
    if len(timeEvolution.survivalContent([1, 4], [1, 0], 220)[0]):
        problems.append('survival probability of a single state is not silent')
    for potential in ('infinite', 'oscillator', 'doublewell', 'lattice', 'circle'):
        for weighting in ('boltzmann', 'equal'):
            req = base | dict(potential=potential, sound='survival', weighting=weighting, freqs=freqs[::12])
            try:
                d = json.loads(json.dumps(server.describe(req), allow_nan=False))
                if any(len(n['m']) > server.liveSurvivalPartials for n in d['notes']):
                    problems.append(f'{potential} survival {weighting}: more partials than are played live')
            except Exception as error:
                problems.append(f'{potential} survival {weighting}: describe failed: {type(error).__name__}: {error}')
    oscillator = server.requestModel(base | dict(potential='oscillator'))
    r, c = server.content(oscillator, base | dict(sound='survival'), 220, 44100, 'boltzmann')
    if not np.allclose(r, np.round(r)):
        problems.append('oscillator survival probability is not harmonic')
    audio = unSospiro.render(notes, lambda f, sr: server.content(oscillator, base | dict(sound='survival'), f, sr, 'equal'), nSeconds=2)
    if not np.all(np.isfinite(audio)) or np.abs(audio).max() == 0:
        problems.append('survival probability render failed')
    # tunnelling in the double well: the lowest pair starts in the left well, crosses at the
    # splitting's rate (in K, played as Hz), and the stereo render has two channels
    model = server.requestModel(base | dict(potential='doublewell', depth=0.2))
    left = model.x < 0
    pair = model.psi[:, 0] + np.sign(np.sum(model.psi[left, 0] * model.psi[left, 1])) * model.psi[:, 1]
    if np.sum(pair[left]**2) / np.sum(pair**2) < 0.999:
        problems.append('doublewell: the lowest pair is not localized in the left well')
    groups = model.tunnellingGroups(220)
    splitting = (model.eng[1] - model.eng[0]) * server.eV / server.kB
    if not groups or groups[0][2] is None or abs(groups[0][2] - splitting) > 1e-9 * splitting:
        problems.append('doublewell: the lowest tunnelling group does not cross at the splitting')
    try:
        req = base | dict(potential='doublewell', depth=0.2, tunnelling=True, freqs=freqs[:3])
        if any('groups' not in n for n in json.loads(json.dumps(server.describe(req), allow_nan=False))['notes']):
            problems.append('doublewell: describe with tunnelling has no groups')
    except Exception as error:
        problems.append(f'doublewell tunnelling: describe failed: {type(error).__name__}: {error}')
    stereo = unSospiro.renderStereo(notes, model.tunnellingGroups, nSeconds=2)
    if stereo.shape[1] != 2 or not np.all(np.isfinite(stereo)) or np.abs(stereo).max() == 0:
        problems.append('doublewell: the stereo tunnelling render failed')
    return problems

if __name__ == '__main__':
    endToEnd = checkEndToEnd()
    for problem in endToEnd:
        print('FAIL end-to-end: ' + problem)
    allCases = list(cases())
    started = time.time()
    workers = int(os.environ.get('TEST_WORKERS', 0)) or os.cpu_count() or 1
    with multiprocessing.Pool(min(len(allCases), workers)) as pool:
        results = pool.map(run, allCases, chunksize=1)
    failures = 0
    byPotential = {}
    for case, problems, buildTime, coefficientTime, note in results:
        byPotential.setdefault(case[0], []).append((buildTime, coefficientTime))
        if problems:
            failures += 1
            print(f'FAIL {case[0]} mass={case[1]} length={case[2]} depth={case[3]}'
                  + (f' fraction={case[4]} phase={case[5]}' if case[0] == 'lattice' else '') + f' ({note})')
            for p in problems:
                print('    ' + p)
    failures += len(endToEnd)
    print(f'\n{len(allCases)} cases + end-to-end checks, {failures} failing, {time.time() - started:.0f} s')
    for potential, times in byPotential.items():
        b = np.array([t[0] for t in times]); c = np.array([t[1] for t in times])
        slowest = max((r for r in results if r[0][0] == potential), key=lambda r: r[2] + np.nan_to_num(r[3]))
        print(f'  {potential:>10}: build median {np.median(b):.2f} s, max {b.max():.2f} s; '
              f'coefficients median {np.nanmedian(c):.2f} s, max {np.nanmax(c):.2f} s; '
              f'slowest mass={slowest[0][1]} length={slowest[0][2]} depth={slowest[0][3]}')
    sys.exit(1 if failures else 0)
