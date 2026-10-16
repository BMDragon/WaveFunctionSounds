import math
import numpy as np
from scipy.io.wavfile import write
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

hbar = 1.054571817e-34  # reduced Planck constant (J s)
kB = 1.380649e-23       # Boltzmann constant (J/K)
eV = 1.602176634e-19    # electronvolt (J)
electronMass = 9.1093837e-31

class NumericalPotential:
    # Eigenstates of any 1D potential, from a finite-difference Hamiltonian on a grid.
    # Subclasses give potential(x) in eV with x in nm, the potential's minimum, a box that
    # holds every state up to a given energy, and an initial energy guess. Positions are in
    # nm and energies in eV throughout.
    #
    # As in the other potentials, each state's unit is the width of the region its energy
    # allows (between its outermost turning points; for a square well that is L at every
    # energy). One period is two units, copies of the state are laid in neighbouring units
    # with their tails overlapping, and states with an even number of nodes flip sign from
    # one unit to the next (for a symmetric potential, exactly the symmetric states, which
    # is what decides it there). Each state is normalized over its own unit.
    coefficientSamples = 8192
    maxStates = 120
    maxPoints = 30000
    pointsPerWavelength = 25    # enough with the 4th-order energy correction below
    boundLimit = math.inf       # energies at or above this are not bound states
    minimum = 0.0
    symmetric = False           # V(-x) = V(x): solve even and odd states separately

    def __init__(self, mass):
        self.mass = mass                                    # kg
        self.kinetic = hbar**2 / (2 * mass) / eV * 1e18     # hbar^2 / 2m in eV nm^2
        self.solve()
        self.stateCoefficients = None

    def solve(self):
        # Solve on a box sized for `energy`; if the highest state kept turns out to lie
        # above that energy the box squeezed it, so enlarge the box and solve again.
        energy = self.initialEnergy()
        for attempt in range(8):
            xmin, xmax = self.box(energy)
            k = math.sqrt(max(energy - self.minimum, 1e-12) / self.kinetic)
            points = int((xmax - xmin) / (2 * math.pi / k / self.pointsPerWavelength))
            points = min(max(points, 2000), self.maxPoints)
            x, V, E, psi, parity = (self.solveSymmetric if self.symmetric else self.solveGrid)(xmin, xmax, points)
            bound = E < self.boundLimit
            E, psi, parity = E[bound], psi[:, bound], parity[bound]
            # a bound state too weakly bound for its tail to fit in the box (Morse states a
            # few microvolts below dissociation) can't be resolved: leave it out
            edge = np.abs(psi[[0, -1], :]).max(axis=0) / np.abs(psi).max(axis=0)
            resolved = np.ones(len(E), bool) if not math.isfinite(self.boundLimit) else edge < 1e-4
            self.unresolved = int((~resolved).sum())
            E, psi, parity = E[resolved], psi[:, resolved], parity[resolved]
            if len(E) == 0:
                raise ValueError('this potential has no bound states')
            if math.isfinite(self.boundLimit) or E[-1] <= energy:
                break
            energy = self.minimum + 1.3 * (E[-1] - self.minimum)
        self.x, self.V, self.eng, self.parity = x, V, E, parity
        self.psi = psi / math.sqrt(x[1] - x[0])     # normalized: integral |psi|^2 dx = 1 (nm)

    # The Hamiltonian uses the 3-point second derivative, a tridiagonal matrix whose lowest
    # states are cheap to find. Its energies are low by about E (k dx)^2 / 12, which for a
    # potential with a hundred close levels is a sizeable part of their spacing, so each
    # energy then gets the first-order correction towards the 4th-order five-point stencil:
    # <psi| (t/12) d4 |psi>, with d4 the fourth difference and t = kinetic / dx^2. That leaves
    # an energy error of order (k dx)^4.
    @staticmethod
    def correctEnergies(E, psi, t):
        padded = np.pad(psi, ((2, 2), (0, 0)))
        fourth = padded[:-4] - 4*padded[1:-3] + 6*padded[2:-2] - 4*padded[3:-1] + padded[4:]
        return E + t / 12 * np.einsum('ij,ij->j', psi, fourth) / np.einsum('ij,ij->j', psi, psi)

    def solveGrid(self, xmin, xmax, points):
        # (x, V, energies, unit-norm eigenvectors, parity) with no parity assumed
        x = np.linspace(xmin, xmax, points)
        V = self.potential(x)
        t = self.kinetic / (x[1] - x[0])**2
        E, psi = eigh_tridiagonal(V + 2*t, -t * np.ones(points - 1), select='i',
                                  select_range=(0, min(self.maxStates, points) - 1))
        E = self.correctEnergies(E, psi, t)
        order = np.argsort(E, kind='stable')
        return x, V, E[order], psi[:, order], np.zeros(len(E), int)

    def solveSymmetric(self, xmin, xmax, points):
        # Even and odd states separately on the half grid x = (j + 1/2) dx: across x = 0 an
        # even state has psi_{-1} = psi_0 and an odd one psi_{-1} = -psi_0. Nearly degenerate
        # even/odd pairs (tunnelling doublets) then never mix, however small their splitting.
        half = points // 2
        dx = xmax / half
        xh = (np.arange(half) + 1/2) * dx
        Vh = self.potential(xh)
        t = self.kinetic / dx**2
        count = min(self.maxStates, half)
        sectors = []
        for parity, centre in ((1, t), (-1, 3*t)):
            diagonal = Vh + 2*t
            diagonal[0] = Vh[0] + centre
            E, psi = eigh_tridiagonal(diagonal, -t * np.ones(half - 1), select='i', select_range=(0, count - 1))
            full = np.vstack([parity * psi[::-1], psi]) / math.sqrt(2)
            E = self.correctEnergies(E, full, t)
            sectors.append((E, full, np.full(len(E), parity)))
        E = np.concatenate([s[0] for s in sectors])
        psi = np.hstack([s[1] for s in sectors])
        parity = np.concatenate([s[2] for s in sectors])
        order = np.argsort(E, kind='stable')[:count]
        E, psi, parity = E[order], psi[:, order], parity[order]
        # a tunnelling pair can be degenerate to machine precision; its symmetric state is
        # the lower one, so the n-th state keeps n nodes
        tie = 1e-12 * np.abs(E).max()
        for i in range(len(E) - 1):
            if parity[i] == -1 and parity[i + 1] == 1 and E[i + 1] - E[i] <= tie:
                order = [i + 1, i]
                E[i:i + 2], psi[:, i:i + 2], parity[i:i + 2] = E[order], psi[:, order], parity[order]
        x = np.concatenate([-xh[::-1], xh])
        return x, self.potential(x), E, psi, parity

    def initialEnergy(self):
        # The energy of the highest state kept, from WKB state counting:
        # N(E) = 1/2 + (1/pi) * integral over V < E of sqrt((E - V) / kinetic) dx.
        # Sizing the box for it usually means a single solve.
        if math.isfinite(self.boundLimit):
            return self.boundLimit
        def count(E):
            x = np.linspace(*self.box(E), 20001)
            return 0.5 + np.sum(np.sqrt(np.maximum(E - self.potential(x), 0) / self.kinetic)) * (x[1] - x[0]) / math.pi
        low, high = self.minimum, self.minimum + 1e-3
        while count(high) < self.maxStates + 1:
            low, high = high, self.minimum + 2 * (high - self.minimum)
        for _ in range(40):
            middle = (low + high) / 2
            low, high = (middle, high) if count(middle) < self.maxStates + 1 else (low, middle)
        return self.minimum + 1.05 * (high - self.minimum)      # a little room for WKB's error

    def decayPoint(self, start, energy, direction, exponent=18):
        # Walk outward from a turning point until the WKB exponent, the integral of
        # sqrt((V - E) / kinetic) dx, reaches `exponent`: a bound state at that energy has
        # fallen by about exp(-exponent) there.
        x, total = start, 0.0
        step = 1e-3
        while total < exponent:
            kappa = math.sqrt(max(float(self.potential(np.array([x]))[0]) - energy, 0) / self.kinetic)
            dx = min(0.05 / kappa, 10 * step) if kappa > 0 else step
            x += direction * dx
            total += kappa * dx
            step = dx
        return x

    def turningPoints(self, n):
        # outermost points where V(x) = E_n, solved between the grid points that bracket them
        E = self.eng[n]
        allowed = np.nonzero(self.V <= E)[0]
        first, last = allowed[0], allowed[-1]
        f = lambda x: float(self.potential(np.array([x]))[0]) - E
        left = brentq(f, self.x[first - 1], self.x[first]) if first > 0 else self.x[0]
        right = brentq(f, self.x[last], self.x[last + 1]) if last < len(self.x) - 1 else self.x[-1]
        return left, right

    def energies(self, count):
        return self.eng[:count] * eV            # J

    def amplitudes(self, temperature, weighting='boltzmann'):
        # Amplitude of every state. 'boltzmann': occupation P_n ~ exp(-E_n/kT) played with
        # amplitude c_n = sqrt(P_n), relative to the ground state; 'equal': all 1;
        # 'inverse': 1/(n+1); or a list of amplitudes set by hand.
        if not isinstance(weighting, str):           # hand-set amplitudes, lowest level first
            amps = np.zeros(len(self.eng))
            given = np.asarray(weighting, float)[:len(self.eng)]
            amps[:len(given)] = given
            return amps
        if weighting == 'equal':
            return np.ones(len(self.eng))
        if weighting == 'inverse':
            return 1 / np.arange(1, len(self.eng) + 1)
        return np.exp(-(self.eng - self.eng[0]) * eV / (2 * kB * temperature))

    def state(self, n, x):
        # the n-th state at positions x (nm^-1/2); zero off the grid
        return np.interp(x, self.x, self.psi[:, n], left=0, right=0)

    def extent(self, n):
        # where the n-th state is not negligible (above 1e-7 of its largest value)
        psi = self.psi[:, n]
        support = np.nonzero(np.abs(psi) > 1e-7 * np.abs(psi).max())[0]
        return self.x[support[0]], self.x[support[-1]]

    def coefficientRow(self, n):
        # Fourier coefficients of the n-th wrapped state over one period
        N = self.ownSamples or self.coefficientSamples
        u = np.linspace(0, 1, N, False)
        left, right = self.turningPoints(n)
        width, centre = right - left, (left + right) / 2
        low, high = self.extent(n)
        images = np.arange(math.floor((centre - high) / width - 1/2) - 1, math.ceil((centre - low) / width + 3/2) + 2)
        X = centre + width * (2*u[None, :] - 1/2) - width * images[:, None]
        flip = self.parity[n] == 1 if self.parity[n] else n % 2 == 0     # n nodes: flip when n is even
        sign = (-1.0)**np.abs(images) if flip else np.ones(len(images))
        wrapped = math.sqrt(width) * (sign @ self.state(n, X))
        row = np.zeros(self.coefficientSamples // 2 + 1, complex)
        row[:N // 2 + 1] = (2 * np.fft.rfft(wrapped) / N)[:len(row)]
        return row

    def computeCoefficients(self, wrapping='own'):
        # every state's coefficients
        row = self.coefficientRow if wrapping == 'own' else self.groundRow
        return np.array([row(n) for n in range(len(self.eng))])

    # The other way to wrap (the "on the ground state's scale" sound): one unit is the ground
    # state's width, between its turning points, for every state, as in the square wells, and
    # one period is two units. Higher states keep their real size: their copies sit a unit apart
    # and, where a state is wider than a unit, it carries over into the neighbouring units and
    # adds there, like the finite well's tails. Each state is sampled once across its extent and
    # folded into one period, which is the same as sampling its Fourier transform at pi m / W.
    ownSamples = None       # samples per period on a state's own unit (2048 where nothing reaches above harmonic 1024)
    tapered = False         # taper the slowly varying outer region (hydrogen's far-reaching states)
    foldSamples = None      # samples per period when folding (2048 where nothing reaches above harmonic 1024)

    def unit(self):
        # (width, centre) of the ground state's classically allowed region
        left, right = self.turningPoints(0)
        return right - left, (left + right) / 2

    def taper(self, n):
        # Out where a state varies slowly, |E - V| below kinetic (pi / 8W)^2 (it neither oscillates
        # nor decays within eight units), it can't reach the harmonics, only the dropped constant;
        # it is tapered smoothly to zero over 64 units (harmonics within about 1e-5)
        W = self.unit()[0]
        limit = (math.pi / W / 8)**2 * self.kinetic
        low, high = self.extent(n)
        x = np.linspace(low, high, 20001)
        fast = np.nonzero(np.abs(self.eng[n] - self.potential(x)) > limit)[0]
        start = x[fast[-1]] if len(fast) else x[0]
        return start, start + 64 * W

    def foldExtent(self, n):
        low, high = self.extent(n)
        return (low, min(high, self.taper(n)[1])) if self.tapered else (low, high)

    def foldState(self, n, x):
        if not self.tapered:
            return self.state(n, x)
        start, end = self.taper(n)
        x = np.asarray(x, float)
        window = np.where(x <= start, 1.0, np.where(x >= end, 0.0, np.cos(np.pi / 2 * (x - start) / (end - start))**2))
        return self.state(n, x) * window

    def groundRow(self, n):
        # Fourier coefficients over one period (two units from half a unit before the unit's
        # centre) of sum over j of sign^j psi(x - j W): samples a period apart add, and the copies
        # one unit apart add with the sign; times sqrt(W)
        W, centre = self.unit()
        N = self.foldSamples or self.coefficientSamples
        h = 2 * W / N
        start = centre - W / 2
        low, high = self.foldExtent(n)
        first, last = math.floor((low - start) / h), math.ceil((high - start) / h)
        folded = np.zeros(N)
        for i0 in range(first, last + 1, 1 << 21):            # in chunks, for very wide states
            i = np.arange(i0, min(i0 + (1 << 21), last + 1))
            folded += np.bincount(i % N, self.foldState(n, start + i * h), N)
        flip = self.parity[n] == 1 if self.parity[n] else n % 2 == 0
        wrapped = math.sqrt(W) * (folded + (-1 if flip else 1) * np.roll(folded, N // 2))
        row = np.zeros(self.coefficientSamples // 2 + 1, complex)
        row[:N // 2 + 1] = (2 * np.fft.rfft(wrapped) / N)[:len(row)]
        return row

    def coefficientsFor(self, amps, wrapping='own'):
        # The coefficient matrix (each state on its own unit, or 'ground': on the ground state's)
        # with every state that `amps` weights computed; a state is only worked out the first
        # time a note gives it a weight above 1e-12 of the largest
        if self.stateCoefficients is None:
            self.stateCoefficients = np.zeros((len(self.eng), self.coefficientSamples // 2 + 1), complex)
            self.coefficientsDone = np.zeros(len(self.eng), bool)
            self.groundCoefficients = np.zeros_like(self.stateCoefficients)
            self.groundDone = np.zeros(len(self.eng), bool)
        matrix, done, row = ((self.stateCoefficients, self.coefficientsDone, self.coefficientRow) if wrapping == 'own'
                             else (self.groundCoefficients, self.groundDone, self.groundRow))
        amps = np.abs(amps)
        for n in np.nonzero((amps > 1e-12 * amps.max()) & ~done)[0]:
            matrix[n] = row(n)
            done[n] = True
        return matrix

    def harmonicContent(self, freq, sampleRate=44100, temperature=None, weighting='boltzmann', wrapping='own'):
        # Harmonic numbers m and complex amplitudes c_m of one period, so the waveform is
        # Re(sum c_m exp(2 pi i m freq t)). The temperature in kelvin equals freq in Hz
        # unless given, and only harmonics below the Nyquist frequency are kept so nothing
        # aliases. `wrapping`: each state on its own unit ('own') or on the ground state's.
        m = np.arange(1, min(int(np.ceil(sampleRate / (2*freq))), self.coefficientSamples // 2))
        amps = self.amplitudes(temperature or freq, weighting)
        return m, amps @ self.coefficientsFor(amps, wrapping)[:, m]

    def plotData(self, count):
        # Position (nm), potential (meV), level energies (meV) and the first `count`
        # wavefunctions (nm^-1/2)
        count = min(count, len(self.eng))
        left, right = self.turningPoints(count - 1)
        margin = 0.3 * (right - left)
        x = np.linspace(left - margin, right + margin, 601)
        return {'x': x, 'V': self.potential(x) * 1e3, 'energies': self.eng[:count] * 1e3,
                'psi': [self.state(n, x) for n in range(count)]}

def tailLength(kinetic, slope):
    # decay length of a state beyond a turning point where the potential rises with `slope` (eV/nm)
    return (kinetic / max(slope, 1e-12))**(1/3)

def writeSample(model, path, freq=220, nSeconds=5, sampleRate=44100):
    # Single-pitch sample of a potential: one period tiled, as for the other potentials, or
    # for inharmonic partials a direct sum of the partials.
    m, c = model.harmonicContent(freq, sampleRate)
    if np.allclose(m, np.round(m)):
        samplesPerPeriod = int(sampleRate/freq)
        m, c = model.harmonicContent(freq, samplesPerPeriod * freq)
        phase = 2 * np.pi * np.outer(np.arange(samplesPerPeriod) / samplesPerPeriod, m)
        superposition = (c.real * np.cos(phase) - c.imag * np.sin(phase)).sum(axis=1)
        write(path, samplesPerPeriod*freq, scale(np.tile(superposition, freq*nSeconds)))
    else:
        t = np.arange(nSeconds * sampleRate) / sampleRate
        wave = np.zeros(len(t))
        for ratio, ci in zip(m, c):
            phase = 2 * np.pi * ratio * freq * t
            wave += ci.real * np.cos(phase) - ci.imag * np.sin(phase)
        write(path, sampleRate, scale(wave))
