import math
import numpy as np
from scipy.io.wavfile import write
from scipy.linalg import eigh_tridiagonal

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
    maxPoints = 60000
    pointsPerWavelength = 50
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
            points = min(max(points, 4000), self.maxPoints)
            x, V, E, psi, parity = (self.solveSymmetric if self.symmetric else self.solveGrid)(xmin, xmax, points)
            bound = E < self.boundLimit
            E, psi, parity = E[bound], psi[:, bound], parity[bound]
            if len(E) == 0:
                raise ValueError('this potential has no bound states')
            if math.isfinite(self.boundLimit) or E[-1] <= energy:
                break
            energy = self.minimum + 1.3 * (E[-1] - self.minimum)
        self.x, self.V, self.eng, self.parity = x, V, E, parity
        self.psi = psi / math.sqrt(x[1] - x[0])     # normalized: integral |psi|^2 dx = 1 (nm)

    def solveGrid(self, xmin, xmax, points):
        # (x, V, energies, unit-norm eigenvectors, parity) with no parity assumed
        x = np.linspace(xmin, xmax, points)
        V = self.potential(x)
        t = self.kinetic / (x[1] - x[0])**2
        E, psi = eigh_tridiagonal(V + 2*t, -t * np.ones(points - 1), select='i',
                                  select_range=(0, min(self.maxStates, points) - 1))
        return x, V, E, psi, np.zeros(len(E), int)

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
            sectors.append((E, full, np.full(len(E), parity)))
        E = np.concatenate([s[0] for s in sectors])
        psi = np.hstack([s[1] for s in sectors])
        parity = np.concatenate([s[2] for s in sectors])
        order = np.argsort(E, kind='stable')[:count]
        x = np.concatenate([-xh[::-1], xh])
        return x, self.potential(x), E[order], psi[:, order], parity[order]

    def turningPoints(self, n):
        allowed = np.nonzero(self.V <= self.eng[n])[0]
        return self.x[allowed[0]], self.x[allowed[-1]]

    def energies(self, count):
        return self.eng[:count] * eV            # J

    def amplitudes(self, temperature, weighting='boltzmann'):
        # Amplitude of every state. 'boltzmann': occupation P_n ~ exp(-E_n/kT) played with
        # amplitude c_n = sqrt(P_n), relative to the ground state; 'equal': all 1;
        # 'inverse': 1/(n+1).
        if weighting == 'equal':
            return np.ones(len(self.eng))
        if weighting == 'inverse':
            return 1 / np.arange(1, len(self.eng) + 1)
        return np.exp(-(self.eng - self.eng[0]) * eV / (2 * kB * temperature))

    def computeCoefficients(self):
        # Fourier coefficients of every wrapped state over one period
        u = np.linspace(0, 1, self.coefficientSamples, False)
        coefficients = []
        for n in range(len(self.eng)):
            psi = self.psi[:, n]
            left, right = self.turningPoints(n)
            width, centre = right - left, (left + right) / 2
            support = np.nonzero(np.abs(psi) > 1e-7 * np.abs(psi).max())[0]
            low, high = self.x[support[0]], self.x[support[-1]]
            images = np.arange(math.floor((centre - high) / width - 1/2) - 1, math.ceil((centre - low) / width + 3/2) + 2)
            X = centre + width * (2*u[None, :] - 1/2) - width * images[:, None]
            flip = self.parity[n] == 1 if self.parity[n] else n % 2 == 0     # n nodes: flip when n is even
            sign = (-1.0)**np.abs(images) if flip else np.ones(len(images))
            wrapped = math.sqrt(width) * (sign @ np.interp(X, self.x, psi, left=0, right=0))
            coefficients.append(2 * np.fft.rfft(wrapped) / self.coefficientSamples)
        return np.array(coefficients)

    def harmonicContent(self, freq, sampleRate=44100, temperature=None, weighting='boltzmann'):
        # Harmonic numbers m and complex amplitudes c_m of one period, so the waveform is
        # Re(sum c_m exp(2 pi i m freq t)). The temperature in kelvin equals freq in Hz
        # unless given, and only harmonics below the Nyquist frequency are kept so nothing
        # aliases.
        if self.stateCoefficients is None:
            self.stateCoefficients = self.computeCoefficients()
        m = np.arange(1, min(int(np.ceil(sampleRate / (2*freq))), self.coefficientSamples // 2))
        return m, self.amplitudes(temperature or freq, weighting) @ self.stateCoefficients[:, m]

    def plotData(self, count):
        # Position (nm), potential (meV), level energies (meV) and the first `count`
        # wavefunctions (nm^-1/2)
        count = min(count, len(self.eng))
        left, right = self.turningPoints(count - 1)
        margin = 0.3 * (right - left)
        x = np.linspace(left - margin, right + margin, 601)
        return {'x': x, 'V': self.potential(x) * 1e3, 'energies': self.eng[:count] * 1e3,
                'psi': [np.interp(x, self.x, self.psi[:, n], left=0, right=0) for n in range(count)]}

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
