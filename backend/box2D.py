import math
import numpy as np
from scipy.special import jn_zeros, jv
from numericalPotential import electronMass, eV, kB, hbar, writeSample

class Box2D:
    # A particle in a two-dimensional box with hard walls (a quantum drum). Every state is a
    # standing wave with a wavenumber k, E = hbar^2 k^2 / 2m, and, as in the 1D square well
    # (where state n is a sine on harmonic n, n = k L / pi), each state is one partial at
    # k / k_1 times the key's frequency, the ground state being the key. In 2D the ratios are
    # not whole numbers, so the partials are inharmonic. States of equal energy (the circle's
    # cos and sin modes, or a square's (a, b) and (b, a)) each count, so their partials add.
    # Subclasses give the wavenumbers of the lowest states (in 1/nm), their labels, and a
    # profile of each along a line through the box for the plot.
    maxStates = 200

    def finish(self, k, labels):
        order = np.argsort(k, kind='stable')[:self.maxStates]
        self.k = k[order]
        self.labels = [labels[i] for i in order]
        self.kinetic = hbar**2 / (2 * self.mass) / eV * 1e18     # hbar^2 / 2m in eV nm^2
        self.eng = self.kinetic * self.k**2                       # eV

    def energies(self, count):
        return self.eng[:count] * eV            # J

    def amplitudes(self, temperature, weighting='boltzmann'):
        # Amplitude of every state, as in the other potentials: 'boltzmann' sqrt(P_n) with
        # P_n ~ exp(-E_n/kT) relative to the ground state, 'equal', 'inverse' 1/(n+1), or a
        # list set by hand
        if not isinstance(weighting, str):
            amps = np.zeros(len(self.eng))
            given = np.asarray(weighting, float)[:len(self.eng)]
            amps[:len(given)] = given
            return amps
        if weighting == 'equal':
            return np.ones(len(self.eng))
        if weighting == 'inverse':
            return 1 / np.arange(1, len(self.eng) + 1)
        return np.exp(-(self.eng - self.eng[0]) * eV / (2 * kB * temperature))

    def harmonicContent(self, freq, sampleRate=44100, temperature=None, weighting='boltzmann', wrapping='own'):
        # `wrapping` makes no difference here (every state already shares one unit, or none is wrapped)
        # Partial ratios k_n / k_1 and amplitudes -i a_n (a sine, like the 1D well's states),
        # states at the same ratio merged; the temperature in K equals freq in Hz unless given,
        # and only partials below the Nyquist frequency are kept
        ratio = self.k / self.k[0]
        amps = self.amplitudes(temperature or freq, weighting)
        audible = ratio * freq < sampleRate / 2
        ratio, amps = ratio[audible], amps[audible]
        merged, index = np.unique(np.round(ratio, 9), return_inverse=True)
        c = np.zeros(len(merged), complex)
        np.add.at(c, index, -1j * amps)
        return merged, c

    def plotData(self, count):
        # Position along a line through the box (nm), potential (meV; None outside), level
        # energies (meV), and each state along the line through its largest values (nm^-1)
        count = min(count, len(self.eng))
        half = self.halfWidth
        x = np.linspace(-1.2 * half, 1.2 * half, 601)
        inside = np.abs(x) <= half
        return {'x': x, 'V': [0.0 if i else None for i in inside], 'energies': self.eng[:count] * 1e3,
                'psi': [np.where(inside, self.profile(n, x), 0) for n in range(count)]}

class CircularBox(Box2D):
    # A circular drum of diameter `diameter`: states J_m(j_{m,n} r / R) cos(m theta) and, for
    # m > 0, the same with sin(m theta), where j_{m,n} is the n-th zero of the Bessel function
    # J_m, so k = j_{m,n} / R.
    def __init__(self, mass=electronMass, diameter=5e-9):
        self.mass = mass
        self.R = diameter / 2 * 1e9             # nm
        self.halfWidth = self.R
        # Weyl's law counts about (k R)^2 / 4 states below k: every zero below twice that
        # bound's k R covers the lowest maxStates comfortably
        limit = 2 * math.sqrt(4 * self.maxStates) + 10
        zeros, labels, m = [], [], 0
        while True:
            j = jn_zeros(m, int(limit / math.pi) + 2)
            j = j[j < limit]
            if not len(j):
                break
            for n, z in enumerate(j, 1):
                for kind in (('cos', 'sin') if m else ('',)):
                    zeros.append(z)
                    labels.append((m, n, kind))
            m += 1
        self.finish(np.array(zeros) / self.R, labels)

    def profile(self, n, x):
        # along a line through the centre at an angle where cos(m theta) (or sin) is 1,
        # normalized over the disc
        m, _, _ = self.labels[n]
        k = self.k[n]
        norm = (1 if m == 0 else math.sqrt(2)) / (math.sqrt(math.pi) * self.R * abs(jv(m + 1, k * self.R)))
        return norm * jv(m, k * np.abs(x)) * np.where(x < 0, (-1.0)**m, 1.0)

class RectangularBox(Box2D):
    # A rectangular drum `width` wide and aspect * width high: states
    # sin(a pi (x / W + 1/2)) sin(b pi (y / H + 1/2)), k = pi sqrt((a / W)^2 + (b / H)^2).
    def __init__(self, mass=electronMass, width=5e-9, aspect=1.5):
        self.mass = mass
        self.W = width * 1e9                    # nm
        self.H = aspect * self.W
        self.halfWidth = self.W / 2
        top = self.maxStates + 1
        a, b = np.meshgrid(np.arange(1, top), np.arange(1, top), indexing='ij')
        k = math.pi * np.sqrt((a / self.W)**2 + (b / self.H)**2)
        self.finish(k.ravel(), list(zip(a.ravel().tolist(), b.ravel().tolist())))

    def profile(self, n, x):
        a, _ = self.labels[n]
        # along the horizontal line where the y factor is largest
        return 2 / math.sqrt(self.W * self.H) * np.sin(a * math.pi * (x / self.W + 1/2))

if __name__ == '__main__':
    writeSample(CircularBox(), './audio/circularBox.wav')
