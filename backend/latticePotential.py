import math
import numpy as np
from scipy.linalg import eigh
from numericalPotential import electronMass, eV, kB, hbar, writeSample

class LatticePotential:
    # Kronig-Penney lattice: wells of spacing a separated by barriers of height V0 and width
    # barrierFraction * a. Its states are Bloch waves, psi(x + a) = exp(i phi) psi(x), one
    # per band at the chosen Bloch phase phi. The lattice cell is the unit and two units are
    # one period of the key's pitch, as in the other potentials, so a Bloch wave puts its
    # partials at (2m + phi/pi) times the key's frequency for every integer m: phi = pi
    # gives odd harmonics (like the wrapped square wells), phi = 0 even harmonics, and any
    # phase in between gives inharmonic partials in pairs around the even harmonics.
    cellPoints = 1024
    maxStates = 60
    maxHarmonic = 400       # Fourier components kept on each side of zero

    def __init__(self, mass=electronMass, spacing=5e-9, barrier=1.6 * eV, barrierFraction=0.2, blochPhase=0.8):
        self.mass = mass                        # kg
        self.spacing = spacing                  # lattice spacing a (m)
        self.barrier = barrier                  # barrier height V0 (J)
        self.barrierFraction = barrierFraction  # barrier width as a fraction of a
        self.blochPhase = blochPhase            # phi in units of pi

        M = self.cellPoints
        a = spacing * 1e9
        kinetic = hbar**2 / (2 * mass) / eV * 1e18      # hbar^2 / 2m in eV nm^2
        phi = math.pi * blochPhase
        # one cell, x in units of a from -1/2 to 1/2, the well in the middle and half a
        # barrier at each edge
        self.s = -1/2 + np.arange(M) / M
        self.V = np.where(np.abs(self.s) > (1 - barrierFraction) / 2, barrier / eV, 0.0)

        # finite-difference Hamiltonian with the Bloch condition psi_M = exp(i phi) psi_0
        t = kinetic / (a / M)**2
        H = np.diag(self.V + 2*t).astype(complex)
        i = np.arange(M - 1)
        H[i, i + 1] = H[i + 1, i] = -t
        H[M - 1, 0] = -t * np.exp(1j * phi)
        H[0, M - 1] = -t * np.exp(-1j * phi)
        E, psi = eigh(H, subset_by_index=[0, self.maxStates - 1])
        self.eng = E                                    # eV
        psi = psi * math.sqrt(M)                        # mean |psi|^2 over a cell is 1

        # Fourier components b_m of the periodic part u = exp(-i phi s) psi, so that
        # psi(s) = sum_m b_m exp(i (phi + 2 pi m) s)
        u = np.exp(-1j * phi * self.s)[:, None] * psi
        B = np.fft.fft(u, axis=0) / M
        self.m = np.arange(-self.maxHarmonic, self.maxHarmonic + 1)
        b = ((-1.0)**np.abs(self.m))[:, None] * B[self.m % M]
        # each band's overall phase is arbitrary: make its strongest component real and positive
        strongest = b[np.abs(b).argmax(axis=0), np.arange(b.shape[1])]
        gauge = np.conj(strongest) / np.abs(strongest)
        self.b = (b * gauge).T                          # (band, m)
        self.psi = psi * gauge

    def energies(self, count):
        return self.eng[:count] * eV            # J

    def amplitudes(self, temperature, weighting='boltzmann'):
        # Amplitude of every band: Boltzmann sqrt(P_n) relative to the lowest band,
        # 'equal': all 1, 'inverse': 1/(n+1)
        if weighting == 'equal':
            return np.ones(len(self.eng))
        if weighting == 'inverse':
            return 1 / np.arange(1, len(self.eng) + 1)
        return np.exp(-(self.eng - self.eng[0]) * eV / (2 * kB * temperature))

    def harmonicContent(self, freq, sampleRate=44100, temperature=None, weighting='boltzmann'):
        # Partial frequencies as multiples r of freq, with complex amplitudes c, so the
        # waveform is Re(sum c exp(2 pi i r freq t)). For phi = 0 or pi the partials are
        # harmonics and are returned in the same form as the other potentials (r = 1, 2, ...).
        # Otherwise r = 2m + phi/pi, negative ones included, all sharing the offset phi/pi.
        c = self.amplitudes(temperature or freq, weighting) @ self.b
        ratios = 2 * self.m + self.blochPhase
        keep = np.abs(ratios) < sampleRate / (2 * freq)
        ratios, c = ratios[keep], c[keep]
        if abs(self.blochPhase - round(self.blochPhase)) > 1e-9:
            return ratios, c
        # Re(c e^{-i theta}) = Re(conj(c) e^{i theta}): fold negative partials onto positive ones
        r = np.round(ratios).astype(int)
        folded = np.zeros(np.abs(r).max() + 1, complex)
        np.add.at(folded, r[r > 0], c[r > 0])
        np.add.at(folded, -r[r < 0], np.conj(c[r < 0]))
        return np.arange(1, len(folded)), folded[1:]

    def plotData(self, count):
        # Position (nm) over three cells, potential (meV), band energies (meV) and the real
        # part of the first `count` Bloch waves (nm^-1/2)
        a = self.spacing * 1e9
        M = self.cellPoints
        s = np.linspace(-1.5, 1.5, 601)
        cell = np.floor(s + 1/2)
        index = np.clip(((s - cell + 1/2) * M).astype(int), 0, M - 1)
        phase = np.exp(1j * math.pi * self.blochPhase * cell)
        return {'x': s * a, 'V': self.V[index] * 1e3, 'energies': self.eng[:count] * 1e3,
                'psi': [np.real(phase * self.psi[index, n]) / math.sqrt(a) for n in range(count)]}

default = LatticePotential()
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    writeSample(default, './audio/latticePotential.wav')
