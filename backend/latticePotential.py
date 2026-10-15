import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from numericalPotential import electronMass, eV, kB, hbar, writeSample

class LatticePotential:
    # Kronig-Penney lattice: wells of spacing a separated by barriers of height V0 and width
    # barrierFraction * a. Its states are Bloch waves, psi(x + a) = exp(i phi) psi(x), one
    # per band at the chosen Bloch phase phi. The lattice cell is the unit and two units are
    # one period of the key's pitch, as in the other potentials, so a Bloch wave puts its
    # partials at (2m + phi/pi) times the key's frequency for every integer m: phi = pi
    # gives odd harmonics (like the wrapped square wells), phi = 0 even harmonics, and any
    # phase in between gives inharmonic partials in pairs around the even harmonics.
    #
    # The potential is piecewise constant, so everything is exact: band energies are roots
    # of the Kronig-Penney equation and the Bloch waves are cos/sin in the wells and
    # cosh/sinh (or cos/sin above the barrier) in the barriers, matched at the walls.
    cellPoints = 4096
    maxStates = 60
    maxHarmonic = 400       # Fourier components kept on each side of zero

    def __init__(self, mass=electronMass, spacing=5e-9, barrier=1.6 * eV, barrierFraction=0.2, blochPhase=0.8):
        self.mass = mass                        # kg
        self.spacing = spacing                  # lattice spacing a (m)
        self.barrier = barrier                  # barrier height V0 (J)
        self.barrierFraction = barrierFraction  # barrier width as a fraction of a
        self.blochPhase = blochPhase            # phi in units of pi

        self.a = spacing * 1e9                              # nm
        self.w = (1 - barrierFraction) * self.a             # well width (nm)
        self.d = barrierFraction * self.a                   # barrier width (nm)
        self.V0 = barrier / eV                              # eV
        self.kinetic = hbar**2 / (2 * mass) / eV * 1e18     # hbar^2 / 2m in eV nm^2
        self.phi = math.pi * blochPhase

        # one cell, x in units of a from -1/2 to 1/2, the well in the middle and half a
        # barrier at each edge
        M = self.cellPoints
        self.s = -1/2 + np.arange(M) / M
        self.V = np.where(np.abs(self.s) > (1 - barrierFraction) / 2, self.V0, 0.0)

        roots = self.bandWavenumbers()
        self.eng = self.kinetic * np.array([k for k, _ in roots])**2       # eV
        psi = np.column_stack([wave for k, count in self.distinct(roots) for wave in self.blochWaves(k, count)])
        psi = psi / np.sqrt((np.abs(psi)**2).mean(axis=0))                 # mean |psi|^2 over a cell is 1

        # Fourier components b_m of the periodic part u = exp(-i phi s) psi, so that
        # psi(s) = sum_m b_m exp(i (phi + 2 pi m) s)
        u = np.exp(-1j * self.phi * self.s)[:, None] * psi
        B = np.fft.fft(u, axis=0) / M
        self.m = np.arange(-self.maxHarmonic, self.maxHarmonic + 1)
        b = ((-1.0)**np.abs(self.m))[:, None] * B[self.m % M]
        # each band's overall phase is arbitrary: make its strongest component real and positive
        strongest = b[np.abs(b).argmax(axis=0), np.arange(b.shape[1])]
        gauge = np.conj(strongest) / np.abs(strongest)
        self.b = (b * gauge).T                          # (band, m)
        self.psi = psi * gauge

    def kronigPenney(self, k):
        # D(E) - cos(phi) for wavenumber k = sqrt(E / kinetic) in the wells, D being half the
        # trace of the cell's transfer matrix. Below the barrier it is divided by
        # cosh(kappa d), which keeps it finite for thick barriers without moving its roots.
        k = np.asarray(k, float)
        w, d = self.w, self.d
        E = self.kinetic * k**2
        kappa = np.sqrt(np.maximum(self.V0 - E, 0) / self.kinetic)
        q = np.sqrt(np.maximum(E - self.V0, 0) / self.kinetic)
        sinkw = w * np.sinc(k * w / np.pi)                      # sin(kw) / k
        z = kappa * d
        tanhOverKappa = d * np.where(z > 1e-8, np.tanh(z) / np.where(z > 1e-8, z, 1), 1)
        sech = 2 * np.exp(-z) / (1 + np.exp(-2 * z))
        below = np.cos(k * w) + 0.5 * (kappa**2 - k**2) * tanhOverKappa * sinkw - math.cos(self.phi) * sech
        above = np.cos(k * w) * np.cos(q * d) - 0.5 * (k**2 + q**2) * d * np.sinc(q * d / np.pi) * sinkw - math.cos(self.phi)
        return np.where(E < self.V0, below, above)

    def bandWavenumbers(self):
        # (k, multiplicity) of the lowest maxStates bands at this Bloch phase. Sign changes
        # on a fine grid in k find the roots; local extrema of the function are refined too,
        # so two roots closer together than the grid (bands meeting across a tiny gap, when
        # phi is near 0 or pi) are found as well, or as one double root if they touch.
        f = lambda k: float(self.kronigPenney(k))
        step = math.pi / (200 * self.a)
        roots, start = [], 1e-9
        while len(roots) < self.maxStates:
            k = start + step * np.arange(200 * (self.maxStates + 2) + 1)
            g = self.kronigPenney(k)
            found = []
            for i in np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]:
                found.append((brentq(f, k[i], k[i + 1], xtol=1e-15 * k[i + 1], rtol=1e-14), 1))
            for i in np.nonzero((g[1:-1] - g[:-2]) * (g[2:] - g[1:-1]) < 0)[0] + 1:
                if abs(g[i]) > 0.1:
                    continue
                side = 1 if g[i] > g[i - 1] else -1          # +1 at a maximum, -1 at a minimum
                best = minimize_scalar(lambda x: -side * f(x), bounds=(k[i - 1], k[i + 1]), method='bounded',
                                       options={'xatol': 1e-14 * k[i + 1]})
                peak, value = best.x, f(best.x)
                if abs(value) < 1e-12:
                    found.append((peak, 2))
                elif np.sign(value) != np.sign(g[i]):
                    if np.sign(g[i - 1]) == np.sign(g[i]) == np.sign(g[i + 1]):
                        found.append((brentq(f, k[i - 1], peak, xtol=1e-15 * k[i + 1]), 1))
                        found.append((brentq(f, peak, k[i + 1], xtol=1e-15 * k[i + 1]), 1))
            roots += sorted(found)
            start = k[-1]
        expanded = [(k, 1) for k, count in roots for _ in range(count)]
        return expanded[:self.maxStates]

    @staticmethod
    def distinct(roots):
        # group equal wavenumbers (a double root) as (k, count)
        groups = []
        for k, _ in roots:
            if groups and abs(k - groups[-1][0]) <= 1e-12 * k:
                groups[-1][1] += 1
            else:
                groups.append([k, 1])
        return groups

    def blochWaves(self, k, count):
        # The Bloch wave(s) at wavenumber k, on the cell grid. In the well (|x| < w/2):
        # A cos(kx) + B sin(kx)/k. In the barrier centred on x = a/2 (y = x - a/2 from -d/2
        # to d/2): C u1(y) + D u2(y), with u1, u2 scaled to stay of order one. The part of the
        # cell left of the well belongs to the previous barrier, so carries exp(-i phi).
        a, w, h = self.a, self.w, self.d / 2
        E = self.kinetic * k**2
        x = self.s * a

        def well(x):
            return np.array([np.cos(k * x), x * np.sinc(k * x / np.pi)]), \
                   np.array([-k * np.sin(k * x), np.cos(k * x)])

        def barrier(y):
            y = np.asarray(y, float)
            if E < self.V0:
                kappa = math.sqrt((self.V0 - E) / self.kinetic)
                H = kappa * h
                if H < 1e-6:            # barely below the top: cosh, sinh are linear
                    return np.array([np.ones_like(y), y / h]), np.array([kappa**2 * y, np.full_like(y, 1 / h)])
                z = kappa * np.abs(y)
                grow = np.exp(z - H)
                c = grow * (1 + np.exp(-2 * z)) / (1 + np.exp(-2 * H))          # cosh(ky)/cosh(kh)
                sn = np.sign(y) * grow * (1 - np.exp(-2 * z)) / (1 - np.exp(-2 * H))   # sinh(ky)/sinh(kh)
                dc = kappa * np.sign(y) * grow * (1 - np.exp(-2 * z)) / (1 + np.exp(-2 * H))
                ds = kappa * grow * (1 + np.exp(-2 * z)) / (1 - np.exp(-2 * H))
                return np.array([c, sn]), np.array([dc, ds])
            q = math.sqrt((E - self.V0) / self.kinetic)
            return np.array([np.cos(q * y), y * np.sinc(q * y / np.pi)]), np.array([-q * np.sin(q * y), np.cos(q * y)])

        bloch = np.exp(-1j * self.phi)
        wR, dwR = well(w / 2); wL, dwL = well(-w / 2)
        bL, dbL = barrier(-h); bR, dbR = barrier(h)
        scale = 1 / max(k, math.sqrt(abs(E - self.V0) / self.kinetic), 1 / a)   # balance the derivative rows
        system = np.array([[wR[0], wR[1], -bL[0], -bL[1]],
                           [scale * dwR[0], scale * dwR[1], -scale * dbL[0], -scale * dbL[1]],
                           [wL[0], wL[1], -bloch * bR[0], -bloch * bR[1]],
                           [scale * dwL[0], scale * dwL[1], -bloch * scale * dbR[0], -bloch * scale * dbR[1]]], complex)
        null = np.linalg.svd(system)[2].conj()[-count:]

        inWell = np.abs(x) <= w / 2
        right = x > w / 2
        waves = []
        for A, B, C, D in null:
            psi = np.zeros(len(x), complex)
            basis, _ = well(x[inWell]); psi[inWell] = A * basis[0] + B * basis[1]
            basis, _ = barrier(x[right] - a / 2); psi[right] = C * basis[0] + D * basis[1]
            left = ~inWell & ~right
            basis, _ = barrier(x[left] + a / 2); psi[left] = bloch * (C * basis[0] + D * basis[1])
            waves.append(psi)
        return waves

    def energies(self, count):
        return self.eng[:count] * eV            # J

    def amplitudes(self, temperature, weighting='boltzmann'):
        # Amplitude of every band: Boltzmann sqrt(P_n) relative to the lowest band,
        # 'equal': all 1, 'inverse': 1/(n+1), or a list of amplitudes set by hand
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
        a = self.a
        M = self.cellPoints
        s = np.linspace(-1.5, 1.5, 601)
        cell = np.floor(s + 1/2)
        index = np.clip(((s - cell + 1/2) * M).astype(int), 0, M - 1)
        phase = np.exp(1j * self.phi * cell)
        return {'x': s * a, 'V': self.V[index] * 1e3, 'energies': self.eng[:count] * 1e3,
                'psi': [np.real(phase * self.psi[index, n]) / math.sqrt(a) for n in range(count)]}

default = LatticePotential()
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    writeSample(default, './audio/latticePotential.wav')
