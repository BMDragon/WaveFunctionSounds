import math
import numpy as np
from scipy.special import eval_genlaguerre, gammaln
from numericalPotential import NumericalPotential, electronMass, eV, hbar, writeSample

bohrRadius = 0.0529177210903       # nm
rydberg = 13.605693122994           # eV

class Hydrogen(NumericalPotential):
    # A hydrogen-like atom: a particle of mass m bound to a nucleus of charge Z, with angular
    # momentum l. Its radial states u(r) = r R(r) live on r > 0 in the effective potential
    # V(r) = -Z e^2 / (4 pi eps0 r) + hbar^2 l (l + 1) / (2 m r^2), with energies
    # E_n = -Z^2 Ry (m / m_e) / n^2 for n = l + 1, l + 2, ... and analytic (Laguerre) states.
    # As in the other potentials, each state's unit is the region its energy allows, between
    # its turning points (from r = 0 when l = 0, where the state meets the nucleus), and it is
    # wrapped into one period of two units.
    maxStates = 60
    tapered = True          # on the ground state's scale: a high state reaches ~n^2 times further
    foldSamples = 2048      # harmonics above 1024 below 1e-6 (from the kink at the nucleus)

    def __init__(self, mass=electronMass, Z=1, l=0):
        self.mass = mass
        self.Z = Z
        self.l = l
        self.kinetic = hbar**2 / (2 * mass) / eV * 1e18          # hbar^2 / 2m in eV nm^2
        self.a = bohrRadius * electronMass / mass                # Bohr radius for this mass (nm)
        self.coulomb = 2 * Z * rydberg * bohrRadius               # Z e^2 / (4 pi eps0) in eV nm
        self.n = np.arange(l + 1, l + 1 + self.maxStates)
        self.eng = -Z**2 * rydberg * (mass / electronMass) / self.n**2
        self.parity = np.zeros(len(self.n), int)
        self.unresolved = 0
        self.stateCoefficients = None

    def potential(self, r):
        r = np.asarray(r, float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(r > 0, -self.coulomb / r + self.kinetic * self.l * (self.l + 1) / r**2, np.inf)

    def u(self, i, r):
        # the i-th radial state r R(r) (nm^-1/2), zero for r <= 0
        n, l = int(self.n[i]), self.l
        rho = 2 * self.Z * np.maximum(r, 0) / (n * self.a)
        logNorm = 0.5 * (3 * math.log(2 * self.Z / (n * self.a)) + gammaln(n - l) - math.log(2 * n) - gammaln(n + l + 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.exp(logNorm - rho / 2 + l * np.log(np.where(rho > 0, rho, 1))) * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
        if l > 0:
            R = np.where(rho > 0, R, 0.0)
        return np.where(r > 0, r * R, 0.0)

    def turningPoints(self, i):
        # roots of E r^2 + coulomb r - kinetic l (l + 1) = 0
        E, L = self.eng[i], self.kinetic * self.l * (self.l + 1)
        root = math.sqrt(max(self.coulomb**2 + 4 * E * L, 0))
        return (-self.coulomb + root) / (2 * E) if L else 0.0, (-self.coulomb - root) / (2 * E)

    def coefficientRow(self, i):
        # Fourier coefficients of the i-th wrapped state over one period. Sampled four times as
        # finely as the other potentials, since the states wiggle near the nucleus on a scale
        # much smaller than their size (the harmonics kept then change by under 1e-5 at 2^21
        # samples).
        samples = 4 * self.coefficientSamples
        u = np.linspace(0, 1, samples, False)
        left, right = self.turningPoints(i)
        width, centre = right - left, (left + right) / 2
        reach = right + 12 * math.sqrt(self.kinetic / -self.eng[i]) + self.a     # past the tail
        images = np.arange(math.floor((centre - reach) / width - 1/2) - 1, math.ceil(centre / width + 3/2) + 2)
        flip = (self.n[i] - self.l - 1) % 2 == 0          # an even number of nodes
        wrapped = np.zeros(samples)
        for image in images:
            r = centre + width * (2*u - 1/2) - width * image
            inside = (r > 0) & (r < reach)
            if inside.any():
                wrapped[inside] += (-1.0)**abs(image) * self.u(i, r[inside]) if flip else self.u(i, r[inside])
        return (2 * np.fft.rfft(wrapped * math.sqrt(width)) / samples)[:self.coefficientSamples // 2 + 1]

    def state(self, i, r):
        return self.u(i, r)

    def extent(self, i):
        # from the nucleus to where the tail has fallen by about exp(-24)
        return 0.0, self.turningPoints(i)[1] + 24 * math.sqrt(self.kinetic / -self.eng[i]) + self.a

    # In units of the ground state's width (a / Z times a number fixed by l) the states depend
    # only on l, so on the ground state's scale their coefficients are shared per (l, n)
    sharedGround = {}

    def groundRow(self, i):
        key = (self.l, i)
        if key not in Hydrogen.sharedGround:
            Hydrogen.sharedGround[key] = super().groundRow(i)
        return Hydrogen.sharedGround[key]

    def plotData(self, count):
        # Radius (nm), effective potential (meV, clipped below the lowest level shown),
        # level energies (meV) and the radial states u(r) (nm^-1/2)
        count = min(count, len(self.eng))
        right = self.turningPoints(count - 1)[1]
        x = np.linspace(0, 1.3 * right, 601)
        V = self.potential(x) * 1e3
        floor = 1.5 * self.eng[0] * 1e3
        V = np.where(np.isfinite(V) & (V > floor), V, floor)
        return {'x': x, 'V': V, 'energies': self.eng[:count] * 1e3, 'psi': [self.u(i, x) for i in range(count)]}

default = Hydrogen()
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    writeSample(default, './audio/hydrogen.wav')
