import math
import numpy as np
from scipy.special import eval_genlaguerre, gammaln
from numericalPotential import NumericalPotential, electronMass, eV, writeSample

class MorsePotential(NumericalPotential):
    # V(x) = D (1 - exp(-x/r))^2, the potential of a molecular bond: a minimum at x = 0, a
    # steep wall for x < 0 and a slow rise to the dissociation energy D for x > 0. Only
    # states below D are bound, and their levels crowd together towards D. The potential is
    # lopsided, so its states have no parity.
    #
    # Solved exactly. With lam = r sqrt(2mD) / hbar there are ceil(lam - 1/2) bound states,
    # E_n = D - (hbar^2 / 2m r^2) (lam - n - 1/2)^2, and with z = 2 lam exp(-x/r) and
    # s = lam - n - 1/2 the states are z^s exp(-z/2) L_n^(2s)(z) (generalized Laguerre). A
    # state bound by less than 1e-6 of D reaches so far that wrapping it costs thousands of
    # copies; those few are left out (counted in `unresolved`).
    weakest = 1e-6
    foldSamples = 2048

    def __init__(self, mass=electronMass, range_=5e-9, depth=1.6 * eV):
        self.range = range_               # r, the range of the potential (m)
        self.depth = depth                # dissociation energy D (J)
        self.r = range_ * 1e9             # nm
        self.D = depth / eV               # eV
        self.boundLimit = self.D
        super().__init__(mass)

    def potential(self, x):
        return self.D * (1 - np.exp(-x / self.r))**2

    def solve(self):
        self.lam = self.r * math.sqrt(self.D / self.kinetic)
        if self.lam <= 0.5:
            raise ValueError('this Morse potential has no bound states; make it deeper, wider or heavier')
        bound = min(math.ceil(self.lam - 0.5), self.maxStates)
        n = np.arange(bound)
        E = self.D - self.kinetic / self.r**2 * (self.lam - n - 0.5)**2
        keep = self.D - E >= self.weakest * self.D
        self.unresolved = int((~keep).sum())
        self.extents = {}
        self.eng = E[keep]
        self.parity = np.zeros(len(self.eng), int)
        s = self.lam - np.arange(len(self.eng)) - 0.5
        # log of the normalization sqrt(n! 2s / Gamma(2 lam - n)) / sqrt(r)
        n = np.arange(len(self.eng))
        self.logNorm = 0.5 * (gammaln(n + 1) + np.log(2 * s) - gammaln(2 * self.lam - n) - math.log(self.r))

    def state(self, n, x):
        # exact, with the sign the grid solver gives (and the other grid potentials keep): positive
        # in its last lobe on the right
        x = np.asarray(x, float)
        s = self.lam - n - 0.5
        logz = math.log(2 * self.lam) - x / self.r
        out = np.zeros(x.shape)
        low, high = self.extent(n)
        live = (x >= low) & (x <= high)               # beyond, the state is below exp(-24)
        z = np.exp(logz[live])
        value = np.exp(self.logNorm[n] + s * logz[live] - z / 2) * eval_genlaguerre(n, 2 * s, z)
        out[live] = value
        return out

    def turningPoints(self, n):
        root = math.sqrt(self.eng[n] / self.D)
        return -self.r * math.log(1 + root), -self.r * math.log(1 - root)

    def extent(self, n):
        # past each turning point until the state has fallen by about exp(-24): where the WKB
        # integral of sqrt((V - E) / kinetic) dx from the turning point reaches 24
        if n not in self.extents:
            left, right = self.turningPoints(n)
            E = self.eng[n]
            far = 24 / math.sqrt((self.D - E) / self.kinetic) + 10 * self.r     # V -> D on the right
            ends = []
            for start, direction, length in ((left, -1, 10 * self.r), (right, 1, far)):
                while True:
                    x = start + direction * np.linspace(0, length, 4001)
                    kappa = np.sqrt(np.maximum(self.potential(x) - E, 0) / self.kinetic)
                    total = np.concatenate([[0], np.cumsum((kappa[1:] + kappa[:-1]) / 2) * length / 4000])
                    if total[-1] >= 24:
                        ends.append(float(x[np.argmax(total >= 24)]))
                        break
                    length *= 2
            self.extents[n] = tuple(ends)
        return self.extents[n]

default = MorsePotential()
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    writeSample(default, './audio/morsePotential.wav')
