import math
import numpy as np
from scipy.special import ai_zeros, airy
from numericalPotential import NumericalPotential, electronMass, eV, writeSample

class LinearPotential(NumericalPotential):
    # V(x) = V0 |x| / (L/2), a V-shaped potential reaching V0 at x = +-L/2. Its states are
    # Airy functions: their levels crowd together as E_n ~ n^(2/3), and each state's
    # wavelength stretches out towards its turning points.
    #
    # Solved exactly. With slope s, length a = (hbar^2 / 2m s)^(1/3) and energy e = s a, each
    # side holds Ai(|x|/a - E/e). Even states need Ai' = 0 at x = 0 and odd states Ai = 0, so
    # the energies are e times minus the zeros of Ai' and of Ai, taken in turn.
    symmetric = True
    foldSamples = 2048      # on the ground state's scale nothing reaches above harmonic 1024
    ownSamples = 2048       # nor on each state's own unit (below 1e-7 there)

    def __init__(self, mass=electronMass, width=5e-9, height=1.6 * eV):
        self.width = width                # L (m)
        self.height = height              # V0 (J)
        self.slope = (height / eV) / (width * 1e9 / 2)     # eV/nm
        super().__init__(mass)

    def potential(self, x):
        return self.slope * np.abs(x)

    def solve(self):
        self.a = (self.kinetic / self.slope)**(1/3)        # nm
        self.e = self.slope * self.a                       # eV
        count = self.maxStates
        a, ap, ai, aip = ai_zeros((count + 1) // 2)
        self.eng = np.empty(count)
        self.parity = np.empty(count, int)
        self.y0 = np.empty(count)                          # -E/e, the zero each state uses
        self.norm = np.empty(count)
        self.eng[0::2], self.eng[1::2] = -ap[:(count + 1) // 2] * self.e, -a[:count // 2] * self.e
        self.parity[0::2], self.parity[1::2] = 1, -1
        self.y0[0::2], self.y0[1::2] = ap[:(count + 1) // 2], a[:count // 2]
        # integral over x > 0 of Ai(x/a + y0)^2 dx is a (Ai'(y0)^2 - y0 Ai(y0)^2); both sides count
        self.norm[0::2] = 1 / np.sqrt(2 * self.a * -ap[:(count + 1) // 2] * ai[:(count + 1) // 2]**2)
        self.norm[1::2] = 1 / np.sqrt(2 * self.a * aip[:count // 2]**2)
        # every state is positive in its last lobe on the right (the outermost lobe of Ai is
        # positive), the sign the grid solver gives
        self.unresolved = 0
        # Ai and Ai' tabulated finely once; states are read from the table by cubic Hermite
        # interpolation, which with the exact derivative is good to about 1e-10
        self.step = 0.002
        self.y = np.arange(self.y0.min() - 4 * self.step, 9 + 4 * self.step, self.step)
        self.ai, self.aip = airy(self.y)[:2]

    def airy(self, y):
        # Ai(y) for y from the table's start; zero past y = 9 (below 1e-7 of its peak)
        out = np.zeros(y.shape)
        inside = y < self.y[-2]
        t = (y[inside] - self.y[0]) / self.step
        i = np.floor(t).astype(int)
        u = t - i
        p0, p1, m0, m1 = self.ai[i], self.ai[i + 1], self.aip[i] * self.step, self.aip[i + 1] * self.step
        out[inside] = ((2*u**3 - 3*u**2 + 1) * p0 + (u**3 - 2*u**2 + u) * m0 + (-2*u**3 + 3*u**2) * p1 + (u**3 - u**2) * m1)
        return out

    def state(self, n, x):
        x = np.asarray(x, float)
        value = self.norm[n] * self.airy(np.abs(x) / self.a + self.y0[n])
        return value if self.parity[n] == 1 else np.sign(x) * value

    # In units of a (and of the ground state's width, a fixed multiple of it) every V-shaped
    # potential has the same states, so their coefficients (on their own units and on the ground
    # state's) are worked out once and shared
    sharedGround, sharedOwn = {}, {}

    def coefficientRow(self, n):
        if n not in LinearPotential.sharedOwn:
            LinearPotential.sharedOwn[n] = super().coefficientRow(n)
        return LinearPotential.sharedOwn[n]

    def groundRow(self, n):
        if n not in LinearPotential.sharedGround:
            LinearPotential.sharedGround[n] = super().groundRow(n)
        return LinearPotential.sharedGround[n]

    def turningPoints(self, n):
        edge = self.eng[n] / self.slope
        return -edge, edge

    def extent(self, n):
        # Ai(y) falls below exp(-18) for y > 9
        edge = self.eng[n] / self.slope + 9 * self.a
        return -edge, edge

default = LinearPotential()
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    writeSample(default, './audio/linearPotential.wav')
