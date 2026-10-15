import math
import numpy as np
from scipy.linalg import eigh_tridiagonal
from numericalPotential import NumericalPotential, electronMass, eV, writeSample

class CustomPotential(NumericalPotential):
    # A potential drawn by hand: `shape` gives V at evenly spaced points across a box of
    # width L, as fractions of `height` (V = height * shape, interpolated linearly), with
    # infinitely high walls at the edges so every state is bound. A flat shape is the
    # infinite square well.
    def __init__(self, mass=electronMass, width=5e-9, height=1.6 * eV, shape=(0.0,) * 64):
        self.width = width                      # L (m)
        self.height = height                    # V at the top of the drawing (J)
        self.shape = np.clip(np.asarray(shape, float), 0, 1)
        self.L = width * 1e9                    # nm
        self.V0 = height / eV                   # eV
        self.minimum = self.V0 * self.shape.min()
        super().__init__(mass)

    def potential(self, x):
        s = np.linspace(-self.L / 2, self.L / 2, len(self.shape))
        return self.V0 * np.interp(x, s, self.shape)

    def box(self, energy):
        return -self.L / 2, self.L / 2

    def solveGrid(self, xmin, xmax, points):
        # interior points only, so psi = 0 falls exactly on the walls at +-L/2
        x = -self.L / 2 + self.L * np.arange(1, points + 1) / (points + 1)
        V = self.potential(x)
        t = self.kinetic / (x[1] - x[0])**2
        E, psi = eigh_tridiagonal(V + 2*t, -t * np.ones(points - 1), select='i',
                                  select_range=(0, min(self.maxStates, points) - 1))
        E = self.correctEnergies(E, psi, t)
        order = np.argsort(E, kind='stable')
        return x, V, E[order], psi[:, order], np.zeros(len(E), int)

    def turningPoints(self, n):
        # inside the walls, the outermost points where V = E_n (the walls themselves when
        # the state reaches them)
        E = self.eng[n]
        allowed = np.nonzero(self.V <= E)[0]
        first, last = allowed[0], allowed[-1]
        x = self.x
        left = -self.L / 2 if first == 0 else np.interp(E, [self.V[first], self.V[first - 1]], [x[first], x[first - 1]])
        right = self.L / 2 if last == len(x) - 1 else np.interp(E, [self.V[last], self.V[last + 1]], [x[last], x[last + 1]])
        return left, right

    def plotData(self, count):
        # the drawn potential with its infinite walls, like the infinite well's diagram
        count = min(count, len(self.eng))
        x = np.linspace(-0.58 * self.L, 0.58 * self.L, 601)
        inside = np.abs(x) < self.L / 2
        V = [float(v) * 1e3 if i else None for v, i in zip(self.potential(x), inside)]
        psi = [np.where(inside, np.interp(x, self.x, self.psi[:, n], left=0, right=0), 0) for n in range(count)]
        return {'x': x, 'V': V, 'energies': self.eng[:count] * 1e3, 'psi': psi}

def encodeShape(shape):
    # 64 heights as a short string for the shareable link: each one of 36 levels, 0-9a-z
    return ''.join(np.base_repr(int(round(v * 35)), 36).lower() for v in shape)

def decodeShape(text):
    return [int(c, 36) / 35 for c in text]

default = CustomPotential(shape=np.concatenate([np.linspace(0.6, 0, 32), np.linspace(0, 0.3, 32)]))
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    writeSample(default, './audio/customPotential.wav')
