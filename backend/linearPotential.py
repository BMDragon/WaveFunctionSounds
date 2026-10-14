import math
import numpy as np
from numericalPotential import NumericalPotential, electronMass, eV, tailLength, writeSample

class LinearPotential(NumericalPotential):
    # V(x) = V0 |x| / (L/2), a V-shaped potential reaching V0 at x = +-L/2. Its states are
    # Airy functions: their levels crowd together as E_n ~ n^(2/3), and each state's
    # wavelength stretches out towards its turning points.
    symmetric = True

    def __init__(self, mass=electronMass, width=5e-9, height=1.6 * eV):
        self.width = width                # L (m)
        self.height = height              # V0 (J)
        self.slope = (height / eV) / (width * 1e9 / 2)     # eV/nm
        super().__init__(mass)

    def potential(self, x):
        return self.slope * np.abs(x)

    def box(self, energy):
        reach = energy / self.slope + 12 * tailLength(self.kinetic, self.slope) + 0.5
        return -reach, reach

    def initialEnergy(self):
        # the k-th zero of Ai is about (3 pi (4k - 1) / 8)^(2/3), in units of (kinetic slope^2)^(1/3)
        return (self.kinetic * self.slope**2)**(1/3) * (3 * math.pi * self.maxStates / 4)**(2/3) * 1.1

default = LinearPotential()
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    writeSample(default, './audio/linearPotential.wav')
