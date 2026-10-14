import math
from numericalPotential import NumericalPotential, electronMass, eV, tailLength, writeSample

class DoubleWell(NumericalPotential):
    # V(x) = V0 ((x/b)^2 - 1)^2: two wells at x = +-b separated by a barrier of height V0 at
    # x = 0. Below the barrier the levels come in nearly degenerate pairs (the symmetric and
    # antisymmetric combinations of the two wells' states), split by tunnelling.
    symmetric = True

    def __init__(self, mass=electronMass, separation=5e-9, barrier=1.6 * eV):
        self.separation = separation      # distance between the two minima (m)
        self.barrier = barrier            # barrier height (J)
        self.b = separation / 2 * 1e9     # nm
        self.V0 = barrier / eV            # eV
        super().__init__(mass)

    def potential(self, x):
        return self.V0 * ((x / self.b)**2 - 1)**2

    def box(self, energy):
        edge = self.b * math.sqrt(1 + math.sqrt(max(energy, 0) / self.V0))
        slope = 4 * self.V0 * edge * (edge**2 / self.b**2 - 1) / self.b**2
        reach = edge + 12 * tailLength(self.kinetic, slope) + 0.5
        return -reach, reach

    def statesBelowBarrier(self):
        return int((self.eng < self.V0).sum())

default = DoubleWell()
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    writeSample(default, './audio/doubleWell.wav')
