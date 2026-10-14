import math
import numpy as np
from numericalPotential import NumericalPotential, electronMass, eV, writeSample

class MorsePotential(NumericalPotential):
    # V(x) = D (1 - exp(-x/r))^2, the potential of a molecular bond: a minimum at x = 0, a
    # steep wall for x < 0 and a slow rise to the dissociation energy D for x > 0. Only
    # states below D are bound, and their levels crowd together towards D. The potential is
    # lopsided, so its states have no parity.
    maxPoints = 10000           # long tails need a long box, not a finer one

    def __init__(self, mass=electronMass, range_=5e-9, depth=1.6 * eV):
        self.range = range_               # r, the range of the potential (m)
        self.depth = depth                # dissociation energy D (J)
        self.r = range_ * 1e9             # nm
        self.D = depth / eV               # eV
        self.boundLimit = self.D
        super().__init__(mass)
        # a state bound by microvolts can be squeezed above D by the finite box; the analytic
        # count of bound states says how many were lost that way or for not fitting
        lam = self.r * math.sqrt(self.D / self.kinetic)
        self.unresolved = max(min(math.ceil(lam - 0.5), self.maxStates) - len(self.eng), self.unresolved)

    def potential(self, x):
        return self.D * (1 - np.exp(-x / self.r))**2

    def box(self, energy):
        # Left: past the wall until the highest kept state has decayed (WKB). Right: past the
        # turning point of the highest state kept plus its exponential tail, using the
        # analytic Morse levels E_n = D [1 - (1 - (n + 1/2)/lam)^2] with lam = r sqrt(D / kinetic),
        # but no further than the grid-point budget allows at full resolution.
        lam = self.r * math.sqrt(self.D / self.kinetic)
        if lam <= 0.5:
            raise ValueError('this Morse potential has no bound states; make it deeper, wider or heavier')
        top = min(math.ceil(lam - 0.5), self.maxStates) - 1    # highest state kept
        gap = self.D * (1 - (top + 0.5) / lam)**2             # D - E_top
        tail = math.sqrt(self.kinetic / gap)
        left = self.decayPoint(-self.r * math.log(2), self.D, -1)
        right = self.r * math.log(2 * self.D / gap) + 14 * tail + 2
        spacing = 2 * math.pi / math.sqrt(self.D / self.kinetic) / self.pointsPerWavelength
        return left, min(right, left + self.maxPoints * spacing)

    def initialEnergy(self):
        return self.D

default = MorsePotential()
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    writeSample(default, './audio/morsePotential.wav')
