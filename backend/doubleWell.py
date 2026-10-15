import math
import numpy as np
from numericalPotential import NumericalPotential, electronMass, eV, kB, tailLength, writeSample

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

    def tunnellingGroups(self, freq, sampleRate=44100, temperature=None, weighting='boltzmann'):
        # The particle starts in the left well. Each pair below the barrier (even state n, odd
        # state n + 1) with amplitudes a_e, a_o is played as the combination localized on the
        # left, a_e psi_e + s a_o psi_o, with s the sign that makes the two add for x < 0. Its
        # probability of being in the left well is P_L(t) = (1 + cos(2 pi rate t)) / 2, where
        # rate = (E_o - E_e) / h; following the convention that a temperature in K plays as
        # a pitch in Hz, the splitting in K is the rate in Hz. States above the barrier, not
        # paired, are shared equally by both wells (rate None), as is any pair whose rate is
        # above the Nyquist frequency, which can't be heard as a rate.
        # Returns [(m, c, rate)] with harmonicContent's m and c for each group.
        if self.stateCoefficients is None:
            self.stateCoefficients = self.computeCoefficients()
        m = np.arange(1, min(int(np.ceil(sampleRate / (2*freq))), self.coefficientSamples // 2))
        amps = self.amplitudes(temperature or freq, weighting)
        C = self.stateCoefficients[:, m]
        left = self.x < 0
        dx = self.x[1] - self.x[0]
        groups, centred, n = [], np.zeros(len(m), complex), 0
        while n < len(self.eng):
            if (n + 1 < len(self.eng) and self.parity[n] == 1 and self.parity[n + 1] == -1
                    and self.eng[n + 1] < self.V0):
                overlap = np.sum(self.psi[left, n] * self.psi[left, n + 1]) * dx
                c = amps[n] * C[n] + math.copysign(1, overlap) * amps[n + 1] * C[n + 1]
                rate = (self.eng[n + 1] - self.eng[n]) * eV / kB
                if rate < sampleRate / 2:
                    groups.append((m, c, rate))
                else:
                    centred += c
                n += 2
            else:
                centred += amps[n] * C[n]
                n += 1
        groups.append((m, centred, None))
        total = sum(np.abs(c).sum() for _, c, _ in groups)
        return [g for g in groups if np.abs(g[1]).sum() > 1e-9 * total]

default = DoubleWell()
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    writeSample(default, './audio/doubleWell.wav')
