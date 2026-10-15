import numpy as np

# The sound of a state evolving in time. The particle starts in psi(0) = sum_n a_n psi_n, with
# the level weights a_n, and evolves as psi(t) = sum_n a_n exp(-i E_n t / hbar) psi_n. What is
# played is its survival probability, the chance of finding it still in the state it started
# in: |<psi(0)|psi(t)>|^2 = |sum_n p_n exp(-i E_n t / hbar)|^2 with p_n = a_n^2 / sum a^2,
#   = sum_n p_n^2 + sum_{n<m} 2 p_n p_m cos((E_m - E_n) t / hbar).
# Unlike psi itself this is observable (it doesn't depend on the zero of energy or the phase
# of the states), and its partials are at the energy differences. They are played with the
# first excitation, E_1 - E_0 (or the first non-zero gap above the ground state), at the key's
# frequency; the constant sum_n p_n^2 is silent.

def survivalContent(energies, amplitudes, freq, sampleRate=44100):
    # Partial ratios and complex amplitudes, as harmonicContent gives them, from the energies
    # of the states (any unit, ascending) and their weights a_n
    E = np.asarray(energies, float)
    p = np.asarray(amplitudes, float)[:len(E)]**2
    E, p = E[:len(p)], p / p.sum()
    gaps = E - E[0]
    unit = gaps[gaps > 1e-12 * max(gaps.max(), 1e-300)]     # exactly degenerate states (to rounding) have no gap
    if not len(unit):
        return np.zeros(0), np.zeros(0, complex)
    unit = unit[0]
    used = p > 1e-12 * p.max()
    E, p = E[used], p[used]
    i, j = np.triu_indices(len(E), 1)
    ratio = (E[j] - E[i]) / unit
    amp = 2 * p[i] * p[j]
    keep = (ratio > 1e-9) & (ratio * freq < sampleRate / 2) & (amp > 1e-9 * amp.max(initial=0))
    merged, index = np.unique(np.round(ratio[keep], 9), return_inverse=True)
    c = np.zeros(len(merged), complex)
    np.add.at(c, index, amp[keep])
    return merged, c
