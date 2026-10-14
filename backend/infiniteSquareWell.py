import numpy as np
from scipy.io.wavfile import write

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

h = 6.62607015e-34      # Planck constant (J s)
kB = 1.380649e-23       # Boltzmann constant (J/K)

freq = 220
nSeconds = 5
sampleRate = 44100

mass = 9.1093837e-31    # electron (kg)
length = 5e-9           # well width (m)
temperature = freq      # kelvin, numerically equal to the fundamental in Hz

samplesPerPeriod = int(sampleRate/freq)

# Boltzmann occupation P_n ~ exp(-E_n/kT) with E_n = n^2 E_1, played with
# amplitude c_n = sqrt(P_n). Z cancels since scale() normalizes the peak.
# Level n completes n cycles per period, so only n < samplesPerPeriod/2 is kept;
# anything at or above the Nyquist frequency would alias.
E1 = h**2 / (8 * mass * length**2)
theta = E1 / (kB * temperature)
levels = np.arange(1, (samplesPerPeriod + 1) // 2)
weights = np.exp(-theta * (levels**2 - 1) / 2)

# One period is the well followed by its sign-flipped copy. psi and -psi are the
# same state (global phase), and the flip removes the kink odd-n states would
# otherwise have at the seam, so level n completes n full cycles per period.
x = np.linspace(0, 2, samplesPerPeriod, False)
superposition = np.zeros(len(x))
for n, w in zip(levels, weights):
    superposition += w * np.sin(n * np.pi * x)

write('./audio/infSqWell.wav', len(x)*freq, scale(np.tile(superposition, freq*nSeconds)))
