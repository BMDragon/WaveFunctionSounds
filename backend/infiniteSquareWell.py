import numpy as np
from scipy.io.wavfile import write

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

freq = 220
nSeconds = 5
sampleRate = 44100
levels = [1, 2, 3, 4, 5]                # energy levels to stack, n = 1 is the ground state
weights = [1, 1/2, 1/3, 1/4, 1/5]       # amplitude of each level in the superposition

# One period is the well followed by its sign-flipped copy. psi and -psi are the
# same state (global phase), and the flip removes the kink odd-n states would
# otherwise have at the seam, so level n completes n full cycles per period.
x = np.linspace(0, 2, int(sampleRate/freq), False)
superposition = np.zeros(len(x))
for n, w in zip(levels, weights):
    superposition += w * np.sin(n * np.pi * x)

write('./audio/infSqWell.wav', len(x)*freq, scale(np.tile(superposition, freq*nSeconds)))
