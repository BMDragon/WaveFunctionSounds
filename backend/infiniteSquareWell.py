import numpy as np
from scipy.io.wavfile import write

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

freq = 80
nSeconds = 5
sampleRate = 44000
harmonic = 13

x = np.linspace(0, 1, int(sampleRate/freq))[:-1]
k = harmonic * np.pi
wavefunction = np.sin(k*x)

write('./audio/infSqWell.wav', len(x)*freq, scale(np.tile(wavefunction, freq*nSeconds)))