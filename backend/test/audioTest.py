import numpy as np
import gc
from scipy.io.wavfile import write

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

freq = 220
nSeconds = 5
x = np.linspace(0, 1, 50)
wavefunction = np.sin(x)

gc.collect()
write('./backend/test/test.wav', len(x)*freq, np.tile(scale(wavefunction), nSeconds*freq))