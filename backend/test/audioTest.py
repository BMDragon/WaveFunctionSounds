import numpy as np
import gc
from scipy.io.wavfile import write

freq = 100
nSeconds = 2
x = np.linspace(0, 1, 50)
waveform = np.sin(x)
gc.collect()
write('./test.wav', len(x)*freq, np.tile(waveform, nSeconds*freq))