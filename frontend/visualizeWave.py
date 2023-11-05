import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

rate, data = read('./audio/finiteSqWell.wav')
freq = 80
numWaveforms = 2

x = np.linspace(0, 1, rate)
plt.plot(x[:int(rate/freq)*numWaveforms], data[:int(rate/freq)*numWaveforms], 'b-')
plt.show()