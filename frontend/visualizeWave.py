import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

rate, data = read('./audio/infSqWell.wav')
freq = 50
numWaveforms = 3

x = np.linspace(0, 1, rate)
plt.plot(x[:int(rate/freq)*numWaveforms], data[:int(rate/freq)*numWaveforms], 'b-')
plt.show()