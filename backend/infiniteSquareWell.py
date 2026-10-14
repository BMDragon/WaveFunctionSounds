import numpy as np
from scipy.io.wavfile import write

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

h = 6.62607015e-34      # Planck constant (J s)
kB = 1.380649e-23       # Boltzmann constant (J/K)

mass = 9.1093837e-31    # electron (kg)
length = 5e-9           # well width (m)

E1 = h**2 / (8 * mass * length**2)

# Boltzmann occupation P_n ~ exp(-E_n/kT) with E_n = n^2 E_1, played with
# amplitude c_n = sqrt(P_n). Z cancels since the output is normalized anyway.
def boltzmannWeights(levels, temperature):
    theta = E1 / (kB * temperature)
    return np.exp(-theta * (levels**2 - 1) / 2)

def harmonicContent(freq, sampleRate=44100):
    # Harmonic numbers m and complex amplitudes c_m of one period, so the waveform is
    # Re(sum c_m exp(2 pi i m freq t)). Level n is a sine on harmonic n, the temperature
    # in kelvin equals freq in Hz, and only harmonics below the Nyquist frequency are kept.
    levels = np.arange(1, int(np.ceil(sampleRate / (2*freq))))
    return levels, -1j * boltzmannWeights(levels, freq)

if __name__ == '__main__':
    freq = 220
    nSeconds = 5
    sampleRate = 44100
    temperature = freq      # kelvin, numerically equal to the fundamental in Hz

    # Level n completes n cycles per period, so only n < samplesPerPeriod/2 is kept;
    # anything at or above the Nyquist frequency would alias.
    samplesPerPeriod = int(sampleRate/freq)
    levels = np.arange(1, (samplesPerPeriod + 1) // 2)
    weights = boltzmannWeights(levels, temperature)

    x = np.linspace(0, 2, samplesPerPeriod, False)
    superposition = np.zeros(len(x))
    for n, w in zip(levels, weights):
        superposition += w * np.sin(n * np.pi * x)

    write('./audio/infSqWell.wav', len(x)*freq, scale(np.tile(superposition, freq*nSeconds)))
