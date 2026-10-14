import numpy as np
import math
from scipy.io.wavfile import write

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

hbar = 1.054571817e-34  # reduced Planck constant (J s)
kB = 1.380649e-23       # Boltzmann constant (J/K)

mass = 9.1093837e-31    # electron (kg)
length = 5e-9           # classical width of the ground state, between its turning points (m)

# The ground state's turning points are at +-a, with a = sqrt(hbar/(m omega)) the
# oscillator length, so its classical width 2a sets the oscillator's frequency.
a = length / 2
hbarOmega = hbar**2 / (mass * a**2)     # level spacing, E_n = hbarOmega (n + 1/2)

# Levels kept. The weights below fall to 1e-5 by n = 400 for notes up to about 2.4 kHz.
numStates = 400

# Boltzmann occupation P_n ~ exp(-n hbarOmega/kT), played with amplitude c_n = sqrt(P_n)
def boltzmannWeights(temperature):
    return np.exp(-np.arange(numStates) * hbarOmega / (2 * kB * temperature))

# From here on x is measured in units of a. Each state's unit is the width of the region
# its energy allows, between the turning points at +-sqrt(2n+1); for a square well that
# width is L at every energy. As in the square wells, one period is two units, copies of
# each state are laid in neighbouring units with their tails overlapping, and even
# (symmetric) states flip sign from one unit to the next. Each state is normalized over
# its own unit, so state n has n+1 half-oscillations per unit like the square wells.
coefficientSamples = 8192
stateCoefficients = None

def computeCoefficients():
    # Fourier coefficients of every wrapped state over one period. The Hermite functions
    # come from the stable recurrence
    # psi_{n+1} = sqrt(2/(n+1)) x psi_n - sqrt(n/(n+1)) psi_{n-1} on a fine grid, then are
    # interpolated at every copy of the state that reaches its period.
    grid = np.arange(-37.5, 37.5, 1e-3)         # beyond every kept state's turning point
    u = np.linspace(0, 1, coefficientSamples, False)
    previous = np.zeros_like(grid)
    current = np.pi**-0.25 * np.exp(-grid**2 / 2)
    coefficients = []
    for n in range(numStates):
        width = 2 * math.sqrt(2*n + 1)
        images = np.arange(-math.ceil(1/2 + 6/width) - 1, math.ceil(1/2 + 6/width) + 2)
        X = width * (2*u[None, :] - 1/2) - width * images[:, None]
        sign = (-1.0)**np.abs(images) if n % 2 == 0 else np.ones(len(images))
        wrapped = math.sqrt(width) * (sign @ np.interp(X, grid, current, left=0, right=0))
        coefficients.append(2 * np.fft.rfft(wrapped) / coefficientSamples)
        previous, current = current, math.sqrt(2/(n+1)) * grid * current - math.sqrt(n/(n+1)) * previous
    return np.array(coefficients)

def harmonicContent(freq, sampleRate=44100):
    # Harmonic numbers m and complex amplitudes c_m of one period, so the waveform is
    # Re(sum c_m exp(2 pi i m freq t)). The temperature in kelvin equals freq in Hz, and
    # only harmonics below the Nyquist frequency are kept so nothing aliases.
    global stateCoefficients
    if stateCoefficients is None:
        stateCoefficients = computeCoefficients()
    m = np.arange(1, min(int(np.ceil(sampleRate / (2*freq))), coefficientSamples // 2))
    return m, boltzmannWeights(freq) @ stateCoefficients[:, m]

if __name__ == '__main__':
    freq = 220
    nSeconds = 5
    sampleRate = 44100

    samplesPerPeriod = int(sampleRate/freq)
    m, c = harmonicContent(freq, samplesPerPeriod * freq)
    phase = 2 * np.pi * np.outer(np.arange(samplesPerPeriod) / samplesPerPeriod, m)
    superposition = (c.real * np.cos(phase) - c.imag * np.sin(phase)).sum(axis=1)

    write('./audio/harmonicOscillator.wav', samplesPerPeriod*freq, scale(np.tile(superposition, freq*nSeconds)))
