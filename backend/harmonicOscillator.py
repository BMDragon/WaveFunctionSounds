import numpy as np
import math
from scipy.io.wavfile import write

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

hbar = 1.054571817e-34  # reduced Planck constant (J s)
kB = 1.380649e-23       # Boltzmann constant (J/K)
electronMass = 9.1093837e-31

# Levels kept. With the default oscillator the weights fall to 1e-5 by n = 400 for notes
# up to about 2.4 kHz.
numStates = 400

def hermiteFunctions(x, count):
    # psi_0..psi_{count-1} with x in units of the oscillator length, from the stable
    # recurrence psi_{n+1} = sqrt(2/(n+1)) x psi_n - sqrt(n/(n+1)) psi_{n-1}
    previous = np.zeros_like(x)
    current = np.pi**-0.25 * np.exp(-x**2 / 2)
    for n in range(count):
        yield current
        previous, current = current, math.sqrt(2/(n+1)) * x * current - math.sqrt(n/(n+1)) * previous

# x is measured in units of a. Each state's unit is the width of the region its energy
# allows, between the turning points at +-sqrt(2n+1); for a square well that width is L at
# every energy. As in the square wells, one period is two units, copies of each state are
# laid in neighbouring units with their tails overlapping, and even (symmetric) states flip
# sign from one unit to the next. Each state is normalized over its own unit, so state n
# has n+1 half-oscillations per unit like the square wells.
coefficientSamples = 8192
stateCoefficients = None

def computeCoefficients():
    # Fourier coefficients of every wrapped state over one period, from the Hermite
    # functions on a fine grid interpolated at every copy of the state that reaches its
    # period. In units of a they are the same for every oscillator.
    grid = np.arange(-37.5, 37.5, 1e-3)         # beyond every kept state's turning point
    u = np.linspace(0, 1, coefficientSamples, False)
    coefficients = []
    for n, psi in enumerate(hermiteFunctions(grid, numStates)):
        width = 2 * math.sqrt(2*n + 1)
        images = np.arange(-math.ceil(1/2 + 6/width) - 1, math.ceil(1/2 + 6/width) + 2)
        X = width * (2*u[None, :] - 1/2) - width * images[:, None]
        sign = (-1.0)**np.abs(images) if n % 2 == 0 else np.ones(len(images))
        wrapped = math.sqrt(width) * (sign @ np.interp(X, grid, psi, left=0, right=0))
        coefficients.append(2 * np.fft.rfft(wrapped) / coefficientSamples)
    return np.array(coefficients)

# The same on the ground state's scale: one unit is the ground state's width (turning points at
# +-1, so 2 in units of a) for every state, the wider ones carrying over into the neighbouring
# units; each Hermite function is sampled once on a grid aligned with the period and folded.
groundCoefficients = None

def computeGroundCoefficients():
    N = coefficientSamples
    h = 4 / N
    i = np.arange(math.floor(-36.5 / h), math.ceil(38.5 / h))
    x = -1 + i * h
    coefficients = []
    for n, psi in enumerate(hermiteFunctions(x, numStates)):
        folded = np.bincount(i % N, psi, N)
        wrapped = math.sqrt(2) * (folded + (-1 if n % 2 == 0 else 1) * np.roll(folded, N // 2))
        coefficients.append(2 * np.fft.rfft(wrapped) / N)
    return np.array(coefficients)

class HarmonicOscillator:
    def __init__(self, mass=electronMass, length=5e-9):
        self.mass = mass        # kg
        self.length = length    # classical width of the ground state, between its turning points (m)
        # The ground state's turning points are at +-a, with a = sqrt(hbar/(m omega)) the
        # oscillator length, so its classical width 2a sets the oscillator's frequency.
        self.a = length / 2
        self.hbarOmega = hbar**2 / (mass * self.a**2)   # level spacing, E_n = hbarOmega (n + 1/2)

    def energies(self, count):
        return self.hbarOmega * (np.arange(count) + 1/2)

    # Boltzmann occupation P_n ~ exp(-n hbarOmega/kT), played with amplitude c_n = sqrt(P_n)
    def boltzmannWeights(self, temperature):
        return np.exp(-np.arange(numStates) * self.hbarOmega / (2 * kB * temperature))

    def amplitudes(self, temperature, weighting='boltzmann'):
        # Amplitude of every kept level: Boltzmann as above, 'equal': all 1, 'inverse': 1/(n+1),
        # or a list of amplitudes set by hand
        if not isinstance(weighting, str):           # hand-set amplitudes, lowest level first
            amps = np.zeros(numStates)
            given = np.asarray(weighting, float)[:numStates]
            amps[:len(given)] = given
            return amps
        if weighting == 'equal':
            return np.ones(numStates)
        if weighting == 'inverse':
            return 1 / np.arange(1, numStates + 1)
        return self.boltzmannWeights(temperature)

    def harmonicContent(self, freq, sampleRate=44100, temperature=None, weighting='boltzmann', wrapping='own'):
        # Harmonic numbers m and complex amplitudes c_m of one period, so the waveform is
        # Re(sum c_m exp(2 pi i m freq t)). The temperature in kelvin equals freq in Hz
        # unless given, and only harmonics below the Nyquist frequency are kept so nothing
        # aliases.
        global stateCoefficients, groundCoefficients
        if wrapping == 'ground':
            if groundCoefficients is None:
                groundCoefficients = computeGroundCoefficients()
            C = groundCoefficients
        else:
            if stateCoefficients is None:
                stateCoefficients = computeCoefficients()
            C = stateCoefficients
        m = np.arange(1, min(int(np.ceil(sampleRate / (2*freq))), coefficientSamples // 2))
        return m, self.amplitudes(temperature or freq, weighting) @ C[:, m]

    def plotData(self, count):
        # Position (nm), potential (meV), level energies (meV) and the first `count`
        # wavefunctions (nm^-1/2)
        reach = math.sqrt(2*count - 1) + 2.5
        x = np.linspace(-reach, reach, 601)
        meV = 1.602176634e-22
        return {'x': x * self.a * 1e9, 'V': self.hbarOmega * x**2 / 2 / meV,
                'energies': self.energies(count) / meV,
                'psi': [psi / math.sqrt(self.a * 1e9) for psi in hermiteFunctions(x, count)]}

default = HarmonicOscillator()
a, hbarOmega = default.a, default.hbarOmega
boltzmannWeights = default.boltzmannWeights
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    freq = 220
    nSeconds = 5
    sampleRate = 44100

    samplesPerPeriod = int(sampleRate/freq)
    m, c = harmonicContent(freq, samplesPerPeriod * freq)
    phase = 2 * np.pi * np.outer(np.arange(samplesPerPeriod) / samplesPerPeriod, m)
    superposition = (c.real * np.cos(phase) - c.imag * np.sin(phase)).sum(axis=1)

    write('./audio/harmonicOscillator.wav', samplesPerPeriod*freq, scale(np.tile(superposition, freq*nSeconds)))
