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

# x is measured in units of a. As in the square wells, one unit is the width of the ground
# state, between its turning points at +-1, and one period is two units. Every state is laid
# on that scale: copies of it go in each unit, the higher (wider) states carrying over into the
# neighbouring units and adding there, and even (symmetric) states flip sign from one copy to
# the next.
coefficientSamples = 8192
stateCoefficients = None

def computeCoefficients():
    # Fourier coefficients over one period (from x = -1, two units long) of every state's
    # copies, sum over j of sign^j psi(x - 2j): each Hermite function is sampled once on a grid
    # aligned with the period, beyond every kept state's turning point, and folded. In units of
    # a they are the same for every oscillator.
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

    def harmonicContent(self, freq, sampleRate=44100, temperature=None, weighting='boltzmann'):
        # Harmonic numbers m and complex amplitudes c_m of one period, so the waveform is
        # Re(sum c_m exp(2 pi i m freq t)). The temperature in kelvin equals freq in Hz
        # unless given, and only harmonics below the Nyquist frequency are kept so nothing
        # aliases.
        global stateCoefficients
        if stateCoefficients is None:
            stateCoefficients = computeCoefficients()
        m = np.arange(1, min(int(np.ceil(sampleRate / (2*freq))), coefficientSamples // 2))
        return m, self.amplitudes(temperature or freq, weighting) @ stateCoefficients[:, m]

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
