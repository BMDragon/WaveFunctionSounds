import numpy as np
from scipy.io.wavfile import write

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

h = 6.62607015e-34      # Planck constant (J s)
kB = 1.380649e-23       # Boltzmann constant (J/K)
electronMass = 9.1093837e-31

class InfiniteSquareWell:
    def __init__(self, mass=electronMass, length=5e-9):
        self.mass = mass        # kg
        self.length = length    # well width (m)
        self.E1 = h**2 / (8 * mass * length**2)

    def energies(self, count):
        # E_n = n^2 E_1 for n = 1..count (J)
        return self.E1 * np.arange(1, count + 1)**2

    def amplitudes(self, levels, temperature, weighting='boltzmann'):
        # Amplitude of each level. 'boltzmann': occupation P_n ~ exp(-E_n/kT) played with
        # amplitude c_n = sqrt(P_n), relative to the ground state; 'equal': all 1;
        # 'inverse': 1/n; or a list of amplitudes set by hand.
        if not isinstance(weighting, str):           # hand-set amplitudes, lowest level first
            amps = np.zeros(len(levels))
            given = np.asarray(weighting, float)[:len(levels)]
            amps[:len(given)] = given
            return amps
        if weighting == 'equal':
            return np.ones(len(levels))
        if weighting == 'inverse':
            return 1 / levels
        theta = self.E1 / (kB * temperature)
        return np.exp(-theta * (levels**2 - 1) / 2)

    def harmonicContent(self, freq, sampleRate=44100, temperature=None, weighting='boltzmann', wrapping='own'):
        # `wrapping` makes no difference here (every state already shares one unit, or none is wrapped)
        # Harmonic numbers m and complex amplitudes c_m of one period, so the waveform is
        # Re(sum c_m exp(2 pi i m freq t)). Level n is a sine on harmonic n, the temperature
        # in kelvin equals freq in Hz unless given, and only harmonics below the Nyquist
        # frequency are kept.
        levels = np.arange(1, int(np.ceil(sampleRate / (2*freq))))
        return levels, -1j * self.amplitudes(levels, temperature or freq, weighting)

    def plotData(self, count):
        # Position (nm), potential (meV, None where infinite), level energies (meV) and the
        # first `count` wavefunctions (nm^-1/2), with the well centred on 0
        L = self.length
        x = np.linspace(-0.6 * L, 0.6 * L, 601)
        inside = np.abs(x) <= L / 2
        psi = [np.where(inside, np.sqrt(2 / L) * np.sin(n * np.pi * (x / L + 1/2)), 0) * np.sqrt(1e-9)
               for n in range(1, count + 1)]
        return {'x': x * 1e9, 'V': [0.0 if i else None for i in inside],
                'energies': self.energies(count) / 1.602176634e-22, 'psi': psi}

default = InfiniteSquareWell()
E1 = default.E1
harmonicContent = default.harmonicContent

def boltzmannWeights(levels, temperature):
    return default.amplitudes(levels, temperature)

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
