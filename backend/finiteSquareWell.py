import numpy as np
import math
from scipy.io.wavfile import write
from scipy.optimize import root

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

hbar = 1.054571817e-34  # reduced Planck constant (J s)
kB = 1.380649e-23       # Boltzmann constant (J/K)
eV = 1.602176634e-19    # electronvolt (J)
electronMass = 9.1093837e-31

class FiniteSquareWell:
    coefficientSamples = 8192
    maxImages = 400         # copies on each side; only barely-bound states in very shallow wells reach it

    def __init__(self, mass=electronMass, length=5e-9, potentialDiff=1.6 * eV, harmonics=True):
        self.mass = mass                    # kg
        self.length = length                # well width (m)
        self.potentialDiff = potentialDiff  # well depth (J)

        u0 = length / (2*hbar) * math.sqrt(2 * mass * potentialDiff)
        numBound = math.ceil(2 * u0 / math.pi)
        solutions = set({})

        # root() passes v as a 1-element array; .item() makes it a float so math.sqrt
        # still raises ValueError past u0, which the scan loop below relies on
        def symmetricTranscendental(v):
            v = np.asarray(v).item()
            return math.sqrt(u0*u0 - v*v) - v*math.tan(v)

        def antiSymTranscendental(v):
            v = np.asarray(v).item()
            return math.sqrt(u0*u0 - v*v) + v*math.tan(v)**-1

        step = u0/(20*numBound)
        for i in np.arange(step, u0, step):
            try:
                if symmetricTranscendental(i) >= 0 and symmetricTranscendental(i+step) <= 0:
                    solutions.add(root(symmetricTranscendental, (i+i+step)/2)['x'][0])
                if antiSymTranscendental(i) >= 0 and antiSymTranscendental(i+step) <= 0:
                    solutions.add(root(antiSymTranscendental, (i+i+step)/2)['x'][0])
            except:
                if symmetricTranscendental(i) >= 0 and symmetricTranscendental(u0) <= 0:
                    solutions.add(root(symmetricTranscendental, u0-step/500)['x'][0])
                if antiSymTranscendental(i) >= 0 and antiSymTranscendental(u0) <= 0:
                    solutions.add(root(antiSymTranscendental, u0-step/500)['x'][0])

        # sort so index n matches energy level n (parity in region1/region2 depends on it)
        vn = np.sort(np.array(list(solutions)))
        if not harmonics:
            vn = vn[:1]

        self.u0 = u0
        self.vn = vn
        self.eng = 2 * (hbar * vn / length)**2 / mass    # E_n = hbar^2 k^2 / 2m with k = 2v/L

        # From here on x is measured in units of the well width, so the well is [-1/2, 1/2]
        self.k = 2 * vn                              # wavenumber inside the well
        self.alpha = 2 * np.sqrt(u0*u0 - vn*vn)      # decay constant of the tails
        self.stateCoefficients = None

    def energies(self, count):
        return self.eng[:count]

    # Boltzmann occupation P_n ~ exp(-E_n/kT), played with amplitude c_n = sqrt(P_n)
    def boltzmannWeights(self, temperature):
        return np.exp(-(self.eng - self.eng[0]) / (2 * kB * temperature))

    def amplitudes(self, temperature, weighting='boltzmann'):
        # Amplitude of every bound state: Boltzmann as above, 'equal': all 1, 'inverse': 1/(n+1)
        if weighting == 'equal':
            return np.ones(len(self.vn))
        if weighting == 'inverse':
            return 1 / np.arange(1, len(self.vn) + 1)
        return self.boltzmannWeights(temperature)

    # x is position, n is energy level where ground is n=0
    def region1(self, x, n):
        return (-1)**n * np.exp(self.alpha[n] * x)

    def amplitude(self, n):
        # inside amplitude that matches the tails at the walls
        k, alpha = self.k, self.alpha
        return math.exp(-alpha[n]/2) / (((n+1)%2) * math.cos(k[n]/2) + (n%2) * math.sin(k[n]/2))

    def region2(self, x, n):
        k = self.k
        return self.amplitude(n) * (((n+1)%2) * np.cos(k[n]*x) + (n%2) * np.sin(k[n]*x))

    def region3(self, x, n):
        return np.exp(-self.alpha[n] * x)

    def wavefunction(self, x, n):
        k, alpha = self.k, self.alpha
        psi = np.where(x < -1/2, self.region1(x, n), np.where(x > 1/2, self.region3(x, n), self.region2(x, n)))
        # integral of |psi|^2: cos^2 or sin^2 inside plus the two exponential tails
        inside = 1/2 + (-1)**n * math.sin(k[n]) / (2*k[n])
        return psi / math.sqrt(self.amplitude(n)**2 * inside + math.exp(-alpha[n]) / alpha[n])

    # One period is two well widths. Copies of each state are laid in neighbouring
    # units with their tails overlapping, even states flipping sign from one unit to
    # the next (psi and -psi are the same state). This matches the infinite well in
    # the deep-well limit and makes every seam smooth.
    def wrappedState(self, x, n):
        sign = -1 if n % 2 == 0 else 1
        images = min(math.ceil(math.log(1e6) / self.alpha[n] + 1/2), self.maxImages)    # tails down to 1e-6
        return sum(sign**abs(j) * self.wavefunction(x - j, n) for j in range(-images, images + 2))

    def harmonicContent(self, freq, sampleRate=44100, temperature=None, weighting='boltzmann'):
        # Harmonic numbers m and complex amplitudes c_m of one period, so the waveform is
        # Re(sum c_m exp(2 pi i m freq t)). The temperature in kelvin equals freq in Hz
        # unless given, and only harmonics below the Nyquist frequency are kept so nothing
        # aliases. The wrapped states' Fourier coefficients don't depend on pitch, so they
        # are computed once, on the first call.
        if self.stateCoefficients is None:
            x = np.linspace(-1/2, 3/2, self.coefficientSamples, False)
            self.stateCoefficients = np.array([2 * np.fft.rfft(self.wrappedState(x, n)) / self.coefficientSamples
                                               for n in range(len(self.vn))])
        m = np.arange(1, min(int(np.ceil(sampleRate / (2*freq))), self.coefficientSamples // 2))
        return m, self.amplitudes(temperature or freq, weighting) @ self.stateCoefficients[:, m]

    def plotData(self, count):
        # Position (nm), potential (meV), level energies (meV) and the first `count` bound
        # wavefunctions (nm^-1/2), with the well centred on 0
        count = min(count, len(self.vn))
        reach = min(1/2 + 4 / self.alpha[:count].min(), 2.5)
        x = np.linspace(-reach, reach, 601)
        V = np.where(np.abs(x) <= 1/2, 0, self.potentialDiff) / (eV * 1e-3)
        psi = [self.wavefunction(x, n) / math.sqrt(self.length * 1e9) for n in range(count)]
        return {'x': x * self.length * 1e9, 'V': V, 'energies': self.eng[:count] / (eV * 1e-3), 'psi': psi}

default = FiniteSquareWell()
vn, eng, u0 = default.vn, default.eng, default.u0
wrappedState = default.wrappedState
boltzmannWeights = default.boltzmannWeights
harmonicContent = default.harmonicContent

if __name__ == '__main__':
    freq = 220
    nSeconds = 5
    sampleRate = 44100
    temperature = freq          # kelvin, numerically equal to the fundamental in Hz

    # Each state is built on a finer grid, then only harmonics below the Nyquist
    # frequency are kept so nothing aliases.
    weights = boltzmannWeights(temperature)
    samplesPerPeriod = int(sampleRate/freq)
    oversample = 16
    x = np.linspace(-1/2, 3/2, oversample * samplesPerPeriod, False)
    superposition = np.zeros(samplesPerPeriod)
    for n in range(len(vn)):
        spectrum = np.fft.rfft(wrappedState(x, n))[:(samplesPerPeriod + 1) // 2]
        superposition += weights[n] * np.fft.irfft(spectrum, samplesPerPeriod)

    write('./audio/finiteSqWell.wav', len(superposition)*freq, scale(np.tile(superposition, freq*nSeconds)))
