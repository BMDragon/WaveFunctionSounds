import numpy as np
import math
from scipy.io.wavfile import write
from scipy.optimize import brentq

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

hbar = 1.054571817e-34  # reduced Planck constant (J s)
kB = 1.380649e-23       # Boltzmann constant (J/K)
eV = 1.602176634e-19    # electronvolt (J)
electronMass = 9.1093837e-31

class FiniteSquareWell:
    coefficientSamples = 8192

    def __init__(self, mass=electronMass, length=5e-9, potentialDiff=1.6 * eV, harmonics=True):
        self.mass = mass                    # kg
        self.length = length                # well width (m)
        self.potentialDiff = potentialDiff  # well depth (J)

        u0 = length / (2*hbar) * math.sqrt(2 * mass * potentialDiff)
        numBound = math.ceil(2 * u0 / math.pi)

        # Exactly one state lies in each interval (j pi/2, (j+1) pi/2) below u0: even states
        # solve sqrt(u0^2 - v^2) = v tan v (j even), odd states sqrt(u0^2 - v^2) = -v cot v
        # (j odd). Multiplied through by cos v or sin v the equations have no poles, so each
        # root is bracketed by its interval, even in deep wells where it sits just below a pole.
        def symmetricTranscendental(v):
            return math.sqrt(max(u0*u0 - v*v, 0)) * math.cos(v) - v * math.sin(v)

        def antiSymTranscendental(v):
            return math.sqrt(max(u0*u0 - v*v, 0)) * math.sin(v) + v * math.cos(v)

        vn = np.array([brentq(symmetricTranscendental if j % 2 == 0 else antiSymTranscendental,
                              j * math.pi / 2, min((j + 1) * math.pi / 2, u0), xtol=1e-14)
                       for j in range(numBound)])
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
        # Amplitude of every bound state: Boltzmann as above, 'equal': all 1, 'inverse': 1/(n+1),
        # or a list of amplitudes set by hand
        if not isinstance(weighting, str):           # hand-set amplitudes, lowest level first
            amps = np.zeros(len(self.vn))
            given = np.asarray(weighting, float)[:len(self.vn)]
            amps[:len(given)] = given
            return amps
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
        # sum over copies j of sign^j psi(x - j). The inside of the well reaches x from at
        # most a few copies; the tails are exponentials, so their sum over all copies is a
        # geometric series, exact however slowly a barely bound state decays.
        sign = -1 if n % 2 == 0 else 1
        alpha = self.alpha[n]
        total = np.zeros_like(x)
        for j in range(math.floor(x.min() - 1/2), math.ceil(x.max() + 1/2) + 1):
            y = x - j
            inside = np.abs(y) <= 1/2
            total[inside] += sign**abs(j) * self.wavefunction(y[inside], n)
        # right tails come from the copies j with x - j > 1/2, left tails from x - j < -1/2;
        # each starts at the copy nearest x and falls by a factor sign exp(-alpha) per copy
        norm = self.wavefunction(np.array([1/2]), n)[0], self.wavefunction(np.array([-1/2]), n)[0]
        ratio = sign * math.exp(-alpha)
        nearestRight = np.ceil(x - 1/2) - 1                 # largest j with x - j > 1/2
        nearestLeft = np.floor(x + 1/2) + 1                 # smallest j with x - j < -1/2
        total += norm[0] * sign**np.abs(nearestRight) * np.exp(-alpha * (x - nearestRight - 1/2)) / (1 - ratio)
        total += norm[1] * sign**np.abs(nearestLeft) * np.exp(alpha * (x - nearestLeft + 1/2)) / (1 - ratio)
        return total

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
