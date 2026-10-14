import numpy as np
import math
from scipy.io.wavfile import write
from scipy.optimize import root

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

hbar = 1.054571817e-34  # reduced Planck constant (J s)
kB = 1.380649e-23       # Boltzmann constant (J/K)
eV = 1.602176634e-19    # electronvolt (J)

harmonics = True

mass = 9.1093837e-31        # electron (kg)
length = 5e-9               # well width (m)
potentialDiff = 1.6 * eV    # well depth (J)

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

eng = 2 * (hbar * vn / length)**2 / mass    # E_n = hbar^2 k^2 / 2m with k = 2v/L

# Boltzmann occupation P_n ~ exp(-E_n/kT), played with amplitude c_n = sqrt(P_n)
def boltzmannWeights(temperature):
    return np.exp(-(eng - eng[0]) / (2 * kB * temperature))

# From here on x is measured in units of the well width, so the well is [-1/2, 1/2]
k = 2 * vn                              # wavenumber inside the well
alpha = 2 * np.sqrt(u0*u0 - vn*vn)      # decay constant of the tails

# x is position, n is energy level where ground is n=0
def region1(x, n):
    return (-1)**n * np.exp(alpha[n] * x)

def amplitude(n):
    # inside amplitude that matches the tails at the walls
    return math.exp(-alpha[n]/2) / (((n+1)%2) * math.cos(k[n]/2) + (n%2) * math.sin(k[n]/2))

def region2(x, n):
    return amplitude(n) * (((n+1)%2) * np.cos(k[n]*x) + (n%2) * np.sin(k[n]*x))

def region3(x, n):
    return np.exp(-alpha[n] * x)

def wavefunction(x, n):
    psi = np.where(x < -1/2, region1(x, n), np.where(x > 1/2, region3(x, n), region2(x, n)))
    # integral of |psi|^2: cos^2 or sin^2 inside plus the two exponential tails
    inside = 1/2 + (-1)**n * math.sin(k[n]) / (2*k[n])
    return psi / math.sqrt(amplitude(n)**2 * inside + math.exp(-alpha[n]) / alpha[n])

# One period is two well widths. Copies of each state are laid in neighbouring
# units with their tails overlapping, even states flipping sign from one unit to
# the next (psi and -psi are the same state). This matches the infinite well in
# the deep-well limit and makes every seam smooth.
def wrappedState(x, n):
    sign = -1 if n % 2 == 0 else 1
    images = math.ceil(math.log(1e6) / alpha[n] + 1/2)    # tails down to 1e-6
    return sum(sign**abs(j) * wavefunction(x - j, n) for j in range(-images, images + 2))

# Fourier coefficients of every wrapped state over one period. They don't depend on
# pitch, so they are computed once, on the first call to harmonicContent().
coefficientSamples = 8192
stateCoefficients = None

def harmonicContent(freq, sampleRate=44100):
    # Harmonic numbers m and complex amplitudes c_m of one period, so the waveform is
    # Re(sum c_m exp(2 pi i m freq t)). The temperature in kelvin equals freq in Hz, and
    # only harmonics below the Nyquist frequency are kept so nothing aliases.
    global stateCoefficients
    if stateCoefficients is None:
        x = np.linspace(-1/2, 3/2, coefficientSamples, False)
        stateCoefficients = np.array([2 * np.fft.rfft(wrappedState(x, n)) / coefficientSamples
                                      for n in range(len(vn))])
    m = np.arange(1, min(int(np.ceil(sampleRate / (2*freq))), coefficientSamples // 2))
    return m, boltzmannWeights(freq) @ stateCoefficients[:, m]

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
