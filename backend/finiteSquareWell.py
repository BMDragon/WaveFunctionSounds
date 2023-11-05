import numpy as np
import math
from scipy.io.wavfile import write
from scipy.optimize import root
import matplotlib.pyplot as plt

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

def normalize(psi):
   return psi / np.linalg.norm(psi)

freq = 80
nSeconds = 5
sampleRate = 44000
harmonics = True

potentialDiff = 500
length = 1
mass = 1

u0 = math.sqrt(mass * length * length * potentialDiff / 2)
numBound = math.ceil(2 * u0 / math.pi)
solutions = set({})

def symmetricTranscendental(v):
    return math.sqrt(u0*u0 - v*v) - v*math.tan(v)

def antiSymTranscendental(v):
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

vn = np.array([])
for j in solutions:
    vn = np.append(vn, j)

eng = np.multiply(vn, vn) * 2 / (mass * length * length)
if not harmonics:
    eng = eng[:1]
k = np.sqrt(2 * mass * eng)
alpha = np.sqrt(2*mass * (potentialDiff - eng))

# x is position, n is energy level where ground is n=0
def region1(x, n):
    return (-1)**n * np.exp(alpha[n] * x)

def region2(x, n):
    scaling = math.exp(-alpha[n] * length/2) / (((n+1)%2) * math.cos(k[n] * length/2) + (n%2) * math.sin(k[n] * length/2))
    return scaling * (((n+1)%2) * np.cos(k[n]*x) + (n%2) * np.sin(k[n]*x))

def region3(x, n):
    return np.exp(-alpha[n] * x)

def makeTail(xlim, unitsize, stepsize):
    currLength = xlim - length/2
    needToAdd = unitsize - (currLength/stepsize)%unitsize
    return np.arange(length/2, xlim + needToAdd*stepsize, stepsize)

reg2 = np.linspace(-length/2, length/2, int(sampleRate/freq))
stepsize = reg2[1]-reg2[0]
superposition = np.zeros(len(reg2))
for z in range(len(eng)):
    wavefunction = np.array([])
    xlim = length/2 - math.log(0.01)/alpha[z]
    reg3 = makeTail(xlim, len(reg2), stepsize)
    reg1 = np.flip(-reg3)

    wavefunction = np.append(wavefunction, region1(reg1, z)[1:])
    wavefunction = np.append(wavefunction, region2(reg2, z))
    wavefunction = np.append(wavefunction, region3(reg3, z)[:-1])
    wavefunction = normalize(wavefunction)

    for m in range(int(len(wavefunction)/len(superposition))):
        superposition = np.add(superposition, wavefunction[m*len(superposition):(m+1)*len(superposition)])
    plt.plot(superposition)
    plt.show()

write('./audio/finiteSqWell.wav', len(superposition)*freq, scale(np.tile(superposition, freq*nSeconds)))