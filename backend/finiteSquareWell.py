import numpy as np
import math
from scipy.io.wavfile import write
from scipy.optimize import root

def scale(waveform):
    return np.int16(waveform / np.max(np.abs(waveform)) * 32767)

def normalize(psi):
   return psi / np.linalg.norm(psi)

freq = 80
nSeconds = 5
sampleRate = 44000
harmonics = True

potentialDiff = 5
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
x = np.linspace(0, length, int(sampleRate/freq))[:-1]
alpha = np.sqrt(2*mass * (potentialDiff - eng))

    

# write('./audio/finiteSqWell.wav', len(x)*freq, scale(np.tile(wavefunction, freq*nSeconds)))