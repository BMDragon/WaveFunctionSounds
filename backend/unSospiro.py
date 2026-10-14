import sys
import math
import numpy as np
from scipy.io.wavfile import write

import infiniteSquareWell
import finiteSquareWell
import harmonicOscillator
import doubleWell
import morsePotential
import latticePotential
import linearPotential

# Plays the opening of Liszt's Un sospiro with every note's timbre built from a potential's
# eigenstates at that note's pitch (temperature in kelvin = pitch in Hz, so higher notes
# are brighter). Usage: python backend/unSospiro.py [path/to/midi]
defaultMidiPath = './midi/unSospiro.mid'

sampleRate = 44100
nSeconds = 65
attackTime = 0.005      # s, ramp at the start of each note so it doesn't click
decayTime = 1.5         # s, exponential decay of a held note, like a struck string
releaseTime = 0.1       # s, fade once the key and sustain pedal are both up
fadeOutTime = 1.0       # s, fade at the end of the excerpt
silence = 1e-5          # envelope level below which a note is no longer computed
tableSize = 16384       # samples in each note's one-period table

def readNotes(path, nSeconds):
    # (start, end, midi note, velocity) for the notes beginning within nSeconds of the
    # first note, with times measured from the first note. A note released while the
    # sustain pedal is down keeps sounding until the pedal comes up.
    import mido     # only needed here, so the web app runs without it until a render is asked for
    t = 0.0
    pedalDown = False
    held = {}           # note -> [(start, velocity)] of keys still down
    sustained = []      # (start, note, velocity) of notes held only by the pedal
    notes = []
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            held.setdefault(msg.note, []).append((t, msg.velocity))
        elif msg.type in ('note_on', 'note_off') and held.get(msg.note):
            start, velocity = held[msg.note].pop(0)
            if pedalDown:
                sustained.append((start, msg.note, velocity))
            else:
                notes.append((start, t, msg.note, velocity))
        elif msg.type == 'control_change' and msg.control == 64:
            pedalDown = msg.value >= 64
            if not pedalDown:
                notes += [(start, t, note, velocity) for start, note, velocity in sustained]
                sustained = []
    # anything still sounding at the end of the file
    notes += [(start, t, note, velocity) for start, note, velocity in sustained]
    notes += [(start, t, note, velocity) for note, keys in held.items() for start, velocity in keys]

    first = min(start for start, _, _, _ in notes)
    return sorted((start - first, end - first, note, velocity)
                  for start, end, note, velocity in notes if start - first < nSeconds)

def render(notes, harmonicContent, nSeconds=nSeconds, attackTime=attackTime, decayTime=decayTime,
           releaseTime=releaseTime):
    out = np.zeros(int(nSeconds * sampleRate))
    for start, end, note, velocity in notes:
        freq = 440 * 2**((note - 69) / 12)
        m, c = harmonicContent(freq, sampleRate)

        # stop once the decay or the release has taken the note below `silence`
        held = end - start
        duration = -math.log(silence) * min(decayTime, releaseTime + held / -math.log(silence))
        i0 = int(start * sampleRate)
        t = np.arange(min(len(out) - i0, int(duration * sampleRate))) / sampleRate

        if np.allclose(m, np.round(m)):
            # one period as a fine table, Re(sum c_m exp(2 pi i m phase)), read out at the
            # note's exact pitch
            spectrum = np.zeros(tableSize // 2 + 1, complex)
            spectrum[m] = c
            table = np.fft.irfft(spectrum, tableSize) * tableSize / 2
            wave = np.interp((freq * t) % 1 * tableSize, np.arange(tableSize + 1), np.append(table, table[0]))
        else:
            # inharmonic partials at (n + offset) freq with integer n and a common offset (the
            # lattice's Bloch phase): one period of the complex sum over n as a table, turned
            # by exp(2 pi i offset freq t)
            offset = m[0] - math.floor(m[0])
            n = np.round(m - offset).astype(int)
            spectrum = np.zeros(tableSize, complex)
            np.add.at(spectrum, n % tableSize, c)
            table = np.fft.ifft(spectrum) * tableSize
            table = np.append(table, table[0])
            phase, grid = (freq * t) % 1 * tableSize, np.arange(tableSize + 1)
            period = np.interp(phase, grid, table.real) + 1j * np.interp(phase, grid, table.imag)
            wave = np.real(np.exp(2j * np.pi * offset * freq * t) * period)

        envelope = np.minimum(t / attackTime, 1) * np.exp(-t / decayTime)
        envelope[t > held] *= np.exp(-(t[t > held] - held) / releaseTime)
        out[i0:i0 + len(t)] += velocity / 127 * envelope * wave / np.abs(c).sum()

    fade = int(min(fadeOutTime, nSeconds / 4) * sampleRate)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out

if __name__ == '__main__':
    midiPath = sys.argv[1] if len(sys.argv) > 1 else defaultMidiPath
    notes = readNotes(midiPath, nSeconds)
    for name, module in (('infiniteSqWell', infiniteSquareWell), ('finiteSqWell', finiteSquareWell),
                         ('harmonicOscillator', harmonicOscillator), ('doubleWell', doubleWell),
                         ('morsePotential', morsePotential), ('latticePotential', latticePotential),
                         ('linearPotential', linearPotential)):
        audio = render(notes, module.harmonicContent)
        write(f'./audio/unSospiro_{name}.wav', sampleRate, infiniteSquareWell.scale(audio))
