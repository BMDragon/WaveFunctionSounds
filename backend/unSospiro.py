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
sustainLevel = 0.0      # level the decay settles at while the note is held (0: decays away, like a piano)
releaseTime = 0.1       # s, fade once the key and sustain pedal are both up
fadeOutTime = 1.0       # s, fade at the end of the excerpt
silence = 1e-5          # envelope level below which a note is no longer computed
tableSize = 16384       # samples in each note's one-period table

def readNotes(path, nSeconds):
    # (start, end, midi note, velocity) for the notes beginning within nSeconds of the
    # first note, with times measured from the first note. A note released while the
    # sustain pedal is down keeps sounding until the pedal comes up. `path` can also be
    # the bytes of a MIDI file.
    import io
    import mido     # only needed here, so the web app runs without it until a render is asked for
    midi = mido.MidiFile(file=io.BytesIO(path)) if isinstance(path, (bytes, bytearray)) else mido.MidiFile(path)
    t = 0.0
    pedalDown = False
    held = {}           # note -> [(start, velocity)] of keys still down
    sustained = []      # (start, note, velocity) of notes held only by the pedal
    notes = []
    for msg in midi:
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

    if not notes:
        raise ValueError('the MIDI file has no notes')
    first = min(start for start, _, _, _ in notes)
    return sorted((start - first, end - first, note, velocity)
                  for start, end, note, velocity in notes if start - first < nSeconds)

def partialWave(m, c, freq, t):
    # Re(sum c exp(2 pi i r freq t)) for partials at multiples r of freq
    if np.allclose(m, np.round(m)):
        # one period as a fine table, Re(sum c_m exp(2 pi i m phase)), read out at the
        # note's exact pitch
        spectrum = np.zeros(tableSize // 2 + 1, complex)
        spectrum[np.round(m).astype(int)] = c
        table = np.fft.irfft(spectrum, tableSize) * tableSize / 2
        return np.interp((freq * t) % 1 * tableSize, np.arange(tableSize + 1), np.append(table, table[0]))
    offset = m[0] - math.floor(m[0])
    n = np.round(m - offset).astype(int)
    if not np.allclose(m - offset, n):
        return additiveWave(m, c, freq, t)
    # inharmonic partials at (n + offset) freq with integer n and a common offset (the
    # lattice's Bloch phase): one period of the complex sum over n as a table, turned
    # by exp(2 pi i offset freq t)
    spectrum = np.zeros(tableSize, complex)
    np.add.at(spectrum, n % tableSize, c)
    table = np.fft.ifft(spectrum) * tableSize
    table = np.append(table, table[0])
    phase, grid = (freq * t) % 1 * tableSize, np.arange(tableSize + 1)
    period = np.interp(phase, grid, table.real) + 1j * np.interp(phase, grid, table.imag)
    return np.real(np.exp(2j * np.pi * offset * freq * t) * period)

def additiveWave(m, c, freq, t, block=2048):
    # Re(sum c exp(2 pi i m freq t)) for any partial ratios m, with t evenly spaced: in blocks,
    # the phasors at each block's start times one block of phasors, as a matrix product
    if len(t) == 0:
        return np.zeros(0)
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    w = 2 * np.pi * np.asarray(m, float) * freq
    blocks = -(-len(t) // block)
    starts = t[0] + np.arange(blocks) * block * dt
    # phases in double precision, the product in single (errors ~1e-7, far below 16-bit audio)
    # exp(i w k dt) for k < block, as exp(i w 32 a dt) exp(i w b dt) with k = 32 a + b: 64 + 32
    # exponentials per partial instead of 2048
    coarse = np.exp(1j * np.outer(w, np.arange(block // 32) * 32 * dt))
    fine = np.exp(1j * np.outer(w, np.arange(32) * dt))
    within = (coarse[:, :, None] * fine[:, None, :]).reshape(len(w), block).astype(np.complex64)   # partials x block
    phasors = (np.asarray(c) * np.exp(1j * np.outer(starts, w))).astype(np.complex64)                # blocks x partials
    return (phasors @ within).real.ravel()[:len(t)].astype(float)

def audible(m, c, tolerance=1e-5):
    # drop the weakest partials whose amplitudes add up to less than `tolerance` of the total:
    # the note (scaled to a total of 1) then changes by under 1e-5 anywhere, a third of a 16-bit step
    m, c = np.asarray(m), np.asarray(c)
    order = np.argsort(np.abs(c))
    weak = np.cumsum(np.abs(c[order])) < tolerance * np.abs(c).sum()
    keep = np.ones(len(c), bool)
    keep[order[weak]] = False
    return m[keep], c[keep]

def pitchWaves(notes, content, length, attackTime, decayTime, releaseTime, sustainLevel):
    # every note's start, envelope and pitch, and each pitch's waveform computed once, as long as
    # its longest note: notes of one pitch share their partials, so they differ only in length
    timed, longest = [], {}
    for start, end, note, velocity in notes:
        i0, t, envelope = noteTimes(start, end, length, attackTime, decayTime, releaseTime, sustainLevel)
        timed.append((i0, envelope, note, velocity))
        longest[note] = max(longest.get(note, 0), len(t))
    waves = {}
    for note, n in longest.items():
        freq = 440 * 2**((note - 69) / 12)
        waves[note] = content(freq, np.arange(n) / sampleRate)
    return timed, waves

def noteTimes(start, end, length, attackTime, decayTime, releaseTime, sustainLevel):
    # first sample, time axis and envelope of one note: attack, decay towards the sustain
    # level, release; computed until the envelope falls below `silence`
    held = end - start
    if sustainLevel > 0:
        duration = held + -math.log(silence) * releaseTime
    else:
        duration = -math.log(silence) * min(decayTime, releaseTime + held / -math.log(silence))
    i0 = int(start * sampleRate)
    t = np.arange(min(length - i0, int(duration * sampleRate))) / sampleRate
    envelope = np.minimum(t / attackTime, 1) * (sustainLevel + (1 - sustainLevel) * np.exp(-t / decayTime))
    envelope[t > held] *= np.exp(-(t[t > held] - held) / releaseTime)
    return i0, t, envelope

# Level calibration: every note is played at the same perceived loudness for the same velocity,
# whatever its timbre, measured by the K-weighted RMS of ITU-R BS.1770 (a high shelf of about +4 dB
# above 2 kHz and a high-pass below about 60 Hz, here as the standard's 48 kHz filters). Notes
# scaled to peak at most 1 (as the partials' amplitudes summed) differ in that loudness by ~16 dB
# between smooth and spiky timbres; each is scaled to the median of the presets, -9.4 dB, the
# correction capped at +-20 dB.
LOUDNESS_TARGET = 10**(-9.4 / 20)
_K = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585],
      [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])

def kWeighting(f):
    # |H(f)| of the BS.1770 K-weighting filters
    from scipy.signal import freqz
    f = np.minimum(np.abs(np.asarray(f, float)), 23999)
    return np.abs(freqz(_K[0], _K[1], worN=f, fs=48000)[1] * freqz(_K[2], _K[3], worN=f, fs=48000)[1])

def loudnessGain(m, c, freq):
    # the factor that brings a note, scaled to sum(|c|) = 1, to the target K-weighted RMS
    m, c = np.asarray(m, float), np.asarray(c)
    total = np.abs(c).sum()
    if not total:
        return 1.0
    krms = math.sqrt((np.abs(c)**2 * kWeighting(m * freq)**2).sum() / 2) / total
    return float(np.clip(LOUDNESS_TARGET / max(krms, 1e-12), 0.1, 10.0))

def groupsGain(groups, freq):
    # the same for a note split into groups of partials scaled together by their summed |c|: the
    # loudness of the whole note, heard from both sides
    total = sum(np.abs(c).sum() for m, c, rate in groups)
    allM = np.concatenate([np.asarray(m, float) for m, c, rate in groups])
    allC = np.concatenate([np.asarray(c) for m, c, rate in groups])
    merged, index = np.unique(np.round(allM, 9), return_inverse=True)
    whole = np.zeros(len(merged), complex); np.add.at(whole, index, allC)
    return loudnessGain(merged, whole, freq) * np.abs(whole).sum() / total if total else 1.0

def render(notes, harmonicContent, nSeconds=nSeconds, attackTime=attackTime, decayTime=decayTime,
           releaseTime=releaseTime, sustainLevel=sustainLevel):
    out = np.zeros(int(nSeconds * sampleRate))
    def content(freq, t):
        m, c = harmonicContent(freq, sampleRate)
        if not np.abs(c).sum():
            return None
        gain = loudnessGain(m, c, freq)
        m, c = audible(m, c)
        return gain * partialWave(m, c, freq, t) / np.abs(c).sum()
    timed, waves = pitchWaves(notes, content, len(out), attackTime, decayTime, releaseTime, sustainLevel)
    for i0, envelope, note, velocity in timed:
        if waves[note] is not None:
            out[i0:i0 + len(envelope)] += velocity / 127 * envelope * waves[note][:len(envelope)]
    fade = int(min(fadeOutTime, nSeconds / 4) * sampleRate)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out

def renderStereo(notes, groupContent, nSeconds=nSeconds, attackTime=attackTime, decayTime=decayTime,
                 releaseTime=releaseTime, sustainLevel=sustainLevel):
    # Stereo, from groups of partials (m, c, rate): each group sounds in the left channel with
    # the probability P_L(t) = (1 + cos(2 pi rate t)) / 2 and in the right with 1 - P_L, from the
    # start of the note; rate None means equally on both sides (the double well's tunnelling).
    out = np.zeros((int(nSeconds * sampleRate), 2))
    def content(freq, t):
        groups = groupContent(freq, sampleRate)
        total = sum(np.abs(c).sum() for m, c, rate in groups)
        gain = groupsGain(groups, freq)
        left = np.zeros(len(t)); right = np.zeros(len(t))
        for m, c, rate in groups:
            wave = gain * partialWave(*audible(m, c), freq, t) / total
            share = 0.5 if rate is None else (1 + np.cos(2 * np.pi * rate * t)) / 2
            left += share * wave
            right += (1 - share) * wave
        return np.stack([left, right], axis=1)
    timed, waves = pitchWaves(notes, content, len(out), attackTime, decayTime, releaseTime, sustainLevel)
    for i0, envelope, note, velocity in timed:
        out[i0:i0 + len(envelope)] += velocity / 127 * envelope[:, None] * waves[note][:len(envelope)]
    fade = int(min(fadeOutTime, nSeconds / 4) * sampleRate)
    out[-fade:] *= np.linspace(1, 0, fade)[:, None]
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
