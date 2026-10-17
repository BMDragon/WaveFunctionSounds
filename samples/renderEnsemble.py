import argparse
import json
import math
import multiprocessing
import os
import re
import sys

# The parts render in parallel on half the cores, one maths-library thread each (set before
# numpy loads)
WORKERS = max(1, (os.cpu_count() or 2) // 2)
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import mido
from scipy.io.wavfile import write

# Render a multi-track MIDI file with every part played by the WaveFunctionSounds instrument
# preset that matches it (by track name, else by General MIDI program; drum-kit notes go to the
# drum and cymbal presets). Not part of the web app.
#
# Usage: python3 samples/renderEnsemble.py piece.mid out.wav [--seconds 90] [--map "Accordion=english_horn" ...]
#        [--drum 35=kick_drum ...]   (a drum-kit note on a chosen preset, e.g. a jazz kit's kick)
#        [--mute Harpsichord ...]    (leave out the tracks whose names match)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # samples/ sits in the repo
sys.path.insert(0, os.path.join(REPO, 'backend'))
sys.path.insert(0, os.path.join(REPO, 'web'))
import server       # noqa: E402  (the web app's models, without starting it)
import unSospiro    # noqa: E402

PRESETS = {p['id']: p for p in json.load(open(os.path.join(REPO, 'web', 'static', 'presets.json')))['presets']}

# track-name keywords, most specific first
NAMES = [
    (r'piccolo', 'piccolo'), (r'contra-?bassoon|contrafagott', 'contrabassoon'), (r'bassoon|fagott', 'bassoon'),
    (r'english horn|cor anglais|englischhorn', 'english_horn'), (r'bass clar', 'bass_clarinet'),
    (r'(e-?flat|eb|e♭|es)[ -]?clar', 'eb_clarinet'), (r'clar', 'bb_clarinet'), (r'oboe', 'oboe'), (r'flute|flöte', 'flute'),
    (r'soprano sax', 'soprano_sax'), (r'alto sax|sax.*alto', 'alto_sax'), (r'tenor sax|sax.*tenor', 'tenor_sax'),
    (r'bari(tone)? sax|sax.*bari', 'baritone_sax'), (r'sax', 'alto_sax'),
    (r'bass trombone', 'bass_trombone'), (r'trombone|posaune', 'tenor_trombone'),
    (r'trumpet|cornet|flugel|trompete', 'trumpet'), (r'tuba|euphonium', 'tuba'), (r'horn', 'french_horn'),
    (r'timpani|pauken', 'timpani'), (r'snare', 'snare_drum'), (r'bass drum|gran cassa', 'bass_drum'), (r'kick', 'kick_drum'),
    (r'hi-?hat', 'hi_hat'), (r'ride', 'ride_cymbal'), (r'cymbal|piatti|becken', 'crash_cymbals'), (r'triangle', 'triangle'),
    (r'glock', 'glockenspiel'), (r'xylo', 'xylophone'), (r'marimba', 'marimba'), (r'vibra', 'vibraphone'),
    (r'chime|tubular', 'chimes'), (r'harpsichord|cembalo|clavecin', 'harpsichord'),
    (r'harp', 'harp'), (r'celest', 'celesta'), (r'piano|klavier', 'piano'),
    (r'electric bass|bass guitar|e-?bass', 'electric_bass'), (r'guitar|gitarre', 'guitar'),
    (r'cello|violoncello', 'cello'), (r'viola|bratsche', 'viola'), (r'violin|violine|geige', 'violin'),
    (r'contrabass|double bass|kontrabass|^basses?$|string bass', 'double_bass'),
]

def presetForProgram(program):
    # General MIDI program (0-127) -> preset
    table = [(0, 5, 'piano'), (6, 6, 'harpsichord'), (7, 7, 'piano'), (8, 8, 'celesta'), (9, 9, 'glockenspiel'), (11, 11, 'vibraphone'), (12, 12, 'marimba'),
             (13, 13, 'xylophone'), (14, 14, 'chimes'), (24, 31, 'guitar'), (32, 32, 'double_bass'), (33, 39, 'electric_bass'),
             (40, 40, 'violin'), (41, 41, 'viola'), (42, 42, 'cello'), (43, 43, 'double_bass'), (44, 45, 'violin'),
             (46, 46, 'harp'), (47, 47, 'timpani'), (48, 51, 'violin'), (56, 56, 'trumpet'), (57, 57, 'tenor_trombone'),
             (58, 58, 'tuba'), (59, 59, 'trumpet'), (60, 60, 'french_horn'), (61, 63, 'trumpet'), (64, 64, 'soprano_sax'),
             (65, 65, 'alto_sax'), (66, 66, 'tenor_sax'), (67, 67, 'baritone_sax'), (68, 68, 'oboe'), (69, 69, 'english_horn'),
             (70, 70, 'bassoon'), (71, 71, 'bb_clarinet'), (72, 72, 'piccolo'), (73, 75, 'flute')]
    return next((p for lo, hi, p in table if lo <= program <= hi), None)

def presetForDrum(note):
    # General MIDI percussion key -> preset
    if note in (35, 36, 41, 43):        # kick and the low floor toms: an orchestra's bass drum
        return 'bass_drum'
    if note in (37, 38, 39, 40):
        return 'snare_drum'
    if note in (42, 44, 46):
        return 'hi_hat'
    if note in (51, 53, 59):
        return 'ride_cymbal'
    if note in (49, 52, 55, 57):
        return 'crash_cymbals'
    if note in (80, 81):
        return 'triangle'
    return 'snare_drum'

def readParts(path):
    # {(track index, channel): {'name', 'program', 'notes': [(start, end, note, velocity)]}}, times in
    # seconds on the file's tempo map; sustain-pedal (CC 64) held notes last until the pedal lifts
    midi = mido.MidiFile(path)
    tempo, ticks = [(0, 500000)], 0
    for msg in mido.merge_tracks(midi.tracks):
        ticks += msg.time
        if msg.type == 'set_tempo':
            tempo.append((ticks, msg.tempo))
    def seconds(tick):
        t, last, current = 0.0, 0, 500000
        for at, value in tempo:
            if at >= tick:
                break
            t += mido.tick2second(at - last, midi.ticks_per_beat, current)
            last, current = at, value
        return t + mido.tick2second(tick - last, midi.ticks_per_beat, current)
    parts = {}
    for index, track in enumerate(midi.tracks):
        name = next((m.name for m in track if m.type == 'track_name'), '')
        tick, held, pedal, sustained = 0, {}, {}, {}
        for msg in track:
            tick += msg.time
            if not hasattr(msg, 'channel'):
                continue
            part = parts.setdefault((index, msg.channel), {'name': name, 'program': None, 'notes': []})
            if msg.type == 'program_change' and part['program'] is None:
                part['program'] = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                held.setdefault((msg.channel, msg.note), []).append((seconds(tick), msg.velocity))
            elif msg.type in ('note_on', 'note_off') and held.get((msg.channel, msg.note)):
                start, velocity = held[(msg.channel, msg.note)].pop(0)
                if pedal.get(msg.channel):
                    sustained.setdefault(msg.channel, []).append((start, msg.note, velocity))
                else:
                    part['notes'].append((start, seconds(tick), msg.note, velocity))
            elif msg.type == 'control_change' and msg.control == 64:
                pedal[msg.channel] = msg.value >= 64
                if not pedal[msg.channel]:
                    part['notes'] += [(s, seconds(tick), n, v) for s, n, v in sustained.pop(msg.channel, [])]
    return {k: v for k, v in parts.items() if v['notes']}

def choosePreset(part, channel, overrides):
    for pattern, preset in overrides:
        if re.search(pattern, part['name'], re.I):
            return preset, 'override'
    if channel == 9:                    # the drum channel: each hit by its note
        return 'drums', 'drum kit'
    for pattern, preset in NAMES:
        if re.search(pattern, part['name'], re.I):
            return preset, 'name'
    if part['program'] is not None and presetForProgram(part['program']):
        return presetForProgram(part['program']), 'GM program'
    return 'default', 'no preset fits'

def renderPart(notes, presetId, nSeconds):
    preset = PRESETS[presetId]
    # the preset as the web app reads it, including the hydrogen atom's Z and l and the sound
    req = dict(preset, Z=preset.get('charge', 1), l=preset.get('ell', 0))
    model = server.requestModel(req)
    content = lambda f, sr: server.content(model, req, f, sr, req['weighting'])
    keys = preset.get('keys')
    if keys and keys[0] == keys[1]:         # unpitched: every hit sounds at the pitch it was matched at
        notes = [(s, e, keys[0], v) for s, e, n, v in notes]
    return unSospiro.render(notes, content, nSeconds=nSeconds, attackTime=preset['attack'], decayTime=preset['decay'],
                            releaseTime=preset['release'], sustainLevel=preset['sustain'])

def renderJob(job):
    return renderPart(*job)

def main():
    parser = argparse.ArgumentParser(description='Render a MIDI file with the WaveFunctionSounds instrument presets')
    parser.add_argument('midi')
    parser.add_argument('out')
    parser.add_argument('--seconds', type=float, default=None, help='length to render (default: the whole piece)')
    parser.add_argument('--map', action='append', default=[], help='"track-name regex=preset id" to override a match')
    parser.add_argument('--drum', action='append', default=[], help='"note=preset id": a drum-kit note on that preset')
    parser.add_argument('--mute', action='append', default=[], help='track-name regex: leave those tracks out')
    args = parser.parse_args()
    overrides = [tuple(m.split('=', 1)) for m in args.map]
    drums = {int(note): preset for note, preset in (d.split('=', 1) for d in args.drum)}
    parts = readParts(args.midi)
    first = min(n[0] for p in parts.values() for n in p['notes'])
    last = max(n[1] for p in parts.values() for n in p['notes']) - first
    nSeconds = min(args.seconds or last + 3, last + 3)
    jobs = []                           # (notes, preset id) for each part, or each drum of a drum track
    for (index, channel), part in sorted(parts.items()):
        notes = [(s - first, e - first, n, v) for s, e, n, v in part['notes'] if s - first < nSeconds]
        if not notes:
            continue
        if any(re.search(pattern, part['name'], re.I) for pattern in args.mute):
            print(f'{part["name"] or "track " + str(index):<28} -> (muted)')
            continue
        presetId, why = choosePreset(part, channel, overrides)
        if presetId == 'drums':
            byPreset = {}
            for note in notes:
                byPreset.setdefault(drums.get(note[2]) or presetForDrum(note[2]), []).append(note)
            jobs += [(hits, drum, nSeconds) for drum, hits in byPreset.items()]
            print(f'{part["name"] or "track " + str(index):<28} -> drum kit: {", ".join(sorted(byPreset))} ({len(notes)} hits)')
            continue
        jobs.append((notes, presetId, nSeconds))
        print(f'{part["name"] or "track " + str(index):<28} -> {PRESETS[presetId]["name"]:<16} ({why}; {len(notes)} notes)')
    # the parts in parallel, added up in their order (so the result doesn't depend on which
    # finishes first)
    mix = np.zeros(int(nSeconds * unSospiro.sampleRate))
    with multiprocessing.Pool(min(WORKERS, len(jobs))) as pool:
        for audio in pool.imap(renderJob, jobs):
            mix += audio
    write(args.out, unSospiro.sampleRate, np.int16(mix / np.abs(mix).max() * 0.95 * 32767))
    print(f'wrote {args.out}: {nSeconds:.1f} s')

if __name__ == '__main__':
    main()
