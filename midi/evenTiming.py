import sys
import numpy as np
import mido

# Evens out the note-to-note timing of a piano-roll scan. The scan of Lamond's Welte roll of
# Un sospiro is uneven from one note to the next (punching and scanning jitter on top of his
# rubato): in running figuration that should be even, consecutive gaps swing between 15 and
# 150 ms. Each onset is moved to a local straight-line fit through its neighbours, which keeps
# the slower shape of the rubato and removes the jitter. Only fast figuration is touched (where
# the typical gap is under 0.2 s), never across a pause, never by more than 45% of the typical
# gap, and notes keep their order and length; slow passages, chords (notes within 8 ms), the
# pedal and everything else stay as played.
#
# Usage: python3 midi/evenTiming.py midi/unSospiro_rollScan.mid midi/unSospiro.mid

def smoothOnsets(times, chord=0.008, half=3, pause=2.5, dense=0.2, cap=0.45):
    # {original onset time: shift in seconds}
    times = np.unique(np.round(times, 6))
    groups = [[times[0]]]
    for x in times[1:]:
        if x - groups[-1][0] < chord:
            groups[-1].append(x)
        else:
            groups.append([x])
    g = np.array([np.mean(c) for c in groups])
    gaps = np.diff(g)
    local = np.array([np.median(gaps[max(0, i - 6):i + 7]) for i in range(len(gaps))])
    breaks = np.nonzero(gaps > pause * local)[0]
    smooth = g.copy()
    start = 0
    for end in list(breaks + 1) + [len(g)]:
        segment = g[start:end]
        for i in range(1, len(segment) - 1):
            j = start + i
            typical = min(local[j - 1], local[min(j, len(local) - 1)])
            if typical > dense:
                continue                                    # a slow passage: leave it as played
            lo, hi = max(0, i - half), min(len(segment), i + half + 1)
            slope, intercept = np.polyfit(np.arange(lo, hi), segment[lo:hi], 1)
            smooth[j] = g[j] + np.clip(slope * i + intercept - g[j], -cap * typical, cap * typical)
        start = end
    assert np.all(np.diff(smooth) > 0), 'notes would change order'
    return {round(x, 6): new - old for c, new, old in zip(groups, smooth, g) for x in c}

def main(source, target):
    midi = mido.MidiFile(source)
    t, events = 0.0, []
    for msg in midi:                                 # every track merged, times in seconds
        t += msg.time
        events.append((t, msg))
    shift = smoothOnsets([t for t, m in events if m.type == 'note_on' and m.velocity])
    moved, sounding = [], {}
    for t, msg in events:
        if msg.type == 'note_on' and msg.velocity:
            d = shift[round(t, 6)]
            sounding.setdefault((msg.channel, msg.note), []).append(d)
            t += d
        elif msg.type in ('note_on', 'note_off') and sounding.get((msg.channel, msg.note)):
            t += sounding[(msg.channel, msg.note)].pop(0)  # the note keeps its length
        if msg.type not in ('set_tempo', 'end_of_track'):
            moved.append((t, msg))
    # one track at 120 bpm with fine ticks; a note's end sorts before a new start at the same time
    out = mido.MidiFile(ticks_per_beat=960)
    track = mido.MidiTrack()
    out.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    order = lambda m: 0 if m.is_meta else 1 if m.type == 'note_off' or (m.type == 'note_on' and not m.velocity) else 2
    last = 0
    for t, msg in sorted(moved, key=lambda e: (round(e[0] * 1920), order(e[1]))):
        tick = round(t * 1920)
        track.append(msg.copy(time=tick - last))
        last = tick
    out.save(target)
    print(f'{len(shift)} onsets; median shift {np.median(np.abs(list(shift.values()))) * 1000:.0f} ms; wrote {target}')

if __name__ == '__main__':
    main(*sys.argv[1:3])
