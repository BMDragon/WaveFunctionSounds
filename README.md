# WaveFunctionSounds
Turning Quantum Wavefunctions into Sound

Each note's timbre is a superposition of a potential's energy eigenstates: every state is a
partial, weighted thermally (the temperature in K equals the pitch in Hz unless fixed), evenly,
by 1/level, or by hand.

## Web app
```
pip install -r requirements.txt
python3 web/server.py        # then open http://localhost:8000
```

- Play the on-screen keyboard, your computer keyboard or a MIDI keyboard, with wavefunction,
  spectrum and live spectrogram views.
- Instrument presets for the orchestra, big band and wind ensemble (the closest each potential gets
  to the real instrument; see `instrumentSettings.txt`), and a link to share any sound.
- Record what you play to WAV, or render Un sospiro or your own MIDI file.
- *Sound*: the wavefunctions (each state a partial, wrapped on its own width), the wavefunctions on
  the ground state's scale (every state at its real size on the ground state's width, the wider
  ones carrying over into neighbouring units), or their time evolution: the survival probability
  |⟨ψ(0)|ψ(t)⟩|², with partials at the energy differences and E₁ − E₀ at the key.
- *Tunnelling* (double well): the particle starts in the left well and each pair of levels moves
  between the speakers at its tunnelling rate (the splitting in K, read as Hz).
- *Customize*: draw your own potential, or set the level weights by hand.

## Potentials
Each script in `backend/` writes a 220 Hz sample to `audio/` when run, and `backend/unSospiro.py`
renders the opening of Un sospiro with every potential.

- `infiniteSquareWell.py`, `finiteSquareWell.py`, `harmonicOscillator.py`: solved analytically.
- `doubleWell.py`: two wells and a barrier; below the barrier the levels pair up by tunnelling.
- `morsePotential.py`: a molecular bond, lopsided, with levels crowding towards dissociation; solved
  exactly (Laguerre-function states).
- `linearPotential.py`: V-shaped |x|, solved exactly with Airy-function states.
- `latticePotential.py`: a Kronig-Penney crystal; Bloch phases other than 0 and π give inharmonic partials.
- `box2D.py`: circular and rectangular drums; each state sounds at its wavenumber, so the partials
  are inharmonic (Bessel zeros for the circle).
- `hydrogen.py`: a hydrogen-like atom with nuclear charge Z and angular momentum l.
- `customPotential.py`: a drawn potential between hard walls.

The double well and drawn potentials are solved numerically on a grid (`numericalPotential.py`), and
the other potentials exactly; `timeEvolution.py` gives the survival-probability sound.

## Samples
`samples/` holds pieces rendered with every part on its instrument preset, by
`samples/renderEnsemble.py piece.mid out.wav`:
- Brahms, Academic Festival Overture (orchestral MIDI sequenced by David Siu)
- Dvořák, Slavonic Dance in F, Op. 46 No. 4 (orchestral MIDI; one stray flute note after the final
  chord removed)
- Handel, "La Réjouissance" from Music for the Royal Fireworks (oboes, bassoon, horns, trumpets,
  timpani, strings, and harpsichord continuo on the harp preset, its strings being plucked)
- Fats Waller, "Ain't Misbehavin'" (1929, public domain in the US; bass, piano, guitar, vibraphone
  and drums, rendered with `--drum 35=kick_drum`)

The first three MIDI files are from https://www.midiworld.com, the last from
https://bitmidi.com/aint-misbehavin-mid.

## Tests
`python3 tests/test_potentials.py` checks every potential across the web app's slider ranges
(`--quick` for a smaller sweep). GitHub runs the quick sweep on every push.

`midi/unSospiro_rollScan.mid` is Frederic Lamond's Welte piano roll of Liszt's Un sospiro, scanned by
Terry Smythe and shared non-commercially via https://www.kunstderfuge.com/liszt.htm. `midi/unSospiro.mid`
is the same performance with the scan's note-to-note jitter evened out (`midi/evenTiming.py`).
