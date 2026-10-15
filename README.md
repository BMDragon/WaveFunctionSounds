# WaveFunctionSounds
Turning Quantum Wavefunctions into Sound

## Web app
Play with the potentials and sound parameters, with wavefunction, spectrum and live spectrogram views:

```
python3 web/server.py        # then open http://localhost:8000
```

Install the dependencies with `pip install -r requirements.txt`.

In the app: pick an instrument preset (or share any sound with its link), play the on-screen keyboard,
your computer keyboard or a MIDI keyboard, record what you play to WAV, and render Un sospiro or any
MIDI file of your own with the current settings.

## Potentials
Each script in `backend/` writes a 220 Hz sample to `audio/` when run, and `backend/unSospiro.py`
renders the opening of Un sospiro with every potential.

- `infiniteSquareWell.py`, `finiteSquareWell.py`, `harmonicOscillator.py`: solved analytically.
- `doubleWell.py`: two wells and a barrier; below the barrier the levels pair up by tunnelling.
- `morsePotential.py`: a molecular bond, lopsided, with levels crowding towards dissociation.
- `linearPotential.py`: V-shaped |x|, with Airy-function states.
- `latticePotential.py`: a Kronig-Penney crystal; Bloch phases other than 0 and pi give inharmonic partials.

The double well, Morse and V-shaped potentials share a finite-difference solver in `numericalPotential.py`.

## Tests
`python3 tests/test_potentials.py` checks every potential across the web app's slider ranges
(`--quick` for a smaller sweep). GitHub runs the quick sweep on every push.

`midi/unSospiro.mid` is Frederic Lamond's Welte piano roll of Liszt's Un sospiro, scanned by Terry
Smythe and shared non-commercially via https://www.kunstderfuge.com/liszt.htm.
