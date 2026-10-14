# WaveFunctionSounds
Turning Quantum Wavefunctions into Sound

## Web app
Play with the potentials and sound parameters, with wavefunction, spectrum and live spectrogram views:

```
python3 web/server.py        # then open http://localhost:8000
```

Install the dependencies with `pip install -r requirements.txt`.

## Potentials
Each script in `backend/` writes a 220 Hz sample to `audio/` when run, and `backend/unSospiro.py`
renders the opening of Un sospiro with every potential.

- `infiniteSquareWell.py`, `finiteSquareWell.py`, `harmonicOscillator.py`: solved analytically.
- `doubleWell.py`: two wells and a barrier; below the barrier the levels pair up by tunnelling.
- `morsePotential.py`: a molecular bond, lopsided, with levels crowding towards dissociation.
- `linearPotential.py`: V-shaped |x|, with Airy-function states.
- `latticePotential.py`: a Kronig-Penney crystal; Bloch phases other than 0 and pi give inharmonic partials.

The double well, Morse and V-shaped potentials share a finite-difference solver in `numericalPotential.py`.

`midi/unSospiro.mid` is Frederic Lamond's Welte piano roll of Liszt's Un sospiro, scanned by Terry
Smythe and shared non-commercially via https://www.kunstderfuge.com/liszt.htm.
