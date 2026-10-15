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
  With *Tunnelling* on (web app), the particle starts in the left well: each pair is heard on the
  left with the probability of finding it there, (1 + cos 2π·rate·t)/2, and on the right otherwise,
  where the rate is the pair's splitting in K read as Hz (as T in K = pitch in Hz).
- `morsePotential.py`: a molecular bond, lopsided, with levels crowding towards dissociation.
- `linearPotential.py`: V-shaped |x|, with Airy-function states.
- `latticePotential.py`: a Kronig-Penney crystal; Bloch phases other than 0 and pi give inharmonic partials.
- `box2D.py`: 2D boxes (circular and rectangular drums). As in the 1D well, each state is one partial
  at its wavenumber relative to the ground state's, so the partials are inharmonic (Bessel zeros for
  the circle, π√((a/W)² + (b/H)²) for the rectangle).

In the web app, *Sound* can instead play the time evolution: the particle starts in the
superposition with the chosen weights, and what is heard is its survival probability
|⟨ψ(0)|ψ(t)⟩|², whose partials are at the energy differences, with E₁ − E₀ at the key
(`backend/timeEvolution.py`).

The double well, Morse and V-shaped potentials share a finite-difference solver in `numericalPotential.py`.

## Tests
`python3 tests/test_potentials.py` checks every potential across the web app's slider ranges
(`--quick` for a smaller sweep). GitHub runs the quick sweep on every push.

`midi/unSospiro.mid` is Frederic Lamond's Welte piano roll of Liszt's Un sospiro, scanned by Terry
Smythe and shared non-commercially via https://www.kunstderfuge.com/liszt.htm.
