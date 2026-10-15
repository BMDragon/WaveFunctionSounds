# To do

Worked through in this order. Customization features stay less prominent in the app than the presets.
Design rule: choices follow from the physics, never from how music or instruments work.

## Housekeeping
- [x] Run `tests/test_potentials.py --quick` on GitHub on every push and pull request

## Easy
- [x] Full ADSR envelope: a sustain level and attacks longer than 300 ms
- [x] Instrument presets (prominent) and shareable links that reopen a sound
- [x] Play from a MIDI keyboard (Web MIDI)
- [x] Render any MIDI file, and record live playing to WAV

## Presets
- [x] Re-run the instrument analysis (several notes per instrument, all potentials)
- [x] Presets for every instrument of a symphony orchestra, jazz big band and wind ensemble that has
      a free recording (40)
- Alto clarinet, contrabass clarinet, cornet, flugelhorn, euphonium: left out, no free recording to match against

## Medium
- [x] Draw your own potential (in a collapsed Customize section)
- [x] Hand-set level weights by dragging bars (Customize section)
- [x] Make tunnelling audible in the double well: well probabilities drive the stereo channels, at the
      splitting's rate (splitting in K = rate in Hz, like T = pitch)

## Bigger
- [x] 2D boxes (circle, rectangle): partials at each mode's wavenumber, as in 1D
- [x] Time-evolution mode: play the survival probability |<psi(0)|psi(t)>|^2 (partials at energy differences)
- [x] Hydrogen atom: the radial Coulomb potential with an l slider, as another potential
