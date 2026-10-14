# WaveFunctionSounds
Turning Quantum Wavefunctions into Sound

## Web app
Play with the potentials and sound parameters, with wavefunction, spectrum and live spectrogram views:

```
python3 web/server.py        # then open http://localhost:8000
```

Install the dependencies with `pip install -r requirements.txt`. Rendering Un sospiro needs mido and
`midi/unSospiro.mid`, which isn't in the repository: download the Frederic Lamond piano roll
("Grand Etude de Concert: A Sigh") from https://www.kunstderfuge.com/liszt.htm and save it there.
