// Passes what it hears back to the page while recording: [left, right] float samples
// (a mono input is sent as the same samples on both sides).
class Recorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = false;
    this.port.onmessage = event => { this.recording = event.data; };
  }

  process(inputs) {
    const input = inputs[0];
    if (this.recording && input.length) this.port.postMessage([input[0].slice(0), (input[1] || input[0]).slice(0)]);
    return true;
  }
}

registerProcessor('recorder', Recorder);
