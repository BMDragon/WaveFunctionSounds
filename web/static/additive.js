// A bank of sine oscillators for partials at any frequencies (the 2D boxes' modes):
// out(t) = Re(sum c_k exp(2 pi i f_k t)). Each partial is a phasor turned by a fixed rotation
// every sample, renormalized every block so rounding never builds up.
class Additive extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { freqs, re, im } = options.processorOptions;
    const n = freqs.length;
    this.zr = Float64Array.from(re);              // c_k exp(2 pi i f_k t), starting at t = 0
    this.zi = Float64Array.from(im);
    this.size = Float64Array.from(re, (r, k) => Math.hypot(r, im[k]));
    this.cr = new Float64Array(n);
    this.ci = new Float64Array(n);
    freqs.forEach((f, k) => {
      this.cr[k] = Math.cos(2 * Math.PI * f / sampleRate);
      this.ci[k] = Math.sin(2 * Math.PI * f / sampleRate);
    });
    this.stopAt = Infinity;
    this.port.onmessage = event => { this.stopAt = event.data; };
  }

  process(inputs, outputs) {
    if (currentTime >= this.stopAt) return false;
    const out = outputs[0][0];
    const { zr, zi, cr, ci, size } = this;
    out.fill(0);
    for (let k = 0; k < zr.length; k++) {
      let r = zr[k], i = zi[k];
      const a = cr[k], b = ci[k];
      for (let j = 0; j < out.length; j++) {
        out[j] += r;
        const t = r * a - i * b;
        i = r * b + i * a;
        r = t;
      }
      const scale = size[k] / (Math.hypot(r, i) || 1);
      zr[k] = r * scale;
      zi[k] = i * scale;
    }
    return true;
  }
}

registerProcessor('additive', Additive);
