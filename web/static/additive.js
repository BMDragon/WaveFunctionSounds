// A bank of sine oscillators for partials at any frequencies (the 2D boxes' modes, the
// survival probability): out(t) = Re(sum c_k exp(2 pi i f_k t)). Each partial keeps its exact
// phasor at the start of every block (turned by a fixed rotation per block and renormalized, so
// rounding never builds up); within a block its samples follow from the first two by the
// cosine recurrence x[j+1] = 2 cos(w) x[j] - x[j-1]. Four partials go through the sample loop
// together, so the output is read and written once per sample for every four.
class Additive extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { freqs, re, im } = options.processorOptions;
    const count = freqs.length;
    const padded = Math.ceil(count / 4) * 4;            // zero-amplitude partials fill the last four
    this.zr = new Float64Array(padded); this.zr.set(re);  // c_k exp(2 pi i f_k t) at the block's start
    this.zi = new Float64Array(padded); this.zi.set(im);
    this.size = Float64Array.from(this.zr, (r, k) => Math.hypot(r, this.zi[k]));
    this.w = new Float64Array(padded); this.w.set(freqs.map(f => 2 * Math.PI * f / sampleRate));
    this.cr = Float64Array.from(this.w, Math.cos);
    this.ci = Float64Array.from(this.w, Math.sin);
    this.blockLength = 0;
    this.stopAt = Infinity;
    this.port.onmessage = event => { this.stopAt = event.data; };
  }

  process(inputs, outputs) {
    if (currentTime >= this.stopAt) return false;
    const out = outputs[0][0], N = out.length;
    if (N !== this.blockLength) {                      // rotation over one block, and a sum buffer
      this.blockLength = N;
      this.br = Float64Array.from(this.w, w => Math.cos(w * N));
      this.bi = Float64Array.from(this.w, w => Math.sin(w * N));
      this.acc = new Float64Array(N);
    }
    const { zr, zi, cr, ci, br, bi, size, acc } = this;
    acc.fill(0);
    for (let k = 0; k < zr.length; k += 4) {
      const t0 = 2 * cr[k], t1 = 2 * cr[k + 1], t2 = 2 * cr[k + 2], t3 = 2 * cr[k + 3];
      let p0 = zr[k], p1 = zr[k + 1], p2 = zr[k + 2], p3 = zr[k + 3];
      let c0 = zr[k] * cr[k] - zi[k] * ci[k], c1 = zr[k + 1] * cr[k + 1] - zi[k + 1] * ci[k + 1];
      let c2 = zr[k + 2] * cr[k + 2] - zi[k + 2] * ci[k + 2], c3 = zr[k + 3] * cr[k + 3] - zi[k + 3] * ci[k + 3];
      acc[0] += p0 + p1 + p2 + p3;
      if (N > 1) acc[1] += c0 + c1 + c2 + c3;
      for (let j = 2; j < N; j++) {
        const n0 = t0 * c0 - p0, n1 = t1 * c1 - p1, n2 = t2 * c2 - p2, n3 = t3 * c3 - p3;
        acc[j] += n0 + n1 + n2 + n3;
        p0 = c0; c0 = n0; p1 = c1; c1 = n1; p2 = c2; c2 = n2; p3 = c3; c3 = n3;
      }
      for (let q = k; q < k + 4; q++) {
        const r = zr[q] * br[q] - zi[q] * bi[q], i = zr[q] * bi[q] + zi[q] * br[q];
        const scale = size[q] / (Math.hypot(r, i) || 1);
        zr[q] = r * scale;
        zi[q] = i * scale;
      }
    }
    for (let j = 0; j < N; j++) out[j] = acc[j];
    return true;
  }
}

registerProcessor('additive', Additive);
