/**
 * K-quant encode/decode tests for gguf-quantization.js
 *
 * Run: node test/gguf-kquant.test.js
 */

import { GGUFQuantizer, GGUFQuantType } from '../src/quantization/gguf-quantization.js';

// ─── minimal harness ─────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    console.log(`  PASS  ${name}`);
    passed++;
  } catch (e) {
    console.log(`  FAIL  ${name}`);
    console.log(`        ${e.message}`);
    failures.push({ name, error: e.message });
    failed++;
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function approxEqual(a, b, tol) {
  return Math.abs(a - b) <= tol;
}

// ─── Seeded LCG for deterministic pseudo-random data ─────────────────────────
// LCG parameters: Numerical Recipes / glibc lcg
function makeLCG(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000; // [0, 1)
  };
}

function lcgFloat32Array(n, seed, lo = -1, hi = 1) {
  const rng = makeLCG(seed);
  const arr = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    arr[i] = lo + rng() * (hi - lo);
  }
  return arr;
}

// Root-mean-square error
function rmsError(a, b) {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum / len);
}

// ─── Float16 round-trip ───────────────────────────────────────────────────────

const q = new GGUFQuantizer();

test('Float16 round-trip: zero', () => {
  const buf = new Uint8Array(4);
  q.writeFloat16(buf, 0, 0.0);
  const v = q.readFloat16(buf, 0);
  assert(v === 0.0, `expected 0, got ${v}`);
});

test('Float16 round-trip: 1.0', () => {
  const buf = new Uint8Array(4);
  q.writeFloat16(buf, 0, 1.0);
  const v = q.readFloat16(buf, 0);
  assert(approxEqual(v, 1.0, 1e-3), `expected 1.0, got ${v}`);
});

test('Float16 round-trip: negative', () => {
  const buf = new Uint8Array(4);
  q.writeFloat16(buf, 0, -3.141);
  const v = q.readFloat16(buf, 0);
  assert(approxEqual(v, -3.141, 0.05), `expected ~-3.141, got ${v}`);
});

// ─── Q4_K block layout ────────────────────────────────────────────────────────
// 256 weights → 144 bytes: 2(d) + 2(dmin) + 12(scales) + 128(nibbles)

test('Q4_K: block byte length is 144 for 256 weights', () => {
  const q4k = new GGUFQuantizer({ quantType: GGUFQuantType.Q4_K });
  const data = lcgFloat32Array(256, 42);
  const result = q4k.quantize(data, [256]);
  assert(result.data.length === 144, `expected 144, got ${result.data.length}`);
});

test('Q4_K: block byte length for 512 weights is 288', () => {
  const q4k = new GGUFQuantizer({ quantType: GGUFQuantType.Q4_K });
  const data = lcgFloat32Array(512, 7);
  const result = q4k.quantize(data, [512]);
  assert(result.data.length === 288, `expected 288, got ${result.data.length}`);
});

test('Q4_K: round-trip RMS error < 0.08 (256 weights, range [-1,1])', () => {
  const q4k = new GGUFQuantizer({ quantType: GGUFQuantType.Q4_K });
  const original = lcgFloat32Array(256, 1234);
  const result = q4k.quantize(original, [256]);
  const recon = q4k.dequantize(result.data, result);
  const rms = rmsError(original, recon);
  assert(rms < 0.08, `RMS error too large: ${rms}`);
});

test('Q4_K: round-trip RMS error < 0.08 (512 weights)', () => {
  const q4k = new GGUFQuantizer({ quantType: GGUFQuantType.Q4_K });
  const original = lcgFloat32Array(512, 9999);
  const result = q4k.quantize(original, [512]);
  const recon = q4k.dequantize(result.data, result);
  const rms = rmsError(original, recon);
  assert(rms < 0.08, `RMS error too large: ${rms}`);
});

test('Q4_K: hand-verifiable block — alternating -0.5 / +0.5', () => {
  // Weights alternate -0.5 and +0.5; sub-block range = 1.0, scale = 1/15 ≈ 0.067.
  // K-quant min-offset encoding: min=-0.5, subScale=1/15, q∈[0,15]
  // Expected: dequant within ~0.07 of original.
  const data = new Float32Array(256);
  for (let i = 0; i < 256; i++) data[i] = i % 2 === 0 ? -0.5 : 0.5;
  const q4k = new GGUFQuantizer({ quantType: GGUFQuantType.Q4_K });
  const result = q4k.quantize(data, [256]);
  const recon = q4k.dequantize(result.data, result);
  const rms = rmsError(data, recon);
  assert(rms < 0.08, `RMS for alternating input should be low: ${rms}`);
});

// ─── Q5_K block layout ────────────────────────────────────────────────────────
// 256 weights → 176 bytes: 2+2+12+32+128

test('Q5_K: block byte length is 176 for 256 weights', () => {
  const q5k = new GGUFQuantizer({ quantType: GGUFQuantType.Q5_K });
  const data = lcgFloat32Array(256, 55);
  const result = q5k.quantize(data, [256]);
  assert(result.data.length === 176, `expected 176, got ${result.data.length}`);
});

test('Q5_K: block byte length for 512 weights is 352', () => {
  const q5k = new GGUFQuantizer({ quantType: GGUFQuantType.Q5_K });
  const data = lcgFloat32Array(512, 77);
  const result = q5k.quantize(data, [512]);
  assert(result.data.length === 352, `expected 352, got ${result.data.length}`);
});

test('Q5_K: round-trip RMS error < 0.05 (256 weights)', () => {
  const q5k = new GGUFQuantizer({ quantType: GGUFQuantType.Q5_K });
  const original = lcgFloat32Array(256, 2024);
  const result = q5k.quantize(original, [256]);
  const recon = q5k.dequantize(result.data, result);
  const rms = rmsError(original, recon);
  assert(rms < 0.05, `RMS error too large: ${rms}`);
});

test('Q5_K: round-trip RMS error < 0.05 (512 weights)', () => {
  const q5k = new GGUFQuantizer({ quantType: GGUFQuantType.Q5_K });
  const original = lcgFloat32Array(512, 3141);
  const result = q5k.quantize(original, [512]);
  const recon = q5k.dequantize(result.data, result);
  const rms = rmsError(original, recon);
  assert(rms < 0.05, `RMS error too large: ${rms}`);
});

test('Q5_K: Q5_K is more accurate than Q4_K on the same data', () => {
  // Both should work but Q5_K has one more bit → lower RMS expected
  const original = lcgFloat32Array(256, 8675309);
  const q4k = new GGUFQuantizer({ quantType: GGUFQuantType.Q4_K });
  const q5k = new GGUFQuantizer({ quantType: GGUFQuantType.Q5_K });

  const r4 = q4k.quantize(original, [256]);
  const r5 = q5k.quantize(original, [256]);

  const rms4 = rmsError(original, q4k.dequantize(r4.data, r4));
  const rms5 = rmsError(original, q5k.dequantize(r5.data, r5));

  assert(rms5 <= rms4 + 0.01, `Q5_K (${rms5}) should be ≤ Q4_K (${rms4}) RMS`);
});

// ─── Q6_K block layout ────────────────────────────────────────────────────────
// 256 weights → 210 bytes: 128(ql) + 64(qh) + 16(scales) + 2(d)

test('Q6_K: block byte length is 210 for 256 weights', () => {
  const q6k = new GGUFQuantizer({ quantType: GGUFQuantType.Q6_K });
  const data = lcgFloat32Array(256, 100);
  const result = q6k.quantize(data, [256]);
  assert(result.data.length === 210, `expected 210, got ${result.data.length}`);
});

test('Q6_K: block byte length for 512 weights is 420', () => {
  const q6k = new GGUFQuantizer({ quantType: GGUFQuantType.Q6_K });
  const data = lcgFloat32Array(512, 200);
  const result = q6k.quantize(data, [512]);
  assert(result.data.length === 420, `expected 420, got ${result.data.length}`);
});

test('Q6_K: round-trip RMS error < 0.03 (256 weights)', () => {
  const q6k = new GGUFQuantizer({ quantType: GGUFQuantType.Q6_K });
  const original = lcgFloat32Array(256, 2026);
  const result = q6k.quantize(original, [256]);
  const recon = q6k.dequantize(result.data, result);
  const rms = rmsError(original, recon);
  assert(rms < 0.03, `RMS error too large: ${rms}`);
});

test('Q6_K: round-trip RMS error < 0.03 (512 weights)', () => {
  const q6k = new GGUFQuantizer({ quantType: GGUFQuantType.Q6_K });
  const original = lcgFloat32Array(512, 4096);
  const result = q6k.quantize(original, [512]);
  const recon = q6k.dequantize(result.data, result);
  const rms = rmsError(original, recon);
  assert(rms < 0.03, `RMS error too large: ${rms}`);
});

test('Q6_K: symmetric format — zero input stays near zero', () => {
  const data = new Float32Array(256).fill(0);
  const q6k = new GGUFQuantizer({ quantType: GGUFQuantType.Q6_K });
  const result = q6k.quantize(data, [256]);
  const recon = q6k.dequantize(result.data, result);
  for (let i = 0; i < recon.length; i++) {
    assert(Math.abs(recon[i]) < 1e-6, `non-zero at [${i}]: ${recon[i]}`);
  }
});

test('Q6_K: accuracy improves over Q5_K on same data', () => {
  const original = lcgFloat32Array(256, 31415926);
  const q5k = new GGUFQuantizer({ quantType: GGUFQuantType.Q5_K });
  const q6k = new GGUFQuantizer({ quantType: GGUFQuantType.Q6_K });

  const r5 = q5k.quantize(original, [256]);
  const r6 = q6k.quantize(original, [256]);

  const rms5 = rmsError(original, q5k.dequantize(r5.data, r5));
  const rms6 = rmsError(original, q6k.dequantize(r6.data, r6));

  assert(rms6 <= rms5 + 0.01, `Q6_K (${rms6}) should be ≤ Q5_K (${rms5}) RMS`);
});

// ─── Scales packing round-trip ────────────────────────────────────────────────

test('_encodeScalesMins / _decodeScalesMins round-trip', () => {
  const sc = [5, 10, 63, 0, 25, 1, 47, 31];
  const mn = [0, 63, 7, 15, 32, 60, 3, 55];

  const buf = new Uint8Array(12);
  q._encodeScalesMins(buf, 0, sc, mn);
  const { sc: sc2, mn: mn2 } = q._decodeScalesMins(buf, 0);

  for (let i = 0; i < 8; i++) {
    assert(sc2[i] === sc[i], `sc[${i}]: encoded ${sc[i]} decoded ${sc2[i]}`);
    assert(mn2[i] === mn[i], `mn[${i}]: encoded ${mn[i]} decoded ${mn2[i]}`);
  }
});

// ─── Summary ─────────────────────────────────────────────────────────────────

console.log();
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failures.length > 0) {
  console.log('\nFailed tests:');
  for (const f of failures) {
    console.log(`  - ${f.name}: ${f.error}`);
  }
  process.exit(1);
}
