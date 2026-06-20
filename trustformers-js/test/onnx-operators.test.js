/**
 * KAT (known-answer tests) for ONNX operators.
 *
 * Run: node --input-type=module < test/onnx-operators.test.js
 *   or: node test/onnx-operators.test.js  (works because this file imports as ESM)
 */

import {
  Tensor,
  MatMul,
  Gemm,
  Add,
  Mul,
  Relu,
  Sigmoid,
  Softmax,
  LayerNormalization,
  Reshape,
  Flatten,
  Gather,
  Slice,
  Transpose,
} from '../src/onnx-operators.js';

// ─── minimal test harness ────────────────────────────────────────────────────

let passed = 0;
let failed = 0;
const failures = [];

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function approxEqual(a, b, tol = 1e-4) {
  return Math.abs(a - b) <= tol;
}

function allClose(arrA, arrB, tol = 1e-4) {
  if (arrA.length !== arrB.length) return false;
  for (let i = 0; i < arrA.length; i++) {
    if (!approxEqual(arrA[i], arrB[i], tol)) return false;
  }
  return true;
}

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

// ─── helpers ─────────────────────────────────────────────────────────────────

function mkTensor(data, shape) {
  return new Tensor(new Float32Array(data), shape);
}

function mkIntTensor(data, shape) {
  // For shape / indices tensors we use Float32Array holding integers.
  return new Tensor(new Float32Array(data), shape);
}

// ─── MatMul ──────────────────────────────────────────────────────────────────

test('MatMul: 2×3 @ 3×2 exact values', () => {
  // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
  // C[0,0] = 1*7+2*9+3*11 = 7+18+33 = 58
  // C[0,1] = 1*8+2*10+3*12 = 8+20+36 = 64
  // C[1,0] = 4*7+5*9+6*11 = 28+45+66 = 139
  // C[1,1] = 4*8+5*10+6*12 = 32+50+72 = 154
  const A = mkTensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const B = mkTensor([7, 8, 9, 10, 11, 12], [3, 2]);
  const [C] = new MatMul().execute([A, B]);
  assert(JSON.stringify(Array.from(C.shape)) === JSON.stringify([2, 2]), 'shape');
  const expected = [58, 64, 139, 154];
  assert(allClose(Array.from(C.data), expected), `got ${Array.from(C.data)}`);
});

test('MatMul: batched 2×(2×3)@(3×2)', () => {
  // batch 0: A0=[[1,2,3],[4,5,6]], B0=[[1,0],[0,1],[0,0]]
  // A0*B0 = [[1,2],[4,5]]
  // batch 1: A1=[[7,8,9],[10,11,12]], B1=[[0,0],[0,0],[1,0]]
  // A1*B1 = [[9,0],[12,0]]
  const A = mkTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);
  const B = mkTensor([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [2, 3, 2]);
  const [C] = new MatMul().execute([A, B]);
  assert(JSON.stringify(Array.from(C.shape)) === JSON.stringify([2, 2, 2]), 'shape');
  // batch 0: [[1,2],[4,5]]
  assert(allClose(Array.from(C.data.slice(0, 4)), [1, 2, 4, 5]), `batch0 = ${Array.from(C.data.slice(0, 4))}`);
  // batch 1: A1*B1: row0=[7,8,9]·cols → col0=9, col1=0; row1=[10,11,12]·cols → col0=12, col1=0
  assert(allClose(Array.from(C.data.slice(4)), [9, 0, 12, 0]), `batch1 = ${Array.from(C.data.slice(4))}`);
});

// ─── Gemm ────────────────────────────────────────────────────────────────────

test('Gemm: Y = alpha*A*B + beta*C with transB=1', () => {
  // A = [[1,2],[3,4]]  B = [[1,0],[0,1]] (transposed identity → still identity)
  // Y = 1.0 * A * B^T + 0 * C = A
  const A = mkTensor([1, 2, 3, 4], [2, 2]);
  const B = mkTensor([1, 0, 0, 1], [2, 2]);
  const op = new Gemm({ alpha: 1.0, beta: 0.0, transA: 0, transB: 1 });
  const [Y] = op.execute([A, B]);
  assert(allClose(Array.from(Y.data), [1, 2, 3, 4]), `got ${Array.from(Y.data)}`);
});

test('Gemm: with bias C', () => {
  const A = mkTensor([1, 0, 0, 1], [2, 2]);
  const B = mkTensor([2, 0, 0, 2], [2, 2]);
  const C = mkTensor([1, 1, 1, 1], [2, 2]);
  const op = new Gemm({ alpha: 1.0, beta: 1.0 });
  const [Y] = op.execute([A, B, C]);
  // A*B = [[2,0],[0,2]], + C = [[3,1],[1,3]]
  assert(allClose(Array.from(Y.data), [3, 1, 1, 3]), `got ${Array.from(Y.data)}`);
});

// ─── Add (broadcasting) ──────────────────────────────────────────────────────

test('Add: element-wise same shape', () => {
  const A = mkTensor([1, 2, 3], [3]);
  const B = mkTensor([4, 5, 6], [3]);
  const [C] = new Add().execute([A, B]);
  assert(allClose(Array.from(C.data), [5, 7, 9]), `got ${Array.from(C.data)}`);
});

test('Add: broadcast (2,3)+(3,)', () => {
  // A = [[1,2,3],[4,5,6]]  B = [10,20,30]
  // Expected = [[11,22,33],[14,25,36]]
  const A = mkTensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const B = mkTensor([10, 20, 30], [3]);
  const [C] = new Add().execute([A, B]);
  assert(JSON.stringify(Array.from(C.shape)) === JSON.stringify([2, 3]), 'shape');
  assert(allClose(Array.from(C.data), [11, 22, 33, 14, 25, 36]), `got ${Array.from(C.data)}`);
});

test('Add: broadcast (1,3)+(2,1)', () => {
  const A = mkTensor([1, 2, 3], [1, 3]);
  const B = mkTensor([10, 20], [2, 1]);
  const [C] = new Add().execute([A, B]);
  // expected [[11,12,13],[21,22,23]]
  assert(allClose(Array.from(C.data), [11, 12, 13, 21, 22, 23]), `got ${Array.from(C.data)}`);
});

// ─── Mul (broadcasting) ──────────────────────────────────────────────────────

test('Mul: broadcast (2,3)*(3,)', () => {
  const A = mkTensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const B = mkTensor([2, 3, 4], [3]);
  const [C] = new Mul().execute([A, B]);
  assert(allClose(Array.from(C.data), [2, 6, 12, 8, 15, 24]), `got ${Array.from(C.data)}`);
});

// ─── Relu ────────────────────────────────────────────────────────────────────

test('Relu: basic', () => {
  const X = mkTensor([-2, -1, 0, 1, 2], [5]);
  const [Y] = new Relu().execute([X]);
  assert(allClose(Array.from(Y.data), [0, 0, 0, 1, 2]), `got ${Array.from(Y.data)}`);
});

// ─── Sigmoid ─────────────────────────────────────────────────────────────────

test('Sigmoid: known values', () => {
  // sigmoid(0) = 0.5, sigmoid(inf) → 1, sigmoid(-inf) → 0
  const X = mkTensor([0, 10, -10], [3]);
  const [Y] = new Sigmoid().execute([X]);
  assert(approxEqual(Y.data[0], 0.5), `sigmoid(0) = ${Y.data[0]}`);
  assert(Y.data[1] > 0.9999, `sigmoid(10) = ${Y.data[1]}`);
  assert(Y.data[2] < 0.0001, `sigmoid(-10) = ${Y.data[2]}`);
});

// ─── Softmax ─────────────────────────────────────────────────────────────────

test('Softmax: rows sum to 1', () => {
  const X = mkTensor([1, 2, 3, 0, 0, 0], [2, 3]);
  const [Y] = new Softmax({ axis: -1 }).execute([X]);
  const row0sum = Y.data[0] + Y.data[1] + Y.data[2];
  const row1sum = Y.data[3] + Y.data[4] + Y.data[5];
  assert(approxEqual(row0sum, 1.0, 1e-5), `row0 sum = ${row0sum}`);
  assert(approxEqual(row1sum, 1.0, 1e-5), `row1 sum = ${row1sum}`);
});

test('Softmax: numerically stable (large inputs)', () => {
  // All same large value → uniform distribution
  const X = mkTensor([1000, 1000, 1000, 1000], [4]);
  const [Y] = new Softmax().execute([X]);
  for (let i = 0; i < 4; i++) {
    assert(approxEqual(Y.data[i], 0.25, 1e-5), `softmax[${i}] = ${Y.data[i]}`);
  }
});

test('Softmax: hand-computed values [1, 2, 3]', () => {
  // e^1=2.71828, e^2=7.38906, e^3=20.08554, sum=30.19288
  // p = [0.09003057, 0.24472847, 0.66524096]
  const X = mkTensor([1, 2, 3], [3]);
  const [Y] = new Softmax().execute([X]);
  const expected = [0.09003057, 0.24472847, 0.66524096];
  assert(allClose(Array.from(Y.data), expected, 1e-5), `got ${Array.from(Y.data)}`);
});

// ─── LayerNorm ───────────────────────────────────────────────────────────────

test('LayerNorm: unit scale, zero bias', () => {
  // Hand-compute: input [1,2,3,4,5] mean=3, var=2, eps=1e-5
  // normalized = (x-3)/sqrt(2+1e-5)  ≈ [-1.414, -0.707, 0, 0.707, 1.414]
  const X = mkTensor([1, 2, 3, 4, 5], [5]);
  const scale = mkTensor([1, 1, 1, 1, 1], [5]);
  const bias = mkTensor([0, 0, 0, 0, 0], [5]);
  const op = new LayerNormalization({ epsilon: 1e-5 });
  const [Y] = op.execute([X, scale, bias]);
  // std = sqrt(2) ≈ 1.41421
  const std = Math.sqrt(2);
  const expected = [-2 / std, -1 / std, 0, 1 / std, 2 / std];
  assert(allClose(Array.from(Y.data), expected, 1e-4), `got ${Array.from(Y.data)}`);
});

test('LayerNorm: with scale and bias', () => {
  const X = mkTensor([0, 1, 2, 3], [4]);
  // mean=1.5, var = ((0-1.5)^2+(1-1.5)^2+(2-1.5)^2+(3-1.5)^2)/4 = (2.25+0.25+0.25+2.25)/4 = 1.25
  // std = sqrt(1.25+1e-5), scale=[2,2,2,2], bias=[1,1,1,1]
  const scale = mkTensor([2, 2, 2, 2], [4]);
  const bias = mkTensor([1, 1, 1, 1], [4]);
  const op = new LayerNormalization({ epsilon: 1e-5 });
  const [Y] = op.execute([X, scale, bias]);
  const std = Math.sqrt(1.25 + 1e-5);
  const norms = [-1.5 / std, -0.5 / std, 0.5 / std, 1.5 / std];
  const expected = norms.map(n => 2 * n + 1);
  assert(allClose(Array.from(Y.data), expected, 1e-4), `got ${Array.from(Y.data)}`);
});

// ─── Reshape ─────────────────────────────────────────────────────────────────

test('Reshape: -1 inference', () => {
  const data = mkTensor([1, 2, 3, 4, 5, 6], [6]);
  const shape = mkIntTensor([2, -1], [2]);
  const [out] = new Reshape().execute([data, shape]);
  assert(JSON.stringify(Array.from(out.shape)) === JSON.stringify([2, 3]), `shape = ${Array.from(out.shape)}`);
  assert(allClose(Array.from(out.data), [1, 2, 3, 4, 5, 6]), `data = ${Array.from(out.data)}`);
});

test('Reshape: 0 passthrough', () => {
  // 0 means "keep this dimension from input"
  // ONNX Reshape with 0: copy input dim. Implemented via simple -1 inference.
  const data = mkTensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const shape = mkIntTensor([3, 2], [2]);
  const [out] = new Reshape().execute([data, shape]);
  assert(JSON.stringify(Array.from(out.shape)) === JSON.stringify([3, 2]), `shape = ${Array.from(out.shape)}`);
});

// ─── Flatten ─────────────────────────────────────────────────────────────────

test('Flatten: axis=1 default', () => {
  const data = mkTensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const [out] = new Flatten().execute([data]);
  assert(JSON.stringify(Array.from(out.shape)) === JSON.stringify([2, 3]), `shape = ${Array.from(out.shape)}`);
});

test('Flatten: axis=0', () => {
  const data = mkTensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const [out] = new Flatten({ axis: 0 }).execute([data]);
  assert(JSON.stringify(Array.from(out.shape)) === JSON.stringify([1, 6]), `shape = ${Array.from(out.shape)}`);
});

test('Flatten: 4D axis=2', () => {
  const data = mkTensor(Array.from({ length: 24 }, (_, i) => i), [2, 3, 2, 2]);
  const [out] = new Flatten({ axis: 2 }).execute([data]);
  // outer = 2*3=6, inner = 2*2=4
  assert(JSON.stringify(Array.from(out.shape)) === JSON.stringify([6, 4]), `shape = ${Array.from(out.shape)}`);
});

// ─── Gather ──────────────────────────────────────────────────────────────────

test('Gather: axis=0 1D indices', () => {
  // data = [[1,2],[3,4],[5,6]], indices = [0,2]
  // output = [[1,2],[5,6]]
  const data = mkTensor([1, 2, 3, 4, 5, 6], [3, 2]);
  const indices = mkIntTensor([0, 2], [2]);
  const [out] = new Gather({ axis: 0 }).execute([data, indices]);
  assert(JSON.stringify(Array.from(out.shape)) === JSON.stringify([2, 2]), `shape = ${Array.from(out.shape)}`);
  assert(allClose(Array.from(out.data), [1, 2, 5, 6]), `got ${Array.from(out.data)}`);
});

test('Gather: axis=1', () => {
  // data = [[1,2,3],[4,5,6]], indices = [2,0]
  // output[:,0] = data[:,2] = [3,6], output[:,1] = data[:,0] = [1,4]
  const data = mkTensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const indices = mkIntTensor([2, 0], [2]);
  const [out] = new Gather({ axis: 1 }).execute([data, indices]);
  assert(JSON.stringify(Array.from(out.shape)) === JSON.stringify([2, 2]), `shape = ${Array.from(out.shape)}`);
  assert(allClose(Array.from(out.data), [3, 1, 6, 4]), `got ${Array.from(out.data)}`);
});

test('Gather: scalar indices', () => {
  // data = [10, 20, 30, 40], indices = [[1,3],[2,0]]
  // output shape = [2,2]
  const data = mkTensor([10, 20, 30, 40], [4]);
  const indices = mkIntTensor([1, 3, 2, 0], [2, 2]);
  const [out] = new Gather({ axis: 0 }).execute([data, indices]);
  assert(JSON.stringify(Array.from(out.shape)) === JSON.stringify([2, 2]), `shape = ${Array.from(out.shape)}`);
  assert(allClose(Array.from(out.data), [20, 40, 30, 10]), `got ${Array.from(out.data)}`);
});

// ─── Slice ───────────────────────────────────────────────────────────────────

test('Slice: 1D basic', () => {
  const data = mkTensor([0, 1, 2, 3, 4, 5], [6]);
  const starts = mkIntTensor([1], [1]);
  const ends = mkIntTensor([4], [1]);
  const [out] = new Slice().execute([data, starts, ends]);
  assert(allClose(Array.from(out.data), [1, 2, 3]), `got ${Array.from(out.data)}`);
});

test('Slice: 2D on axis 1', () => {
  // data = [[0,1,2],[3,4,5]], slice axis=1 from 1 to 3 → [[1,2],[4,5]]
  const data = mkTensor([0, 1, 2, 3, 4, 5], [2, 3]);
  const starts = mkIntTensor([1], [1]);
  const ends = mkIntTensor([3], [1]);
  const axes = mkIntTensor([1], [1]);
  const [out] = new Slice().execute([data, starts, ends, axes]);
  assert(JSON.stringify(Array.from(out.shape)) === JSON.stringify([2, 2]), `shape = ${Array.from(out.shape)}`);
  assert(allClose(Array.from(out.data), [1, 2, 4, 5]), `got ${Array.from(out.data)}`);
});

test('Slice: with step=2', () => {
  const data = mkTensor([0, 1, 2, 3, 4, 5, 6, 7], [8]);
  const starts = mkIntTensor([0], [1]);
  const ends = mkIntTensor([8], [1]);
  const axes = mkIntTensor([0], [1]);
  const steps = mkIntTensor([2], [1]);
  const [out] = new Slice().execute([data, starts, ends, axes, steps]);
  assert(allClose(Array.from(out.data), [0, 2, 4, 6]), `got ${Array.from(out.data)}`);
});

// ─── Transpose ───────────────────────────────────────────────────────────────

test('Transpose: 2D perm=[1,0]', () => {
  // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
  const data = mkTensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const [out] = new Transpose({ perm: [1, 0] }).execute([data]);
  assert(JSON.stringify(Array.from(out.shape)) === JSON.stringify([3, 2]), `shape = ${Array.from(out.shape)}`);
  assert(allClose(Array.from(out.data), [1, 4, 2, 5, 3, 6]), `got ${Array.from(out.data)}`);
});

// ─── Error cases ─────────────────────────────────────────────────────────────

test('MatMul: shape mismatch throws', () => {
  const A = mkTensor([1, 2, 3], [1, 3]);
  const B = mkTensor([1, 2, 3, 4], [2, 2]);
  let threw = false;
  try { new MatMul().execute([A, B]); } catch { threw = true; }
  assert(threw, 'should throw on shape mismatch');
});

test('Add: incompatible shapes throw', () => {
  const A = mkTensor([1, 2, 3], [3]);
  const B = mkTensor([1, 2], [2]);
  let threw = false;
  try { new Add().execute([A, B]); } catch { threw = true; }
  assert(threw, 'should throw on incompatible broadcast');
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
