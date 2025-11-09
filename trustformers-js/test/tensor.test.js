/**
 * Comprehensive Tensor Operations Tests
 */

import { 
  describe, 
  test, 
  beforeAll, 
  afterEach, 
  expect 
} from './test-runner.js';

import { 
  initialize, 
  tensor, 
  zeros, 
  ones, 
  randn,
  eye,
  utils,
  memory 
} from '../src/index.js';

// Track tensors for cleanup
let createdTensors = [];

function trackTensor(t) {
  createdTensors.push(t);
  return t;
}

describe('Tensor Operations', () => {
  beforeAll(async () => {
    console.log('Initializing TrustformeRS for tensor tests...');
    await initialize({
      wasmPath: '../pkg/trustformers_wasm_bg.wasm'
    });
  });

  afterEach(() => {
    // Cleanup all tensors created in tests
    createdTensors.forEach(t => {
      try {
        if (t && typeof t.free === 'function') {
          t.free();
        }
      } catch (e) {
        // Ignore cleanup errors
      }
    });
    createdTensors = [];
  });

  describe('Tensor Creation', () => {
    test('creates tensor from array with explicit shape', () => {
      const t = trackTensor(tensor([1, 2, 3, 4, 5, 6], [2, 3]));
      expect(t).toBeTruthy();
      expect(Array.from(t.shape)).toEqual([2, 3]);
      expect(t.numel()).toBe(6);
    });

    test('creates tensor from nested array', () => {
      const t = trackTensor(tensor([[1, 2], [3, 4]]));
      expect(t).toBeTruthy();
      expect(Array.from(t.shape)).toEqual([2, 2]);
      expect(t.numel()).toBe(4);
    });

    test('creates scalar tensor', () => {
      const t = trackTensor(tensor(42));
      expect(t).toBeTruthy();
      expect(Array.from(t.shape)).toEqual([]);
      expect(t.numel()).toBe(1);
    });

    test('creates 1D tensor', () => {
      const t = trackTensor(tensor([1, 2, 3, 4, 5]));
      expect(t).toBeTruthy();
      expect(Array.from(t.shape)).toEqual([5]);
      expect(t.numel()).toBe(5);
    });

    test('creates zeros tensor', () => {
      const t = trackTensor(zeros([3, 3]));
      expect(t).toBeTruthy();
      expect(Array.from(t.shape)).toEqual([3, 3]);
      expect(t.sum()).toBeCloseTo(0, 5);
    });

    test('creates ones tensor', () => {
      const t = trackTensor(ones([2, 4]));
      expect(t).toBeTruthy();
      expect(Array.from(t.shape)).toEqual([2, 4]);
      expect(t.sum()).toBeCloseTo(8, 5);
    });

    test('creates identity matrix', () => {
      const t = trackTensor(eye(3));
      expect(t).toBeTruthy();
      expect(Array.from(t.shape)).toEqual([3, 3]);
      expect(t.sum()).toBeCloseTo(3, 5);
    });

    test('creates random normal tensor', () => {
      const t = trackTensor(randn([2, 3]));
      expect(t).toBeTruthy();
      expect(Array.from(t.shape)).toEqual([2, 3]);
      expect(t.numel()).toBe(6);
      // Random values should not all be exactly zero
      expect(Math.abs(t.sum())).toBeGreaterThan(0.01);
    });
  });

  describe('Tensor Properties', () => {
    test('gets tensor shape correctly', () => {
      const t = trackTensor(tensor([[1, 2, 3], [4, 5, 6]]));
      const shape = Array.from(t.shape);
      expect(shape).toEqual([2, 3]);
    });

    test('gets tensor size (numel)', () => {
      const t = trackTensor(tensor([1, 2, 3, 4, 5, 6], [2, 3]));
      expect(t.numel()).toBe(6);
    });

    test('gets tensor ndim', () => {
      const t1 = trackTensor(tensor(42)); // scalar
      const t2 = trackTensor(tensor([1, 2, 3])); // 1D
      const t3 = trackTensor(tensor([[1, 2], [3, 4]])); // 2D
      
      expect(t1.ndim()).toBe(0);
      expect(t2.ndim()).toBe(1);
      expect(t3.ndim()).toBe(2);
    });

    test('checks if tensor requires grad', () => {
      const t = trackTensor(tensor([1, 2, 3]));
      // Default should be false for inference
      expect(typeof t.requires_grad()).toBe('boolean');
    });

    test('gets tensor data type', () => {
      const t = trackTensor(tensor([1.5, 2.5, 3.5]));
      const dtype = t.dtype();
      expect(typeof dtype).toBe('string');
      expect(['f32', 'f64', 'i32', 'i64'].includes(dtype)).toBeTruthy();
    });
  });

  describe('Basic Arithmetic Operations', () => {
    test('adds two tensors', () => {
      const t1 = trackTensor(tensor([1, 2, 3, 4], [2, 2]));
      const t2 = trackTensor(tensor([5, 6, 7, 8], [2, 2]));
      const result = trackTensor(t1.add(t2));
      
      expect(Array.from(result.shape)).toEqual([2, 2]);
      expect(result.sum()).toBeCloseTo(36, 5);
    });

    test('subtracts two tensors', () => {
      const t1 = trackTensor(tensor([5, 6, 7, 8], [2, 2]));
      const t2 = trackTensor(tensor([1, 2, 3, 4], [2, 2]));
      const result = trackTensor(t1.sub(t2));
      
      expect(Array.from(result.shape)).toEqual([2, 2]);
      expect(result.sum()).toBeCloseTo(16, 5);
    });

    test('multiplies two tensors element-wise', () => {
      const t1 = trackTensor(tensor([2, 3, 4, 5], [2, 2]));
      const t2 = trackTensor(tensor([1, 2, 3, 4], [2, 2]));
      const result = trackTensor(t1.mul(t2));
      
      expect(Array.from(result.shape)).toEqual([2, 2]);
      expect(result.sum()).toBeCloseTo(40, 5); // 2*1 + 3*2 + 4*3 + 5*4 = 40
    });

    test('divides two tensors element-wise', () => {
      const t1 = trackTensor(tensor([8, 6, 4, 2], [2, 2]));
      const t2 = trackTensor(tensor([2, 2, 2, 2], [2, 2]));
      const result = trackTensor(t1.div(t2));
      
      expect(Array.from(result.shape)).toEqual([2, 2]);
      expect(result.sum()).toBeCloseTo(10, 5); // 4 + 3 + 2 + 1 = 10
    });

    test('adds scalar to tensor', () => {
      const t = trackTensor(tensor([1, 2, 3, 4], [2, 2]));
      const result = trackTensor(t.add_scalar(5));
      
      expect(Array.from(result.shape)).toEqual([2, 2]);
      expect(result.sum()).toBeCloseTo(30, 5); // (1+5) + (2+5) + (3+5) + (4+5) = 30
    });

    test('multiplies tensor by scalar', () => {
      const t = trackTensor(tensor([1, 2, 3, 4], [2, 2]));
      const result = trackTensor(t.mul_scalar(3));
      
      expect(Array.from(result.shape)).toEqual([2, 2]);
      expect(result.sum()).toBeCloseTo(30, 5); // 3*(1+2+3+4) = 30
    });
  });

  describe('Matrix Operations', () => {
    test('performs matrix multiplication', () => {
      const t1 = trackTensor(tensor([[1, 2], [3, 4]])); // 2x2
      const t2 = trackTensor(tensor([[5, 6], [7, 8]])); // 2x2
      const result = trackTensor(t1.matmul(t2));
      
      expect(Array.from(result.shape)).toEqual([2, 2]);
      // [1*5+2*7, 1*6+2*8] = [19, 22]
      // [3*5+4*7, 3*6+4*8] = [43, 50]
      expect(result.sum()).toBeCloseTo(134, 5); // 19+22+43+50 = 134
    });

    test('transposes matrix', () => {
      const t = trackTensor(tensor([[1, 2, 3], [4, 5, 6]])); // 2x3
      const result = trackTensor(t.transpose());
      
      expect(Array.from(result.shape)).toEqual([3, 2]);
      expect(result.sum()).toBeCloseTo(21, 5); // Same sum as original
    });

    test('transposes with specific dimensions', () => {
      const t = trackTensor(tensor([1, 2, 3, 4, 5, 6], [2, 3])); 
      const result = trackTensor(t.transpose_dims(0, 1));
      
      expect(Array.from(result.shape)).toEqual([3, 2]);
      expect(result.sum()).toBeCloseTo(21, 5);
    });
  });

  describe('Shape Manipulation', () => {
    test('reshapes tensor', () => {
      const t = trackTensor(tensor([1, 2, 3, 4, 5, 6])); // 1D [6]
      const result = trackTensor(t.reshape([2, 3]));
      
      expect(Array.from(result.shape)).toEqual([2, 3]);
      expect(result.numel()).toBe(6);
      expect(result.sum()).toBeCloseTo(21, 5);
    });

    test('views tensor with new shape', () => {
      const t = trackTensor(tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]));
      const result = trackTensor(t.view([4, 2]));
      
      expect(Array.from(result.shape)).toEqual([4, 2]);
      expect(result.numel()).toBe(8);
      expect(result.sum()).toBeCloseTo(36, 5);
    });

    test('squeezes tensor (removes size-1 dimensions)', () => {
      const t = trackTensor(tensor([1, 2, 3], [1, 3, 1]));
      const result = trackTensor(t.squeeze());
      
      expect(Array.from(result.shape)).toEqual([3]);
      expect(result.sum()).toBeCloseTo(6, 5);
    });

    test('unsqueezes tensor (adds size-1 dimension)', () => {
      const t = trackTensor(tensor([1, 2, 3])); // [3]
      const result = trackTensor(t.unsqueeze(0));
      
      expect(Array.from(result.shape)).toEqual([1, 3]);
      expect(result.sum()).toBeCloseTo(6, 5);
    });
  });

  describe('Indexing and Slicing', () => {
    test('slices tensor along dimension', () => {
      const t = trackTensor(tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])); // 3x3
      const result = trackTensor(t.slice(0, 0, 2)); // First 2 rows
      
      expect(Array.from(result.shape)).toEqual([2, 3]);
      expect(result.sum()).toBeCloseTo(21, 5); // 1+2+3+4+5+6 = 21
    });

    test('indexes tensor with single index', () => {
      const t = trackTensor(tensor([[1, 2, 3], [4, 5, 6]])); // 2x3
      const result = trackTensor(t.index([0])); // First row
      
      expect(Array.from(result.shape)).toEqual([3]);
      expect(result.sum()).toBeCloseTo(6, 5); // 1+2+3 = 6
    });
  });

  describe('Reduction Operations', () => {
    test('sums all elements', () => {
      const t = trackTensor(tensor([1, 2, 3, 4, 5]));
      const sum = t.sum();
      expect(sum).toBeCloseTo(15, 5);
    });

    test('sums along dimension', () => {
      const t = trackTensor(tensor([[1, 2, 3], [4, 5, 6]])); // 2x3
      const result = trackTensor(t.sum_dim(0)); // Sum along rows
      
      expect(Array.from(result.shape)).toEqual([3]);
      expect(result.sum()).toBeCloseTo(21, 5); // [5, 7, 9] = 21
    });

    test('finds mean of elements', () => {
      const t = trackTensor(tensor([2, 4, 6, 8]));
      const mean = t.mean();
      expect(mean).toBeCloseTo(5, 5);
    });

    test('finds max element', () => {
      const t = trackTensor(tensor([3, 1, 4, 1, 5, 9, 2, 6]));
      const max = t.max();
      expect(max).toBeCloseTo(9, 5);
    });

    test('finds min element', () => {
      const t = trackTensor(tensor([3, 1, 4, 1, 5, 9, 2, 6]));
      const min = t.min();
      expect(min).toBeCloseTo(1, 5);
    });

    test('finds argmax along dimension', () => {
      const t = trackTensor(tensor([[1, 3, 2], [6, 4, 5]])); // 2x3
      const result = trackTensor(t.argmax(1)); // Argmax along columns
      
      expect(Array.from(result.shape)).toEqual([2]);
      // First row: max is 3 at index 1, second row: max is 6 at index 0
    });
  });

  describe('Mathematical Functions', () => {
    test('applies sin function', () => {
      const t = trackTensor(tensor([0, Math.PI/2, Math.PI]));
      const result = trackTensor(t.sin());
      
      expect(Array.from(result.shape)).toEqual([3]);
      const data = Array.from(result.data);
      expect(data[0]).toBeCloseTo(0, 5);
      expect(data[1]).toBeCloseTo(1, 5);
      expect(data[2]).toBeCloseTo(0, 5);
    });

    test('applies exp function', () => {
      const t = trackTensor(tensor([0, 1, 2]));
      const result = trackTensor(t.exp());
      
      expect(Array.from(result.shape)).toEqual([3]);
      const data = Array.from(result.data);
      expect(data[0]).toBeCloseTo(1, 5); // e^0 = 1
      expect(data[1]).toBeCloseTo(Math.E, 5); // e^1 = e
      expect(data[2]).toBeCloseTo(Math.E * Math.E, 5); // e^2 = e^2
    });

    test('applies log function', () => {
      const t = trackTensor(tensor([1, Math.E, Math.E * Math.E]));
      const result = trackTensor(t.log());
      
      expect(Array.from(result.shape)).toEqual([3]);
      const data = Array.from(result.data);
      expect(data[0]).toBeCloseTo(0, 5); // ln(1) = 0
      expect(data[1]).toBeCloseTo(1, 5); // ln(e) = 1
      expect(data[2]).toBeCloseTo(2, 5); // ln(e^2) = 2
    });

    test('applies sqrt function', () => {
      const t = trackTensor(tensor([1, 4, 9, 16]));
      const result = trackTensor(t.sqrt());
      
      expect(Array.from(result.shape)).toEqual([4]);
      const data = Array.from(result.data);
      expect(data[0]).toBeCloseTo(1, 5);
      expect(data[1]).toBeCloseTo(2, 5);
      expect(data[2]).toBeCloseTo(3, 5);
      expect(data[3]).toBeCloseTo(4, 5);
    });

    test('applies abs function', () => {
      const t = trackTensor(tensor([-3, -1, 0, 1, 3]));
      const result = trackTensor(t.abs());
      
      expect(Array.from(result.shape)).toEqual([5]);
      expect(result.sum()).toBeCloseTo(8, 5); // 3+1+0+1+3 = 8
    });
  });

  describe('Activation Functions', () => {
    test('applies ReLU function', () => {
      const t = trackTensor(tensor([-2, -1, 0, 1, 2]));
      const result = trackTensor(t.relu());
      
      expect(Array.from(result.shape)).toEqual([5]);
      expect(result.sum()).toBeCloseTo(3, 5); // 0+0+0+1+2 = 3
    });

    test('applies sigmoid function', () => {
      const t = trackTensor(tensor([0]));
      const result = trackTensor(t.sigmoid());
      
      expect(Array.from(result.shape)).toEqual([1]);
      const data = Array.from(result.data);
      expect(data[0]).toBeCloseTo(0.5, 5); // sigmoid(0) = 0.5
    });

    test('applies tanh function', () => {
      const t = trackTensor(tensor([0]));
      const result = trackTensor(t.tanh());
      
      expect(Array.from(result.shape)).toEqual([1]);
      const data = Array.from(result.data);
      expect(data[0]).toBeCloseTo(0, 5); // tanh(0) = 0
    });
  });

  describe('Memory Management', () => {
    test('tracks tensor memory correctly', () => {
      const initialStats = memory.getStats();
      
      const t1 = trackTensor(tensor([1, 2, 3, 4, 5], [5]));
      const t2 = trackTensor(tensor([[1, 2], [3, 4]], [2, 2]));
      
      const currentStats = memory.getStats();
      expect(currentStats.used_mb).toBeGreaterThan(initialStats.used_mb);
    });

    test('frees tensor memory correctly', () => {
      const initialStats = memory.getStats();
      
      const t = tensor([1, 2, 3, 4, 5], [5]);
      const afterCreate = memory.getStats();
      expect(afterCreate.used_mb).toBeGreaterThan(initialStats.used_mb);
      
      t.free();
      const afterFree = memory.getStats();
      expect(afterFree.used_mb).toBeLessThan(afterCreate.used_mb);
    });

    test('clones tensor correctly', () => {
      const original = trackTensor(tensor([1, 2, 3, 4]));
      const cloned = trackTensor(original.clone());
      
      expect(Array.from(cloned.shape)).toEqual(Array.from(original.shape));
      expect(cloned.sum()).toBeCloseTo(original.sum(), 5);
      
      // Modify original and ensure clone is unchanged
      const modified = trackTensor(original.add_scalar(10));
      expect(cloned.sum()).toBeCloseTo(10, 5); // Original sum
      expect(modified.sum()).toBeCloseTo(50, 5); // Modified sum
    });
  });

  describe('Error Handling', () => {
    test('throws error for invalid shape in tensor creation', () => {
      expect(() => {
        tensor([1, 2, 3], [2, 2]); // Wrong shape
      }).toThrow();
    });

    test('throws error for shape mismatch in operations', () => {
      const t1 = trackTensor(tensor([1, 2], [2]));
      const t2 = trackTensor(tensor([1, 2, 3], [3]));
      
      expect(() => {
        t1.add(t2); // Shape mismatch
      }).toThrow();
    });

    test('throws error for invalid matrix multiplication', () => {
      const t1 = trackTensor(tensor([[1, 2]], [1, 2])); // 1x2
      const t2 = trackTensor(tensor([[1], [2], [3]], [3, 1])); // 3x1
      
      expect(() => {
        t1.matmul(t2); // Incompatible shapes for matmul
      }).toThrow();
    });

    test('throws error for invalid reshape', () => {
      const t = trackTensor(tensor([1, 2, 3, 4, 5], [5]));
      
      expect(() => {
        t.reshape([2, 3]); // 5 elements can't be reshaped to [2,3] = 6 elements
      }).toThrow();
    });
  });
});