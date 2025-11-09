/**
 * Enhanced Tensor Operations Example for TrustformeRS JavaScript API
 * Demonstrates the comprehensive tensor operations and modern JavaScript features
 */

import { 
  initialize, 
  tensor, 
  zeros, 
  ones, 
  randn,
  tensor_ops,
  activations,
  streaming,
  async_utils,
  tensor_utils,
  webgpu,
  memory
} from '../src/index.js';

async function main() {
  console.log('🚀 TrustformeRS Enhanced Tensor Operations Demo');
  console.log('===============================================');
  
  // Initialize the WASM module
  await initialize({
    wasmPath: '../pkg/trustformers_wasm_bg.wasm',
    initPanicHook: true
  });
  
  console.log('✅ TrustformeRS initialized successfully');
  console.log('📊 Memory usage:', memory.getStats());
  
  // Demo 1: Basic tensor operations
  console.log('\n1. Basic Tensor Operations');
  console.log('-------------------------');
  
  const a = tensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const b = tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]);
  
  console.log('Tensor A:', a.toString());
  console.log('Tensor B:', b.toString());
  
  // Arithmetic operations
  const sum = tensor_ops.add(a, b);
  const product = tensor_ops.mul(a, b);
  const matrix_mult = tensor_ops.matmul(a, tensor_ops.transpose(b));
  
  console.log('A + B:', sum.toString());
  console.log('A * B:', product.toString());
  console.log('A @ B^T:', matrix_mult.toString());
  
  // Demo 2: Advanced tensor operations
  console.log('\n2. Advanced Tensor Operations');
  console.log('-----------------------------');
  
  const data = randn([4, 5, 6]);
  console.log('Random tensor shape:', data.shape);
  
  // Shape operations
  const reshaped = tensor_ops.reshape(data, [20, 6]);
  const squeezed = tensor_ops.squeeze(reshaped, 0);
  const transposed = tensor_ops.transpose(reshaped, [1, 0]);
  
  console.log('Reshaped to [20, 6]:', reshaped.shape);
  console.log('Transposed to [6, 20]:', transposed.shape);
  
  // Reduction operations
  const mean_all = tensor_ops.mean(data);
  const sum_dim = tensor_ops.sum(data, 1, true);
  const max_vals = tensor_ops.max(data, 2);
  
  console.log('Mean (all):', mean_all.toString());
  console.log('Sum along dim 1 (keepDim=true):', sum_dim.shape);
  console.log('Max along dim 2:', max_vals.shape);
  
  // Demo 3: Tensor creation utilities
  console.log('\n3. Tensor Creation Utilities');
  console.log('----------------------------');
  
  // Create from nested array
  const nested = [
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
  ];
  const from_nested = tensor_utils.fromNestedArray(nested);
  console.log('From nested array:', from_nested.shape);
  
  // Create from typed array
  const float_data = new Float32Array([1, 2, 3, 4, 5, 6]);
  const from_typed = tensor_utils.fromTypedArray(float_data, [2, 3]);
  console.log('From Float32Array:', from_typed.toString());
  
  // Create random with different distributions
  const normal = tensor_utils.random([3, 3], 'normal', { mean: 0, std: 1 });
  const uniform = tensor_utils.random([3, 3], 'uniform', { low: -1, high: 1 });
  
  console.log('Normal distribution:', normal.toString());
  console.log('Uniform distribution:', uniform.toString());
  
  // Demo 4: Activation functions
  console.log('\n4. Activation Functions');
  console.log('----------------------');
  
  const input = tensor([-2, -1, 0, 1, 2], [5]);
  
  console.log('Input:', input.toString());
  console.log('ReLU:', activations.relu(input).toString());
  console.log('Leaky ReLU:', activations.leakyRelu(input, 0.1).toString());
  console.log('Sigmoid:', activations.sigmoid(input).toString());
  console.log('Tanh:', activations.tanh(input).toString());
  console.log('GELU:', activations.gelu(input).toString());
  console.log('Swish:', activations.swish(input).toString());
  
  // Demo 5: Mathematical operations
  console.log('\n5. Mathematical Operations');
  console.log('-------------------------');
  
  const math_input = tensor([1, 4, 9, 16], [4]);
  console.log('Input:', math_input.toString());
  console.log('Square root:', tensor_ops.sqrt(math_input).toString());
  console.log('Logarithm:', tensor_ops.log(math_input).toString());
  console.log('Exponential:', tensor_ops.exp(tensor([-1, 0, 1, 2], [4])).toString());
  console.log('Power (^2):', tensor_ops.pow(math_input, 2).toString());
  console.log('Absolute:', tensor_ops.abs(tensor([-2, -1, 0, 1, 2], [5])).toString());
  
  // Demo 6: Concatenation and stacking
  console.log('\n6. Concatenation and Stacking');
  console.log('-----------------------------');
  
  const t1 = ones([2, 3]);
  const t2 = zeros([2, 3]);
  const t3 = tensor_ops.mulScalar(ones([2, 3]), 2);
  
  const concatenated = tensor_ops.cat([t1, t2, t3], 0);
  const stacked = tensor_ops.stack([t1, t2, t3], 0);
  
  console.log('Concatenated shape:', concatenated.shape);
  console.log('Stacked shape:', stacked.shape);
  
  // Demo 7: Async utilities
  console.log('\n7. Async Utilities');
  console.log('------------------');
  
  const tensors = [
    randn([10, 10]),
    randn([10, 10]),
    randn([10, 10]),
    randn([10, 10])
  ];
  
  // Process tensors in batch
  const processed = await async_utils.processBatch(
    tensors,
    async (tensor) => {
      // Simulate some async processing
      await new Promise(resolve => setTimeout(resolve, 10));
      return tensor_ops.mean(tensor);
    },
    2 // batch size
  );
  
  console.log('Batch processed results:', processed.map(t => t.toString()));
  
  // Demo 8: WebGPU support (if available)
  console.log('\n8. WebGPU Support');
  console.log('----------------');
  
  if (webgpu.isAvailable()) {
    console.log('✅ WebGPU is available');
    console.log('Status:', webgpu.getStatus());
    
    try {
      const deviceInfo = await webgpu.getDeviceInfo();
      console.log('Device info:', deviceInfo);
    } catch (error) {
      console.log('Could not get device info:', error.message);
    }
  } else {
    console.log('❌ WebGPU is not available');
  }
  
  // Demo 9: Memory management
  console.log('\n9. Memory Management');
  console.log('-------------------');
  
  console.log('Memory before cleanup:', memory.getStats());
  
  // Clean up tensors
  [a, b, sum, product, matrix_mult, data, reshaped, transposed, 
   from_nested, from_typed, normal, uniform, input, math_input, 
   t1, t2, t3, concatenated, stacked, ...tensors, ...processed]
    .forEach(tensor => tensor.free && tensor.free());
  
  console.log('Memory after cleanup:', memory.getStats());
  
  console.log('\n✅ Demo completed successfully!');
}

// Run the demo
main().catch(console.error);