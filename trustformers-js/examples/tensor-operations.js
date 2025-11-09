/**
 * Example: Basic Tensor Operations with TrustformeRS
 */

import { initialize, tensor, zeros, ones, randn, utils, memory } from '../src/index.js';

async function runTensorOperations() {
  try {
    // Initialize TrustformeRS
    console.log('Initializing TrustformeRS...');
    await initialize({
      wasmPath: '../pkg/trustformers_wasm_bg.wasm'
    });
    
    console.log('\n=== Basic Tensor Creation ===');
    
    // Create tensors
    const t1 = tensor([1, 2, 3, 4, 5, 6], [2, 3]);
    console.log('Tensor 1:', t1.toString());
    
    const t2 = tensor([7, 8, 9, 10, 11, 12], [2, 3]);
    console.log('Tensor 2:', t2.toString());
    
    // Create special tensors
    const zerosT = zeros([3, 3]);
    console.log('Zeros tensor:', zerosT.toString());
    
    const onesT = ones([2, 4]);
    console.log('Ones tensor:', onesT.toString());
    
    const randomT = randn([3, 3]);
    console.log('Random tensor:', randomT.toString());
    
    console.log('\n=== Tensor Operations ===');
    
    // Addition
    const sum = t1.add(t2);
    console.log('t1 + t2:', sum.toString());
    
    // Subtraction
    const diff = t2.sub(t1);
    console.log('t2 - t1:', diff.toString());
    
    // Element-wise multiplication
    const product = t1.mul(t2);
    console.log('t1 * t2:', product.toString());
    
    // Matrix multiplication
    const t3 = tensor([1, 2, 3, 4, 5, 6], [3, 2]);
    const matmulResult = t1.matmul(t3);
    console.log('t1 @ t3:', matmulResult.toString());
    
    console.log('\n=== Tensor Transformations ===');
    
    // Transpose
    const transposed = t1.transpose();
    console.log('t1.T:', transposed.toString());
    
    // Reshape
    const reshaped = t1.reshape(new Uint32Array([3, 2]));
    console.log('t1 reshaped to [3, 2]:', reshaped.toString());
    
    // Sum and mean
    console.log('Sum of t1:', t1.sum());
    console.log('Mean of t1:', t1.mean());
    
    console.log('\n=== Activation Functions ===');
    
    // Create a tensor with negative values
    const t4 = tensor([-2, -1, 0, 1, 2], [5]);
    console.log('Original tensor:', t4.toString());
    
    // ReLU
    const reluResult = t4.relu();
    console.log('ReLU:', reluResult.toString());
    
    // Softmax
    const softmaxResult = t4.softmax(0);
    console.log('Softmax:', softmaxResult.toString());
    
    // Exponential
    const expResult = t4.exp();
    console.log('Exp:', expResult.toString());
    
    console.log('\n=== Memory Usage ===');
    const memStats = memory.getStats();
    console.log(`Memory used: ${memStats.used_mb.toFixed(2)} MB`);
    console.log(`Memory limit: ${memStats.limit_mb.toFixed(2)} MB`);
    
    // Clean up
    console.log('\n=== Cleanup ===');
    console.log('Freeing tensors...');
    [t1, t2, t3, t4, zerosT, onesT, randomT, sum, diff, product, 
     matmulResult, transposed, reshaped, reluResult, softmaxResult, expResult].forEach(t => {
      if (t && t.free) t.free();
    });
    
    console.log('Done!');
    
  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the example
runTensorOperations();