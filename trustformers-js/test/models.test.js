/**
 * Comprehensive Model Tests
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
  createModel, 
  createTokenizer,
  tensor,
  utils,
  memory 
} from '../src/index.js';

// Track resources for cleanup
let createdResources = [];

function trackResource(resource) {
  createdResources.push(resource);
  return resource;
}

describe('Model Operations', () => {
  beforeAll(async () => {
    console.log('Initializing TrustformeRS for model tests...');
    await initialize({
      wasmPath: '../pkg/trustformers_wasm_bg.wasm'
    });
  });

  afterEach(() => {
    // Cleanup all resources created in tests
    createdResources.forEach(resource => {
      try {
        if (resource && typeof resource.free === 'function') {
          resource.free();
        }
      } catch (e) {
        // Ignore cleanup errors
      }
    });
    createdResources = [];
  });

  describe('Model Creation', () => {
    test('creates BERT base model', () => {
      const model = trackResource(createModel('bert_base'));
      expect(model).toBeTruthy();
      expect(model.config).toBeTruthy();
      expect(model.config.hidden_size).toBe(768);
      expect(model.config.num_layers).toBe(12);
      expect(model.config.num_heads).toBe(12);
    });

    test('creates BERT large model', () => {
      const model = trackResource(createModel('bert_large'));
      expect(model).toBeTruthy();
      expect(model.config).toBeTruthy();
      expect(model.config.hidden_size).toBe(1024);
      expect(model.config.num_layers).toBe(24);
      expect(model.config.num_heads).toBe(16);
    });

    test('creates GPT-2 base model', () => {
      const model = trackResource(createModel('gpt2_base'));
      expect(model).toBeTruthy();
      expect(model.config).toBeTruthy();
      expect(model.config.vocab_size).toBe(50257);
      expect(model.config.hidden_size).toBe(768);
      expect(model.config.num_layers).toBe(12);
    });

    test('creates GPT-2 medium model', () => {
      const model = trackResource(createModel('gpt2_medium'));
      expect(model).toBeTruthy();
      expect(model.config).toBeTruthy();
      expect(model.config.hidden_size).toBe(1024);
      expect(model.config.num_layers).toBe(24);
    });

    test('creates T5 base model', () => {
      const model = trackResource(createModel('t5_base'));
      expect(model).toBeTruthy();
      expect(model.config).toBeTruthy();
      expect(model.config.d_model).toBe(768);
      expect(model.config.num_layers).toBe(12);
    });

    test('throws error for invalid model type', () => {
      expect(() => {
        createModel('invalid_model_type');
      }).toThrow();
    });
  });

  describe('Model Configuration', () => {
    test('BERT model has correct configuration', () => {
      const model = trackResource(createModel('bert_base'));
      const config = model.config;
      
      expect(config.model_type).toBe('bert');
      expect(config.hidden_size).toBe(768);
      expect(config.intermediate_size).toBe(3072);
      expect(config.num_layers).toBe(12);
      expect(config.num_heads).toBe(12);
      expect(config.max_position_embeddings).toBe(512);
      expect(config.vocab_size).toBe(30522);
    });

    test('GPT-2 model has correct configuration', () => {
      const model = trackResource(createModel('gpt2_base'));
      const config = model.config;
      
      expect(config.model_type).toBe('gpt2');
      expect(config.hidden_size).toBe(768);
      expect(config.num_layers).toBe(12);
      expect(config.num_heads).toBe(12);
      expect(config.vocab_size).toBe(50257);
      expect(config.max_position_embeddings).toBe(1024);
    });

    test('can update model configuration', () => {
      const model = trackResource(createModel('bert_base'));
      
      // Update dropout rate
      model.config.hidden_dropout_prob = 0.2;
      expect(model.config.hidden_dropout_prob).toBeCloseTo(0.2, 5);
      
      // Update attention dropout
      model.config.attention_probs_dropout_prob = 0.15;
      expect(model.config.attention_probs_dropout_prob).toBeCloseTo(0.15, 5);
    });
  });

  describe('Model Inference', () => {
    test('performs forward pass with BERT model', () => {
      const model = trackResource(createModel('bert_base'));
      const inputTensor = trackResource(tensor([101, 2003, 102, 0, 0], [1, 5])); // [CLS] is [SEP] [PAD] [PAD]
      
      const output = trackResource(model.forward(inputTensor));
      expect(output).toBeTruthy();
      expect(Array.from(output.shape)).toEqual([1, 5, 768]); // batch_size, seq_len, hidden_size
    });

    test('performs forward pass with GPT-2 model', () => {
      const model = trackResource(createModel('gpt2_base'));
      const inputTensor = trackResource(tensor([1, 2, 3, 4, 5], [1, 5]));
      
      const output = trackResource(model.forward(inputTensor));
      expect(output).toBeTruthy();
      expect(Array.from(output.shape)).toEqual([1, 5, 768]); // batch_size, seq_len, hidden_size
    });

    test('handles different batch sizes', () => {
      const model = trackResource(createModel('bert_base'));
      
      // Single sample
      const input1 = trackResource(tensor([101, 2003, 102], [1, 3]));
      const output1 = trackResource(model.forward(input1));
      expect(Array.from(output1.shape)).toEqual([1, 3, 768]);
      
      // Batch of 2 samples
      const input2 = trackResource(tensor([101, 2003, 102, 101, 2004, 102], [2, 3]));
      const output2 = trackResource(model.forward(input2));
      expect(Array.from(output2.shape)).toEqual([2, 3, 768]);
    });

    test('handles different sequence lengths', () => {
      const model = trackResource(createModel('bert_base'));
      
      // Short sequence
      const input1 = trackResource(tensor([101, 102], [1, 2]));
      const output1 = trackResource(model.forward(input1));
      expect(Array.from(output1.shape)).toEqual([1, 2, 768]);
      
      // Longer sequence
      const input2 = trackResource(tensor([101, 2003, 2004, 2005, 102], [1, 5]));
      const output2 = trackResource(model.forward(input2));
      expect(Array.from(output2.shape)).toEqual([1, 5, 768]);
    });

    test('validates input tensor dimensions', () => {
      const model = trackResource(createModel('bert_base'));
      
      // Wrong number of dimensions
      const invalidInput = trackResource(tensor([101, 2003, 102])); // 1D instead of 2D
      expect(() => {
        model.forward(invalidInput);
      }).toThrow();
    });

    test('validates maximum sequence length', () => {
      const model = trackResource(createModel('bert_base'));
      
      // Sequence too long for BERT (max 512)
      const longSequence = new Array(600).fill(101);
      const longInput = trackResource(tensor(longSequence, [1, 600]));
      
      expect(() => {
        model.forward(longInput);
      }).toThrow();
    });
  });

  describe('Model Memory Management', () => {
    test('reports memory usage', () => {
      const model = trackResource(createModel('bert_base'));
      const memoryUsage = model.memory_usage_mb();
      
      expect(typeof memoryUsage).toBe('number');
      expect(memoryUsage).toBeGreaterThan(0);
      expect(memoryUsage).toBeLessThan(1000); // Reasonable upper bound
    });

    test('memory usage varies by model size', () => {
      const bertBase = trackResource(createModel('bert_base'));
      const bertLarge = trackResource(createModel('bert_large'));
      
      const baseMemory = bertBase.memory_usage_mb();
      const largeMemory = bertLarge.memory_usage_mb();
      
      expect(largeMemory).toBeGreaterThan(baseMemory);
    });

    test('frees model memory correctly', () => {
      const initialStats = memory.getStats();
      
      const model = createModel('bert_base');
      const afterCreate = memory.getStats();
      expect(afterCreate.used_mb).toBeGreaterThan(initialStats.used_mb);
      
      model.free();
      const afterFree = memory.getStats();
      expect(afterFree.used_mb).toBeLessThan(afterCreate.used_mb);
    });
  });

  describe('Model Serialization', () => {
    test('exports model state dict', () => {
      const model = trackResource(createModel('bert_base'));
      const stateDict = model.state_dict();
      
      expect(stateDict).toBeTruthy();
      expect(typeof stateDict).toBe('object');
      expect(Object.keys(stateDict).length).toBeGreaterThan(0);
    });

    test('loads model state dict', () => {
      const model1 = trackResource(createModel('bert_base'));
      const stateDict = model1.state_dict();
      
      const model2 = trackResource(createModel('bert_base'));
      model2.load_state_dict(stateDict);
      
      // Models should now have the same weights
      const input = trackResource(tensor([101, 2003, 102], [1, 3]));
      const output1 = trackResource(model1.forward(input));
      const output2 = trackResource(model2.forward(input));
      
      // Outputs should be identical
      const diff = trackResource(output1.sub(output2));
      expect(Math.abs(diff.sum())).toBeLessThan(1e-6);
    });

    test('saves and loads model to/from file', async () => {
      const model1 = trackResource(createModel('bert_base'));
      
      // Save model
      await model1.save('/tmp/test_model.bin');
      
      // Load model
      const model2 = trackResource(createModel('bert_base'));
      await model2.load('/tmp/test_model.bin');
      
      // Verify models are equivalent
      const input = trackResource(tensor([101, 2003, 102], [1, 3]));
      const output1 = trackResource(model1.forward(input));
      const output2 = trackResource(model2.forward(input));
      
      const diff = trackResource(output1.sub(output2));
      expect(Math.abs(diff.sum())).toBeLessThan(1e-6);
    });
  });

  describe('Model Evaluation Mode', () => {
    test('switches between training and evaluation modes', () => {
      const model = trackResource(createModel('bert_base'));
      
      // Default should be training mode
      expect(model.training).toBeTruthy();
      
      // Switch to eval mode
      model.eval();
      expect(model.training).toBeFalsy();
      
      // Switch back to training mode
      model.train();
      expect(model.training).toBeTruthy();
    });

    test('evaluation mode affects dropout behavior', () => {
      const model = trackResource(createModel('bert_base'));
      const input = trackResource(tensor([101, 2003, 102], [1, 3]));
      
      // In eval mode, output should be deterministic
      model.eval();
      const output1 = trackResource(model.forward(input));
      const output2 = trackResource(model.forward(input));
      
      const diff = trackResource(output1.sub(output2));
      expect(Math.abs(diff.sum())).toBeLessThan(1e-6);
    });
  });

  describe('Model Device Management', () => {
    test('reports model device', () => {
      const model = trackResource(createModel('bert_base'));
      const device = model.device();
      
      expect(typeof device).toBe('string');
      expect(['cpu', 'cuda', 'metal', 'rocm'].includes(device)).toBeTruthy();
    });

    test('moves model to different device', () => {
      const model = trackResource(createModel('bert_base'));
      const originalDevice = model.device();
      
      // Try to move to CPU (should always be available)
      model.to('cpu');
      expect(model.device()).toBe('cpu');
    });
  });

  describe('Model Performance', () => {
    test('measures inference time', () => {
      const model = trackResource(createModel('bert_base'));
      const input = trackResource(tensor([101, 2003, 102], [1, 3]));
      
      const timer = utils.timer('inference_test');
      const output = trackResource(model.forward(input));
      const elapsed = timer.elapsed();
      
      expect(elapsed).toBeGreaterThan(0);
      expect(elapsed).toBeLessThan(1000); // Should be under 1 second
    });

    test('batch processing is more efficient', () => {
      const model = trackResource(createModel('bert_base'));
      
      // Single sample processing
      const singleTimer = utils.timer('single_processing');
      for (let i = 0; i < 4; i++) {
        const input = trackResource(tensor([101, 2003, 102], [1, 3]));
        const output = trackResource(model.forward(input));
      }
      const singleTime = singleTimer.elapsed();
      
      // Batch processing
      const batchTimer = utils.timer('batch_processing');
      const batchInput = trackResource(tensor([
        101, 2003, 102,
        101, 2003, 102,
        101, 2003, 102,
        101, 2003, 102
      ], [4, 3]));
      const batchOutput = trackResource(model.forward(batchInput));
      const batchTime = batchTimer.elapsed();
      
      // Batch processing should be more efficient
      expect(batchTime).toBeLessThan(singleTime);
    });
  });

  describe('Model Error Handling', () => {
    test('handles null input gracefully', () => {
      const model = trackResource(createModel('bert_base'));
      
      expect(() => {
        model.forward(null);
      }).toThrow();
    });

    test('handles empty input gracefully', () => {
      const model = trackResource(createModel('bert_base'));
      const emptyInput = trackResource(tensor([], [0, 0]));
      
      expect(() => {
        model.forward(emptyInput);
      }).toThrow();
    });

    test('handles invalid token IDs', () => {
      const model = trackResource(createModel('bert_base'));
      // Token ID larger than vocab size
      const invalidInput = trackResource(tensor([999999], [1, 1]));
      
      expect(() => {
        model.forward(invalidInput);
      }).toThrow();
    });

    test('handles negative token IDs', () => {
      const model = trackResource(createModel('bert_base'));
      const negativeInput = trackResource(tensor([-1, -2, -3], [1, 3]));
      
      expect(() => {
        model.forward(negativeInput);
      }).toThrow();
    });
  });

  describe('Model Information', () => {
    test('provides model summary', () => {
      const model = trackResource(createModel('bert_base'));
      const summary = model.summary();
      
      expect(summary).toBeTruthy();
      expect(typeof summary).toBe('string');
      expect(summary.includes('BERT')).toBeTruthy();
      expect(summary.includes('768')).toBeTruthy(); // Hidden size
    });

    test('counts model parameters', () => {
      const model = trackResource(createModel('bert_base'));
      const paramCount = model.num_parameters();
      
      expect(typeof paramCount).toBe('number');
      expect(paramCount).toBeGreaterThan(100000000); // BERT base has ~110M parameters
      expect(paramCount).toBeLessThan(200000000);
    });

    test('lists model layers', () => {
      const model = trackResource(createModel('bert_base'));
      const layers = model.named_modules();
      
      expect(Array.isArray(layers)).toBeTruthy();
      expect(layers.length).toBeGreaterThan(0);
      expect(layers.some(layer => layer.includes('attention'))).toBeTruthy();
      expect(layers.some(layer => layer.includes('embeddings'))).toBeTruthy();
    });
  });
});