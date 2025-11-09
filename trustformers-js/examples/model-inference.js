/**
 * Example: Model Inference with TrustformeRS
 */

import { 
  initialize, 
  createModel, 
  createTokenizer, 
  tensor,
  utils,
  memory 
} from '../src/index.js';

async function runModelInference() {
  try {
    // Initialize TrustformeRS
    console.log('Initializing TrustformeRS...');
    await initialize({
      wasmPath: '../pkg/trustformers_wasm_bg.wasm'
    });
    
    console.log('\n=== Creating Model ===');
    
    // Create a BERT model
    const timer = utils.timer('model_creation');
    const model = createModel('bert_base');
    timer.log_elapsed();
    
    console.log('Model created successfully');
    console.log(`Model architecture: BERT`);
    console.log(`Hidden size: ${model.config.hidden_size}`);
    console.log(`Number of layers: ${model.config.num_layers}`);
    console.log(`Number of attention heads: ${model.config.num_heads}`);
    
    console.log('\n=== Creating Tokenizer ===');
    
    // Create a tokenizer
    const tokenizer = createTokenizer('wordpiece');
    
    // Load a simple vocabulary for demonstration
    const simpleVocab = {
      '[PAD]': 0,
      '[UNK]': 1,
      '[CLS]': 2,
      '[SEP]': 3,
      '[MASK]': 4,
      'hello': 5,
      'world': 6,
      'how': 7,
      'are': 8,
      'you': 9,
      'i': 10,
      'am': 11,
      'fine': 12,
      'thanks': 13,
      'the': 14,
      'a': 15,
      'is': 16,
      'it': 17,
      'good': 18,
      'day': 19
    };
    
    tokenizer.load_vocab(simpleVocab);
    console.log(`Tokenizer loaded with vocab size: ${tokenizer.vocab_size}`);
    
    console.log('\n=== Tokenizing Text ===');
    
    // Tokenize some text
    const text = "hello world";
    const encoded = tokenizer.encode(text, true);
    console.log(`Text: "${text}"`);
    console.log(`Encoded: [${Array.from(encoded).join(', ')}]`);
    
    // Decode back
    const decoded = tokenizer.decode(encoded, true);
    console.log(`Decoded: "${decoded}"`);
    
    console.log('\n=== Running Inference ===');
    
    // Create input tensor from token IDs
    const inputTensor = tensor(Array.from(encoded), [1, encoded.length]);
    console.log('Input tensor shape:', Array.from(inputTensor.shape));
    
    // Note: In a real scenario, you would load pre-trained weights
    // For this example, we're using randomly initialized weights
    console.log('Note: Using randomly initialized weights for demonstration');
    
    const inferenceTimer = utils.timer('inference');
    const output = model.forward(inputTensor);
    inferenceTimer.log_elapsed();
    
    console.log('Output tensor shape:', Array.from(output.shape));
    console.log('Output sample values:', Array.from(output.data).slice(0, 10).map(v => v.toFixed(4)));
    
    console.log('\n=== Batch Processing ===');
    
    // Tokenize multiple texts
    const texts = [
      "hello world",
      "how are you",
      "it is a good day"
    ];
    
    const batchEncoded = tokenizer.batch_encode(texts, true);
    console.log(`Batch size: ${batchEncoded.len()}`);
    
    for (let i = 0; i < batchEncoded.len(); i++) {
      const seq = batchEncoded.get_sequence(i);
      if (seq) {
        console.log(`  Text ${i}: [${Array.from(seq).join(', ')}]`);
      }
    }
    
    console.log('\n=== Memory Usage ===');
    
    console.log(`Model memory usage: ${model.memory_usage_mb().toFixed(2)} MB`);
    
    const memStats = memory.getStats();
    console.log(`Total WASM memory used: ${memStats.used_mb.toFixed(2)} MB`);
    console.log(`Total WASM memory limit: ${memStats.limit_mb.toFixed(2)} MB`);
    
    console.log('\n=== Performance Stats ===');
    
    // Run multiple inferences to measure performance
    const numRuns = 10;
    const perfTimer = utils.timer('performance_test');
    
    for (let i = 0; i < numRuns; i++) {
      const output = model.forward(inputTensor);
      output.free();
    }
    
    const elapsed = perfTimer.elapsed();
    console.log(`${numRuns} inferences completed in ${elapsed.toFixed(2)}ms`);
    console.log(`Average time per inference: ${(elapsed / numRuns).toFixed(2)}ms`);
    
    // Clean up
    console.log('\n=== Cleanup ===');
    inputTensor.free();
    output.free();
    batchEncoded.free();
    tokenizer.free();
    model.free();
    
    console.log('Done!');
    
  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the example
runModelInference();