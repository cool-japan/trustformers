/**
 * Basic tests for TrustformeRS JavaScript API
 */

import { 
  initialize, 
  tensor, 
  zeros, 
  ones,
  createModel,
  createTokenizer,
  utils
} from '../src/index.js';

async function runTests() {
  console.log('🧪 Running TrustformeRS JavaScript API Tests\n');
  
  let passed = 0;
  let failed = 0;
  
  function test(name, fn) {
    try {
      fn();
      console.log(`✅ ${name}`);
      passed++;
    } catch (error) {
      console.log(`❌ ${name}`);
      console.log(`   Error: ${error.message}`);
      failed++;
    }
  }
  
  // Initialize
  console.log('Initializing TrustformeRS...');
  await initialize({
    wasmPath: '../pkg/trustformers_wasm_bg.wasm'
  });
  
  console.log('\n--- Tensor Tests ---');
  
  test('Create tensor from array', () => {
    const t = tensor([1, 2, 3, 4], [2, 2]);
    if (!t) throw new Error('Failed to create tensor');
    if (t.shape.length !== 2) throw new Error('Wrong shape dimensions');
    t.free();
  });
  
  test('Create zeros tensor', () => {
    const t = zeros([3, 3]);
    if (!t) throw new Error('Failed to create zeros tensor');
    const sum = t.sum();
    if (sum !== 0) throw new Error(`Expected sum 0, got ${sum}`);
    t.free();
  });
  
  test('Create ones tensor', () => {
    const t = ones([2, 3]);
    if (!t) throw new Error('Failed to create ones tensor');
    const sum = t.sum();
    if (sum !== 6) throw new Error(`Expected sum 6, got ${sum}`);
    t.free();
  });
  
  test('Tensor addition', () => {
    const t1 = tensor([1, 2, 3, 4], [2, 2]);
    const t2 = tensor([5, 6, 7, 8], [2, 2]);
    const result = t1.add(t2);
    const sum = result.sum();
    if (sum !== 36) throw new Error(`Expected sum 36, got ${sum}`);
    t1.free();
    t2.free();
    result.free();
  });
  
  test('Tensor multiplication', () => {
    const t1 = tensor([2, 2, 2, 2], [2, 2]);
    const t2 = tensor([3, 3, 3, 3], [2, 2]);
    const result = t1.mul(t2);
    const sum = result.sum();
    if (sum !== 24) throw new Error(`Expected sum 24, got ${sum}`);
    t1.free();
    t2.free();
    result.free();
  });
  
  test('Tensor transpose', () => {
    const t = tensor([1, 2, 3, 4, 5, 6], [2, 3]);
    const transposed = t.transpose();
    const shape = Array.from(transposed.shape);
    if (shape[0] !== 3 || shape[1] !== 2) {
      throw new Error(`Expected shape [3, 2], got [${shape.join(', ')}]`);
    }
    t.free();
    transposed.free();
  });
  
  console.log('\n--- Model Tests ---');
  
  test('Create BERT model', () => {
    const model = createModel('bert_base');
    if (!model) throw new Error('Failed to create model');
    if (model.config.hidden_size !== 768) {
      throw new Error(`Expected hidden size 768, got ${model.config.hidden_size}`);
    }
    model.free();
  });
  
  test('Create GPT-2 model', () => {
    const model = createModel('gpt2_base');
    if (!model) throw new Error('Failed to create model');
    if (model.config.vocab_size !== 50257) {
      throw new Error(`Expected vocab size 50257, got ${model.config.vocab_size}`);
    }
    model.free();
  });
  
  console.log('\n--- Tokenizer Tests ---');
  
  test('Create WordPiece tokenizer', () => {
    const tokenizer = createTokenizer('wordpiece');
    if (!tokenizer) throw new Error('Failed to create tokenizer');
    tokenizer.free();
  });
  
  test('Create BPE tokenizer', () => {
    const tokenizer = createTokenizer('bpe');
    if (!tokenizer) throw new Error('Failed to create tokenizer');
    tokenizer.free();
  });
  
  test('Tokenizer encode/decode', () => {
    const tokenizer = createTokenizer('wordpiece');
    const vocab = { 'hello': 1, 'world': 2, '[UNK]': 0 };
    tokenizer.load_vocab(vocab);
    
    const encoded = tokenizer.encode('hello world', false);
    if (encoded.length === 0) throw new Error('Encoding failed');
    
    const decoded = tokenizer.decode(encoded, false);
    if (typeof decoded !== 'string') throw new Error('Decoding failed');
    
    tokenizer.free();
  });
  
  console.log('\n--- Utility Tests ---');
  
  test('Version check', () => {
    const version = utils.version();
    if (!version || typeof version !== 'string') {
      throw new Error('Failed to get version');
    }
  });
  
  test('Feature check', () => {
    const features = utils.features();
    if (!Array.isArray(features)) {
      throw new Error('Failed to get features');
    }
  });
  
  test('Timer functionality', () => {
    const timer = utils.timer('test_timer');
    // Small delay
    for (let i = 0; i < 1000000; i++) {}
    const elapsed = timer.elapsed();
    if (typeof elapsed !== 'number' || elapsed < 0) {
      throw new Error('Timer failed');
    }
    timer.free();
  });
  
  console.log('\n--- Results ---');
  console.log(`Total tests: ${passed + failed}`);
  console.log(`Passed: ${passed}`);
  console.log(`Failed: ${failed}`);
  
  if (failed > 0) {
    process.exit(1);
  } else {
    console.log('\n🎉 All tests passed!');
  }
}

// Run tests
runTests().catch(error => {
  console.error('Test suite failed:', error);
  process.exit(1);
});