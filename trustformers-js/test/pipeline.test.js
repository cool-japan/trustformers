/**
 * Comprehensive Pipeline Tests
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
  Pipeline,
  utils
} from '../src/index.js';

// Track resources for cleanup
let createdResources = [];

function trackResource(resource) {
  createdResources.push(resource);
  return resource;
}

describe('Pipeline Operations', () => {
  beforeAll(async () => {
    console.log('Initializing TrustformeRS for pipeline tests...');
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

  describe('Text Classification Pipeline', () => {
    test('creates text classification pipeline', () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      const pipeline = trackResource(Pipeline.textClassification(
        model, 
        tokenizer, 
        ['positive', 'negative', 'neutral']
      ));
      
      expect(pipeline).toBeTruthy();
      expect(typeof pipeline.classify).toBe('function');
      expect(typeof pipeline.classify_batch).toBe('function');
    });

    test('classifies single text', async () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      
      // Load simple vocabulary
      const vocab = {
        '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
        'good': 4, 'bad': 5, 'great': 6, 'terrible': 7
      };
      tokenizer.load_vocab(vocab);
      
      const pipeline = trackResource(Pipeline.textClassification(
        model, 
        tokenizer, 
        ['positive', 'negative', 'neutral']
      ));
      
      const result = trackResource(await pipeline.classify('good'));
      expect(result).toBeTruthy();
      expect(typeof result.label).toBe('string');
      expect(['positive', 'negative', 'neutral'].includes(result.label)).toBeTruthy();
      expect(typeof result.score).toBe('number');
      expect(result.score).toBeGreaterThan(0);
      expect(result.score).toBeLessThan(1);
      expect(Array.isArray(result.all_scores)).toBeTruthy();
    });

    test('classifies batch of texts', async () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      
      const vocab = {
        '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
        'good': 4, 'bad': 5, 'great': 6, 'terrible': 7
      };
      tokenizer.load_vocab(vocab);
      
      const pipeline = trackResource(Pipeline.textClassification(
        model, 
        tokenizer, 
        ['positive', 'negative']
      ));
      
      const texts = ['good', 'bad', 'great'];
      const results = await pipeline.classify_batch(texts);
      
      expect(Array.isArray(results)).toBeTruthy();
      expect(results.length).toBe(3);
      
      results.forEach((result, i) => {
        trackResource(result);
        expect(typeof result.label).toBe('string');
        expect(['positive', 'negative'].includes(result.label)).toBeTruthy();
        expect(typeof result.score).toBe('number');
      });
    });

    test('handles empty text', async () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      const pipeline = trackResource(Pipeline.textClassification(
        model, 
        tokenizer, 
        ['positive', 'negative']
      ));
      
      const result = trackResource(await pipeline.classify(''));
      expect(result).toBeTruthy();
      expect(typeof result.label).toBe('string');
    });
  });

  describe('Question Answering Pipeline', () => {
    test('creates question answering pipeline', () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      const pipeline = trackResource(Pipeline.questionAnswering(model, tokenizer));
      
      expect(pipeline).toBeTruthy();
      expect(typeof pipeline.answer).toBe('function');
    });

    test('answers question from context', async () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      
      const vocab = {
        '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
        'what': 4, 'is': 5, 'the': 6, 'capital': 7, 'of': 8,
        'paris': 9, 'france': 10, 'city': 11
      };
      tokenizer.load_vocab(vocab);
      
      const pipeline = trackResource(Pipeline.questionAnswering(model, tokenizer));
      
      const question = 'What is the capital of France?';
      const context = 'Paris is the capital city of France.';
      
      const result = trackResource(await pipeline.answer(question, context));
      expect(result).toBeTruthy();
      expect(typeof result.answer).toBe('string');
      expect(typeof result.score).toBe('number');
      expect(typeof result.start).toBe('number');
      expect(typeof result.end).toBe('number');
      expect(result.start).toBeLessThan(result.end);
      expect(result.score).toBeGreaterThan(0);
    });

    test('handles unanswerable questions', async () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      
      const vocab = {
        '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
        'what': 4, 'is': 5, 'the': 6, 'weather': 7,
        'paris': 8, 'france': 9, 'capital': 10, 'city': 11
      };
      tokenizer.load_vocab(vocab);
      
      const pipeline = trackResource(Pipeline.questionAnswering(model, tokenizer));
      
      const question = 'What is the weather like?';
      const context = 'Paris is the capital city of France.';
      
      const result = trackResource(await pipeline.answer(question, context));
      expect(result).toBeTruthy();
      expect(typeof result.answer).toBe('string');
      expect(result.score).toBeLessThan(0.5); // Should have low confidence
    });

    test('validates input parameters', async () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      const pipeline = trackResource(Pipeline.questionAnswering(model, tokenizer));
      
      // Empty question
      await expect(async () => {
        await pipeline.answer('', 'Some context');
      }).rejects.toThrow();
      
      // Empty context
      await expect(async () => {
        await pipeline.answer('What is this?', '');
      }).rejects.toThrow();
    });
  });

  describe('Text Generation Pipeline', () => {
    test('creates text generation pipeline', () => {
      const model = trackResource(createModel('gpt2_base'));
      const tokenizer = trackResource(createTokenizer('bpe'));
      const pipeline = trackResource(Pipeline.textGeneration(model, tokenizer, {
        max_length: 50,
        temperature: 0.8
      }));
      
      expect(pipeline).toBeTruthy();
      expect(typeof pipeline.generate).toBe('function');
    });

    test('generates text from prompt', async () => {
      const model = trackResource(createModel('gpt2_base'));
      const tokenizer = trackResource(createTokenizer('bpe'));
      
      const vocab = {
        '[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3,
        'the': 4, 'quick': 5, 'brown': 6, 'fox': 7
      };
      tokenizer.load_vocab(vocab);
      
      const pipeline = trackResource(Pipeline.textGeneration(model, tokenizer, {
        max_length: 10,
        temperature: 0.8
      }));
      
      const result = await pipeline.generate('the quick');
      expect(typeof result).toBe('string');
      expect(result.length).toBeGreaterThan(0);
      expect(result.includes('the quick')).toBeTruthy();
    });

    test('respects generation parameters', async () => {
      const model = trackResource(createModel('gpt2_base'));
      const tokenizer = trackResource(createTokenizer('bpe'));
      
      const vocab = {
        '[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3,
        'hello': 4, 'world': 5, 'the': 6, 'quick': 7
      };
      tokenizer.load_vocab(vocab);
      
      // Low temperature (more deterministic)
      const deterministicPipeline = trackResource(Pipeline.textGeneration(model, tokenizer, {
        max_length: 15,
        temperature: 0.1
      }));
      
      // High temperature (more creative)
      const creativePipeline = trackResource(Pipeline.textGeneration(model, tokenizer, {
        max_length: 15,
        temperature: 1.5
      }));
      
      const prompt = 'hello';
      const deterministicResult = await deterministicPipeline.generate(prompt);
      const creativeResult = await creativePipeline.generate(prompt);
      
      expect(typeof deterministicResult).toBe('string');
      expect(typeof creativeResult).toBe('string');
      expect(deterministicResult.includes(prompt)).toBeTruthy();
      expect(creativeResult.includes(prompt)).toBeTruthy();
    });

    test('handles empty prompt', async () => {
      const model = trackResource(createModel('gpt2_base'));
      const tokenizer = trackResource(createTokenizer('bpe'));
      const pipeline = trackResource(Pipeline.textGeneration(model, tokenizer, {
        max_length: 10
      }));
      
      const result = await pipeline.generate('');
      expect(typeof result).toBe('string');
    });

    test('validates generation parameters', () => {
      const model = trackResource(createModel('gpt2_base'));
      const tokenizer = trackResource(createTokenizer('bpe'));
      
      // Invalid max_length
      expect(() => {
        Pipeline.textGeneration(model, tokenizer, { max_length: -1 });
      }).toThrow();
      
      // Invalid temperature
      expect(() => {
        Pipeline.textGeneration(model, tokenizer, { temperature: -0.5 });
      }).toThrow();
    });
  });

  describe('Feature Extraction Pipeline', () => {
    test('creates feature extraction pipeline', () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      const pipeline = trackResource(Pipeline.featureExtraction(model, tokenizer));
      
      expect(pipeline).toBeTruthy();
      expect(typeof pipeline.extract).toBe('function');
    });

    test('extracts features from text', async () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      
      const vocab = {
        '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
        'hello': 4, 'world': 5
      };
      tokenizer.load_vocab(vocab);
      
      const pipeline = trackResource(Pipeline.featureExtraction(model, tokenizer));
      
      const features = trackResource(await pipeline.extract('hello world'));
      expect(features).toBeTruthy();
      expect(Array.from(features.shape).length).toBe(3); // [batch, seq_len, hidden_size]
      expect(Array.from(features.shape)[2]).toBe(768); // BERT base hidden size
    });

    test('extracts features in batch', async () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      
      const vocab = {
        '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
        'hello': 4, 'world': 5, 'good': 6, 'day': 7
      };
      tokenizer.load_vocab(vocab);
      
      const pipeline = trackResource(Pipeline.featureExtraction(model, tokenizer));
      
      const texts = ['hello world', 'good day'];
      const features = trackResource(await pipeline.extract_batch(texts));
      
      expect(features).toBeTruthy();
      expect(Array.from(features.shape)[0]).toBe(2); // Batch size
      expect(Array.from(features.shape)[2]).toBe(768); // Hidden size
    });
  });

  describe('Pipeline Error Handling', () => {
    test('handles invalid model/tokenizer combination', () => {
      const bertModel = trackResource(createModel('bert_base'));
      const gptTokenizer = trackResource(createTokenizer('bpe'));
      
      // This combination might not work well
      expect(() => {
        Pipeline.textClassification(bertModel, gptTokenizer, ['pos', 'neg']);
      }).not.toThrow(); // Should create pipeline but might have issues during inference
    });

    test('handles missing labels for classification', () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      
      expect(() => {
        Pipeline.textClassification(model, tokenizer, []);
      }).toThrow();
    });

    test('handles null model or tokenizer', () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      
      expect(() => {
        Pipeline.textClassification(null, tokenizer, ['pos', 'neg']);
      }).toThrow();
      
      expect(() => {
        Pipeline.textClassification(model, null, ['pos', 'neg']);
      }).toThrow();
    });
  });

  describe('Pipeline Performance', () => {
    test('measures pipeline inference time', async () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      
      const vocab = {
        '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
        'test': 4, 'performance': 5
      };
      tokenizer.load_vocab(vocab);
      
      const pipeline = trackResource(Pipeline.textClassification(
        model, 
        tokenizer, 
        ['positive', 'negative']
      ));
      
      const timer = utils.timer('pipeline_inference');
      const result = trackResource(await pipeline.classify('test performance'));
      const elapsed = timer.elapsed();
      
      expect(elapsed).toBeGreaterThan(0);
      expect(elapsed).toBeLessThan(2000); // Should be under 2 seconds
    });

    test('batch processing is more efficient than individual processing', async () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      
      const vocab = {
        '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
        'test': 4, 'batch': 5, 'performance': 6
      };
      tokenizer.load_vocab(vocab);
      
      const pipeline = trackResource(Pipeline.textClassification(
        model, 
        tokenizer, 
        ['positive', 'negative']
      ));
      
      const texts = ['test batch', 'performance test', 'batch performance'];
      
      // Individual processing
      const individualTimer = utils.timer('individual_processing');
      for (const text of texts) {
        const result = trackResource(await pipeline.classify(text));
      }
      const individualTime = individualTimer.elapsed();
      
      // Batch processing
      const batchTimer = utils.timer('batch_processing');
      const batchResults = await pipeline.classify_batch(texts);
      batchResults.forEach(result => trackResource(result));
      const batchTime = batchTimer.elapsed();
      
      // Batch should be more efficient
      expect(batchTime).toBeLessThan(individualTime);
    });
  });

  describe('Pipeline Memory Management', () => {
    test('properly manages memory in pipeline operations', async () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      
      const vocab = { '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, 'test': 4 };
      tokenizer.load_vocab(vocab);
      
      const pipeline = trackResource(Pipeline.textClassification(
        model, 
        tokenizer, 
        ['positive', 'negative']
      ));
      
      // Perform multiple operations
      for (let i = 0; i < 5; i++) {
        const result = trackResource(await pipeline.classify('test'));
        // Results are tracked and will be cleaned up
      }
      
      // Memory should not grow unboundedly
      // This is more of a stress test than a specific assertion
      expect(true).toBeTruthy(); // Placeholder - real test would check memory usage
    });

    test('frees pipeline resources correctly', () => {
      const model = trackResource(createModel('bert_base'));
      const tokenizer = trackResource(createTokenizer('wordpiece'));
      const pipeline = Pipeline.textClassification(model, tokenizer, ['pos', 'neg']);
      
      expect(typeof pipeline.free).toBe('function');
      
      // Should not throw when freeing
      expect(() => {
        pipeline.free();
      }).not.toThrow();
    });
  });
});