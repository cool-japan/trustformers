/**
 * Streaming and Async Processing Demo for TrustformeRS JavaScript API
 * Demonstrates streaming text generation, async batch processing, and modern JavaScript patterns
 */

import { 
  initialize, 
  createModel,
  createTokenizer,
  Pipeline,
  streaming,
  async_utils,
  tensor_utils,
  memory,
  utils
} from '../src/index.js';

async function main() {
  console.log('🌊 TrustformeRS Streaming & Async Processing Demo');
  console.log('================================================');
  
  // Initialize the WASM module
  await initialize({
    wasmPath: '../pkg/trustformers_wasm_bg.wasm',
    initPanicHook: true
  });
  
  console.log('✅ TrustformeRS initialized successfully');
  console.log('📊 Initial memory usage:', memory.getStats());
  
  // Create a model and tokenizer for demonstrations
  console.log('\n1. Model and Tokenizer Setup');
  console.log('----------------------------');
  
  const model = createModel('gpt2_base');
  const tokenizer = createTokenizer('bpe');
  
  console.log('Model architecture:', model.config.architecture);
  console.log('Tokenizer type:', tokenizer.constructor.name);
  
  // Demo 1: Streaming text generation
  console.log('\n2. Streaming Text Generation');
  console.log('-----------------------------');
  
  const prompt = "The future of artificial intelligence is";
  console.log('Prompt:', prompt);
  console.log('Streaming response:');
  
  try {
    // Create a streaming generator
    const generator = streaming.textGeneration(model, tokenizer, {
      max_length: 100,
      temperature: 0.8,
      top_k: 50,
      top_p: 0.9
    });
    
    let fullText = '';
    let chunkCount = 0;
    
    for await (const chunk of generator) {
      process.stdout.write(chunk);
      fullText += chunk;
      chunkCount++;
      
      // Add some delay to simulate real-time streaming
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    
    console.log('\n');
    console.log('📝 Generated text length:', fullText.length);
    console.log('🔢 Total chunks:', chunkCount);
    
  } catch (error) {
    console.log('Note: Streaming generation requires WASM implementation');
    console.log('This is a demonstration of the API structure');
  }
  
  // Demo 2: Streaming tokenization
  console.log('\n3. Streaming Tokenization');
  console.log('-------------------------');
  
  const longText = "Natural language processing is a fascinating field that combines linguistics, computer science, and artificial intelligence to help machines understand and generate human language.";
  console.log('Text to tokenize:', longText);
  console.log('Streaming tokens:');
  
  try {
    const tokens = [];
    for await (const token of streaming.tokenize(tokenizer, longText)) {
      tokens.push(token);
      process.stdout.write(`${token} `);
    }
    console.log('\n');
    console.log('📊 Total tokens:', tokens.length);
    
  } catch (error) {
    console.log('Note: Streaming tokenization requires WASM implementation');
    console.log('This is a demonstration of the API structure');
  }
  
  // Demo 3: Async batch processing
  console.log('\n4. Async Batch Processing');
  console.log('-------------------------');
  
  const sentences = [
    "This is a positive sentence.",
    "I love this product!",
    "This is terrible and I hate it.",
    "Neutral statement here.",
    "Amazing quality and great service!",
    "Could be better, but not bad.",
    "Absolutely fantastic experience!",
    "This is just okay.",
    "Worst purchase ever made.",
    "Perfect in every way!"
  ];
  
  console.log('Processing', sentences.length, 'sentences in batches...');
  
  // Simulate async text processing
  const processText = async (text) => {
    // Simulate network delay or heavy computation
    await new Promise(resolve => setTimeout(resolve, Math.random() * 100));
    
    // Simulate sentiment analysis
    const positiveWords = ['love', 'amazing', 'great', 'fantastic', 'perfect'];
    const negativeWords = ['hate', 'terrible', 'worst', 'bad'];
    
    const words = text.toLowerCase().split(' ');
    let score = 0;
    
    words.forEach(word => {
      if (positiveWords.some(pos => word.includes(pos))) score += 1;
      if (negativeWords.some(neg => word.includes(neg))) score -= 1;
    });
    
    return {
      text: text,
      sentiment: score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral',
      score: score
    };
  };
  
  const timer = utils.timer('batch_processing');
  
  const results = await async_utils.processBatch(
    sentences,
    processText,
    3 // Process in batches of 3
  );
  
  timer.log_elapsed();
  
  console.log('\n📊 Processing Results:');
  results.forEach((result, i) => {
    console.log(`${i + 1}. ${result.sentiment.toUpperCase()}: ${result.text}`);
  });
  
  // Demo 4: Async inference with timeout and cleanup
  console.log('\n5. Async Inference with Management');
  console.log('----------------------------------');
  
  // Create some test tensors
  const testInputs = [
    tensor_utils.random([1, 10], 'normal'),
    tensor_utils.random([1, 10], 'normal'),
    tensor_utils.random([1, 10], 'normal')
  ];
  
  console.log('Running inference with automatic cleanup...');
  
  try {
    const inferencePromises = testInputs.map(async (input, i) => {
      console.log(`Starting inference ${i + 1}...`);
      
      try {
        const result = await async_utils.runInference(model, input, {
          autoCleanup: true,
          timeout: 5000  // 5 second timeout
        });
        
        console.log(`✅ Inference ${i + 1} completed`);
        return result;
        
      } catch (error) {
        console.log(`❌ Inference ${i + 1} failed:`, error.message);
        return null;
      }
    });
    
    const results = await Promise.all(inferencePromises);
    const successCount = results.filter(r => r !== null).length;
    console.log(`📊 Successful inferences: ${successCount}/${testInputs.length}`);
    
  } catch (error) {
    console.log('Note: Async inference requires WASM implementation');
    console.log('This is a demonstration of the API structure');
  }
  
  // Demo 5: Memory monitoring during processing
  console.log('\n6. Memory Monitoring');
  console.log('-------------------');
  
  const memoryBefore = memory.getStats();
  console.log('Memory before tensor operations:', memoryBefore);
  
  // Create and process many tensors
  const tensors = [];
  for (let i = 0; i < 50; i++) {
    tensors.push(tensor_utils.random([100, 100], 'normal'));
  }
  
  const memoryPeak = memory.getStats();
  console.log('Memory at peak usage:', memoryPeak);
  
  // Process tensors in batches to avoid memory issues
  const processedTensors = await async_utils.processBatch(
    tensors,
    async (tensor) => {
      // Simulate some processing
      await new Promise(resolve => setTimeout(resolve, 1));
      const result = tensor.mean();
      tensor.free(); // Clean up immediately
      return result;
    },
    10 // Small batch size to manage memory
  );
  
  const memoryAfter = memory.getStats();
  console.log('Memory after processing:', memoryAfter);
  console.log('📊 Processed tensors:', processedTensors.length);
  
  // Demo 6: Error handling and recovery
  console.log('\n7. Error Handling and Recovery');
  console.log('-----------------------------');
  
  const faultyOperations = [
    () => Promise.resolve('success'),
    () => Promise.reject(new Error('Operation failed')),
    () => Promise.resolve('success'),
    () => Promise.reject(new Error('Timeout')),
    () => Promise.resolve('success')
  ];
  
  console.log('Processing operations with error handling...');
  
  const robustProcess = async (operation, index) => {
    try {
      const result = await operation();
      console.log(`✅ Operation ${index + 1}: ${result}`);
      return result;
    } catch (error) {
      console.log(`❌ Operation ${index + 1} failed: ${error.message}`);
      return null;
    }
  };
  
  const robustResults = await async_utils.processBatch(
    faultyOperations,
    robustProcess,
    2
  );
  
  const successful = robustResults.filter(r => r !== null).length;
  console.log(`📊 Success rate: ${successful}/${faultyOperations.length}`);
  
  // Demo 7: Performance monitoring
  console.log('\n8. Performance Monitoring');
  console.log('-------------------------');
  
  const performanceTest = async () => {
    const timer = utils.timer('performance_test');
    
    // Simulate various operations
    const operations = [
      () => new Promise(resolve => setTimeout(resolve, 100)),
      () => new Promise(resolve => setTimeout(resolve, 200)),
      () => new Promise(resolve => setTimeout(resolve, 50)),
      () => new Promise(resolve => setTimeout(resolve, 150)),
      () => new Promise(resolve => setTimeout(resolve, 75))
    ];
    
    console.log('Running performance test...');
    
    const results = await async_utils.processBatch(
      operations,
      async (op, i) => {
        const opTimer = utils.timer(`operation_${i + 1}`);
        await op();
        const elapsed = opTimer.elapsed();
        console.log(`  Operation ${i + 1}: ${elapsed}ms`);
        return elapsed;
      },
      3
    );
    
    timer.log_elapsed();
    
    const totalTime = results.reduce((sum, time) => sum + time, 0);
    const averageTime = totalTime / results.length;
    
    console.log(`📊 Total operation time: ${totalTime}ms`);
    console.log(`📊 Average operation time: ${averageTime.toFixed(2)}ms`);
  };
  
  await performanceTest();
  
  // Cleanup
  console.log('\n9. Cleanup');
  console.log('----------');
  
  [model, tokenizer, ...processedTensors].forEach(obj => {
    if (obj && obj.free) obj.free();
  });
  
  const finalMemory = memory.getStats();
  console.log('Final memory usage:', finalMemory);
  
  console.log('\n🎉 Streaming & Async Demo completed successfully!');
  console.log('This demonstrates the modern JavaScript patterns available in TrustformeRS');
}

// Run the demo
main().catch(console.error);