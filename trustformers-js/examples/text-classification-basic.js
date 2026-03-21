/**
 * Basic Text Classification Example
 * 
 * This example demonstrates simple text classification using TrustformeRS.
 * It shows how to classify text into predefined categories.
 */

import { 
  initialize, 
  createModel, 
  createTokenizer,
  Pipeline,
  utils 
} from '../src/index.js';

async function runTextClassification() {
  try {
    console.log('🚀 Starting Text Classification Example\n');
    
    // Initialize TrustformeRS
    console.log('Initializing TrustformeRS...');
    await initialize({
      wasmPath: '../pkg/trustformers_wasm_bg.wasm'
    });
    console.log('✅ TrustformeRS initialized\n');
    
    // Create BERT model for classification
    console.log('Creating BERT model...');
    const model = createModel('bert_base');
    const tokenizer = createTokenizer('wordpiece');
    
    // Load vocabulary for demonstration
    const vocab = {
      '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
      'this': 5, 'is': 6, 'great': 7, 'amazing': 8, 'wonderful': 9,
      'terrible': 10, 'awful': 11, 'bad': 12, 'horrible': 13, 'hate': 14,
      'love': 15, 'like': 16, 'enjoy': 17, 'excellent': 18, 'perfect': 19,
      'worst': 20, 'best': 21, 'good': 22, 'poor': 23, 'fantastic': 24,
      'disappointing': 25, 'impressive': 26, 'mediocre': 27, 'outstanding': 28,
      'product': 29, 'service': 30, 'experience': 31, 'quality': 32,
      'the': 33, 'a': 34, 'an': 35, 'i': 36, 'it': 37, 'was': 38, 'very': 39
    };
    
    tokenizer.load_vocab(vocab);
    console.log(`✅ Model and tokenizer ready (vocab size: ${tokenizer.vocab_size})\n`);
    
    // Create classification pipeline
    const classifier = Pipeline.textClassification(
      model, 
      tokenizer,
      ['positive', 'negative', 'neutral']
    );
    
    // Test texts for classification
    const testTexts = [
      "This product is amazing and works perfectly!",
      "Terrible experience, worst service ever.",
      "It's okay, nothing special about it.",
      "I love this! Excellent quality and great value.",
      "Very disappointing, poor quality product."
    ];
    
    console.log('📊 Classifying test texts:\n');
    
    // Classify each text
    for (let i = 0; i < testTexts.length; i++) {
      const text = testTexts[i];
      console.log(`${i + 1}. Text: "${text}"`);
      
      try {
        const timer = utils.timer(`classification_${i}`);
        const result = await classifier.classify(text);
        const elapsed = timer.elapsed();
        
        console.log(`   📈 Prediction: ${result.label.toUpperCase()}`);
        console.log(`   🎯 Confidence: ${(result.score * 100).toFixed(1)}%`);
        console.log(`   ⏱️  Time: ${elapsed.toFixed(2)}ms`);
        console.log(`   📊 All scores: ${Array.from(result.all_scores).map(s => (s * 100).toFixed(1) + '%').join(', ')}\n`);
        
        result.free();
      } catch (error) {
        console.log(`   ❌ Classification failed: ${error.message}\n`);
      }
    }
    
    // Batch classification demo
    console.log('⚡ Batch Classification Demo:\n');
    
    try {
      const batchTimer = utils.timer('batch_classification');
      const batchResults = await classifier.classify_batch(testTexts);
      const batchElapsed = batchTimer.elapsed();
      
      console.log(`✅ Classified ${testTexts.length} texts in ${batchElapsed.toFixed(2)}ms`);
      console.log(`📊 Average time per text: ${(batchElapsed / testTexts.length).toFixed(2)}ms\n`);
      
      batchResults.forEach((result, i) => {
        console.log(`${i + 1}. ${result.label.toUpperCase()} (${(result.score * 100).toFixed(1)}%)`);
        result.free();
      });
      
    } catch (error) {
      console.log(`❌ Batch classification failed: ${error.message}`);
    }
    
    console.log('\n🎯 Classification Tips:');
    console.log('• Higher confidence scores indicate more certain predictions');
    console.log('• Batch processing is more efficient for multiple texts');
    console.log('• Pre-trained models work better with domain-specific fine-tuning');
    
    // Cleanup
    console.log('\n🧹 Cleaning up resources...');
    classifier.free();
    tokenizer.free();
    model.free();
    
    console.log('✅ Text Classification Example completed successfully!');
    
  } catch (error) {
    console.error('❌ Error in text classification example:', error);
    console.error('Stack trace:', error.stack);
  }
}

// Check if running directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTextClassification();
}

export { runTextClassification };