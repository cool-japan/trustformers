/**
 * Example: Pipeline Usage with TrustformeRS
 */

import { 
  initialize, 
  createModel, 
  createTokenizer,
  Pipeline,
  utils 
} from '../src/index.js';

async function runPipelineExamples() {
  try {
    // Initialize TrustformeRS
    console.log('Initializing TrustformeRS...');
    await initialize({
      wasmPath: '../pkg/trustformers_wasm_bg.wasm'
    });
    
    // Create models and tokenizers for different tasks
    const bertModel = createModel('bert_base');
    const gpt2Model = createModel('gpt2_base');
    
    // Create tokenizers with sample vocabularies
    const bertTokenizer = createTokenizer('wordpiece');
    const gpt2Tokenizer = createTokenizer('bpe');
    
    // Load sample vocabularies
    const sampleVocab = {
      '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
      'the': 5, 'quick': 6, 'brown': 7, 'fox': 8, 'jumps': 9,
      'over': 10, 'lazy': 11, 'dog': 12, 'what': 13, 'is': 14,
      'your': 15, 'name': 16, 'i': 17, 'love': 18, 'programming': 19,
      'machine': 20, 'learning': 21, 'great': 22, 'terrible': 23,
      'good': 24, 'bad': 25, 'happy': 26, 'sad': 27, 'angry': 28,
      'peaceful': 29, 'exciting': 30, 'boring': 31, 'when': 32,
      'where': 33, 'who': 34, 'why': 35, 'how': 36, 'paris': 37,
      'france': 38, 'capital': 39, 'city': 40, 'country': 41,
      'located': 42, 'in': 43, 'europe': 44, 'it': 45, 'beautiful': 46
    };
    
    bertTokenizer.load_vocab(sampleVocab);
    gpt2Tokenizer.load_vocab(sampleVocab);
    
    console.log('\n=== Text Classification Pipeline ===');
    
    // Create text classification pipeline
    const classificationPipeline = Pipeline.textClassification(
      bertModel, 
      bertTokenizer,
      ['positive', 'negative', 'neutral']
    );
    
    const sentiments = [
      "I love this product, it's amazing!",
      "This is terrible, I hate it.",
      "It's okay, nothing special."
    ];
    
    for (const text of sentiments) {
      try {
        console.log(`\nText: "${text}"`);
        const result = await classificationPipeline.classify(text);
        console.log(`Label: ${result.label}`);
        console.log(`Score: ${result.score.toFixed(4)}`);
        console.log(`All scores:`, Array.from(result.all_scores).map(s => s.toFixed(4)));
        result.free();
      } catch (e) {
        console.log(`Classification failed: ${e.message}`);
      }
    }
    
    // Batch classification
    console.log('\n--- Batch Classification ---');
    try {
      const batchResults = await classificationPipeline.classify_batch(sentiments);
      batchResults.forEach((result, i) => {
        console.log(`Text ${i}: ${result.label} (${result.score.toFixed(4)})`);
        result.free();
      });
    } catch (e) {
      console.log(`Batch classification failed: ${e.message}`);
    }
    
    console.log('\n=== Text Generation Pipeline ===');
    
    // Create text generation pipeline
    const generationPipeline = Pipeline.textGeneration(gpt2Model, gpt2Tokenizer, {
      max_length: 50,
      temperature: 0.8,
      top_p: 0.9,
      do_sample: true
    });
    
    const prompts = [
      "The quick brown fox",
      "Machine learning is",
      "In the future, we will"
    ];
    
    for (const prompt of prompts) {
      try {
        console.log(`\nPrompt: "${prompt}"`);
        const generated = await generationPipeline.generate(prompt);
        console.log(`Generated: "${generated}"`);
      } catch (e) {
        console.log(`Generation failed: ${e.message}`);
      }
    }
    
    console.log('\n=== Question Answering Pipeline ===');
    
    // Create question answering pipeline
    const qaPipeline = Pipeline.questionAnswering(bertModel, bertTokenizer);
    
    const qaExamples = [
      {
        question: "What is the capital of France?",
        context: "Paris is the capital city of France. It is located in Europe and is known for its beautiful architecture."
      },
      {
        question: "Where is Paris located?",
        context: "Paris is the capital city of France. It is located in Europe and is known for its beautiful architecture."
      }
    ];
    
    for (const example of qaExamples) {
      try {
        console.log(`\nQuestion: "${example.question}"`);
        console.log(`Context: "${example.context}"`);
        const answer = await qaPipeline.answer(example.question, example.context);
        console.log(`Answer: "${answer.answer}"`);
        console.log(`Score: ${answer.score.toFixed(4)}`);
        console.log(`Position: ${answer.start}-${answer.end}`);
        answer.free();
      } catch (e) {
        console.log(`QA failed: ${e.message}`);
      }
    }
    
    console.log('\n=== Pipeline Performance Comparison ===');
    
    // Compare performance of different pipelines
    const perfTimer = utils.timer('pipeline_performance');
    const numIterations = 5;
    
    console.log(`\nRunning ${numIterations} iterations of each pipeline...`);
    
    // Classification performance
    perfTimer.elapsed(); // Reset
    for (let i = 0; i < numIterations; i++) {
      try {
        const result = await classificationPipeline.classify(sentiments[0]);
        result.free();
      } catch (e) {}
    }
    const classificationTime = perfTimer.elapsed();
    console.log(`Classification: ${(classificationTime / numIterations).toFixed(2)}ms per inference`);
    
    // Generation performance  
    perfTimer.elapsed(); // Reset
    for (let i = 0; i < numIterations; i++) {
      try {
        await generationPipeline.generate(prompts[0]);
      } catch (e) {}
    }
    const generationTime = perfTimer.elapsed() - classificationTime;
    console.log(`Generation: ${(generationTime / numIterations).toFixed(2)}ms per inference`);
    
    // QA performance
    perfTimer.elapsed(); // Reset
    for (let i = 0; i < numIterations; i++) {
      try {
        const answer = await qaPipeline.answer(qaExamples[0].question, qaExamples[0].context);
        answer.free();
      } catch (e) {}
    }
    const qaTime = perfTimer.elapsed() - generationTime - classificationTime;
    console.log(`Question Answering: ${(qaTime / numIterations).toFixed(2)}ms per inference`);
    
    // Clean up
    console.log('\n=== Cleanup ===');
    classificationPipeline.free();
    generationPipeline.free();
    qaPipeline.free();
    bertTokenizer.free();
    gpt2Tokenizer.free();
    bertModel.free();
    gpt2Model.free();
    
    console.log('Done!');
    
  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the example
runPipelineExamples();