/**
 * Basic Text Generation Example
 * 
 * This example demonstrates text generation using TrustformeRS with GPT-style models.
 * It shows how to generate text continuations from prompts with various settings.
 */

import { 
  initialize, 
  createModel, 
  createTokenizer,
  Pipeline,
  utils 
} from '../src/index.js';

async function runTextGeneration() {
  try {
    console.log('✍️  Starting Text Generation Example\n');
    
    // Initialize TrustformeRS
    console.log('Initializing TrustformeRS...');
    await initialize({
      wasmPath: '../pkg/trustformers_wasm_bg.wasm'
    });
    console.log('✅ TrustformeRS initialized\n');
    
    // Create GPT model for text generation
    console.log('Creating GPT model for text generation...');
    const model = createModel('gpt2_base');
    const tokenizer = createTokenizer('bpe');
    
    // Load vocabulary for text generation
    const vocab = {
      '[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3,
      'the': 4, 'a': 5, 'an': 6, 'and': 7, 'or': 8, 'but': 9, 'in': 10,
      'on': 11, 'at': 12, 'to': 13, 'for': 14, 'of': 15, 'with': 16,
      'by': 17, 'from': 18, 'about': 19, 'into': 20, 'through': 21,
      'quick': 22, 'brown': 23, 'fox': 24, 'jumps': 25, 'over': 26,
      'lazy': 27, 'dog': 28, 'cat': 29, 'bird': 30, 'fish': 31,
      'machine': 32, 'learning': 33, 'artificial': 34, 'intelligence': 35,
      'computer': 36, 'science': 37, 'technology': 38, 'data': 39,
      'algorithm': 40, 'model': 41, 'neural': 42, 'network': 43,
      'future': 44, 'will': 45, 'can': 46, 'could': 47, 'should': 48,
      'would': 49, 'might': 50, 'may': 51, 'must': 52, 'shall': 53,
      'amazing': 54, 'incredible': 55, 'fantastic': 56, 'wonderful': 57,
      'beautiful': 58, 'powerful': 59, 'advanced': 60, 'innovative': 61,
      'revolutionary': 62, 'breakthrough': 63, 'discovery': 64,
      'research': 65, 'development': 66, 'progress': 67, 'evolution': 68,
      'transform': 69, 'change': 70, 'improve': 71, 'enhance': 72,
      'optimize': 73, 'accelerate': 74, 'enable': 75, 'provide': 76
    };
    
    tokenizer.load_vocab(vocab);
    console.log(`✅ Model and tokenizer ready (vocab size: ${tokenizer.vocab_size})\n`);
    
    // Create text generation pipeline with different configurations
    console.log('🎛️  Configuring generation pipelines:\n');
    
    const conservativeGenerator = Pipeline.textGeneration(model, tokenizer, {
      max_length: 30,
      temperature: 0.3,
      top_p: 0.8,
      do_sample: true
    });
    
    const creativeGenerator = Pipeline.textGeneration(model, tokenizer, {
      max_length: 40,
      temperature: 0.9,
      top_p: 0.95,
      do_sample: true
    });
    
    const deterministicGenerator = Pipeline.textGeneration(model, tokenizer, {
      max_length: 25,
      temperature: 0.0,
      do_sample: false
    });
    
    // Test prompts for generation
    const prompts = [
      "The quick brown fox",
      "Machine learning is",
      "In the future, artificial intelligence will",
      "The most amazing discovery in science",
      "Technology has the power to"
    ];
    
    console.log('📝 Conservative Generation (Low Temperature):\n');
    await runGenerationSet(conservativeGenerator, prompts, 'Conservative');
    
    console.log('\n🎨 Creative Generation (High Temperature):\n');
    await runGenerationSet(creativeGenerator, prompts, 'Creative');
    
    console.log('\n🎯 Deterministic Generation (Greedy):\n');
    await runGenerationSet(deterministicGenerator, prompts, 'Deterministic');
    
    // Interactive generation demo
    console.log('\n🔄 Interactive Generation Demo:\n');
    
    const interactivePrompts = [
      "Once upon a time",
      "The secret to happiness",
      "Scientists recently discovered"
    ];
    
    console.log('Generating with different settings for comparison...\n');
    
    for (const prompt of interactivePrompts) {
      console.log(`🌟 Prompt: "${prompt}"`);
      console.log('─'.repeat(50));
      
      try {
        // Conservative
        const conservative = await conservativeGenerator.generate(prompt);
        console.log(`🔒 Conservative: "${conservative}"`);
        
        // Creative  
        const creative = await creativeGenerator.generate(prompt);
        console.log(`🎨 Creative: "${creative}"`);
        
        // Deterministic
        const deterministic = await deterministicGenerator.generate(prompt);
        console.log(`🎯 Deterministic: "${deterministic}"\n`);
        
      } catch (error) {
        console.log(`❌ Generation failed for "${prompt}": ${error.message}\n`);
      }
    }
    
    // Performance comparison
    console.log('⚡ Performance Comparison:\n');
    
    const perfPrompt = "The future of technology";
    const numIterations = 5;
    
    console.log(`Testing "${perfPrompt}" with ${numIterations} iterations each:\n`);
    
    // Conservative performance
    const conservativeTimer = utils.timer('conservative_perf');
    for (let i = 0; i < numIterations; i++) {
      try {
        await conservativeGenerator.generate(perfPrompt);
      } catch (e) {}
    }
    const conservativeTime = conservativeTimer.elapsed();
    
    // Creative performance
    const creativeTimer = utils.timer('creative_perf');
    for (let i = 0; i < numIterations; i++) {
      try {
        await creativeGenerator.generate(perfPrompt);
      } catch (e) {}
    }
    const creativeTime = creativeTimer.elapsed();
    
    // Deterministic performance
    const detTimer = utils.timer('deterministic_perf');
    for (let i = 0; i < numIterations; i++) {
      try {
        await deterministicGenerator.generate(perfPrompt);
      } catch (e) {}
    }
    const detTime = detTimer.elapsed();
    
    console.log(`🔒 Conservative: ${(conservativeTime / numIterations).toFixed(2)}ms avg`);
    console.log(`🎨 Creative: ${(creativeTime / numIterations).toFixed(2)}ms avg`);
    console.log(`🎯 Deterministic: ${(detTime / numIterations).toFixed(2)}ms avg\n`);
    
    console.log('💡 Generation Tips:');
    console.log('• Lower temperature = more focused, predictable text');
    console.log('• Higher temperature = more creative, diverse text');
    console.log('• top_p controls nucleus sampling diversity');
    console.log('• Deterministic mode always produces the same output');
    console.log('• Longer max_length allows for more complete thoughts');
    
    // Cleanup
    console.log('\n🧹 Cleaning up resources...');
    conservativeGenerator.free();
    creativeGenerator.free();
    deterministicGenerator.free();
    tokenizer.free();
    model.free();
    
    console.log('✅ Text Generation Example completed successfully!');
    
  } catch (error) {
    console.error('❌ Error in text generation example:', error);
    console.error('Stack trace:', error.stack);
  }
}

// Helper function to run generation with a set of prompts
async function runGenerationSet(generator, prompts, name) {
  for (let i = 0; i < prompts.length; i++) {
    const prompt = prompts[i];
    console.log(`${i + 1}. Prompt: "${prompt}"`);
    
    try {
      const timer = utils.timer(`${name.toLowerCase()}_${i}`);
      const generated = await generator.generate(prompt);
      const elapsed = timer.elapsed();
      
      console.log(`   ✨ Generated: "${generated}"`);
      console.log(`   ⏱️  Time: ${elapsed.toFixed(2)}ms\n`);
      
    } catch (error) {
      console.log(`   ❌ Generation failed: ${error.message}\n`);
    }
  }
}

// Check if running directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTextGeneration();
}

export { runTextGeneration };