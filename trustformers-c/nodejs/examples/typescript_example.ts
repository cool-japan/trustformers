/**
 * TypeScript example for TrustformeRS Node.js bindings
 * Demonstrates type-safe usage and advanced features
 */

import {
  trustformers,
  TrustformeRS,
  Model,
  Tokenizer,
  Pipeline,
  createTextGenerationPipeline,
  createClassificationPipeline,
  loadModel,
  loadTokenizer,
  quickSetup,
  benchmark,
  getLibraryStatus,
  GenerationConfig,
  ModelConfig,
  TokenizerConfig,
  PipelineConfig,
  TrustformersOptimizationConfig,
  TrustformersNativeError,
  TrustformersError
} from '../src';

/**
 * Example class demonstrating advanced usage patterns
 */
class TrustformersExample {
  private api: TrustformeRS;
  private model?: Model;
  private tokenizer?: Tokenizer;
  private pipeline?: Pipeline;

  constructor() {
    this.api = TrustformeRS.getInstance();
  }

  /**
   * Initialize the example with optimizations
   */
  public async initialize(): Promise<void> {
    console.log('Initializing TrustformeRS with TypeScript...\n');

    // Setup with type-safe configuration
    quickSetup({
      logLevel: 3,
      memoryLimitMb: 2048,
      enableProfiling: true,
      optimizationLevel: 3
    });

    // Display library information
    this.displayLibraryInfo();
  }

  /**
   * Display comprehensive library information
   */
  private displayLibraryInfo(): void {
    const status = getLibraryStatus();
    
    console.log('Library Status:');
    console.log('===============');
    console.log(`Version: ${status.version}`);
    console.log(`Build Date: ${status.buildInfo.buildDate}`);
    console.log(`Target: ${status.buildInfo.target}`);
    console.log(`Features: ${status.buildInfo.features}`);
    console.log(`Platform: ${status.platformInfo.platform} (${status.platformInfo.arch})`);
    console.log(`CPU Cores: ${status.platformInfo.numCores}`);
    console.log(`Total Memory: ${(status.platformInfo.totalMemory / 1024 / 1024 / 1024).toFixed(2)} GB`);
    console.log(`GPU Support: ${status.platformInfo.hasGpu ? 'Yes' : 'No'}`);
    console.log(`CUDA Support: ${status.platformInfo.hasCuda ? 'Yes' : 'No'}`);
    console.log('');
  }

  /**
   * Demonstrate type-safe model loading and configuration
   */
  public async loadModelExample(): Promise<void> {
    console.log('Model Loading Example:');
    console.log('=====================');

    try {
      const modelConfig: ModelConfig = {
        modelPath: 'gpt2',
        device: 'auto',
        quantization: 'fp16',
        batchSize: 4,
        maxLength: 1024,
        numThreads: 0 // Auto-detect
      };

      this.model = loadModel(modelConfig.modelPath, {
        device: modelConfig.device,
        quantization: modelConfig.quantization,
        numThreads: modelConfig.numThreads
      });

      // Get model metadata with type safety
      const metadata = this.model.getMetadata();
      console.log('Model Metadata:', JSON.stringify(metadata, null, 2));

      const config = this.model.getConfig();
      console.log('Model Config:', JSON.stringify(config, null, 2));

      // Validate model
      const validation = this.model.validate();
      console.log('Model Validation:', JSON.stringify(validation, null, 2));

    } catch (error) {
      if (error instanceof TrustformersNativeError) {
        console.log(`Model loading failed with code ${error.code}: ${error.message}`);
        console.log('Context:', error.context);
      } else {
        console.log('Model loading failed:', (error as Error).message);
      }
    }
    console.log('');
  }

  /**
   * Demonstrate tokenizer usage with type safety
   */
  public async tokenizerExample(): Promise<void> {
    console.log('Tokenizer Example:');
    console.log('=================');

    try {
      const tokenizerConfig: TokenizerConfig = {
        tokenizerPath: 'tokenizer.json',
        addSpecialTokens: true,
        padding: 'max_length',
        truncation: true,
        maxLength: 512,
        returnTensors: 'array'
      };

      this.tokenizer = loadTokenizer(tokenizerConfig.tokenizerPath, {
        addSpecialTokens: tokenizerConfig.addSpecialTokens,
        padding: tokenizerConfig.padding,
        truncation: tokenizerConfig.truncation,
        maxLength: tokenizerConfig.maxLength
      });

      const text = "Hello, TypeScript world! This is a test of type-safe tokenization.";
      console.log(`Input: "${text}"`);

      // Type-safe encoding
      const encoded = this.tokenizer.encode(text, {
        addSpecialTokens: true,
        padding: false,
        truncation: true,
        maxLength: 100,
        returnSpecialTokensMask: true,
        returnAttentionMask: true
      });

      console.log('Encoded IDs:', encoded.inputIds);
      console.log('Attention Mask:', encoded.attentionMask);
      console.log('Special Tokens Mask:', encoded.specialTokensMask);

      // Type-safe decoding
      const decoded = this.tokenizer.decode(encoded.inputIds, {
        skipSpecialTokens: true,
        cleanUpTokenizationSpaces: true
      });

      console.log(`Decoded: "${decoded}"`);
      console.log(`Vocabulary Size: ${this.tokenizer.getVocabSize()}`);

      // Batch encoding example
      const texts = [
        "First example text",
        "Second example with different length",
        "Third short text"
      ];

      const batchEncoded = this.tokenizer.encodeBatch(texts, {
        padding: 'longest',
        truncation: true,
        maxLength: 50
      });

      console.log('Batch Encoded:', batchEncoded.map(result => ({
        length: result.inputIds.length,
        tokens: result.inputIds.slice(0, 5) // Show first 5 tokens
      })));

    } catch (error) {
      console.log('Tokenizer example failed:', (error as Error).message);
    }
    console.log('');
  }

  /**
   * Demonstrate pipeline usage with different tasks
   */
  public async pipelineExample(): Promise<void> {
    console.log('Pipeline Example:');
    console.log('================');

    try {
      // Text generation pipeline
      await this.textGenerationExample();
      await this.classificationExample();
      await this.conversationExample();
      await this.streamingExample();

    } catch (error) {
      console.log('Pipeline example failed:', (error as Error).message);
    }
    console.log('');
  }

  /**
   * Text generation pipeline example
   */
  private async textGenerationExample(): Promise<void> {
    console.log('Text Generation:');
    console.log('---------------');

    try {
      const pipeline = await createTextGenerationPipeline('gpt2', {
        device: 'cpu',
        maxLength: 150
      });

      const generationConfig: GenerationConfig = {
        maxLength: 100,
        temperature: 0.8,
        topK: 50,
        topP: 0.95,
        repetitionPenalty: 1.1,
        doSample: true,
        numBeams: 1,
        earlyStopping: false
      };

      const input = "The future of artificial intelligence";
      const result = await pipeline.run(input, generationConfig);

      console.log(`Input: "${input}"`);
      console.log(`Generated: "${result.text || result}"`);
      console.log(`Finish Reason: ${result.finishReason || 'unknown'}`);

      pipeline.dispose();

    } catch (error) {
      console.log('Text generation failed:', (error as Error).message);
    }
  }

  /**
   * Classification pipeline example
   */
  private async classificationExample(): Promise<void> {
    console.log('\nText Classification:');
    console.log('-------------------');

    try {
      const pipeline = await createClassificationPipeline('distilbert-base-uncased', {
        device: 'cpu'
      });

      const texts = [
        "I love this product, it's amazing!",
        "This is terrible, I hate it.",
        "It's okay, could be better."
      ];

      const results = await pipeline.runBatch(texts);
      
      texts.forEach((text, index) => {
        const result = results[index];
        console.log(`Text: "${text}"`);
        console.log(`Classification: ${result.label} (${(result.score * 100).toFixed(2)}%)`);
      });

      pipeline.dispose();

    } catch (error) {
      console.log('Classification failed:', (error as Error).message);
    }
  }

  /**
   * Conversation pipeline example
   */
  private async conversationExample(): Promise<void> {
    console.log('\nConversation:');
    console.log('------------');

    try {
      const pipeline = await createTextGenerationPipeline('gpt2');

      const conversationId = pipeline.startConversation(
        "You are a helpful AI assistant. Be concise and friendly."
      );

      // Add user turn
      pipeline.addConversationTurn(conversationId, 'user', 'Hello! How are you?');
      
      const response1 = await pipeline.generateConversationResponse(conversationId, {
        maxLength: 50,
        temperature: 0.7
      });

      console.log('User: Hello! How are you?');
      console.log('Assistant:', response1.text || response1);

      // Add assistant response and continue conversation
      pipeline.addConversationTurn(conversationId, 'assistant', response1.text || response1);
      pipeline.addConversationTurn(conversationId, 'user', 'What can you help me with?');

      const response2 = await pipeline.generateConversationResponse(conversationId, {
        maxLength: 80,
        temperature: 0.7
      });

      console.log('User: What can you help me with?');
      console.log('Assistant:', response2.text || response2);

      pipeline.dispose();

    } catch (error) {
      console.log('Conversation failed:', (error as Error).message);
    }
  }

  /**
   * Streaming generation example
   */
  private async streamingExample(): Promise<void> {
    console.log('\nStreaming Generation:');
    console.log('--------------------');

    try {
      const pipeline = await createTextGenerationPipeline('gpt2');

      const input = "In the distant future";
      console.log(`Input: "${input}"`);
      console.log('Streaming output: ');

      await pipeline.stream(input, {
        maxLength: 80,
        temperature: 0.8
      }, (chunk: string, isComplete: boolean) => {
        process.stdout.write(chunk);
        if (isComplete) {
          console.log('\n[Stream complete]');
        }
      });

      pipeline.dispose();

    } catch (error) {
      console.log('Streaming failed:', (error as Error).message);
    }
  }

  /**
   * Performance optimization example
   */
  public async optimizationExample(): Promise<void> {
    console.log('Optimization Example:');
    console.log('====================');

    // Apply custom optimizations
    const optimizationConfig: TrustformersOptimizationConfig = {
      enableTracking: true,
      enableCaching: true,
      cacheSizeMb: 512,
      numThreads: 0, // Auto-detect
      enableSimd: true,
      optimizeBatchSize: true,
      memoryOptimizationLevel: 3
    };

    this.api.applyOptimizations(optimizationConfig);

    // Get and display performance metrics
    const metrics = this.api.getPerformanceMetrics();
    console.log('Performance Metrics:');
    console.log(`- Operations: ${metrics.totalOperations}`);
    console.log(`- Avg Time: ${metrics.avgOperationTimeMs.toFixed(2)} ms`);
    console.log(`- Cache Hit Rate: ${(metrics.cacheHitRate * 100).toFixed(2)}%`);
    console.log(`- Performance Score: ${metrics.performanceScore.toFixed(2)}/100`);
    console.log(`- Optimization Hints: ${metrics.numOptimizationHints}`);

    if (metrics.optimizationHintsJson) {
      const hints = JSON.parse(metrics.optimizationHintsJson);
      if (hints.length > 0) {
        console.log('Optimization Hints:');
        hints.forEach((hint: any, index: number) => {
          console.log(`  ${index + 1}. ${hint.description} (${hint.potential_improvement}% improvement)`);
        });
      }
    }
    console.log('');
  }

  /**
   * Memory monitoring example
   */
  public async memoryMonitoringExample(): Promise<void> {
    console.log('Memory Monitoring Example:');
    console.log('=========================');

    const basicUsage = this.api.getMemoryUsage();
    console.log('Basic Memory Usage:');
    console.log(`- Total: ${(basicUsage.totalMemoryBytes / 1024 / 1024).toFixed(2)} MB`);
    console.log(`- Peak: ${(basicUsage.peakMemoryBytes / 1024 / 1024).toFixed(2)} MB`);
    console.log(`- Models: ${basicUsage.allocatedModels}`);
    console.log(`- Tokenizers: ${basicUsage.allocatedTokenizers}`);
    console.log(`- Pipelines: ${basicUsage.allocatedPipelines}`);

    const advancedUsage = this.api.getAdvancedMemoryUsage();
    console.log('\nAdvanced Memory Usage:');
    console.log(`- Fragmentation: ${(advancedUsage.fragmentationRatio * 100).toFixed(2)}%`);
    console.log(`- Avg Allocation: ${advancedUsage.avgAllocationSize} bytes`);
    console.log(`- Pressure Level: ${this.getPressureLevelName(advancedUsage.pressureLevel)}`);
    console.log(`- Allocation Rate: ${advancedUsage.allocationRate.toFixed(2)}/min`);

    // Check for memory leaks
    const leakReport = this.api.checkMemoryLeaks();
    console.log('\nMemory Leak Check:');
    console.log(JSON.stringify(leakReport, null, 2));
    console.log('');
  }

  /**
   * Run a comprehensive benchmark
   */
  public async benchmarkExample(): Promise<void> {
    console.log('Benchmark Example:');
    console.log('=================');

    try {
      const benchmarkResult = await benchmark({
        modelPath: 'gpt2',
        inputText: 'The quick brown fox jumps over the lazy dog.',
        iterations: 5,
        warmupIterations: 2
      });

      console.log('Benchmark Results:');
      console.log(`- Total Time: ${benchmarkResult.totalTime} ms`);
      console.log(`- Average Time: ${benchmarkResult.averageTime.toFixed(2)} ms`);
      console.log(`- Iterations: ${benchmarkResult.iterations}`);
      console.log(`- Memory Usage: ${(benchmarkResult.memoryUsage.totalMemoryBytes / 1024 / 1024).toFixed(2)} MB`);
      console.log(`- Performance Score: ${benchmarkResult.performanceMetrics.performanceScore.toFixed(2)}`);

    } catch (error) {
      console.log('Benchmark failed:', (error as Error).message);
    }
    console.log('');
  }

  /**
   * Clean up all resources
   */
  public async cleanup(): Promise<void> {
    console.log('Cleaning up resources...');

    if (this.model) {
      this.model.dispose();
      this.model = undefined;
    }

    if (this.tokenizer) {
      this.tokenizer.dispose();
      this.tokenizer = undefined;
    }

    if (this.pipeline) {
      this.pipeline.dispose();
      this.pipeline = undefined;
    }

    this.api.memoryCleanup();
    this.api.gc();

    console.log('Cleanup complete!');
  }

  /**
   * Get human-readable pressure level name
   */
  private getPressureLevelName(level: number): string {
    switch (level) {
      case 0: return 'Low';
      case 1: return 'Medium';
      case 2: return 'High';
      case 3: return 'Critical';
      default: return 'Unknown';
    }
  }
}

/**
 * Main function to run all examples
 */
async function main(): Promise<void> {
  const example = new TrustformersExample();

  try {
    await example.initialize();
    await example.loadModelExample();
    await example.tokenizerExample();
    await example.pipelineExample();
    await example.optimizationExample();
    await example.memoryMonitoringExample();
    await example.benchmarkExample();

  } catch (error) {
    console.error('Example failed:', error);
  } finally {
    await example.cleanup();
  }
}

// Handle process signals for cleanup
process.on('SIGINT', async () => {
  console.log('\nReceived SIGINT, cleaning up...');
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\nReceived SIGTERM, cleaning up...');
  process.exit(0);
});

// Run the example if this file is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { TrustformersExample, main };