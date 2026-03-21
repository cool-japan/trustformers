/**
 * Basic usage example for TrustformeRS Node.js bindings
 */

const { trustformers, createTextGenerationPipeline, quickSetup } = require('../dist');

async function main() {
  try {
    console.log('TrustformeRS Node.js Bindings Example');
    console.log('====================================\n');
    
    // Quick setup with basic optimization
    console.log('Setting up TrustformeRS...');
    quickSetup({
      logLevel: 3, // Info level
      memoryLimitMb: 1024, // 1GB limit
      enableProfiling: true,
      optimizationLevel: 2
    });
    
    // Get library information
    console.log('Library Version:', trustformers.getVersion());
    console.log('Build Info:', trustformers.getBuildInfo());
    console.log('Platform Info:', trustformers.getPlatformInfo());
    console.log('Capabilities:', trustformers.getCapabilities());
    console.log('');
    
    // Example 1: Basic text generation
    console.log('Example 1: Text Generation');
    console.log('--------------------------');
    
    try {
      // Note: You'll need to provide actual model paths
      const pipeline = await createTextGenerationPipeline('gpt2', {
        device: 'cpu',
        maxLength: 100
      });
      
      const input = "The future of artificial intelligence is";
      console.log('Input:', input);
      
      const result = await pipeline.run(input, {
        maxLength: 50,
        temperature: 0.7,
        topK: 50
      });
      
      console.log('Generated text:', result.text || result);
      console.log('');
      
      // Cleanup
      pipeline.dispose();
    } catch (error) {
      console.log('Text generation example failed (model not found):', error.message);
      console.log('To run this example, provide a valid model path.');
      console.log('');
    }
    
    // Example 2: Memory usage monitoring
    console.log('Example 2: Memory Usage Monitoring');
    console.log('----------------------------------');
    
    const memoryUsage = trustformers.getMemoryUsage();
    console.log('Basic Memory Usage:');
    console.log('- Total Memory:', (memoryUsage.totalMemoryBytes / 1024 / 1024).toFixed(2), 'MB');
    console.log('- Peak Memory:', (memoryUsage.peakMemoryBytes / 1024 / 1024).toFixed(2), 'MB');
    console.log('- Allocated Models:', memoryUsage.allocatedModels);
    console.log('- Allocated Tokenizers:', memoryUsage.allocatedTokenizers);
    console.log('- Allocated Pipelines:', memoryUsage.allocatedPipelines);
    console.log('');
    
    const advancedMemoryUsage = trustformers.getAdvancedMemoryUsage();
    console.log('Advanced Memory Usage:');
    console.log('- Fragmentation Ratio:', (advancedMemoryUsage.fragmentationRatio * 100).toFixed(2) + '%');
    console.log('- Average Allocation Size:', advancedMemoryUsage.avgAllocationSize, 'bytes');
    console.log('- Memory Pressure Level:', advancedMemoryUsage.pressureLevel);
    console.log('- Allocation Rate:', advancedMemoryUsage.allocationRate.toFixed(2), 'allocs/min');
    console.log('');
    
    // Example 3: Performance metrics
    console.log('Example 3: Performance Metrics');
    console.log('------------------------------');
    
    const performanceMetrics = trustformers.getPerformanceMetrics();
    console.log('Performance Metrics:');
    console.log('- Total Operations:', performanceMetrics.totalOperations);
    console.log('- Average Operation Time:', performanceMetrics.avgOperationTimeMs.toFixed(2), 'ms');
    console.log('- Cache Hit Rate:', (performanceMetrics.cacheHitRate * 100).toFixed(2) + '%');
    console.log('- Performance Score:', performanceMetrics.performanceScore.toFixed(2));
    console.log('- Optimization Hints:', performanceMetrics.numOptimizationHints);
    console.log('');
    
    // Example 4: Tokenizer usage
    console.log('Example 4: Tokenizer Usage');
    console.log('--------------------------');
    
    try {
      const { Tokenizer } = require('../dist');
      
      // Note: You'll need to provide an actual tokenizer path
      const tokenizer = new Tokenizer('tokenizer.json');
      
      const text = "Hello, how are you today?";
      console.log('Input text:', text);
      
      const encoded = tokenizer.encode(text, {
        addSpecialTokens: true,
        padding: false,
        truncation: true,
        maxLength: 512
      });
      
      console.log('Encoded tokens:', encoded.inputIds);
      console.log('Token count:', encoded.inputIds.length);
      
      const decoded = tokenizer.decode(encoded.inputIds, {
        skipSpecialTokens: true
      });
      
      console.log('Decoded text:', decoded);
      console.log('Vocabulary size:', tokenizer.getVocabSize());
      console.log('');
      
      // Cleanup
      tokenizer.dispose();
    } catch (error) {
      console.log('Tokenizer example failed (tokenizer not found):', error.message);
      console.log('To run this example, provide a valid tokenizer path.');
      console.log('');
    }
    
    // Example 5: Batch processing
    console.log('Example 5: Batch Processing');
    console.log('---------------------------');
    
    try {
      const pipeline = await createTextGenerationPipeline('gpt2');
      
      const inputs = [
        "The weather today is",
        "Machine learning is",
        "The future of technology"
      ];
      
      console.log('Processing batch of', inputs.length, 'inputs...');
      const results = await pipeline.runBatch(inputs, {
        maxLength: 30,
        temperature: 0.8
      });
      
      results.forEach((result, index) => {
        console.log(`Result ${index + 1}:`, result.text || result);
      });
      console.log('');
      
      // Cleanup
      pipeline.dispose();
    } catch (error) {
      console.log('Batch processing example failed:', error.message);
      console.log('');
    }
    
    // Example 6: Memory leak detection
    console.log('Example 6: Memory Leak Detection');
    console.log('--------------------------------');
    
    const leakReport = trustformers.checkMemoryLeaks();
    console.log('Memory Leak Report:');
    console.log(JSON.stringify(leakReport, null, 2));
    console.log('');
    
    // Example 7: Performance profiling
    console.log('Example 7: Performance Profiling Report');
    console.log('---------------------------------------');
    
    const profilingReport = trustformers.stopProfiling();
    console.log('Profiling Report:');
    console.log(JSON.stringify(profilingReport, null, 2));
    console.log('');
    
    // Cleanup
    console.log('Performing final cleanup...');
    trustformers.memoryCleanup();
    trustformers.gc();
    
    console.log('Example completed successfully!');
    
  } catch (error) {
    console.error('Example failed:', error);
    process.exit(1);
  }
}

// Handle cleanup on exit
process.on('exit', () => {
  console.log('Cleaning up resources...');
});

process.on('SIGINT', () => {
  console.log('Received SIGINT, cleaning up...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('Received SIGTERM, cleaning up...');
  process.exit(0);
});

// Run the example
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { main };