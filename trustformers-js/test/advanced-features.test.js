/**
 * Integration tests for advanced features:
 * - WebNN Backend
 * - Advanced Quantization
 * - Benchmark Suite
 * - Advanced Caching
 */

import {
  // WebNN
  WebNNBackend,
  WebNNBackendManager,
  WebNNCapabilities,
  WebNNOperations,
  getWebNNManager,
  initWebNN,
  isWebNNAvailable,

  // Quantization
  QuantizationType,
  Float16Utils,
  Int8Quantizer,
  Int4Quantizer,
  GGMLQuantizer,
  QuantizationCalibrator,
  MixedPrecisionQuantizer,
  QuantizationAwareTraining,

  // Benchmarking
  BenchmarkConfig,
  BenchmarkSuite,
  TensorBenchmarks,
  ModelBenchmarks,
  PerformanceStats,

  // Caching
  LRUCache,
  TTLCache,
  LFUCache,
  MultiLevelCache,
  PersistentCache,
  CacheManager
} from '../src/index.js';

// Test utilities
class TestRunner {
  constructor(name) {
    this.name = name;
    this.tests = [];
    this.results = {
      passed: 0,
      failed: 0,
      errors: []
    };
  }

  test(description, fn) {
    this.tests.push({ description, fn });
  }

  async run() {
    console.log(`\n=== ${this.name} ===\n`);

    for (const { description, fn } of this.tests) {
      try {
        await fn();
        console.log(`✓ ${description}`);
        this.results.passed++;
      } catch (error) {
        console.error(`✗ ${description}`);
        console.error(`  Error: ${error.message}`);
        this.results.failed++;
        this.results.errors.push({ description, error: error.message });
      }
    }

    console.log(`\nResults: ${this.results.passed} passed, ${this.results.failed} failed\n`);
    return this.results;
  }
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message || 'Assertion failed');
  }
}

function assertAlmostEqual(actual, expected, tolerance = 0.01, message) {
  const diff = Math.abs(actual - expected);
  if (diff > tolerance) {
    throw new Error(message || `Expected ${actual} to be close to ${expected} (diff: ${diff})`);
  }
}

// ===== WebNN Backend Tests =====
async function testWebNNBackend() {
  const runner = new TestRunner('WebNN Backend Tests');

  runner.test('WebNN capability detection', async () => {
    const supported = isWebNNAvailable();
    console.log(`  WebNN supported: ${supported}`);
    // Don't fail if not supported, just log
  });

  runner.test('WebNNCapabilities.detect()', async () => {
    const caps = await WebNNCapabilities.detect();
    assert(caps !== null, 'Capabilities should not be null');
    assert(typeof caps.available === 'boolean', 'Available should be boolean');
    console.log(`  Available: ${caps.available}`);
    console.log(`  Devices: ${caps.devices.join(', ') || 'none'}`);
  });

  runner.test('WebNN backend initialization', async () => {
    if (!isWebNNAvailable()) {
      console.log('  Skipped: WebNN not available');
      return;
    }

    const success = await initWebNN({ deviceType: 'cpu' });
    console.log(`  Initialization: ${success ? 'success' : 'failed'}`);
  });

  runner.test('WebNN operations helpers', () => {
    // Test operation builder structure
    assert(typeof WebNNOperations.linear === 'function', 'linear should be a function');
    assert(typeof WebNNOperations.conv2d === 'function', 'conv2d should be a function');
    assert(typeof WebNNOperations.gelu === 'function', 'gelu should be a function');
  });

  return runner.run();
}

// ===== Quantization Tests =====
async function testQuantization() {
  const runner = new TestRunner('Advanced Quantization Tests');

  runner.test('Float16 conversion', () => {
    const f32 = 3.14159;
    const f16 = Float16Utils.float32ToFloat16(f32);
    const back = Float16Utils.float16ToFloat32(f16);

    assert(typeof f16 === 'number', 'Float16 should be a number');
    assertAlmostEqual(back, f32, 0.01, 'Conversion should be reversible');
    console.log(`  ${f32} -> ${f16} (uint16) -> ${back}`);
  });

  runner.test('Float16 array conversion', () => {
    const data = new Float32Array([1.0, 2.5, 3.14, -1.5, 0.0]);
    const f16Array = Float16Utils.float32ArrayToFloat16Array(data);
    const f32Array = Float16Utils.float16ArrayToFloat32Array(f16Array);

    assert(f16Array instanceof Uint16Array, 'Should return Uint16Array');
    assert(f32Array instanceof Float32Array, 'Should return Float32Array');
    assert(f32Array.length === data.length, 'Length should match');

    for (let i = 0; i < data.length; i++) {
      assertAlmostEqual(f32Array[i], data[i], 0.01);
    }
  });

  runner.test('INT8 quantization and dequantization', () => {
    const data = new Float32Array([-10, -5, 0, 5, 10]);
    const quantized = Int8Quantizer.quantize(data);

    assert(quantized.data instanceof Int8Array, 'Quantized data should be Int8Array');
    assert(typeof quantized.scale === 'number', 'Should have scale');
    assert(typeof quantized.zeroPoint === 'number', 'Should have zero point');

    const dequantized = Int8Quantizer.dequantize(
      quantized.data,
      quantized.scale,
      quantized.zeroPoint
    );

    assert(dequantized instanceof Float32Array, 'Dequantized should be Float32Array');
    assert(dequantized.length === data.length, 'Length should match');

    console.log(`  Original: [${data.join(', ')}]`);
    console.log(`  Dequantized: [${Array.from(dequantized).map(v => v.toFixed(2)).join(', ')}]`);
  });

  runner.test('INT4 quantization', () => {
    const data = new Float32Array([-8, -4, 0, 4, 8]);
    const quantized = Int4Quantizer.quantize(data);

    assert(quantized.data instanceof Uint8Array, 'Quantized data should be Uint8Array');
    assert(quantized.originalLength === data.length, 'Should store original length');
    assert(quantized.data.length === Math.ceil(data.length / 2), 'Should be packed');

    const dequantized = Int4Quantizer.dequantize(
      quantized.data,
      quantized.scale,
      quantized.zeroPoint,
      quantized.originalLength
    );

    assert(dequantized.length === data.length, 'Length should match');
  });

  runner.test('GGML Q4_0 quantization', () => {
    const data = new Float32Array(64).map(() => Math.random() * 10 - 5);
    const quantized = GGMLQuantizer.quantizeQ4_0(data);

    assert(quantized.data instanceof Uint8Array, 'Should return Uint8Array');
    assert(quantized.blockSize === 32, 'Block size should be 32');
    assert(quantized.type === QuantizationType.GGML_Q4_0, 'Type should match');

    const dequantized = GGMLQuantizer.dequantizeQ4_0(quantized.data, data.length);

    assert(dequantized instanceof Float32Array, 'Dequantized should be Float32Array');
    assert(dequantized.length === data.length, 'Length should match');
  });

  runner.test('GGML Q8_0 quantization', () => {
    const data = new Float32Array(64).map(() => Math.random() * 10 - 5);
    const quantized = GGMLQuantizer.quantizeQ8_0(data);

    assert(quantized.data instanceof Uint8Array, 'Should return Uint8Array');
    assert(quantized.type === QuantizationType.GGML_Q8_0, 'Type should match');

    const dequantized = GGMLQuantizer.dequantizeQ8_0(quantized.data, data.length);
    assert(dequantized.length === data.length, 'Length should match');
  });

  runner.test('Quantization calibrator', () => {
    const calibrator = new QuantizationCalibrator();
    const data = new Float32Array(1000).map(() => Math.random() * 20 - 10);

    calibrator.collectStatistics('test', data);
    const scale = calibrator.computeOptimalScale('test', 8);

    assert(typeof scale === 'number', 'Scale should be a number');
    assert(scale > 0, 'Scale should be positive');
    console.log(`  Optimal scale: ${scale.toFixed(4)}`);
  });

  runner.test('Mixed precision quantization', () => {
    const mixer = new MixedPrecisionQuantizer();

    mixer.setLayerQuantization('layer1', QuantizationType.FP16);
    mixer.setLayerQuantization('layer2', QuantizationType.INT8);
    mixer.setLayerQuantization('layer3', QuantizationType.INT4);

    const data1 = new Float32Array([1, 2, 3, 4, 5]);
    const data2 = new Float32Array([1, 2, 3, 4, 5]);
    const data3 = new Float32Array([1, 2, 3, 4, 5]);

    const q1 = mixer.quantizeLayer('layer1', data1);
    const q2 = mixer.quantizeLayer('layer2', data2);
    const q3 = mixer.quantizeLayer('layer3', data3);

    assert(q1.type === QuantizationType.FP16, 'Layer1 should be FP16');
    assert(q2.type === QuantizationType.INT8, 'Layer2 should be INT8');
    assert(q3.type === QuantizationType.INT4, 'Layer3 should be INT4');

    const report = mixer.generateReport();
    assert(report.layers.layer1.bitsPerWeight === 16, 'FP16 should be 16 bits');
    assert(report.layers.layer2.bitsPerWeight === 8, 'INT8 should be 8 bits');
    assert(report.layers.layer3.bitsPerWeight === 4, 'INT4 should be 4 bits');
  });

  runner.test('Quantization-aware training', () => {
    const data = new Float32Array([1.5, 2.7, -3.2, 4.1, -5.8]);
    const scale = 0.1;
    const zeroPoint = 0;

    const fakeQuantized = QuantizationAwareTraining.fakeQuantize(data, scale, zeroPoint, 8);

    assert(fakeQuantized instanceof Float32Array, 'Should return Float32Array');
    assert(fakeQuantized.length === data.length, 'Length should match');

    // Values should be quantized and dequantized
    for (let i = 0; i < data.length; i++) {
      const diff = Math.abs(fakeQuantized[i] - data[i]);
      assert(diff <= scale, 'Quantization error should be within scale');
    }
  });

  return runner.run();
}

// ===== Benchmark Suite Tests =====
async function testBenchmarking() {
  const runner = new TestRunner('Benchmark Suite Tests');

  runner.test('Performance statistics calculation', () => {
    const measurements = [10, 12, 11, 13, 14, 10, 12, 15, 11, 13];
    const stats = PerformanceStats.calculate(measurements);

    assert(stats.count === measurements.length, 'Count should match');
    assert(stats.min === 10, 'Min should be 10');
    assert(stats.max === 15, 'Max should be 15');
    assert(stats.mean > 0, 'Mean should be positive');
    assert(stats.median > 0, 'Median should be positive');
    assert(stats.stdDev >= 0, 'Std dev should be non-negative');

    console.log(`  Mean: ${stats.mean.toFixed(2)}ms`);
    console.log(`  Median: ${stats.median}ms`);
    console.log(`  Std Dev: ${stats.stdDev.toFixed(2)}ms`);
  });

  runner.test('Performance comparison', () => {
    const baseline = PerformanceStats.calculate([100, 110, 105, 108, 102]);
    const current = PerformanceStats.calculate([90, 88, 92, 89, 91]);

    const comparison = PerformanceStats.compare(baseline, current);

    assert(comparison.faster === true, 'Current should be faster');
    assert(comparison.improvement > 0, 'Should show improvement');
    assert(comparison.status === 'faster', 'Status should be faster');

    console.log(`  Improvement: ${comparison.improvementPercent}%`);
    console.log(`  Status: ${comparison.status}`);
  });

  runner.test('Benchmark configuration', () => {
    const config = new BenchmarkConfig({
      warmupRuns: 10,
      benchmarkRuns: 100,
      timeout: 60000
    });

    assert(config.warmupRuns === 10, 'Warmup runs should match');
    assert(config.benchmarkRuns === 100, 'Benchmark runs should match');
    assert(config.timeout === 60000, 'Timeout should match');
  });

  runner.test('Benchmark suite creation', () => {
    const suite = new BenchmarkSuite({
      warmupRuns: 3,
      benchmarkRuns: 10
    });

    assert(suite.results.metadata !== undefined, 'Should have metadata');
    assert(suite.results.benchmarks !== undefined, 'Should have benchmarks object');
  });

  runner.test('Benchmark HTML report generation', () => {
    const suite = new BenchmarkSuite();
    const html = suite.generateHTMLReport();

    assert(typeof html === 'string', 'Should return string');
    assert(html.includes('<!DOCTYPE html>'), 'Should be valid HTML');
    assert(html.includes('TrustformeRS Benchmark Report'), 'Should have title');
  });

  runner.test('Benchmark JSON report generation', () => {
    const suite = new BenchmarkSuite();
    const json = suite.generateJSONReport();

    assert(typeof json === 'string', 'Should return string');
    const parsed = JSON.parse(json);
    assert(parsed.metadata !== undefined, 'Should have metadata');
  });

  return runner.run();
}

// ===== Caching Tests =====
async function testCaching() {
  const runner = new TestRunner('Advanced Caching Tests');

  runner.test('LRU cache basic operations', () => {
    const cache = new LRUCache({ maxSize: 3 });

    cache.set('a', 1);
    cache.set('b', 2);
    cache.set('c', 3);

    assert(cache.get('a') === 1, 'Should retrieve value a');
    assert(cache.get('b') === 2, 'Should retrieve value b');
    assert(cache.get('c') === 3, 'Should retrieve value c');

    // Add one more, should evict least recently used
    cache.set('d', 4);

    assert(cache.get('a') === null, 'Should have evicted a');
    assert(cache.get('d') === 4, 'Should have d');

    const stats = cache.getStatistics();
    console.log(`  Hits: ${stats.hits}, Misses: ${stats.misses}`);
  });

  runner.test('LRU cache memory limits', () => {
    const cache = new LRUCache({ maxMemory: 1000 });

    const largeData = new Float32Array(100); // 400 bytes
    cache.set('data1', largeData, { size: 400 });
    cache.set('data2', largeData, { size: 400 });
    cache.set('data3', largeData, { size: 400 }); // Should trigger eviction

    assert(cache.currentMemory <= 1000, 'Should stay within memory limit');
  });

  runner.test('TTL cache expiration', async () => {
    const cache = new TTLCache({ defaultTTL: 100 }); // 100ms TTL

    try {
      cache.set('key1', 'value1');
      assert(cache.get('key1') === 'value1', 'Should get value immediately');

      // Wait for expiration
      await new Promise(resolve => setTimeout(resolve, 150));

      assert(cache.get('key1') === null, 'Should have expired');

      const stats = cache.getStatistics();
      console.log(`  Expirations: ${stats.expirations}`);
    } finally {
      // Clean up to prevent test hanging
      cache.dispose();
    }
  });

  runner.test('TTL cache custom expiration', () => {
    const cache = new TTLCache({ defaultTTL: 1000 });

    try {
      cache.set('key1', 'value1', 100); // Custom 100ms TTL
      cache.set('key2', 'value2', 10000); // 10s TTL

      assert(cache.has('key1') === true, 'key1 should exist');
      assert(cache.has('key2') === true, 'key2 should exist');
    } finally {
      // Clean up to prevent test hanging
      cache.dispose();
    }
  });

  runner.test('LFU cache frequency tracking', () => {
    const cache = new LFUCache({ maxSize: 3 });

    cache.set('a', 1);
    cache.set('b', 2);
    cache.set('c', 3);

    // Access 'a' and 'b' multiple times
    cache.get('a');
    cache.get('a');
    cache.get('a');
    cache.get('b');
    cache.get('b');

    // Add new item, should evict 'c' (least frequently used)
    cache.set('d', 4);

    assert(cache.get('a') === 1, 'a should still exist');
    assert(cache.get('b') === 2, 'b should still exist');
    assert(cache.get('c') === null, 'c should have been evicted');
    assert(cache.get('d') === 4, 'd should exist');
  });

  runner.test('Multi-level cache L1/L2/L3', async () => {
    const cache = new MultiLevelCache({
      l1Size: 2,
      l2Size: 5,
      l3Size: 10
    });

    try {
      await cache.set('key1', 'value1');
      const value = await cache.get('key1');

      assert(value === 'value1', 'Should retrieve value');

      const stats = cache.getStatistics();
      assert(stats.l1Hits >= 0, 'Should track L1 hits');
      assert(stats.hitRate !== undefined, 'Should calculate hit rate');

      console.log(`  Hit rate: ${stats.hitRate}`);
    } finally {
      cache.dispose();
    }
  });

  runner.test('Multi-level cache promotion', async () => {
    const cache = new MultiLevelCache({
      l1Size: 1,
      l2Size: 2,
      l3Size: 5
    });

    try {
      // Fill caches
      await cache.set('a', 1);
      await cache.set('b', 2);
      await cache.set('c', 3);

      // Access 'c' should promote to higher levels
      const value = await cache.get('c');
      assert(value === 3, 'Should get value');

      // Now 'c' should be in L1
      cache.l1.get('c'); // This should be a hit
    } finally {
      cache.dispose();
    }
  });

  runner.test('Cache manager with loader', async () => {
    const manager = new CacheManager({ maxSize: 5 });

    const loader = async (key) => {
      // Simulate async data loading
      return `loaded_${key}`;
    };

    const value1 = await manager.get('key1', loader);
    assert(value1 === 'loaded_key1', 'Should load value');

    const value2 = await manager.get('key1'); // Should use cache
    assert(value2 === 'loaded_key1', 'Should use cached value');
  });

  runner.test('Cache manager prefetching', async () => {
    const manager = new CacheManager({ maxSize: 10 });

    const loader = async (key) => `prefetched_${key}`;

    await manager.prefetch(['key1', 'key2', 'key3'], loader);

    // Give prefetch time to process
    await new Promise(resolve => setTimeout(resolve, 100));

    // Keys should be in cache
    const value = await manager.get('key1');
    console.log(`  Prefetched value: ${value || 'not yet cached'}`);
  });

  runner.test('Cache manager warming', async () => {
    const manager = new CacheManager({ maxSize: 10 });

    await manager.warm({
      key1: 'value1',
      key2: 'value2',
      key3: 'value3'
    });

    assert(await manager.get('key1') === 'value1', 'Should have warmed key1');
    assert(await manager.get('key2') === 'value2', 'Should have warmed key2');
    assert(await manager.get('key3') === 'value3', 'Should have warmed key3');
  });

  return runner.run();
}

// ===== Main Test Runner =====
async function runAllTests() {
  console.log('\n╔════════════════════════════════════════╗');
  console.log('║  TrustformeRS Advanced Features Tests  ║');
  console.log('╚════════════════════════════════════════╝');

  const results = {
    webnn: await testWebNNBackend(),
    quantization: await testQuantization(),
    benchmarking: await testBenchmarking(),
    caching: await testCaching()
  };

  // Summary
  console.log('\n╔════════════════════════════════════════╗');
  console.log('║            Test Summary                ║');
  console.log('╚════════════════════════════════════════╝\n');

  let totalPassed = 0;
  let totalFailed = 0;

  for (const [category, result] of Object.entries(results)) {
    console.log(`${category}:`);
    console.log(`  ✓ Passed: ${result.passed}`);
    console.log(`  ✗ Failed: ${result.failed}`);
    totalPassed += result.passed;
    totalFailed += result.failed;
  }

  console.log(`\n${'='.repeat(40)}`);
  console.log(`Total: ${totalPassed} passed, ${totalFailed} failed`);
  console.log(`${'='.repeat(40)}\n`);

  // Exit with error code if any tests failed
  if (totalFailed > 0) {
    process.exit(1);
  }
}

// Run tests if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllTests().catch(error => {
    console.error('Test runner error:', error);
    process.exit(1);
  });
}

export { runAllTests };
