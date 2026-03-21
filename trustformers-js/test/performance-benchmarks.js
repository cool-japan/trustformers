/**
 * Performance Benchmarks for TrustformeRS JavaScript API
 *
 * Comprehensive performance testing and benchmarking utilities
 */

import { performance } from 'perf_hooks';
import process from 'process';
import { TestRunner, describe, test, beforeAll, afterAll, expect } from './test-runner.js';

class PerformanceBenchmark {
  constructor(name, options = {}) {
    this.name = name;
    this.options = {
      iterations: 100,
      warmup: 10,
      timeout: 30000,
      memoryTracking: true,
      gcBetweenRuns: true,
      ...options
    };
    this.results = [];
    this.startMemory = null;
    this.endMemory = null;
  }

  getMemoryUsage() {
    if (process.memoryUsage) {
      const usage = process.memoryUsage();
      return {
        rss: Math.round(usage.rss / 1024 / 1024), // MB
        heapTotal: Math.round(usage.heapTotal / 1024 / 1024),
        heapUsed: Math.round(usage.heapUsed / 1024 / 1024),
        external: Math.round(usage.external / 1024 / 1024)
      };
    }
    return { rss: 0, heapTotal: 0, heapUsed: 0, external: 0 };
  }

  async run(fn) {
    console.log(`🚀 Running benchmark: ${this.name}`);
    console.log(`   Iterations: ${this.options.iterations}, Warmup: ${this.options.warmup}`);

    this.startMemory = this.getMemoryUsage();

    // Warmup phase
    console.log('   ⚡ Warming up...');
    for (let i = 0; i < this.options.warmup; i++) {
      await fn();
      if (this.options.gcBetweenRuns && global.gc) {
        global.gc();
      }
    }

    // Benchmark phase
    console.log('   📊 Benchmarking...');
    const results = [];

    for (let i = 0; i < this.options.iterations; i++) {
      const start = performance.now();
      const memBefore = this.getMemoryUsage();

      await fn();

      const end = performance.now();
      const memAfter = this.getMemoryUsage();
      const duration = end - start;

      results.push({
        duration,
        memoryDelta: memAfter.heapUsed - memBefore.heapUsed,
        timestamp: Date.now()
      });

      if (this.options.gcBetweenRuns && global.gc) {
        global.gc();
      }

      // Show progress every 10%
      if ((i + 1) % Math.max(1, Math.floor(this.options.iterations / 10)) === 0) {
        const progress = ((i + 1) / this.options.iterations * 100).toFixed(0);
        console.log(`   ⏳ Progress: ${progress}% (${i + 1}/${this.options.iterations})`);
      }
    }

    this.endMemory = this.getMemoryUsage();
    this.results = results;

    return this.analyze();
  }

  analyze() {
    const durations = this.results.map(r => r.duration);
    const memoryDeltas = this.results.map(r => r.memoryDelta);

    // Statistical analysis
    const mean = durations.reduce((a, b) => a + b, 0) / durations.length;
    const sorted = [...durations].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const min = Math.min(...durations);
    const max = Math.max(...durations);
    const p95 = sorted[Math.floor(sorted.length * 0.95)];
    const p99 = sorted[Math.floor(sorted.length * 0.99)];

    // Standard deviation
    const variance = durations.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / durations.length;
    const stdDev = Math.sqrt(variance);

    // Operations per second
    const opsPerSecond = 1000 / mean;

    // Memory analysis
    const avgMemoryDelta = memoryDeltas.reduce((a, b) => a + b, 0) / memoryDeltas.length;
    const totalMemoryDelta = this.endMemory.heapUsed - this.startMemory.heapUsed;

    const analysis = {
      name: this.name,
      iterations: this.options.iterations,
      timing: {
        mean: mean.toFixed(3),
        median: median.toFixed(3),
        min: min.toFixed(3),
        max: max.toFixed(3),
        p95: p95.toFixed(3),
        p99: p99.toFixed(3),
        stdDev: stdDev.toFixed(3),
        opsPerSecond: opsPerSecond.toFixed(2)
      },
      memory: {
        avgDeltaPerOp: avgMemoryDelta.toFixed(2),
        totalDelta: totalMemoryDelta.toFixed(2),
        startHeap: this.startMemory.heapUsed,
        endHeap: this.endMemory.heapUsed,
        peakRSS: Math.max(this.startMemory.rss, this.endMemory.rss)
      },
      rawResults: this.results
    };

    this.printResults(analysis);
    return analysis;
  }

  printResults(analysis) {
    console.log(`\n📊 Benchmark Results: ${analysis.name}`);
    console.log('─'.repeat(60));
    console.log('⏱️  Timing Statistics:');
    console.log(`   Mean:     ${analysis.timing.mean}ms`);
    console.log(`   Median:   ${analysis.timing.median}ms`);
    console.log(`   Min:      ${analysis.timing.min}ms`);
    console.log(`   Max:      ${analysis.timing.max}ms`);
    console.log(`   P95:      ${analysis.timing.p95}ms`);
    console.log(`   P99:      ${analysis.timing.p99}ms`);
    console.log(`   StdDev:   ${analysis.timing.stdDev}ms`);
    console.log(`   Ops/sec:  ${analysis.timing.opsPerSecond}`);

    console.log('\n💾 Memory Statistics:');
    console.log(`   Avg per op:     ${analysis.memory.avgDeltaPerOp}MB`);
    console.log(`   Total delta:    ${analysis.memory.totalDelta}MB`);
    console.log(`   Start heap:     ${analysis.memory.startHeap}MB`);
    console.log(`   End heap:       ${analysis.memory.endHeap}MB`);
    console.log(`   Peak RSS:       ${analysis.memory.peakRSS}MB`);

    // Performance warnings
    console.log('\n🔍 Performance Analysis:');
    if (parseFloat(analysis.timing.mean) > 100) {
      console.log('   ⚠️  High latency detected (>100ms average)');
    }
    if (parseFloat(analysis.timing.stdDev) / parseFloat(analysis.timing.mean) > 0.5) {
      console.log('   ⚠️  High variance detected (inconsistent performance)');
    }
    if (parseFloat(analysis.memory.totalDelta) > 100) {
      console.log('   ⚠️  Significant memory growth detected (>100MB)');
    }
    if (parseFloat(analysis.memory.avgDeltaPerOp) > 1) {
      console.log('   ⚠️  High memory usage per operation (>1MB)');
    }
  }
}

class PerformanceTestSuite {
  constructor() {
    this.benchmarks = [];
    this.results = [];
    this.config = {
      defaultIterations: 100,
      defaultWarmup: 10,
      memoryThreshold: 1000, // MB
      latencyThreshold: 100,  // ms
      varianceThreshold: 0.5  // coefficient of variation
    };
  }

  addBenchmark(name, fn, options = {}) {
    const benchmark = new PerformanceBenchmark(name, {
      iterations: this.config.defaultIterations,
      warmup: this.config.defaultWarmup,
      ...options
    });

    this.benchmarks.push({ benchmark, fn });
  }

  async runAll() {
    console.log('🏁 Performance Test Suite');
    console.log('═'.repeat(80));
    console.log(`Running ${this.benchmarks.length} benchmarks\n`);

    const startTime = performance.now();
    const startMemory = this.getMemoryUsage();

    for (const { benchmark, fn } of this.benchmarks) {
      try {
        const result = await benchmark.run(fn);
        this.results.push(result);
      } catch (error) {
        console.error(`❌ Benchmark failed: ${benchmark.name}`);
        console.error(`   Error: ${error.message}`);
        this.results.push({
          name: benchmark.name,
          error: error.message,
          failed: true
        });
      }
      console.log(); // Add spacing between benchmarks
    }

    const endTime = performance.now();
    const endMemory = this.getMemoryUsage();

    this.generateSummaryReport(startTime, endTime, startMemory, endMemory);
    return this.results;
  }

  getMemoryUsage() {
    if (process.memoryUsage) {
      const usage = process.memoryUsage();
      return {
        rss: Math.round(usage.rss / 1024 / 1024),
        heapTotal: Math.round(usage.heapTotal / 1024 / 1024),
        heapUsed: Math.round(usage.heapUsed / 1024 / 1024),
        external: Math.round(usage.external / 1024 / 1024)
      };
    }
    return { rss: 0, heapTotal: 0, heapUsed: 0, external: 0 };
  }

  generateSummaryReport(startTime, endTime, startMemory, endMemory) {
    const totalDuration = endTime - startTime;
    const successfulBenchmarks = this.results.filter(r => !r.failed);
    const failedBenchmarks = this.results.filter(r => r.failed);

    console.log('📈 Performance Test Suite Summary');
    console.log('═'.repeat(80));
    console.log(`Total duration: ${totalDuration.toFixed(2)}ms`);
    console.log(`Successful benchmarks: ${successfulBenchmarks.length}`);
    console.log(`Failed benchmarks: ${failedBenchmarks.length}`);
    console.log(`Memory change: ${endMemory.heapUsed - startMemory.heapUsed}MB`);

    if (successfulBenchmarks.length > 0) {
      console.log('\n🏆 Top Performers:');
      const fastest = successfulBenchmarks.sort((a, b) =>
        parseFloat(a.timing.mean) - parseFloat(b.timing.mean)
      ).slice(0, 3);

      fastest.forEach((result, i) => {
        console.log(`   ${i + 1}. ${result.name}: ${result.timing.mean}ms avg`);
      });

      console.log('\n⚠️  Performance Concerns:');
      const concerns = successfulBenchmarks.filter(result =>
        parseFloat(result.timing.mean) > this.config.latencyThreshold ||
        parseFloat(result.timing.stdDev) / parseFloat(result.timing.mean) > this.config.varianceThreshold ||
        parseFloat(result.memory.totalDelta) > 50
      );

      if (concerns.length > 0) {
        concerns.forEach(result => {
          const issues = [];
          if (parseFloat(result.timing.mean) > this.config.latencyThreshold) {
            issues.push('high latency');
          }
          if (parseFloat(result.timing.stdDev) / parseFloat(result.timing.mean) > this.config.varianceThreshold) {
            issues.push('high variance');
          }
          if (parseFloat(result.memory.totalDelta) > 50) {
            issues.push('memory growth');
          }
          console.log(`   • ${result.name}: ${issues.join(', ')}`);
        });
      } else {
        console.log('   ✅ No significant performance concerns detected');
      }
    }

    console.log('\n🌍 Environment Information:');
    console.log(`   Node.js: ${process.version}`);
    console.log(`   Platform: ${process.platform} ${process.arch}`);
    console.log(`   Total memory: ${Math.round(require('os').totalmem() / 1024 / 1024 / 1024)}GB`);
    console.log(`   Free memory: ${Math.round(require('os').freemem() / 1024 / 1024 / 1024)}GB`);
    console.log(`   CPU cores: ${require('os').cpus().length}`);
  }

  exportResults(filename = 'benchmark-results.json') {
    const exportData = {
      timestamp: new Date().toISOString(),
      environment: {
        node: process.version,
        platform: process.platform,
        arch: process.arch,
        cpus: require('os').cpus().length,
        totalMemory: Math.round(require('os').totalmem() / 1024 / 1024 / 1024),
      },
      config: this.config,
      results: this.results
    };

    return JSON.stringify(exportData, null, 2);
  }
}

// Enhanced test utilities for performance testing
export class PerformanceTestUtilities {
  static createMockTensor(shape, dtype = 'float32') {
    // Mock tensor for testing when WASM is not available
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);

    // Fill with random data
    for (let i = 0; i < size; i++) {
      data[i] = Math.random();
    }

    return {
      shape,
      dtype,
      data,
      size,
      toString: () => `MockTensor(shape=[${shape.join(', ')}], dtype=${dtype})`,
      // Mock common tensor operations
      add: () => PerformanceTestUtilities.createMockTensor(shape, dtype),
      mul: () => PerformanceTestUtilities.createMockTensor(shape, dtype),
      matmul: (other) => {
        const lastDim = shape[shape.length - 1];
        const newShape = [...shape.slice(0, -1), other.shape[other.shape.length - 1]];
        return PerformanceTestUtilities.createMockTensor(newShape, dtype);
      }
    };
  }

  static async measureAsyncOperation(operation, label = 'Operation') {
    const start = performance.now();
    const memBefore = process.memoryUsage();

    const result = await operation();

    const end = performance.now();
    const memAfter = process.memoryUsage();

    const metrics = {
      duration: end - start,
      memoryDelta: Math.round((memAfter.heapUsed - memBefore.heapUsed) / 1024 / 1024),
      label,
      timestamp: Date.now()
    };

    console.log(`⏱️  ${label}: ${metrics.duration.toFixed(2)}ms, Memory: ${metrics.memoryDelta >= 0 ? '+' : ''}${metrics.memoryDelta}MB`);

    return { result, metrics };
  }

  static createStressTest(operationFn, options = {}) {
    const {
      duration = 10000,
      concurrency = 10,
      rampUp = 1000,
      memoryLimit = 2048
    } = options;

    return async () => {
      console.log(`🔥 Stress test: ${duration}ms duration, ${concurrency} concurrent operations`);

      const startTime = Date.now();
      const activeOperations = [];
      let completedOps = 0;
      let errors = 0;

      // Ramp up phase
      const rampUpInterval = rampUp / concurrency;

      for (let i = 0; i < concurrency; i++) {
        await new Promise(resolve => setTimeout(resolve, rampUpInterval));

        const runOperation = async () => {
          while (Date.now() - startTime < duration) {
            try {
              // Check memory limit
              const memory = process.memoryUsage();
              if (memory.heapUsed / 1024 / 1024 > memoryLimit) {
                console.warn('⚠️  Memory limit reached, pausing operations');
                await new Promise(resolve => setTimeout(resolve, 100));
                continue;
              }

              await operationFn();
              completedOps++;

              // Brief pause to prevent overwhelming
              await new Promise(resolve => setTimeout(resolve, 10));
            } catch (error) {
              errors++;
              console.error(`Error in stress test operation: ${error.message}`);
            }
          }
        };

        activeOperations.push(runOperation());
      }

      // Wait for all operations to complete
      await Promise.all(activeOperations);

      const totalDuration = Date.now() - startTime;
      const opsPerSecond = (completedOps / totalDuration * 1000).toFixed(2);

      console.log(`🏁 Stress test completed:`);
      console.log(`   Operations: ${completedOps}`);
      console.log(`   Errors: ${errors}`);
      console.log(`   Duration: ${totalDuration}ms`);
      console.log(`   Ops/sec: ${opsPerSecond}`);

      return {
        completedOps,
        errors,
        duration: totalDuration,
        opsPerSecond: parseFloat(opsPerSecond)
      };
    };
  }
}

// Performance test definitions
export function definePerformanceBenchmarks() {
  const suite = new PerformanceTestSuite();

  // Basic operations benchmarks
  suite.addBenchmark('Array Creation and Processing', async () => {
    const size = 10000;
    const arr = new Array(size);
    for (let i = 0; i < size; i++) {
      arr[i] = Math.random();
    }
    return arr.reduce((sum, val) => sum + val, 0);
  }, { iterations: 1000, warmup: 50 });

  suite.addBenchmark('TypedArray Operations', async () => {
    const size = 10000;
    const arr = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      arr[i] = Math.random();
    }
    // Simulate tensor operations
    const result = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      result[i] = arr[i] * 2 + 1;
    }
    return result;
  }, { iterations: 1000, warmup: 50 });

  suite.addBenchmark('Mock Tensor Operations', async () => {
    const tensor1 = PerformanceTestUtilities.createMockTensor([100, 100]);
    const tensor2 = PerformanceTestUtilities.createMockTensor([100, 100]);

    // Simulate tensor operations
    const result1 = tensor1.add(tensor2);
    const result2 = result1.mul(tensor1);
    return result2;
  }, { iterations: 500, warmup: 25 });

  suite.addBenchmark('Matrix Multiplication Simulation', async () => {
    const matrixA = PerformanceTestUtilities.createMockTensor([64, 64]);
    const matrixB = PerformanceTestUtilities.createMockTensor([64, 64]);

    return matrixA.matmul(matrixB);
  }, { iterations: 100, warmup: 10 });

  suite.addBenchmark('JSON Serialization Performance', async () => {
    const data = {
      model: 'test-model',
      tensors: Array.from({ length: 100 }, (_, i) => ({
        name: `tensor_${i}`,
        shape: [32, 32],
        data: Array.from({ length: 1024 }, () => Math.random())
      })),
      metadata: {
        version: '1.0.0',
        timestamp: Date.now(),
        description: 'Performance test data'
      }
    };

    const serialized = JSON.stringify(data);
    const parsed = JSON.parse(serialized);
    return parsed;
  }, { iterations: 50, warmup: 5 });

  return suite;
}

// Memory leak detection test
export function createMemoryLeakTest() {
  return describe('Memory Leak Detection', () => {
    test('detects memory leaks in repeated operations', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      const operations = 1000;

      for (let i = 0; i < operations; i++) {
        // Simulate operation that might leak
        const tensor = PerformanceTestUtilities.createMockTensor([10, 10]);
        const result = tensor.add(tensor);

        // Force garbage collection every 100 operations
        if (i % 100 === 0 && global.gc) {
          global.gc();
        }
      }

      // Final garbage collection
      if (global.gc) {
        global.gc();
        // Wait for GC to complete
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = (finalMemory - initialMemory) / 1024 / 1024; // MB

      console.log(`Memory increase after ${operations} operations: ${memoryIncrease.toFixed(2)}MB`);

      // Allow for some memory growth, but flag significant leaks
      expect(memoryIncrease).toBeLessThan(100); // Less than 100MB increase
    }, 30000); // 30 second timeout
  });
}

export { PerformanceBenchmark, PerformanceTestSuite };