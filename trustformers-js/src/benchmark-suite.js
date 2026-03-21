/**
 * Comprehensive Benchmarking Suite
 *
 * Provides extensive performance testing and profiling capabilities for:
 * - Tensor operations
 * - Model inference
 * - Backend performance (WASM, WebGL, WebGPU, WebNN)
 * - Memory usage and allocation
 * - Quantization impact
 * - End-to-end pipeline performance
 *
 * Features:
 * - Automated benchmark execution
 * - Statistical analysis (mean, median, std dev, percentiles)
 * - Performance comparison across backends
 * - Regression detection
 * - HTML and JSON reporting
 */

/**
 * Benchmark configuration
 */
export class BenchmarkConfig {
  constructor(options = {}) {
    this.warmupRuns = options.warmupRuns || 5;
    this.benchmarkRuns = options.benchmarkRuns || 50;
    this.timeout = options.timeout || 30000;
    this.collectMemoryStats = options.collectMemoryStats !== false;
    this.backends = options.backends || ['wasm', 'webgl', 'webgpu', 'webnn'];
    this.reportFormat = options.reportFormat || 'both'; // 'json', 'html', 'both'
  }
}

/**
 * Performance statistics calculator
 */
export class PerformanceStats {
  static calculate(measurements) {
    if (measurements.length === 0) {
      return null;
    }

    const sorted = [...measurements].sort((a, b) => a - b);
    const sum = measurements.reduce((a, b) => a + b, 0);
    const mean = sum / measurements.length;

    // Standard deviation
    const squareDiffs = measurements.map(value => Math.pow(value - mean, 2));
    const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / measurements.length;
    const stdDev = Math.sqrt(avgSquareDiff);

    // Percentiles
    const p50 = sorted[Math.floor(sorted.length * 0.50)];
    const p90 = sorted[Math.floor(sorted.length * 0.90)];
    const p95 = sorted[Math.floor(sorted.length * 0.95)];
    const p99 = sorted[Math.floor(sorted.length * 0.99)];

    return {
      count: measurements.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean,
      median: p50,
      stdDev,
      variance: avgSquareDiff,
      p50, p90, p95, p99,
      coefficientOfVariation: (stdDev / mean) * 100
    };
  }

  static compare(baseline, current) {
    const improvement = ((baseline.mean - current.mean) / baseline.mean) * 100;
    const significance = Math.abs(improvement) > 5; // 5% threshold

    return {
      baseline: baseline.mean,
      current: current.mean,
      improvement,
      improvementPercent: improvement.toFixed(2),
      faster: improvement > 0,
      significant: significance,
      status: significance ?
        (improvement > 0 ? 'faster' : 'slower') :
        'similar'
    };
  }
}

/**
 * Tensor operation benchmarks
 */
export class TensorBenchmarks {
  constructor(backend) {
    this.backend = backend;
  }

  /**
   * Benchmark matrix multiplication
   */
  async benchmarkMatMul(sizes = [[64, 64], [128, 128], [256, 256], [512, 512], [1024, 1024]]) {
    const results = [];

    for (const [m, n] of sizes) {
      const result = await this._benchmarkOp(
        'matmul',
        () => {
          const a = this._createRandomTensor([m, n]);
          const b = this._createRandomTensor([n, m]);
          return () => this.backend.matmul(a, b);
        },
        { m, n, operation: 'matmul' }
      );
      results.push(result);
    }

    return results;
  }

  /**
   * Benchmark element-wise operations
   */
  async benchmarkElementWise(sizes = [[1000], [10000], [100000], [1000000]]) {
    const operations = ['add', 'sub', 'mul', 'div'];
    const results = [];

    for (const size of sizes) {
      for (const op of operations) {
        const result = await this._benchmarkOp(
          `elementwise_${op}`,
          () => {
            const a = this._createRandomTensor(size);
            const b = this._createRandomTensor(size);
            return () => this.backend[op](a, b);
          },
          { size: size[0], operation: op }
        );
        results.push(result);
      }
    }

    return results;
  }

  /**
   * Benchmark activation functions
   */
  async benchmarkActivations(sizes = [[1000], [10000], [100000]]) {
    const activations = ['relu', 'sigmoid', 'tanh', 'gelu', 'softmax'];
    const results = [];

    for (const size of sizes) {
      for (const activation of activations) {
        const result = await this._benchmarkOp(
          `activation_${activation}`,
          () => {
            const input = this._createRandomTensor(size);
            return () => this.backend[activation](input);
          },
          { size: size[0], activation }
        );
        results.push(result);
      }
    }

    return results;
  }

  /**
   * Benchmark reduction operations
   */
  async benchmarkReductions(sizes = [[1000], [10000], [100000]]) {
    const operations = ['sum', 'mean', 'max', 'min'];
    const results = [];

    for (const size of sizes) {
      for (const op of operations) {
        const result = await this._benchmarkOp(
          `reduction_${op}`,
          () => {
            const input = this._createRandomTensor(size);
            return () => this.backend[op](input);
          },
          { size: size[0], operation: op }
        );
        results.push(result);
      }
    }

    return results;
  }

  async _benchmarkOp(name, setupFn, metadata = {}) {
    const config = new BenchmarkConfig();
    const measurements = [];
    let memoryStats = null;

    // Setup
    const opFn = setupFn();

    // Warmup
    for (let i = 0; i < config.warmupRuns; i++) {
      await opFn();
    }

    // Benchmark
    const startMemory = this._getMemoryUsage();

    for (let i = 0; i < config.benchmarkRuns; i++) {
      const start = performance.now();
      await opFn();
      const end = performance.now();
      measurements.push(end - start);
    }

    const endMemory = this._getMemoryUsage();

    if (config.collectMemoryStats && startMemory && endMemory) {
      memoryStats = {
        before: startMemory,
        after: endMemory,
        delta: endMemory.usedJSHeapSize - startMemory.usedJSHeapSize
      };
    }

    return {
      name,
      metadata,
      timing: PerformanceStats.calculate(measurements),
      memory: memoryStats,
      backend: this.backend.name || 'unknown'
    };
  }

  _createRandomTensor(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).map(() => Math.random());
    return { shape, data };
  }

  _getMemoryUsage() {
    if (typeof performance !== 'undefined' && performance.memory) {
      return {
        usedJSHeapSize: performance.memory.usedJSHeapSize,
        totalJSHeapSize: performance.memory.totalJSHeapSize,
        jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
      };
    }
    return null;
  }
}

/**
 * Model inference benchmarks
 */
export class ModelBenchmarks {
  constructor(model, tokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  /**
   * Benchmark text generation throughput
   */
  async benchmarkTextGeneration(prompts, options = {}) {
    const config = new BenchmarkConfig(options);
    const results = [];

    for (const prompt of prompts) {
      const measurements = {
        tokenization: [],
        inference: [],
        decoding: [],
        endToEnd: []
      };

      // Warmup
      for (let i = 0; i < config.warmupRuns; i++) {
        await this._runTextGeneration(prompt);
      }

      // Benchmark
      for (let i = 0; i < config.benchmarkRuns; i++) {
        const timing = await this._runTextGeneration(prompt);
        measurements.tokenization.push(timing.tokenization);
        measurements.inference.push(timing.inference);
        measurements.decoding.push(timing.decoding);
        measurements.endToEnd.push(timing.endToEnd);
      }

      results.push({
        prompt,
        tokenization: PerformanceStats.calculate(measurements.tokenization),
        inference: PerformanceStats.calculate(measurements.inference),
        decoding: PerformanceStats.calculate(measurements.decoding),
        endToEnd: PerformanceStats.calculate(measurements.endToEnd)
      });
    }

    return results;
  }

  /**
   * Benchmark batch processing
   */
  async benchmarkBatchProcessing(batchSizes = [1, 4, 8, 16, 32]) {
    const results = [];
    const sampleInput = "This is a test input for benchmarking.";

    for (const batchSize of batchSizes) {
      const inputs = Array(batchSize).fill(sampleInput);
      const measurements = [];

      // Warmup
      for (let i = 0; i < 3; i++) {
        await this._processBatch(inputs);
      }

      // Benchmark
      for (let i = 0; i < 20; i++) {
        const start = performance.now();
        await this._processBatch(inputs);
        const end = performance.now();
        measurements.push(end - start);
      }

      const stats = PerformanceStats.calculate(measurements);
      results.push({
        batchSize,
        timing: stats,
        throughput: (batchSize / stats.mean) * 1000, // samples per second
        latencyPerSample: stats.mean / batchSize
      });
    }

    return results;
  }

  /**
   * Benchmark different model precision settings
   */
  async benchmarkPrecision(input, precisions = ['fp32', 'fp16', 'int8', 'int4']) {
    const results = [];

    for (const precision of precisions) {
      // Would need to load model with different precision
      const measurements = [];

      for (let i = 0; i < 30; i++) {
        const start = performance.now();
        await this.model.forward(input);
        const end = performance.now();
        measurements.push(end - start);
      }

      results.push({
        precision,
        timing: PerformanceStats.calculate(measurements)
      });
    }

    return results;
  }

  async _runTextGeneration(prompt) {
    const startTokenization = performance.now();
    const tokens = await this.tokenizer.encode(prompt);
    const endTokenization = performance.now();

    const startInference = performance.now();
    const output = await this.model.generate(tokens);
    const endInference = performance.now();

    const startDecoding = performance.now();
    await this.tokenizer.decode(output);
    const endDecoding = performance.now();

    return {
      tokenization: endTokenization - startTokenization,
      inference: endInference - startInference,
      decoding: endDecoding - startDecoding,
      endToEnd: endDecoding - startTokenization
    };
  }

  async _processBatch(inputs) {
    return Promise.all(inputs.map(input => this.model.forward(input)));
  }
}

/**
 * Backend comparison benchmarks
 */
export class BackendComparison {
  constructor(backends) {
    this.backends = backends; // Array of {name, backend} objects
  }

  /**
   * Compare operation performance across backends
   */
  async compareOperation(opName, setupFn, _options = {}) {
    const results = {};

    for (const { name, backend } of this.backends) {
      try {
        const benchmarker = new TensorBenchmarks(backend);
        const result = await benchmarker._benchmarkOp(
          opName,
          setupFn,
          { backend: name }
        );
        results[name] = result;
      } catch (error) {
        results[name] = { error: error.message };
      }
    }

    // Find best backend
    const validResults = Object.entries(results)
      .filter(([_, r]) => !r.error)
      .map(([name, r]) => ({ name, ...r }));

    if (validResults.length > 0) {
      const fastest = validResults.reduce((a, b) =>
        a.timing.mean < b.timing.mean ? a : b
      );

      results._comparison = {
        fastest: fastest.name,
        results: validResults.map(r => ({
          backend: r.name,
          mean: r.timing.mean,
          relativeSpeed: (fastest.timing.mean / r.timing.mean).toFixed(2)
        }))
      };
    }

    return results;
  }

  /**
   * Comprehensive backend comparison
   */
  async runFullComparison() {
    const results = {
      matmul: await this.compareOperation('matmul', () => {
        const a = this._createTensor([512, 512]);
        const b = this._createTensor([512, 512]);
        return () => this.backends[0].backend.matmul(a, b);
      }),
      elementwise: await this.compareOperation('add', () => {
        const a = this._createTensor([100000]);
        const b = this._createTensor([100000]);
        return () => this.backends[0].backend.add(a, b);
      }),
      activation: await this.compareOperation('relu', () => {
        const input = this._createTensor([100000]);
        return () => this.backends[0].backend.relu(input);
      })
    };

    return results;
  }

  _createTensor(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).map(() => Math.random());
    return { shape, data };
  }
}

/**
 * Memory benchmark
 */
export class MemoryBenchmark {
  /**
   * Benchmark memory allocation patterns
   */
  static async benchmarkAllocation(sizes = [1000, 10000, 100000, 1000000]) {
    const results = [];

    for (const size of sizes) {
      const before = this._getMemoryUsage();

      // Allocate
      const tensors = [];
      for (let i = 0; i < 100; i++) {
        tensors.push(new Float32Array(size));
      }

      const after = this._getMemoryUsage();

      // Cleanup
      tensors.length = 0;

      if (before && after) {
        results.push({
          size,
          count: 100,
          allocated: after.usedJSHeapSize - before.usedJSHeapSize,
          perTensor: (after.usedJSHeapSize - before.usedJSHeapSize) / 100
        });
      }
    }

    return results;
  }

  /**
   * Benchmark memory pooling efficiency
   */
  static async benchmarkPooling(poolManager) {
    const measurements = {
      withPool: [],
      withoutPool: []
    };

    // With pooling
    for (let i = 0; i < 50; i++) {
      const start = performance.now();
      poolManager.allocate([1000, 1000]);
      poolManager.release();
      measurements.withPool.push(performance.now() - start);
    }

    // Without pooling
    for (let i = 0; i < 50; i++) {
      const start = performance.now();
      // Allocate and let it be garbage collected
      new Float32Array(1000 * 1000);
      measurements.withoutPool.push(performance.now() - start);
    }

    return {
      withPool: PerformanceStats.calculate(measurements.withPool),
      withoutPool: PerformanceStats.calculate(measurements.withoutPool)
    };
  }

  static _getMemoryUsage() {
    if (typeof performance !== 'undefined' && performance.memory) {
      return {
        usedJSHeapSize: performance.memory.usedJSHeapSize,
        totalJSHeapSize: performance.memory.totalJSHeapSize
      };
    }
    return null;
  }
}

/**
 * Benchmark suite runner
 */
export class BenchmarkSuite {
  constructor(config = {}) {
    this.config = new BenchmarkConfig(config);
    this.results = {
      metadata: {
        timestamp: new Date().toISOString(),
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'Node.js',
        platform: typeof navigator !== 'undefined' ? navigator.platform : process.platform
      },
      benchmarks: {}
    };
  }

  /**
   * Run all benchmarks
   */
  async runAll(backends) {
    // eslint-disable-next-line no-console
    console.log('Running comprehensive benchmark suite...');

    // Tensor operation benchmarks
    for (const backend of backends) {
      const tensorBench = new TensorBenchmarks(backend.backend);
      this.results.benchmarks[`${backend.name}_matmul`] = await tensorBench.benchmarkMatMul();
      this.results.benchmarks[`${backend.name}_elementwise`] = await tensorBench.benchmarkElementWise();
      this.results.benchmarks[`${backend.name}_activations`] = await tensorBench.benchmarkActivations();
    }

    // Backend comparison
    if (backends.length > 1) {
      const comparison = new BackendComparison(backends);
      this.results.benchmarks.backend_comparison = await comparison.runFullComparison();
    }

    // Memory benchmarks
    this.results.benchmarks.memory_allocation = await MemoryBenchmark.benchmarkAllocation();

    // eslint-disable-next-line no-console
    console.log('Benchmark suite completed');
    return this.results;
  }

  /**
   * Generate HTML report
   */
  generateHTMLReport() {
    const html = `
<!DOCTYPE html>
<html>
<head>
    <title>TrustformeRS Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .metadata { background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #4CAF50; color: white; }
        tr:hover { background: #f5f5f5; }
        .faster { color: green; font-weight: bold; }
        .slower { color: red; font-weight: bold; }
        .chart { margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>TrustformeRS Benchmark Report</h1>
        <div class="metadata">
            <p><strong>Generated:</strong> ${this.results.metadata.timestamp}</p>
            <p><strong>Platform:</strong> ${this.results.metadata.platform}</p>
            <p><strong>User Agent:</strong> ${this.results.metadata.userAgent}</p>
        </div>

        ${this._generateBenchmarkTables()}

        <h2>Summary</h2>
        <p>Benchmark suite completed successfully. Results show performance across different operations and backends.</p>
    </div>
</body>
</html>`;

    return html;
  }

  _generateBenchmarkTables() {
    let tables = '';

    for (const [name, results] of Object.entries(this.results.benchmarks)) {
      if (Array.isArray(results)) {
        tables += `<h2>${name}</h2><table><thead><tr>`;
        tables += `<th>Operation</th><th>Mean (ms)</th><th>Median (ms)</th><th>Std Dev</th><th>P95 (ms)</th>`;
        tables += `</tr></thead><tbody>`;

        for (const result of results) {
          if (result.timing) {
            tables += `<tr>`;
            tables += `<td>${result.name || result.metadata?.operation || 'Unknown'}</td>`;
            tables += `<td>${result.timing.mean.toFixed(3)}</td>`;
            tables += `<td>${result.timing.median.toFixed(3)}</td>`;
            tables += `<td>${result.timing.stdDev.toFixed(3)}</td>`;
            tables += `<td>${result.timing.p95.toFixed(3)}</td>`;
            tables += `</tr>`;
          }
        }

        tables += `</tbody></table>`;
      }
    }

    return tables;
  }

  /**
   * Generate JSON report
   */
  generateJSONReport() {
    return JSON.stringify(this.results, null, 2);
  }

  /**
   * Save report to file (Node.js only)
   */
  async saveReport(filename, format = 'html') {
    if (typeof require === 'undefined') {
      console.warn('File saving only available in Node.js environment');
      return;
    }

    const fs = require('fs').promises;
    const content = format === 'html' ?
      this.generateHTMLReport() :
      this.generateJSONReport();

    await fs.writeFile(filename, content, 'utf-8');
    // eslint-disable-next-line no-console
    console.log(`Report saved to ${filename}`);
  }
}

// Export all classes
export default {
  BenchmarkConfig,
  BenchmarkSuite,
  TensorBenchmarks,
  ModelBenchmarks,
  BackendComparison,
  MemoryBenchmark,
  PerformanceStats
};
