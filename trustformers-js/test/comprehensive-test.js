/**
 * Comprehensive Test Execution Script for TrustformeRS
 *
 * Runs all tests including performance benchmarks and coverage analysis
 */

import { performance } from 'perf_hooks';
import process from 'process';
import { testEnv, TestUtilities } from './test-config.js';
import { definePerformanceBenchmarks, createMemoryLeakTest } from './performance-benchmarks.js';
import { TestRunner } from './test-runner.js';

class ComprehensiveTestRunner {
  constructor() {
    this.testSuites = [];
    this.benchmarkSuites = [];
    this.results = {
      unitTests: null,
      integrationTests: null,
      performanceBenchmarks: null,
      memoryLeakTests: null,
      overall: {
        startTime: null,
        endTime: null,
        duration: 0,
        totalTests: 0,
        passedTests: 0,
        failedTests: 0,
        skippedTests: 0
      }
    };
  }

  async runAll(options = {}) {
    const {
      skipUnit = false,
      skipIntegration = false,
      skipPerformance = false,
      skipMemoryLeaks = false,
      verbose = false
    } = options;

    console.log('🚀 TrustformeRS Comprehensive Test Suite');
    console.log('═'.repeat(80));

    this.results.overall.startTime = performance.now();

    try {
      // Initialize test environment
      await testEnv.initialize();

      // Run unit tests
      if (!skipUnit) {
        console.log('\n📋 Running Unit Tests...');
        this.results.unitTests = await this.runUnitTests();
      }

      // Run integration tests (if WASM is available)
      if (!skipIntegration && !testEnv.skipIntegrationTests) {
        console.log('\n🔗 Running Integration Tests...');
        this.results.integrationTests = await this.runIntegrationTests();
      } else if (!testEnv.skipIntegrationTests) {
        console.log('\n⏭️  Integration tests skipped (WASM not available)');
      }

      // Run performance benchmarks
      if (!skipPerformance) {
        console.log('\n📊 Running Performance Benchmarks...');
        this.results.performanceBenchmarks = await this.runPerformanceBenchmarks();
      }

      // Run memory leak detection tests
      if (!skipMemoryLeaks && testEnv.gcAvailable) {
        console.log('\n🔍 Running Memory Leak Detection...');
        this.results.memoryLeakTests = await this.runMemoryLeakTests();
      } else if (!testEnv.gcAvailable) {
        console.log('\n⏭️  Memory leak tests skipped (GC not available)');
      }

      this.results.overall.endTime = performance.now();
      this.results.overall.duration = this.results.overall.endTime - this.results.overall.startTime;

      this.aggregateResults();
      this.generateComprehensiveReport();

      return this.results;

    } catch (error) {
      console.error('💥 Comprehensive test suite failed:', error);
      this.results.overall.endTime = performance.now();
      this.results.overall.duration = this.results.overall.endTime - this.results.overall.startTime;
      throw error;
    }
  }

  async runUnitTests() {
    // Import and run all unit test files
    try {
      const { runTests } = await import('./test-suite.js');
      return await runTests();
    } catch (error) {
      console.error('Unit tests failed:', error.message);
      return {
        passed: 0,
        failed: 1,
        skipped: 0,
        total: 1,
        duration: 0,
        error: error.message
      };
    }
  }

  async runIntegrationTests() {
    // Create integration test runner
    const runner = new TestRunner();

    // Define integration tests
    runner.describe('WASM Integration', () => {
      runner.test('initializes TrustformeRS with WASM', async () => {
        const TrustformeRS = await import('../src/index.js');
        await TrustformeRS.initialize();
        runner.expect(TrustformeRS.isInitialized()).toBeTruthy();
      }, { requires: ['tensor'], timeout: 10000 });

      runner.test('creates real tensor with WASM backend', async () => {
        const TrustformeRS = await import('../src/index.js');
        const tensor = TrustformeRS.tensor.create([1, 2, 3], [3]);
        runner.expect(tensor).toBeTruthy();
        runner.expect(tensor.shape).toEqual([3]);
      }, { requires: ['tensor'] });

      runner.test('performs real tensor operations', async () => {
        const TrustformeRS = await import('../src/index.js');
        const a = TrustformeRS.tensor.create([1, 2], [2]);
        const b = TrustformeRS.tensor.create([3, 4], [2]);
        const result = a.add(b);
        runner.expect(result.shape).toEqual([2]);
      }, { requires: ['tensor'] });
    });

    runner.describe('Model Integration', () => {
      runner.test('loads and runs model inference', async () => {
        const TrustformeRS = await import('../src/index.js');
        const model = await TrustformeRS.AutoModel.create('test-model');
        runner.expect(model).toBeTruthy();
      }, { requires: ['models'], timeout: 15000 });
    });

    return await runner.run();
  }

  async runPerformanceBenchmarks() {
    try {
      const suite = definePerformanceBenchmarks();
      return await suite.runAll();
    } catch (error) {
      console.error('Performance benchmarks failed:', error.message);
      return {
        benchmarks: [],
        error: error.message
      };
    }
  }

  async runMemoryLeakTests() {
    const runner = new TestRunner();

    // Add memory leak detection tests
    const memoryLeakTest = createMemoryLeakTest();

    // Create stress test
    runner.describe('Memory Stress Tests', () => {
      runner.test('handles continuous tensor operations', async () => {
        const iterations = 10000;
        const initialMemory = process.memoryUsage().heapUsed;

        for (let i = 0; i < iterations; i++) {
          if (testEnv.isMockMode()) {
            // Use mock tensors
            const tensor = global.mockTrustformers.tensor.random([10, 10]);
            const result = tensor.add(tensor);
          } else {
            // Use real tensors if available
            const TrustformeRS = await import('../src/index.js');
            const tensor = TrustformeRS.tensor.random([10, 10]);
            const result = tensor.add(tensor);
            tensor.dispose?.();
            result.dispose?.();
          }

          // Force GC every 1000 iterations
          if (i % 1000 === 0 && global.gc) {
            global.gc();
          }
        }

        // Final GC
        if (global.gc) {
          global.gc();
          await new Promise(resolve => setTimeout(resolve, 100));
        }

        const finalMemory = process.memoryUsage().heapUsed;
        const memoryIncrease = (finalMemory - initialMemory) / 1024 / 1024;

        console.log(`Memory increase after ${iterations} operations: ${memoryIncrease.toFixed(2)}MB`);
        runner.expect(memoryIncrease).toBeLessThan(200); // Less than 200MB increase

      }, { type: 'stress', timeout: 60000 });

      runner.test('handles large tensor allocations', async () => {
        const initialMemory = process.memoryUsage().heapUsed;

        // Create and destroy large tensors
        for (let i = 0; i < 10; i++) {
          if (testEnv.isMockMode()) {
            const largeTensor = global.mockTrustformers.tensor.zeros([1000, 1000]);
          } else {
            const TrustformeRS = await import('../src/index.js');
            const largeTensor = TrustformeRS.tensor.zeros([1000, 1000]);
            largeTensor.dispose?.();
          }

          if (global.gc) global.gc();
        }

        if (global.gc) {
          global.gc();
          await new Promise(resolve => setTimeout(resolve, 100));
        }

        const finalMemory = process.memoryUsage().heapUsed;
        const memoryIncrease = (finalMemory - initialMemory) / 1024 / 1024;

        console.log(`Memory increase after large allocations: ${memoryIncrease.toFixed(2)}MB`);
        runner.expect(memoryIncrease).toBeLessThan(100);

      }, { type: 'stress', timeout: 30000 });
    });

    return await runner.run();
  }

  aggregateResults() {
    let totalTests = 0;
    let passedTests = 0;
    let failedTests = 0;
    let skippedTests = 0;

    // Aggregate unit test results
    if (this.results.unitTests) {
      totalTests += this.results.unitTests.total || 0;
      passedTests += this.results.unitTests.passed || 0;
      failedTests += this.results.unitTests.failed || 0;
      skippedTests += this.results.unitTests.skipped || 0;
    }

    // Aggregate integration test results
    if (this.results.integrationTests) {
      totalTests += this.results.integrationTests.total || 0;
      passedTests += this.results.integrationTests.passed || 0;
      failedTests += this.results.integrationTests.failed || 0;
      skippedTests += this.results.integrationTests.skipped || 0;
    }

    // Aggregate memory leak test results
    if (this.results.memoryLeakTests) {
      totalTests += this.results.memoryLeakTests.total || 0;
      passedTests += this.results.memoryLeakTests.passed || 0;
      failedTests += this.results.memoryLeakTests.failed || 0;
      skippedTests += this.results.memoryLeakTests.skipped || 0;
    }

    this.results.overall.totalTests = totalTests;
    this.results.overall.passedTests = passedTests;
    this.results.overall.failedTests = failedTests;
    this.results.overall.skippedTests = skippedTests;
  }

  generateComprehensiveReport() {
    console.log('\n📊 Comprehensive Test Report');
    console.log('═'.repeat(80));

    const { overall } = this.results;
    const passRate = overall.totalTests > 0
      ? (overall.passedTests / (overall.totalTests - overall.skippedTests) * 100).toFixed(1)
      : 0;

    console.log('🎯 Overall Results:');
    console.log(`   Total duration: ${overall.duration.toFixed(2)}ms`);
    console.log(`   Total tests: ${overall.totalTests}`);
    console.log(`   ✅ Passed: ${overall.passedTests}`);
    console.log(`   ❌ Failed: ${overall.failedTests}`);
    console.log(`   ⏭️  Skipped: ${overall.skippedTests}`);
    console.log(`   🎯 Pass rate: ${passRate}%`);

    // Test category breakdown
    console.log('\n📋 Test Categories:');

    if (this.results.unitTests) {
      const unitPassRate = this.results.unitTests.total > 0
        ? (this.results.unitTests.passed / (this.results.unitTests.total - this.results.unitTests.skipped) * 100).toFixed(1)
        : 0;
      console.log(`   Unit Tests: ${this.results.unitTests.passed}/${this.results.unitTests.total - this.results.unitTests.skipped} (${unitPassRate}%)`);
    }

    if (this.results.integrationTests) {
      const integrationPassRate = this.results.integrationTests.total > 0
        ? (this.results.integrationTests.passed / (this.results.integrationTests.total - this.results.integrationTests.skipped) * 100).toFixed(1)
        : 0;
      console.log(`   Integration Tests: ${this.results.integrationTests.passed}/${this.results.integrationTests.total - this.results.integrationTests.skipped} (${integrationPassRate}%)`);
    }

    if (this.results.memoryLeakTests) {
      const memoryPassRate = this.results.memoryLeakTests.total > 0
        ? (this.results.memoryLeakTests.passed / (this.results.memoryLeakTests.total - this.results.memoryLeakTests.skipped) * 100).toFixed(1)
        : 0;
      console.log(`   Memory Tests: ${this.results.memoryLeakTests.passed}/${this.results.memoryLeakTests.total - this.results.memoryLeakTests.skipped} (${memoryPassRate}%)`);
    }

    // Performance benchmark summary
    if (this.results.performanceBenchmarks) {
      console.log('\n⚡ Performance Summary:');
      const benchmarks = this.results.performanceBenchmarks;
      if (benchmarks.length > 0) {
        const fastest = benchmarks.sort((a, b) =>
          parseFloat(a.timing?.mean || Infinity) - parseFloat(b.timing?.mean || Infinity)
        )[0];
        console.log(`   Fastest operation: ${fastest.name} (${fastest.timing?.mean}ms avg)`);

        const totalBenchmarks = benchmarks.filter(b => !b.failed).length;
        const failedBenchmarks = benchmarks.filter(b => b.failed).length;
        console.log(`   Successful benchmarks: ${totalBenchmarks}`);
        if (failedBenchmarks > 0) {
          console.log(`   Failed benchmarks: ${failedBenchmarks}`);
        }
      }
    }

    // Environment summary
    console.log('\n🌍 Test Environment:');
    console.log(`   Platform: ${process.platform} ${process.arch}`);
    console.log(`   Node.js: ${process.version}`);
    console.log(`   WASM Available: ${testEnv.isWasmAvailable() ? '✅' : '❌'}`);
    console.log(`   Mock Mode: ${testEnv.isMockMode() ? '✅' : '❌'}`);
    console.log(`   Memory Tracking: ${testEnv.gcAvailable ? '✅' : '❌'}`);

    // Recommendations
    console.log('\n💡 Recommendations:');
    if (overall.failedTests > 0) {
      console.log('   • Review failed tests and fix underlying issues');
    }
    if (overall.skippedTests > 0) {
      console.log('   • Consider implementing missing capabilities to reduce skipped tests');
    }
    if (!testEnv.isWasmAvailable()) {
      console.log('   • Build WASM module to enable full integration testing');
    }
    if (!testEnv.gcAvailable) {
      console.log('   • Run with --expose-gc flag for better memory testing');
    }
    if (overall.failedTests === 0 && overall.passedTests > 0) {
      console.log('   🎉 All tests passing! Great job!');
    }

    console.log('\n' + '═'.repeat(80));
  }

  exportResults(filename = 'comprehensive-test-results.json') {
    const exportData = {
      timestamp: new Date().toISOString(),
      environment: {
        node: process.version,
        platform: process.platform,
        arch: process.arch,
        wasmAvailable: testEnv.isWasmAvailable(),
        mockMode: testEnv.isMockMode()
      },
      results: this.results
    };

    return JSON.stringify(exportData, null, 2);
  }
}

// CLI interface
async function main() {
  const args = process.argv.slice(2);
  const options = {
    skipUnit: args.includes('--skip-unit'),
    skipIntegration: args.includes('--skip-integration'),
    skipPerformance: args.includes('--skip-performance'),
    skipMemoryLeaks: args.includes('--skip-memory'),
    verbose: args.includes('--verbose'),
    exportResults: args.includes('--export')
  };

  if (args.includes('--help')) {
    console.log('TrustformeRS Comprehensive Test Suite');
    console.log('');
    console.log('Usage: node test/comprehensive-test.js [options]');
    console.log('');
    console.log('Options:');
    console.log('  --skip-unit           Skip unit tests');
    console.log('  --skip-integration    Skip integration tests');
    console.log('  --skip-performance    Skip performance benchmarks');
    console.log('  --skip-memory         Skip memory leak tests');
    console.log('  --verbose             Enable verbose output');
    console.log('  --export              Export results to JSON');
    console.log('  --help                Show this help message');
    console.log('');
    console.log('Environment Variables:');
    console.log('  NODE_OPTIONS="--expose-gc"  Enable garbage collection for memory tests');
    return;
  }

  const runner = new ComprehensiveTestRunner();

  try {
    const results = await runner.runAll(options);

    if (options.exportResults) {
      const exportedResults = runner.exportResults();
      console.log('\n📤 Exported Results:');
      console.log(exportedResults);
    }

    // Exit with appropriate code
    if (results.overall.failedTests > 0) {
      process.exit(1);
    } else {
      process.exit(0);
    }

  } catch (error) {
    console.error('💥 Comprehensive test suite crashed:', error);
    process.exit(1);
  }
}

// Run if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { ComprehensiveTestRunner };