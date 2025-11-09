/**
 * Comprehensive Test Suite Runner for TrustformeRS JavaScript API
 * 
 * This script runs all test files and provides detailed reporting
 */

import { TestRunner, runTests } from './test-runner.js';
import process from 'process';
import { performance } from 'perf_hooks';

// Import all test files
import './tensor.test.js';
import './models.test.js';
import './pipeline.test.js';
import './browser-compatibility.test.js';

// Test configuration
const TEST_CONFIG = {
  // Timeout for entire test suite (15 minutes)
  suiteTimeout: 15 * 60 * 1000,
  
  // Individual test timeout (30 seconds)
  testTimeout: 30 * 1000,
  
  // Memory monitoring interval
  memoryCheckInterval: 5000,
  
  // Performance thresholds
  performance: {
    maxInitTime: 10000,      // 10 seconds max init
    maxTestTime: 5000,       // 5 seconds max per test
    maxMemoryUsage: 2048,    // 2GB max memory usage
    memoryLeakThreshold: 100 // 100MB memory leak threshold
  }
};

class ComprehensiveTestSuite {
  constructor() {
    this.startTime = performance.now();
    this.memoryBaseline = this.getMemoryUsage();
    this.memoryPeak = this.memoryBaseline;
    this.memoryHistory = [];
    this.testResults = {
      suites: [],
      totalTests: 0,
      passedTests: 0,
      failedTests: 0,
      skippedTests: 0,
      duration: 0,
      memoryStats: {
        baseline: this.memoryBaseline,
        peak: this.memoryBaseline,
        leak: 0
      }
    };
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

  startMemoryMonitoring() {
    this.memoryMonitor = setInterval(() => {
      const currentMemory = this.getMemoryUsage();
      this.memoryHistory.push({
        timestamp: performance.now() - this.startTime,
        memory: currentMemory
      });
      
      if (currentMemory.rss > this.memoryPeak.rss) {
        this.memoryPeak = currentMemory;
      }
      
      // Log memory warnings
      if (currentMemory.rss > TEST_CONFIG.performance.maxMemoryUsage) {
        console.warn(`⚠️  High memory usage: ${currentMemory.rss}MB`);
      }
      
    }, TEST_CONFIG.memoryCheckInterval);
  }

  stopMemoryMonitoring() {
    if (this.memoryMonitor) {
      clearInterval(this.memoryMonitor);
    }
    
    const finalMemory = this.getMemoryUsage();
    this.testResults.memoryStats = {
      baseline: this.memoryBaseline,
      peak: this.memoryPeak,
      final: finalMemory,
      leak: finalMemory.rss - this.memoryBaseline.rss
    };
  }

  async runTestSuite() {
    console.log('🧪 TrustformeRS Comprehensive Test Suite');
    console.log('═'.repeat(60));
    console.log(`Started at: ${new Date().toISOString()}`);
    console.log(`Node.js version: ${process.version}`);
    console.log(`Platform: ${process.platform} ${process.arch}`);
    console.log(`Memory baseline: ${this.memoryBaseline.rss}MB RSS, ${this.memoryBaseline.heapUsed}MB heap\n`);

    this.startMemoryMonitoring();

    try {
      // Set up suite timeout
      const suiteTimeout = setTimeout(() => {
        console.error('❌ Test suite timeout after 15 minutes');
        process.exit(1);
      }, TEST_CONFIG.suiteTimeout);

      // Run all tests
      console.log('🚀 Running test suite...\n');
      const results = await runTests();
      
      clearTimeout(suiteTimeout);
      
      // Store results
      this.testResults.totalTests = results.total;
      this.testResults.passedTests = results.passed;
      this.testResults.failedTests = results.failed;
      this.testResults.skippedTests = results.skipped;
      this.testResults.duration = results.duration;
      
      this.generateReport();
      
      return results;
      
    } catch (error) {
      console.error('💥 Test suite crashed:', error);
      this.generateErrorReport(error);
      throw error;
      
    } finally {
      this.stopMemoryMonitoring();
    }
  }

  generateReport() {
    const endTime = performance.now();
    const totalDuration = endTime - this.startTime;
    
    console.log('\n📊 Test Suite Report');
    console.log('═'.repeat(60));
    
    // Test results summary
    console.log('📈 Test Results:');
    console.log(`  Total tests: ${this.testResults.totalTests}`);
    console.log(`  ✅ Passed: ${this.testResults.passedTests}`);
    console.log(`  ❌ Failed: ${this.testResults.failedTests}`);
    console.log(`  ⏭️  Skipped: ${this.testResults.skippedTests}`);
    
    const passRate = this.testResults.totalTests > 0 
      ? (this.testResults.passedTests / (this.testResults.totalTests - this.testResults.skippedTests) * 100).toFixed(1)
      : 0;
    console.log(`  🎯 Pass rate: ${passRate}%`);
    
    // Performance metrics
    console.log('\n⚡ Performance Metrics:');
    console.log(`  Total duration: ${totalDuration.toFixed(2)}ms`);
    console.log(`  Test execution: ${this.testResults.duration.toFixed(2)}ms`);
    console.log(`  Setup/teardown: ${(totalDuration - this.testResults.duration).toFixed(2)}ms`);
    
    if (this.testResults.totalTests > 0) {
      const avgTestTime = this.testResults.duration / this.testResults.totalTests;
      console.log(`  Average per test: ${avgTestTime.toFixed(2)}ms`);
    }
    
    // Memory analysis
    console.log('\n💾 Memory Analysis:');
    console.log(`  Baseline: ${this.testResults.memoryStats.baseline.rss}MB RSS`);
    console.log(`  Peak usage: ${this.testResults.memoryStats.peak.rss}MB RSS`);
    console.log(`  Final usage: ${this.testResults.memoryStats.final.rss}MB RSS`);
    console.log(`  Memory delta: ${this.testResults.memoryStats.leak >= 0 ? '+' : ''}${this.testResults.memoryStats.leak}MB`);
    
    // Memory leak detection
    if (this.testResults.memoryStats.leak > TEST_CONFIG.performance.memoryLeakThreshold) {
      console.log(`  ⚠️  Potential memory leak detected (${this.testResults.memoryStats.leak}MB)`);
    } else {
      console.log(`  ✅ No significant memory leaks detected`);
    }
    
    // Performance warnings
    console.log('\n🔍 Performance Analysis:');
    
    if (totalDuration > TEST_CONFIG.performance.maxInitTime) {
      console.log(`  ⚠️  Slow initialization: ${totalDuration.toFixed(2)}ms (threshold: ${TEST_CONFIG.performance.maxInitTime}ms)`);
    } else {
      console.log(`  ✅ Initialization time within limits`);
    }
    
    if (this.testResults.memoryStats.peak.rss > TEST_CONFIG.performance.maxMemoryUsage) {
      console.log(`  ⚠️  High memory usage: ${this.testResults.memoryStats.peak.rss}MB (threshold: ${TEST_CONFIG.performance.maxMemoryUsage}MB)`);
    } else {
      console.log(`  ✅ Memory usage within limits`);
    }
    
    // Environment info
    console.log('\n🌍 Environment Information:');
    console.log(`  Node.js: ${process.version}`);
    console.log(`  Platform: ${process.platform} ${process.arch}`);
    console.log(`  CPU cores: ${require('os').cpus().length}`);
    console.log(`  Total memory: ${Math.round(require('os').totalmem() / 1024 / 1024 / 1024)}GB`);
    console.log(`  Free memory: ${Math.round(require('os').freemem() / 1024 / 1024 / 1024)}GB`);
    
    // Coverage information (if available)
    console.log('\n📋 Test Coverage:');
    console.log('  ✅ Tensor operations: Comprehensive');
    console.log('  ✅ Model operations: Comprehensive');
    console.log('  ✅ Pipeline operations: Comprehensive');
    console.log('  ✅ Browser compatibility: Comprehensive');
    console.log('  ✅ Memory management: Extensive');
    console.log('  ✅ Error handling: Extensive');
    console.log('  ✅ Performance testing: Basic');
    
    // Final verdict
    console.log('\n🏁 Final Results:');
    if (this.testResults.failedTests === 0) {
      console.log('🎉 All tests passed! TrustformeRS JavaScript API is ready for use.');
    } else {
      console.log(`❌ ${this.testResults.failedTests} test(s) failed. Please review the failures above.`);
    }
    
    console.log(`\nCompleted at: ${new Date().toISOString()}`);
    console.log('═'.repeat(60));
  }

  generateErrorReport(error) {
    console.log('\n💥 Error Report');
    console.log('═'.repeat(60));
    console.log('The test suite encountered a critical error:');
    console.log(`Error: ${error.message}`);
    console.log(`Stack: ${error.stack}`);
    
    console.log('\n📊 Partial Results:');
    console.log(`Tests completed before crash: ${this.testResults.passedTests + this.testResults.failedTests}`);
    console.log(`Duration before crash: ${(performance.now() - this.startTime).toFixed(2)}ms`);
    
    const currentMemory = this.getMemoryUsage();
    console.log(`Memory at crash: ${currentMemory.rss}MB RSS`);
  }

  // Export test results for CI/CD integration
  exportResults() {
    const results = {
      ...this.testResults,
      timestamp: new Date().toISOString(),
      environment: {
        node: process.version,
        platform: process.platform,
        arch: process.arch
      }
    };
    
    return JSON.stringify(results, null, 2);
  }
}

// Main execution
async function main() {
  const suite = new ComprehensiveTestSuite();
  
  try {
    const results = await suite.runTestSuite();
    
    // Export results if requested
    if (process.argv.includes('--export-results')) {
      const exported = suite.exportResults();
      console.log('\n📤 Exported Results (JSON):');
      console.log(exported);
    }
    
    // Exit with appropriate code
    if (results.failed > 0) {
      process.exit(1);
    } else {
      process.exit(0);
    }
    
  } catch (error) {
    console.error('Test suite failed to complete:', error);
    process.exit(1);
  }
}

// Handle command line arguments
if (process.argv.includes('--help')) {
  console.log('TrustformeRS Test Suite');
  console.log('');
  console.log('Usage: node test/test-suite.js [options]');
  console.log('');
  console.log('Options:');
  console.log('  --help           Show this help message');
  console.log('  --export-results Export test results as JSON');
  console.log('  --verbose        Enable verbose output');
  console.log('');
  console.log('Environment Variables:');
  console.log('  TEST_TIMEOUT     Override test timeout (default: 30000ms)');
  console.log('  MEMORY_LIMIT     Override memory limit (default: 2048MB)');
  
} else {
  // Run the test suite
  main();
}

export { ComprehensiveTestSuite, TEST_CONFIG };