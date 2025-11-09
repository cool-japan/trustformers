/**
 * Test Runner for TrustformeRS JavaScript API
 * 
 * A lightweight test framework for running comprehensive tests
 */

import { performance } from 'perf_hooks';
import process from 'process';
import { testEnv, TestUtilities } from './test-config.js';

class TestRunner {
  constructor() {
    this.tests = [];
    this.suites = new Map();
    this.results = {
      passed: 0,
      failed: 0,
      skipped: 0,
      total: 0,
      duration: 0,
      failures: []
    };
    this.currentSuite = null;
    this.beforeEachCallbacks = [];
    this.afterEachCallbacks = [];
    this.beforeAllCallbacks = [];
    this.afterAllCallbacks = [];
  }

  // Test suite management
  describe(name, fn) {
    const previousSuite = this.currentSuite;
    this.currentSuite = name;
    
    if (!this.suites.has(name)) {
      this.suites.set(name, {
        name,
        tests: [],
        beforeEach: [],
        afterEach: [],
        beforeAll: [],
        afterAll: []
      });
    }
    
    fn();
    this.currentSuite = previousSuite;
  }

  // Test definition
  test(name, fn, options = {}) {
    // Handle legacy timeout parameter or new options object
    const timeout = typeof options === 'number' ? options :
                   options.timeout || testEnv?.getTestTimeout('unit') || 5000;

    const testCase = {
      name,
      fn,
      timeout,
      suite: this.currentSuite,
      skip: false,
      type: options.type || 'unit',
      requiredCapabilities: options.requires || []
    };
    
    if (this.currentSuite) {
      this.suites.get(this.currentSuite).tests.push(testCase);
    } else {
      this.tests.push(testCase);
    }
  }

  // Alias for test
  it(name, fn, options = {}) {
    this.test(name, fn, options);
  }

  // Skip test
  skip(name, fn) {
    this.test(name, fn);
    const testCase = this.tests[this.tests.length - 1] || 
                     this.suites.get(this.currentSuite).tests.slice(-1)[0];
    if (testCase) testCase.skip = true;
  }

  // Lifecycle hooks
  beforeEach(fn) {
    if (this.currentSuite) {
      this.suites.get(this.currentSuite).beforeEach.push(fn);
    } else {
      this.beforeEachCallbacks.push(fn);
    }
  }

  afterEach(fn) {
    if (this.currentSuite) {
      this.suites.get(this.currentSuite).afterEach.push(fn);
    } else {
      this.afterEachCallbacks.push(fn);
    }
  }

  beforeAll(fn) {
    if (this.currentSuite) {
      this.suites.get(this.currentSuite).beforeAll.push(fn);
    } else {
      this.beforeAllCallbacks.push(fn);
    }
  }

  afterAll(fn) {
    if (this.currentSuite) {
      this.suites.get(this.currentSuite).afterAll.push(fn);
    } else {
      this.afterAllCallbacks.push(fn);
    }
  }

  // Assertions
  expect(actual) {
    return {
      toBe: (expected) => {
        if (actual !== expected) {
          throw new Error(`Expected ${expected}, but got ${actual}`);
        }
      },
      toEqual: (expected) => {
        if (JSON.stringify(actual) !== JSON.stringify(expected)) {
          throw new Error(`Expected ${JSON.stringify(expected)}, but got ${JSON.stringify(actual)}`);
        }
      },
      toBeNull: () => {
        if (actual !== null) {
          throw new Error(`Expected null, but got ${actual}`);
        }
      },
      toBeUndefined: () => {
        if (actual !== undefined) {
          throw new Error(`Expected undefined, but got ${actual}`);
        }
      },
      toBeTruthy: () => {
        if (!actual) {
          throw new Error(`Expected truthy value, but got ${actual}`);
        }
      },
      toBeFalsy: () => {
        if (actual) {
          throw new Error(`Expected falsy value, but got ${actual}`);
        }
      },
      toThrow: () => {
        let threw = false;
        try {
          if (typeof actual === 'function') {
            actual();
          }
        } catch (e) {
          threw = true;
        }
        if (!threw) {
          throw new Error('Expected function to throw an error');
        }
      },
      toBeInstanceOf: (constructor) => {
        if (!(actual instanceof constructor)) {
          throw new Error(`Expected instance of ${constructor.name}, but got ${actual?.constructor?.name || typeof actual}`);
        }
      },
      toHaveLength: (length) => {
        if (actual.length !== length) {
          throw new Error(`Expected length ${length}, but got ${actual.length}`);
        }
      },
      toBeGreaterThan: (value) => {
        if (actual <= value) {
          throw new Error(`Expected ${actual} to be greater than ${value}`);
        }
      },
      toBeLessThan: (value) => {
        if (actual >= value) {
          throw new Error(`Expected ${actual} to be less than ${value}`);
        }
      },
      toBeCloseTo: (expected, precision = 2) => {
        const diff = Math.abs(actual - expected);
        const tolerance = Math.pow(10, -precision) / 2;
        if (diff > tolerance) {
          throw new Error(`Expected ${actual} to be close to ${expected} (precision: ${precision})`);
        }
      }
    };
  }

  // Run a single test
  async runTest(testCase) {
    const startTime = performance.now();

    try {
      // Check if test should be skipped based on capabilities
      if (testEnv && testCase.requiredCapabilities.length > 0) {
        if (testEnv.shouldSkipTest(testCase.requiredCapabilities)) {
          this.results.skipped++;
          console.log(`  ⏭️  ${testCase.name} (skipped - missing capabilities: ${testCase.requiredCapabilities.join(', ')})`);
          return;
        }
      }

      // Setup
      for (const fn of this.beforeEachCallbacks) {
        await fn();
      }

      if (testCase.suite) {
        const suite = this.suites.get(testCase.suite);
        for (const fn of suite.beforeEach) {
          await fn();
        }
      }

      // Run test with timeout using TestUtilities if available
      const testOperation = async () => testCase.fn();

      if (TestUtilities) {
        await TestUtilities.withTimeout(testOperation, testCase.timeout);
      } else {
        await Promise.race([
          testOperation(),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error(`Test timeout after ${testCase.timeout}ms`)), testCase.timeout)
          )
        ]);
      }

      // Cleanup
      for (const fn of this.afterEachCallbacks) {
        await fn();
      }
      
      if (testCase.suite) {
        const suite = this.suites.get(testCase.suite);
        for (const fn of suite.afterEach) {
          await fn();
        }
      }

      const duration = performance.now() - startTime;
      this.results.passed++;
      console.log(`  ✅ ${testCase.name} (${duration.toFixed(2)}ms)`);
      
    } catch (error) {
      const duration = performance.now() - startTime;
      this.results.failed++;
      this.results.failures.push({
        name: testCase.name,
        suite: testCase.suite,
        error: error.message,
        duration
      });
      console.log(`  ❌ ${testCase.name} (${duration.toFixed(2)}ms)`);
      console.log(`     Error: ${error.message}`);
    }
  }

  // Run all tests
  async run() {
    console.log('🧪 TrustformeRS Test Suite\n');

    // Initialize test environment
    await testEnv.initialize();

    const overallStart = performance.now();

    try {
      // Run beforeAll hooks
      for (const fn of this.beforeAllCallbacks) {
        await fn();
      }

      // Run standalone tests
      if (this.tests.length > 0) {
        console.log('📋 Standalone Tests');
        for (const testCase of this.tests) {
          if (testCase.skip) {
            this.results.skipped++;
            console.log(`  ⏭️  ${testCase.name} (skipped)`);
            continue;
          }
          await this.runTest(testCase);
        }
        console.log();
      }

      // Run test suites
      for (const [suiteName, suite] of this.suites) {
        if (suite.tests.length === 0) continue;
        
        console.log(`📂 ${suiteName}`);

        // Run suite beforeAll hooks
        for (const fn of suite.beforeAll) {
          await fn();
        }

        for (const testCase of suite.tests) {
          if (testCase.skip) {
            this.results.skipped++;
            console.log(`  ⏭️  ${testCase.name} (skipped)`);
            continue;
          }
          await this.runTest(testCase);
        }

        // Run suite afterAll hooks
        for (const fn of suite.afterAll) {
          await fn();
        }

        console.log();
      }

      // Run afterAll hooks
      for (const fn of this.afterAllCallbacks) {
        await fn();
      }

    } catch (error) {
      console.error('Test runner error:', error);
    }

    this.results.duration = performance.now() - overallStart;
    this.results.total = this.results.passed + this.results.failed + this.results.skipped;

    this.printResults();
    return this.results;
  }

  // Print test results
  printResults() {
    console.log('📊 Test Results');
    console.log('─'.repeat(50));
    console.log(`Total:   ${this.results.total}`);
    console.log(`✅ Passed: ${this.results.passed}`);
    console.log(`❌ Failed: ${this.results.failed}`);
    console.log(`⏭️  Skipped: ${this.results.skipped}`);
    console.log(`⏱️  Duration: ${this.results.duration.toFixed(2)}ms`);
    
    if (this.results.failed > 0) {
      console.log('\n💥 Failures:');
      this.results.failures.forEach((failure, i) => {
        console.log(`${i + 1}. ${failure.suite ? `${failure.suite} > ` : ''}${failure.name}`);
        console.log(`   ${failure.error}`);
      });
    }

    const passRate = (this.results.passed / (this.results.total - this.results.skipped) * 100).toFixed(1);
    console.log(`\n🎯 Pass Rate: ${passRate}%`);

    if (this.results.failed === 0) {
      console.log('\n🎉 All tests passed!');
    }
  }
}

// Global test runner instance
const runner = new TestRunner();

// Export global functions
export const describe = (name, fn) => runner.describe(name, fn);
export const test = (name, fn, options) => runner.test(name, fn, options);
export const it = (name, fn, options) => runner.it(name, fn, options);
export const skip = (name, fn) => runner.skip(name, fn);
export const beforeEach = (fn) => runner.beforeEach(fn);
export const afterEach = (fn) => runner.afterEach(fn);
export const beforeAll = (fn) => runner.beforeAll(fn);
export const afterAll = (fn) => runner.afterAll(fn);
export const expect = (actual) => runner.expect(actual);

// Run tests if this is the main module
export async function runTests() {
  const results = await runner.run();
  
  // Exit with appropriate code
  if (results.failed > 0) {
    process.exit(1);
  } else {
    process.exit(0);
  }
}

export { TestRunner };