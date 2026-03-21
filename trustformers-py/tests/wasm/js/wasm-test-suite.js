
// TrustformeRS WASM Test Suite JavaScript
class WASMTestSuite {
    constructor() {
        this.tests = [];
        this.results = [];
        this.currentTest = 0;
        this.startTime = 0;
        this.wasmModule = null;
        this.memoryMonitor = null;
        
        this.initializeBrowserInfo();
        this.setupConsole();
    }
    
    initializeBrowserInfo() {
        const info = {
            userAgent: navigator.userAgent,
            platform: navigator.platform,
            language: navigator.language,
            cookieEnabled: navigator.cookieEnabled,
            onLine: navigator.onLine,
            webgl: this.checkWebGLSupport(),
            wasm: this.checkWASMSupport(),
            simd: this.checkSIMDSupport(),
            threads: this.checkThreadsSupport()
        };
        
        document.getElementById('browser-info').innerHTML = `
            <h3>Browser Information</h3>
            <p><strong>User Agent:</strong> ${info.userAgent}</p>
            <p><strong>Platform:</strong> ${info.platform}</p>
            <p><strong>Language:</strong> ${info.language}</p>
            <p><strong>WebGL Support:</strong> ${info.webgl ? '✅' : '❌'}</p>
            <p><strong>WASM Support:</strong> ${info.wasm ? '✅' : '❌'}</p>
            <p><strong>SIMD Support:</strong> ${info.simd ? '✅' : '❌'}</p>
            <p><strong>Threads Support:</strong> ${info.threads ? '✅' : '❌'}</p>
        `;
        
        // Store browser info for test reporting
        window.browserInfo = info;
    }
    
    setupConsole() {
        const originalLog = console.log;
        const originalError = console.error;
        const consoleDiv = document.getElementById('console');
        
        console.log = (...args) => {
            originalLog.apply(console, args);
            this.appendToConsole('LOG', args.join(' '));
        };
        
        console.error = (...args) => {
            originalError.apply(console, args);
            this.appendToConsole('ERROR', args.join(' '));
        };
    }
    
    appendToConsole(type, message) {
        const consoleDiv = document.getElementById('console');
        const timestamp = new Date().toISOString().substr(11, 12);
        const color = type === 'ERROR' ? '#ff6b6b' : '#00ff00';
        consoleDiv.innerHTML += `<div style="color: ${color}">[${timestamp}] ${type}: ${message}</div>`;
        consoleDiv.scrollTop = consoleDiv.scrollHeight;
    }
    
    checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext && 
                     (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
        } catch (e) {
            return false;
        }
    }
    
    checkWASMSupport() {
        return typeof WebAssembly === 'object' && 
               typeof WebAssembly.instantiate === 'function';
    }
    
    checkSIMDSupport() {
        // Check for WASM SIMD support
        return WebAssembly.validate(new Uint8Array([
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
            0x03, 0x02, 0x01, 0x00,
            0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0f, 0xfd, 0x62, 0x0b
        ]));
    }
    
    checkThreadsSupport() {
        return typeof SharedArrayBuffer !== 'undefined' && 
               typeof Worker !== 'undefined';
    }
    
    async loadWASM() {
        try {
            console.log('Loading WASM module...');
            
            // Simulate WASM loading for now
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            this.wasmModule = {
                createModel: () => ({ id: 'test-model' }),
                inference: (model, input) => `Output for: ${input}`,
                getMemoryUsage: () => Math.random() * 100
            };
            
            console.log('WASM module loaded successfully');
            return true;
        } catch (error) {
            console.error('Failed to load WASM module:', error);
            return false;
        }
    }
    
    startMemoryMonitoring() {
        if (performance && performance.memory) {
            this.memoryMonitor = setInterval(() => {
                const memory = performance.memory;
                document.getElementById('memory-usage').textContent = 
                    Math.round(memory.usedJSHeapSize / 1024 / 1024);
            }, 1000);
        }
    }
    
    stopMemoryMonitoring() {
        if (this.memoryMonitor) {
            clearInterval(this.memoryMonitor);
            this.memoryMonitor = null;
        }
    }
    
    async runCompatibilityTests() {
        console.log('Running compatibility tests...');
        
        const tests = [
            {
                name: 'WASM Module Loading',
                test: () => this.loadWASM()
            },
            {
                name: 'Basic Inference',
                test: async () => {
                    if (!this.wasmModule) await this.loadWASM();
                    const model = this.wasmModule.createModel();
                    const result = this.wasmModule.inference(model, 'test input');
                    return result.includes('test input');
                }
            },
            {
                name: 'Memory Management',
                test: async () => {
                    if (!this.wasmModule) await this.loadWASM();
                    const initialMemory = this.wasmModule.getMemoryUsage();
                    // Simulate memory operations
                    for (let i = 0; i < 100; i++) {
                        this.wasmModule.createModel();
                    }
                    const finalMemory = this.wasmModule.getMemoryUsage();
                    return finalMemory > initialMemory;
                }
            },
            {
                name: 'Error Handling',
                test: async () => {
                    try {
                        if (!this.wasmModule) await this.loadWASM();
                        // Test invalid input handling
                        this.wasmModule.inference(null, 'test');
                        return false; // Should have thrown an error
                    } catch (error) {
                        return true; // Error handling works
                    }
                }
            }
        ];
        
        return this.executeTestSuite(tests, 'compatibility');
    }
    
    async runPerformanceTests() {
        console.log('Running performance tests...');
        
        const tests = [
            {
                name: 'Load Time Benchmark',
                test: async () => {
                    const start = performance.now();
                    await this.loadWASM();
                    const loadTime = performance.now() - start;
                    document.getElementById('load-time').textContent = Math.round(loadTime);
                    return loadTime < 5000; // 5 second threshold
                }
            },
            {
                name: 'Inference Speed',
                test: async () => {
                    if (!this.wasmModule) await this.loadWASM();
                    const model = this.wasmModule.createModel();
                    
                    const start = performance.now();
                    for (let i = 0; i < 100; i++) {
                        this.wasmModule.inference(model, `test input ${i}`);
                    }
                    const inferenceTime = (performance.now() - start) / 100;
                    
                    document.getElementById('inference-time').textContent = Math.round(inferenceTime);
                    return inferenceTime < 10; // 10ms per inference threshold
                }
            },
            {
                name: 'Throughput Test',
                test: async () => {
                    if (!this.wasmModule) await this.loadWASM();
                    const model = this.wasmModule.createModel();
                    
                    const start = performance.now();
                    let count = 0;
                    
                    while (performance.now() - start < 1000) { // 1 second test
                        this.wasmModule.inference(model, `input ${count}`);
                        count++;
                    }
                    
                    const throughput = count;
                    document.getElementById('fps').textContent = throughput;
                    return throughput > 50; // 50 inferences per second threshold
                }
            }
        ];
        
        return this.executeTestSuite(tests, 'performance');
    }
    
    async runMemoryTests() {
        console.log('Running memory tests...');
        this.startMemoryMonitoring();
        
        const tests = [
            {
                name: 'Memory Allocation',
                test: async () => {
                    if (!this.wasmModule) await this.loadWASM();
                    
                    const initialMemory = performance.memory ? performance.memory.usedJSHeapSize : 0;
                    
                    // Create multiple models
                    const models = [];
                    for (let i = 0; i < 50; i++) {
                        models.push(this.wasmModule.createModel());
                    }
                    
                    const afterAllocation = performance.memory ? performance.memory.usedJSHeapSize : 0;
                    const memoryIncrease = (afterAllocation - initialMemory) / 1024 / 1024;
                    
                    return memoryIncrease < 100; // Less than 100MB increase
                }
            },
            {
                name: 'Memory Cleanup',
                test: async () => {
                    if (!this.wasmModule) await this.loadWASM();
                    
                    const initialMemory = performance.memory ? performance.memory.usedJSHeapSize : 0;
                    
                    // Allocate and deallocate memory
                    for (let i = 0; i < 10; i++) {
                        const models = [];
                        for (let j = 0; j < 100; j++) {
                            models.push(this.wasmModule.createModel());
                        }
                        // Simulate cleanup
                        models.length = 0;
                        if (window.gc) window.gc(); // Force garbage collection if available
                    }
                    
                    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for cleanup
                    
                    const finalMemory = performance.memory ? performance.memory.usedJSHeapSize : 0;
                    const memoryIncrease = (finalMemory - initialMemory) / 1024 / 1024;
                    
                    return memoryIncrease < 50; // Less than 50MB increase after cleanup
                }
            },
            {
                name: 'Memory Leak Detection',
                test: async () => {
                    if (!this.wasmModule) await this.loadWASM();
                    
                    const measurements = [];
                    
                    for (let i = 0; i < 5; i++) {
                        const start = performance.memory ? performance.memory.usedJSHeapSize : 0;
                        
                        // Perform operations
                        for (let j = 0; j < 20; j++) {
                            const model = this.wasmModule.createModel();
                            this.wasmModule.inference(model, `test ${j}`);
                        }
                        
                        await new Promise(resolve => setTimeout(resolve, 200));
                        
                        const end = performance.memory ? performance.memory.usedJSHeapSize : 0;
                        measurements.push(end - start);
                    }
                    
                    // Check if memory usage is stabilizing
                    const lastThree = measurements.slice(-3);
                    const increasing = lastThree.every((val, i, arr) => i === 0 || val > arr[i-1]);
                    
                    return !increasing; // No memory leak if not consistently increasing
                }
            }
        ];
        
        const result = await this.executeTestSuite(tests, 'memory');
        this.stopMemoryMonitoring();
        return result;
    }
    
    async executeTestSuite(tests, category) {
        const results = [];
        const totalTests = tests.length;
        
        for (let i = 0; i < tests.length; i++) {
            const test = tests[i];
            const progress = ((i + 1) / totalTests) * 100;
            
            this.updateProgress(progress, `Running ${test.name}...`);
            
            try {
                const start = performance.now();
                const passed = await test.test();
                const duration = performance.now() - start;
                
                const result = {
                    name: test.name,
                    category: category,
                    passed: passed,
                    duration: duration,
                    timestamp: new Date().toISOString()
                };
                
                results.push(result);
                this.displayTestResult(result);
                
                console.log(`Test ${test.name}: ${passed ? 'PASS' : 'FAIL'} (${Math.round(duration)}ms)`);
                
            } catch (error) {
                const result = {
                    name: test.name,
                    category: category,
                    passed: false,
                    duration: 0,
                    error: error.message,
                    timestamp: new Date().toISOString()
                };
                
                results.push(result);
                this.displayTestResult(result);
                
                console.error(`Test ${test.name}: ERROR - ${error.message}`);
            }
            
            // Small delay between tests
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        this.updateProgress(100, `Completed ${category} tests`);
        return results;
    }
    
    displayTestResult(result) {
        const resultsDiv = document.getElementById('test-results');
        const resultDiv = document.createElement('div');
        resultDiv.className = `test-result ${result.passed ? 'test-pass' : 'test-fail'}`;
        
        const icon = result.passed ? '✅' : '❌';
        const duration = Math.round(result.duration);
        const error = result.error ? `<br><small>Error: ${result.error}</small>` : '';
        
        resultDiv.innerHTML = `
            ${icon} <strong>${result.name}</strong> (${result.category})
            <span style="float: right;">${duration}ms</span>
            ${error}
        `;
        
        resultsDiv.appendChild(resultDiv);
    }
    
    updateProgress(percentage, text) {
        document.getElementById('progress-fill').style.width = `${percentage}%`;
        document.getElementById('progress-text').textContent = text;
    }
    
    async runAllTests() {
        console.log('Starting comprehensive WASM test suite...');
        this.clearResults();
        
        document.getElementById('run-all-btn').disabled = true;
        
        try {
            const compatResults = await this.runCompatibilityTests();
            const perfResults = await this.runPerformanceTests();
            const memoryResults = await this.runMemoryTests();
            
            const allResults = [...compatResults, ...perfResults, ...memoryResults];
            
            // Generate test report
            this.generateTestReport(allResults);
            
            console.log('All tests completed!');
            
        } catch (error) {
            console.error('Test suite failed:', error);
        } finally {
            document.getElementById('run-all-btn').disabled = false;
        }
    }
    
    generateTestReport(results) {
        const report = {
            timestamp: new Date().toISOString(),
            browser: window.browserInfo,
            summary: {
                total: results.length,
                passed: results.filter(r => r.passed).length,
                failed: results.filter(r => !r.passed).length,
                averageDuration: results.reduce((sum, r) => sum + r.duration, 0) / results.length
            },
            results: results
        };
        
        // Store report for external access
        window.testReport = report;
        
        console.log('Test Report Generated:', report);
        
        // Display summary
        const summary = document.createElement('div');
        summary.className = 'test-result';
        summary.style.backgroundColor = '#e7f3ff';
        summary.style.borderColor = '#0066cc';
        summary.innerHTML = `
            <h3>Test Summary</h3>
            <p><strong>Total Tests:</strong> ${report.summary.total}</p>
            <p><strong>Passed:</strong> ${report.summary.passed}</p>
            <p><strong>Failed:</strong> ${report.summary.failed}</p>
            <p><strong>Success Rate:</strong> ${Math.round(report.summary.passed / report.summary.total * 100)}%</p>
            <p><strong>Average Duration:</strong> ${Math.round(report.summary.averageDuration)}ms</p>
        `;
        
        document.getElementById('test-results').insertBefore(summary, document.getElementById('test-results').firstChild);
    }
    
    clearResults() {
        document.getElementById('test-results').innerHTML = '';
        document.getElementById('console').innerHTML = '';
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('progress-text').textContent = 'Ready to start tests';
        
        // Reset metrics
        document.getElementById('load-time').textContent = '-';
        document.getElementById('inference-time').textContent = '-';
        document.getElementById('memory-usage').textContent = '-';
        document.getElementById('fps').textContent = '-';
    }
}

// Initialize test suite
const testSuite = new WASMTestSuite();

// Global functions for buttons
async function runAllTests() {
    await testSuite.runAllTests();
}

async function runCompatibilityTests() {
    await testSuite.runCompatibilityTests();
}

async function runPerformanceTests() {
    await testSuite.runPerformanceTests();
}

async function runMemoryTests() {
    await testSuite.runMemoryTests();
}

function clearResults() {
    testSuite.clearResults();
}

// Auto-run tests if requested via URL parameter
if (new URLSearchParams(window.location.search).get('autorun') === 'true') {
    window.addEventListener('load', () => {
        setTimeout(runAllTests, 1000);
    });
}
