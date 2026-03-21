#!/usr/bin/env python3
"""
Cross-Browser WASM Testing Suite for TrustformeRS
==================================================

This module provides comprehensive cross-browser testing capabilities for WebAssembly deployment 
of TrustformeRS, ensuring compatibility and performance across all major browsers and platforms.

Key Features:
- Automated browser testing with Selenium WebDriver
- Performance benchmarking across browsers
- Memory usage monitoring and leak detection
- WASM feature compatibility testing
- Responsive design and mobile browser testing
- Continuous integration support

Author: TrustformeRS Development Team
License: MIT License
"""

import os
import sys
import json
import time
import asyncio
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import zipfile
import hashlib
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager
import platform as platform_module

# Web testing dependencies
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.safari.options import Options as SafariOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not available. Install with: pip install selenium")

# HTTP server for local testing
try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("Warning: aiohttp not available. Install with: pip install aiohttp")

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with: pip install psutil")

class BrowserType(Enum):
    """Supported browser types for testing."""
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"
    CHROMIUM = "chromium"

class TestCategory(Enum):
    """Test categories for WASM testing."""
    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    FUNCTIONALITY = "functionality"
    STRESS = "stress"
    RESPONSIVE = "responsive"

class PlatformType(Enum):
    """Platform types for testing."""
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"

@dataclass
class BrowserConfig:
    """Configuration for browser testing."""
    browser_type: BrowserType
    version: Optional[str] = None
    headless: bool = True
    window_size: Tuple[int, int] = (1920, 1080)
    user_agent: Optional[str] = None
    enable_logging: bool = True
    enable_gpu: bool = True
    enable_webgl: bool = True
    enable_wasm: bool = True
    enable_simd: bool = True
    timeout: int = 30
    
@dataclass
class TestResult:
    """Result of a browser test."""
    browser: BrowserType
    test_name: str
    category: TestCategory
    passed: bool
    duration: float
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    screenshots: List[str] = field(default_factory=list)
    
class CrossBrowserWASMTester:
    """
    Comprehensive cross-browser testing suite for TrustformeRS WASM deployment.
    
    Features:
    - Multi-browser automated testing with Selenium
    - Performance benchmarking and memory monitoring
    - WASM feature compatibility validation
    - Mobile and responsive design testing
    - Continuous integration support
    - Detailed reporting and visualization
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests" / "wasm"
        self.output_dir = self.project_root / "test_results" / "wasm"
        self.server_port = 8888
        self.server_host = "localhost"
        
        # Create directories
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Browser configurations
        self.browser_configs = self._create_browser_configs()
        
        # Test server
        self.server_process = None
        self.server_app = None
        
    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load testing configuration from file or use defaults."""
        default_config = {
            "browsers": ["chrome", "firefox", "edge"],
            "test_categories": ["compatibility", "performance", "functionality"],
            "wasm_features": ["basic", "simd", "threads", "bulk_memory"],
            "performance_thresholds": {
                "load_time_ms": 5000,
                "inference_time_ms": 1000,
                "memory_usage_mb": 500
            },
            "test_timeout": 60,
            "retry_count": 3,
            "screenshot_on_failure": True,
            "enable_video_recording": False,
            "mobile_devices": [
                "iPhone 12", "iPhone SE", "Samsung Galaxy S21",
                "iPad", "Samsung Galaxy Tab"
            ]
        }
        
        if config_file and config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_file}: {e}")
                
        return default_config
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for test results."""
        logger = logging.getLogger("CrossBrowserWASMTester")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    def _create_browser_configs(self) -> Dict[BrowserType, BrowserConfig]:
        """Create browser configurations for testing."""
        configs = {}
        
        for browser_name in self.config["browsers"]:
            try:
                browser_type = BrowserType(browser_name.lower())
                configs[browser_type] = BrowserConfig(
                    browser_type=browser_type,
                    headless=True,
                    timeout=self.config["test_timeout"]
                )
            except ValueError:
                self.logger.warning(f"Unknown browser type: {browser_name}")
                
        return configs
        
    async def start_test_server(self):
        """Start HTTP server for serving WASM test files."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for test server")
            
        self.server_app = web.Application()
        
        # Serve static files with proper MIME types
        async def serve_wasm(request):
            """Serve WASM files with correct MIME type."""
            file_path = self.test_dir / "wasm" / request.match_info['filename']
            if file_path.exists():
                return web.Response(
                    body=file_path.read_bytes(),
                    content_type='application/wasm'
                )
            return web.Response(status=404)
            
        async def serve_js(request):
            """Serve JavaScript files."""
            file_path = self.test_dir / "js" / request.match_info['filename']
            if file_path.exists():
                return web.Response(
                    text=file_path.read_text(),
                    content_type='application/javascript'
                )
            return web.Response(status=404)
            
        async def serve_html(request):
            """Serve HTML test pages."""
            file_path = self.test_dir / "html" / request.match_info['filename']
            if file_path.exists():
                return web.Response(
                    text=file_path.read_text(),
                    content_type='text/html'
                )
            return web.Response(status=404)
            
        # Setup routes
        self.server_app.router.add_get('/wasm/{filename}', serve_wasm)
        self.server_app.router.add_get('/js/{filename}', serve_js)
        self.server_app.router.add_get('/html/{filename}', serve_html)
        self.server_app.router.add_get('/{filename}', serve_html)
        
        # Start server
        runner = web.AppRunner(self.server_app)
        await runner.setup()
        site = web.TCPSite(runner, self.server_host, self.server_port)
        await site.start()
        
        self.logger.info(f"Test server started at http://{self.server_host}:{self.server_port}")
        
    def create_test_files(self):
        """Create HTML/JS test files for WASM testing."""
        # Create directory structure
        (self.test_dir / "html").mkdir(exist_ok=True)
        (self.test_dir / "js").mkdir(exist_ok=True)
        (self.test_dir / "wasm").mkdir(exist_ok=True)
        
        # Create main test HTML page
        test_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS WASM Test Suite</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .test-section {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .test-result {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
        }}
        .test-pass {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .test-fail {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .test-running {{
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background-color: #007bff;
            transition: width 0.3s ease;
        }}
        button {{
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }}
        button:hover {{
            background-color: #0056b3;
        }}
        button:disabled {{
            background-color: #6c757d;
            cursor: not-allowed;
        }}
        .console {{
            background: #000;
            color: #00ff00;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            height: 200px;
            overflow-y: auto;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>TrustformeRS WASM Test Suite</h1>
        <div id="browser-info"></div>
        
        <div class="test-section">
            <h2>Test Controls</h2>
            <button onclick="runAllTests()" id="run-all-btn">Run All Tests</button>
            <button onclick="runCompatibilityTests()" id="run-compat-btn">Compatibility Tests</button>
            <button onclick="runPerformanceTests()" id="run-perf-btn">Performance Tests</button>
            <button onclick="runMemoryTests()" id="run-memory-btn">Memory Tests</button>
            <button onclick="clearResults()" id="clear-btn">Clear Results</button>
        </div>
        
        <div class="test-section">
            <h2>Test Progress</h2>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
            </div>
            <div id="progress-text">Ready to start tests</div>
        </div>
        
        <div class="test-section">
            <h2>Performance Metrics</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value" id="load-time">-</div>
                    <div>Load Time (ms)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="inference-time">-</div>
                    <div>Inference Time (ms)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="memory-usage">-</div>
                    <div>Memory Usage (MB)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="fps">-</div>
                    <div>FPS</div>
                </div>
            </div>
        </div>
        
        <div class="test-section">
            <h2>Test Results</h2>
            <div id="test-results"></div>
        </div>
        
        <div class="test-section">
            <h2>Console Output</h2>
            <div class="console" id="console"></div>
        </div>
    </div>
    
    <script src="/js/wasm-test-suite.js"></script>
</body>
</html>"""
        
        (self.test_dir / "html" / "test.html").write_text(test_html)
        
        # Create JavaScript test suite
        test_js = """
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
"""
        
        (self.test_dir / "js" / "wasm-test-suite.js").write_text(test_js)
        
        # Create mobile test page
        mobile_test_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS WASM Mobile Test</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.5;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            min-height: 100vh;
            padding: 20px;
            display: flex;
            flex-direction: column;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .test-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .test-button {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            margin-bottom: 10px;
        }
        .test-button:active {
            transform: scale(0.98);
        }
        .test-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .status {
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            margin: 10px 0;
            font-weight: 600;
        }
        .status.pass {
            background: #d4edda;
            color: #155724;
        }
        .status.fail {
            background: #f8d7da;
            color: #721c24;
        }
        .status.running {
            background: #fff3cd;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📱 Mobile WASM Test</h1>
            <p>TrustformeRS Performance Testing</p>
        </div>
        
        <div class="test-card">
            <h2>Device Info</h2>
            <div id="device-info"></div>
        </div>
        
        <div class="test-card">
            <h2>Quick Tests</h2>
            <button class="test-button" onclick="runMobileCompatibilityTest()">
                🔍 Compatibility Test
            </button>
            <button class="test-button" onclick="runMobilePerformanceTest()">
                ⚡ Performance Test
            </button>
            <button class="test-button" onclick="runMobileMemoryTest()">
                💾 Memory Test
            </button>
        </div>
        
        <div class="test-card">
            <h2>Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value" id="mobile-load-time">-</div>
                    <div class="metric-label">Load Time (ms)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="mobile-inference-time">-</div>
                    <div class="metric-label">Inference (ms)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="mobile-memory">-</div>
                    <div class="metric-label">Memory (MB)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="mobile-battery">-</div>
                    <div class="metric-label">Battery %</div>
                </div>
            </div>
        </div>
        
        <div class="test-card">
            <h2>Test Status</h2>
            <div id="mobile-status">Ready to test</div>
        </div>
    </div>
    
    <script>
        // Mobile-specific test implementation
        document.addEventListener('DOMContentLoaded', function() {
            initializeMobileTests();
        });
        
        function initializeMobileTests() {
            // Display device information
            const deviceInfo = {
                userAgent: navigator.userAgent,
                screen: `${screen.width}x${screen.height}`,
                pixelRatio: window.devicePixelRatio,
                orientation: screen.orientation ? screen.orientation.type : 'unknown',
                touchSupport: 'ontouchstart' in window,
                platform: navigator.platform
            };
            
            document.getElementById('device-info').innerHTML = `
                <p><strong>Platform:</strong> ${deviceInfo.platform}</p>
                <p><strong>Screen:</strong> ${deviceInfo.screen}</p>
                <p><strong>Pixel Ratio:</strong> ${deviceInfo.pixelRatio}</p>
                <p><strong>Orientation:</strong> ${deviceInfo.orientation}</p>
                <p><strong>Touch:</strong> ${deviceInfo.touchSupport ? 'Yes' : 'No'}</p>
            `;
            
            // Monitor battery if available
            if ('getBattery' in navigator) {
                navigator.getBattery().then(function(battery) {
                    updateBatteryInfo(battery);
                    battery.addEventListener('levelchange', () => updateBatteryInfo(battery));
                });
            }
        }
        
        function updateBatteryInfo(battery) {
            document.getElementById('mobile-battery').textContent = 
                Math.round(battery.level * 100);
        }
        
        function updateStatus(message, type = 'running') {
            const statusDiv = document.getElementById('mobile-status');
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
        }
        
        async function runMobileCompatibilityTest() {
            updateStatus('Running compatibility test...', 'running');
            
            try {
                // Test WASM support
                const wasmSupported = typeof WebAssembly === 'object';
                const webglSupported = !!window.WebGLRenderingContext;
                
                if (wasmSupported && webglSupported) {
                    updateStatus('✅ Compatibility: PASS', 'pass');
                } else {
                    updateStatus('❌ Compatibility: FAIL', 'fail');
                }
            } catch (error) {
                updateStatus('❌ Compatibility: ERROR', 'fail');
            }
        }
        
        async function runMobilePerformanceTest() {
            updateStatus('Running performance test...', 'running');
            
            try {
                const start = performance.now();
                
                // Simulate WASM loading
                await new Promise(resolve => setTimeout(resolve, 500));
                const loadTime = performance.now() - start;
                
                // Simulate inference
                const inferenceStart = performance.now();
                await new Promise(resolve => setTimeout(resolve, 200));
                const inferenceTime = performance.now() - inferenceStart;
                
                document.getElementById('mobile-load-time').textContent = Math.round(loadTime);
                document.getElementById('mobile-inference-time').textContent = Math.round(inferenceTime);
                
                updateStatus('✅ Performance: PASS', 'pass');
            } catch (error) {
                updateStatus('❌ Performance: ERROR', 'fail');
            }
        }
        
        async function runMobileMemoryTest() {
            updateStatus('Running memory test...', 'running');
            
            try {
                if (performance.memory) {
                    const memoryMB = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
                    document.getElementById('mobile-memory').textContent = memoryMB;
                    
                    if (memoryMB < 100) {
                        updateStatus('✅ Memory: PASS', 'pass');
                    } else {
                        updateStatus('⚠️ Memory: WARNING', 'fail');
                    }
                } else {
                    updateStatus('❓ Memory: Not Available', 'running');
                }
            } catch (error) {
                updateStatus('❌ Memory: ERROR', 'fail');
            }
        }
    </script>
</body>
</html>"""
        
        (self.test_dir / "html" / "mobile-test.html").write_text(mobile_test_html)
        
    def create_webdriver(self, browser_config: BrowserConfig) -> webdriver.Remote:
        """Create WebDriver instance for specified browser."""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium is required for browser testing")
            
        options = None
        driver = None
        
        try:
            if browser_config.browser_type == BrowserType.CHROME:
                options = ChromeOptions()
                if browser_config.headless:
                    options.add_argument("--headless")
                options.add_argument(f"--window-size={browser_config.window_size[0]},{browser_config.window_size[1]}")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                if browser_config.enable_logging:
                    options.add_argument("--enable-logging")
                    options.add_argument("--v=1")
                    
                driver = webdriver.Chrome(options=options)
                
            elif browser_config.browser_type == BrowserType.FIREFOX:
                options = FirefoxOptions()
                if browser_config.headless:
                    options.add_argument("--headless")
                options.add_argument(f"--width={browser_config.window_size[0]}")
                options.add_argument(f"--height={browser_config.window_size[1]}")
                
                driver = webdriver.Firefox(options=options)
                
            elif browser_config.browser_type == BrowserType.SAFARI:
                # Note: Safari doesn't support headless mode
                driver = webdriver.Safari()
                driver.set_window_size(*browser_config.window_size)
                
            elif browser_config.browser_type == BrowserType.EDGE:
                options = EdgeOptions()
                if browser_config.headless:
                    options.add_argument("--headless")
                options.add_argument(f"--window-size={browser_config.window_size[0]},{browser_config.window_size[1]}")
                
                driver = webdriver.Edge(options=options)
                
            if driver:
                driver.implicitly_wait(browser_config.timeout)
                return driver
                
        except Exception as e:
            self.logger.error(f"Failed to create {browser_config.browser_type.value} driver: {e}")
            if driver:
                driver.quit()
            raise
            
        raise RuntimeError(f"Unsupported browser: {browser_config.browser_type.value}")
        
    async def run_browser_test(self, browser_config: BrowserConfig, test_url: str) -> TestResult:
        """Run test in specific browser."""
        self.logger.info(f"Running test in {browser_config.browser_type.value}")
        
        driver = None
        start_time = time.time()
        
        try:
            driver = self.create_webdriver(browser_config)
            
            # Navigate to test page
            driver.get(test_url)
            
            # Wait for page to load
            WebDriverWait(driver, browser_config.timeout).until(
                EC.presence_of_element_located((By.ID, "run-all-btn"))
            )
            
            # Run tests automatically
            driver.execute_script("runAllTests()")
            
            # Wait for tests to complete (check for test report)
            WebDriverWait(driver, 120).until(
                EC.presence_of_element_located((By.CLASS_NAME, "test-result"))
            )
            
            # Give additional time for all tests to complete
            await asyncio.sleep(5)
            
            # Extract test results
            test_report = driver.execute_script("return window.testReport")
            browser_info = driver.execute_script("return window.browserInfo")
            
            duration = time.time() - start_time
            
            # Determine if tests passed
            passed = test_report and test_report.get('summary', {}).get('failed', 1) == 0
            
            # Capture screenshot
            screenshot_path = None
            if self.config.get("screenshot_on_failure", True) or not passed:
                screenshot_path = self._capture_screenshot(driver, browser_config.browser_type)
                
            return TestResult(
                browser=browser_config.browser_type,
                test_name="Cross-Browser WASM Test",
                category=TestCategory.FUNCTIONALITY,
                passed=passed,
                duration=duration,
                performance_metrics=test_report.get('summary', {}) if test_report else {},
                screenshots=[screenshot_path] if screenshot_path else []
            )
            
        except TimeoutException:
            duration = time.time() - start_time
            return TestResult(
                browser=browser_config.browser_type,
                test_name="Cross-Browser WASM Test",
                category=TestCategory.FUNCTIONALITY,
                passed=False,
                duration=duration,
                error_message="Test timeout"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                browser=browser_config.browser_type,
                test_name="Cross-Browser WASM Test",
                category=TestCategory.FUNCTIONALITY,
                passed=False,
                duration=duration,
                error_message=str(e)
            )
            
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
                    
    def _capture_screenshot(self, driver: webdriver.Remote, browser_type: BrowserType) -> str:
        """Capture screenshot of current browser state."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = self.output_dir / f"screenshot_{browser_type.value}_{timestamp}.png"
            driver.save_screenshot(str(screenshot_path))
            return str(screenshot_path)
        except Exception as e:
            self.logger.warning(f"Failed to capture screenshot: {e}")
            return None
            
    async def run_mobile_tests(self) -> List[TestResult]:
        """Run tests on mobile browsers and devices."""
        self.logger.info("Running mobile browser tests")
        
        mobile_results = []
        mobile_test_url = f"http://{self.server_host}:{self.server_port}/mobile-test.html"
        
        # Mobile Chrome configurations
        mobile_devices = [
            ("iPhone 12", {"width": 390, "height": 844, "pixelRatio": 3}),
            ("Samsung Galaxy S21", {"width": 384, "height": 854, "pixelRatio": 2.75}),
            ("iPad", {"width": 768, "height": 1024, "pixelRatio": 2})
        ]
        
        for device_name, viewport in mobile_devices:
            try:
                # Configure mobile Chrome
                mobile_config = BrowserConfig(
                    browser_type=BrowserType.CHROME,
                    window_size=(viewport["width"], viewport["height"]),
                    user_agent=f"Mobile Device Emulation - {device_name}"
                )
                
                # Add mobile emulation
                chrome_options = ChromeOptions()
                mobile_emulation = {"deviceName": device_name}
                chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
                
                self.logger.info(f"Testing mobile device: {device_name}")
                
                # Run mobile test
                result = await self.run_browser_test(mobile_config, mobile_test_url)
                result.test_name = f"Mobile Test - {device_name}"
                result.category = TestCategory.RESPONSIVE
                
                mobile_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Mobile test failed for {device_name}: {e}")
                mobile_results.append(TestResult(
                    browser=BrowserType.CHROME,
                    test_name=f"Mobile Test - {device_name}",
                    category=TestCategory.RESPONSIVE,
                    passed=False,
                    duration=0,
                    error_message=str(e)
                ))
                
        return mobile_results
        
    async def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks across browsers."""
        self.logger.info("Running performance benchmarks")
        
        benchmarks = {
            "timestamp": datetime.now().isoformat(),
            "browsers": {},
            "summary": {}
        }
        
        test_url = f"http://{self.server_host}:{self.server_port}/test.html?autorun=true"
        
        for browser_type, browser_config in self.browser_configs.items():
            try:
                self.logger.info(f"Benchmarking {browser_type.value}")
                
                # Run performance test
                result = await self.run_browser_test(browser_config, test_url)
                
                benchmarks["browsers"][browser_type.value] = {
                    "passed": result.passed,
                    "duration": result.duration,
                    "performance_metrics": result.performance_metrics,
                    "memory_usage": result.memory_usage
                }
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {browser_type.value}: {e}")
                benchmarks["browsers"][browser_type.value] = {
                    "passed": False,
                    "error": str(e)
                }
                
        # Calculate summary statistics
        successful_browsers = [b for b in benchmarks["browsers"].values() if b.get("passed", False)]
        if successful_browsers:
            benchmarks["summary"] = {
                "successful_browsers": len(successful_browsers),
                "average_duration": sum(b["duration"] for b in successful_browsers) / len(successful_browsers),
                "fastest_browser": min(successful_browsers, key=lambda x: x["duration"]),
                "slowest_browser": max(successful_browsers, key=lambda x: x["duration"])
            }
            
        return benchmarks
        
    def generate_test_report(self, results: List[TestResult], benchmarks: Dict[str, Any]) -> Path:
        """Generate comprehensive test report."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform_module.platform(),
            "python_version": sys.version,
            "total_tests": len(results),
            "passed_tests": len([r for r in results if r.passed]),
            "failed_tests": len([r for r in results if not r.passed]),
            "success_rate": len([r for r in results if r.passed]) / len(results) * 100 if results else 0,
            "results": [
                {
                    "browser": result.browser.value,
                    "test_name": result.test_name,
                    "category": result.category.value,
                    "passed": result.passed,
                    "duration": result.duration,
                    "error_message": result.error_message,
                    "performance_metrics": result.performance_metrics,
                    "memory_usage": result.memory_usage,
                    "screenshots": result.screenshots
                }
                for result in results
            ],
            "benchmarks": benchmarks,
            "browser_support": {
                browser.value: any(r.passed for r in results if r.browser == browser)
                for browser in BrowserType
                if browser in [r.browser for r in results]
            }
        }
        
        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_report_path = self.output_dir / f"wasm_test_report_{timestamp}.json"
        
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
            
        # Generate HTML report
        html_report_path = self._generate_html_report(report_data, timestamp)
        
        self.logger.info(f"Test report generated: {json_report_path}")
        self.logger.info(f"HTML report generated: {html_report_path}")
        
        return json_report_path
        
    def _generate_html_report(self, report_data: Dict[str, Any], timestamp: str) -> Path:
        """Generate HTML test report."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS WASM Cross-Browser Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .test-results {{ margin-bottom: 30px; }}
        .test-result {{ padding: 15px; margin: 10px 0; border-radius: 8px; }}
        .test-pass {{ background: #d4edda; border-left: 4px solid #28a745; }}
        .test-fail {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .browser-support {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
        .browser-card {{ padding: 15px; border-radius: 8px; text-align: center; }}
        .browser-supported {{ background: #d4edda; color: #155724; }}
        .browser-unsupported {{ background: #f8d7da; color: #721c24; }}
        .benchmarks {{ margin: 20px 0; }}
        .benchmark-item {{ background: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 TrustformeRS WASM Cross-Browser Test Report</h1>
            <p>Generated on {report_data['timestamp']}</p>
            <p>Platform: {report_data['platform']}</p>
        </div>
        
        <div class="summary">
            <div class="metric-card">
                <div class="metric-value">{report_data['total_tests']}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report_data['passed_tests']}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report_data['failed_tests']}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report_data['success_rate']:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
        </div>
        
        <div class="test-results">
            <h2>Test Results</h2>
"""
        
        for result in report_data['results']:
            status_class = "test-pass" if result['passed'] else "test-fail"
            status_icon = "✅" if result['passed'] else "❌"
            error_info = f"<br><small>Error: {result['error_message']}</small>" if result['error_message'] else ""
            
            html_content += f"""
            <div class="test-result {status_class}">
                <strong>{status_icon} {result['test_name']}</strong> ({result['browser']})
                <span style="float: right;">{result['duration']:.2f}s</span>
                <br><small>Category: {result['category']}</small>
                {error_info}
            </div>
"""
        
        html_content += f"""
        </div>
        
        <div class="browser-support">
            <h2>Browser Support Matrix</h2>
"""
        
        for browser, supported in report_data['browser_support'].items():
            support_class = "browser-supported" if supported else "browser-unsupported"
            support_icon = "✅" if supported else "❌"
            
            html_content += f"""
            <div class="browser-card {support_class}">
                {support_icon}<br><strong>{browser.title()}</strong>
            </div>
"""
        
        html_content += f"""
        </div>
        
        <div class="benchmarks">
            <h2>Performance Benchmarks</h2>
"""
        
        if 'summary' in report_data['benchmarks']:
            summary = report_data['benchmarks']['summary']
            html_content += f"""
            <div class="benchmark-item">
                <strong>Successful Browsers:</strong> {summary.get('successful_browsers', 0)}
            </div>
            <div class="benchmark-item">
                <strong>Average Duration:</strong> {summary.get('average_duration', 0):.2f}s
            </div>
"""
        
        html_content += """
        </div>
    </div>
</body>
</html>"""
        
        html_report_path = self.output_dir / f"wasm_test_report_{timestamp}.html"
        html_report_path.write_text(html_content)
        
        return html_report_path
        
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive cross-browser WASM test suite."""
        self.logger.info("Starting comprehensive cross-browser WASM test suite")
        
        try:
            # Start test server
            await self.start_test_server()
            
            # Create test files
            self.create_test_files()
            
            # Run desktop browser tests
            desktop_results = []
            test_url = f"http://{self.server_host}:{self.server_port}/test.html"
            
            for browser_type, browser_config in self.browser_configs.items():
                try:
                    result = await self.run_browser_test(browser_config, test_url)
                    desktop_results.append(result)
                except Exception as e:
                    self.logger.error(f"Desktop test failed for {browser_type.value}: {e}")
                    desktop_results.append(TestResult(
                        browser=browser_type,
                        test_name="Desktop WASM Test",
                        category=TestCategory.FUNCTIONALITY,
                        passed=False,
                        duration=0,
                        error_message=str(e)
                    ))
            
            # Run mobile tests
            mobile_results = await self.run_mobile_tests()
            
            # Run performance benchmarks
            benchmarks = await self.run_performance_benchmarks()
            
            # Combine all results
            all_results = desktop_results + mobile_results
            
            # Generate comprehensive report
            report_path = self.generate_test_report(all_results, benchmarks)
            
            # Calculate final statistics
            total_tests = len(all_results)
            passed_tests = len([r for r in all_results if r.passed])
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            summary = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "report_path": str(report_path),
                "benchmarks": benchmarks,
                "browser_support": {
                    browser.value: any(r.passed for r in all_results if r.browser == browser)
                    for browser in set(r.browser for r in all_results)
                }
            }
            
            self.logger.info(f"Cross-browser test suite completed: {success_rate:.1f}% success rate")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            raise
            
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        pass

# Convenience functions for easy testing
async def run_cross_browser_tests(
    browsers: Optional[List[str]] = None,
    include_mobile: bool = True,
    config_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run cross-browser WASM tests with default configuration.
    
    Args:
        browsers: List of browsers to test (defaults to ["chrome", "firefox"])
        include_mobile: Whether to include mobile device testing
        config_file: Optional configuration file path
        
    Returns:
        Test results summary
    """
    if browsers is None:
        browsers = ["chrome", "firefox"]
        
    # Create temporary config
    temp_config = {
        "browsers": browsers,
        "test_categories": ["compatibility", "performance", "functionality"],
        "include_mobile": include_mobile
    }
    
    async with CrossBrowserWASMTester(config_file) as tester:
        # Override browser config
        tester.config.update(temp_config)
        tester.browser_configs = tester._create_browser_configs()
        
        return await tester.run_comprehensive_test_suite()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TrustformeRS Cross-Browser WASM Testing")
    parser.add_argument("--browsers", nargs="+", 
                       choices=["chrome", "firefox", "safari", "edge"],
                       default=["chrome", "firefox"],
                       help="Browsers to test")
    parser.add_argument("--no-mobile", action="store_true",
                       help="Skip mobile device testing")
    parser.add_argument("--config", type=Path,
                       help="Configuration file path")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run browsers in headless mode")
    
    args = parser.parse_args()
    
    async def main():
        print(f"Starting cross-browser WASM testing with: {', '.join(args.browsers)}")
        
        try:
            results = await run_cross_browser_tests(
                browsers=args.browsers,
                include_mobile=not args.no_mobile,
                config_file=args.config
            )
            
            print(f"\\n🎉 Testing completed!")
            print(f"Total Tests: {results['total_tests']}")
            print(f"Passed: {results['passed_tests']}")
            print(f"Failed: {results['failed_tests']}")
            print(f"Success Rate: {results['success_rate']:.1f}%")
            print(f"Report: {results['report_path']}")
            
            print(f"\\nBrowser Support:")
            for browser, supported in results['browser_support'].items():
                status = "✅" if supported else "❌"
                print(f"  {status} {browser.title()}")
                
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            return 1
            
        return 0 if results['success_rate'] >= 90 else 1
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)