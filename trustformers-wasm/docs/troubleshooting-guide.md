# TrustformeRS WASM Troubleshooting Guide

This guide helps diagnose and resolve common issues when using TrustformeRS WASM.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Common Issues](#common-issues)
- [Error Codes](#error-codes)
- [Browser Compatibility](#browser-compatibility)
- [Memory Issues](#memory-issues)
- [Performance Problems](#performance-problems)
- [WebGPU Issues](#webgpu-issues)
- [Network and Loading](#network-and-loading)
- [Edge Computing](#edge-computing)
- [Debugging Tools](#debugging-tools)
- [Getting Help](#getting-help)

## Quick Diagnostics

### Environment Check

Run this diagnostic script to check your environment:

```javascript
import { TrustformersWasm } from 'trustformers-wasm';

async function runDiagnostics() {
    console.log('=== TrustformeRS WASM Diagnostics ===');
    
    // Browser support
    console.log('WebAssembly support:', !!WebAssembly);
    console.log('WebGPU support:', !!navigator.gpu);
    console.log('SharedArrayBuffer support:', !!SharedArrayBuffer);
    console.log('Cross-origin isolated:', crossOriginIsolated);
    console.log('Service Worker support:', 'serviceWorker' in navigator);
    console.log('Web Workers support:', !!Worker);
    
    // Device info
    console.log('User agent:', navigator.userAgent);
    console.log('Device memory:', navigator.deviceMemory || 'unknown');
    console.log('Hardware concurrency:', navigator.hardwareConcurrency);
    
    // Try initialization
    try {
        const tf = new TrustformersWasm();
        console.log('✅ TrustformeRS initialization: SUCCESS');
        console.log('Version:', tf.version);
    } catch (error) {
        console.log('❌ TrustformeRS initialization: FAILED');
        console.error(error);
    }
    
    // WebGPU check
    if (navigator.gpu) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            console.log('✅ WebGPU adapter: SUCCESS');
            console.log('Adapter info:', adapter);
        } catch (error) {
            console.log('❌ WebGPU adapter: FAILED');
            console.error(error);
        }
    }
}

runDiagnostics();
```

### Quick Health Check

```javascript
async function healthCheck() {
    const issues = [];
    
    // Critical checks
    if (!WebAssembly) issues.push('WebAssembly not supported');
    if (!crossOriginIsolated) issues.push('Cross-origin isolation required');
    
    // Performance checks
    if (!navigator.gpu) issues.push('WebGPU not available (CPU-only mode)');
    if (!SharedArrayBuffer) issues.push('SharedArrayBuffer not available (no multi-threading)');
    
    // Memory checks
    const memory = navigator.deviceMemory;
    if (memory && memory < 2) issues.push('Low device memory (< 2GB)');
    
    if (issues.length === 0) {
        console.log('✅ All checks passed');
    } else {
        console.log('⚠️ Issues detected:');
        issues.forEach(issue => console.log('  - ' + issue));
    }
    
    return issues;
}
```

## Common Issues

### 1. Module Not Found

**Problem**: `Cannot resolve module 'trustformers-wasm'`

**Causes**:
- Package not installed
- Incorrect import path
- Build system configuration

**Solutions**:
```bash
# Install the package
npm install trustformers-wasm

# Verify installation
npm list trustformers-wasm

# Clear cache and reinstall
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**Webpack configuration**:
```javascript
// webpack.config.js
module.exports = {
  experiments: {
    asyncWebAssembly: true,
  },
  resolve: {
    extensions: ['.js', '.wasm'],
  },
};
```

### 2. WASM Loading Failed

**Problem**: `Failed to load WASM module`

**Causes**:
- CORS issues
- Incorrect MIME type
- File not found

**Solutions**:

Check server configuration:
```nginx
# nginx
location ~* \.wasm$ {
    add_header Content-Type application/wasm;
    add_header Access-Control-Allow-Origin *;
}
```

Verify file path:
```javascript
// Check if WASM file is accessible
fetch('./trustformers_wasm_bg.wasm')
  .then(response => {
    if (!response.ok) {
      console.error('WASM file not found:', response.status);
    }
  })
  .catch(error => console.error('WASM fetch error:', error));
```

### 3. Cross-Origin Isolation Required

**Problem**: `SharedArrayBuffer is not defined`

**Causes**:
- Missing security headers
- Not served over HTTPS
- Incorrect server configuration

**Solutions**:

Add required headers:
```html
<meta http-equiv="Cross-Origin-Opener-Policy" content="same-origin">
<meta http-equiv="Cross-Origin-Embedder-Policy" content="require-corp">
```

Server configuration:
```nginx
add_header Cross-Origin-Opener-Policy same-origin;
add_header Cross-Origin-Embedder-Policy require-corp;
```

Check isolation status:
```javascript
if (!crossOriginIsolated) {
    console.error('Cross-origin isolation is required for SharedArrayBuffer');
    console.log('Add security headers to enable multi-threading');
}
```

### 4. WebGPU Not Available

**Problem**: `WebGPU not supported`

**Causes**:
- Browser doesn't support WebGPU
- WebGPU disabled in flags
- Hardware limitations

**Solutions**:

Enable WebGPU (Chrome):
1. Go to `chrome://flags`
2. Enable "Unsafe WebGPU"
3. Restart browser

Fallback to CPU:
```javascript
import { is_webgpu_supported } from 'trustformers-wasm';

if (!is_webgpu_supported()) {
    console.warn('WebGPU not supported, falling back to CPU');
    await session.initialize_with_device('CPU');
} else {
    await session.initialize_with_device('GPU');
}
```

### 5. Out of Memory

**Problem**: `RangeError: Maximum call stack size exceeded` or browser crash

**Causes**:
- Model too large for device
- Memory leaks
- Insufficient device memory

**Solutions**:

Enable quantization:
```javascript
import { QuantizationConfig, QuantizationPrecision } from 'trustformers-wasm';

const quantConfig = new QuantizationConfig();
quantConfig.precision = QuantizationPrecision.Int8; // Reduce memory by ~75%
session.enable_quantization(quantConfig);
```

Monitor memory usage:
```javascript
import { get_memory_stats } from 'trustformers-wasm';

function monitorMemory() {
    const stats = get_memory_stats();
    console.log('Memory usage:', stats.wasm_memory / 1024 / 1024, 'MB');
    
    if (stats.wasm_memory > 1024 * 1024 * 1024) { // 1GB
        console.warn('High memory usage detected');
        session.cleanup(); // Force cleanup
    }
}

setInterval(monitorMemory, 5000);
```

### 6. Slow Performance

**Problem**: Inference takes too long

**Causes**:
- CPU-only execution
- Large model size
- Inefficient batching

**Solutions**:

Check device type:
```javascript
const deviceType = session.current_device_type;
console.log('Current device:', deviceType);

if (deviceType === 'CPU') {
    console.log('Consider enabling WebGPU for better performance');
}
```

Enable batch processing:
```javascript
import { BatchConfig, BatchingStrategy } from 'trustformers-wasm';

const batchConfig = new BatchConfig();
batchConfig.strategy = BatchingStrategy.Dynamic;
batchConfig.max_batch_size = 8;
session.enable_batch_processing(batchConfig);
```

Profile performance:
```javascript
import { PerformanceProfiler } from 'trustformers-wasm';

const profiler = new PerformanceProfiler();
profiler.start_profiling();

const result = session.predict(input);

const profile = profiler.stop_profiling();
console.log('Bottlenecks:', profile.bottlenecks);
```

## Error Codes

### E1xxx: Model Errors

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| E1001 | Invalid model format | Corrupted or unsupported model | Use compatible model format |
| E1002 | Model too large | Exceeds memory limit | Enable quantization or use smaller model |
| E1003 | Unsupported architecture | Model architecture not supported | Check supported architectures |
| E1004 | Model loading timeout | Network or processing timeout | Increase timeout or check network |

### E2xxx: Inference Errors

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| E2001 | Input shape mismatch | Wrong tensor dimensions | Check input tensor shape |
| E2002 | Inference failed | Computation error | Check input data validity |
| E2003 | Output buffer overflow | Result too large | Increase output buffer size |
| E2004 | Inference timeout | Processing took too long | Optimize model or increase timeout |

### E3xxx: Device Errors

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| E3001 | Device initialization failed | GPU/device setup error | Check device drivers |
| E3002 | WebGPU unavailable | Browser/hardware limitation | Use CPU fallback |
| E3003 | Memory allocation failed | Insufficient device memory | Reduce memory usage |
| E3004 | Device context lost | GPU driver crash/reset | Reinitialize session |

### E4xxx: Memory Errors

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| E4001 | Out of memory | Insufficient system memory | Enable quantization |
| E4002 | Memory allocation failed | Cannot allocate required memory | Reduce model size |
| E4003 | Memory access violation | Invalid memory access | Report bug |
| E4004 | Memory leak detected | Memory not properly released | Check cleanup calls |

### E5xxx: Configuration Errors

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| E5001 | Invalid configuration | Wrong config parameter | Check configuration docs |
| E5002 | Feature not available | Feature disabled/unsupported | Check feature availability |
| E5003 | Version mismatch | Incompatible versions | Update to compatible version |
| E5004 | Environment not supported | Unsupported runtime | Check system requirements |

## Browser Compatibility

### Chrome Issues

**WebGPU not working**:
1. Enable `chrome://flags/#enable-unsafe-webgpu`
2. Restart browser
3. Check for GPU driver updates

**SharedArrayBuffer issues**:
- Ensure site is served over HTTPS
- Add required security headers
- Check `chrome://settings/content/sharedArrayBuffer`

### Firefox Issues

**WebGPU support**:
```javascript
// Firefox WebGPU is behind flag
if (navigator.userAgent.includes('Firefox')) {
    console.log('Firefox WebGPU support is experimental');
    console.log('Enable dom.webgpu.enabled in about:config');
}
```

**WASM performance**:
- Firefox may have slower WASM execution
- Consider CPU optimizations for Firefox users

### Safari Issues

**WebGPU support**:
- Safari 18+ has experimental WebGPU support
- May need to enable in Developer settings

**Memory limits**:
- Safari has stricter memory limits on iOS
- Use smaller models for Safari/iOS

```javascript
function isSafari() {
    return /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
}

if (isSafari()) {
    // Use Safari-optimized settings
    const safariConfig = {
        memory_limit_mb: 512,
        enable_quantization: true,
        device_type: 'CPU'
    };
}
```

## Memory Issues

### Detecting Memory Leaks

```javascript
class MemoryMonitor {
    constructor() {
        this.baseline = null;
        this.samples = [];
    }
    
    startMonitoring() {
        this.baseline = this.getCurrentMemory();
        this.interval = setInterval(() => {
            this.recordSample();
        }, 5000);
    }
    
    stopMonitoring() {
        clearInterval(this.interval);
    }
    
    getCurrentMemory() {
        return {
            used: performance.memory?.usedJSHeapSize || 0,
            total: performance.memory?.totalJSHeapSize || 0,
            limit: performance.memory?.jsHeapSizeLimit || 0
        };
    }
    
    recordSample() {
        const current = this.getCurrentMemory();
        this.samples.push({
            timestamp: Date.now(),
            memory: current
        });
        
        // Check for memory growth
        if (this.samples.length > 10) {
            const growth = this.detectMemoryGrowth();
            if (growth > 50 * 1024 * 1024) { // 50MB growth
                console.warn('Potential memory leak detected:', growth / 1024 / 1024, 'MB');
            }
        }
    }
    
    detectMemoryGrowth() {
        const recent = this.samples.slice(-5);
        const older = this.samples.slice(-10, -5);
        
        const recentAvg = recent.reduce((sum, s) => sum + s.memory.used, 0) / recent.length;
        const olderAvg = older.reduce((sum, s) => sum + s.memory.used, 0) / older.length;
        
        return recentAvg - olderAvg;
    }
}

const monitor = new MemoryMonitor();
monitor.startMonitoring();
```

### Memory Optimization

```javascript
// Proper cleanup pattern
class SessionManager {
    constructor() {
        this.sessions = new Map();
    }
    
    async createSession(id, modelType) {
        // Clean up old session if exists
        if (this.sessions.has(id)) {
            this.sessions.get(id).cleanup();
        }
        
        const session = new InferenceSession(modelType);
        await session.initialize_with_auto_device();
        
        this.sessions.set(id, session);
        return session;
    }
    
    destroySession(id) {
        const session = this.sessions.get(id);
        if (session) {
            session.cleanup();
            this.sessions.delete(id);
        }
    }
    
    cleanup() {
        for (const session of this.sessions.values()) {
            session.cleanup();
        }
        this.sessions.clear();
        
        // Force garbage collection if available
        if (window.gc) {
            window.gc();
        }
    }
}
```

## Performance Problems

### Profiling Inference

```javascript
async function profileInference(session, input, iterations = 100) {
    const results = {
        times: [],
        memory_before: [],
        memory_after: []
    };
    
    for (let i = 0; i < iterations; i++) {
        const memBefore = performance.memory?.usedJSHeapSize || 0;
        
        const start = performance.now();
        await session.predict(input);
        const end = performance.now();
        
        const memAfter = performance.memory?.usedJSHeapSize || 0;
        
        results.times.push(end - start);
        results.memory_before.push(memBefore);
        results.memory_after.push(memAfter);
    }
    
    // Calculate statistics
    const avgTime = results.times.reduce((a, b) => a + b) / results.times.length;
    const p95Time = results.times.sort()[Math.floor(results.times.length * 0.95)];
    const avgMemoryGrowth = results.memory_after.map((after, i) => 
        after - results.memory_before[i]
    ).reduce((a, b) => a + b) / results.memory_after.length;
    
    console.log('Performance Profile:');
    console.log('  Average time:', avgTime.toFixed(2), 'ms');
    console.log('  P95 time:', p95Time.toFixed(2), 'ms');
    console.log('  Memory growth per inference:', avgMemoryGrowth / 1024, 'KB');
    
    return {
        avg_time_ms: avgTime,
        p95_time_ms: p95Time,
        memory_growth_kb: avgMemoryGrowth / 1024
    };
}
```

### Performance Bottleneck Detection

```javascript
class PerformanceAnalyzer {
    constructor() {
        this.operations = new Map();
    }
    
    startOperation(name) {
        this.operations.set(name, {
            start: performance.now(),
            memory_start: performance.memory?.usedJSHeapSize || 0
        });
    }
    
    endOperation(name) {
        const op = this.operations.get(name);
        if (!op) return null;
        
        const result = {
            duration: performance.now() - op.start,
            memory_delta: (performance.memory?.usedJSHeapSize || 0) - op.memory_start
        };
        
        this.operations.delete(name);
        return result;
    }
    
    async analyzeInference(session, input) {
        this.startOperation('total');
        
        this.startOperation('preprocessing');
        // Preprocessing would go here
        const preprocessing = this.endOperation('preprocessing');
        
        this.startOperation('inference');
        const result = await session.predict(input);
        const inference = this.endOperation('inference');
        
        this.startOperation('postprocessing');
        // Postprocessing would go here
        const postprocessing = this.endOperation('postprocessing');
        
        const total = this.endOperation('total');
        
        const analysis = {
            total: total.duration,
            breakdown: {
                preprocessing: preprocessing?.duration || 0,
                inference: inference.duration,
                postprocessing: postprocessing?.duration || 0
            },
            bottleneck: this.identifyBottleneck({
                preprocessing: preprocessing?.duration || 0,
                inference: inference.duration,
                postprocessing: postprocessing?.duration || 0
            })
        };
        
        console.log('Performance Analysis:', analysis);
        return analysis;
    }
    
    identifyBottleneck(breakdown) {
        const max = Math.max(...Object.values(breakdown));
        return Object.keys(breakdown).find(key => breakdown[key] === max);
    }
}
```

## WebGPU Issues

### WebGPU Adapter Problems

```javascript
async function diagnoseWebGPU() {
    if (!navigator.gpu) {
        console.log('❌ WebGPU not available in this browser');
        return false;
    }
    
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.log('❌ No WebGPU adapter found');
            return false;
        }
        
        console.log('✅ WebGPU adapter available');
        console.log('Adapter limits:', adapter.limits);
        console.log('Adapter features:', Array.from(adapter.features));
        
        const device = await adapter.requestDevice();
        console.log('✅ WebGPU device created');
        console.log('Device limits:', device.limits);
        
        return true;
    } catch (error) {
        console.log('❌ WebGPU initialization failed:', error);
        return false;
    }
}
```

### GPU Memory Issues

```javascript
class GPUMemoryMonitor {
    constructor(device) {
        this.device = device;
        this.allocatedBuffers = new Set();
    }
    
    createBuffer(descriptor) {
        const buffer = this.device.createBuffer(descriptor);
        this.allocatedBuffers.add(buffer);
        
        console.log(`GPU buffer created: ${descriptor.size} bytes`);
        console.log(`Total buffers: ${this.allocatedBuffers.size}`);
        
        return buffer;
    }
    
    destroyBuffer(buffer) {
        if (this.allocatedBuffers.has(buffer)) {
            buffer.destroy();
            this.allocatedBuffers.delete(buffer);
            console.log(`GPU buffer destroyed. Remaining: ${this.allocatedBuffers.size}`);
        }
    }
    
    cleanup() {
        for (const buffer of this.allocatedBuffers) {
            buffer.destroy();
        }
        this.allocatedBuffers.clear();
        console.log('All GPU buffers cleaned up');
    }
}
```

## Network and Loading

### Model Loading Issues

```javascript
async function debugModelLoading(modelUrl) {
    console.log('Debugging model loading for:', modelUrl);
    
    try {
        // Check if URL is accessible
        const headResponse = await fetch(modelUrl, { method: 'HEAD' });
        console.log('Model URL accessible:', headResponse.ok);
        console.log('Content-Length:', headResponse.headers.get('Content-Length'));
        console.log('Content-Type:', headResponse.headers.get('Content-Type'));
        
        // Check download speed
        const start = performance.now();
        const response = await fetch(modelUrl);
        const buffer = await response.arrayBuffer();
        const end = performance.now();
        
        const sizeMB = buffer.byteLength / 1024 / 1024;
        const timeSec = (end - start) / 1000;
        const speedMBps = sizeMB / timeSec;
        
        console.log(`Model downloaded: ${sizeMB.toFixed(2)} MB in ${timeSec.toFixed(2)}s`);
        console.log(`Download speed: ${speedMBps.toFixed(2)} MB/s`);
        
        return buffer;
    } catch (error) {
        console.error('Model loading failed:', error);
        throw error;
    }
}
```

### Network Optimization

```javascript
class NetworkOptimizer {
    static async checkConnection() {
        if ('connection' in navigator) {
            const conn = navigator.connection;
            return {
                effective_type: conn.effectiveType,
                downlink: conn.downlink,
                rtt: conn.rtt,
                save_data: conn.saveData
            };
        }
        return null;
    }
    
    static async optimizeForConnection() {
        const conn = await this.checkConnection();
        if (!conn) return null;
        
        let config = {};
        
        if (conn.save_data || conn.effective_type === 'slow-2g') {
            config = {
                quantization: 'Int4',
                model_size: 'small',
                batch_size: 1,
                cache_enabled: true
            };
        } else if (conn.effective_type === '2g' || conn.effective_type === '3g') {
            config = {
                quantization: 'Int8',
                model_size: 'medium',
                batch_size: 4,
                cache_enabled: true
            };
        } else {
            config = {
                quantization: 'FP16',
                model_size: 'large',
                batch_size: 8,
                cache_enabled: true
            };
        }
        
        console.log('Optimized config for connection:', config);
        return config;
    }
}
```

## Edge Computing

### Cold Start Issues

```javascript
class EdgeOptimizer {
    static async optimizeForEdge() {
        const isEdge = this.detectEdgeEnvironment();
        
        if (isEdge) {
            return {
                memory_limit_mb: 128,
                model_size_limit_mb: 50,
                timeout_ms: 10000,
                enable_quantization: true,
                precision: 'Int8',
                enable_model_splitting: true
            };
        }
        
        return null;
    }
    
    static detectEdgeEnvironment() {
        // Common edge environment indicators
        const userAgent = navigator.userAgent;
        const edgeKeywords = [
            'cloudflare-worker',
            'vercel-edge',
            'netlify-edge',
            'deno-deploy'
        ];
        
        return edgeKeywords.some(keyword => 
            userAgent.toLowerCase().includes(keyword)
        );
    }
    
    static async warmupModel(session) {
        // Warm up with minimal input
        const dummyInput = new Float32Array([1, 2, 3, 4]);
        const dummyTensor = new WasmTensor(dummyInput, [1, 4]);
        
        console.log('Warming up model...');
        const start = performance.now();
        
        try {
            await session.predict(dummyTensor);
            const warmupTime = performance.now() - start;
            console.log(`Model warmed up in ${warmupTime.toFixed(2)}ms`);
            return warmupTime;
        } catch (error) {
            console.error('Model warmup failed:', error);
            throw error;
        }
    }
}
```

## Debugging Tools

### Debug Logger

```javascript
class TrustformersDebugger {
    constructor() {
        this.logs = [];
        this.enabled = true;
    }
    
    log(level, message, context = {}) {
        if (!this.enabled) return;
        
        const entry = {
            timestamp: new Date().toISOString(),
            level,
            message,
            context,
            stack: new Error().stack
        };
        
        this.logs.push(entry);
        
        // Console output with styling
        const style = this.getLogStyle(level);
        console.log(`%c[TrustformeRS ${level.toUpperCase()}]`, style, message, context);
    }
    
    getLogStyle(level) {
        const styles = {
            error: 'color: red; font-weight: bold',
            warn: 'color: orange; font-weight: bold',
            info: 'color: blue',
            debug: 'color: gray'
        };
        return styles[level] || '';
    }
    
    exportLogs() {
        return JSON.stringify(this.logs, null, 2);
    }
    
    clearLogs() {
        this.logs = [];
    }
    
    enable() {
        this.enabled = true;
    }
    
    disable() {
        this.enabled = false;
    }
}

// Global debugger instance
window.TrustformersDebugger = new TrustformersDebugger();
```

### Performance Inspector

```javascript
class PerformanceInspector {
    constructor() {
        this.measurements = new Map();
    }
    
    mark(name) {
        performance.mark(name);
    }
    
    measure(name, startMark, endMark) {
        performance.measure(name, startMark, endMark);
        const measure = performance.getEntriesByName(name, 'measure')[0];
        this.measurements.set(name, measure.duration);
        return measure.duration;
    }
    
    getReport() {
        const report = {
            measurements: Object.fromEntries(this.measurements),
            memory: performance.memory ? {
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit
            } : null,
            navigation: performance.getEntriesByType('navigation')[0],
            resources: performance.getEntriesByType('resource')
        };
        
        return report;
    }
    
    clear() {
        performance.clearMarks();
        performance.clearMeasures();
        this.measurements.clear();
    }
}
```

## Getting Help

### Bug Report Template

When reporting issues, please include:

```javascript
// Run this and include the output
async function generateBugReport() {
    const report = {
        timestamp: new Date().toISOString(),
        user_agent: navigator.userAgent,
        trustformers_version: '0.1.0', // Replace with actual version
        
        environment: {
            webassembly: !!WebAssembly,
            webgpu: !!navigator.gpu,
            shared_array_buffer: !!SharedArrayBuffer,
            cross_origin_isolated: crossOriginIsolated,
            device_memory: navigator.deviceMemory,
            hardware_concurrency: navigator.hardwareConcurrency
        },
        
        error_details: {
            // Include specific error message and stack trace
        },
        
        reproduction_steps: [
            // Include minimal code to reproduce the issue
        ],
        
        performance_info: await new PerformanceInspector().getReport()
    };
    
    console.log('Bug Report:', JSON.stringify(report, null, 2));
    return report;
}
```

### Community Resources

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/trustformers/trustformers/issues)
- **Discussions**: [Ask questions and share solutions](https://github.com/trustformers/trustformers/discussions)
- **Discord**: [Real-time community chat](https://discord.gg/trustformers)
- **Documentation**: [Complete API reference](./api-reference.md)

### Professional Support

For production deployments requiring professional support:

- **Enterprise Support**: Contact enterprise@trustformers.ai
- **Consulting Services**: Available for custom implementations
- **Training**: Developer training and workshops available

## FAQ

**Q: Why is my inference slow?**
A: Check if WebGPU is enabled and working. Use the diagnostics script to verify.

**Q: How do I reduce memory usage?**
A: Enable quantization with Int8 precision, which can reduce memory by 75%.

**Q: My model won't load, what should I check?**
A: Verify CORS headers, MIME types, and file accessibility. Use the model loading debugger.

**Q: WebGPU isn't working in my browser**
A: Check browser compatibility and enable WebGPU flags if needed. Fallback to CPU mode.

**Q: How do I optimize for mobile devices?**
A: Use smaller models, enable quantization, and implement proper memory cleanup.

**Q: What's the best way to deploy to production?**
A: Follow the [deployment guide](./deployment-guide.md) for your specific platform.

This troubleshooting guide should help resolve most common issues. If you encounter problems not covered here, please refer to the community resources for additional support.