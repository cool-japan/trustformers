// Node.js example for trustformers-wasm

const { 
    TrustformersWasm,
    WasmTensor,
    Linear,
    LayerNorm,
    BertConfig,
    BertModelWasm,
    Timer,
    get_memory_stats,
    version,
    features,
    enable_simd
} = require('../pkg-node/trustformers_wasm.js');

async function tensorDemo() {
    console.log('\n=== Tensor Operations Demo ===');
    
    // Create tensors
    const a = WasmTensor.new([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = WasmTensor.new([7, 8, 9, 10, 11, 12], [2, 3]);
    
    console.log('Tensor A:', a.toString());
    console.log('Data:', a.data);
    
    // Operations
    const c = a.add(b);
    console.log('A + B:', c.data);
    
    const d = a.transpose();
    console.log('A.T shape:', d.shape);
    console.log('A.T data:', d.data);
    
    // Activation functions
    const e = a.relu();
    console.log('ReLU(A):', e.data);
    
    const f = a.gelu();
    console.log('GELU(A):', f.data.map(x => x.toFixed(3)));
}

async function layerDemo() {
    console.log('\n=== Layer Operations Demo ===');
    
    // Linear layer
    const linear = new Linear(4, 2, true);
    const input = WasmTensor.new([1, 2, 3, 4], [1, 4]);
    
    const output = linear.forward(input);
    console.log('Linear output shape:', output.shape);
    console.log('Linear output:', output.data.map(x => x.toFixed(3)));
    
    // Layer normalization
    const ln = new LayerNorm([4], 1e-5);
    const ln_input = WasmTensor.new([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const ln_output = ln.forward(ln_input);
    console.log('LayerNorm output:', ln_output.data.map(x => x.toFixed(3)));
}

async function modelDemo() {
    console.log('\n=== Model Inference Demo ===');
    
    // Create a tiny BERT model
    const config = BertConfig.tiny();
    console.log('Creating model with config:');
    console.log(`  Hidden size: ${config.hidden_size}`);
    console.log(`  Num layers: ${config.num_hidden_layers}`);
    console.log(`  Num heads: ${config.num_attention_heads}`);
    
    const model = new BertModelWasm(config);
    
    // Run inference
    const inputIds = [101, 2003, 1037, 2742, 102]; // Example token IDs
    console.log('Input IDs:', inputIds);
    
    const timer = new Timer("Inference");
    const output = model.forward(inputIds, null);
    const elapsed = timer.elapsed();
    
    console.log('Output shape:', output.shape);
    console.log('First 5 values:', output.data.slice(0, 5).map(x => x.toFixed(3)));
    console.log(`Inference time: ${elapsed.toFixed(2)}ms`);
}

async function performanceDemo() {
    console.log('\n=== Performance & Capabilities ===');
    
    console.log('Version:', version());
    console.log('Features:', features());
    console.log('SIMD enabled:', enable_simd());
    
    // Memory stats
    const memStats = get_memory_stats();
    console.log('\nMemory usage:');
    console.log(`  Used: ${memStats.used_mb.toFixed(2)} MB`);
    console.log(`  Limit: ${memStats.limit_mb.toFixed(2)} MB`);
    
    // Benchmark
    console.log('\nBenchmarking matrix multiplication:');
    const sizes = [10, 50, 100, 200];
    
    for (const size of sizes) {
        const timer = new Timer(`MatMul ${size}x${size}`);
        
        const a = WasmTensor.randn([size, size]);
        const b = WasmTensor.randn([size, size]);
        const c = a.matmul(b);
        
        const elapsed = timer.elapsed();
        console.log(`  ${size}x${size}: ${elapsed.toFixed(2)}ms`);
    }
}

async function main() {
    console.log('TrustformeRS WASM Node.js Example');
    console.log('=================================');
    
    // Initialize
    const tf = new TrustformersWasm();
    console.log('Initialized:', tf.initialized);
    console.log('Version:', tf.version);
    
    try {
        await tensorDemo();
        await layerDemo();
        await modelDemo();
        await performanceDemo();
        
        console.log('\n✅ All demos completed successfully!');
    } catch (error) {
        console.error('❌ Error:', error);
    }
}

// Run the demo
main().catch(console.error);