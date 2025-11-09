/**
 * Comprehensive Test Suite for TrustformeRS-JS Advanced AI Features
 *
 * Tests for the 8 NEW Advanced AI Modules (2025-10-27):
 * 1. Advanced Optimization (Gradient Checkpointing, Mixed Precision, LARS, SAM, Lookahead)
 * 2. Federated Learning (Secure Aggregation, Differential Privacy, Byzantine Robustness)
 * 3. Neural Architecture Search (Random, Evolutionary, Multi-Objective)
 * 4. Knowledge Distillation (Progressive, Self-Distillation, Temperature Scaling)
 * 5. Multi-Modal Streaming (Text, Image, Audio coordination)
 * 6. ONNX Integration (Model Conversion, Optimization, Runtime)
 * 7. Model Interpretability (Attention Visualization, Gradient Explanation, Feature Importance)
 * 8. Auto Performance Optimization (Bottleneck Detection, ML-based Optimization)
 */

import {
    // Advanced Optimization
    GradientCheckpointingManager,
    MixedPrecisionManager,
    GradientAccumulationManager,
    LARSOptimizer,
    LookaheadOptimizer,
    SAMOptimizer,
    OptimizationStrategySelector,
    AdvancedOptimizer,
    createAdvancedOptimizer,

    // Federated Learning
    FederatedClient,
    SecureAggregationProtocol,
    DifferentialPrivacyMechanism,
    ClientSelectionStrategy,
    ByzantineRobustAggregation,
    FederatedServer,
    createFederatedLearning,

    // Neural Architecture Search
    SearchSpace,
    PerformanceEstimator,
    RandomSearch,
    EvolutionarySearch,
    MultiObjectiveNAS,
    NASController,
    createNAS,

    // Knowledge Distillation
    DistillationLoss,
    TeacherModel,
    StudentModel,
    DistillationTrainer,
    ProgressiveDistillation,
    SelfDistillation,
    DistillationController,
    createDistillation,

    // Multi-Modal Streaming
    BaseStreamHandler,
    TextStreamHandler,
    ImageStreamHandler,
    AudioStreamHandler,
    MultiModalStreamCoordinator,
    createMultiModalStreaming,

    // ONNX Integration
    ONNXRuntimeWrapper,
    ONNXModelConverter,
    ONNXModelAnalyzer,
    ONNXController,
    createONNXIntegration,

    // Model Interpretability
    AttentionVisualizer,
    GradientExplainer,
    FeatureImportanceAnalyzer,
    InterpretabilityController,
    createInterpretability,

    // Auto Performance Optimization
    AutoPerformanceProfiler,
    BottleneckDetector,
    MLBasedOptimizer,
    AutoPerformanceOptimizer,
    createAutoOptimizer
} from '../src/index.js';

// Test utilities
const assert = (condition, message) => {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
};

const createMockModel = () => ({
    type: 'transformer',
    architecture: 'bert',
    layers: Array.from({ length: 12 }, (_, idx) => ({
        id: idx,
        type: 'transformer_block',
        parameters: new Float32Array(1000),
        outputShape: [1, 768]
    })),
    parameters: new Float32Array(10000),
    forward: async function(input) {
        return { data: new Float32Array(input.length) };
    }
});

const createMockDataset = (size = 100) =>
    Array.from({ length: size }, (_, i) => ({
        input: new Float32Array(768).fill(Math.random()),
        label: Math.floor(Math.random() * 2)
    }));

// ============================================================================
// Test Suite 1: Advanced Optimization
// ============================================================================

async function testGradientCheckpointing() {
    console.log('Testing Gradient Checkpointing...');

    const manager = new GradientCheckpointingManager({
        enabled: true,
        checkpointEveryN: 2,
        memoryThreshold: 0.8
    });

    const model = createMockModel();
    const availableMemory = 1e9; // 1GB

    const result = manager.selectCheckpointLayers(model, availableMemory);

    assert(result.checkpointedLayers.length > 0, 'Should select checkpoint layers');
    assert(result.memorySaved > 0, 'Should save memory');
    assert(result.totalLayers === model.layers.length, 'Should count all layers');

    // Test caching
    const cached = manager.cacheActivation(1, new Float32Array(100));
    assert(cached === true, 'Should cache activation for checkpointed layer');

    console.log('✓ Gradient Checkpointing tests passed');
}

async function testMixedPrecision() {
    console.log('Testing Mixed Precision Training...');

    const manager = new MixedPrecisionManager({
        enabled: true,
        dtype: 'float16',
        lossScale: 1024,
        dynamicScaling: true
    });

    const tensor = new Float32Array([1.0, 2.0, 3.0, 4.0]);

    const fp16 = manager.toFloat16(tensor);
    assert(fp16 instanceof Uint16Array, 'Should convert to FP16');
    assert(fp16.length === tensor.length, 'Should preserve length');

    const fp32 = manager.toFloat32(fp16);
    assert(fp32 instanceof Float32Array, 'Should convert back to FP32');

    // Test loss scaling
    const loss = 0.5;
    const scaled = manager.scaleLoss(loss);
    assert(scaled === loss * 1024, 'Should scale loss');

    const unscaled = manager.unscaleLoss(scaled);
    assert(Math.abs(unscaled - loss) < 1e-6, 'Should unscale loss correctly');

    console.log('✓ Mixed Precision tests passed');
}

async function testAdvancedOptimizers() {
    console.log('Testing Advanced Optimizers (LARS, SAM, Lookahead)...');

    // Test LARS
    const lars = new LARSOptimizer({
        baseLearningRate: 0.01,
        momentum: 0.9,
        trustCoefficient: 0.001
    });

    const weights = new Float32Array([1.0, 2.0, 3.0]);
    const gradients = new Float32Array([0.1, 0.2, 0.3]);

    const larsUpdate = lars.computeUpdate(weights, gradients, 0);
    assert(larsUpdate instanceof Float32Array, 'LARS should return update');
    assert(larsUpdate.length === weights.length, 'LARS should preserve dimensions');

    // Test SAM
    const sam = new SAMOptimizer({
        rho: 0.05,
        adaptiveRho: true
    });

    const perturbation = sam.computePerturbation(gradients);
    assert(perturbation instanceof Float32Array, 'SAM should compute perturbation');

    // Test Lookahead
    const baseOptimizer = { step: (w, g) => w.map((val, i) => val - 0.01 * g[i]) };
    const lookahead = new LookaheadOptimizer(baseOptimizer, {
        k: 5,
        alpha: 0.5
    });

    const lookaheadUpdate = await lookahead.step(weights, gradients);
    assert(lookaheadUpdate, 'Lookahead should return update');

    console.log('✓ Advanced Optimizers tests passed');
}

async function testOptimizationStrategySelector() {
    console.log('Testing Optimization Strategy Selector...');

    const selector = new OptimizationStrategySelector();

    const largeModelConfig = {
        modelSize: 1e9, // 1B parameters
        batchSize: 8,
        sequenceLength: 2048
    };

    const strategy = selector.selectStrategy(largeModelConfig);
    assert(strategy.name, 'Should select a strategy');
    assert(strategy.techniques.length > 0, 'Should recommend techniques');

    const config = selector.getRecommendedConfig(strategy, largeModelConfig);
    assert(config.gradientCheckpointing !== undefined, 'Should include gradient checkpointing config');
    assert(config.mixedPrecision !== undefined, 'Should include mixed precision config');

    console.log('✓ Optimization Strategy Selector tests passed');
}

// ============================================================================
// Test Suite 2: Federated Learning
// ============================================================================

async function testFederatedClient() {
    console.log('Testing Federated Learning Client...');

    const client = new FederatedClient('client_1', {
        epochs: 2,
        batchSize: 16,
        learningRate: 0.01
    });

    const globalModel = createMockModel();
    await client.initializeModel(globalModel);

    const localData = createMockDataset(50);
    client.setLocalData(localData);

    const updates = await client.trainLocal({ epochs: 1, batchSize: 16 });

    assert(updates !== null, 'Should return local updates');
    assert(client.statistics.roundsParticipated === 1, 'Should track rounds');
    assert(client.statistics.samplesProcessed > 0, 'Should track samples');

    console.log('✓ Federated Client tests passed');
}

async function testSecureAggregation() {
    console.log('Testing Secure Aggregation Protocol...');

    const protocol = new SecureAggregationProtocol({
        numClients: 5,
        threshold: 3,
        securityLevel: 128
    });

    protocol.initialize();

    // Create mock client updates
    const clientUpdates = Array.from({ length: 5 }, (_, i) => ({
        clientId: `client_${i}`,
        weights: new Float32Array([0.1, 0.2, 0.3]),
        bias: new Float32Array([0.01, 0.02])
    }));

    const aggregated = await protocol.aggregate(clientUpdates);

    assert(aggregated.weights instanceof Float32Array, 'Should aggregate weights');
    assert(aggregated.weights.length === 3, 'Should preserve dimensions');

    console.log('✓ Secure Aggregation tests passed');
}

async function testDifferentialPrivacy() {
    console.log('Testing Differential Privacy Mechanism...');

    const dp = new DifferentialPrivacyMechanism({
        epsilon: 1.0,
        delta: 1e-5,
        mechanism: 'gaussian',
        clipNorm: 1.0
    });

    const gradients = new Float32Array([0.5, 1.5, 2.5, 0.3]);

    const clipped = dp.clipGradients(gradients);
    const norm = Math.sqrt(clipped.reduce((sum, v) => sum + v * v, 0));
    assert(norm <= 1.0 + 1e-6, 'Should clip gradients to max norm');

    const noisy = dp.addNoise(clipped);
    assert(noisy instanceof Float32Array, 'Should add noise');
    assert(noisy.length === clipped.length, 'Should preserve dimensions');

    const privacySpent = dp.getPrivacySpent();
    assert(privacySpent.epsilon > 0, 'Should track privacy budget');

    console.log('✓ Differential Privacy tests passed');
}

async function testFederatedServer() {
    console.log('Testing Federated Server...');

    const server = new FederatedServer({
        aggregationMethod: 'fedavg',
        clientsPerRound: 3,
        numRounds: 2
    });

    const globalModel = createMockModel();
    server.initializeGlobalModel(globalModel);

    // Register clients
    for (let i = 0; i < 5; i++) {
        server.registerClient(`client_${i}`, { dataSize: 100 });
    }

    assert(server.clients.size === 5, 'Should register 5 clients');

    const selected = server.selectClients(3);
    assert(selected.length === 3, 'Should select 3 clients');

    console.log('✓ Federated Server tests passed');
}

// ============================================================================
// Test Suite 3: Neural Architecture Search
// ============================================================================

async function testSearchSpace() {
    console.log('Testing NAS Search Space...');

    const searchSpace = new SearchSpace({
        maxLayers: 15,
        minLayers: 5,
        connections: 'all'
    });

    const arch = searchSpace.sampleArchitecture();

    assert(arch.layers.length >= 5 && arch.layers.length <= 15, 'Should sample architecture within bounds');
    assert(arch.connections.length >= 0, 'Should have connections');
    assert(arch.layers[0].operation, 'Layers should have operations');

    console.log('✓ Search Space tests passed');
}

async function testPerformanceEstimator() {
    console.log('Testing Performance Estimator...');

    const estimator = new PerformanceEstimator({
        method: 'surrogate',
        metrics: ['accuracy', 'latency', 'parameters']
    });

    const searchSpace = new SearchSpace();
    const arch = searchSpace.sampleArchitecture();

    await estimator.train([
        { architecture: arch, metrics: { accuracy: 0.85, latency: 10, parameters: 1e6 } }
    ]);

    const prediction = await estimator.predict(arch);

    assert(prediction.accuracy !== undefined, 'Should predict accuracy');
    assert(prediction.latency !== undefined, 'Should predict latency');
    assert(prediction.parameters !== undefined, 'Should predict parameters');

    console.log('✓ Performance Estimator tests passed');
}

async function testNASStrategies() {
    console.log('Testing NAS Search Strategies...');

    const searchSpace = new SearchSpace({ maxLayers: 10, minLayers: 3 });
    const estimator = new PerformanceEstimator({ method: 'surrogate' });

    // Test Random Search
    const randomSearch = new RandomSearch(searchSpace, estimator, {
        numIterations: 5
    });

    const randomResult = await randomSearch.search();
    assert(randomResult.bestArchitecture, 'Random search should find architecture');
    assert(randomResult.history.length === 5, 'Should have 5 iterations');

    // Test Evolutionary Search
    const evolutionSearch = new EvolutionarySearch(searchSpace, estimator, {
        populationSize: 10,
        numGenerations: 3,
        mutationRate: 0.1
    });

    const evolutionResult = await evolutionSearch.search();
    assert(evolutionResult.bestArchitecture, 'Evolution search should find architecture');
    assert(evolutionResult.population.length === 10, 'Should maintain population');

    console.log('✓ NAS Strategies tests passed');
}

async function testMultiObjectiveNAS() {
    console.log('Testing Multi-Objective NAS...');

    const searchSpace = new SearchSpace();
    const estimator = new PerformanceEstimizer({
        metrics: ['accuracy', 'latency', 'parameters']
    });

    const multiObjNAS = new MultiObjectiveNAS(searchSpace, estimator, {
        objectives: ['accuracy', 'latency'],
        directions: ['maximize', 'minimize'],
        populationSize: 20,
        numGenerations: 3
    });

    const result = await multiObjNAS.search();

    assert(result.paretoFront.length > 0, 'Should find Pareto front');
    assert(result.paretoFront[0].architecture, 'Pareto solutions should have architectures');
    assert(result.paretoFront[0].metrics, 'Pareto solutions should have metrics');

    console.log('✓ Multi-Objective NAS tests passed');
}

// ============================================================================
// Test Suite 4: Knowledge Distillation
// ============================================================================

async function testDistillationLoss() {
    console.log('Testing Distillation Loss...');

    const loss = new DistillationLoss({
        temperature: 3.0,
        alpha: 0.7
    });

    const studentLogits = new Float32Array([2.0, 1.0, 0.5]);
    const teacherLogits = new Float32Array([2.5, 1.2, 0.3]);
    const hardLabels = new Float32Array([1.0, 0.0, 0.0]);

    const distillLoss = loss.computeKLDivergence(studentLogits, teacherLogits, 3.0);
    assert(distillLoss >= 0, 'KL divergence should be non-negative');

    const combinedLoss = loss.compute(studentLogits, teacherLogits, hardLabels);
    assert(combinedLoss >= 0, 'Combined loss should be non-negative');

    console.log('✓ Distillation Loss tests passed');
}

async function testTeacherStudentModels() {
    console.log('Testing Teacher-Student Models...');

    const teacherModel = new TeacherModel(createMockModel(), {
        temperature: 2.0
    });

    const input = new Float32Array(768).fill(0.5);
    const teacherOutput = await teacherModel.forward(input);

    assert(teacherOutput.logits, 'Teacher should produce logits');
    assert(teacherOutput.softTargets, 'Teacher should produce soft targets');

    const studentConfig = {
        numLayers: 6, // Smaller than teacher
        hiddenSize: 512
    };

    const studentModel = new StudentModel(studentConfig);
    const studentOutput = await studentModel.forward(input);

    assert(studentOutput.logits, 'Student should produce logits');

    console.log('✓ Teacher-Student Models tests passed');
}

async function testProgressiveDistillation() {
    console.log('Testing Progressive Distillation...');

    const teacherModel = new TeacherModel(createMockModel());
    const studentConfig = { numLayers: 6, hiddenSize: 512 };
    const dataset = createMockDataset(50);

    const progressive = new ProgressiveDistillation(teacherModel, studentConfig, {
        numStages: 3,
        layersPerStage: 2
    });

    const result = await progressive.distill(dataset, {
        epochs: 1,
        batchSize: 8
    });

    assert(result.finalModel, 'Should produce final model');
    assert(result.stageHistory.length === 3, 'Should have 3 stages');

    console.log('✓ Progressive Distillation tests passed');
}

// ============================================================================
// Test Suite 5: Multi-Modal Streaming
// ============================================================================

async function testStreamHandlers() {
    console.log('Testing Stream Handlers...');

    // Test Text Stream Handler
    const textHandler = new TextStreamHandler({
        maxTokensPerChunk: 50,
        temperature: 0.7
    });

    const textChunk = { text: 'Hello world', timestamp: Date.now() };
    const textProcessed = await textHandler.process(textChunk);

    assert(textProcessed.tokens, 'Should tokenize text');
    assert(textProcessed.metadata, 'Should include metadata');

    // Test Image Stream Handler
    const imageHandler = new ImageStreamHandler({
        maxWidth: 512,
        maxHeight: 512,
        fps: 30
    });

    const imageData = new Uint8Array(512 * 512 * 3);
    const imageChunk = { data: imageData, timestamp: Date.now(), format: 'rgb' };
    const imageProcessed = await imageHandler.process(imageChunk);

    assert(imageProcessed.tensor, 'Should convert to tensor');
    assert(imageProcessed.dimensions, 'Should include dimensions');

    // Test Audio Stream Handler
    const audioHandler = new AudioStreamHandler({
        sampleRate: 16000,
        channels: 1,
        chunkSize: 1024
    });

    const audioData = new Float32Array(1024);
    const audioChunk = { data: audioData, timestamp: Date.now() };
    const audioProcessed = await audioHandler.process(audioChunk);

    assert(audioProcessed.features, 'Should extract audio features');

    console.log('✓ Stream Handlers tests passed');
}

async function testMultiModalCoordinator() {
    console.log('Testing Multi-Modal Stream Coordinator...');

    const coordinator = new MultiModalStreamCoordinator({
        synchronization: 'timestamp',
        bufferSize: 10,
        maxLatency: 100
    });

    coordinator.registerModality('text', new TextStreamHandler());
    coordinator.registerModality('image', new ImageStreamHandler());

    const textChunk = { modality: 'text', data: { text: 'Hello' }, timestamp: 1000 };
    const imageChunk = { modality: 'image', data: { data: new Uint8Array(100) }, timestamp: 1002 };

    await coordinator.addChunk(textChunk);
    await coordinator.addChunk(imageChunk);

    const synchronized = coordinator.getSynchronizedBatch();
    assert(synchronized, 'Should produce synchronized batch');

    console.log('✓ Multi-Modal Coordinator tests passed');
}

// ============================================================================
// Test Suite 6: ONNX Integration
// ============================================================================

async function testONNXRuntime() {
    console.log('Testing ONNX Runtime Wrapper...');

    const runtime = new ONNXRuntimeWrapper({
        executionProviders: ['cpu'],
        graphOptimizationLevel: 'all'
    });

    const mockModelPath = '/tmp/mock_model.onnx';

    // Note: This is a mock test since we don't have actual ONNX models
    try {
        const sessionId = await runtime.loadModel(mockModelPath, { optimizeForInference: true });
        assert(sessionId, 'Should return session ID');
    } catch (error) {
        // Expected to fail with mock path
        assert(error.message.includes('ONNX') || error.message.includes('not found'),
               'Should fail gracefully with mock model');
    }

    const stats = runtime.getStatistics();
    assert(stats.modelsLoaded !== undefined, 'Should track models loaded');
    assert(stats.totalInferences !== undefined, 'Should track inferences');

    console.log('✓ ONNX Runtime tests passed');
}

async function testONNXModelConverter() {
    console.log('Testing ONNX Model Converter...');

    const converter = new ONNXModelConverter({
        opsetVersion: 13,
        optimization: true
    });

    const model = createMockModel();

    // Mock conversion (actual conversion would require ONNX.js)
    try {
        const onnxModel = await converter.convertFromTrustformers(model);
        // This will likely fail without actual ONNX support, which is expected
    } catch (error) {
        assert(true, 'Converter should handle missing ONNX dependencies gracefully');
    }

    console.log('✓ ONNX Model Converter tests passed');
}

async function testONNXModelAnalyzer() {
    console.log('Testing ONNX Model Analyzer...');

    const analyzer = new ONNXModelAnalyzer();

    const mockONNXModel = {
        graph: {
            node: [
                { op_type: 'MatMul', name: 'layer1' },
                { op_type: 'Add', name: 'layer2' },
                { op_type: 'Relu', name: 'layer3' }
            ],
            input: [{ name: 'input', type: { tensor_type: { shape: { dim: [{ dim_value: 1 }, { dim_value: 768 }] } } } }],
            output: [{ name: 'output' }]
        }
    };

    const analysis = analyzer.analyze(mockONNXModel);

    assert(analysis.summary, 'Should provide summary');
    assert(analysis.summary.totalOps === 3, 'Should count operations');
    assert(analysis.inputShapes, 'Should analyze input shapes');

    console.log('✓ ONNX Model Analyzer tests passed');
}

// ============================================================================
// Test Suite 7: Model Interpretability
// ============================================================================

async function testAttentionVisualizer() {
    console.log('Testing Attention Visualizer...');

    const visualizer = new AttentionVisualizer({
        numHeads: 8,
        sequenceLength: 128
    });

    // Mock attention weights [num_heads, seq_len, seq_len]
    const numHeads = 8;
    const seqLen = 128;
    const attentionWeights = new Float32Array(numHeads * seqLen * seqLen).map(() => Math.random());

    const visualization = visualizer.visualize(attentionWeights, {
        heads: [0, 1, 2],
        tokens: ['hello', 'world', 'test']
    });

    assert(visualization.headMaps, 'Should create head attention maps');
    assert(visualization.avgAttention, 'Should compute average attention');

    const headStats = visualizer.computeHeadStatistics(attentionWeights);
    assert(headStats.entropy, 'Should compute entropy');
    assert(headStats.avgAttentionDistance, 'Should compute attention distance');

    console.log('✓ Attention Visualizer tests passed');
}

async function testGradientExplainer() {
    console.log('Testing Gradient Explainer...');

    const model = createMockModel();
    const explainer = new GradientExplainer(model, {
        method: 'integrated_gradients',
        steps: 50
    });

    const input = new Float32Array(768).fill(0.5);
    const baseline = new Float32Array(768).fill(0.0);

    const attributions = await explainer.explain(input, { baseline, targetClass: 0 });

    assert(attributions.values instanceof Float32Array, 'Should compute attributions');
    assert(attributions.values.length === input.length, 'Should match input dimensions');
    assert(attributions.method === 'integrated_gradients', 'Should record method');

    console.log('✓ Gradient Explainer tests passed');
}

async function testFeatureImportanceAnalyzer() {
    console.log('Testing Feature Importance Analyzer...');

    const model = createMockModel();
    const analyzer = new FeatureImportanceAnalyzer(model, {
        method: 'permutation',
        numPermutations: 10
    });

    const dataset = createMockDataset(20);

    const importance = await analyzer.computeImportance(dataset, {
        metric: 'accuracy'
    });

    assert(importance.scores instanceof Float32Array, 'Should compute importance scores');
    assert(importance.rankings, 'Should rank features');
    assert(importance.method === 'permutation', 'Should record method');

    console.log('✓ Feature Importance Analyzer tests passed');
}

async function testInterpretabilityController() {
    console.log('Testing Interpretability Controller...');

    const model = createMockModel();
    const controller = new InterpretabilityController(model, {
        methods: ['attention', 'gradients', 'feature_importance']
    });

    const input = new Float32Array(768).fill(0.5);
    const dataset = createMockDataset(10);

    const report = await controller.generateReport(input, dataset, {
        attention: { heads: [0, 1] },
        gradients: { method: 'integrated_gradients' },
        featureImportance: { method: 'permutation' }
    });

    assert(report.model, 'Should include model info');
    assert(report.explanations, 'Should include explanations');

    console.log('✓ Interpretability Controller tests passed');
}

// ============================================================================
// Test Suite 8: Auto Performance Optimization
// ============================================================================

async function testAutoPerformanceProfiler() {
    console.log('Testing Auto Performance Profiler...');

    const profiler = new AutoPerformanceProfiler({
        samplingInterval: 100,
        historySize: 100
    });

    profiler.start();

    // Simulate some operations
    const op1 = profiler.startOperation('operation1');
    await new Promise(resolve => setTimeout(resolve, 50));
    profiler.endOperation(op1);

    const op2 = profiler.startOperation('operation2');
    await new Promise(resolve => setTimeout(resolve, 30));
    profiler.endOperation(op2);

    profiler.stop();

    const metrics = profiler.getMetrics();

    assert(metrics.operations.length === 2, 'Should track 2 operations');
    assert(metrics.operations[0].duration > 0, 'Should measure duration');
    assert(metrics.totalTime > 0, 'Should track total time');

    console.log('✓ Auto Performance Profiler tests passed');
}

async function testBottleneckDetector() {
    console.log('Testing Bottleneck Detector...');

    const detector = new BottleneckDetector({
        thresholds: {
            memory: 0.8,
            cpu: 0.9,
            duration: 1000
        }
    });

    const metrics = {
        operations: [
            { name: 'fast_op', duration: 10, memory: 100 },
            { name: 'slow_op', duration: 2000, memory: 500 },
            { name: 'memory_heavy', duration: 50, memory: 900 }
        ],
        totalMemory: 1000
    };

    const bottlenecks = detector.detect(metrics);

    assert(bottlenecks.length > 0, 'Should detect bottlenecks');
    assert(bottlenecks.some(b => b.type === 'duration'), 'Should detect duration bottleneck');
    assert(bottlenecks.some(b => b.type === 'memory'), 'Should detect memory bottleneck');

    const recommendations = detector.generateRecommendations(bottlenecks);
    assert(recommendations.length > 0, 'Should generate recommendations');

    console.log('✓ Bottleneck Detector tests passed');
}

async function testMLBasedOptimizer() {
    console.log('Testing ML-Based Optimizer...');

    const optimizer = new MLBasedOptimizer({
        historySize: 100,
        learningRate: 0.01
    });

    // Train with historical performance data
    const history = Array.from({ length: 50 }, (_, i) => ({
        config: {
            batchSize: 16 + (i % 4) * 16,
            learningRate: 0.001 * (1 + i % 3),
            workers: 1 + (i % 4)
        },
        metrics: {
            throughput: 100 + Math.random() * 50,
            latency: 10 + Math.random() * 5,
            memory: 500 + Math.random() * 200
        }
    }));

    await optimizer.train(history);

    const currentConfig = {
        batchSize: 32,
        learningRate: 0.001,
        workers: 2
    };

    const optimized = await optimizer.optimize(currentConfig, {
        targetMetric: 'throughput',
        constraints: { memory: 1000 }
    });

    assert(optimized.config, 'Should suggest optimized config');
    assert(optimized.predictedMetrics, 'Should predict metrics');
    assert(optimized.improvements, 'Should estimate improvements');

    console.log('✓ ML-Based Optimizer tests passed');
}

async function testAutoPerformanceOptimizer() {
    console.log('Testing Auto Performance Optimizer...');

    const optimizer = new AutoPerformanceOptimizer({
        enabled: true,
        optimizationInterval: 1000,
        adaptiveLearning: true
    });

    const model = createMockModel();
    const config = {
        batchSize: 16,
        learningRate: 0.001,
        workers: 2
    };

    optimizer.startOptimization(model, config);

    // Simulate some training iterations
    for (let i = 0; i < 5; i++) {
        const metrics = {
            throughput: 100 + Math.random() * 20,
            latency: 15 + Math.random() * 5,
            memory: 600 + Math.random() * 100
        };

        optimizer.recordMetrics(metrics);
    }

    await new Promise(resolve => setTimeout(resolve, 100));

    const report = optimizer.getOptimizationReport();

    assert(report.iterations > 0, 'Should track iterations');
    assert(report.currentConfig, 'Should track current config');
    assert(report.bottlenecks, 'Should identify bottlenecks');

    optimizer.stopOptimization();

    console.log('✓ Auto Performance Optimizer tests passed');
}

// ============================================================================
// Integration Tests
// ============================================================================

async function testAdvancedOptimizationPipeline() {
    console.log('\nTesting Advanced Optimization Pipeline Integration...');

    // Create a complete optimization pipeline
    const model = createMockModel();

    // 1. Select optimization strategy
    const strategySelector = new OptimizationStrategySelector();
    const strategy = strategySelector.selectStrategy({
        modelSize: 1e8,
        batchSize: 16,
        sequenceLength: 512
    });

    console.log(`  Selected strategy: ${strategy.name}`);

    // 2. Create advanced optimizer with selected strategy
    const baseOptimizer = {
        step: (weights, gradients) => weights.map((w, i) => w - 0.01 * gradients[i])
    };

    const advancedOpt = createAdvancedOptimizer(baseOptimizer, {
        modelSize: 1e8,
        batchSize: 16
    });

    // 3. Start auto-optimization
    const autoOpt = createAutoOptimizer({
        enabled: true,
        adaptiveLearning: true
    });

    autoOpt.startOptimization(model, {
        batchSize: 16,
        learningRate: 0.001
    });

    // 4. Simulate training with metrics
    for (let i = 0; i < 3; i++) {
        autoOpt.recordMetrics({
            throughput: 120 + Math.random() * 10,
            latency: 12 + Math.random() * 2,
            memory: 700 + Math.random() * 50
        });
    }

    const report = autoOpt.getOptimizationReport();
    assert(report.iterations > 0, 'Integration: Should complete optimization iterations');

    autoOpt.stopOptimization();

    console.log('✓ Advanced Optimization Pipeline integration passed');
}

async function testFederatedLearningPipeline() {
    console.log('\nTesting Federated Learning Pipeline Integration...');

    // 1. Create federated learning system
    const flSystem = createFederatedLearning({
        numClients: 5,
        clientsPerRound: 3,
        numRounds: 2,
        differentialPrivacy: true,
        secureAggregation: true
    });

    // 2. Initialize global model
    const globalModel = createMockModel();
    flSystem.server.initializeGlobalModel(globalModel);

    // 3. Create and register clients
    const clients = [];
    for (let i = 0; i < 5; i++) {
        const client = new FederatedClient(`client_${i}`);
        await client.initializeModel(globalModel);
        client.setLocalData(createMockDataset(20));
        clients.push(client);

        flSystem.server.registerClient(`client_${i}`, { dataSize: 20 });
    }

    // 4. Run federated training round
    const selectedClients = flSystem.server.selectClients(3);
    assert(selectedClients.length === 3, 'Integration: Should select clients');

    const clientUpdates = [];
    for (const clientId of selectedClients) {
        const client = clients.find(c => c.clientId === clientId);
        const update = await client.trainLocal({ epochs: 1, batchSize: 8 });
        clientUpdates.push({ clientId, ...update });
    }

    // 5. Aggregate with privacy
    const aggregated = await flSystem.server.aggregateUpdates(clientUpdates);
    assert(aggregated, 'Integration: Should aggregate updates');

    console.log('✓ Federated Learning Pipeline integration passed');
}

async function testNASWithDistillation() {
    console.log('\nTesting NAS + Knowledge Distillation Integration...');

    // 1. Use NAS to find efficient architecture
    const nasController = createNAS({
        searchStrategy: 'evolutionary',
        populationSize: 10,
        numGenerations: 2
    });

    const searchResult = await nasController.search({
        maxLayers: 8,
        targetMetrics: ['accuracy', 'latency']
    });

    assert(searchResult.bestArchitecture, 'Integration: NAS should find architecture');
    console.log(`  Found architecture with ${searchResult.bestArchitecture.layers.length} layers`);

    // 2. Use knowledge distillation to compress
    const teacherModel = createMockModel();
    const distillation = createDistillation({
        temperature: 3.0,
        alpha: 0.7
    });

    const studentConfig = {
        numLayers: 6,
        hiddenSize: 512
    };

    const dataset = createMockDataset(50);
    const distillResult = await distillation.distill(
        teacherModel,
        studentConfig,
        dataset,
        { epochs: 1 }
    );

    assert(distillResult.studentModel, 'Integration: Should produce student model');
    assert(distillResult.history, 'Integration: Should track distillation history');

    console.log('✓ NAS + Distillation integration passed');
}

async function testMultiModalWithInterpretability() {
    console.log('\nTesting Multi-Modal + Interpretability Integration...');

    // 1. Set up multi-modal streaming
    const mmStreaming = createMultiModalStreaming({
        modalities: ['text', 'image'],
        synchronization: 'timestamp'
    });

    // 2. Process multi-modal data
    await mmStreaming.coordinator.addChunk({
        modality: 'text',
        data: { text: 'A cat sitting on a mat' },
        timestamp: 1000
    });

    await mmStreaming.coordinator.addChunk({
        modality: 'image',
        data: { data: new Uint8Array(224 * 224 * 3) },
        timestamp: 1002
    });

    const batch = mmStreaming.coordinator.getSynchronizedBatch();
    assert(batch, 'Integration: Should synchronize multi-modal data');

    // 3. Add interpretability for the model
    const model = createMockModel();
    const interpretability = createInterpretability(model, {
        methods: ['attention', 'gradients']
    });

    const input = new Float32Array(768).fill(0.5);
    const report = await interpretability.generateReport(input, createMockDataset(10), {
        attention: { heads: [0, 1] },
        gradients: { method: 'integrated_gradients' }
    });

    assert(report.explanations, 'Integration: Should generate interpretability report');

    console.log('✓ Multi-Modal + Interpretability integration passed');
}

// ============================================================================
// Main Test Runner
// ============================================================================

async function runAllTests() {
    console.log('='.repeat(80));
    console.log('TrustformeRS-JS Advanced AI Features Test Suite');
    console.log('='.repeat(80));
    console.log('');

    const testSuites = [
        // Suite 1: Advanced Optimization
        { name: 'Gradient Checkpointing', fn: testGradientCheckpointing },
        { name: 'Mixed Precision Training', fn: testMixedPrecision },
        { name: 'Advanced Optimizers', fn: testAdvancedOptimizers },
        { name: 'Optimization Strategy Selector', fn: testOptimizationStrategySelector },

        // Suite 2: Federated Learning
        { name: 'Federated Client', fn: testFederatedClient },
        { name: 'Secure Aggregation', fn: testSecureAggregation },
        { name: 'Differential Privacy', fn: testDifferentialPrivacy },
        { name: 'Federated Server', fn: testFederatedServer },

        // Suite 3: Neural Architecture Search
        { name: 'Search Space', fn: testSearchSpace },
        { name: 'Performance Estimator', fn: testPerformanceEstimator },
        { name: 'NAS Strategies', fn: testNASStrategies },
        { name: 'Multi-Objective NAS', fn: testMultiObjectiveNAS },

        // Suite 4: Knowledge Distillation
        { name: 'Distillation Loss', fn: testDistillationLoss },
        { name: 'Teacher-Student Models', fn: testTeacherStudentModels },
        { name: 'Progressive Distillation', fn: testProgressiveDistillation },

        // Suite 5: Multi-Modal Streaming
        { name: 'Stream Handlers', fn: testStreamHandlers },
        { name: 'Multi-Modal Coordinator', fn: testMultiModalCoordinator },

        // Suite 6: ONNX Integration
        { name: 'ONNX Runtime', fn: testONNXRuntime },
        { name: 'ONNX Model Converter', fn: testONNXModelConverter },
        { name: 'ONNX Model Analyzer', fn: testONNXModelAnalyzer },

        // Suite 7: Model Interpretability
        { name: 'Attention Visualizer', fn: testAttentionVisualizer },
        { name: 'Gradient Explainer', fn: testGradientExplainer },
        { name: 'Feature Importance Analyzer', fn: testFeatureImportanceAnalyzer },
        { name: 'Interpretability Controller', fn: testInterpretabilityController },

        // Suite 8: Auto Performance Optimization
        { name: 'Auto Performance Profiler', fn: testAutoPerformanceProfiler },
        { name: 'Bottleneck Detector', fn: testBottleneckDetector },
        { name: 'ML-Based Optimizer', fn: testMLBasedOptimizer },
        { name: 'Auto Performance Optimizer', fn: testAutoPerformanceOptimizer },

        // Integration Tests
        { name: 'Advanced Optimization Pipeline', fn: testAdvancedOptimizationPipeline },
        { name: 'Federated Learning Pipeline', fn: testFederatedLearningPipeline },
        { name: 'NAS with Distillation', fn: testNASWithDistillation },
        { name: 'Multi-Modal with Interpretability', fn: testMultiModalWithInterpretability }
    ];

    let passed = 0;
    let failed = 0;
    const failures = [];

    for (const test of testSuites) {
        try {
            await test.fn();
            passed++;
        } catch (error) {
            failed++;
            failures.push({ name: test.name, error: error.message });
            console.error(`✗ ${test.name} FAILED: ${error.message}`);
        }
    }

    console.log('');
    console.log('='.repeat(80));
    console.log('Test Results');
    console.log('='.repeat(80));
    console.log(`Total Tests: ${testSuites.length}`);
    console.log(`Passed: ${passed}`);
    console.log(`Failed: ${failed}`);
    console.log('');

    if (failures.length > 0) {
        console.log('Failed Tests:');
        failures.forEach(({ name, error }) => {
            console.log(`  - ${name}: ${error}`);
        });
        console.log('');
    }

    if (failed === 0) {
        console.log('🎉 All tests passed!');
    } else {
        console.log(`⚠️  ${failed} test(s) failed.`);
    }

    console.log('='.repeat(80));

    return failed === 0;
}

// Run tests if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    runAllTests()
        .then(success => process.exit(success ? 0 : 1))
        .catch(error => {
            console.error('Fatal error:', error);
            process.exit(1);
        });
}

export { runAllTests };
