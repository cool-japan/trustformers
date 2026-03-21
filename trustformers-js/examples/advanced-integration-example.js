/**
 * Advanced AI Features Integration Example
 *
 * This example demonstrates how to use multiple advanced AI features together
 * to build a complete end-to-end machine learning pipeline with:
 * - Neural Architecture Search to find optimal model architecture
 * - Knowledge Distillation to create an efficient model
 * - Federated Learning for privacy-preserving training
 * - Advanced Optimization techniques for efficient training
 * - Model Interpretability for understanding decisions
 * - Auto Performance Optimization for continuous improvement
 */

import {
    // NAS
    createNAS,
    SearchSpace,
    PerformanceEstimator,

    // Distillation
    createDistillation,
    TeacherModel,
    StudentModel,

    // Federated Learning
    createFederatedLearning,
    FederatedClient,
    FederatedServer,
    DifferentialPrivacyMechanism,

    // Advanced Optimization
    createAdvancedOptimizer,
    OptimizationStrategySelector,
    GradientCheckpointingManager,
    MixedPrecisionManager,

    // Interpretability
    createInterpretability,
    AttentionVisualizer,
    GradientExplainer,

    // Auto Optimization
    createAutoOptimizer,

    // Multi-Modal
    createMultiModalStreaming,
    TextStreamHandler,
    ImageStreamHandler,

    // ONNX
    createONNXIntegration
} from '../src/index.js';

/**
 * Complete ML Pipeline Example
 */
class AdvancedMLPipeline {
    constructor(config = {}) {
        this.config = config;
        this.components = {};
        this.statistics = {
            nasIterations: 0,
            distillationEpochs: 0,
            federatedRounds: 0,
            optimizationSteps: 0
        };
    }

    /**
     * Step 1: Use NAS to find optimal architecture
     */
    async discoverArchitecture(searchConfig = {}) {
        console.log('Step 1: Neural Architecture Search...');

        const nasController = createNAS({
            searchStrategy: 'evolutionary',
            populationSize: searchConfig.populationSize || 20,
            numGenerations: searchConfig.numGenerations || 5
        });

        const searchResult = await nasController.search({
            maxLayers: searchConfig.maxLayers || 12,
            minLayers: searchConfig.minLayers || 3,
            targetMetrics: ['accuracy', 'latency', 'parameters']
        });

        this.statistics.nasIterations = searchResult.history?.length || 0;
        this.components.architecture = searchResult.bestArchitecture;

        console.log(`✓ Found architecture with ${searchResult.bestArchitecture.layers.length} layers`);
        console.log(`  Estimated accuracy: ${searchResult.bestMetrics.accuracy?.toFixed(4) || 'N/A'}`);
        console.log(`  Estimated latency: ${searchResult.bestMetrics.latency?.toFixed(2) || 'N/A'}ms`);

        return searchResult;
    }

    /**
     * Step 2: Create teacher model from discovered architecture
     */
    async createTeacherModel(architecture) {
        console.log('\nStep 2: Creating Teacher Model...');

        // Mock teacher model creation (in practice, would train on full dataset)
        const teacherModel = {
            type: 'transformer',
            architecture: architecture,
            layers: architecture.layers,
            parameters: new Float32Array(100000), // Simulated parameters
            forward: async (input) => ({
                logits: new Float32Array(input.length).map(() => Math.random()),
                hidden: input
            })
        };

        this.components.teacher = new TeacherModel(teacherModel, {
            temperature: 3.0
        });

        console.log(`✓ Teacher model created with ${architecture.layers.length} layers`);

        return this.components.teacher;
    }

    /**
     * Step 3: Knowledge Distillation to create efficient student model
     */
    async distillKnowledge(dataset, studentConfig = {}) {
        console.log('\nStep 3: Knowledge Distillation...');

        const distillation = createDistillation({
            temperature: studentConfig.temperature || 3.0,
            alpha: studentConfig.alpha || 0.7
        });

        const result = await distillation.distill(
            this.components.teacher,
            {
                numLayers: studentConfig.numLayers || 6,
                hiddenSize: studentConfig.hiddenSize || 512
            },
            dataset,
            {
                epochs: studentConfig.epochs || 5,
                batchSize: studentConfig.batchSize || 32
            }
        );

        this.statistics.distillationEpochs = studentConfig.epochs || 5;
        this.components.studentModel = result.studentModel;

        console.log(`✓ Student model created with ${studentConfig.numLayers || 6} layers`);
        console.log(`  Compression ratio: ${(this.components.teacher.model.layers.length / (studentConfig.numLayers || 6)).toFixed(2)}x`);

        return result;
    }

    /**
     * Step 4: Set up federated learning for privacy-preserving training
     */
    async setupFederatedLearning(numClients = 10) {
        console.log('\nStep 4: Setting Up Federated Learning...');

        const flSystem = createFederatedLearning({
            numClients,
            clientsPerRound: Math.min(5, numClients),
            numRounds: 3,
            differentialPrivacy: true,
            secureAggregation: true
        });

        // Initialize global model with student model
        flSystem.server.initializeGlobalModel(this.components.studentModel);

        // Create and register clients
        this.components.federatedClients = [];
        for (let i = 0; i < numClients; i++) {
            const client = new FederatedClient(`client_${i}`);
            await client.initializeModel(this.components.studentModel);

            // In practice, each client would have their own private data
            const clientData = this.generateMockData(50);
            client.setLocalData(clientData);

            this.components.federatedClients.push(client);
            flSystem.server.registerClient(`client_${i}`, { dataSize: 50 });
        }

        this.components.federatedServer = flSystem.server;

        console.log(`✓ Federated learning setup with ${numClients} clients`);
        console.log(`  Privacy: Differential Privacy + Secure Aggregation enabled`);

        return flSystem;
    }

    /**
     * Step 5: Train with advanced optimization techniques
     */
    async trainWithAdvancedOptimization(numRounds = 3) {
        console.log('\nStep 5: Federated Training with Advanced Optimization...');

        // Select optimization strategy
        const strategySelector = new OptimizationStrategySelector();
        const strategy = strategySelector.selectStrategy({
            modelSize: 50e6, // 50M parameters
            batchSize: 32,
            sequenceLength: 512
        });

        console.log(`  Selected strategy: ${strategy.name}`);

        // Create advanced optimizer
        const baseOptimizer = {
            step: (weights, gradients) => weights.map((w, i) => w - 0.001 * gradients[i])
        };

        const advancedOpt = createAdvancedOptimizer(baseOptimizer, {
            modelSize: 50e6,
            batchSize: 32
        });

        // Setup auto performance optimizer
        const autoOpt = createAutoOptimizer({
            enabled: true,
            adaptiveLearning: true,
            optimizationInterval: 1000
        });

        autoOpt.startOptimization(this.components.studentModel, {
            batchSize: 32,
            learningRate: 0.001,
            workers: 4
        });

        // Run federated training rounds
        for (let round = 0; round < numRounds; round++) {
            console.log(`\n  Round ${round + 1}/${numRounds}:`);

            // Select clients
            const selectedClients = this.components.federatedServer.selectClients(5);
            console.log(`    - Selected ${selectedClients.length} clients`);

            // Clients train locally
            const clientUpdates = [];
            for (const clientId of selectedClients) {
                const client = this.components.federatedClients.find(c => c.clientId === clientId);
                const update = await client.trainLocal({
                    epochs: 1,
                    batchSize: 16
                });
                clientUpdates.push({ clientId, ...update });
            }

            // Server aggregates updates
            const aggregated = await this.components.federatedServer.aggregateUpdates(clientUpdates);
            console.log(`    - Aggregated updates from ${clientUpdates.length} clients`);

            // Record metrics for auto-optimization
            autoOpt.recordMetrics({
                throughput: 100 + Math.random() * 20,
                latency: 15 + Math.random() * 5,
                memory: 700 + Math.random() * 100
            });

            this.statistics.federatedRounds++;
            this.statistics.optimizationSteps += selectedClients.length;
        }

        const optimizationReport = autoOpt.getOptimizationReport();
        autoOpt.stopOptimization();

        console.log(`\n✓ Training completed:`);
        console.log(`  - Federated rounds: ${this.statistics.federatedRounds}`);
        console.log(`  - Total optimization steps: ${this.statistics.optimizationSteps}`);
        console.log(`  - Bottlenecks detected: ${optimizationReport.bottlenecks?.length || 0}`);

        return optimizationReport;
    }

    /**
     * Step 6: Analyze model interpretability
     */
    async analyzeInterpretability(sampleInput) {
        console.log('\nStep 6: Model Interpretability Analysis...');

        const interpretability = createInterpretability(this.components.studentModel, {
            methods: ['attention', 'gradients', 'feature_importance']
        });

        const report = await interpretability.generateReport(
            sampleInput,
            this.generateMockData(20),
            {
                attention: { heads: [0, 1, 2] },
                gradients: { method: 'integrated_gradients', steps: 50 },
                featureImportance: { method: 'permutation', numPermutations: 10 }
            }
        );

        console.log(`✓ Interpretability analysis completed:`);
        console.log(`  - Attention patterns analyzed for ${report.explanations?.attention?.heads?.length || 0} heads`);
        console.log(`  - Gradient attributions computed`);
        console.log(`  - Feature importance rankings generated`);

        return report;
    }

    /**
     * Step 7: Export to ONNX for deployment
     */
    async exportToONNX() {
        console.log('\nStep 7: ONNX Export for Deployment...');

        const onnx = createONNXIntegration({
            opsetVersion: 13,
            optimization: true
        });

        try {
            const exportResult = await onnx.convert(this.components.studentModel, {
                optimize: true,
                quantize: false
            });

            console.log(`✓ Model exported to ONNX format`);
            console.log(`  - Optimization level: all`);
            console.log(`  - Ready for cross-platform deployment`);

            return exportResult;
        } catch (error) {
            console.log(`  Note: ONNX export requires additional dependencies (${error.message})`);
            return null;
        }
    }

    /**
     * Step 8: Multi-modal inference pipeline
     */
    async setupMultiModalInference() {
        console.log('\nStep 8: Multi-Modal Inference Pipeline...');

        const mmStreaming = createMultiModalStreaming({
            modalities: ['text', 'image', 'audio'],
            synchronization: 'timestamp',
            bufferSize: 10
        });

        console.log(`✓ Multi-modal pipeline initialized:`);
        console.log(`  - Modalities: text, image, audio`);
        console.log(`  - Synchronization: timestamp-based`);
        console.log(`  - Real-time processing enabled`);

        return mmStreaming;
    }

    /**
     * Helper: Generate mock dataset
     */
    generateMockData(size) {
        return Array.from({ length: size }, (_, i) => ({
            input: new Float32Array(768).fill(Math.random()),
            label: Math.floor(Math.random() * 2)
        }));
    }

    /**
     * Generate comprehensive pipeline report
     */
    generateReport() {
        return {
            pipeline: 'Advanced AI Features Integration',
            timestamp: new Date().toISOString(),
            statistics: this.statistics,
            components: {
                architecture: this.components.architecture ?
                    `${this.components.architecture.layers.length} layers` : 'N/A',
                teacher: this.components.teacher ? 'Created' : 'N/A',
                student: this.components.studentModel ? 'Distilled' : 'N/A',
                federatedClients: this.components.federatedClients?.length || 0,
                optimizationEnabled: true,
                interpretabilityEnabled: true
            },
            features: [
                'Neural Architecture Search',
                'Knowledge Distillation',
                'Federated Learning',
                'Advanced Optimization',
                'Model Interpretability',
                'Auto Performance Optimization',
                'Multi-Modal Streaming',
                'ONNX Integration'
            ]
        };
    }
}

/**
 * Run the complete pipeline
 */
async function runCompletePipeline() {
    console.log('='.repeat(80));
    console.log('Advanced AI Features - Complete Integration Pipeline');
    console.log('='.repeat(80));
    console.log('');

    const pipeline = new AdvancedMLPipeline({
        name: 'text-classification-pipeline',
        task: 'sentiment-analysis'
    });

    try {
        // Step 1: Architecture Discovery
        const nasResult = await pipeline.discoverArchitecture({
            populationSize: 10,
            numGenerations: 3,
            maxLayers: 10,
            minLayers: 4
        });

        // Step 2: Create Teacher Model
        await pipeline.createTeacherModel(nasResult.bestArchitecture);

        // Step 3: Knowledge Distillation
        const trainingData = pipeline.generateMockData(200);
        await pipeline.distillKnowledge(trainingData, {
            numLayers: 6,
            hiddenSize: 512,
            epochs: 3,
            batchSize: 32
        });

        // Step 4: Federated Learning Setup
        await pipeline.setupFederatedLearning(10);

        // Step 5: Advanced Optimization Training
        await pipeline.trainWithAdvancedOptimization(3);

        // Step 6: Interpretability Analysis
        const sampleInput = new Float32Array(768).fill(0.5);
        await pipeline.analyzeInterpretability(sampleInput);

        // Step 7: ONNX Export
        await pipeline.exportToONNX();

        // Step 8: Multi-Modal Setup
        await pipeline.setupMultiModalInference();

        // Generate final report
        const report = pipeline.generateReport();

        console.log('\n' + '='.repeat(80));
        console.log('Pipeline Completed Successfully!');
        console.log('='.repeat(80));
        console.log('\nFinal Report:');
        console.log(JSON.stringify(report, null, 2));

        return report;

    } catch (error) {
        console.error('\n❌ Pipeline failed:', error.message);
        console.error(error.stack);
        throw error;
    }
}

/**
 * Example: Specific use cases
 */

// Use Case 1: Privacy-Preserving Healthcare AI
async function healthcareAIExample() {
    console.log('\n\n' + '='.repeat(80));
    console.log('Use Case 1: Privacy-Preserving Healthcare AI');
    console.log('='.repeat(80));

    const pipeline = new AdvancedMLPipeline({ task: 'medical-diagnosis' });

    // Use federated learning for patient privacy
    const flSystem = await pipeline.setupFederatedLearning(5); // 5 hospitals

    console.log('\n✓ Healthcare AI pipeline configured:');
    console.log('  - Each hospital keeps data locally');
    console.log('  - Differential privacy protects individual patients');
    console.log('  - Secure aggregation prevents data leakage');
    console.log('  - Model interpretability for clinical decisions');

    return pipeline;
}

// Use Case 2: Edge Device Deployment
async function edgeDeploymentExample() {
    console.log('\n\n' + '='.repeat(80));
    console.log('Use Case 2: Edge Device Deployment');
    console.log('='.repeat(80));

    const pipeline = new AdvancedMLPipeline({ task: 'mobile-inference' });

    // Use NAS to find efficient architecture
    await pipeline.discoverArchitecture({
        maxLayers: 6, // Constraint for mobile
        targetMetrics: ['latency', 'parameters'] // Prioritize efficiency
    });

    // Use distillation for model compression
    const mockTeacher = {
        model: { layers: Array(12).fill({}) },
        forward: async () => ({})
    };
    pipeline.components.teacher = mockTeacher;

    await pipeline.distillKnowledge(pipeline.generateMockData(100), {
        numLayers: 4, // Very small for edge
        epochs: 5
    });

    console.log('\n✓ Edge deployment pipeline configured:');
    console.log('  - Efficient architecture discovered');
    console.log('  - Model compressed via distillation');
    console.log('  - Ready for ONNX export');
    console.log('  - Optimized for mobile/edge devices');

    return pipeline;
}

// Use Case 3: Multi-Modal Content Moderation
async function contentModerationExample() {
    console.log('\n\n' + '='.repeat(80));
    console.log('Use Case 3: Multi-Modal Content Moderation');
    console.log('='.repeat(80));

    const mmStreaming = createMultiModalStreaming({
        modalities: ['text', 'image', 'audio'],
        synchronization: 'timestamp'
    });

    // Process different content types
    await mmStreaming.coordinator.addChunk({
        modality: 'text',
        data: { text: 'Sample comment' },
        timestamp: 1000
    });

    await mmStreaming.coordinator.addChunk({
        modality: 'image',
        data: { data: new Uint8Array(100) },
        timestamp: 1001
    });

    console.log('\n✓ Content moderation pipeline configured:');
    console.log('  - Multi-modal analysis (text + image + audio)');
    console.log('  - Real-time processing');
    console.log('  - Synchronized modality handling');
    console.log('  - Interpretable decisions');

    return mmStreaming;
}

// Export for use in other modules
export {
    AdvancedMLPipeline,
    runCompletePipeline,
    healthcareAIExample,
    edgeDeploymentExample,
    contentModerationExample
};

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    runCompletePipeline()
        .then(() => {
            console.log('\n\nRunning specific use cases...\n');
            return Promise.all([
                healthcareAIExample(),
                edgeDeploymentExample(),
                contentModerationExample()
            ]);
        })
        .then(() => {
            console.log('\n\n✨ All examples completed successfully!');
            process.exit(0);
        })
        .catch(error => {
            console.error('\n\n❌ Error:', error);
            process.exit(1);
        });
}
