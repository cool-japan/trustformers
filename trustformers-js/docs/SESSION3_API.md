# Session 3 Features API Documentation

This document provides comprehensive API documentation for TrustformeRS-JS Session 3 features (2025-11-10).

## Table of Contents

1. [ENAS NAS Algorithm](#1-enas-nas-algorithm)
2. [Enhanced Federated Learning](#2-enhanced-federated-learning)
3. [ONNX Operators](#3-onnx-operators)
4. [Real-time Collaboration](#4-real-time-collaboration)
5. [Integration Examples](#5-integration-examples)

---

## 1. ENAS NAS Algorithm

**Efficient Neural Architecture Search** using reinforcement learning with parameter sharing. ENAS is 1000x faster than traditional NAS methods by sharing parameters across child architectures.

### 1.1 Quick Start

```javascript
import { createENASSearcher, ENASSearchSpaces } from 'trustformers-js';

// Create searcher with compact search space
const searcher = createENASSearcher(ENASSearchSpaces.compact, {
    controllerEpochs: 50,
    childEpochs: 300,
    controllerLearningRate: 0.00035,
    entropyWeight: 0.0001
});

// Prepare data
const trainData = [
    { input: new Float32Array([...]), target: 0 },
    // ... more samples
];
const validData = [...];

// Run search
const results = await searcher.search(trainData, validData);

console.log('Best architecture:', results.bestArchitecture);
console.log('Best reward:', results.bestReward);
```

### 1.2 Search Spaces

ENAS provides three predefined search spaces:

```javascript
import { ENASSearchSpaces } from 'trustformers-js';

// Compact search space (quick experiments)
const compactSpace = ENASSearchSpaces.compact;
// {
//   numLayers: 4,
//   operations: ['conv3x3', 'maxpool', 'identity', 'zero'],
//   inputDim: 32,
//   outputDim: 10
// }

// CNN search space
const cnnSpace = ENASSearchSpaces.cnn;

// Transformer search space
const transformerSpace = ENASSearchSpaces.transformer;
```

### 1.3 Custom Search Space

```javascript
const customSpace = {
    numLayers: 8,
    operations: [
        'conv3x3',
        'conv5x5',
        'separable_conv3x3',
        'maxpool3x3',
        'avgpool3x3',
        'identity',
        'zero'
    ],
    inputDim: 64,
    outputDim: 20
};

const searcher = createENASSearcher(customSpace, {
    controllerEpochs: 100,
    childEpochs: 500,
    numSamples: 20
});
```

### 1.4 Configuration Options

```typescript
interface ENASSearcherConfig {
    // Number of controller training epochs (default: 50)
    controllerEpochs?: number;

    // Number of child model training epochs (default: 300)
    childEpochs?: number;

    // Learning rate for controller (default: 0.00035)
    controllerLearningRate?: number;

    // Learning rate for child model (default: 0.1)
    childLearningRate?: number;

    // Entropy weight for exploration (default: 0.0001)
    entropyWeight?: number;

    // Baseline for reward normalization (default: null)
    baseline?: number | null;

    // Number of architectures to sample per controller step (default: 10)
    numSamples?: number;

    // Early stopping patience (default: 10)
    patience?: number;
}
```

### 1.5 Advanced Usage

#### Controller Operations

```javascript
import { ENASController } from 'trustformers-js';

const controller = new ENASController({
    numLayers: 6,
    numOperations: 7,
    hiddenSize: 256,
    temperature: 5.0,
    entropyWeight: 0.0001
});

// Sample architecture
const architecture = controller.sampleArchitecture();
// {
//   layers: [
//     { operation: 2, input: 0 },
//     { operation: 0, input: 1 },
//     { operation: 5, input: 0 },
//     ...
//   ]
// }

// Get log probabilities
const logProbs = controller.getLogProbabilities(architecture);

// Update controller with reward
controller.update(architecture, reward, baseline);
```

#### Shared Model Training

```javascript
import { ENASSharedModel } from 'trustformers-js';

const sharedModel = new ENASSharedModel({
    inputDim: 32,
    outputDim: 10,
    numNodes: 6,
    operations: ['conv3x3', 'conv5x5', 'maxpool', 'identity']
});

// Forward pass
const output = sharedModel.forward(input, architecture);

// Training step
const loss = sharedModel.train(batch, architecture);
```

### 1.6 Performance Tips

1. **Use smaller search spaces** for quick experiments
2. **Increase `numSamples`** for better exploration
3. **Tune `entropyWeight`** to balance exploration vs exploitation
4. **Use early stopping** (`patience` parameter) to save compute

---

## 2. Enhanced Federated Learning

Advanced federated learning with **FedBN** (preserves local batch normalization) and **FedNova** (normalized averaging for heterogeneous clients).

### 2.1 FedBN (Federated Batch Normalization)

**Key Innovation**: Keeps batch normalization statistics local to handle non-IID data better.

```javascript
import { FedBNAggregator } from 'trustformers-js';

const aggregator = new FedBNAggregator({
    bnParamNames: ['running_mean', 'running_var', 'num_batches_tracked']
});

// Client updates
const clientUpdates = [
    {
        modelUpdate: {
            layer1: {
                weights: new Float32Array([0.1, 0.2, 0.3]),
                bias: new Float32Array([0.01, 0.02]),
                running_mean: new Float32Array([0.5, 0.6]),  // Local BN stats
                running_var: new Float32Array([0.1, 0.1])
            }
        },
        numSamples: 100,
        clientId: 'client_1'
    },
    // ... more clients
];

// Aggregate while preserving local BN stats
const result = aggregator.aggregate(clientUpdates, {
    preserveBNStats: true,  // FedBN core feature!
    weightingScheme: 'dataSize'
});

console.log('Global model:', result.globalModel);
// BN parameters are NOT aggregated (kept local)
```

#### Weighting Schemes

```javascript
// Uniform weighting (all clients equal)
aggregator.aggregate(updates, { weightingScheme: 'uniform' });

// Data size weighting (proportional to number of samples)
aggregator.aggregate(updates, { weightingScheme: 'dataSize' });

// Inverse gradient weighting
aggregator.aggregate(updates, { weightingScheme: 'inverseGradient' });

// Custom weights
aggregator.aggregate(updates, {
    weightingScheme: 'custom',
    customWeights: [0.3, 0.5, 0.2]  // Must sum to 1.0
});
```

### 2.2 FedNova (Federated Normalized Averaging)

**Key Innovation**: Normalized averaging to handle heterogeneous local training steps.

```javascript
import { FedNovaAggregator } from 'trustformers-js';

const aggregator = new FedNovaAggregator({
    rho: 0.9,  // Momentum parameter
    tauEffStrategy: 'gradient'  // or 'model'
});

// Clients with DIFFERENT numbers of local steps
const clientUpdates = [
    {
        modelUpdate: { weights: new Float32Array([...]), bias: new Float32Array([...]) },
        numSamples: 100,
        numLocalSteps: 5,  // Different local steps
        clientId: 'client_1'
    },
    {
        modelUpdate: { weights: new Float32Array([...]), bias: new Float32Array([...]) },
        numSamples: 150,
        numLocalSteps: 10,  // Different local steps
        clientId: 'client_2'
    }
];

// Aggregate with normalized averaging
const result = aggregator.aggregate(clientUpdates, {
    globalLearningRate: 1.0,
    useMomentum: true,
    normalizationScheme: 'gradient'  // or 'model'
});

console.log('Effective tau:', result.metadata.tau);
console.log('Global model:', result.globalModel);
console.log('Momentum buffer:', result.momentumBuffer);
```

### 2.3 Enhanced Federated Server

Supports multiple aggregation strategies (FedAvg, FedBN, FedNova).

```javascript
import { EnhancedFederatedServer } from 'trustformers-js';

const server = new EnhancedFederatedServer({
    model: initialModel,
    aggregationStrategy: 'fedbn',  // 'fedavg', 'fedbn', or 'fednova'
    minClients: 10,
    clientsPerRound: 5
});

// Register clients
server.registerClient({ clientId: 'client_1', capabilities: {} });
server.registerClient({ clientId: 'client_2', capabilities: {} });

// Select clients for training round
const selectedClients = server.selectClients();

// Aggregate updates
const globalModel = server.aggregate(clientUpdates);

// Switch aggregation strategy dynamically
server.setAggregationStrategy('fednova');
```

### 2.4 When to Use Which Algorithm

| Algorithm | Best For | Key Benefit |
|-----------|----------|-------------|
| **FedAvg** | IID data, homogeneous clients | Simple, baseline |
| **FedBN** | Non-IID data | Better handling of distribution shift |
| **FedNova** | Heterogeneous local steps | Corrects objective inconsistency |

### 2.5 Performance Metrics

```javascript
// FedBN typically shows:
// - 10-20% accuracy improvement on non-IID data
// - Better convergence stability
// - No additional communication cost

// FedNova typically shows:
// - Better convergence with heterogeneous clients
// - Reduced variance in final model performance
// - Faster convergence (fewer rounds)
```

---

## 3. ONNX Operators

20+ standard ONNX operators for cross-framework compatibility and model interoperability.

### 3.1 Operator Registry

```javascript
import { createOperatorRegistry } from 'trustformers-js';

const registry = createOperatorRegistry();

// Check supported operators
const operators = registry.getSupportedOperators();
console.log('Supported:', operators);
// ['Add', 'Sub', 'Mul', 'Div', 'MatMul', 'Gemm', 'Relu', 'Gelu', ...]

// Check if operator is supported
if (registry.isSupported('MatMul')) {
    const op = registry.create('MatMul');
}
```

### 3.2 Basic Math Operators

```javascript
import { createOperatorRegistry, Tensor } from 'trustformers-js';

const registry = createOperatorRegistry();

// Create tensors
const a = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
const b = new Tensor(new Float32Array([5, 6, 7, 8]), [2, 2]);

// Add
const addOp = registry.create('Add');
const [sum] = addOp.execute([a, b]);
// sum.data = [6, 8, 10, 12]

// Subtract
const subOp = registry.create('Sub');
const [diff] = subOp.execute([a, b]);
// diff.data = [-4, -4, -4, -4]

// Multiply (element-wise)
const mulOp = registry.create('Mul');
const [product] = mulOp.execute([a, b]);
// product.data = [5, 12, 21, 32]

// Divide
const divOp = registry.create('Div');
const [quotient] = divOp.execute([a, b]);
// quotient.data = [0.2, 0.333, 0.428, 0.5]
```

### 3.3 Matrix Operations

```javascript
// Matrix Multiplication
const matmulOp = registry.create('MatMul');
const A = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const B = new Tensor(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]);
const [C] = matmulOp.execute([A, B]);
// C.shape = [2, 2]
// C.data = [58, 64, 139, 154]

// GEMM (General Matrix Multiply): Y = alpha * A * B + beta * C
const gemmOp = registry.create('Gemm', { alpha: 2.0, beta: 1.0, transA: false, transB: false });
const [Y] = gemmOp.execute([A, B, C]);

// Transpose
const transposeOp = registry.create('Transpose', { perm: [1, 0] });
const [At] = transposeOp.execute([A]);
// At.shape = [3, 2]
```

### 3.4 Activation Functions

```javascript
const input = new Tensor(new Float32Array([-2, -1, 0, 1, 2]), [5]);

// ReLU
const reluOp = registry.create('Relu');
const [reluOut] = reluOp.execute([input]);
// reluOut.data = [0, 0, 0, 1, 2]

// GELU
const geluOp = registry.create('Gelu');
const [geluOut] = geluOp.execute([input]);

// Sigmoid
const sigmoidOp = registry.create('Sigmoid');
const [sigmoidOut] = sigmoidOp.execute([input]);
// sigmoidOut.data = [0.119, 0.269, 0.5, 0.731, 0.881]

// Tanh
const tanhOp = registry.create('Tanh');
const [tanhOut] = tanhOp.execute([input]);

// Softmax
const softmaxOp = registry.create('Softmax', { axis: -1 });
const logits = new Tensor(new Float32Array([1, 2, 3, 4]), [4]);
const [probs] = softmaxOp.execute([logits]);
// probs.data = [0.032, 0.087, 0.237, 0.644] (sums to 1.0)

// Swish (SiLU)
const swishOp = registry.create('Swish');
const [swishOut] = swishOp.execute([input]);
```

### 3.5 Normalization

```javascript
// Batch Normalization
const bnOp = registry.create('BatchNormalization', {
    epsilon: 1e-5,
    momentum: 0.9
});

const input = new Tensor(new Float32Array([...]), [batch, channels, height, width]);
const scale = new Tensor(new Float32Array([...]), [channels]);
const bias = new Tensor(new Float32Array([...]), [channels]);
const runningMean = new Tensor(new Float32Array([...]), [channels]);
const runningVar = new Tensor(new Float32Array([...]), [channels]);

const [normalized] = bnOp.execute([input, scale, bias, runningMean, runningVar]);

// Layer Normalization
const lnOp = registry.create('LayerNormalization', {
    epsilon: 1e-5,
    axis: -1
});

const input = new Tensor(new Float32Array([...]), [batch, seqLen, hiddenSize]);
const [lnOut] = lnOp.execute([input]);
```

### 3.6 Shape Operations

```javascript
// Reshape
const reshapeOp = registry.create('Reshape');
const input = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const newShape = new Tensor(new Int32Array([3, 2]), [2]);
const [reshaped] = reshapeOp.execute([input, newShape]);
// reshaped.shape = [3, 2]

// Concat
const concatOp = registry.create('Concat', { axis: 0 });
const a = new Tensor(new Float32Array([1, 2, 3]), [1, 3]);
const b = new Tensor(new Float32Array([4, 5, 6]), [1, 3]);
const [concatenated] = concatOp.execute([a, b]);
// concatenated.shape = [2, 3]

// Slice
const sliceOp = registry.create('Slice', {
    starts: [0, 1],
    ends: [2, 3],
    axes: [0, 1]
});
const [sliced] = sliceOp.execute([input]);
```

### 3.7 Reduction Operations

```javascript
const input = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);

// ReduceSum
const sumOp = registry.create('ReduceSum', { axes: [1], keepdims: false });
const [sum] = sumOp.execute([input]);
// sum.data = [6, 15] (sum across dimension 1)

// ReduceMean
const meanOp = registry.create('ReduceMean', { axes: [1], keepdims: false });
const [mean] = meanOp.execute([input]);
// mean.data = [2, 5]

// ReduceMax
const maxOp = registry.create('ReduceMax', { axes: [1], keepdims: true });
const [max] = maxOp.execute([input]);
// max.data = [3, 6], max.shape = [2, 1]
```

### 3.8 Custom Operators

```javascript
// Register custom operator
class CustomOperator {
    constructor(attributes = {}) {
        this.name = 'Custom';
        this.attributes = attributes;
    }

    execute(inputs) {
        // Custom logic
        const [input] = inputs;
        const output = new Tensor(
            input.data.map(x => x * 2),
            input.shape
        );
        return [output];
    }
}

registry.register('Custom', CustomOperator);

// Use custom operator
const customOp = registry.create('Custom', { param: 1.0 });
const [result] = customOp.execute([input]);
```

---

## 4. Real-time Collaboration

WebSocket-based collaborative features for distributed ML experiments.

### 4.1 Collaborative Session

```javascript
import { createCollaborativeSession } from 'trustformers-js';

const session = createCollaborativeSession({
    serverUrl: 'ws://your-server.com:8080',
    userId: 'researcher_1',
    userName: 'Alice',
    autoReconnect: true,
    reconnectInterval: 5000
});

// Event listeners
session.on('connected', () => {
    console.log('Connected to collaboration server');
});

session.on('peerJoined', (peer) => {
    console.log(`${peer.userName} joined`);
});

session.on('peerLeft', (peer) => {
    console.log(`${peer.userName} left`);
});

session.on('modelUpdated', ({ model, updatedBy }) => {
    console.log('Model updated by:', updatedBy);
    // Apply model update
});

// Connect
await session.connect();

// Share model update
await session.shareModelUpdate({
    layer1: { weights: new Float32Array([...]) }
}, {
    description: 'Improved layer 1'
});

// Update presence
session.updatePresence('active');

// Disconnect
session.disconnect();
```

### 4.2 Collaborative Experiments

**Bayesian Hyperparameter Optimization** with real-time result sharing.

```javascript
import { createCollaborativeExperiment } from 'trustformers-js';

const experiment = createCollaborativeExperiment({
    name: 'Learning Rate Tuning',
    searchSpace: {
        learningRate: {
            type: 'continuous',
            min: 1e-5,
            max: 1e-2,
            logScale: true
        },
        batchSize: {
            type: 'integer',
            min: 16,
            max: 128,
            step: 16
        },
        optimizer: {
            type: 'categorical',
            choices: ['adam', 'sgd', 'adamw', 'lamb']
        },
        warmupSteps: {
            type: 'integer',
            min: 0,
            max: 1000,
            step: 100
        }
    },
    metric: 'accuracy',
    goal: 'maximize',  // or 'minimize'
    session: session
});

// Submit results
experiment.submitResult(
    { learningRate: 0.001, batchSize: 32, optimizer: 'adam', warmupSteps: 500 },
    { accuracy: 0.92, loss: 0.15, f1: 0.91 },
    'researcher_1'
);

// Get best result
const best = experiment.getBestResult();
console.log('Best config:', best.config);
console.log('Best metrics:', best.metrics);

// Get next configuration suggestion (Bayesian optimization)
const nextConfig = experiment.suggestConfiguration();
console.log('Try next:', nextConfig);

// Get all results
const allResults = experiment.getResults();
```

### 4.3 Metrics Dashboard

Real-time metrics broadcasting and visualization.

```javascript
import { createMetricsDashboard } from 'trustformers-js';

const dashboard = createMetricsDashboard({
    session: session,
    updateInterval: 1000,  // Update every 1 second
    metricsToTrack: ['loss', 'accuracy', 'f1_score', 'latency']
});

// Start dashboard
dashboard.start();

// Update metrics
dashboard.updateMetric('loss', 0.234);
dashboard.updateMetric('accuracy', 0.921);
dashboard.updateMetric('f1_score', 0.905);
dashboard.updateMetric('latency', 45.2);  // ms

// Get current metrics
const current = dashboard.getCurrentMetrics();
console.log(current);
// { loss: 0.234, accuracy: 0.921, f1_score: 0.905, latency: 45.2 }

// Get metrics history
const lossHistory = dashboard.getMetricsHistory('loss', 100);  // Last 100 entries
lossHistory.forEach(entry => {
    console.log(`${new Date(entry.timestamp).toISOString()}: ${entry.value}`);
});

// Subscribe to metric updates
dashboard.subscribeToMetric('accuracy', (value) => {
    console.log('New accuracy:', value);
    if (value > 0.95) {
        console.log('🎉 Target accuracy reached!');
    }
});

// Stop dashboard
dashboard.stop();
```

### 4.4 Operational Transformation

For conflict-free collaborative editing.

```javascript
// Operations are automatically transformed when concurrent edits occur
// The system uses Operational Transformation (OT) to ensure consistency

session.on('conflict', ({ operation, resolved }) => {
    console.log('Conflict resolved:', resolved);
});
```

### 4.5 Presence Awareness

```javascript
// Update your presence
session.updatePresence('active');   // 'active', 'idle', 'away', 'offline'

// Listen for presence updates
session.on('peerStatusChanged', ({ userId, status }) => {
    console.log(`${userId} is now ${status}`);
});

// Get all peers
const peers = session.peers;
peers.forEach((peer, userId) => {
    console.log(`${peer.userName} (${userId}): ${peer.status}`);
});
```

---

## 5. Integration Examples

### 5.1 Distributed NAS with Federated Learning

```javascript
import {
    createENASSearcher,
    ENASSearchSpaces,
    FedNovaAggregator,
    createCollaborativeSession,
    createCollaborativeExperiment
} from 'trustformers-js';

// 1. Setup collaboration
const session = createCollaborativeSession({
    serverUrl: 'ws://nas-server.com:8080',
    userId: 'researcher_1',
    userName: 'Researcher 1'
});

await session.connect();

// 2. Create collaborative NAS experiment
const nasExperiment = createCollaborativeExperiment({
    name: 'Distributed ENAS Search',
    searchSpace: {
        numLayers: { type: 'integer', min: 4, max: 12 },
        hiddenSize: { type: 'integer', min: 128, max: 512, step: 64 }
    },
    metric: 'reward',
    goal: 'maximize',
    session: session
});

// 3. Run ENAS search locally
const searcher = createENASSearcher(ENASSearchSpaces.transformer, {
    controllerEpochs: 50,
    childEpochs: 300
});

const results = await searcher.search(trainData, validData);

// 4. Share results with collaborators
nasExperiment.submitResult(
    { architecture: results.bestArchitecture },
    { reward: results.bestReward },
    'researcher_1'
);

// 5. Aggregate architectures from multiple researchers
const fedNova = new FedNovaAggregator({ rho: 0.9 });

const researcherUpdates = [
    {
        modelUpdate: researcher1Architecture,
        numSamples: 100,
        numLocalSteps: 50,
        clientId: 'researcher_1'
    },
    {
        modelUpdate: researcher2Architecture,
        numSamples: 150,
        numLocalSteps: 75,
        clientId: 'researcher_2'
    }
];

const globalArchitecture = fedNova.aggregate(researcherUpdates);
```

### 5.2 ONNX Model Validation in Federated Setting

```javascript
import {
    createOperatorRegistry,
    Tensor,
    FedBNAggregator
} from 'trustformers-js';

// Use ONNX operators for model validation
const registry = createOperatorRegistry();

const validateModel = (modelUpdate) => {
    // Convert to ONNX tensors
    const weights = new Tensor(modelUpdate.weights, modelUpdate.shape);

    // Apply ONNX operators for validation
    const softmaxOp = registry.create('Softmax', { axis: -1 });
    const [normalized] = softmaxOp.execute([weights]);

    // Compute validation metric
    const reluOp = registry.create('Relu');
    const [activated] = reluOp.execute([normalized]);

    return activated;
};

// FedBN aggregation with ONNX validation
const fedBN = new FedBNAggregator();

const clientUpdates = clients.map(client => ({
    modelUpdate: validateModel(client.model),
    numSamples: client.dataSize,
    clientId: client.id
}));

const aggregated = fedBN.aggregate(clientUpdates);
```

### 5.3 Real-time Collaborative NAS Dashboard

```javascript
import {
    createCollaborativeSession,
    createMetricsDashboard,
    createENASSearcher
} from 'trustformers-js';

// Setup collaboration
const session = createCollaborativeSession({
    serverUrl: 'ws://dashboard.com:8080',
    userId: 'researcher_1',
    userName: 'Researcher 1'
});

await session.connect();

// Create dashboard
const dashboard = createMetricsDashboard({
    session: session,
    metricsToTrack: ['reward', 'architectures_searched', 'parameters', 'latency']
});

dashboard.start();

// Run ENAS with real-time updates
const searcher = createENASSearcher(ENASSearchSpaces.compact);

searcher.on('architectureEvaluated', ({ architecture, reward, parameters, latency }) => {
    // Update dashboard in real-time
    dashboard.updateMetric('reward', reward);
    dashboard.updateMetric('architectures_searched', searcher.history.length);
    dashboard.updateMetric('parameters', parameters);
    dashboard.updateMetric('latency', latency);

    // Broadcast to collaborators
    session.shareModelUpdate({
        architecture,
        reward,
        timestamp: Date.now()
    });
});

const results = await searcher.search(trainData, validData);
```

---

## Best Practices

### ENAS NAS
1. Start with compact search space for prototyping
2. Use early stopping to save compute
3. Tune entropy weight based on search space size
4. Monitor controller's entropy to ensure exploration

### Federated Learning
1. Use **FedBN** for non-IID data
2. Use **FedNova** when clients have different computational capabilities
3. Monitor effective tau in FedNova (should be stable)
4. Use data-size weighting for heterogeneous client datasets

### ONNX Operators
1. Use operator registry for consistency
2. Validate tensor shapes before operations
3. Use appropriate activation functions for your task
4. Leverage ONNX for cross-framework compatibility

### Real-time Collaboration
1. Handle connection failures gracefully (auto-reconnect)
2. Use Bayesian optimization for efficient hyperparameter search
3. Monitor presence to know who's active
4. Subscribe to specific metrics for automated alerts

---

## Performance Considerations

### ENAS
- **Search time**: O(C * E * S), where C = controller epochs, E = child epochs, S = samples
- **Memory**: Shared parameters reduce memory by 1000x vs independent training
- **Recommendation**: Use GPU if available, start with small search spaces

### Federated Learning
- **Communication cost**: FedBN = FedAvg (same), FedNova = FedAvg + tau
- **Computation**: FedBN ≈ FedAvg, FedNova slightly higher (normalization)
- **Recommendation**: Batch updates to reduce network overhead

### ONNX Operators
- **Overhead**: Minimal (<5% vs native operations)
- **Broadcasting**: Automatic, but be aware of memory implications
- **Recommendation**: Use WebGPU backend for large tensors

### Collaboration
- **Latency**: WebSocket ~10-50ms per message
- **Bandwidth**: Depends on model size, use compression if needed
- **Recommendation**: Update metrics in batches, not per-iteration

---

## Troubleshooting

### Common Issues

**ENAS not converging**
- Increase controller learning rate
- Reduce entropy weight
- Use more samples per controller step

**FedNova tau unstable**
- Check that numLocalSteps is correctly reported
- Try 'model' normalization instead of 'gradient'
- Reduce momentum (rho)

**ONNX shape mismatch**
- Validate input shapes before operations
- Check broadcasting compatibility
- Use reshape/transpose to align dimensions

**Collaboration connection fails**
- Check WebSocket server is running
- Verify firewall settings
- Enable auto-reconnect

---

## API Reference

For complete TypeScript definitions, see `types/session3.d.ts`.

For additional examples, see `examples/session3-integration-demo.html`.

For tests, see `test/session3-features.test.js`.
