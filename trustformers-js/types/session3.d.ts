/**
 * TrustformeRS-JS Session 3 Features Type Definitions
 *
 * Comprehensive TypeScript definitions for:
 * 1. ENAS NAS Algorithm
 * 2. Enhanced Federated Learning (FedBN & FedNova)
 * 3. ONNX Operators (20+ operators)
 * 4. Real-time Collaboration
 */

// ============================================================================
// 1. ENAS NAS Algorithm
// ============================================================================

/**
 * ENAS operation types available in the search space
 */
export type ENASOperationType =
    | 'conv3x3'
    | 'conv5x5'
    | 'separable_conv3x3'
    | 'separable_conv5x5'
    | 'maxpool3x3'
    | 'avgpool3x3'
    | 'identity'
    | 'zero';

/**
 * ENAS search space configuration
 */
export interface ENASSearchSpace {
    /** Number of layers in the search space */
    numLayers: number;
    /** Available operations */
    operations: ENASOperationType[];
    /** Input dimension */
    inputDim?: number;
    /** Output dimension */
    outputDim?: number;
}

/**
 * Architecture layer specification
 */
export interface ENASLayer {
    /** Operation index (corresponding to operation in search space) */
    operation: number;
    /** Input node index */
    input?: number;
}

/**
 * Complete architecture specification
 */
export interface ENASArchitecture {
    /** Layers in the architecture */
    layers: ENASLayer[];
    /** Optional metadata */
    metadata?: {
        searchTime?: number;
        reward?: number;
        parameters?: number;
    };
}

/**
 * ENAS controller configuration
 */
export interface ENASControllerConfig {
    /** Number of layers to search */
    numLayers: number;
    /** Number of operations */
    numOperations: number;
    /** Hidden size of controller RNN */
    hiddenSize?: number;
    /** Temperature for softmax */
    temperature?: number;
    /** Tanh constant */
    tanhConstant?: number;
    /** Entropy weight for exploration */
    entropyWeight?: number;
}

/**
 * ENAS searcher configuration
 */
export interface ENASSearcherConfig {
    /** Number of controller training epochs */
    controllerEpochs?: number;
    /** Number of child model training epochs */
    childEpochs?: number;
    /** Learning rate for controller */
    controllerLearningRate?: number;
    /** Learning rate for child model */
    childLearningRate?: number;
    /** Entropy weight for exploration */
    entropyWeight?: number;
    /** Baseline for reward normalization */
    baseline?: number | null;
    /** Number of architectures to sample per controller step */
    numSamples?: number;
    /** Early stopping patience */
    patience?: number;
}

/**
 * ENAS search results
 */
export interface ENASSearchResults {
    /** Best architecture found */
    bestArchitecture: ENASArchitecture;
    /** Best reward achieved */
    bestReward: number;
    /** Search history */
    history: Array<{
        architecture: ENASArchitecture;
        reward: number;
        epoch: number;
    }>;
    /** Total search time in milliseconds */
    searchTime: number;
}

/**
 * ENAS Operations manager
 */
export declare class ENASOperations {
    constructor();

    /** Get available operation types */
    getOperationTypes(): ENASOperationType[];

    /** Execute operation on input */
    execute(operationType: ENASOperationType, input: Float32Array): Float32Array;
}

/**
 * ENAS Controller (RNN-based architecture sampler)
 */
export declare class ENASController {
    constructor(config: ENASControllerConfig);

    /** Sample architecture from controller */
    sampleArchitecture(): ENASArchitecture;

    /** Get log probabilities for architecture */
    getLogProbabilities(architecture: ENASArchitecture): number[];

    /** Update controller with reward */
    update(architecture: ENASArchitecture, reward: number, baseline?: number): void;

    /** Get controller parameters */
    getParameters(): Record<string, Float32Array>;

    /** Set controller parameters */
    setParameters(params: Record<string, Float32Array>): void;
}

/**
 * ENAS Shared Model (efficient parameter sharing)
 */
export declare class ENASSharedModel {
    constructor(config: {
        inputDim: number;
        outputDim: number;
        numNodes: number;
        operations: ENASOperationType[];
    });

    /** Forward pass with given architecture */
    forward(input: Float32Array, architecture: ENASArchitecture): Float32Array;

    /** Compute loss */
    computeLoss(output: Float32Array, target: Float32Array): number;

    /** Train on batch */
    train(batch: Array<{ input: Float32Array; target: Float32Array }>, architecture: ENASArchitecture): number;
}

/**
 * ENAS Searcher (main interface)
 */
export declare class ENASSearcher {
    constructor(searchSpace: ENASSearchSpace, config?: ENASSearcherConfig);

    /** Run architecture search */
    search(
        trainData: Array<{ input: Float32Array; target: number }>,
        validData: Array<{ input: Float32Array; target: number }>
    ): Promise<ENASSearchResults>;

    /** Export best architecture */
    exportArchitecture(): ENASArchitecture;
}

/**
 * Predefined ENAS search spaces
 */
export declare const ENASSearchSpaces: {
    /** Compact search space for quick experiments */
    compact: ENASSearchSpace;
    /** CNN search space */
    cnn: ENASSearchSpace;
    /** Transformer search space */
    transformer: ENASSearchSpace;
};

/**
 * Create ENAS searcher
 */
export declare function createENASSearcher(
    searchSpace: ENASSearchSpace,
    config?: ENASSearcherConfig
): ENASSearcher;

// ============================================================================
// 2. Enhanced Federated Learning (FedBN & FedNova)
// ============================================================================

/**
 * Client update for federated learning
 */
export interface ClientUpdate {
    /** Model update (gradients or model diff) */
    modelUpdate: Record<string, {
        weights?: Float32Array;
        bias?: Float32Array;
        running_mean?: Float32Array;
        running_var?: Float32Array;
        [key: string]: Float32Array | undefined;
    }>;
    /** Number of samples used for training */
    numSamples: number;
    /** Client identifier */
    clientId: string;
    /** Number of local training steps (for FedNova) */
    numLocalSteps?: number;
}

/**
 * Aggregation result
 */
export interface AggregationResult {
    /** Aggregated global model */
    globalModel: Record<string, {
        weights?: Float32Array;
        bias?: Float32Array;
        running_mean?: Float32Array;
        running_var?: Float32Array;
        [key: string]: Float32Array | undefined;
    }>;
    /** Metadata about aggregation */
    metadata: {
        /** Total samples aggregated */
        totalSamples?: number;
        /** Effective tau (FedNova) */
        tau?: number;
        /** Whether momentum was used */
        momentumUsed?: boolean;
        /** Local BN stats preserved (FedBN) */
        localBNStats?: boolean;
        [key: string]: any;
    };
    /** Momentum buffer (if momentum is used) */
    momentumBuffer?: Record<string, {
        weights?: Float32Array;
        bias?: Float32Array;
        [key: string]: Float32Array | undefined;
    }>;
}

/**
 * Weighting scheme for aggregation
 */
export type WeightingScheme = 'uniform' | 'dataSize' | 'inverseGradient' | 'custom';

/**
 * FedBN (Federated Batch Normalization) Aggregator
 *
 * Key innovation: Preserves local batch normalization statistics
 * to handle non-IID data better.
 */
export declare class FedBNAggregator {
    constructor(config?: {
        /** Names of batch normalization parameters to keep local */
        bnParamNames?: string[];
    });

    /** Aggregate client updates */
    aggregate(
        clientUpdates: ClientUpdate[],
        options?: {
            /** Whether to preserve BN statistics (FedBN core feature) */
            preserveBNStats?: boolean;
            /** Weighting scheme for aggregation */
            weightingScheme?: WeightingScheme;
            /** Custom weights (if using 'custom' scheme) */
            customWeights?: number[];
        }
    ): AggregationResult;
}

/**
 * FedNova (Federated Normalized Averaging) Aggregator
 *
 * Key innovation: Normalized averaging to handle heterogeneous
 * local training steps across clients.
 */
export declare class FedNovaAggregator {
    constructor(config?: {
        /** Momentum parameter (rho) */
        rho?: number;
        /** Effective tau computation strategy */
        tauEffStrategy?: 'gradient' | 'model';
    });

    /** Aggregate client updates */
    aggregate(
        clientUpdates: ClientUpdate[],
        options?: {
            /** Global learning rate */
            globalLearningRate?: number;
            /** Whether to use server-side momentum */
            useMomentum?: boolean;
            /** Normalization scheme */
            normalizationScheme?: 'gradient' | 'model';
        }
    ): AggregationResult;
}

/**
 * Enhanced Federated Server
 *
 * Supports multiple aggregation strategies (FedAvg, FedBN, FedNova)
 */
export declare class EnhancedFederatedServer {
    /** Registered clients */
    clients: Map<string, {
        clientId: string;
        capabilities: Record<string, any>;
    }>;

    /** Current aggregation strategy */
    aggregationStrategy: 'fedavg' | 'fedbn' | 'fednova';

    constructor(config: {
        /** Initial model */
        model: any;
        /** Aggregation strategy */
        aggregationStrategy?: 'fedavg' | 'fedbn' | 'fednova';
        /** Minimum number of clients required */
        minClients?: number;
        /** Number of clients to select per round */
        clientsPerRound?: number;
    });

    /** Register a client */
    registerClient(client: { clientId: string; capabilities: Record<string, any> }): void;

    /** Select clients for training round */
    selectClients(): string[];

    /** Aggregate client updates */
    aggregate(clientUpdates: ClientUpdate[]): AggregationResult;

    /** Switch aggregation strategy */
    setAggregationStrategy(strategy: 'fedavg' | 'fedbn' | 'fednova'): void;
}

/**
 * Create enhanced federated learning system
 */
export declare function createEnhancedFederatedLearning(config: {
    model: any;
    aggregationStrategy?: 'fedavg' | 'fedbn' | 'fednova';
    minClients?: number;
    clientsPerRound?: number;
}): EnhancedFederatedServer;

// ============================================================================
// 3. ONNX Operators
// ============================================================================

/**
 * ONNX Tensor (compatible with ONNX specification)
 */
export declare class Tensor {
    /** Tensor data */
    data: Float32Array;
    /** Tensor shape */
    shape: number[];

    constructor(data: Float32Array, shape: number[]);

    /** Get total number of elements */
    size(): number;

    /** Reshape tensor */
    reshape(newShape: number[]): Tensor;

    /** Get element at index */
    get(...indices: number[]): number;

    /** Set element at index */
    set(value: number, ...indices: number[]): void;
}

/**
 * ONNX Operator attributes
 */
export interface OperatorAttributes {
    /** Axis for reduction/softmax operations */
    axis?: number;
    /** Axes for reduction operations */
    axes?: number[];
    /** Keep dimensions after reduction */
    keepdims?: boolean;
    /** Permutation for transpose */
    perm?: number[];
    /** Alpha parameter (for operators like LeakyRelu) */
    alpha?: number;
    /** Beta parameter */
    beta?: number;
    /** Epsilon for normalization */
    epsilon?: number;
    /** Momentum for batch normalization */
    momentum?: number;
    [key: string]: any;
}

/**
 * Base ONNX Operator interface
 */
export interface ONNXOperator {
    /** Operator name */
    name: string;
    /** Execute operator */
    execute(inputs: Tensor[], attributes?: OperatorAttributes): Tensor[];
}

/**
 * ONNX Operator Registry
 */
export declare class ONNXOperatorRegistry {
    constructor();

    /** Get supported operators */
    getSupportedOperators(): string[];

    /** Check if operator is supported */
    isSupported(operatorName: string): boolean;

    /** Create operator instance */
    create(operatorName: string, attributes?: OperatorAttributes): ONNXOperator;

    /** Register custom operator */
    register(operatorName: string, operatorClass: new (attributes?: OperatorAttributes) => ONNXOperator): void;
}

// Math operators
export declare class Add implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Sub implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Mul implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Div implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class MatMul implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Gemm implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

// Activation functions
export declare class Relu implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Gelu implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Sigmoid implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Tanh implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Softmax implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Swish implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

// Normalization
export declare class BatchNormalization implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class LayerNormalization implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

// Shape manipulation
export declare class Reshape implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Transpose implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Concat implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class Slice implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

// Reduction operations
export declare class ReduceSum implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class ReduceMean implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

export declare class ReduceMax implements ONNXOperator {
    name: string;
    constructor(attributes?: OperatorAttributes);
    execute(inputs: Tensor[]): Tensor[];
}

/**
 * Create ONNX operator registry
 */
export declare function createOperatorRegistry(): ONNXOperatorRegistry;

// ============================================================================
// 4. Real-time Collaboration
// ============================================================================

/**
 * Presence status for collaborative users
 */
export type PresenceStatus = 'active' | 'idle' | 'away' | 'offline';

/**
 * Peer information
 */
export interface Peer {
    /** User ID */
    userId: string;
    /** User name */
    userName: string;
    /** Presence status */
    status?: PresenceStatus;
    /** Capabilities */
    capabilities?: Record<string, any>;
    /** Last seen timestamp */
    lastSeen?: number;
}

/**
 * Collaborative session event types
 */
export type CollaborativeEvent =
    | 'connected'
    | 'disconnected'
    | 'peerJoined'
    | 'peerLeft'
    | 'peerStatusChanged'
    | 'modelUpdated'
    | 'experimentUpdated'
    | 'metricsUpdated'
    | 'error';

/**
 * Event listener callback
 */
export type EventListener<T = any> = (data: T) => void;

/**
 * Collaborative Session
 */
export declare class CollaborativeSession {
    /** User ID */
    userId: string;
    /** User name */
    userName: string;
    /** Connected peers */
    peers: Map<string, Peer>;

    constructor(config: {
        /** WebSocket server URL */
        serverUrl: string;
        /** User ID */
        userId: string;
        /** User name */
        userName: string;
        /** Auto-reconnect on disconnect */
        autoReconnect?: boolean;
        /** Reconnect interval in ms */
        reconnectInterval?: number;
    });

    /** Connect to collaboration server */
    connect(): Promise<void>;

    /** Disconnect from server */
    disconnect(): void;

    /** Check if connected */
    isConnected(): boolean;

    /** Register event listener */
    on<T = any>(event: CollaborativeEvent, listener: EventListener<T>): void;

    /** Remove event listener */
    off<T = any>(event: CollaborativeEvent, listener: EventListener<T>): void;

    /** Emit event (for testing) */
    emit<T = any>(event: CollaborativeEvent, data: T): void;

    /** Share model update */
    shareModelUpdate(modelUpdate: any, metadata?: Record<string, any>): Promise<void>;

    /** Update presence status */
    updatePresence(status: PresenceStatus): void;
}

/**
 * Search space parameter type
 */
export type SearchSpaceParamType = 'continuous' | 'integer' | 'categorical';

/**
 * Search space parameter definition
 */
export interface SearchSpaceParam {
    /** Parameter type */
    type: SearchSpaceParamType;
    /** Minimum value (for continuous/integer) */
    min?: number;
    /** Maximum value (for continuous/integer) */
    max?: number;
    /** Step size (for integer) */
    step?: number;
    /** Use log scale (for continuous) */
    logScale?: boolean;
    /** Choices (for categorical) */
    choices?: any[];
}

/**
 * Experiment result
 */
export interface ExperimentResult {
    /** Configuration used */
    config: Record<string, any>;
    /** Metrics achieved */
    metrics: Record<string, number>;
    /** User who submitted */
    userId: string;
    /** Timestamp */
    timestamp: number;
}

/**
 * Collaborative Experiment (Bayesian Optimization)
 */
export declare class CollaborativeExperiment {
    /** Experiment name */
    name: string;
    /** Search space definition */
    searchSpace: Record<string, SearchSpaceParam>;
    /** Optimization metric name */
    metric: string;
    /** Optimization goal */
    goal: 'maximize' | 'minimize';
    /** Experiment results */
    results: ExperimentResult[];

    constructor(config: {
        /** Experiment name */
        name: string;
        /** Search space */
        searchSpace: Record<string, SearchSpaceParam>;
        /** Metric to optimize */
        metric: string;
        /** Optimization goal */
        goal: 'maximize' | 'minimize';
        /** Collaborative session */
        session: CollaborativeSession;
    });

    /** Submit experiment result */
    submitResult(
        config: Record<string, any>,
        metrics: Record<string, number>,
        userId: string
    ): void;

    /** Get best result */
    getBestResult(): ExperimentResult | null;

    /** Suggest next configuration (Bayesian optimization) */
    suggestConfiguration(): Record<string, any>;

    /** Get all results */
    getResults(): ExperimentResult[];

    /** Clear all results */
    clearResults(): void;
}

/**
 * Collaborative Metrics Dashboard
 */
export declare class CollaborativeMetricsDashboard {
    /** Metrics to track */
    metricsToTrack: string[];

    constructor(config: {
        /** Collaborative session */
        session: CollaborativeSession;
        /** Update interval in ms */
        updateInterval?: number;
        /** Metrics to track */
        metricsToTrack?: string[];
    });

    /** Start dashboard */
    start(): void;

    /** Stop dashboard */
    stop(): void;

    /** Update metric value */
    updateMetric(metricName: string, value: number): void;

    /** Get current metrics */
    getCurrentMetrics(): Record<string, number>;

    /** Get metrics history */
    getMetricsHistory(metricName: string, limit?: number): Array<{
        value: number;
        timestamp: number;
    }>;

    /** Subscribe to metric updates */
    subscribeToMetric(metricName: string, callback: (value: number) => void): void;
}

/**
 * Create collaborative session
 */
export declare function createCollaborativeSession(config: {
    serverUrl: string;
    userId: string;
    userName: string;
    autoReconnect?: boolean;
    reconnectInterval?: number;
}): CollaborativeSession;

/**
 * Create collaborative experiment
 */
export declare function createCollaborativeExperiment(config: {
    name: string;
    searchSpace: Record<string, SearchSpaceParam>;
    metric: string;
    goal: 'maximize' | 'minimize';
    session: CollaborativeSession;
}): CollaborativeExperiment;

/**
 * Create metrics dashboard
 */
export declare function createMetricsDashboard(config: {
    session: CollaborativeSession;
    updateInterval?: number;
    metricsToTrack?: string[];
}): CollaborativeMetricsDashboard;
