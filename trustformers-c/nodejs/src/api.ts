/**
 * High-level API wrapper for TrustformeRS C bindings
 */

import ref from 'ref-napi';
import { EventEmitter } from 'events';
import {
  getLibrary,
  ensureInitialized,
  checkError,
  allocCString,
  readCString,
  freeCString,
  TrustformersMemoryUsageStruct,
  TrustformersAdvancedMemoryUsageStruct,
  TrustformersBuildInfoStruct,
  TrustformersPerformanceMetricsStruct,
  TrustformersOptimizationConfigStruct
} from './ffi';
import {
  TrustformersError,
  ModelConfig,
  TokenizerConfig,
  PipelineConfig,
  GenerationConfig,
  GenerationResult,
  ClassificationResult,
  QuestionAnsweringResult,
  TokenizationResult,
  EncodeOptions,
  DecodeOptions,
  ConversationState,
  ConversationTurn,
  TrustformersMemoryUsage,
  TrustformersAdvancedMemoryUsage,
  TrustformersBuildInfo,
  TrustformersPerformanceMetrics,
  TrustformersOptimizationConfig,
  StreamingCallback,
  ModelHandle,
  TokenizerHandle,
  PipelineHandle,
  TrustformersNativeError,
  LibraryCapabilities,
  PlatformInfo
} from './types';

/**
 * Main TrustformeRS API class
 */
export class TrustformeRS extends EventEmitter {
  private static _instance: TrustformeRS | null = null;
  
  private constructor() {
    super();
    ensureInitialized();
  }
  
  /**
   * Get the singleton instance of TrustformeRS
   */
  public static getInstance(): TrustformeRS {
    if (!TrustformeRS._instance) {
      TrustformeRS._instance = new TrustformeRS();
    }
    return TrustformeRS._instance;
  }
  
  /**
   * Get version information
   */
  public getVersion(): string {
    const lib = getLibrary();
    const versionPtr = lib.trustformers_version();
    return readCString(versionPtr);
  }
  
  /**
   * Get build information
   */
  public getBuildInfo(): TrustformersBuildInfo {
    const lib = getLibrary();
    const buildInfo = new TrustformersBuildInfoStruct();
    const result = lib.trustformers_build_info(buildInfo.ref());
    checkError(result, 'get build info');
    
    return {
      version: readCString(buildInfo.version),
      features: readCString(buildInfo.features),
      buildDate: readCString(buildInfo.build_date),
      target: readCString(buildInfo.target)
    };
  }
  
  /**
   * Check if a feature is available
   */
  public hasFeature(feature: string): boolean {
    const lib = getLibrary();
    const featureCStr = allocCString(feature);
    const result = lib.trustformers_has_feature(featureCStr);
    return result === 1;
  }
  
  /**
   * Set log level (0=off, 1=error, 2=warn, 3=info, 4=debug, 5=trace)
   */
  public setLogLevel(level: number): void {
    const lib = getLibrary();
    const result = lib.trustformers_set_log_level(level);
    checkError(result, 'set log level');
  }
  
  /**
   * Get current memory usage
   */
  public getMemoryUsage(): TrustformersMemoryUsage {
    const lib = getLibrary();
    const usage = new TrustformersMemoryUsageStruct();
    const result = lib.trustformers_get_memory_usage(usage.ref());
    checkError(result, 'get memory usage');
    
    return {
      totalMemoryBytes: Number(usage.total_memory_bytes),
      peakMemoryBytes: Number(usage.peak_memory_bytes),
      allocatedModels: Number(usage.allocated_models),
      allocatedTokenizers: Number(usage.allocated_tokenizers),
      allocatedPipelines: Number(usage.allocated_pipelines),
      allocatedTensors: Number(usage.allocated_tensors)
    };
  }
  
  /**
   * Get advanced memory usage statistics
   */
  public getAdvancedMemoryUsage(): TrustformersAdvancedMemoryUsage {
    const lib = getLibrary();
    const usage = new TrustformersAdvancedMemoryUsageStruct();
    const result = lib.trustformers_get_advanced_memory_usage(usage.ref());
    checkError(result, 'get advanced memory usage');
    
    const typeUsageJson = readCString(usage.type_usage_json);
    const typeUsage = typeUsageJson ? JSON.parse(typeUsageJson) : {};
    
    const advancedUsage: TrustformersAdvancedMemoryUsage = {
      basic: {
        totalMemoryBytes: Number(usage.basic.total_memory_bytes),
        peakMemoryBytes: Number(usage.basic.peak_memory_bytes),
        allocatedModels: Number(usage.basic.allocated_models),
        allocatedTokenizers: Number(usage.basic.allocated_tokenizers),
        allocatedPipelines: Number(usage.basic.allocated_pipelines),
        allocatedTensors: Number(usage.basic.allocated_tensors)
      },
      fragmentationRatio: usage.fragmentation_ratio,
      avgAllocationSize: Number(usage.avg_allocation_size),
      typeUsageJson: typeUsageJson,
      pressureLevel: usage.pressure_level,
      allocationRate: usage.allocation_rate,
      deallocationRate: usage.deallocation_rate
    };
    
    // Free the allocated string
    lib.trustformers_advanced_memory_usage_free(usage.ref());
    
    return advancedUsage;
  }
  
  /**
   * Perform memory cleanup
   */
  public memoryCleanup(): void {
    const lib = getLibrary();
    const result = lib.trustformers_memory_cleanup();
    checkError(result, 'memory cleanup');
  }
  
  /**
   * Set memory limits
   */
  public setMemoryLimits(maxMemoryMb: number, warningThresholdMb: number): void {
    const lib = getLibrary();
    const result = lib.trustformers_set_memory_limits(maxMemoryMb, warningThresholdMb);
    checkError(result, 'set memory limits');
  }
  
  /**
   * Check for memory leaks
   */
  public checkMemoryLeaks(): object {
    const lib = getLibrary();
    const reportPtr = ref.alloc(ref.refType(ref.types.char));
    const result = lib.trustformers_check_memory_leaks(reportPtr);
    checkError(result, 'check memory leaks');
    
    const reportStr = readCString(reportPtr.deref());
    freeCString(reportPtr.deref());
    
    return JSON.parse(reportStr);
  }
  
  /**
   * Get performance metrics
   */
  public getPerformanceMetrics(): TrustformersPerformanceMetrics {
    const lib = getLibrary();
    const metrics = new TrustformersPerformanceMetricsStruct();
    const result = lib.trustformers_get_performance_metrics(metrics.ref());
    checkError(result, 'get performance metrics');
    
    const hintsJson = readCString(metrics.optimization_hints_json);
    const hints = hintsJson ? JSON.parse(hintsJson) : [];
    
    const performanceMetrics: TrustformersPerformanceMetrics = {
      totalOperations: Number(metrics.total_operations),
      avgOperationTimeMs: metrics.avg_operation_time_ms,
      minOperationTimeMs: metrics.min_operation_time_ms,
      maxOperationTimeMs: metrics.max_operation_time_ms,
      cacheHitRate: metrics.cache_hit_rate,
      performanceScore: metrics.performance_score,
      numOptimizationHints: metrics.num_optimization_hints,
      optimizationHintsJson: hintsJson
    };
    
    // Free the allocated string
    lib.trustformers_performance_metrics_free(metrics.ref());
    
    return performanceMetrics;
  }
  
  /**
   * Apply performance optimizations
   */
  public applyOptimizations(config: TrustformersOptimizationConfig): void {
    const lib = getLibrary();
    const configStruct = new TrustformersOptimizationConfigStruct();
    
    configStruct.enable_tracking = config.enableTracking ? 1 : 0;
    configStruct.enable_caching = config.enableCaching ? 1 : 0;
    configStruct.cache_size_mb = config.cacheSizeMb;
    configStruct.num_threads = config.numThreads;
    configStruct.enable_simd = config.enableSimd ? 1 : 0;
    configStruct.optimize_batch_size = config.optimizeBatchSize ? 1 : 0;
    configStruct.memory_optimization_level = config.memoryOptimizationLevel;
    
    const result = lib.trustformers_apply_optimizations(configStruct.ref());
    checkError(result, 'apply optimizations');
  }
  
  /**
   * Start performance profiling
   */
  public startProfiling(): void {
    const lib = getLibrary();
    const result = lib.trustformers_start_profiling();
    checkError(result, 'start profiling');
  }
  
  /**
   * Stop performance profiling and get report
   */
  public stopProfiling(): object {
    const lib = getLibrary();
    const reportPtr = ref.alloc(ref.refType(ref.types.char));
    const result = lib.trustformers_stop_profiling(reportPtr);
    checkError(result, 'stop profiling');
    
    const reportStr = readCString(reportPtr.deref());
    freeCString(reportPtr.deref());
    
    return JSON.parse(reportStr);
  }
  
  /**
   * Force garbage collection
   */
  public gc(): void {
    const lib = getLibrary();
    const result = lib.trustformers_gc();
    checkError(result, 'garbage collection');
  }
  
  /**
   * Get library capabilities
   */
  public getCapabilities(): LibraryCapabilities {
    const buildInfo = this.getBuildInfo();
    const features = buildInfo.features.split(',');
    
    return {
      version: buildInfo.version,
      buildInfo,
      supportedFeatures: features,
      supportedDevices: this.getSupportedDevices(),
      supportedFormats: ['onnx', 'safetensors', 'pytorch'],
      maxBatchSize: 256,
      maxSequenceLength: 4096
    };
  }
  
  /**
   * Get platform information
   */
  public getPlatformInfo(): PlatformInfo {
    const os = require('os');
    
    return {
      arch: os.arch(),
      platform: os.platform(),
      features: this.getBuildInfo().features.split(','),
      hasGpu: this.hasFeature('gpu'),
      hasCuda: this.hasFeature('cuda'),
      hasRocm: this.hasFeature('rocm'),
      numCores: os.cpus().length,
      totalMemory: os.totalmem()
    };
  }
  
  /**
   * Create a new model instance
   */
  public createModel(config: ModelConfig | string): Model {
    return new Model(config);
  }
  
  /**
   * Create a new tokenizer instance
   */
  public createTokenizer(config: TokenizerConfig | string): Tokenizer {
    return new Tokenizer(config);
  }
  
  /**
   * Create a new pipeline instance
   */
  public createPipeline(config: PipelineConfig): Pipeline {
    return new Pipeline(config);
  }
  
  private getSupportedDevices(): string[] {
    const devices = ['cpu'];
    if (this.hasFeature('cuda')) devices.push('cuda');
    if (this.hasFeature('rocm')) devices.push('rocm');
    return devices;
  }
}

/**
 * Model class for loading and managing models
 */
export class Model {
  private _handle: ModelHandle;
  private _disposed = false;
  
  constructor(config: ModelConfig | string) {
    ensureInitialized();
    
    const lib = getLibrary();
    const handlePtr = ref.alloc(ref.types.int);
    
    if (typeof config === 'string') {
      const modelPathCStr = allocCString(config);
      const result = lib.trustformers_model_load(modelPathCStr, handlePtr);
      checkError(result, 'model load');
    } else {
      const modelPathCStr = allocCString(config.modelPath);
      const configJson = JSON.stringify(config);
      const configCStr = allocCString(configJson);
      const result = lib.trustformers_model_load_with_config(modelPathCStr, configCStr, handlePtr);
      checkError(result, 'model load with config');
    }
    
    this._handle = handlePtr.deref();
  }
  
  /**
   * Get model handle
   */
  public get handle(): ModelHandle {
    this.checkDisposed();
    return this._handle;
  }
  
  /**
   * Get model metadata
   */
  public getMetadata(): object {
    this.checkDisposed();
    
    const lib = getLibrary();
    const metadataPtr = ref.alloc(ref.refType(ref.types.char));
    const result = lib.trustformers_model_get_metadata(this._handle, metadataPtr);
    checkError(result, 'get model metadata');
    
    const metadataStr = readCString(metadataPtr.deref());
    freeCString(metadataPtr.deref());
    
    return JSON.parse(metadataStr);
  }
  
  /**
   * Get model configuration
   */
  public getConfig(): object {
    this.checkDisposed();
    
    const lib = getLibrary();
    const configPtr = ref.alloc(ref.refType(ref.types.char));
    const result = lib.trustformers_model_get_config(this._handle, configPtr);
    checkError(result, 'get model config');
    
    const configStr = readCString(configPtr.deref());
    freeCString(configPtr.deref());
    
    return JSON.parse(configStr);
  }
  
  /**
   * Quantize the model
   */
  public quantize(quantizationType: string, outputPath: string): void {
    this.checkDisposed();
    
    const lib = getLibrary();
    const quantizationCStr = allocCString(quantizationType);
    const outputCStr = allocCString(outputPath);
    const result = lib.trustformers_model_quantize(this._handle, quantizationCStr, outputCStr);
    checkError(result, 'model quantization');
  }
  
  /**
   * Validate the model
   */
  public validate(): object {
    this.checkDisposed();
    
    const lib = getLibrary();
    const reportPtr = ref.alloc(ref.refType(ref.types.char));
    const result = lib.trustformers_model_validate(this._handle, reportPtr);
    checkError(result, 'model validation');
    
    const reportStr = readCString(reportPtr.deref());
    freeCString(reportPtr.deref());
    
    return JSON.parse(reportStr);
  }
  
  /**
   * Dispose of the model and free resources
   */
  public dispose(): void {
    if (!this._disposed) {
      const lib = getLibrary();
      const result = lib.trustformers_model_free(this._handle);
      checkError(result, 'model free');
      this._disposed = true;
    }
  }
  
  private checkDisposed(): void {
    if (this._disposed) {
      throw new Error('Model has been disposed');
    }
  }
}

/**
 * Tokenizer class for text tokenization
 */
export class Tokenizer {
  private _handle: TokenizerHandle;
  private _disposed = false;
  
  constructor(config: TokenizerConfig | string) {
    ensureInitialized();
    
    const lib = getLibrary();
    const handlePtr = ref.alloc(ref.types.int);
    
    if (typeof config === 'string') {
      const tokenizerPathCStr = allocCString(config);
      const result = lib.trustformers_tokenizer_load(tokenizerPathCStr, handlePtr);
      checkError(result, 'tokenizer load');
    } else {
      const tokenizerPathCStr = allocCString(config.tokenizerPath);
      const configJson = JSON.stringify(config);
      const configCStr = allocCString(configJson);
      const result = lib.trustformers_tokenizer_load_with_config(tokenizerPathCStr, configCStr, handlePtr);
      checkError(result, 'tokenizer load with config');
    }
    
    this._handle = handlePtr.deref();
  }
  
  /**
   * Get tokenizer handle
   */
  public get handle(): TokenizerHandle {
    this.checkDisposed();
    return this._handle;
  }
  
  /**
   * Encode text to tokens
   */
  public encode(text: string, options?: EncodeOptions): TokenizationResult {
    this.checkDisposed();
    
    const lib = getLibrary();
    const textCStr = allocCString(text);
    const optionsCStr = allocCString(JSON.stringify(options || {}));
    const resultPtr = ref.alloc(ref.refType(ref.types.char));
    
    const result = lib.trustformers_tokenizer_encode(this._handle, textCStr, optionsCStr, resultPtr);
    checkError(result, 'tokenizer encode');
    
    const resultStr = readCString(resultPtr.deref());
    freeCString(resultPtr.deref());
    
    return JSON.parse(resultStr);
  }
  
  /**
   * Encode multiple texts to tokens
   */
  public encodeBatch(texts: string[], options?: EncodeOptions): TokenizationResult[] {
    this.checkDisposed();
    
    const lib = getLibrary();
    const textPtrs = texts.map(text => allocCString(text));
    const textsArray = Buffer.alloc(textPtrs.length * ref.sizeof.pointer);
    
    for (let i = 0; i < textPtrs.length; i++) {
      ref.writePointer(textsArray, i * ref.sizeof.pointer, textPtrs[i]);
    }
    
    const optionsCStr = allocCString(JSON.stringify(options || {}));
    const resultPtr = ref.alloc(ref.refType(ref.types.char));
    
    const result = lib.trustformers_tokenizer_encode_batch(
      this._handle,
      textsArray,
      texts.length,
      optionsCStr,
      resultPtr
    );
    checkError(result, 'tokenizer encode batch');
    
    const resultStr = readCString(resultPtr.deref());
    freeCString(resultPtr.deref());
    
    return JSON.parse(resultStr);
  }
  
  /**
   * Decode tokens to text
   */
  public decode(tokenIds: number[], options?: DecodeOptions): string {
    this.checkDisposed();
    
    const lib = getLibrary();
    const tokenIdsBuffer = Buffer.alloc(tokenIds.length * ref.sizeof.int);
    
    for (let i = 0; i < tokenIds.length; i++) {
      ref.writeInt32LE(tokenIdsBuffer, i * ref.sizeof.int, tokenIds[i]);
    }
    
    const optionsCStr = allocCString(JSON.stringify(options || {}));
    const resultPtr = ref.alloc(ref.refType(ref.types.char));
    
    const result = lib.trustformers_tokenizer_decode(
      this._handle,
      tokenIdsBuffer,
      tokenIds.length,
      optionsCStr,
      resultPtr
    );
    checkError(result, 'tokenizer decode');
    
    const resultStr = readCString(resultPtr.deref());
    freeCString(resultPtr.deref());
    
    const decoded = JSON.parse(resultStr);
    return decoded.text || decoded;
  }
  
  /**
   * Get vocabulary size
   */
  public getVocabSize(): number {
    this.checkDisposed();
    
    const lib = getLibrary();
    const sizePtr = ref.alloc(ref.types.int);
    const result = lib.trustformers_tokenizer_get_vocab_size(this._handle, sizePtr);
    checkError(result, 'get vocab size');
    
    return sizePtr.deref();
  }
  
  /**
   * Get vocabulary
   */
  public getVocab(): object {
    this.checkDisposed();
    
    const lib = getLibrary();
    const vocabPtr = ref.alloc(ref.refType(ref.types.char));
    const result = lib.trustformers_tokenizer_get_vocab(this._handle, vocabPtr);
    checkError(result, 'get vocab');
    
    const vocabStr = readCString(vocabPtr.deref());
    freeCString(vocabPtr.deref());
    
    return JSON.parse(vocabStr);
  }
  
  /**
   * Add special token
   */
  public addSpecialToken(token: string, tokenId: number): void {
    this.checkDisposed();
    
    const lib = getLibrary();
    const tokenCStr = allocCString(token);
    const result = lib.trustformers_tokenizer_add_special_token(this._handle, tokenCStr, tokenId);
    checkError(result, 'add special token');
  }
  
  /**
   * Dispose of the tokenizer and free resources
   */
  public dispose(): void {
    if (!this._disposed) {
      const lib = getLibrary();
      const result = lib.trustformers_tokenizer_free(this._handle);
      checkError(result, 'tokenizer free');
      this._disposed = true;
    }
  }
  
  private checkDisposed(): void {
    if (this._disposed) {
      throw new Error('Tokenizer has been disposed');
    }
  }
}

/**
 * Pipeline class for high-level inference tasks
 */
export class Pipeline extends EventEmitter {
  private _handle: PipelineHandle;
  private _disposed = false;
  private _config: PipelineConfig;
  
  constructor(config: PipelineConfig) {
    super();
    ensureInitialized();
    
    this._config = config;
    const lib = getLibrary();
    const handlePtr = ref.alloc(ref.types.int);
    
    if (typeof config.model === 'string') {
      const taskCStr = allocCString(config.task);
      const modelCStr = allocCString(config.model);
      const tokenizerCStr = allocCString(config.tokenizer || '');
      const configCStr = allocCString(JSON.stringify(config));
      
      const result = lib.trustformers_pipeline_create(
        taskCStr,
        modelCStr,
        tokenizerCStr,
        configCStr,
        handlePtr
      );
      checkError(result, 'pipeline create');
    } else {
      // Assuming model and tokenizer are already loaded Model and Tokenizer instances
      const modelId = (config.model as any).handle || 0;
      const tokenizerId = (config.tokenizer as any)?.handle || 0;
      
      const taskCStr = allocCString(config.task);
      const configCStr = allocCString(JSON.stringify(config));
      
      const result = lib.trustformers_pipeline_create_with_model(
        taskCStr,
        modelId,
        tokenizerId,
        configCStr,
        handlePtr
      );
      checkError(result, 'pipeline create with model');
    }
    
    this._handle = handlePtr.deref();
  }
  
  /**
   * Get pipeline handle
   */
  public get handle(): PipelineHandle {
    this.checkDisposed();
    return this._handle;
  }
  
  /**
   * Run inference on input
   */
  public run(input: string, config?: Partial<GenerationConfig>): any {
    this.checkDisposed();
    
    const lib = getLibrary();
    const inputCStr = allocCString(input);
    const configCStr = allocCString(JSON.stringify(config || {}));
    const resultPtr = ref.alloc(ref.refType(ref.types.char));
    
    const result = lib.trustformers_pipeline_run(this._handle, inputCStr, configCStr, resultPtr);
    checkError(result, 'pipeline run');
    
    const resultStr = readCString(resultPtr.deref());
    freeCString(resultPtr.deref());
    
    return JSON.parse(resultStr);
  }
  
  /**
   * Run batch inference
   */
  public runBatch(inputs: string[], config?: Partial<GenerationConfig>): any[] {
    this.checkDisposed();
    
    const lib = getLibrary();
    const inputPtrs = inputs.map(input => allocCString(input));
    const inputsArray = Buffer.alloc(inputPtrs.length * ref.sizeof.pointer);
    
    for (let i = 0; i < inputPtrs.length; i++) {
      ref.writePointer(inputsArray, i * ref.sizeof.pointer, inputPtrs[i]);
    }
    
    const configCStr = allocCString(JSON.stringify(config || {}));
    const resultPtr = ref.alloc(ref.refType(ref.types.char));
    
    const result = lib.trustformers_pipeline_run_batch(
      this._handle,
      inputsArray,
      inputs.length,
      configCStr,
      resultPtr
    );
    checkError(result, 'pipeline run batch');
    
    const resultStr = readCString(resultPtr.deref());
    freeCString(resultPtr.deref());
    
    return JSON.parse(resultStr);
  }
  
  /**
   * Run streaming inference
   */
  public async stream(
    input: string,
    config?: Partial<GenerationConfig>,
    onChunk?: StreamingCallback
  ): Promise<any> {
    this.checkDisposed();
    
    // For now, we'll implement this as a regular run since streaming requires callbacks
    // In a full implementation, you'd need to set up callback mechanisms
    const result = this.run(input, config);
    
    if (onChunk) {
      // Simulate streaming by sending the full result
      onChunk(result.text || JSON.stringify(result), true);
    }
    
    return result;
  }
  
  /**
   * Start a conversation
   */
  public startConversation(systemPrompt?: string): string {
    this.checkDisposed();
    
    const lib = getLibrary();
    const systemCStr = allocCString(systemPrompt || '');
    const conversationIdPtr = ref.alloc(ref.refType(ref.types.char));
    
    const result = lib.trustformers_pipeline_conversation_start(
      this._handle,
      systemCStr,
      conversationIdPtr
    );
    checkError(result, 'conversation start');
    
    const conversationId = readCString(conversationIdPtr.deref());
    freeCString(conversationIdPtr.deref());
    
    return conversationId;
  }
  
  /**
   * Add turn to conversation
   */
  public addConversationTurn(
    conversationId: string,
    role: 'user' | 'assistant' | 'system',
    content: string
  ): void {
    this.checkDisposed();
    
    const lib = getLibrary();
    const conversationIdCStr = allocCString(conversationId);
    const roleCStr = allocCString(role);
    const contentCStr = allocCString(content);
    
    const result = lib.trustformers_pipeline_conversation_add_turn(
      this._handle,
      conversationIdCStr,
      roleCStr,
      contentCStr
    );
    checkError(result, 'conversation add turn');
  }
  
  /**
   * Generate response in conversation
   */
  public generateConversationResponse(
    conversationId: string,
    config?: Partial<GenerationConfig>
  ): any {
    this.checkDisposed();
    
    const lib = getLibrary();
    const conversationIdCStr = allocCString(conversationId);
    const configCStr = allocCString(JSON.stringify(config || {}));
    const resultPtr = ref.alloc(ref.refType(ref.types.char));
    
    const result = lib.trustformers_pipeline_conversation_generate(
      this._handle,
      conversationIdCStr,
      configCStr,
      resultPtr
    );
    checkError(result, 'conversation generate');
    
    const resultStr = readCString(resultPtr.deref());
    freeCString(resultPtr.deref());
    
    return JSON.parse(resultStr);
  }
  
  /**
   * Get pipeline statistics
   */
  public getStats(): object {
    this.checkDisposed();
    
    const lib = getLibrary();
    const statsPtr = ref.alloc(ref.refType(ref.types.char));
    const result = lib.trustformers_pipeline_get_stats(this._handle, statsPtr);
    checkError(result, 'get pipeline stats');
    
    const statsStr = readCString(statsPtr.deref());
    freeCString(statsPtr.deref());
    
    return JSON.parse(statsStr);
  }
  
  /**
   * Dispose of the pipeline and free resources
   */
  public dispose(): void {
    if (!this._disposed) {
      const lib = getLibrary();
      const result = lib.trustformers_pipeline_free(this._handle);
      checkError(result, 'pipeline free');
      this._disposed = true;
    }
  }
  
  private checkDisposed(): void {
    if (this._disposed) {
      throw new Error('Pipeline has been disposed');
    }
  }
}

// Export the main API instance
export const trustformers = TrustformeRS.getInstance();