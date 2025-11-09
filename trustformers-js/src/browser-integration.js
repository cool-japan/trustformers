/**
 * Browser Integration for TrustformeRS
 * 
 * High-level integration utilities that combine browser APIs with TrustformeRS functionality
 */

import { 
  fileAPI, 
  indexedDBCache, 
  webStreams, 
  broadcastChannel,
  BrowserAPIsManager 
} from './browser-apis.js';

import { tensor, createModel } from './index.js';

/**
 * Model Caching and Storage
 */
export class ModelStorage {
  constructor() {
    this.cache = indexedDBCache;
    this.storeName = 'models';
  }

  // Save model to IndexedDB
  async saveModel(modelId, model) {
    try {
      const stateDict = model.state_dict();
      const {config} = model;
      
      await this.cache.store(this.storeName, modelId, {
        stateDict,
        config,
        type: 'model',
        modelType: config.model_type,
        timestamp: Date.now()
      });
      
      return true;
    } catch (error) {
      console.error('Failed to save model:', error);
      return false;
    }
  }

  // Load model from IndexedDB
  async loadModel(modelId) {
    try {
      const record = await this.cache.get(this.storeName, modelId);
      if (!record) return null;

      // Create model with config
      const model = createModel(record.data.config.model_type);
      
      // Load state dict
      model.load_state_dict(record.data.stateDict);
      
      return {
        model,
        metadata: {
          timestamp: record.timestamp,
          modelType: record.data.modelType
        }
      };
    } catch (error) {
      console.error('Failed to load model:', error);
      return null;
    }
  }

  // Check if model exists in cache
  async hasModel(modelId) {
    return await this.cache.exists(this.storeName, modelId);
  }

  // List cached models
  async listModels() {
    try {
      const keys = await this.cache.getAllKeys(this.storeName);
      const models = [];
      
      for (const key of keys) {
        const record = await this.cache.get(this.storeName, key);
        if (record && record.data.type === 'model') {
          models.push({
            id: key,
            modelType: record.data.modelType,
            timestamp: record.timestamp,
            size: record.size
          });
        }
      }
      
      return models;
    } catch (error) {
      console.error('Failed to list models:', error);
      return [];
    }
  }

  // Delete model from cache
  async deleteModel(modelId) {
    return await this.cache.delete(this.storeName, modelId);
  }
}

/**
 * File Upload and Processing
 */
export class FileProcessor {
  constructor() {
    this.file = fileAPI;
    this.supportedFormats = {
      model: ['.bin', '.safetensors', '.onnx'],
      config: ['.json'],
      tokenizer: ['.json', '.txt'],
      text: ['.txt', '.csv']
    };
  }

  // Process uploaded model file
  async processModelFile(file) {
    if (!this.file.validateFileType(file, ['application/octet-stream'])) {
      throw new Error('Invalid model file type');
    }

    const arrayBuffer = await this.file.readAsArrayBuffer(file);
    
    // Here you would implement model loading from binary data
    // For now, we'll return the buffer for processing
    return {
      name: file.name,
      size: file.size,
      data: arrayBuffer,
      type: 'model'
    };
  }

  // Process uploaded config file
  async processConfigFile(file) {
    if (!this.file.validateFileType(file, ['application/json', 'text/plain'])) {
      throw new Error('Invalid config file type');
    }

    const text = await this.file.readAsText(file);
    
    try {
      const config = JSON.parse(text);
      return {
        name: file.name,
        size: file.size,
        data: config,
        type: 'config'
      };
    } catch (_error) {
      throw new Error('Invalid JSON in config file');
    }
  }

  // Process uploaded text file for training/inference
  async processTextFile(file) {
    if (!this.file.validateFileType(file, ['text/plain'])) {
      throw new Error('Invalid text file type');
    }

    const text = await this.file.readAsText(file);
    
    // Split into lines and filter empty ones
    const lines = text.split('\n').filter(line => line.trim().length > 0);
    
    return {
      name: file.name,
      size: file.size,
      data: lines,
      type: 'text',
      lineCount: lines.length
    };
  }

  // Download tensor as file
  async downloadTensor(tensor, filename = 'tensor.bin') {
    const {data} = tensor;
    const blob = this.file.createDownloadBlob(data, 'application/octet-stream');
    this.file.downloadFile(blob, filename);
  }

  // Download model
  async downloadModel(model, filename = 'model.bin') {
    const stateDict = model.state_dict();
    const jsonString = JSON.stringify(stateDict);
    const blob = this.file.createDownloadBlob(jsonString, 'application/json');
    this.file.downloadFile(blob, filename);
  }
}

/**
 * Real-time Model Sharing
 */
export class ModelSharing {
  constructor() {
    this.broadcast = broadcastChannel;
    this.channelName = 'trustformers-models';
  }

  // Broadcast model inference results
  broadcastInference(modelId, input, output, metadata = {}) {
    this.broadcast.send(this.channelName, {
      type: 'inference',
      modelId,
      input: Array.from(input.data),
      inputShape: Array.from(input.shape),
      output: Array.from(output.data),
      outputShape: Array.from(output.shape),
      timestamp: Date.now(),
      ...metadata
    });
  }

  // Broadcast model loading status
  broadcastModelStatus(modelId, status, details = {}) {
    this.broadcast.send(this.channelName, {
      type: 'model-status',
      modelId,
      status, // 'loading', 'loaded', 'error'
      details,
      timestamp: Date.now()
    });
  }

  // Listen for shared model events
  onModelEvent(callback) {
    return this.broadcast.listen(this.channelName, (message) => {
      callback(message.data);
    });
  }

  // Broadcast performance metrics
  broadcastPerformance(metrics) {
    this.broadcast.send(this.channelName, {
      type: 'performance',
      metrics,
      timestamp: Date.now()
    });
  }
}

/**
 * Streaming Tensor Processing
 */
export class StreamingProcessor {
  constructor() {
    this.streams = webStreams;
  }

  // Process large tensor data in chunks
  async processTensorStream(tensorData, processor, chunkSize = 1024) {
    const stream = this.streams.createReadableStream((controller) => {
      let offset = 0;
      
      const processChunk = () => {
        if (offset < tensorData.length) {
          const chunk = tensorData.slice(offset, offset + chunkSize);
          const processed = processor(chunk, offset);
          controller.enqueue(processed);
          offset += chunkSize;
          
          // Use setTimeout to yield control
          setTimeout(processChunk, 0);
        } else {
          controller.close();
        }
      };
      
      processChunk();
    });

    const results = [];
    for await (const result of this.streams.streamToAsyncIterator(stream)) {
      results.push(result);
    }
    
    return results;
  }

  // Stream text processing for large documents
  async processTextStream(text, tokenizer, processor) {
    const lines = text.split('\n');
    
    const stream = this.streams.createReadableStream((controller) => {
      let lineIndex = 0;
      
      const processLine = () => {
        if (lineIndex < lines.length) {
          const line = lines[lineIndex];
          if (line.trim()) {
            try {
              const tokens = tokenizer.encode(line);
              const processed = processor(tokens, line, lineIndex);
              controller.enqueue(processed);
            } catch (error) {
              controller.enqueue({ error: error.message, line: lineIndex });
            }
          }
          lineIndex++;
          setTimeout(processLine, 0);
        } else {
          controller.close();
        }
      };
      
      processLine();
    });

    const results = [];
    for await (const result of this.streams.streamToAsyncIterator(stream)) {
      results.push(result);
    }
    
    return results;
  }

  // Streaming model inference for batched data
  async streamInference(model, dataStream, batchSize = 8) {
    const transform = this.streams.createTransformStream((chunk) => {
      if (chunk.length >= batchSize) {
        // Process batch
        const batchTensor = tensor(chunk, [chunk.length]);
        const output = model.forward(batchTensor);
        const result = Array.from(output.data);
        
        // Cleanup
        batchTensor.free();
        output.free();
        
        return result;
      }
      return chunk; // Pass through if not enough for batch
    });

    return dataStream.pipeThrough(transform);
  }
}

/**
 * Progressive Model Loading
 */
export class ProgressiveLoader {
  constructor() {
    this.cache = indexedDBCache;
    this.file = fileAPI;
  }

  // Load model with progress reporting
  async loadModelWithProgress(modelPath, onProgress) {
    try {
      onProgress({ stage: 'fetching', progress: 0 });
      
      // Fetch model file
      const response = await fetch(modelPath);
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.statusText}`);
      }

      const contentLength = response.headers.get('content-length');
      const total = contentLength ? parseInt(contentLength, 10) : 0;
      
      onProgress({ stage: 'downloading', progress: 0, total });

      // Read with progress
      const reader = response.body.getReader();
      const chunks = [];
      let loaded = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        chunks.push(value);
        loaded += value.length;
        
        if (total > 0) {
          onProgress({ 
            stage: 'downloading', 
            progress: (loaded / total) * 100,
            loaded,
            total 
          });
        }
      }

      onProgress({ stage: 'parsing', progress: 0 });

      // Combine chunks
      const arrayBuffer = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        arrayBuffer.set(chunk, offset);
        offset += chunk.length;
      }

      onProgress({ stage: 'loading', progress: 50 });

      // Create model (this would need actual model loading logic)
      // For now, we'll simulate the process
      await new Promise(resolve => setTimeout(resolve, 500));
      
      onProgress({ stage: 'complete', progress: 100 });

      return {
        data: arrayBuffer.buffer,
        size: loaded
      };

    } catch (error) {
      onProgress({ stage: 'error', error: error.message });
      throw error;
    }
  }

  // Cache model during loading
  async loadAndCacheModel(modelId, modelPath, onProgress) {
    // Check cache first
    if (await this.cache.exists('models', modelId)) {
      onProgress({ stage: 'cache-hit', progress: 100 });
      return await this.cache.get('models', modelId);
    }

    // Load with progress
    const modelData = await this.loadModelWithProgress(modelPath, onProgress);
    
    // Cache the result
    onProgress({ stage: 'caching', progress: 90 });
    await this.cache.store('models', modelId, modelData, {
      type: 'model',
      source: modelPath
    });

    onProgress({ stage: 'complete', progress: 100 });
    return modelData;
  }
}

/**
 * Offline Support Manager
 */
export class OfflineManager {
  constructor() {
    this.cache = indexedDBCache;
    this.isOnline = navigator?.onLine ?? true;
    this.offlineQueue = [];
    
    // Listen for online/offline events
    if (typeof window !== 'undefined') {
      window.addEventListener('online', () => {
        this.isOnline = true;
        this.processOfflineQueue();
      });
      
      window.addEventListener('offline', () => {
        this.isOnline = false;
      });
    }
  }

  // Check if we're online
  checkOnlineStatus() {
    return this.isOnline;
  }

  // Cache essential models for offline use
  async cacheForOffline(modelIds) {
    const cached = [];
    const failed = [];

    for (const modelId of modelIds) {
      try {
        if (!(await this.cache.exists('models', modelId))) {
          // Would need to download and cache
          console.warn(`Model ${modelId} not cached - would download if online`);
        }
        cached.push(modelId);
      } catch (error) {
        failed.push({ modelId, error: error.message });
      }
    }

    return { cached, failed };
  }

  // Queue operations for when online
  queueForOnline(operation) {
    this.offlineQueue.push({
      operation,
      timestamp: Date.now()
    });
  }

  // Process queued operations when online
  async processOfflineQueue() {
    if (!this.isOnline || this.offlineQueue.length === 0) return;

    const queue = [...this.offlineQueue];
    this.offlineQueue = [];

    for (const item of queue) {
      try {
        await item.operation();
      } catch (error) {
        console.error('Failed to process queued operation:', error);
        // Could re-queue on failure
      }
    }
  }

  // Get offline capabilities
  getOfflineCapabilities() {
    return {
      hasCache: this.cache.isSupported(),
      isOnline: this.isOnline,
      queuedOperations: this.offlineQueue.length
    };
  }
}

// Create singleton instances
export const modelStorage = new ModelStorage();
export const fileProcessor = new FileProcessor();
export const modelSharing = new ModelSharing();
export const streamingProcessor = new StreamingProcessor();
export const progressiveLoader = new ProgressiveLoader();
export const offlineManager = new OfflineManager();

// Export unified browser integration manager
export class BrowserIntegration {
  constructor() {
    this.apis = new BrowserAPIsManager();
    this.storage = modelStorage;
    this.files = fileProcessor;
    this.sharing = modelSharing;
    this.streaming = streamingProcessor;
    this.loader = progressiveLoader;
    this.offline = offlineManager;
  }

  // Initialize all browser integrations
  async init() {
    return await this.apis.init();
  }

  // Check what features are available
  getCapabilities() {
    return {
      ...this.apis.checkSupport(),
      offline: this.offline.getOfflineCapabilities()
    };
  }

  // Cleanup all resources
  cleanup() {
    this.apis.cleanup();
  }
}

export default new BrowserIntegration();