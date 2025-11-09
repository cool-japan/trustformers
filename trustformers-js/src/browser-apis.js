/**
 * Browser APIs Integration for TrustformeRS
 * 
 * This module provides integration with various browser APIs including
 * File API, IndexedDB, Web Streams, Broadcast Channel, and Shared Array Buffer
 */

// Check for browser environment
const isBrowser = typeof window !== 'undefined' && typeof document !== 'undefined';

/**
 * File API Integration
 */
export class FileAPIManager {
  constructor() {
    this.supportedTypes = [
      'application/octet-stream',  // WASM files
      'application/json',          // Model configs
      'text/plain',               // Text files
      'image/*'                   // Images (future support)
    ];
  }

  // Check if File API is supported
  isSupported() {
    return isBrowser && 
           typeof File !== 'undefined' && 
           typeof FileReader !== 'undefined' && 
           typeof Blob !== 'undefined';
  }

  // Read file as ArrayBuffer (for WASM modules, model weights)
  async readAsArrayBuffer(file) {
    if (!this.isSupported()) {
      throw new Error('File API not supported in this environment');
    }

    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => resolve(event.target.result);
      reader.onerror = (error) => reject(new Error(`Failed to read file: ${error}`));
      
      reader.readAsArrayBuffer(file);
    });
  }

  // Read file as text (for configs, vocabularies)
  async readAsText(file) {
    if (!this.isSupported()) {
      throw new Error('File API not supported in this environment');
    }

    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => resolve(event.target.result);
      reader.onerror = (error) => reject(new Error(`Failed to read file: ${error}`));
      
      reader.readAsText(file);
    });
  }

  // Read file as Data URL (for small files, debugging)
  async readAsDataURL(file) {
    if (!this.isSupported()) {
      throw new Error('File API not supported in this environment');
    }

    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => resolve(event.target.result);
      reader.onerror = (error) => reject(new Error(`Failed to read file: ${error}`));
      
      reader.readAsDataURL(file);
    });
  }

  // Create downloadable blob from data
  createDownloadBlob(data, type = 'application/octet-stream') {
    if (!this.isSupported()) {
      throw new Error('Blob API not supported in this environment');
    }

    return new Blob([data], { type });
  }

  // Trigger file download
  downloadFile(blob, filename) {
    if (!isBrowser) {
      throw new Error('File download not supported in this environment');
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.style.display = 'none';
    
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    // Clean up object URL
    setTimeout(() => URL.revokeObjectURL(url), 100);
  }

  // Validate file type
  validateFileType(file, allowedTypes = this.supportedTypes) {
    return allowedTypes.some(type => {
      if (type.endsWith('/*')) {
        const prefix = type.slice(0, -2);
        return file.type.startsWith(prefix);
      }
      return file.type === type;
    });
  }

  // Get file size in MB
  getFileSizeMB(file) {
    return (file.size / 1024 / 1024).toFixed(2);
  }
}

/**
 * IndexedDB Caching Manager
 */
export class IndexedDBCache {
  constructor(dbName = 'TrustformeRS', version = 1) {
    this.dbName = dbName;
    this.version = version;
    this.db = null;
    this.stores = {
      models: 'models',
      tensors: 'tensors',
      configs: 'configs',
      cache: 'cache'
    };
  }

  // Check if IndexedDB is supported
  isSupported() {
    return isBrowser && typeof indexedDB !== 'undefined';
  }

  // Initialize database
  async init() {
    if (!this.isSupported()) {
      throw new Error('IndexedDB not supported in this environment');
    }

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => reject(new Error('Failed to open IndexedDB'));

      request.onsuccess = (event) => {
        this.db = event.target.result;
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // Create object stores
        Object.values(this.stores).forEach(storeName => {
          if (!db.objectStoreNames.contains(storeName)) {
            const store = db.createObjectStore(storeName, { keyPath: 'id' });
            store.createIndex('timestamp', 'timestamp', { unique: false });
            store.createIndex('type', 'type', { unique: false });
          }
        });
      };
    });
  }

  // Store data in cache
  async store(storeName, id, data, metadata = {}) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);

      const record = {
        id,
        data,
        timestamp: Date.now(),
        size: this._calculateSize(data),
        ...metadata
      };

      const request = store.put(record);

      request.onsuccess = () => resolve(record);
      request.onerror = () => reject(new Error(`Failed to store data in ${storeName}`));
    });
  }

  // Retrieve data from cache
  async get(storeName, id) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(id);

      request.onsuccess = (event) => {
        const {result} = event.target;
        if (result) {
          resolve(result);
        } else {
          resolve(null);
        }
      };

      request.onerror = () => reject(new Error(`Failed to retrieve data from ${storeName}`));
    });
  }

  // Check if data exists in cache
  async exists(storeName, id) {
    const data = await this.get(storeName, id);
    return data !== null;
  }

  // Delete data from cache
  async delete(storeName, id) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.delete(id);

      request.onsuccess = () => resolve(true);
      request.onerror = () => reject(new Error(`Failed to delete data from ${storeName}`));
    });
  }

  // Clear entire store
  async clear(storeName) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.clear();

      request.onsuccess = () => resolve(true);
      request.onerror = () => reject(new Error(`Failed to clear ${storeName}`));
    });
  }

  // List all keys in store
  async getAllKeys(storeName) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.getAllKeys();

      request.onsuccess = (event) => resolve(event.target.result);
      request.onerror = () => reject(new Error(`Failed to get keys from ${storeName}`));
    });
  }

  // Get cache statistics
  async getStats() {
    if (!this.db) await this.init();

    const stats = {};
    
    for (const storeName of Object.values(this.stores)) {
      try {
        const keys = await this.getAllKeys(storeName);
        let totalSize = 0;
        let count = 0;

        for (const key of keys) {
          const record = await this.get(storeName, key);
          if (record) {
            totalSize += record.size || 0;
            count++;
          }
        }

        stats[storeName] = {
          count,
          totalSize,
          totalSizeMB: (totalSize / 1024 / 1024).toFixed(2)
        };
      } catch (error) {
        stats[storeName] = { error: error.message };
      }
    }

    return stats;
  }

  // Cleanup old entries
  async cleanup(storeName, maxAge = 7 * 24 * 60 * 60 * 1000) { // 7 days default
    if (!this.db) await this.init();

    const cutoffTime = Date.now() - maxAge;
    const keys = await this.getAllKeys(storeName);
    let deletedCount = 0;

    for (const key of keys) {
      const record = await this.get(storeName, key);
      if (record && record.timestamp < cutoffTime) {
        await this.delete(storeName, key);
        deletedCount++;
      }
    }

    return deletedCount;
  }

  // Calculate approximate size of data
  _calculateSize(data) {
    try {
      return new Blob([JSON.stringify(data)]).size;
    } catch (_e) {
      return 0;
    }
  }
}

/**
 * Web Streams Support
 */
export class WebStreamsManager {
  constructor() {
    this.activeStreams = new Set();
  }

  // Check if Web Streams are supported
  isSupported() {
    return isBrowser && 
           typeof ReadableStream !== 'undefined' && 
           typeof WritableStream !== 'undefined';
  }

  // Create readable stream for large data processing
  createReadableStream(source, options = {}) {
    if (!this.isSupported()) {
      throw new Error('Web Streams not supported in this environment');
    }

    const stream = new ReadableStream({
      start(controller) {
        if (typeof source === 'function') {
          source(controller);
        } else if (Array.isArray(source)) {
          source.forEach(chunk => controller.enqueue(chunk));
          controller.close();
        }
      },
      
      pull(controller) {
        // Called when stream wants more data
        if (options.onPull) {
          options.onPull(controller);
        }
      },
      
      cancel() {
        // Called when stream is cancelled
        if (options.onCancel) {
          options.onCancel();
        }
      }
    });

    this.activeStreams.add(stream);
    return stream;
  }

  // Create transform stream for data processing
  createTransformStream(transformer) {
    if (!this.isSupported()) {
      throw new Error('Web Streams not supported in this environment');
    }

    return new TransformStream({
      transform(chunk, controller) {
        const result = transformer(chunk);
        if (result !== undefined) {
          controller.enqueue(result);
        }
      }
    });
  }

  // Stream tensor data processing
  async processTensorStream(tensorData, chunkSize = 1024) {
    const stream = this.createReadableStream((controller) => {
      let offset = 0;
      
      const pushChunk = () => {
        if (offset < tensorData.length) {
          const chunk = tensorData.slice(offset, offset + chunkSize);
          controller.enqueue(chunk);
          offset += chunkSize;
          setTimeout(pushChunk, 0); // Yield control
        } else {
          controller.close();
        }
      };
      
      pushChunk();
    });

    return stream;
  }

  // Process stream with async iterator
  async *streamToAsyncIterator(stream) {
    const reader = stream.getReader();
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        yield value;
      }
    } finally {
      reader.releaseLock();
    }
  }

  // Cleanup streams
  cleanup() {
    this.activeStreams.forEach(stream => {
      try {
        if (stream.cancel) {
          stream.cancel();
        }
      } catch (_e) {
        // Ignore cleanup errors
      }
    });
    this.activeStreams.clear();
  }
}

/**
 * Broadcast Channel Manager
 */
export class BroadcastChannelManager {
  constructor() {
    this.channels = new Map();
  }

  // Check if Broadcast Channel is supported
  isSupported() {
    return isBrowser && typeof BroadcastChannel !== 'undefined';
  }

  // Create or get channel
  getChannel(name) {
    if (!this.isSupported()) {
      throw new Error('BroadcastChannel not supported in this environment');
    }

    if (!this.channels.has(name)) {
      const channel = new BroadcastChannel(name);
      this.channels.set(name, channel);
    }

    return this.channels.get(name);
  }

  // Send message to channel
  send(channelName, message) {
    const channel = this.getChannel(channelName);
    channel.postMessage({
      timestamp: Date.now(),
      data: message
    });
  }

  // Listen to channel messages
  listen(channelName, callback) {
    const channel = this.getChannel(channelName);
    
    const handler = (event) => {
      callback(event.data);
    };
    
    channel.addEventListener('message', handler);
    
    // Return cleanup function
    return () => {
      channel.removeEventListener('message', handler);
    };
  }

  // Close specific channel
  closeChannel(name) {
    if (this.channels.has(name)) {
      this.channels.get(name).close();
      this.channels.delete(name);
    }
  }

  // Close all channels
  closeAll() {
    this.channels.forEach((channel, _name) => {
      channel.close();
    });
    this.channels.clear();
  }
}

/**
 * Shared Array Buffer Manager
 */
export class SharedArrayBufferManager {
  constructor() {
    this.buffers = new Map();
  }

  // Check if SharedArrayBuffer is supported
  isSupported() {
    return typeof SharedArrayBuffer !== 'undefined';
  }

  // Create shared buffer
  createBuffer(name, size) {
    if (!this.isSupported()) {
      // Fallback to regular ArrayBuffer
      console.warn('SharedArrayBuffer not supported, falling back to ArrayBuffer');
      return new ArrayBuffer(size);
    }

    const buffer = new SharedArrayBuffer(size);
    this.buffers.set(name, buffer);
    return buffer;
  }

  // Get existing buffer
  getBuffer(name) {
    return this.buffers.get(name);
  }

  // Create typed view of shared buffer
  createView(bufferName, ViewType, offset = 0, length) {
    const buffer = this.getBuffer(bufferName);
    if (!buffer) {
      throw new Error(`Buffer '${bufferName}' not found`);
    }

    return new ViewType(buffer, offset, length);
  }

  // Check if buffer is shared
  isShared(buffer) {
    return this.isSupported() && buffer instanceof SharedArrayBuffer;
  }

  // Get buffer info
  getBufferInfo(name) {
    const buffer = this.getBuffer(name);
    if (!buffer) return null;

    return {
      name,
      size: buffer.byteLength,
      sizeMB: (buffer.byteLength / 1024 / 1024).toFixed(2),
      isShared: this.isShared(buffer)
    };
  }

  // List all buffers
  listBuffers() {
    return Array.from(this.buffers.keys()).map(name => 
      this.getBufferInfo(name)
    );
  }

  // Cleanup buffer
  removeBuffer(name) {
    this.buffers.delete(name);
  }

  // Cleanup all buffers
  cleanup() {
    this.buffers.clear();
  }
}

// Create singleton instances
export const fileAPI = new FileAPIManager();
export const indexedDBCache = new IndexedDBCache();
export const webStreams = new WebStreamsManager();
export const broadcastChannel = new BroadcastChannelManager();
export const sharedArrayBuffer = new SharedArrayBufferManager();

// Export unified browser APIs manager
export class BrowserAPIsManager {
  constructor() {
    this.file = fileAPI;
    this.cache = indexedDBCache;
    this.streams = webStreams;
    this.broadcast = broadcastChannel;
    this.sharedMemory = sharedArrayBuffer;
  }

  // Check overall browser support
  checkSupport() {
    return {
      fileAPI: this.file.isSupported(),
      indexedDB: this.cache.isSupported(),
      webStreams: this.streams.isSupported(),
      broadcastChannel: this.broadcast.isSupported(),
      sharedArrayBuffer: this.sharedMemory.isSupported(),
      isBrowser
    };
  }

  // Initialize all supported APIs
  async init() {
    const support = this.checkSupport();
    const initialized = {};

    if (support.indexedDB) {
      try {
        await this.cache.init();
        initialized.indexedDB = true;
      } catch (error) {
        initialized.indexedDB = { error: error.message };
      }
    }

    initialized.support = support;
    return initialized;
  }

  // Cleanup all resources
  cleanup() {
    this.streams.cleanup();
    this.broadcast.closeAll();
    this.sharedMemory.cleanup();
  }
}

export default new BrowserAPIsManager();