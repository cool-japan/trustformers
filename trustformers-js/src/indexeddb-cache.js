/* global Response, CompressionStream, DecompressionStream */

/**
 * TrustformeRS IndexedDB Cache Manager
 *
 * Provides persistent caching of models, weights, and other large assets
 * using IndexedDB for efficient browser storage.
 *
 * Features:
 * - Persistent model caching (survives page refreshes)
 * - Automatic cache invalidation and versioning
 * - LRU eviction policy for size management
 * - Compression support for reduced storage
 * - Efficient chunked storage for large models
 * - Cache statistics and monitoring
 *
 * @module indexeddb-cache
 */

/**
 * Cache configuration
 * @typedef {Object} CacheConfig
 * @property {string} [dbName='trustformers-cache'] - Database name
 * @property {number} [version=1] - Database version
 * @property {number} [maxSize=500*1024*1024] - Maximum cache size in bytes (500MB default)
 * @property {number} [chunkSize=10*1024*1024] - Chunk size for large items (10MB default)
 * @property {boolean} [compression=true] - Enable compression
 * @property {number} [ttl=7*24*60*60*1000] - Time to live in milliseconds (7 days default)
 * @property {string} [evictionPolicy='lru'] - Eviction policy: 'lru', 'lfu', 'fifo'
 */

/**
 * Cache entry metadata
 * @typedef {Object} CacheEntry
 * @property {string} key - Entry key
 * @property {number} size - Size in bytes
 * @property {number} createdAt - Creation timestamp
 * @property {number} lastAccessed - Last access timestamp
 * @property {number} accessCount - Number of accesses
 * @property {number} version - Entry version
 * @property {string} [contentType] - Content type
 * @property {boolean} compressed - Whether content is compressed
 * @property {Object} [metadata] - Additional metadata
 */

/**
 * IndexedDB Cache Manager
 */
export class IndexedDBCache {
  /**
   * Create a new cache manager
   * @param {CacheConfig} config - Cache configuration
   */
  constructor(config = {}) {
    this.config = {
      dbName: 'trustformers-cache',
      version: 1,
      maxSize: 500 * 1024 * 1024, // 500MB
      chunkSize: 10 * 1024 * 1024, // 10MB chunks
      compression: true,
      ttl: 7 * 24 * 60 * 60 * 1000, // 7 days
      evictionPolicy: 'lru', // LRU by default
      ...config
    };

    this.db = null;
    this.initialized = false;
    this.currentSize = 0;

    // Statistics
    this.stats = {
      hits: 0,
      misses: 0,
      sets: 0,
      evictions: 0,
      compressionSavings: 0
    };
  }

  /**
   * Initialize the cache
   * @returns {Promise<void>}
   */
  async initialize() {
    if (this.initialized) return Promise.resolve();

    if (typeof indexedDB === 'undefined') {
      throw new Error('IndexedDB is not supported in this environment');
    }

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.config.dbName, this.config.version);

      request.onerror = () => reject(new Error('Failed to open IndexedDB'));

      request.onsuccess = (event) => {
        this.db = event.target.result;
        this.initialized = true;
        this.calculateCurrentSize().then(() => resolve()).catch(reject);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        // Create object stores
        if (!db.objectStoreNames.contains('entries')) {
          const entryStore = db.createObjectStore('entries', { keyPath: 'key' });
          entryStore.createIndex('lastAccessed', 'lastAccessed', { unique: false });
          entryStore.createIndex('createdAt', 'createdAt', { unique: false });
          entryStore.createIndex('accessCount', 'accessCount', { unique: false });
        }

        if (!db.objectStoreNames.contains('chunks')) {
          db.createObjectStore('chunks', { keyPath: ['entryKey', 'chunkIndex'] });
        }
      };
    });
  }

  /**
   * Get an item from the cache
   * @param {string} key - Cache key
   * @returns {Promise<any|null>} Cached value or null
   */
  async get(key) {
    if (!this.initialized) await this.initialize();

    const entry = await this.getEntry(key);
    if (!entry) {
      this.stats.misses++;
      return null;
    }

    // Check TTL
    if (Date.now() - entry.createdAt > this.config.ttl) {
      await this.delete(key);
      this.stats.misses++;
      return null;
    }

    // Update access metadata
    entry.lastAccessed = Date.now();
    entry.accessCount++;
    await this.updateEntry(entry);

    // Load data
    const data = await this.loadData(entry);

    this.stats.hits++;
    return data;
  }

  /**
   * Set an item in the cache
   * @param {string} key - Cache key
   * @param {any} value - Value to cache
   * @param {Object} [metadata] - Additional metadata
   * @returns {Promise<void>}
   */
  async set(key, value, metadata = {}) {
    if (!this.initialized) await this.initialize();

    // Serialize value
    let data;
    let contentType;

    if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
      data = value instanceof ArrayBuffer ? value : value.buffer;
      contentType = 'binary';
    } else if (typeof value === 'string') {
      data = new TextEncoder().encode(value);
      contentType = 'text';
    } else {
      data = new TextEncoder().encode(JSON.stringify(value));
      contentType = 'json';
    }

    const originalSize = data.byteLength;
    let compressed = false;

    // Compress if enabled and beneficial
    if (this.config.compression && originalSize > 1024) {
      try {
        const compressedData = await this.compress(data);
        if (compressedData.byteLength < originalSize * 0.9) {
          this.stats.compressionSavings += originalSize - compressedData.byteLength;
          data = compressedData;
          compressed = true;
        }
      } catch (error) {
        console.warn('Compression failed, storing uncompressed:', error);
      }
    }

    const size = data.byteLength;

    // Check if we need to evict items
    while (this.currentSize + size > this.config.maxSize) {
      const evicted = await this.evictOne();
      if (!evicted) break; // No more items to evict
    }

    // Delete existing entry if present
    await this.delete(key);

    // Create entry metadata
    const entry = {
      key,
      size,
      createdAt: Date.now(),
      lastAccessed: Date.now(),
      accessCount: 0,
      version: 1,
      contentType,
      compressed,
      metadata
    };

    // Store data in chunks if needed
    await this.storeData(entry, data);

    // Store entry metadata
    await this.setEntry(entry);

    this.currentSize += size;
    this.stats.sets++;
  }

  /**
   * Delete an item from the cache
   * @param {string} key - Cache key
   * @returns {Promise<boolean>} Whether item was deleted
   */
  async delete(key) {
    if (!this.initialized) await this.initialize();

    const entry = await this.getEntry(key);
    if (!entry) return false;

    // Delete chunks
    await this.deleteChunks(key);

    // Delete entry
    await this.deleteEntry(key);

    this.currentSize -= entry.size;
    return true;
  }

  /**
   * Check if an item exists in the cache
   * @param {string} key - Cache key
   * @returns {Promise<boolean>}
   */
  async has(key) {
    if (!this.initialized) await this.initialize();

    const entry = await this.getEntry(key);
    if (!entry) return false;

    // Check TTL
    if (Date.now() - entry.createdAt > this.config.ttl) {
      await this.delete(key);
      return false;
    }

    return true;
  }

  /**
   * Clear all items from the cache
   * @returns {Promise<void>}
   */
  async clear() {
    if (!this.initialized) await this.initialize();

    await Promise.all([
      this.clearObjectStore('entries'),
      this.clearObjectStore('chunks')
    ]);

    this.currentSize = 0;
  }

  /**
   * Get cache statistics
   * @returns {Object} Statistics
   */
  getStats() {
    const hitRate = this.stats.hits + this.stats.misses > 0
      ? this.stats.hits / (this.stats.hits + this.stats.misses)
      : 0;

    return {
      ...this.stats,
      hitRate,
      currentSize: this.currentSize,
      maxSize: this.config.maxSize,
      utilizationPercent: (this.currentSize / this.config.maxSize) * 100
    };
  }

  /**
   * List all cache keys
   * @returns {Promise<string[]>}
   */
  async keys() {
    if (!this.initialized) await this.initialize();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['entries'], 'readonly');
      const store = transaction.objectStore('entries');
      const request = store.getAllKeys();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(new Error('Failed to get keys'));
    });
  }

  /**
   * Get entry metadata
   * @param {string} key - Entry key
   * @returns {Promise<CacheEntry|null>}
   */
  async getEntry(key) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['entries'], 'readonly');
      const store = transaction.objectStore('entries');
      const request = store.get(key);

      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(new Error('Failed to get entry'));
    });
  }

  /**
   * Set entry metadata
   * @param {CacheEntry} entry - Entry metadata
   * @returns {Promise<void>}
   */
  async setEntry(entry) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['entries'], 'readwrite');
      const store = transaction.objectStore('entries');
      const request = store.put(entry);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('Failed to set entry'));
    });
  }

  /**
   * Update entry metadata
   * @param {CacheEntry} entry - Entry metadata
   * @returns {Promise<void>}
   */
  async updateEntry(entry) {
    return this.setEntry(entry);
  }

  /**
   * Delete entry metadata
   * @param {string} key - Entry key
   * @returns {Promise<void>}
   */
  async deleteEntry(key) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['entries'], 'readwrite');
      const store = transaction.objectStore('entries');
      const request = store.delete(key);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('Failed to delete entry'));
    });
  }

  /**
   * Store data in chunks
   * @param {CacheEntry} entry - Entry metadata
   * @param {ArrayBuffer} data - Data to store
   * @returns {Promise<void>}
   */
  async storeData(entry, data) {
    const numChunks = Math.ceil(data.byteLength / this.config.chunkSize);

    const transaction = this.db.transaction(['chunks'], 'readwrite');
    const store = transaction.objectStore('chunks');

    for (let i = 0; i < numChunks; i++) {
      const start = i * this.config.chunkSize;
      const end = Math.min(start + this.config.chunkSize, data.byteLength);
      const chunk = data.slice(start, end);

      store.put({
        entryKey: entry.key,
        chunkIndex: i,
        data: chunk
      });
    }

    return new Promise((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(new Error('Failed to store data'));
    });
  }

  /**
   * Load data from chunks
   * @param {CacheEntry} entry - Entry metadata
   * @returns {Promise<any>}
   */
  async loadData(entry) {
    const numChunks = Math.ceil(entry.size / this.config.chunkSize);
    const chunks = [];

    const transaction = this.db.transaction(['chunks'], 'readonly');
    const store = transaction.objectStore('chunks');

    for (let i = 0; i < numChunks; i++) {
      const chunk = await new Promise((resolve, reject) => {
        const request = store.get([entry.key, i]);
        request.onsuccess = () => resolve(request.result?.data);
        request.onerror = () => reject(new Error('Failed to load chunk'));
      });
      chunks.push(chunk);
    }

    // Combine chunks
    let data = new Uint8Array(entry.size);
    let offset = 0;
    for (const chunk of chunks) {
      data.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }

    // Decompress if needed
    if (entry.compressed) {
      data = await this.decompress(data.buffer);
    }

    // Convert to original type
    switch (entry.contentType) {
      case 'text':
        return new TextDecoder().decode(data);
      case 'json':
        return JSON.parse(new TextDecoder().decode(data));
      case 'binary':
        return data.buffer;
      default:
        return data;
    }
  }

  /**
   * Delete chunks for an entry
   * @param {string} key - Entry key
   * @returns {Promise<void>}
   */
  async deleteChunks(key) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['chunks'], 'readwrite');
      const store = transaction.objectStore('chunks');

      // Get all chunk keys for this entry
      const range = IDBKeyRange.bound([key, 0], [key, Number.MAX_SAFE_INTEGER]);
      const request = store.openCursor(range);

      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        }
      };

      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(new Error('Failed to delete chunks'));
    });
  }

  /**
   * Evict one item from the cache
   * @returns {Promise<boolean>} Whether an item was evicted
   */
  async evictOne() {
    const entries = await this.getAllEntries();
    if (entries.length === 0) return false;

    let entryToEvict;

    switch (this.config.evictionPolicy) {
      case 'lru': // Least Recently Used
        entryToEvict = entries.reduce((oldest, entry) =>
          entry.lastAccessed < oldest.lastAccessed ? entry : oldest
        );
        break;

      case 'lfu': // Least Frequently Used
        entryToEvict = entries.reduce((least, entry) =>
          entry.accessCount < least.accessCount ? entry : least
        );
        break;

      case 'fifo': // First In First Out
        entryToEvict = entries.reduce((oldest, entry) =>
          entry.createdAt < oldest.createdAt ? entry : oldest
        );
        break;

      default:
        [entryToEvict] = entries;
    }

    await this.delete(entryToEvict.key);
    this.stats.evictions++;
    return true;
  }

  /**
   * Get all entries
   * @returns {Promise<CacheEntry[]>}
   */
  async getAllEntries() {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['entries'], 'readonly');
      const store = transaction.objectStore('entries');
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(new Error('Failed to get all entries'));
    });
  }

  /**
   * Calculate current cache size
   * @returns {Promise<void>}
   */
  async calculateCurrentSize() {
    const entries = await this.getAllEntries();
    this.currentSize = entries.reduce((sum, entry) => sum + entry.size, 0);
  }

  /**
   * Clear an object store
   * @param {string} storeName - Store name
   * @returns {Promise<void>}
   */
  async clearObjectStore(storeName) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.clear();

      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error(`Failed to clear ${storeName}`));
    });
  }

  /**
   * Compress data using CompressionStream API
   * @param {ArrayBuffer} data - Data to compress
   * @returns {Promise<ArrayBuffer>}
   */
  async compress(data) {
    if (typeof CompressionStream === 'undefined') {
      throw new Error('CompressionStream not supported');
    }

    const stream = new Response(data).body.pipeThrough(
      new CompressionStream('gzip')
    );
    const compressedResponse = await new Response(stream).arrayBuffer();
    return compressedResponse;
  }

  /**
   * Decompress data using DecompressionStream API
   * @param {ArrayBuffer} data - Data to decompress
   * @returns {Promise<ArrayBuffer>}
   */
  async decompress(data) {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('DecompressionStream not supported');
    }

    const stream = new Response(data).body.pipeThrough(
      new DecompressionStream('gzip')
    );
    const decompressedResponse = await new Response(stream).arrayBuffer();
    return decompressedResponse;
  }

  /**
   * Close the database connection
   * @returns {Promise<void>}
   */
  async close() {
    if (this.db) {
      this.db.close();
      this.db = null;
      this.initialized = false;
    }
  }
}

/**
 * Model cache specifically for TrustformeRS models
 */
export class ModelCache extends IndexedDBCache {
  constructor(config = {}) {
    super({
      dbName: 'trustformers-model-cache',
      maxSize: 2 * 1024 * 1024 * 1024, // 2GB for models
      ...config
    });
  }

  /**
   * Cache a model
   * @param {string} modelId - Model ID
   * @param {Object} modelData - Model data (weights, config, etc.)
   * @returns {Promise<void>}
   */
  async cacheModel(modelId, modelData) {
    const metadata = {
      type: 'model',
      modelId,
      architecture: modelData.architecture || 'unknown',
      version: modelData.version || '1.0.0'
    };

    await this.set(`model:${modelId}`, modelData, metadata);
  }

  /**
   * Load a cached model
   * @param {string} modelId - Model ID
   * @returns {Promise<Object|null>}
   */
  async loadModel(modelId) {
    return await this.get(`model:${modelId}`);
  }

  /**
   * Cache model weights
   * @param {string} modelId - Model ID
   * @param {ArrayBuffer} weights - Model weights
   * @returns {Promise<void>}
   */
  async cacheWeights(modelId, weights) {
    const metadata = {
      type: 'weights',
      modelId
    };

    await this.set(`weights:${modelId}`, weights, metadata);
  }

  /**
   * Load cached model weights
   * @param {string} modelId - Model ID
   * @returns {Promise<ArrayBuffer|null>}
   */
  async loadWeights(modelId) {
    return await this.get(`weights:${modelId}`);
  }

  /**
   * List all cached models
   * @returns {Promise<string[]>}
   */
  async listModels() {
    const keys = await this.keys();
    return keys
      .filter(key => key.startsWith('model:'))
      .map(key => key.substring(6));
  }
}

/**
 * Create a cache instance
 * @param {CacheConfig} config - Cache configuration
 * @returns {IndexedDBCache}
 */
export function createCache(config) {
  return new IndexedDBCache(config);
}

/**
 * Create a model cache instance
 * @param {CacheConfig} config - Cache configuration
 * @returns {ModelCache}
 */
export function createModelCache(config) {
  return new ModelCache(config);
}

export default {
  IndexedDBCache,
  ModelCache,
  createCache,
  createModelCache
};
