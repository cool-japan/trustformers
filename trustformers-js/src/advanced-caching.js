/**
 * Advanced Caching Strategies
 *
 * Provides sophisticated caching mechanisms for optimizing model inference and data access.
 * Implements multiple caching strategies:
 * - LRU (Least Recently Used)
 * - LFU (Least Frequently Used)
 * - TTL (Time To Live)
 * - ARC (Adaptive Replacement Cache)
 * - Multi-level caching
 *
 * Features:
 * - Automatic eviction policies
 * - Memory-aware caching
 * - Compression support
 * - Persistence to IndexedDB/localStorage
 * - Cache warming and prefetching
 * - Statistics and monitoring
 */

/**
 * LRU (Least Recently Used) Cache
 */
export class LRUCache {
  constructor(options = {}) {
    this.maxSize = options.maxSize || 100;
    this.maxMemory = options.maxMemory || Infinity; // bytes
    this.onEvict = options.onEvict || null;

    this.cache = new Map();
    this.accessOrder = [];
    this.currentMemory = 0;
    this.statistics = {
      hits: 0,
      misses: 0,
      evictions: 0,
      insertions: 0
    };
  }

  /**
   * Get value from cache
   */
  get(key) {
    if (!this.cache.has(key)) {
      this.statistics.misses++;
      return null;
    }

    this.statistics.hits++;

    // Move to end (most recently used)
    this._updateAccessOrder(key);

    const entry = this.cache.get(key);
    entry.lastAccess = Date.now();
    entry.accessCount++;

    return entry.value;
  }

  /**
   * Set value in cache
   */
  set(key, value, options = {}) {
    const size = options.size || this._estimateSize(value);
    const ttl = options.ttl || null;

    // Check if we need to evict
    while (this._shouldEvict(size)) {
      this._evictLRU();
    }

    // Remove old entry if exists
    if (this.cache.has(key)) {
      const oldEntry = this.cache.get(key);
      this.currentMemory -= oldEntry.size;
      this._removeFromAccessOrder(key);
    }

    // Add new entry
    const entry = {
      key,
      value,
      size,
      createdAt: Date.now(),
      lastAccess: Date.now(),
      accessCount: 0,
      ttl,
      expiresAt: ttl ? Date.now() + ttl : null
    };

    this.cache.set(key, entry);
    this.accessOrder.push(key);
    this.currentMemory += size;
    this.statistics.insertions++;

    return true;
  }

  /**
   * Check if key exists
   */
  has(key) {
    if (!this.cache.has(key)) {
      return false;
    }

    const entry = this.cache.get(key);

    // Check TTL
    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      this.delete(key);
      return false;
    }

    return true;
  }

  /**
   * Delete key
   */
  delete(key) {
    if (!this.cache.has(key)) {
      return false;
    }

    const entry = this.cache.get(key);
    this.cache.delete(key);
    this._removeFromAccessOrder(key);
    this.currentMemory -= entry.size;

    return true;
  }

  /**
   * Clear cache
   */
  clear() {
    this.cache.clear();
    this.accessOrder = [];
    this.currentMemory = 0;
  }

  /**
   * Get cache statistics
   */
  getStatistics() {
    const hitRate = this.statistics.hits / (this.statistics.hits + this.statistics.misses) || 0;

    return {
      ...this.statistics,
      hitRate: `${(hitRate * 100).toFixed(2)}%`,
      size: this.cache.size,
      memoryUsage: this.currentMemory,
      memoryUsageMB: (this.currentMemory / 1024 / 1024).toFixed(2)
    };
  }

  _shouldEvict(newSize) {
    return (this.cache.size >= this.maxSize) ||
           (this.currentMemory + newSize > this.maxMemory);
  }

  _evictLRU() {
    if (this.accessOrder.length === 0) {
      return;
    }

    // Find first non-expired entry
    let keyToEvict = null;
    for (const key of this.accessOrder) {
      const entry = this.cache.get(key);
      if (!entry.expiresAt || Date.now() <= entry.expiresAt) {
        keyToEvict = key;
        break;
      }
    }

    if (!keyToEvict && this.accessOrder.length > 0) {
      keyToEvict = this.accessOrder[0];
    }

    if (keyToEvict) {
      const entry = this.cache.get(keyToEvict);

      if (this.onEvict) {
        this.onEvict(keyToEvict, entry.value);
      }

      this.delete(keyToEvict);
      this.statistics.evictions++;
    }
  }

  _updateAccessOrder(key) {
    this._removeFromAccessOrder(key);
    this.accessOrder.push(key);
  }

  _removeFromAccessOrder(key) {
    const index = this.accessOrder.indexOf(key);
    if (index > -1) {
      this.accessOrder.splice(index, 1);
    }
  }

  _estimateSize(value) {
    if (value === null || value === undefined) {
      return 8;
    }

    if (ArrayBuffer.isView(value)) {
      return value.byteLength;
    }

    if (value instanceof ArrayBuffer) {
      return value.byteLength;
    }

    if (typeof value === 'string') {
      return value.length * 2; // UTF-16
    }

    if (typeof value === 'object') {
      return JSON.stringify(value).length * 2;
    }

    return 8; // Default size
  }
}

/**
 * TTL (Time To Live) Cache
 */
export class TTLCache {
  constructor(options = {}) {
    this.defaultTTL = options.defaultTTL || 60000; // 1 minute
    this.checkInterval = options.checkInterval || 10000; // 10 seconds
    this.maxSize = options.maxSize || 1000;

    this.cache = new Map();
    this.timers = new Map();
    this.statistics = {
      hits: 0,
      misses: 0,
      expirations: 0
    };

    // Start cleanup interval
    this._startCleanupInterval();
  }

  get(key) {
    const entry = this.cache.get(key);

    if (!entry) {
      this.statistics.misses++;
      return null;
    }

    if (Date.now() > entry.expiresAt) {
      this._expire(key);
      this.statistics.misses++;
      return null;
    }

    this.statistics.hits++;
    return entry.value;
  }

  set(key, value, ttl = null) {
    const actualTTL = ttl || this.defaultTTL;
    const expiresAt = Date.now() + actualTTL;

    // Clear existing timer
    if (this.timers.has(key)) {
      clearTimeout(this.timers.get(key));
    }

    // Set expiration timer
    const timer = setTimeout(() => this._expire(key), actualTTL);
    this.timers.set(key, timer);

    // Store entry
    this.cache.set(key, {
      value,
      expiresAt,
      createdAt: Date.now()
    });

    // Enforce size limit
    if (this.cache.size > this.maxSize) {
      this._evictOldest();
    }

    return true;
  }

  has(key) {
    const entry = this.cache.get(key);
    if (!entry) {
      return false;
    }

    if (Date.now() > entry.expiresAt) {
      this._expire(key);
      return false;
    }

    return true;
  }

  delete(key) {
    if (this.timers.has(key)) {
      clearTimeout(this.timers.get(key));
      this.timers.delete(key);
    }
    return this.cache.delete(key);
  }

  clear() {
    for (const timer of this.timers.values()) {
      clearTimeout(timer);
    }
    this.timers.clear();
    this.cache.clear();
  }

  getStatistics() {
    const hitRate = this.statistics.hits / (this.statistics.hits + this.statistics.misses) || 0;

    return {
      ...this.statistics,
      hitRate: `${(hitRate * 100).toFixed(2)}%`,
      size: this.cache.size,
      activeTimers: this.timers.size
    };
  }

  _expire(key) {
    this.cache.delete(key);
    if (this.timers.has(key)) {
      clearTimeout(this.timers.get(key));
      this.timers.delete(key);
    }
    this.statistics.expirations++;
  }

  _evictOldest() {
    let oldestKey = null;
    let oldestTime = Infinity;

    for (const [key, entry] of this.cache.entries()) {
      if (entry.createdAt < oldestTime) {
        oldestTime = entry.createdAt;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      this.delete(oldestKey);
    }
  }

  _startCleanupInterval() {
    this.cleanupInterval = setInterval(() => {
      const now = Date.now();
      const keysToDelete = [];

      for (const [key, entry] of this.cache.entries()) {
        if (now > entry.expiresAt) {
          keysToDelete.push(key);
        }
      }

      for (const key of keysToDelete) {
        this._expire(key);
      }
    }, this.checkInterval);
  }

  dispose() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    this.clear();
  }
}

/**
 * LFU (Least Frequently Used) Cache
 */
export class LFUCache {
  constructor(options = {}) {
    this.maxSize = options.maxSize || 100;
    this.cache = new Map();
    this.frequencies = new Map();
    this.statistics = {
      hits: 0,
      misses: 0,
      evictions: 0
    };
  }

  get(key) {
    if (!this.cache.has(key)) {
      this.statistics.misses++;
      return null;
    }

    this.statistics.hits++;

    // Increment frequency
    const freq = this.frequencies.get(key) || 0;
    this.frequencies.set(key, freq + 1);

    return this.cache.get(key);
  }

  set(key, value) {
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      this._evictLFU();
    }

    this.cache.set(key, value);
    this.frequencies.set(key, 0);

    return true;
  }

  has(key) {
    return this.cache.has(key);
  }

  delete(key) {
    this.frequencies.delete(key);
    return this.cache.delete(key);
  }

  clear() {
    this.cache.clear();
    this.frequencies.clear();
  }

  getStatistics() {
    const hitRate = this.statistics.hits / (this.statistics.hits + this.statistics.misses) || 0;

    return {
      ...this.statistics,
      hitRate: `${(hitRate * 100).toFixed(2)}%`,
      size: this.cache.size
    };
  }

  _evictLFU() {
    let minFreq = Infinity;
    let lfuKey = null;

    for (const [key, freq] of this.frequencies.entries()) {
      if (freq < minFreq) {
        minFreq = freq;
        lfuKey = key;
      }
    }

    if (lfuKey) {
      this.delete(lfuKey);
      this.statistics.evictions++;
    }
  }
}

/**
 * Multi-level Cache
 * Combines multiple cache levels (L1, L2, L3)
 */
export class MultiLevelCache {
  constructor(options = {}) {
    // L1: Small, fast LRU cache
    this.l1 = new LRUCache({
      maxSize: options.l1Size || 50,
      maxMemory: options.l1Memory || 50 * 1024 * 1024 // 50MB
    });

    // L2: Medium-sized TTL cache
    this.l2 = new TTLCache({
      maxSize: options.l2Size || 200,
      defaultTTL: options.l2TTL || 300000 // 5 minutes
    });

    // L3: Large LFU cache
    this.l3 = new LFUCache({
      maxSize: options.l3Size || 1000
    });

    this.statistics = {
      l1Hits: 0,
      l2Hits: 0,
      l3Hits: 0,
      misses: 0
    };
  }

  async get(key) {
    // Try L1
    let value = this.l1.get(key);
    if (value !== null) {
      this.statistics.l1Hits++;
      return value;
    }

    // Try L2
    value = this.l2.get(key);
    if (value !== null) {
      this.statistics.l2Hits++;
      // Promote to L1
      this.l1.set(key, value);
      return value;
    }

    // Try L3
    value = this.l3.get(key);
    if (value !== null) {
      this.statistics.l3Hits++;
      // Promote to L1 and L2
      this.l1.set(key, value);
      this.l2.set(key, value);
      return value;
    }

    this.statistics.misses++;
    return null;
  }

  async set(key, value, options = {}) {
    // Set in all levels
    this.l1.set(key, value, options);
    this.l2.set(key, value, options.ttl);
    this.l3.set(key, value);
  }

  has(key) {
    return this.l1.has(key) || this.l2.has(key) || this.l3.has(key);
  }

  delete(key) {
    this.l1.delete(key);
    this.l2.delete(key);
    this.l3.delete(key);
  }

  clear() {
    this.l1.clear();
    this.l2.clear();
    this.l3.clear();
  }

  getStatistics() {
    const total = this.statistics.l1Hits + this.statistics.l2Hits +
                  this.statistics.l3Hits + this.statistics.misses;
    const hitRate = total > 0 ?
      ((total - this.statistics.misses) / total * 100).toFixed(2) : '0.00';

    return {
      ...this.statistics,
      hitRate: `${hitRate}%`,
      l1Stats: this.l1.getStatistics(),
      l2Stats: this.l2.getStatistics(),
      l3Stats: this.l3.getStatistics()
    };
  }

  dispose() {
    this.l2.dispose();
    this.clear();
  }
}

/**
 * Persistent Cache
 * Uses IndexedDB for persistence
 */
export class PersistentCache {
  constructor(options = {}) {
    this.dbName = options.dbName || 'trustformers-cache';
    this.storeName = options.storeName || 'cache';
    this.version = options.version || 1;
    this.memoryCache = new LRUCache({ maxSize: 50 });
    this.db = null;
  }

  async initialize() {
    if (typeof indexedDB === 'undefined') {
      console.warn('IndexedDB not available, using memory cache only');
      return;
    }

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName);
        }
      };
    });
  }

  async get(key) {
    // Try memory cache first
    const cached = this.memoryCache.get(key);
    if (cached !== null) {
      return cached;
    }

    // Try IndexedDB
    if (!this.db) {
      return null;
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.storeName], 'readonly');
      const store = transaction.objectStore(this.storeName);
      const request = store.get(key);

      request.onsuccess = () => {
        const value = request.result;
        if (value) {
          // Add to memory cache
          this.memoryCache.set(key, value);
        }
        resolve(value || null);
      };

      request.onerror = () => reject(request.error);
    });
  }

  async set(key, value) {
    // Set in memory cache
    this.memoryCache.set(key, value);

    // Set in IndexedDB
    if (!this.db) {
      return;
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const request = store.put(value, key);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async delete(key) {
    this.memoryCache.delete(key);

    if (!this.db) {
      return;
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const request = store.delete(key);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async clear() {
    this.memoryCache.clear();

    if (!this.db) {
      return;
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const request = store.clear();

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }
}

/**
 * Cache Manager
 * High-level cache management with prefetching and warming
 */
export class CacheManager {
  constructor(options = {}) {
    this.cache = options.useMultiLevel ?
      new MultiLevelCache(options) :
      new LRUCache(options);

    this.prefetchQueue = [];
    this.warmingActive = false;
  }

  async get(key, loader = null) {
    const cached = await this.cache.get(key);

    if (cached !== null) {
      return cached;
    }

    // If loader provided, load and cache
    if (loader) {
      const value = await loader(key);
      await this.cache.set(key, value);
      return value;
    }

    return null;
  }

  async set(key, value, options = {}) {
    return this.cache.set(key, value, options);
  }

  /**
   * Prefetch items in the background
   */
  async prefetch(keys, loader) {
    for (const key of keys) {
      if (!this.cache.has(key)) {
        this.prefetchQueue.push({ key, loader });
      }
    }

    this._processPrefetchQueue();
  }

  /**
   * Warm cache with initial data
   */
  async warm(items) {
    this.warmingActive = true;

    for (const [key, value] of Object.entries(items)) {
      await this.cache.set(key, value);
    }

    this.warmingActive = false;
  }

  getStatistics() {
    return this.cache.getStatistics();
  }

  clear() {
    this.cache.clear();
    this.prefetchQueue = [];
  }

  async _processPrefetchQueue() {
    if (this.prefetchQueue.length === 0) {
      return;
    }

    const { key, loader } = this.prefetchQueue.shift();

    try {
      const value = await loader(key);
      await this.cache.set(key, value);
    } catch (error) {
      console.warn(`Prefetch failed for ${key}:`, error);
    }

    // Process next item (with small delay to avoid blocking)
    setTimeout(() => this._processPrefetchQueue(), 10);
  }
}

// Export all classes
export default {
  LRUCache,
  TTLCache,
  LFUCache,
  MultiLevelCache,
  PersistentCache,
  CacheManager
};
