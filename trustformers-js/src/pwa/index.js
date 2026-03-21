/**
 * Progressive Web App features for TrustformeRS
 * Provides offline caching, background sync, and push notification capabilities
 */

/**
 * PWA Manager
 * Main class for managing Progressive Web App features
 */
export class PWAManager {
  constructor(options = {}) {
    this.options = {
      serviceWorkerPath: '/service-worker.js',
      enableNotifications: true,
      enableBackgroundSync: true,
      cacheStrategy: 'cache-first',
      updateInterval: 24 * 60 * 60 * 1000, // 24 hours
      ...options,
    };

    this.serviceWorker = null;
    this.notificationPermission = 'default';
    this.isOnline = navigator.onLine;
    this.updateAvailable = false;
    this.installPrompt = null;

    this._init();
  }

  /**
   * Initialize PWA features
   * @private
   */
  async _init() {
    // Register service worker
    if ('serviceWorker' in navigator) {
      await this.registerServiceWorker();
    }

    // Setup event listeners
    this._setupEventListeners();

    // Check for app updates
    this._checkForUpdates();

    // Initialize push notifications
    if (this.options.enableNotifications) {
      await this.initializeNotifications();
    }
  }

  /**
   * Register service worker
   * @returns {Promise<ServiceWorkerRegistration>}
   */
  async registerServiceWorker() {
    try {
      const registration = await navigator.serviceWorker.register(this.options.serviceWorkerPath, {
        scope: '/',
      });

      this.serviceWorker = registration;

      console.warn('ServiceWorker registered successfully');

      // Listen for service worker updates
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing;
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            this.updateAvailable = true;
            this._notifyUpdateAvailable();
          }
        });
      });

      return registration;
    } catch (error) {
      console.error('ServiceWorker registration failed:', error);
      throw error;
    }
  }

  /**
   * Setup event listeners
   * @private
   */
  _setupEventListeners() {
    // Online/offline status
    window.addEventListener('online', () => {
      this.isOnline = true;
      this._onNetworkChange('online');
    });

    window.addEventListener('offline', () => {
      this.isOnline = false;
      this._onNetworkChange('offline');
    });

    // App install prompt
    window.addEventListener('beforeinstallprompt', event => {
      event.preventDefault();
      this.installPrompt = event;
    });

    // Service worker messages
    navigator.serviceWorker?.addEventListener('message', event => {
      this._handleServiceWorkerMessage(event.data);
    });
  }

  /**
   * Handle network status changes
   * @param {string} status - 'online' or 'offline'
   * @private
   */
  _onNetworkChange(status) {
    console.warn(`Network status changed: ${status}`);

    if (status === 'online' && this.options.enableBackgroundSync) {
      this._triggerBackgroundSync();
    }

    // Dispatch custom event
    window.dispatchEvent(
      new CustomEvent('trustformers-network-change', {
        detail: { status, isOnline: this.isOnline },
      })
    );
  }

  /**
   * Handle service worker messages
   * @param {Object} message - Message from service worker
   * @private
   */
  _handleServiceWorkerMessage(message) {
    const { type, data } = message;

    switch (type) {
      case 'MODEL_DOWNLOADED':
        this._onModelDownloaded(data);
        break;
      case 'MODEL_CACHED':
        this._onModelCached(data);
        break;
      case 'INFERENCE_COMPLETED':
        this._onInferenceCompleted(data);
        break;
      default:
        console.warn('Unknown service worker message:', message);
    }
  }

  /**
   * Check for app updates
   * @private
   */
  async _checkForUpdates() {
    if (!this.serviceWorker) return;

    try {
      await this.serviceWorker.update();
    } catch (error) {
      console.error('Update check failed:', error);
    }

    // Schedule next update check
    setTimeout(() => this._checkForUpdates(), this.options.updateInterval);
  }

  /**
   * Notify about available updates
   * @private
   */
  _notifyUpdateAvailable() {
    window.dispatchEvent(
      new CustomEvent('trustformers-update-available', {
        detail: { updateAvailable: true },
      })
    );
  }

  /**
   * Initialize push notifications
   * @returns {Promise<boolean>}
   */
  async initializeNotifications() {
    if (!('Notification' in window) || !('serviceWorker' in navigator)) {
      console.warn('Push notifications not supported');
      return false;
    }

    // Check current permission
    this.notificationPermission = Notification.permission;

    if (this.notificationPermission === 'granted') {
      await this._setupPushSubscription();
      return true;
    }

    return false;
  }

  /**
   * Request notification permission
   * @returns {Promise<boolean>}
   */
  async requestNotificationPermission() {
    if (!('Notification' in window)) {
      throw new Error('Notifications not supported');
    }

    const permission = await Notification.requestPermission();
    this.notificationPermission = permission;

    if (permission === 'granted') {
      await this._setupPushSubscription();
      return true;
    }

    return false;
  }

  /**
   * Setup push subscription
   * @private
   */
  async _setupPushSubscription() {
    if (!this.serviceWorker) return;

    try {
      const subscription = await this.serviceWorker.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: this._urlBase64ToUint8Array(this.options.vapidPublicKey),
      });

      console.warn('Push subscription created:', subscription);

      // Send subscription to server
      await this._sendSubscriptionToServer(subscription);
    } catch (error) {
      console.error('Push subscription failed:', error);
    }
  }

  /**
   * Send push subscription to server
   * @param {PushSubscription} subscription
   * @private
   */
  async _sendSubscriptionToServer(subscription) {
    try {
      await fetch('/api/push-subscription', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(subscription),
      });
    } catch (error) {
      console.error('Failed to send subscription to server:', error);
    }
  }

  /**
   * Convert VAPID key to Uint8Array
   * @param {string} base64String
   * @returns {Uint8Array}
   * @private
   */
  _urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
    const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
  }

  /**
   * Cache model for offline use
   * @param {Object} modelInfo - Model information
   * @returns {Promise<void>}
   */
  async cacheModel(modelInfo) {
    if (!this.serviceWorker) {
      throw new Error('Service worker not available');
    }

    const channel = new MessageChannel();

    return new Promise((resolve, reject) => {
      channel.port1.onmessage = event => {
        if (event.data.error) {
          reject(new Error(event.data.error));
        } else {
          resolve(event.data);
        }
      };

      navigator.serviceWorker.controller?.postMessage(
        {
          type: 'CACHE_MODEL',
          data: modelInfo,
        },
        [channel.port2]
      );

      // Timeout after 30 seconds
      setTimeout(() => reject(new Error('Cache operation timeout')), 30000);
    });
  }

  /**
   * Get cache status
   * @returns {Promise<Object>}
   */
  async getCacheStatus() {
    if (!this.serviceWorker) {
      return {};
    }

    const channel = new MessageChannel();

    return new Promise(resolve => {
      channel.port1.onmessage = event => {
        resolve(event.data);
      };

      navigator.serviceWorker.controller?.postMessage(
        {
          type: 'GET_CACHE_STATUS',
        },
        [channel.port2]
      );

      // Timeout with empty result
      setTimeout(() => resolve({}), 5000);
    });
  }

  /**
   * Clear cache
   * @param {string} cacheType - Type of cache to clear
   * @returns {Promise<void>}
   */
  async clearCache(cacheType = 'all') {
    if (!this.serviceWorker) return;

    navigator.serviceWorker.controller?.postMessage({
      type: 'CLEAR_CACHE',
      data: cacheType,
    });
  }

  /**
   * Schedule model download for background sync
   * @param {Object} downloadInfo - Download information
   * @returns {Promise<void>}
   */
  async scheduleModelDownload(downloadInfo) {
    if (!this.serviceWorker) {
      throw new Error('Service worker not available');
    }

    // Store download info in IndexedDB
    await this._storeDownloadInfo(downloadInfo);

    // Register background sync
    await this.serviceWorker.sync.register('model-download');
  }

  /**
   * Store download info in IndexedDB
   * @param {Object} downloadInfo
   * @private
   */
  async _storeDownloadInfo(downloadInfo) {
    const db = await this._openIndexedDB();
    const transaction = db.transaction(['downloads'], 'readwrite');
    const store = transaction.objectStore('downloads');

    await store.add({
      ...downloadInfo,
      id: Date.now(),
      status: 'pending',
      createdAt: new Date().toISOString(),
    });
  }

  /**
   * Open IndexedDB connection
   * @returns {Promise<IDBDatabase>}
   * @private
   */
  _openIndexedDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('TrustformersDB', 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = event => {
        const db = event.target.result;

        // Create object stores
        if (!db.objectStoreNames.contains('downloads')) {
          db.createObjectStore('downloads', { keyPath: 'id' });
        }

        if (!db.objectStoreNames.contains('inference-requests')) {
          db.createObjectStore('inference-requests', { keyPath: 'id' });
        }

        if (!db.objectStoreNames.contains('telemetry')) {
          db.createObjectStore('telemetry', { keyPath: 'id', autoIncrement: true });
        }
      };
    });
  }

  /**
   * Show app install prompt
   * @returns {Promise<boolean>}
   */
  async showInstallPrompt() {
    if (!this.installPrompt) {
      return false;
    }

    try {
      this.installPrompt.prompt();
      const result = await this.installPrompt.userChoice;

      this.installPrompt = null;

      return result.outcome === 'accepted';
    } catch (error) {
      console.error('Install prompt failed:', error);
      return false;
    }
  }

  /**
   * Check if app can be installed
   * @returns {boolean}
   */
  canInstall() {
    return this.installPrompt !== null;
  }

  /**
   * Apply app update
   * @returns {Promise<void>}
   */
  async applyUpdate() {
    if (!this.updateAvailable || !this.serviceWorker) return;

    const newWorker = this.serviceWorker.waiting;
    if (newWorker) {
      newWorker.postMessage({ type: 'SKIP_WAITING' });

      // Reload the page to activate new service worker
      window.location.reload();
    }
  }

  /**
   * Trigger background sync
   * @private
   */
  async _triggerBackgroundSync() {
    if (!this.serviceWorker || !this.options.enableBackgroundSync) return;

    try {
      await this.serviceWorker.sync.register('inference-result');
      await this.serviceWorker.sync.register('telemetry');
    } catch (error) {
      console.error('Background sync registration failed:', error);
    }
  }

  /**
   * Handle model downloaded event
   * @param {Object} data
   * @private
   */
  _onModelDownloaded(data) {
    window.dispatchEvent(
      new CustomEvent('trustformers-model-downloaded', {
        detail: data,
      })
    );
  }

  /**
   * Handle model cached event
   * @param {Object} data
   * @private
   */
  _onModelCached(data) {
    window.dispatchEvent(
      new CustomEvent('trustformers-model-cached', {
        detail: data,
      })
    );
  }

  /**
   * Handle inference completed event
   * @param {Object} data
   * @private
   */
  _onInferenceCompleted(data) {
    window.dispatchEvent(
      new CustomEvent('trustformers-inference-completed', {
        detail: data,
      })
    );
  }

  /**
   * Get network status
   * @returns {Object}
   */
  getNetworkStatus() {
    return {
      isOnline: this.isOnline,
      connection: navigator.connection || null,
    };
  }

  /**
   * Get PWA status
   * @returns {Object}
   */
  getStatus() {
    return {
      serviceWorkerRegistered: !!this.serviceWorker,
      notificationPermission: this.notificationPermission,
      updateAvailable: this.updateAvailable,
      canInstall: this.canInstall(),
      isOnline: this.isOnline,
    };
  }
}

/**
 * Offline Storage Manager
 * Manages offline data storage using IndexedDB
 */
export class OfflineStorageManager {
  constructor() {
    this.db = null;
    this.dbName = 'TrustformersOfflineDB';
    this.dbVersion = 1;
  }

  /**
   * Initialize storage
   * @returns {Promise<void>}
   */
  async init() {
    this.db = await this._openDatabase();
  }

  /**
   * Open IndexedDB database
   * @returns {Promise<IDBDatabase>}
   * @private
   */
  _openDatabase() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = event => {
        const db = event.target.result;

        // Models store
        if (!db.objectStoreNames.contains('models')) {
          const modelStore = db.createObjectStore('models', { keyPath: 'id' });
          modelStore.createIndex('name', 'name', { unique: false });
          modelStore.createIndex('type', 'type', { unique: false });
        }

        // Inference results store
        if (!db.objectStoreNames.contains('inference-results')) {
          const resultsStore = db.createObjectStore('inference-results', { keyPath: 'id' });
          resultsStore.createIndex('modelId', 'modelId', { unique: false });
          resultsStore.createIndex('timestamp', 'timestamp', { unique: false });
        }

        // Cache metadata store
        if (!db.objectStoreNames.contains('cache-metadata')) {
          db.createObjectStore('cache-metadata', { keyPath: 'key' });
        }
      };
    });
  }

  /**
   * Store model metadata
   * @param {Object} modelData - Model data
   * @returns {Promise<void>}
   */
  async storeModel(modelData) {
    const transaction = this.db.transaction(['models'], 'readwrite');
    const store = transaction.objectStore('models');

    await store.put({
      ...modelData,
      id: modelData.id || Date.now(),
      storedAt: new Date().toISOString(),
    });
  }

  /**
   * Get model by ID
   * @param {string} modelId - Model ID
   * @returns {Promise<Object|null>}
   */
  async getModel(modelId) {
    const transaction = this.db.transaction(['models'], 'readonly');
    const store = transaction.objectStore('models');

    return await store.get(modelId);
  }

  /**
   * Get all models
   * @returns {Promise<Array>}
   */
  async getAllModels() {
    const transaction = this.db.transaction(['models'], 'readonly');
    const store = transaction.objectStore('models');

    return await store.getAll();
  }

  /**
   * Store inference result
   * @param {Object} result - Inference result
   * @returns {Promise<void>}
   */
  async storeInferenceResult(result) {
    const transaction = this.db.transaction(['inference-results'], 'readwrite');
    const store = transaction.objectStore('inference-results');

    await store.put({
      ...result,
      id: result.id || Date.now(),
      timestamp: new Date().toISOString(),
    });
  }

  /**
   * Get inference results by model ID
   * @param {string} modelId - Model ID
   * @returns {Promise<Array>}
   */
  async getInferenceResults(modelId) {
    const transaction = this.db.transaction(['inference-results'], 'readonly');
    const store = transaction.objectStore('inference-results');
    const index = store.index('modelId');

    return await index.getAll(modelId);
  }

  /**
   * Clear old data
   * @param {number} maxAge - Maximum age in milliseconds
   * @returns {Promise<void>}
   */
  async clearOldData(maxAge = 7 * 24 * 60 * 60 * 1000) {
    // 7 days
    const cutoffDate = new Date(Date.now() - maxAge).toISOString();

    const transaction = this.db.transaction(['inference-results'], 'readwrite');
    const store = transaction.objectStore('inference-results');
    const index = store.index('timestamp');

    const request = index.openCursor(IDBKeyRange.upperBound(cutoffDate));

    return new Promise((resolve, reject) => {
      request.onsuccess = event => {
        const cursor = event.target.result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        } else {
          resolve();
        }
      };

      request.onerror = () => reject(request.error);
    });
  }
}

/**
 * Notification Manager
 * Manages push notifications and local notifications
 */
export class NotificationManager {
  constructor(pwaManager) {
    this.pwaManager = pwaManager;
  }

  /**
   * Show local notification
   * @param {string} title - Notification title
   * @param {Object} options - Notification options
   * @returns {Promise<Notification>}
   */
  async showNotification(title, options = {}) {
    if (this.pwaManager.notificationPermission !== 'granted') {
      const granted = await this.pwaManager.requestNotificationPermission();
      if (!granted) {
        throw new Error('Notification permission denied');
      }
    }

    return new Notification(title, {
      icon: '/icons/icon-192x192.png',
      badge: '/icons/badge-72x72.png',
      ...options,
    });
  }

  /**
   * Show model ready notification
   * @param {string} modelName - Model name
   * @returns {Promise<Notification>}
   */
  async notifyModelReady(modelName) {
    return this.showNotification('Model Ready', {
      body: `${modelName} is now available for inference`,
      tag: 'model-ready',
      icon: '/icons/model-ready.png',
    });
  }

  /**
   * Show inference complete notification
   * @param {Object} result - Inference result
   * @returns {Promise<Notification>}
   */
  async notifyInferenceComplete(result) {
    return this.showNotification('Inference Complete', {
      body: 'Your model inference has finished processing',
      tag: 'inference-complete',
      icon: '/icons/inference-complete.png',
      data: result,
    });
  }

  /**
   * Show offline notification
   * @returns {Promise<Notification>}
   */
  async notifyOfflineMode() {
    return this.showNotification('Offline Mode', {
      body: 'You are now offline. Cached models are still available.',
      tag: 'offline-mode',
      icon: '/icons/offline.png',
    });
  }
}

// Global PWA instance
let globalPWAManager = null;

/**
 * Initialize PWA features
 * @param {Object} options - PWA options
 * @returns {Promise<PWAManager>}
 */
export async function initializePWA(options = {}) {
  if (!globalPWAManager) {
    globalPWAManager = new PWAManager(options);
  }
  return globalPWAManager;
}

/**
 * Get global PWA manager instance
 * @returns {PWAManager|null}
 */
export function getPWAManager() {
  return globalPWAManager;
}

export default {
  PWAManager,
  OfflineStorageManager,
  NotificationManager,
  initializePWA,
  getPWAManager,
};
