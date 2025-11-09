/**
 * Service Worker for TrustformeRS Progressive Web App
 * Handles offline caching, background sync, and push notifications
 */

const CACHE_NAME = 'trustformers-v1';
const RUNTIME_CACHE = 'trustformers-runtime-v1';
const MODEL_CACHE = 'trustformers-models-v1';
const WASM_CACHE = 'trustformers-wasm-v1';

// Core files to cache immediately
const CORE_FILES = [
  '/',
  '/index.html',
  '/dist/trustformers.esm.min.js',
  '/dist/trustformers.umd.min.js',
  '/dist/trustformers_wasm_bg.wasm',
  '/pkg/trustformers_wasm.js',
  '/pkg/trustformers_wasm_bg.wasm',
];

// Runtime caching patterns
const RUNTIME_PATTERNS = {
  models: /\/models\/.*\.(bin|safetensors|onnx)$/,
  config: /\/config\.json$/,
  tokenizer: /\/tokenizer.*\.json$/,
  vocab: /\/vocab\.txt$/,
};

// Background sync tags
const SYNC_TAGS = {
  MODEL_DOWNLOAD: 'model-download',
  INFERENCE_RESULT: 'inference-result',
  TELEMETRY: 'telemetry',
};

/**
 * Install Event
 * Pre-cache core files
 */
self.addEventListener('install', event => {
  console.warn('[SW] Installing service worker...');

  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then(cache => {
        console.warn('[SW] Pre-caching core files');
        return cache.addAll(CORE_FILES);
      })
      .then(() => {
        console.warn('[SW] Core files cached successfully');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('[SW] Failed to cache core files:', error);
      })
  );
});

/**
 * Activate Event
 * Clean up old caches
 */
self.addEventListener('activate', event => {
  console.warn('[SW] Activating service worker...');

  event.waitUntil(
    Promise.all([
      // Clean up old caches
      caches.keys().then(cacheNames =>
        Promise.all(
          cacheNames.map(cacheName => {
            if (
              cacheName !== CACHE_NAME &&
              cacheName !== RUNTIME_CACHE &&
              cacheName !== MODEL_CACHE &&
              cacheName !== WASM_CACHE
            ) {
              console.warn('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        )
      ),
      // Take control of all pages
      self.clients.claim(),
    ])
  );
});

/**
 * Fetch Event
 * Implement caching strategies
 */
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Handle different resource types
  if (RUNTIME_PATTERNS.models.test(url.pathname)) {
    // Model files: Cache First with Network Fallback
    event.respondWith(handleModelRequest(request));
  } else if (
    RUNTIME_PATTERNS.config.test(url.pathname) ||
    RUNTIME_PATTERNS.tokenizer.test(url.pathname) ||
    RUNTIME_PATTERNS.vocab.test(url.pathname)
  ) {
    // Config files: Network First with Cache Fallback
    event.respondWith(handleConfigRequest(request));
  } else if (url.pathname.endsWith('.wasm')) {
    // WASM files: Cache First
    event.respondWith(handleWasmRequest(request));
  } else if (url.pathname.startsWith('/api/')) {
    // API requests: Network Only with Background Sync
    event.respondWith(handleApiRequest(request));
  } else {
    // Static assets: Cache First with Network Fallback
    event.respondWith(handleStaticRequest(request));
  }
});

/**
 * Handle model file requests
 * Cache First strategy for large model files
 */
async function handleModelRequest(request) {
  try {
    const cache = await caches.open(MODEL_CACHE);
    const cachedResponse = await cache.match(request);

    if (cachedResponse) {
      console.warn('[SW] Serving model from cache:', request.url);

      // Update in background if needed
      fetchAndCache(request, cache).catch(console.error);

      return cachedResponse;
    }

    console.warn('[SW] Fetching model from network:', request.url);
    const response = await fetch(request);

    if (response.ok) {
      await cache.put(request, response.clone());
      console.warn('[SW] Model cached:', request.url);
    }

    return response;
  } catch (error) {
    console.error('[SW] Model request failed:', error);
    return new Response('Model unavailable offline', {
      status: 503,
      statusText: 'Service Unavailable',
    });
  }
}

/**
 * Handle config file requests
 * Network First strategy for frequently updated config
 */
async function handleConfigRequest(request) {
  try {
    const response = await fetch(request);

    if (response.ok) {
      const cache = await caches.open(RUNTIME_CACHE);
      await cache.put(request, response.clone());
      console.warn('[SW] Config updated:', request.url);
    }

    return response;
  } catch (error) {
    console.warn('[SW] Network failed, trying cache:', request.url);

    const cache = await caches.open(RUNTIME_CACHE);
    const cachedResponse = await cache.match(request);

    if (cachedResponse) {
      return cachedResponse;
    }

    return new Response('Config unavailable offline', {
      status: 503,
      statusText: 'Service Unavailable',
    });
  }
}

/**
 * Handle WASM file requests
 * Cache First strategy for WASM modules
 */
async function handleWasmRequest(request) {
  try {
    const cache = await caches.open(WASM_CACHE);
    const cachedResponse = await cache.match(request);

    if (cachedResponse) {
      console.warn('[SW] Serving WASM from cache:', request.url);
      return cachedResponse;
    }

    const response = await fetch(request);

    if (response.ok) {
      await cache.put(request, response.clone());
      console.warn('[SW] WASM cached:', request.url);
    }

    return response;
  } catch (error) {
    console.error('[SW] WASM request failed:', error);
    return new Response('WASM module unavailable offline', {
      status: 503,
      statusText: 'Service Unavailable',
    });
  }
}

/**
 * Handle API requests
 * Network Only with Background Sync fallback
 */
async function handleApiRequest(request) {
  try {
    return await fetch(request);
  } catch (error) {
    console.warn('[SW] API request failed, scheduling background sync');

    // Schedule background sync for retry
    await self.registration.sync.register(SYNC_TAGS.INFERENCE_RESULT);

    return new Response(
      JSON.stringify({
        error: 'Request scheduled for background sync',
        syncTag: SYNC_TAGS.INFERENCE_RESULT,
      }),
      {
        status: 202,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

/**
 * Handle static asset requests
 * Cache First with Network Fallback
 */
async function handleStaticRequest(request) {
  try {
    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(request);

    if (cachedResponse) {
      return cachedResponse;
    }

    const response = await fetch(request);

    if (response.ok && response.status < 400) {
      await cache.put(request, response.clone());
    }

    return response;
  } catch (error) {
    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(request);

    if (cachedResponse) {
      return cachedResponse;
    }

    return new Response('Resource unavailable offline', {
      status: 503,
      statusText: 'Service Unavailable',
    });
  }
}

/**
 * Fetch and cache helper function
 */
async function fetchAndCache(request, cache) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      await cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.error('[SW] Background fetch failed:', error);
  }
}

/**
 * Background Sync Event
 * Handle deferred operations
 */
self.addEventListener('sync', event => {
  console.warn('[SW] Background sync triggered:', event.tag);

  switch (event.tag) {
    case SYNC_TAGS.MODEL_DOWNLOAD:
      event.waitUntil(handleModelDownloadSync());
      break;
    case SYNC_TAGS.INFERENCE_RESULT:
      event.waitUntil(handleInferenceResultSync());
      break;
    case SYNC_TAGS.TELEMETRY:
      event.waitUntil(handleTelemetrySync());
      break;
    default:
      console.warn('[SW] Unknown sync tag:', event.tag);
  }
});

/**
 * Handle model download background sync
 */
async function handleModelDownloadSync() {
  try {
    console.warn('[SW] Processing model download sync');

    // Get pending downloads from IndexedDB
    const pendingDownloads = await getPendingDownloads();

    for (const download of pendingDownloads) {
      try {
        await downloadModel(download);
        await removePendingDownload(download.id);

        // Notify clients
        await notifyClients({
          type: 'MODEL_DOWNLOADED',
          data: download,
        });
      } catch (error) {
        console.error('[SW] Model download failed:', error);
      }
    }
  } catch (error) {
    console.error('[SW] Model download sync failed:', error);
  }
}

/**
 * Handle inference result background sync
 */
async function handleInferenceResultSync() {
  try {
    console.warn('[SW] Processing inference result sync');

    // Get pending inference requests from IndexedDB
    const pendingRequests = await getPendingInferenceRequests();

    for (const request of pendingRequests) {
      try {
        const result = await processInferenceRequest(request);
        await removePendingInferenceRequest(request.id);

        // Notify clients
        await notifyClients({
          type: 'INFERENCE_COMPLETED',
          data: { request, result },
        });
      } catch (error) {
        console.error('[SW] Inference request failed:', error);
      }
    }
  } catch (error) {
    console.error('[SW] Inference result sync failed:', error);
  }
}

/**
 * Handle telemetry background sync
 */
async function handleTelemetrySync() {
  try {
    console.warn('[SW] Processing telemetry sync');

    const telemetryData = await getPendingTelemetry();

    if (telemetryData.length > 0) {
      await fetch('/api/telemetry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ events: telemetryData }),
      });

      await clearPendingTelemetry();
      console.warn('[SW] Telemetry data sent');
    }
  } catch (error) {
    console.error('[SW] Telemetry sync failed:', error);
  }
}

/**
 * Push Event
 * Handle push notifications
 */
self.addEventListener('push', event => {
  console.warn('[SW] Push notification received');

  const options = {
    body: 'Your TrustformeRS model is ready',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    vibrate: [200, 100, 200],
    data: { url: '/' },
    actions: [
      {
        action: 'open',
        title: 'Open App',
        icon: '/icons/action-open.png',
      },
      {
        action: 'dismiss',
        title: 'Dismiss',
        icon: '/icons/action-dismiss.png',
      },
    ],
  };

  if (event.data) {
    try {
      const payload = event.data.json();
      options.title = payload.title || 'TrustformeRS';
      options.body = payload.body || options.body;
      options.icon = payload.icon || options.icon;
      options.data = payload.data || options.data;
    } catch (error) {
      console.error('[SW] Invalid push payload:', error);
      options.title = 'TrustformeRS';
    }
  } else {
    options.title = 'TrustformeRS';
  }

  event.waitUntil(self.registration.showNotification(options.title, options));
});

/**
 * Notification Click Event
 * Handle notification interactions
 */
self.addEventListener('notificationclick', event => {
  console.warn('[SW] Notification clicked:', event.action);

  event.notification.close();

  if (event.action === 'dismiss') {
    return;
  }

  const urlToOpen = event.notification.data?.url || '/';

  event.waitUntil(
    self.clients.matchAll({ type: 'window' }).then(clients => {
      // Check if app is already open
      for (const client of clients) {
        if (client.url === urlToOpen && 'focus' in client) {
          return client.focus();
        }
      }

      // Open new window
      if (self.clients.openWindow) {
        return self.clients.openWindow(urlToOpen);
      }
    })
  );
});

/**
 * Message Event
 * Handle messages from main thread
 */
self.addEventListener('message', event => {
  console.warn('[SW] Message received:', event.data);

  const { type, data } = event.data;

  switch (type) {
    case 'CACHE_MODEL':
      event.waitUntil(cacheModel(data));
      break;
    case 'CLEAR_CACHE':
      event.waitUntil(clearCache(data));
      break;
    case 'GET_CACHE_STATUS':
      event.waitUntil(
        getCacheStatus().then(status => {
          event.ports[0].postMessage(status);
        })
      );
      break;
    case 'SCHEDULE_DOWNLOAD':
      event.waitUntil(scheduleModelDownload(data));
      break;
    default:
      console.warn('[SW] Unknown message type:', type);
  }
});

/**
 * Cache model manually
 */
async function cacheModel(modelData) {
  try {
    const cache = await caches.open(MODEL_CACHE);
    const { url, name } = modelData;

    const response = await fetch(url);
    if (response.ok) {
      await cache.put(url, response);
      console.warn('[SW] Model cached manually:', name);

      await notifyClients({
        type: 'MODEL_CACHED',
        data: { name, url },
      });
    }
  } catch (error) {
    console.error('[SW] Manual model caching failed:', error);
  }
}

/**
 * Clear specific cache
 */
async function clearCache(cacheType) {
  try {
    let cacheName;
    switch (cacheType) {
      case 'models':
        cacheName = MODEL_CACHE;
        break;
      case 'runtime':
        cacheName = RUNTIME_CACHE;
        break;
      case 'wasm':
        cacheName = WASM_CACHE;
        break;
      case 'all':
        await Promise.all([
          caches.delete(CACHE_NAME),
          caches.delete(RUNTIME_CACHE),
          caches.delete(MODEL_CACHE),
          caches.delete(WASM_CACHE),
        ]);
        return;
      default:
        cacheName = CACHE_NAME;
    }

    await caches.delete(cacheName);
    console.warn('[SW] Cache cleared:', cacheName);
  } catch (error) {
    console.error('[SW] Cache clearing failed:', error);
  }
}

/**
 * Get cache status
 */
async function getCacheStatus() {
  try {
    const cacheNames = await caches.keys();
    const status = {};

    for (const cacheName of cacheNames) {
      const cache = await caches.open(cacheName);
      const keys = await cache.keys();
      status[cacheName] = {
        count: keys.length,
        size: await calculateCacheSize(cache, keys),
      };
    }

    return status;
  } catch (error) {
    console.error('[SW] Cache status check failed:', error);
    return {};
  }
}

/**
 * Calculate cache size
 */
async function calculateCacheSize(cache, keys) {
  let totalSize = 0;

  for (const key of keys.slice(0, 10)) {
    // Limit to avoid performance issues
    try {
      const response = await cache.match(key);
      if (response) {
        const blob = await response.blob();
        totalSize += blob.size;
      }
    } catch (error) {
      // Ignore individual errors
    }
  }

  return totalSize;
}

/**
 * Notify all clients
 */
async function notifyClients(message) {
  const clients = await self.clients.matchAll();
  clients.forEach(client => client.postMessage(message));
}

/**
 * IndexedDB helper functions
 * These would integrate with the actual IndexedDB implementation
 */

async function getPendingDownloads() {
  // Placeholder - integrate with actual IndexedDB
  return [];
}

async function removePendingDownload(id) {
  // Placeholder - integrate with actual IndexedDB
}

async function downloadModel(download) {
  // Placeholder - integrate with actual download logic
}

async function getPendingInferenceRequests() {
  // Placeholder - integrate with actual IndexedDB
  return [];
}

async function removePendingInferenceRequest(id) {
  // Placeholder - integrate with actual IndexedDB
}

async function processInferenceRequest(request) {
  // Placeholder - integrate with actual inference logic
}

async function getPendingTelemetry() {
  // Placeholder - integrate with actual IndexedDB
  return [];
}

async function clearPendingTelemetry() {
  // Placeholder - integrate with actual IndexedDB
}

async function scheduleModelDownload(data) {
  // Placeholder - integrate with actual scheduling logic
}
