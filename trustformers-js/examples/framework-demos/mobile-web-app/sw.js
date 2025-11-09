/**
 * TrustformeRS Mobile Web App Service Worker
 * Provides offline functionality, caching, and background sync
 */

const CACHE_NAME = 'trustformers-mobile-v1.0.0';
const RUNTIME_CACHE = 'trustformers-runtime-v1.0.0';
const OFFLINE_URL = './offline.html';

// Core app files to cache immediately
const CORE_FILES = [
  './',
  './index.html',
  './css/styles.css',
  './js/main.js',
  './manifest.json',
  './icons/icon-192x192.png',
  './icons/icon-512x512.png',
  OFFLINE_URL
];

// API patterns that should be cached
const API_CACHE_PATTERNS = [
  /\/api\/analyze/,
  /\/api\/generate/,
  /\/api\/models/
];

// Files that should never be cached
const NEVER_CACHE_PATTERNS = [
  /\/api\/auth/,
  /\/api\/upload/,
  /\.hot-update\./
];

// Maximum age for different types of cached content (in milliseconds)
const CACHE_MAX_AGE = {
  static: 24 * 60 * 60 * 1000,     // 24 hours
  api: 5 * 60 * 1000,              // 5 minutes  
  images: 7 * 24 * 60 * 60 * 1000, // 7 days
  fonts: 30 * 24 * 60 * 60 * 1000  // 30 days
};

// Install event - cache core files
self.addEventListener('install', event => {
  console.log('📦 Service Worker: Installing...');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('📦 Service Worker: Caching core files');
        return cache.addAll(CORE_FILES);
      })
      .then(() => {
        console.log('✅ Service Worker: Core files cached');
        return self.skipWaiting(); // Force activation
      })
      .catch(error => {
        console.error('❌ Service Worker: Install failed', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('🚀 Service Worker: Activating...');
  
  event.waitUntil(
    Promise.all([
      // Clean up old caches
      caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames
            .filter(cacheName => 
              cacheName.startsWith('trustformers-') && 
              cacheName !== CACHE_NAME && 
              cacheName !== RUNTIME_CACHE
            )
            .map(cacheName => {
              console.log('🗑️ Service Worker: Deleting old cache', cacheName);
              return caches.delete(cacheName);
            })
        );
      }),
      
      // Take control immediately
      self.clients.claim()
    ]).then(() => {
      console.log('✅ Service Worker: Activated successfully');
    })
  );
});

// Fetch event - handle network requests
self.addEventListener('fetch', event => {
  const request = event.request;
  const url = new URL(request.url);
  
  // Skip non-GET requests and chrome-extension requests
  if (request.method !== 'GET' || url.protocol === 'chrome-extension:') {
    return;
  }
  
  // Skip requests that should never be cached
  if (NEVER_CACHE_PATTERNS.some(pattern => pattern.test(url.pathname))) {
    return;
  }
  
  event.respondWith(handleRequest(request));
});

// Background sync for offline actions
self.addEventListener('sync', event => {
  console.log('🔄 Service Worker: Background sync triggered', event.tag);
  
  if (event.tag === 'background-sync-conversations') {
    event.waitUntil(syncConversations());
  }
  
  if (event.tag === 'background-sync-analytics') {
    event.waitUntil(syncAnalytics());
  }
});

// Push notifications
self.addEventListener('push', event => {
  console.log('🔔 Service Worker: Push notification received');
  
  const options = {
    body: event.data ? event.data.text() : 'New message from TrustformeRS AI',
    icon: './icons/icon-192x192.png',
    badge: './icons/badge-72x72.png',
    vibrate: [200, 100, 200],
    tag: 'trustformers-notification',
    actions: [
      {
        action: 'open',
        title: 'Open App',
        icon: './icons/action-open.png'
      },
      {
        action: 'dismiss',
        title: 'Dismiss',
        icon: './icons/action-dismiss.png'
      }
    ],
    data: {
      url: './',
      timestamp: Date.now()
    }
  };
  
  event.waitUntil(
    self.registration.showNotification('TrustformeRS AI', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', event => {
  console.log('🔔 Service Worker: Notification clicked', event.action);
  
  event.notification.close();
  
  if (event.action === 'open' || !event.action) {
    event.waitUntil(
      clients.matchAll({ type: 'window' }).then(clientList => {
        // Focus existing window if available
        for (const client of clientList) {
          if (client.url.includes('trustformers') && 'focus' in client) {
            return client.focus();
          }
        }
        
        // Open new window
        if (clients.openWindow) {
          return clients.openWindow('./');
        }
      })
    );
  }
});

// Message handler for communication with main app
self.addEventListener('message', event => {
  console.log('💬 Service Worker: Message received', event.data);
  
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
    return;
  }
  
  if (event.data && event.data.type === 'CACHE_URLS') {
    event.waitUntil(
      cacheUrls(event.data.urls).then(() => {
        event.ports[0].postMessage({ success: true });
      }).catch(error => {
        event.ports[0].postMessage({ success: false, error: error.message });
      })
    );
    return;
  }
  
  if (event.data && event.data.type === 'GET_CACHE_SIZE') {
    event.waitUntil(
      getCacheSize().then(size => {
        event.ports[0].postMessage({ size });
      })
    );
    return;
  }
});

// Main request handler
async function handleRequest(request) {
  const url = new URL(request.url);
  
  // Handle different types of requests
  if (url.origin === location.origin) {
    // Same origin - use appropriate caching strategy
    return handleSameOriginRequest(request);
  } else {
    // Cross-origin - cache with caution
    return handleCrossOriginRequest(request);
  }
}

// Handle same-origin requests
async function handleSameOriginRequest(request) {
  const url = new URL(request.url);
  
  // API requests - use network first with fallback
  if (url.pathname.startsWith('/api/')) {
    return networkFirstStrategy(request, RUNTIME_CACHE);
  }
  
  // Static assets - use cache first
  if (isStaticAsset(url.pathname)) {
    return cacheFirstStrategy(request, CACHE_NAME);
  }
  
  // HTML pages - use network first with offline fallback
  if (request.headers.get('accept')?.includes('text/html')) {
    return htmlNetworkFirstStrategy(request);
  }
  
  // Default - use cache first
  return cacheFirstStrategy(request, CACHE_NAME);
}

// Handle cross-origin requests
async function handleCrossOriginRequest(request) {
  const url = new URL(request.url);
  
  // CDN resources - use cache first
  if (isCDNResource(url.hostname)) {
    return cacheFirstStrategy(request, CACHE_NAME);
  }
  
  // Default - network only for security
  return fetch(request);
}

// Network first strategy (with cache fallback)
async function networkFirstStrategy(request, cacheName) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      // Cache successful responses
      const cache = await caches.open(cacheName);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('🌐 Service Worker: Network failed, trying cache', request.url);
    
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline response for API calls
    return new Response(
      JSON.stringify({
        error: 'Network unavailable',
        offline: true,
        message: 'This request requires an internet connection'
      }),
      {
        status: 503,
        statusText: 'Service Unavailable',
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

// Cache first strategy (with network fallback)
async function cacheFirstStrategy(request, cacheName) {
  const cachedResponse = await caches.match(request);
  
  if (cachedResponse) {
    // Check if cache is still fresh
    const cacheDate = cachedResponse.headers.get('date');
    const maxAge = getCacheMaxAge(request.url);
    
    if (cacheDate && (Date.now() - new Date(cacheDate).getTime() < maxAge)) {
      return cachedResponse;
    }
  }
  
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    if (cachedResponse) {
      return cachedResponse;
    }
    
    throw error;
  }
}

// HTML network first strategy with offline fallback
async function htmlNetworkFirstStrategy(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline page
    return caches.match(OFFLINE_URL);
  }
}

// Background sync functions
async function syncConversations() {
  try {
    console.log('🔄 Service Worker: Syncing conversations');
    
    // Get pending conversations from IndexedDB
    const pendingData = await getFromIndexedDB('pendingSync');
    
    if (pendingData && pendingData.conversations) {
      // Send to server
      const response = await fetch('/api/sync/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pendingData.conversations)
      });
      
      if (response.ok) {
        // Clear pending data
        await deleteFromIndexedDB('pendingSync');
        console.log('✅ Service Worker: Conversations synced successfully');
      }
    }
  } catch (error) {
    console.error('❌ Service Worker: Conversation sync failed', error);
  }
}

async function syncAnalytics() {
  try {
    console.log('🔄 Service Worker: Syncing analytics');
    
    // Get pending analytics from IndexedDB
    const pendingAnalytics = await getFromIndexedDB('pendingAnalytics');
    
    if (pendingAnalytics && pendingAnalytics.length > 0) {
      // Send to server
      const response = await fetch('/api/analytics/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pendingAnalytics)
      });
      
      if (response.ok) {
        // Clear pending data
        await deleteFromIndexedDB('pendingAnalytics');
        console.log('✅ Service Worker: Analytics synced successfully');
      }
    }
  } catch (error) {
    console.error('❌ Service Worker: Analytics sync failed', error);
  }
}

// Utility functions
function isStaticAsset(pathname) {
  return /\.(css|js|png|jpg|jpeg|gif|svg|woff2?|ttf|ico)$/.test(pathname);
}

function isCDNResource(hostname) {
  return [
    'fonts.googleapis.com',
    'fonts.gstatic.com',
    'cdn.jsdelivr.net',
    'unpkg.com'
  ].includes(hostname);
}

function getCacheMaxAge(url) {
  if (url.includes('/api/')) return CACHE_MAX_AGE.api;
  if (/\.(png|jpg|jpeg|gif|svg)$/.test(url)) return CACHE_MAX_AGE.images;
  if (/\.(woff2?|ttf)$/.test(url)) return CACHE_MAX_AGE.fonts;
  return CACHE_MAX_AGE.static;
}

async function cacheUrls(urls) {
  const cache = await caches.open(CACHE_NAME);
  return cache.addAll(urls);
}

async function getCacheSize() {
  const cacheNames = await caches.keys();
  let totalSize = 0;
  
  for (const cacheName of cacheNames) {
    if (cacheName.startsWith('trustformers-')) {
      const cache = await caches.open(cacheName);
      const requests = await cache.keys();
      
      for (const request of requests) {
        const response = await cache.match(request);
        if (response) {
          const blob = await response.blob();
          totalSize += blob.size;
        }
      }
    }
  }
  
  return totalSize;
}

// IndexedDB helpers for background sync
async function getFromIndexedDB(key) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('TrustformersMobile', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['sync'], 'readonly');
      const store = transaction.objectStore('sync');
      const getRequest = store.get(key);
      
      getRequest.onsuccess = () => resolve(getRequest.result);
      getRequest.onerror = () => reject(getRequest.error);
    };
    
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('sync')) {
        db.createObjectStore('sync');
      }
    };
  });
}

async function deleteFromIndexedDB(key) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('TrustformersMobile', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['sync'], 'readwrite');
      const store = transaction.objectStore('sync');
      const deleteRequest = store.delete(key);
      
      deleteRequest.onsuccess = () => resolve();
      deleteRequest.onerror = () => reject(deleteRequest.error);
    };
  });
}

// Error handling for uncaught errors
self.addEventListener('error', event => {
  console.error('❌ Service Worker: Uncaught error', event.error);
});

self.addEventListener('unhandledrejection', event => {
  console.error('❌ Service Worker: Unhandled promise rejection', event.reason);
});

console.log('🤖 TrustformeRS Mobile Service Worker loaded successfully');