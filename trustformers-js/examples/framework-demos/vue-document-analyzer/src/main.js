import { createApp } from 'vue'
import App from './App.vue'
import './style.css'

// Error handling for initialization
const initializeApp = async () => {
  try {
    // Check for WebAssembly support
    if (!window.WebAssembly) {
      throw new Error('WebAssembly is not supported in this browser')
    }

    // Pre-load TrustformeRS if possible
    console.log('Initializing TrustformeRS Vue Document Analyzer...')
    
    // Create and mount the Vue app
    const app = createApp(App)
    
    // Global error handler
    app.config.errorHandler = (err, instance, info) => {
      console.error('Vue application error:', err)
      console.error('Component instance:', instance)
      console.error('Error info:', info)
      
      // You could send this to an error reporting service
      if (window.errorReporting) {
        window.errorReporting.report(err, { component: instance, info })
      }
    }

    // Global properties (if needed)
    app.config.globalProperties.$appVersion = '1.0.0'
    app.config.globalProperties.$buildTime = import.meta.env.VITE_BUILD_TIME || new Date().toISOString()

    // Performance mark for initialization
    if (window.performance && window.performance.mark) {
      performance.mark('vue-app-init-start')
    }

    // Mount the app
    const mountedApp = app.mount('#app')

    // Performance measurement
    if (window.performance && window.performance.mark && window.performance.measure) {
      performance.mark('vue-app-init-end')
      performance.measure('vue-app-initialization', 'vue-app-init-start', 'vue-app-init-end')
      
      const measurement = performance.getEntriesByName('vue-app-initialization')[0]
      console.log(`Vue app initialized in ${measurement.duration.toFixed(2)}ms`)
    }

    console.log('Vue application successfully mounted')
    return mountedApp

  } catch (error) {
    console.error('Failed to initialize Vue application:', error)
    
    // Render error fallback
    const errorApp = createApp({
      template: `
        <div class="min-h-screen bg-red-50 flex items-center justify-center p-4">
          <div class="bg-white rounded-lg shadow-lg p-6 max-w-md w-full">
            <div class="text-center">
              <div class="text-red-500 text-6xl mb-4">⚠️</div>
              <h1 class="text-xl font-bold text-gray-800 mb-2">
                Application Failed to Load
              </h1>
              <p class="text-gray-600 mb-4">
                The Vue document analyzer could not be initialized. 
                {{ errorMessage }}
              </p>
              <button 
                @click="retry"
                class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors mr-2"
              >
                Retry
              </button>
              <button 
                @click="showDetails = !showDetails"
                class="bg-gray-200 text-gray-700 px-4 py-2 rounded hover:bg-gray-300 transition-colors"
              >
                {{ showDetails ? 'Hide' : 'Show' }} Details
              </button>
              <div v-if="showDetails" class="mt-4 text-left">
                <pre class="text-xs bg-gray-100 p-2 rounded overflow-auto max-h-32">{{ errorDetails }}</pre>
              </div>
            </div>
          </div>
        </div>
      `,
      data() {
        return {
          showDetails: false,
          errorMessage: getErrorMessage(error),
          errorDetails: error.stack || error.toString()
        }
      },
      methods: {
        retry() {
          window.location.reload()
        }
      }
    })
    
    errorApp.mount('#app')
  }
}

// Helper function to get user-friendly error messages
const getErrorMessage = (error) => {
  if (error.message.includes('WebAssembly')) {
    return 'Your browser does not support WebAssembly. Please update to a modern browser version.'
  }
  if (error.message.includes('TrustformeRS')) {
    return 'Failed to load machine learning models. Please check your internet connection.'
  }
  if (error.message.includes('network') || error.message.includes('fetch')) {
    return 'Network error occurred. Please check your internet connection and try again.'
  }
  return 'An unexpected error occurred. Please try refreshing the page.'
}

// Browser compatibility checks
const checkBrowserCompatibility = () => {
  const issues = []
  
  // Check for essential features
  if (!window.fetch) {
    issues.push('Fetch API not supported')
  }
  
  if (!window.Promise) {
    issues.push('Promises not supported')
  }
  
  if (!window.WebAssembly) {
    issues.push('WebAssembly not supported')
  }
  
  if (!window.Worker) {
    issues.push('Web Workers not supported')
  }
  
  // Check for Vue 3 compatibility
  if (!window.Proxy) {
    issues.push('Proxy not supported (required for Vue 3 reactivity)')
  }
  
  return issues
}

// Development helpers
if (import.meta.env.DEV) {
  // Enable Vue devtools in development
  window.__VUE_DEVTOOLS_GLOBAL_HOOK__ = window.__VUE_DEVTOOLS_GLOBAL_HOOK__ || {}
  
  // Log performance information
  window.addEventListener('load', () => {
    setTimeout(() => {
      const perfData = performance.getEntriesByType('navigation')[0]
      console.log('Development Performance Stats:')
      console.log(`- DOM Content Loaded: ${perfData.domContentLoadedEventEnd - perfData.fetchStart}ms`)
      console.log(`- Page Load Complete: ${perfData.loadEventEnd - perfData.fetchStart}ms`)
      console.log(`- First Paint: ${performance.getEntriesByType('paint').find(p => p.name === 'first-paint')?.startTime || 'N/A'}ms`)
    }, 100)
  })
}

// Check browser compatibility before initializing
const compatibilityIssues = checkBrowserCompatibility()
if (compatibilityIssues.length > 0) {
  console.warn('Browser compatibility issues detected:', compatibilityIssues)
  
  // Show compatibility warning but still try to initialize
  const warning = document.createElement('div')
  warning.innerHTML = `
    <div style="background: #fef3cd; border: 1px solid #ffeaa7; padding: 12px; margin-bottom: 20px; border-radius: 4px;">
      <strong>Browser Compatibility Warning:</strong><br>
      ${compatibilityIssues.join(', ')}<br>
      <small>The application may not work correctly. Please update your browser.</small>
    </div>
  `
  document.body.insertBefore(warning, document.getElementById('app'))
}

// Initialize the application
initializeApp().catch(error => {
  console.error('Critical application initialization error:', error)
})