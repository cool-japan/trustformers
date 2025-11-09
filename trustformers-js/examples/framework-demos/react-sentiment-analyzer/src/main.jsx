import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css';

// Initialize TrustformeRS before mounting the app
async function initializeApp() {
  try {
    // Import TrustformeRS and initialize
    const { TrustformersReact } = await import('@trustformers/js/frameworks');
    
    // Initialize the React integration
    await TrustformersReact.initialize({
      // Configuration options
      wasmPath: '/wasm/',
      modelCache: true,
      performanceOptimizations: true,
      debug: process.env.NODE_ENV === 'development'
    });

    console.log('TrustformeRS React integration initialized successfully');
    
    // Mount the React app
    ReactDOM.createRoot(document.getElementById('root')).render(
      <React.StrictMode>
        <App />
      </React.StrictMode>,
    );
    
  } catch (error) {
    console.error('Failed to initialize TrustformeRS:', error);
    
    // Render error fallback
    ReactDOM.createRoot(document.getElementById('root')).render(
      <div className="min-h-screen bg-red-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-lg p-6 max-w-md w-full">
          <div className="text-center">
            <div className="text-red-500 text-6xl mb-4">⚠️</div>
            <h1 className="text-xl font-bold text-gray-800 mb-2">
              Initialization Failed
            </h1>
            <p className="text-gray-600 mb-4">
              Failed to initialize TrustformeRS. Please check your browser compatibility and try again.
            </p>
            <button 
              onClick={() => window.location.reload()}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors"
            >
              Retry
            </button>
            <details className="mt-4 text-left">
              <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
                Show technical details
              </summary>
              <pre className="mt-2 text-xs bg-gray-100 p-2 rounded overflow-auto">
                {error.message || error.toString()}
              </pre>
            </details>
          </div>
        </div>
      </div>
    );
  }
}

// Check for WebAssembly support
if (!window.WebAssembly) {
  ReactDOM.createRoot(document.getElementById('root')).render(
    <div className="min-h-screen bg-yellow-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-lg p-6 max-w-md w-full text-center">
        <div className="text-yellow-500 text-6xl mb-4">⚠️</div>
        <h1 className="text-xl font-bold text-gray-800 mb-2">
          WebAssembly Not Supported
        </h1>
        <p className="text-gray-600 mb-4">
          Your browser doesn't support WebAssembly, which is required for TrustformeRS.
          Please update your browser to a modern version.
        </p>
        <a 
          href="https://webassembly.org/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-500 hover:text-blue-600 underline"
        >
          Learn more about WebAssembly
        </a>
      </div>
    </div>
  );
} else {
  // Initialize the app
  initializeApp();
}