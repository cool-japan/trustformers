import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react({
      // Enable Fast Refresh for better development experience
      fastRefresh: true,
      // Include JSX runtime automatically
      jsxRuntime: 'automatic'
    })
  ],
  
  // Development server configuration
  server: {
    port: 3000,
    host: true,
    open: true,
    cors: true,
    // Enable HMR for better development experience
    hmr: {
      overlay: true
    }
  },

  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: true,
    minify: 'terser',
    target: ['es2020', 'chrome80', 'firefox80', 'safari14'],
    
    // Chunk splitting for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          // Separate vendor chunks for better caching
          vendor: ['react', 'react-dom'],
          trustformers: ['@trustformers/js'],
          icons: ['lucide-react']
        }
      }
    },
    
    // Optimize bundle size
    chunkSizeWarningLimit: 1000,
    
    // Asset optimization
    assetsInlineLimit: 4096
  },

  // Path resolution
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@assets': resolve(__dirname, 'src/assets')
    }
  },

  // CSS configuration
  css: {
    postcss: {
      plugins: [
        // Add autoprefixer for better browser compatibility
        require('autoprefixer')
      ]
    },
    modules: {
      // Enable CSS modules for component-scoped styles
      localsConvention: 'camelCase'
    }
  },

  // Environment variables
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString())
  },

  // Optimization for production
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      '@trustformers/js',
      'lucide-react'
    ],
    // Exclude packages that should not be pre-bundled
    exclude: ['@trustformers/js/wasm']
  },

  // PWA configuration (if using vite-plugin-pwa)
  // pwa: {
  //   registerType: 'autoUpdate',
  //   includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'masked-icon.svg'],
  //   manifest: {
  //     name: 'TrustformeRS Sentiment Analyzer',
  //     short_name: 'Sentiment Analyzer',
  //     description: 'Real-time sentiment analysis with TrustformeRS',
  //     theme_color: '#3b82f6',
  //     icons: [
  //       {
  //         src: 'pwa-192x192.png',
  //         sizes: '192x192',
  //         type: 'image/png'
  //       }
  //     ]
  //   }
  // },

  // Worker configuration for WebAssembly
  worker: {
    format: 'es',
    plugins: []
  },

  // Enable WebAssembly support
  assetsInclude: ['**/*.wasm'],

  // Preview server (for production builds)
  preview: {
    port: 3001,
    host: true,
    cors: true
  },

  // Experimental features
  experimental: {
    // Enable build optimization
    buildAdvancedBaseOptions: true
  },

  // Base path for deployment
  base: process.env.NODE_ENV === 'production' ? '/demos/react-sentiment/' : '/',

  // Public directory
  publicDir: 'public',

  // Environment files
  envPrefix: ['VITE_', 'TRUSTFORMERS_'],

  // ESBuild configuration
  esbuild: {
    // Remove console logs in production
    drop: process.env.NODE_ENV === 'production' ? ['console', 'debugger'] : [],
    // Keep JSX for better debugging
    jsx: 'automatic'
  },

  // JSON configuration
  json: {
    namedExports: true,
    stringify: false
  },

  // Plugin configuration for different environments
  ...(process.env.NODE_ENV === 'development' && {
    // Development-specific configuration
    logLevel: 'info',
    clearScreen: false
  }),

  ...(process.env.NODE_ENV === 'production' && {
    // Production-specific configuration
    logLevel: 'warn'
  })
});