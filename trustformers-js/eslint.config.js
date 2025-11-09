import js from '@eslint/js';

export default [
  {
    name: 'trustformers-js/base',
    files: ['**/*.js', '**/*.mjs'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        // Browser globals
        window: 'readonly',
        document: 'readonly',
        console: 'readonly',
        navigator: 'readonly',
        performance: 'readonly',
        crypto: 'readonly',

        // Node.js globals
        global: 'readonly',
        process: 'readonly',
        Buffer: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        require: 'readonly',
        module: 'readonly',
        exports: 'readonly',

        // Browser APIs
        fetch: 'readonly',
        URL: 'readonly',
        URLSearchParams: 'readonly',
        FileReader: 'readonly',
        FormData: 'readonly',
        Blob: 'readonly',
        File: 'readonly',
        localStorage: 'readonly',
        sessionStorage: 'readonly',
        indexedDB: 'readonly',
        WebSocket: 'readonly',
        XMLHttpRequest: 'readonly',
        PerformanceObserver: 'readonly',
        IntersectionObserver: 'readonly',
        MutationObserver: 'readonly',
        ResizeObserver: 'readonly',

        // Timers
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        setImmediate: 'readonly',
        clearImmediate: 'readonly',
        requestAnimationFrame: 'readonly',
        cancelAnimationFrame: 'readonly',

        // Framework globals
        React: 'readonly',
        Vue: 'readonly',
        Angular: 'readonly',
        Svelte: 'readonly',

        // Worker globals
        self: 'readonly',
        importScripts: 'readonly',
        postMessage: 'readonly',

        // Other common globals
        CustomEvent: 'readonly',
        Event: 'readonly',
        ErrorEvent: 'readonly',
        MessageEvent: 'readonly',
        ProgressEvent: 'readonly',

        // Additional Browser APIs
        Notification: 'readonly',
        MessageChannel: 'readonly',
        MessagePort: 'readonly',
        IDBKeyRange: 'readonly',
        caches: 'readonly',
        clients: 'readonly',
        registration: 'readonly',
        TextEncoder: 'readonly',
        TextDecoder: 'readonly',
        btoa: 'readonly',
        atob: 'readonly',

        // Streaming APIs
        ReadableStream: 'readonly',
        WritableStream: 'readonly',
        TransformStream: 'readonly',
        BroadcastChannel: 'readonly',

        // Custom utilities
        RandomUtils: 'readonly',

        // Service Worker globals
        skipWaiting: 'readonly',
      },
    },
    rules: {
      ...js.configs.recommended.rules,

      // ES6+ Rules
      'no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      'prefer-const': 'error',
      'no-var': 'error',
      'object-shorthand': 'error',
      'prefer-arrow-callback': 'error',
      'arrow-body-style': ['error', 'as-needed'],
      'prefer-template': 'error',
      'template-curly-spacing': 'error',
      'prefer-destructuring': ['error', {
        array: true,
        object: true
      }, {
        enforceForRenamedProperties: false
      }],

      // Class rules
      'no-useless-constructor': 'error',
      'class-methods-use-this': 'off',

      // Control flow
      'consistent-return': 'error',
      'no-else-return': 'error',
      'no-unneeded-ternary': 'error',
      'one-var': ['error', 'never'],

      // Equality and comparisons
      'eqeqeq': ['error', 'always'],

      // Security
      'no-eval': 'error',
      'no-implied-eval': 'error',
      'no-new-func': 'error',
      'no-script-url': 'error',

      // Best practices
      'no-self-compare': 'error',
      'no-sequences': 'error',
      'radix': 'error',
      'wrap-iife': ['error', 'outside'],
      'yoda': 'error',

      // Style
      'no-nested-ternary': 'error',
      'no-mixed-operators': 'error',
      'nonblock-statement-body-position': ['error', 'beside']
    },
  },

  // Test files configuration
  {
    name: 'trustformers-js/test-files',
    files: ['**/*.test.js', '**/*.spec.js', 'test/**/*.js'],
    languageOptions: {
      globals: {
        // Test environment globals
        describe: 'readonly',
        it: 'readonly',
        test: 'readonly',
        expect: 'readonly',
        beforeEach: 'readonly',
        afterEach: 'readonly',
        beforeAll: 'readonly',
        afterAll: 'readonly',
        jest: 'readonly',
      },
    },
    rules: {
      'no-console': 'off',
    },
  },

  // Scripts configuration
  {
    name: 'trustformers-js/scripts',
    files: ['scripts/**/*.js'],
    rules: {
      'no-console': 'off',
    },
  },

  // Examples configuration
  {
    name: 'trustformers-js/examples',
    files: ['examples/**/*.js'],
    rules: {
      'no-console': 'off',
    },
  },

  // JSX files (React)
  {
    name: 'trustformers-js/jsx-files',
    files: ['**/*.jsx', '**/*react*.js'],
    languageOptions: {
      parserOptions: {
        ecmaFeatures: {
          jsx: true
        }
      }
    },
    rules: {
      // Allow JSX syntax
      'no-undef': 'off', // React JSX can have custom elements
    },
  },

  // Ignore patterns
  {
    name: 'trustformers-js/ignores',
    ignores: [
      'dist/',
      'pkg/',
      'node_modules/',
      '*.min.js',
      'packages/*/dist/',
      'packages/*/pkg/',
    ],
  },
];