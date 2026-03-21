import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import babel from '@rollup/plugin-babel';
import terser from '@rollup/plugin-terser';
import replace from '@rollup/plugin-replace';
import json from '@rollup/plugin-json';
import { wasm } from '@rollup/plugin-wasm';
import filesize from 'rollup-plugin-filesize';
import progress from 'rollup-plugin-progress';
import analyze from 'rollup-plugin-analyzer';
import gzipPlugin from 'rollup-plugin-gzip';

const isProduction = process.env.NODE_ENV === 'production';
const isDevelopment = !isProduction;

const external = [
  'fs',
  'path',
  'crypto',
  'util',
  'stream',
  'worker_threads',
  'cluster',
  'os'
];

const globals = {
  'fs': 'fs',
  'path': 'path',
  'crypto': 'crypto',
  'util': 'util',
  'stream': 'stream',
  'worker_threads': 'worker_threads',
  'cluster': 'cluster',
  'os': 'os'
};

const basePlugins = [
  progress({
    clearLine: false
  }),
  json(),
  resolve({
    browser: true,
    preferBuiltins: false,
    exportConditions: ['browser', 'module', 'import', 'default']
  }),
  commonjs({
    include: /node_modules/,
    transformMixedEsModules: true
  }),
  wasm({
    sync: ['*.wasm'],
    maxFileSize: 10000000 // 10MB
  }),
  replace({
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
    'process.env.TRUSTFORMERS_VERSION': JSON.stringify(require('./package.json').version),
    preventAssignment: true
  }),
  babel({
    babelHelpers: 'bundled',
    exclude: 'node_modules/**',
    presets: [
      ['@babel/preset-env', {
        targets: {
          browsers: ['> 1%', 'last 2 versions', 'not ie <= 11']
        },
        modules: false,
        useBuiltIns: 'usage',
        corejs: 3
      }]
    ],
    plugins: [
      '@babel/plugin-syntax-dynamic-import',
      '@babel/plugin-proposal-class-properties'
    ]
  })
];

const productionPlugins = [
  terser({
    compress: {
      drop_console: true,
      drop_debugger: true,
      pure_funcs: ['console.log', 'console.info', 'console.debug']
    },
    format: {
      comments: false
    },
    mangle: {
      safari10: true
    }
  }),
  gzipPlugin({
    filter: /\.(js|json|css|html|svg)$/
  }),
  filesize({
    showMinifiedSize: true,
    showGzippedSize: true,
    showBrotliSize: true
  }),
  analyze({
    summaryOnly: true,
    limit: 10
  })
];

// ESM Build (tree-shakable)
const esmConfig = {
  input: 'src/index.js',
  output: {
    file: `dist/trustformers.esm${isProduction ? '.min' : ''}.js`,
    format: 'es',
    sourcemap: true,
    exports: 'named'
  },
  external: external.filter(dep => dep !== 'buffer' && dep !== 'process'),
  plugins: [
    ...basePlugins,
    ...(isProduction ? productionPlugins : [])
  ],
  treeshake: {
    moduleSideEffects: false,
    propertyReadSideEffects: false,
    unknownGlobalSideEffects: false
  }
};

// UMD Build (for CDN)
const umdConfig = {
  input: 'src/index.js',
  output: {
    file: `dist/trustformers.umd${isProduction ? '.min' : ''}.js`,
    format: 'umd',
    name: 'TrustformeRS',
    sourcemap: true,
    exports: 'named',
    globals
  },
  external: external,
  plugins: [
    ...basePlugins,
    ...(isProduction ? productionPlugins : [])
  ]
};

// CommonJS Build (for Node.js)
const cjsConfig = {
  input: 'src/index.js',
  output: {
    file: `dist/trustformers.cjs${isProduction ? '.min' : ''}.js`,
    format: 'cjs',
    sourcemap: true,
    exports: 'named'
  },
  external: external,
  plugins: [
    ...basePlugins,
    ...(isProduction ? productionPlugins : [])
  ]
};

// IIFE Build (for direct browser usage)
const iifeConfig = {
  input: 'src/index.js',
  output: {
    file: `dist/trustformers.iife${isProduction ? '.min' : ''}.js`,
    format: 'iife',
    name: 'TrustformeRS',
    sourcemap: true,
    exports: 'named'
  },
  plugins: [
    ...basePlugins,
    resolve({
      browser: true,
      preferBuiltins: false
    }),
    ...(isProduction ? productionPlugins : [])
  ]
};

// Modular builds for tree-shaking
const modularBuilds = [
  {
    input: 'src/tensor/index.js',
    output: {
      file: `dist/modules/tensor${isProduction ? '.min' : ''}.js`,
      format: 'es',
      sourcemap: true
    }
  },
  {
    input: 'src/models/index.js',
    output: {
      file: `dist/modules/models${isProduction ? '.min' : ''}.js`,
      format: 'es',
      sourcemap: true
    }
  },
  {
    input: 'src/pipeline/index.js',
    output: {
      file: `dist/modules/pipeline${isProduction ? '.min' : ''}.js`,
      format: 'es',
      sourcemap: true
    }
  },
  {
    input: 'src/utils/index.js',
    output: {
      file: `dist/modules/utils${isProduction ? '.min' : ''}.js`,
      format: 'es',
      sourcemap: true
    }
  }
].map(config => ({
  ...config,
  external: external,
  plugins: [
    ...basePlugins,
    ...(isProduction ? productionPlugins : [])
  ],
  treeshake: {
    moduleSideEffects: false,
    propertyReadSideEffects: false,
    unknownGlobalSideEffects: false
  }
}));

export default [
  esmConfig,
  umdConfig,
  cjsConfig,
  iifeConfig,
  ...(isProduction ? modularBuilds : [])
];