const path = require('path');
const webpack = require('webpack');
const TerserPlugin = require('terser-webpack-plugin');
const CompressionPlugin = require('compression-webpack-plugin');

const isProduction = process.env.NODE_ENV === 'production';
const isDevelopment = !isProduction;

const baseConfig = {
  entry: './src/index.js',
  mode: isProduction ? 'production' : 'development',
  devtool: isProduction ? 'source-map' : 'eval-source-map',
  resolve: {
    extensions: ['.js', '.ts', '.wasm'],
    fallback: {
      "crypto": false,
      "fs": false,
      "path": false,
      "util": false,
      "stream": false,
      "buffer": require.resolve("buffer/"),
      "process": require.resolve("process/browser")
    }
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
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
          }
        }
      },
      {
        test: /\.wasm$/,
        type: 'webassembly/async'
      }
    ]
  },
  plugins: [
    new webpack.DefinePlugin({
      'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
      'process.env.TRUSTFORMERS_VERSION': JSON.stringify(require('./package.json').version)
    }),
    new webpack.ProvidePlugin({
      Buffer: ['buffer', 'Buffer'],
      process: 'process/browser'
    })
  ],
  optimization: {
    minimize: isProduction,
    minimizer: [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: isProduction,
            drop_debugger: isProduction,
            pure_funcs: isProduction ? ['console.log', 'console.info'] : []
          },
          mangle: {
            safari10: true
          },
          format: {
            comments: false
          }
        },
        extractComments: false
      })
    ],
    sideEffects: false,
    usedExports: true
  },
  experiments: {
    asyncWebAssembly: true,
    topLevelAwait: true
  }
};

// ESM Build (for modern bundlers and browsers)
const esmConfig = {
  ...baseConfig,
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: isProduction ? 'trustformers.esm.min.js' : 'trustformers.esm.js',
    library: {
      type: 'module'
    },
    environment: {
      module: true,
      dynamicImport: true
    },
    clean: true
  },
  target: ['web', 'es2020'],
  experiments: {
    ...baseConfig.experiments,
    outputModule: true
  }
};

// UMD Build (for CDN and legacy browsers)
const umdConfig = {
  ...baseConfig,
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: isProduction ? 'trustformers.umd.min.js' : 'trustformers.umd.js',
    library: {
      name: 'TrustformeRS',
      type: 'umd',
      export: 'default'
    },
    globalObject: 'typeof self !== "undefined" ? self : this'
  },
  target: ['web', 'es5']
};

// CommonJS Build (for Node.js)
const cjsConfig = {
  ...baseConfig,
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: isProduction ? 'trustformers.cjs.min.js' : 'trustformers.cjs.js',
    library: {
      type: 'commonjs2'
    }
  },
  target: 'node16',
  externals: {
    'fs': 'fs',
    'path': 'path',
    'crypto': 'crypto',
    'util': 'util',
    'stream': 'stream',
    'worker_threads': 'worker_threads',
    'cluster': 'cluster',
    'os': 'os'
  }
};

// Web Worker Build
const workerConfig = {
  ...baseConfig,
  entry: './src/worker.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: isProduction ? 'trustformers.worker.min.js' : 'trustformers.worker.js',
    globalObject: 'self'
  },
  target: 'webworker'
};

// CDN Optimized Build
const cdnConfig = {
  ...umdConfig,
  output: {
    ...umdConfig.output,
    filename: 'trustformers.cdn.min.js'
  },
  plugins: [
    ...baseConfig.plugins,
    new CompressionPlugin({
      filename: '[path][base].gz',
      algorithm: 'gzip',
      test: /\.(js|css|html|svg)$/,
      threshold: 8192,
      minRatio: 0.8
    }),
    new CompressionPlugin({
      filename: '[path][base].br',
      algorithm: 'brotliCompress',
      test: /\.(js|css|html|svg)$/,
      threshold: 8192,
      minRatio: 0.8
    })
  ],
  optimization: {
    ...baseConfig.optimization,
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendor',
          chunks: 'all'
        },
        wasm: {
          test: /\.wasm$/,
          name: 'wasm',
          chunks: 'all'
        }
      }
    }
  }
};

module.exports = (env, argv) => {
  const configs = [];
  
  if (env && env.format) {
    switch (env.format) {
      case 'esm':
        configs.push(esmConfig);
        break;
      case 'umd':
        configs.push(umdConfig);
        break;
      case 'cjs':
        configs.push(cjsConfig);
        break;
      case 'worker':
        configs.push(workerConfig);
        break;
      case 'cdn':
        configs.push(cdnConfig);
        break;
      default:
        configs.push(esmConfig, umdConfig, cjsConfig);
    }
  } else {
    // Build all formats by default
    configs.push(esmConfig, umdConfig, cjsConfig);
    
    if (isProduction) {
      configs.push(workerConfig, cdnConfig);
    }
  }
  
  return configs;
};