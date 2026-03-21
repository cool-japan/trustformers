module.exports = {
  presets: [
    [
      '@babel/preset-env',
      {
        targets: {
          browsers: ['> 1%', 'last 2 versions', 'not ie <= 11'],
          node: '16'
        },
        modules: false,
        useBuiltIns: 'usage',
        corejs: 3,
        debug: false
      }
    ]
  ],
  plugins: [
    '@babel/plugin-syntax-dynamic-import',
    '@babel/plugin-proposal-class-properties'
  ],
  env: {
    test: {
      presets: [
        [
          '@babel/preset-env',
          {
            targets: {
              node: 'current'
            },
            modules: 'commonjs'
          }
        ]
      ]
    },
    production: {
      plugins: [
        [
          'transform-remove-console',
          {
            exclude: ['error', 'warn']
          }
        ]
      ]
    }
  }
};