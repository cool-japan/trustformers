module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true,
    worker: true
  },
  extends: [
    'eslint:recommended',
    'prettier'
  ],
  plugins: [
    'prettier'
  ],
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module'
  },
  rules: {
    'prettier/prettier': 'error',
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
    'no-useless-constructor': 'error',
    'class-methods-use-this': 'off',
    'import/prefer-default-export': 'off',
    'consistent-return': 'error',
    'no-else-return': 'error',
    'no-unneeded-ternary': 'error',
    'one-var': ['error', 'never'],
    'eqeqeq': ['error', 'always'],
    'no-eval': 'error',
    'no-implied-eval': 'error',
    'no-new-func': 'error',
    'no-script-url': 'error',
    'no-self-compare': 'error',
    'no-sequences': 'error',
    'radix': 'error',
    'wrap-iife': ['error', 'outside'],
    'yoda': 'error',
    'no-nested-ternary': 'error',
    'no-mixed-operators': 'error',
    'nonblock-statement-body-position': ['error', 'beside']
  },
  overrides: [
    {
      files: ['**/*.test.js', '**/*.spec.js'],
      env: {
        jest: true
      },
      rules: {
        'no-console': 'off'
      }
    },
    {
      files: ['scripts/**/*.js'],
      rules: {
        'no-console': 'off'
      }
    },
    {
      files: ['examples/**/*.js'],
      rules: {
        'no-console': 'off',
        'import/no-unresolved': 'off'
      }
    }
  ],
  ignorePatterns: [
    'dist/',
    'pkg/',
    'node_modules/',
    '*.min.js'
  ]
};