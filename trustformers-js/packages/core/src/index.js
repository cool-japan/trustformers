/**
 * TrustformeRS Core Package
 * Main entry point for all core functionality
 */

// Re-export everything from the main source
export * from '../../../src/index.js';
export * from '../../../src/tensor/index.js';
export * from '../../../src/models/index.js';
export * from '../../../src/pipeline/index.js';
export * from '../../../src/utils/index.js';

// WASM bindings
export * from '../pkg/trustformers_wasm.js';

// Package-specific utilities
export const PACKAGE_VERSION = '0.1.0';
export const PACKAGE_NAME = '@trustformers/core';

export default {
  version: PACKAGE_VERSION,
  name: PACKAGE_NAME,
};
