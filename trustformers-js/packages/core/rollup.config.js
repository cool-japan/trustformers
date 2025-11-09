import { createConfig } from '../../rollup.config.shared.js';
import wasm from '@rollup/plugin-wasm';

const config = createConfig('@trustformers/core');

// Add WASM plugin for core package
config.plugins.splice(1, 0, wasm({
  sync: ['*.wasm'],
  maxFileSize: 0 // No file size limit for WASM
}));

// Core package should bundle WASM files
config.external = config.external.filter(ext => {
  if (typeof ext === 'string') {
    return !ext.includes('.wasm');
  }
  return true; // Keep regex and function externals
});

export default config;