import { createConfig } from '../../rollup.config.shared.js';
import svelte from 'rollup-plugin-svelte';

const config = createConfig('@trustformers/svelte', ['svelte']);

// Add Svelte plugin
config.plugins.splice(1, 0, svelte({
  compilerOptions: {
    css: false
  }
}));

export default config;