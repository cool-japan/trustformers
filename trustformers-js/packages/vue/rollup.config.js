import { createConfig } from '../../rollup.config.shared.js';
import vue from 'rollup-plugin-vue';

const config = createConfig('@trustformers/vue', ['vue']);

// Add Vue plugin
config.plugins.splice(1, 0, vue({
  css: true,
  compileTemplate: true
}));

export default config;