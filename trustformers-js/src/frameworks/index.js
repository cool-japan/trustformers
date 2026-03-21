/**
 * Framework Integrations Index
 * Exports all framework-specific integrations for TrustformeRS
 */

// React Integration
export * from './react.js';
export { default as React } from './react.js';

// Vue.js Integration
export * from './vue.js';
export { default as Vue } from './vue.js';

// Angular Integration
export * from './angular.js';
export { default as Angular } from './angular.js';

// Svelte Integration
export * from './svelte.js';
export { default as Svelte } from './svelte.js';

/**
 * Framework detection utility
 * @returns {string} Detected framework name
 */
export function detectFramework() {
  // Check for React
  if (typeof window !== 'undefined' && window.React) {
    return 'react';
  }

  // Check for Vue
  if (typeof window !== 'undefined' && window.Vue) {
    return 'vue';
  }

  // Check for Angular
  if (typeof window !== 'undefined' && window.ng) {
    return 'angular';
  }

  // Check for Svelte (harder to detect, look for compilation artifacts)
  if (typeof window !== 'undefined' && document.querySelector('[data-svelte]')) {
    return 'svelte';
  }

  // Check for framework-specific global variables
  if (typeof process !== 'undefined' && process.env) {
    if (process.env.REACT_APP_VERSION) return 'react';
    if (process.env.VUE_APP_VERSION) return 'vue';
  }

  return 'vanilla';
}

/**
 * Get framework-specific integration
 * @param {string} framework - Framework name
 * @returns {Object} Framework integration object
 */
export function getFrameworkIntegration(framework) {
  switch (framework.toLowerCase()) {
    case 'react':
      return React;
    case 'vue':
      return Vue;
    case 'angular':
      return Angular;
    case 'svelte':
      return Svelte;
    default:
      return null;
  }
}

/**
 * Auto-configure for detected framework
 * @param {Object} options - Configuration options
 * @returns {Object} Framework-specific configuration
 */
export function autoConfigureFramework(options = {}) {
  const framework = detectFramework();
  const integration = getFrameworkIntegration(framework);

  if (!integration) {
    console.warn(`No integration found for framework: ${framework}`);
    return null;
  }

  console.warn(`Auto-configured for ${framework}`);
  return {
    framework,
    integration,
    ...options,
  };
}

export default {
  React,
  Vue,
  Angular,
  Svelte,
  detectFramework,
  getFrameworkIntegration,
  autoConfigureFramework,
};
