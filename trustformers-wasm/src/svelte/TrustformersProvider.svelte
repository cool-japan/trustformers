<!--
  TrustformersProvider.svelte
  
  Root provider component for TrustformeRS WASM integration.
  Handles initialization and provides context to child components.
-->

<script>
  import { onMount, onDestroy, setContext, createEventDispatcher } from 'svelte';
  import { 
    initializeTrustformers, 
    enableWebGPU, 
    wasmState, 
    memoryUsage, 
    webGpuAvailable,
    cleanupTensors,
    createErrorHandler
  } from '../svelte_bindings.js';

  // Props
  export let autoInit = true;
  export let enableGpu = false;
  export let onInitialized = null;
  export let onError = null;

  // Event dispatcher
  const dispatch = createEventDispatcher();

  // Error handling
  const errorHandler = createErrorHandler('TrustformersProvider');

  // State
  let initializationPromise = null;
  let mounted = false;

  // Provide context for child components
  setContext('trustformers', {
    wasmState,
    memoryUsage,
    webGpuAvailable,
    errorHandler,
    enableWebGPU: handleEnableWebGPU
  });

  onMount(async () => {
    mounted = true;
    
    if (autoInit) {
      await initialize();
    }
  });

  onDestroy(() => {
    mounted = false;
    cleanupTensors();
  });

  async function initialize() {
    if (initializationPromise) {
      return initializationPromise;
    }

    initializationPromise = (async () => {
      try {
        dispatch('init-start');
        
        const instance = await initializeTrustformers();
        
        if (enableGpu && $webGpuAvailable) {
          try {
            await enableWebGPU();
            dispatch('webgpu-enabled');
          } catch (gpuError) {
            console.warn('WebGPU enablement failed:', gpuError.message);
            dispatch('webgpu-failed', { error: gpuError });
          }
        }
        
        dispatch('initialized', { instance });
        
        if (onInitialized) {
          onInitialized(instance);
        }
        
        return instance;
      } catch (error) {
        errorHandler.handle(error, 'initialization');
        dispatch('init-error', { error });
        
        if (onError) {
          onError(error);
        }
        
        throw error;
      }
    })();

    return initializationPromise;
  }

  async function handleEnableWebGPU() {
    if (!$wasmState.initialized) {
      throw new Error('WASM must be initialized before enabling WebGPU');
    }

    try {
      await enableWebGPU();
      dispatch('webgpu-enabled');
      return true;
    } catch (error) {
      errorHandler.handle(error, 'WebGPU enablement');
      dispatch('webgpu-failed', { error });
      throw error;
    }
  }

  // Reactive statements
  $: if ($wasmState.error) {
    dispatch('error', { error: $wasmState.error });
  }

  $: if ($wasmState.initialized && mounted) {
    dispatch('ready');
  }

  // Expose methods to parent
  export { initialize, handleEnableWebGPU as enableWebGPU };
</script>

<!-- Loading state -->
{#if $wasmState.loading}
  <div class="trustformers-loading" aria-live="polite">
    <div class="loading-spinner"></div>
    <p>Initializing TrustformeRS WASM...</p>
    {#if enableGpu && $webGpuAvailable}
      <p class="loading-subtext">Preparing WebGPU acceleration...</p>
    {/if}
  </div>
{/if}

<!-- Error state -->
{#if $wasmState.error}
  <div class="trustformers-error" role="alert" aria-live="assertive">
    <h3>TrustformeRS Initialization Error</h3>
    <p>{$wasmState.error}</p>
    <button on:click={() => initialize()} class="retry-button">
      Retry Initialization
    </button>
  </div>
{/if}

<!-- Success state - render children -->
{#if $wasmState.initialized && !$wasmState.error}
  <div class="trustformers-provider" class:gpu-enabled={$wasmState.device === 'gpu'}>
    <!-- Status bar -->
    <div class="status-bar">
      <span class="device-status" class:gpu={$wasmState.device === 'gpu'}>
        Device: {$wasmState.device.toUpperCase()}
      </span>
      <span class="memory-usage">
        Memory: {($memoryUsage.estimatedMemoryMB).toFixed(2)} MB
      </span>
      <span class="tensor-count">
        Tensors: {$memoryUsage.totalTensors}
      </span>
    </div>
    
    <!-- Child components -->
    <slot />
  </div>
{/if}

<style>
  .trustformers-loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    text-align: center;
  }

  .loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 1rem;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  .loading-subtext {
    font-size: 0.9rem;
    color: #666;
    margin-top: 0.5rem;
  }

  .trustformers-error {
    background-color: #ffebee;
    border: 1px solid #ffcdd2;
    border-radius: 4px;
    padding: 1rem;
    margin: 1rem 0;
    color: #c62828;
  }

  .trustformers-error h3 {
    margin: 0 0 0.5rem 0;
    color: #d32f2f;
  }

  .retry-button {
    background-color: #2196f3;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    margin-top: 1rem;
  }

  .retry-button:hover {
    background-color: #1976d2;
  }

  .trustformers-provider {
    position: relative;
  }

  .trustformers-provider.gpu-enabled {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-size: 200% 200%;
    animation: gradientShift 3s ease infinite;
  }

  @keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  .status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 1rem;
    background-color: rgba(0, 0, 0, 0.05);
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    font-size: 0.8rem;
    font-family: monospace;
  }

  .device-status {
    font-weight: bold;
    padding: 0.2rem 0.5rem;
    border-radius: 3px;
    background-color: #4caf50;
    color: white;
  }

  .device-status.gpu {
    background-color: #ff9800;
  }

  .memory-usage, .tensor-count {
    color: #666;
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .status-bar {
      flex-direction: column;
      gap: 0.25rem;
    }
    
    .loading-spinner {
      width: 30px;
      height: 30px;
    }
  }

  /* High contrast mode support */
  @media (prefers-contrast: high) {
    .trustformers-error {
      background-color: #fff;
      border: 2px solid #000;
      color: #000;
    }
    
    .device-status, .retry-button {
      border: 2px solid #000;
    }
  }

  /* Reduced motion support */
  @media (prefers-reduced-motion: reduce) {
    .loading-spinner {
      animation: none;
      border: 4px solid #3498db;
    }
    
    .trustformers-provider.gpu-enabled {
      animation: none;
      background: #667eea;
    }
  }
</style>