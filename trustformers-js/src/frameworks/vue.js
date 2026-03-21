/**
 * Vue.js composition functions and components for TrustformeRS
 * Provides Vue 3 Composition API integrations for the TrustformeRS ecosystem
 */

import {
  ref,
  reactive,
  computed,
  watch,
  onMounted,
  onUnmounted,
  nextTick,
  toRefs,
  readonly,
  shallowRef,
} from 'vue';
import {
  initialize,
  initializeEnhanced,
  createModel,
  createTokenizer,
  Pipeline,
  tensor,
  enhanced_inference,
  performance,
  devTools,
  memory,
} from '../index.js';

/**
 * Composition function for initializing TrustformeRS with Vue reactivity
 * @param {Object} options - Initialization options
 * @returns {Object} Reactive initialization state and methods
 */
export function useTrustformersInit(options = {}) {
  const state = reactive({
    isInitialized: false,
    isLoading: false,
    error: null,
    capabilities: null,
  });

  const initRef = ref(false);

  const initTrustformers = async () => {
    if (initRef.value) return;

    state.isLoading = true;
    state.error = null;
    initRef.value = true;

    try {
      const caps = await initializeEnhanced({
        enableWebGL: true,
        enableMemoryPool: true,
        enableProfiling: true,
        enableZeroCopy: true,
        ...options,
      });

      state.capabilities = caps;
      state.isInitialized = true;
    } catch (err) {
      state.error = err;
      initRef.value = false;
    } finally {
      state.isLoading = false;
    }
  };

  const reinitialize = async (newOptions = {}) => {
    initRef.value = false;
    state.isInitialized = false;
    state.capabilities = null;
    await initTrustformers({ ...options, ...newOptions });
  };

  onMounted(() => {
    initTrustformers();
  });

  return {
    ...toRefs(state),
    reinitialize,
  };
}

/**
 * Composition function for creating and managing models
 * @param {Ref|string|Object} modelConfig - Reactive model configuration or type string
 * @param {Object} options - Model options
 * @returns {Object} Reactive model state and methods
 */
export function useModel(modelConfig, options = {}) {
  const state = reactive({
    model: null,
    isLoading: false,
    error: null,
  });

  const modelRef = shallowRef(null);

  const createModelInstance = async config => {
    if (!config) return;

    state.isLoading = true;
    state.error = null;

    try {
      // Cleanup previous model
      if (modelRef.value && modelRef.value.free) {
        modelRef.value.free();
      }

      const modelInstance = createModel(config);
      modelRef.value = modelInstance;
      state.model = modelInstance;
    } catch (err) {
      state.error = err;
    } finally {
      state.isLoading = false;
    }
  };

  const runInference = async (inputs, inferenceOptions = {}) => {
    if (!state.model) throw new Error('Model not initialized');

    return await enhanced_inference.runInference(state.model, inputs, {
      profile: true,
      useMemoryPool: true,
      autoCleanup: true,
      ...inferenceOptions,
    });
  };

  const batchInference = async (batchInputs, batchOptions = {}) => {
    if (!state.model) throw new Error('Model not initialized');

    return await enhanced_inference.batchInference(state.model, batchInputs, {
      batchSize: 32,
      profile: true,
      ...batchOptions,
    });
  };

  // Watch for changes in modelConfig
  watch(
    () => modelConfig,
    newConfig => {
      if (newConfig) {
        createModelInstance(newConfig);
      }
    },
    { immediate: true }
  );

  onUnmounted(() => {
    if (modelRef.value && modelRef.value.free) {
      modelRef.value.free();
    }
  });

  return {
    ...toRefs(state),
    runInference,
    batchInference,
  };
}

/**
 * Composition function for creating and managing tokenizers
 * @param {Ref|string} type - Reactive tokenizer type
 * @param {Ref|Object} vocab - Reactive vocabulary object
 * @returns {Object} Reactive tokenizer state and methods
 */
export function useTokenizer(type, vocab = null) {
  const state = reactive({
    tokenizer: null,
    isLoading: false,
    error: null,
  });

  const tokenizerRef = shallowRef(null);

  const createTokenizerInstance = async (tokType, vocabData) => {
    if (!tokType) return;

    state.isLoading = true;
    state.error = null;

    try {
      // Cleanup previous tokenizer
      if (tokenizerRef.value && tokenizerRef.value.free) {
        tokenizerRef.value.free();
      }

      const tokenizerInstance = createTokenizer(tokType, vocabData);
      tokenizerRef.value = tokenizerInstance;
      state.tokenizer = tokenizerInstance;
    } catch (err) {
      state.error = err;
    } finally {
      state.isLoading = false;
    }
  };

  const encode = text => {
    if (!state.tokenizer) throw new Error('Tokenizer not initialized');
    return state.tokenizer.encode(text);
  };

  const decode = tokens => {
    if (!state.tokenizer) throw new Error('Tokenizer not initialized');
    return state.tokenizer.decode(tokens);
  };

  const batchEncode = texts => {
    if (!state.tokenizer) throw new Error('Tokenizer not initialized');
    return texts.map(text => state.tokenizer.encode(text));
  };

  // Watch for changes in type and vocab
  watch(
    [() => type, () => vocab],
    ([newType, newVocab]) => {
      if (newType) {
        createTokenizerInstance(newType, newVocab);
      }
    },
    { immediate: true }
  );

  onUnmounted(() => {
    if (tokenizerRef.value && tokenizerRef.value.free) {
      tokenizerRef.value.free();
    }
  });

  return {
    ...toRefs(state),
    encode,
    decode,
    batchEncode,
  };
}

/**
 * Composition function for creating and managing pipelines
 * @param {Ref|string} task - Reactive pipeline task type
 * @param {Ref|Object} model - Reactive model object
 * @param {Ref|Object} tokenizer - Reactive tokenizer object
 * @param {Object} config - Pipeline configuration
 * @returns {Object} Reactive pipeline state and methods
 */
export function usePipeline(task, model, tokenizer, config = {}) {
  const state = reactive({
    pipeline: null,
    isLoading: false,
    error: null,
  });

  const pipelineRef = shallowRef(null);

  const createPipelineInstance = async (taskType, modelObj, tokenizerObj, pipelineConfig) => {
    if (!taskType || !modelObj || !tokenizerObj) return;

    state.isLoading = true;
    state.error = null;

    try {
      // Cleanup previous pipeline
      if (pipelineRef.value && pipelineRef.value.free) {
        pipelineRef.value.free();
      }

      let pipelineInstance;

      switch (taskType) {
        case 'text-generation':
          pipelineInstance = Pipeline.textGeneration(modelObj, tokenizerObj, pipelineConfig);
          break;
        case 'text-classification':
          pipelineInstance = Pipeline.textClassification(
            modelObj,
            tokenizerObj,
            pipelineConfig.labels || []
          );
          break;
        case 'question-answering':
          pipelineInstance = Pipeline.questionAnswering(modelObj, tokenizerObj);
          break;
        default:
          pipelineInstance = await Pipeline.fromPretrained(
            taskType,
            pipelineConfig.modelName || 'default'
          );
      }

      pipelineRef.value = pipelineInstance;
      state.pipeline = pipelineInstance;
    } catch (err) {
      state.error = err;
    } finally {
      state.isLoading = false;
    }
  };

  const run = async (input, options = {}) => {
    if (!state.pipeline) throw new Error('Pipeline not initialized');

    return await state.pipeline.run(input, {
      maxLength: 50,
      temperature: 0.7,
      ...options,
    });
  };

  const batch = async (inputs, options = {}) => {
    if (!state.pipeline) throw new Error('Pipeline not initialized');

    return Promise.all(inputs.map(input => state.pipeline.run(input, options)));
  };

  // Watch for changes in task, model, tokenizer
  watch(
    [() => task, () => model, () => tokenizer],
    ([newTask, newModel, newTokenizer]) => {
      if (newTask && newModel && newTokenizer) {
        createPipelineInstance(newTask, newModel, newTokenizer, config);
      }
    },
    { immediate: true }
  );

  onUnmounted(() => {
    if (pipelineRef.value && pipelineRef.value.free) {
      pipelineRef.value.free();
    }
  });

  return {
    ...toRefs(state),
    run,
    batch,
  };
}

/**
 * Composition function for tensor operations with Vue reactivity
 * @param {Ref|Array} initialData - Reactive initial tensor data
 * @param {Ref|Array} initialShape - Reactive initial tensor shape
 * @returns {Object} Reactive tensor state and operations
 */
export function useTensor(initialData = null, initialShape = null) {
  const state = reactive({
    tensor: null,
    shape: null,
    size: null,
    dtype: null,
    error: null,
  });

  const tensorRef = shallowRef(null);

  const createTensor = async (data, shape) => {
    try {
      // Cleanup previous tensor
      if (tensorRef.value && tensorRef.value.free) {
        tensorRef.value.free();
      }

      const newTensor = tensor(data, shape);
      tensorRef.value = newTensor;

      state.tensor = newTensor;
      state.shape = Array.from(newTensor.shape());
      state.size = newTensor.size();
      state.dtype = newTensor.dtype();
      state.error = null;
    } catch (err) {
      state.error = err;
    }
  };

  const operations = computed(() => ({
    reshape: newShape => {
      if (!state.tensor) return null;
      try {
        return state.tensor.reshape(newShape);
      } catch (err) {
        state.error = err;
        return null;
      }
    },

    transpose: (dims = null) => {
      if (!state.tensor) return null;
      try {
        return dims ? state.tensor.transpose(dims) : state.tensor.transpose();
      } catch (err) {
        state.error = err;
        return null;
      }
    },

    slice: (start, end, step = 1) => {
      if (!state.tensor) return null;
      try {
        return state.tensor.slice(start, end, step);
      } catch (err) {
        state.error = err;
        return null;
      }
    },

    add: other => {
      if (!state.tensor) return null;
      try {
        return state.tensor.add(other);
      } catch (err) {
        state.error = err;
        return null;
      }
    },

    multiply: other => {
      if (!state.tensor) return null;
      try {
        return state.tensor.mul(other);
      } catch (err) {
        state.error = err;
        return null;
      }
    },

    matmul: other => {
      if (!state.tensor) return null;
      try {
        return state.tensor.matmul(other);
      } catch (err) {
        state.error = err;
        return null;
      }
    },
  }));

  // Watch for changes in initialData and initialShape
  watch(
    [() => initialData, () => initialShape],
    ([newData, newShape]) => {
      if (newData && newShape) {
        createTensor(newData, newShape);
      }
    },
    { immediate: true }
  );

  onUnmounted(() => {
    if (tensorRef.value && tensorRef.value.free) {
      tensorRef.value.free();
    }
  });

  return {
    ...toRefs(state),
    operations,
    createTensor,
  };
}

/**
 * Composition function for performance monitoring in Vue applications
 * @param {Ref|boolean} enabled - Reactive enabled state
 * @returns {Object} Reactive performance monitoring state and methods
 */
export function usePerformance(enabled = ref(true)) {
  const state = reactive({
    metrics: null,
    sessionId: null,
  });

  let intervalId = null;

  const startSession = (name, metadata = {}) => {
    if (!enabled.value) return null;

    const id = performance.startSession(name, {
      framework: 'vue',
      timestamp: new Date().toISOString(),
      ...metadata,
    });
    state.sessionId = id;
    return id;
  };

  const endSession = () => {
    if (!enabled.value || !state.sessionId) return null;

    const report = performance.endSession();
    state.sessionId = null;
    return report;
  };

  const profileOperation = async (name, operation, metadata = {}) => {
    if (!enabled.value) return await operation();

    return await performance.profile(name, operation, {
      framework: 'vue',
      ...metadata,
    });
  };

  const updateMetrics = () => {
    if (!enabled.value) return;

    const currentMetrics = {
      memory: performance.getMemoryUsage(),
      report: performance.getReport(),
      timestamp: Date.now(),
    };
    state.metrics = currentMetrics;
  };

  const cleanup = () => {
    performance.cleanup();
    updateMetrics();
  };

  // Watch enabled state
  watch(
    enabled,
    isEnabled => {
      if (isEnabled) {
        updateMetrics();
        intervalId = setInterval(updateMetrics, 5000); // Update every 5 seconds
      } else {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
      }
    },
    { immediate: true }
  );

  onUnmounted(() => {
    if (intervalId) {
      clearInterval(intervalId);
    }
    if (state.sessionId) {
      endSession();
    }
  });

  return {
    ...toRefs(state),
    startSession,
    endSession,
    profileOperation,
    updateMetrics,
    cleanup,
  };
}

/**
 * Composition function for memory management in Vue applications
 * @returns {Object} Reactive memory management state and methods
 */
export function useMemory() {
  const memoryStats = ref(null);
  let intervalId = null;

  const updateMemoryStats = () => {
    const stats = memory.getStats();
    memoryStats.value = {
      ...stats,
      timestamp: Date.now(),
    };
  };

  const cleanup = () => {
    if (typeof window !== 'undefined' && window.gc) {
      window.gc();
    }
    performance.cleanup();
    updateMemoryStats();
  };

  onMounted(() => {
    updateMemoryStats();
    intervalId = setInterval(updateMemoryStats, 3000); // Update every 3 seconds
  });

  onUnmounted(() => {
    if (intervalId) {
      clearInterval(intervalId);
    }
  });

  return {
    memoryStats: readonly(memoryStats),
    updateMemoryStats,
    cleanup,
  };
}

/**
 * Composition function for development tools integration
 * @param {Ref|boolean} enabled - Reactive enabled state
 * @returns {Object} Reactive development tools state and methods
 */
export function useDevTools(enabled = ref(process.env.NODE_ENV === 'development')) {
  const state = reactive({
    debugState: null,
    isEnabled: enabled.value,
  });

  const enableDebug = (options = {}) => {
    devTools.debug.enable({
      trackTensors: true,
      trackOperations: true,
      validateOperations: true,
      ...options,
    });
    state.isEnabled = true;
  };

  const disableDebug = () => {
    devTools.debug.disable();
    state.isEnabled = false;
  };

  const startDebugSession = (name, metadata = {}) => {
    if (!state.isEnabled) return null;

    return devTools.debug.startSession(name, {
      framework: 'vue',
      timestamp: new Date().toISOString(),
      ...metadata,
    });
  };

  const trackTensor = (tensor, operation, metadata = {}) => {
    if (!state.isEnabled) return;

    devTools.debug.trackTensor(tensor, operation, {
      framework: 'vue',
      ...metadata,
    });
  };

  const validateOperation = (operation, tensors, options = {}) => {
    if (!state.isEnabled) return true;

    return devTools.debug.validateOperation(operation, tensors, options);
  };

  const generateReport = () => {
    if (!state.isEnabled) return null;

    const report = devTools.debug.generateReport();
    state.debugState = report;
    return report;
  };

  const analyzeComprehensive = (options = {}) => {
    if (!state.isEnabled) return null;

    return devTools.analyze.comprehensive({
      includeDebug: true,
      ...options,
    });
  };

  // Watch enabled state
  watch(enabled, isEnabled => {
    state.isEnabled = isEnabled;
  });

  return {
    ...toRefs(state),
    enableDebug,
    disableDebug,
    startDebugSession,
    trackTensor,
    validateOperation,
    generateReport,
    analyzeComprehensive,
  };
}

/**
 * Vue component definitions for templates
 */

// TrustformersProvider component template
export const TrustformersProviderTemplate = {
  name: 'TrustformersProvider',
  props: {
    options: {
      type: Object,
      default: () => ({}),
    },
  },
  emits: ['init', 'error'],
  setup(props, { emit, slots }) {
    const { isInitialized, isLoading, error, capabilities } = useTrustformersInit(props.options);

    watch(isInitialized, initialized => {
      if (initialized) {
        emit('init', capabilities.value);
      }
    });

    watch(error, err => {
      if (err) {
        emit('error', err);
      }
    });

    return {
      isInitialized,
      isLoading,
      error,
      capabilities,
      slots,
    };
  },
  template: `
    <div class="trustformers-provider" :data-initialized="isInitialized">
      <div v-if="error" class="trustformers-error">
        <h3>TrustformeRS Initialization Error</h3>
        <p>{{ error.message }}</p>
        <pre>{{ error.stack }}</pre>
      </div>
      <div v-else-if="isLoading" class="trustformers-loading">
        <p>Initializing TrustformeRS...</p>
      </div>
      <template v-else>
        <slot></slot>
      </template>
    </div>
  `,
};

// PerformanceMonitor component template
export const PerformanceMonitorTemplate = {
  name: 'PerformanceMonitor',
  props: {
    enabled: {
      type: Boolean,
      default: true,
    },
    updateInterval: {
      type: Number,
      default: 5000,
    },
  },
  setup(props) {
    const { metrics } = usePerformance(ref(props.enabled));
    const expanded = ref(false);

    return {
      metrics,
      expanded,
    };
  },
  template: `
    <div 
      v-if="enabled && metrics" 
      class="performance-monitor"
      :style="{
        position: 'fixed',
        top: '10px',
        right: '10px',
        background: 'rgba(0,0,0,0.8)',
        color: 'white',
        padding: '10px',
        borderRadius: '5px',
        fontSize: '12px',
        fontFamily: 'monospace',
        zIndex: 9999,
        cursor: 'pointer'
      }"
      @click="expanded = !expanded"
    >
      <div>🚀 TrustformeRS Performance</div>
      <div v-if="metrics.memory">
        Memory: {{ (metrics.memory.used / 1024 / 1024).toFixed(1) }}MB
      </div>
      <div v-if="expanded && metrics.report" :style="{ marginTop: '10px', fontSize: '10px' }">
        <div>
          Capabilities: {{ 
            Object.entries(metrics.report.capabilities)
              .filter(([, v]) => v)
              .map(([k]) => k)
              .join(', ') 
          }}
        </div>
        <div v-if="metrics.report.profiling">
          Operations: {{ metrics.report.profiling.totalOperations || 0 }}
        </div>
      </div>
    </div>
  `,
};

// MemoryMonitor component template
export const MemoryMonitorTemplate = {
  name: 'MemoryMonitor',
  props: {
    enabled: {
      type: Boolean,
      default: true,
    },
  },
  setup(props) {
    const { memoryStats } = useMemory();
    const expanded = ref(false);

    const usedMB = computed(() =>
      memoryStats.value?.used ? (memoryStats.value.used / 1024 / 1024).toFixed(1) : 'N/A'
    );

    const limitMB = computed(() =>
      memoryStats.value?.limit ? (memoryStats.value.limit / 1024 / 1024).toFixed(1) : 'N/A'
    );

    const usagePercent = computed(() =>
      memoryStats.value?.used && memoryStats.value?.limit
        ? ((memoryStats.value.used / memoryStats.value.limit) * 100).toFixed(1)
        : 'N/A'
    );

    return {
      memoryStats,
      expanded,
      usedMB,
      limitMB,
      usagePercent,
    };
  },
  template: `
    <div 
      v-if="enabled && memoryStats" 
      class="memory-monitor"
      :style="{
        position: 'fixed',
        top: '80px',
        right: '10px',
        background: 'rgba(0,0,0,0.8)',
        color: 'white',
        padding: '10px',
        borderRadius: '5px',
        fontSize: '12px',
        fontFamily: 'monospace',
        zIndex: 9999,
        cursor: 'pointer'
      }"
      @click="expanded = !expanded"
    >
      <div>💾 Memory</div>
      <div>{{ usedMB }}MB / {{ limitMB }}MB</div>
      <div v-if="expanded" :style="{ marginTop: '10px', fontSize: '10px' }">
        <div>Usage: {{ usagePercent }}%</div>
        <div>Updated: {{ new Date(memoryStats.timestamp).toLocaleTimeString() }}</div>
      </div>
    </div>
  `,
};

/**
 * Vue plugin for global installation
 */
export const TrustformersVuePlugin = {
  install(app, options = {}) {
    // Register global properties
    app.config.globalProperties.$trustformers = {
      useTrustformersInit,
      useModel,
      useTokenizer,
      usePipeline,
      useTensor,
      usePerformance,
      useMemory,
      useDevTools,
    };

    // Register components if requested
    if (options.components !== false) {
      app.component('TrustformersProvider', TrustformersProviderTemplate);
      app.component('PerformanceMonitor', PerformanceMonitorTemplate);
      app.component('MemoryMonitor', MemoryMonitorTemplate);
    }

    // Provide global composition functions
    app.provide('trustformers', {
      useTrustformersInit,
      useModel,
      useTokenizer,
      usePipeline,
      useTensor,
      usePerformance,
      useMemory,
      useDevTools,
    });
  },
};

export default {
  // Composition functions
  useTrustformersInit,
  useModel,
  useTokenizer,
  usePipeline,
  useTensor,
  usePerformance,
  useMemory,
  useDevTools,

  // Component templates
  TrustformersProviderTemplate,
  PerformanceMonitorTemplate,
  MemoryMonitorTemplate,

  // Plugin
  TrustformersVuePlugin,
};
