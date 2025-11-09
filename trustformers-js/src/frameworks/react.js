/**
 * React hooks and components for TrustformeRS
 * Provides React-specific integrations for the TrustformeRS ecosystem
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
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
 * Hook for initializing TrustformeRS with React lifecycle management
 * @param {Object} options - Initialization options
 * @returns {Object} Initialization state and methods
 */
export function useTrustformersInit(options = {}) {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const initRef = useRef(false);

  const initTrustformers = useCallback(async () => {
    if (initRef.current) return;

    setIsLoading(true);
    setError(null);
    initRef.current = true;

    try {
      const caps = await initializeEnhanced({
        enableWebGL: true,
        enableMemoryPool: true,
        enableProfiling: true,
        enableZeroCopy: true,
        ...options,
      });

      setCapabilities(caps);
      setIsInitialized(true);
    } catch (err) {
      setError(err);
      initRef.current = false;
    } finally {
      setIsLoading(false);
    }
  }, [options]);

  useEffect(() => {
    initTrustformers();
  }, [initTrustformers]);

  const reinitialize = useCallback(
    async (newOptions = {}) => {
      initRef.current = false;
      setIsInitialized(false);
      setCapabilities(null);
      await initTrustformers(newOptions);
    },
    [initTrustformers]
  );

  return {
    isInitialized,
    isLoading,
    error,
    capabilities,
    reinitialize,
  };
}

/**
 * Hook for creating and managing models
 * @param {string|Object} modelConfig - Model configuration or type string
 * @param {Object} options - Model options
 * @returns {Object} Model state and methods
 */
export function useModel(modelConfig, options = {}) {
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const modelRef = useRef(null);

  const createModelInstance = useCallback(async () => {
    if (!modelConfig) return;

    setIsLoading(true);
    setError(null);

    try {
      const modelInstance = createModel(modelConfig);
      modelRef.current = modelInstance;
      setModel(modelInstance);
    } catch (err) {
      setError(err);
    } finally {
      setIsLoading(false);
    }
  }, [modelConfig]);

  useEffect(() => {
    createModelInstance();

    return () => {
      if (modelRef.current && modelRef.current.free) {
        modelRef.current.free();
      }
    };
  }, [createModelInstance]);

  const runInference = useCallback(
    async (inputs, inferenceOptions = {}) => {
      if (!model) throw new Error('Model not initialized');

      return await enhanced_inference.runInference(model, inputs, {
        profile: true,
        useMemoryPool: true,
        autoCleanup: true,
        ...inferenceOptions,
      });
    },
    [model]
  );

  const batchInference = useCallback(
    async (batchInputs, batchOptions = {}) => {
      if (!model) throw new Error('Model not initialized');

      return await enhanced_inference.batchInference(model, batchInputs, {
        batchSize: 32,
        profile: true,
        ...batchOptions,
      });
    },
    [model]
  );

  return {
    model,
    isLoading,
    error,
    runInference,
    batchInference,
  };
}

/**
 * Hook for creating and managing tokenizers
 * @param {string} type - Tokenizer type
 * @param {Object} vocab - Vocabulary object
 * @returns {Object} Tokenizer state and methods
 */
export function useTokenizer(type, vocab = null) {
  const [tokenizer, setTokenizer] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const tokenizerRef = useRef(null);

  const createTokenizerInstance = useCallback(async () => {
    if (!type) return;

    setIsLoading(true);
    setError(null);

    try {
      const tokenizerInstance = createTokenizer(type, vocab);
      tokenizerRef.current = tokenizerInstance;
      setTokenizer(tokenizerInstance);
    } catch (err) {
      setError(err);
    } finally {
      setIsLoading(false);
    }
  }, [type, vocab]);

  useEffect(() => {
    createTokenizerInstance();

    return () => {
      if (tokenizerRef.current && tokenizerRef.current.free) {
        tokenizerRef.current.free();
      }
    };
  }, [createTokenizerInstance]);

  const encode = useCallback(
    text => {
      if (!tokenizer) throw new Error('Tokenizer not initialized');
      return tokenizer.encode(text);
    },
    [tokenizer]
  );

  const decode = useCallback(
    tokens => {
      if (!tokenizer) throw new Error('Tokenizer not initialized');
      return tokenizer.decode(tokens);
    },
    [tokenizer]
  );

  const batchEncode = useCallback(
    texts => {
      if (!tokenizer) throw new Error('Tokenizer not initialized');
      return texts.map(text => tokenizer.encode(text));
    },
    [tokenizer]
  );

  return {
    tokenizer,
    isLoading,
    error,
    encode,
    decode,
    batchEncode,
  };
}

/**
 * Hook for creating and managing pipelines
 * @param {string} task - Pipeline task type
 * @param {Object} model - Model object
 * @param {Object} tokenizer - Tokenizer object
 * @param {Object} config - Pipeline configuration
 * @returns {Object} Pipeline state and methods
 */
export function usePipeline(task, model, tokenizer, config = {}) {
  const [pipeline, setPipeline] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const pipelineRef = useRef(null);

  const createPipelineInstance = useCallback(async () => {
    if (!task || !model || !tokenizer) return;

    setIsLoading(true);
    setError(null);

    try {
      let pipelineInstance;

      switch (task) {
        case 'text-generation':
          pipelineInstance = Pipeline.textGeneration(model, tokenizer, config);
          break;
        case 'text-classification':
          pipelineInstance = Pipeline.textClassification(model, tokenizer, config.labels || []);
          break;
        case 'question-answering':
          pipelineInstance = Pipeline.questionAnswering(model, tokenizer);
          break;
        default:
          pipelineInstance = await Pipeline.fromPretrained(task, config.modelName || 'default');
      }

      pipelineRef.current = pipelineInstance;
      setPipeline(pipelineInstance);
    } catch (err) {
      setError(err);
    } finally {
      setIsLoading(false);
    }
  }, [task, model, tokenizer, config]);

  useEffect(() => {
    createPipelineInstance();

    return () => {
      if (pipelineRef.current && pipelineRef.current.free) {
        pipelineRef.current.free();
      }
    };
  }, [createPipelineInstance]);

  const run = useCallback(
    async (input, options = {}) => {
      if (!pipeline) throw new Error('Pipeline not initialized');

      return await pipeline.run(input, {
        maxLength: 50,
        temperature: 0.7,
        ...options,
      });
    },
    [pipeline]
  );

  const batch = useCallback(
    async (inputs, options = {}) => {
      if (!pipeline) throw new Error('Pipeline not initialized');

      return Promise.all(inputs.map(input => pipeline.run(input, options)));
    },
    [pipeline]
  );

  return {
    pipeline,
    isLoading,
    error,
    run,
    batch,
  };
}

/**
 * Hook for tensor operations with React state management
 * @param {Array|null} initialData - Initial tensor data
 * @param {Array|null} initialShape - Initial tensor shape
 * @returns {Object} Tensor state and operations
 */
export function useTensor(initialData = null, initialShape = null) {
  const [tensorState, setTensorState] = useState(null);
  const [error, setError] = useState(null);
  const tensorRef = useRef(null);

  const createTensor = useCallback((data, shape) => {
    try {
      if (tensorRef.current && tensorRef.current.free) {
        tensorRef.current.free();
      }

      const newTensor = tensor(data, shape);
      tensorRef.current = newTensor;
      setTensorState({
        tensor: newTensor,
        shape: Array.from(newTensor.shape()),
        size: newTensor.size(),
        dtype: newTensor.dtype(),
      });
      setError(null);
    } catch (err) {
      setError(err);
    }
  }, []);

  useEffect(() => {
    if (initialData && initialShape) {
      createTensor(initialData, initialShape);
    }

    return () => {
      if (tensorRef.current && tensorRef.current.free) {
        tensorRef.current.free();
      }
    };
  }, [initialData, initialShape, createTensor]);

  const operations = useMemo(
    () => ({
      reshape: newShape => {
        if (!tensorState) return null;
        try {
          const reshaped = tensorState.tensor.reshape(newShape);
          return reshaped;
        } catch (err) {
          setError(err);
          return null;
        }
      },

      transpose: (dims = null) => {
        if (!tensorState) return null;
        try {
          const transposed = dims
            ? tensorState.tensor.transpose(dims)
            : tensorState.tensor.transpose();
          return transposed;
        } catch (err) {
          setError(err);
          return null;
        }
      },

      slice: (start, end, step = 1) => {
        if (!tensorState) return null;
        try {
          const sliced = tensorState.tensor.slice(start, end, step);
          return sliced;
        } catch (err) {
          setError(err);
          return null;
        }
      },

      add: other => {
        if (!tensorState) return null;
        try {
          const result = tensorState.tensor.add(other);
          return result;
        } catch (err) {
          setError(err);
          return null;
        }
      },

      multiply: other => {
        if (!tensorState) return null;
        try {
          const result = tensorState.tensor.mul(other);
          return result;
        } catch (err) {
          setError(err);
          return null;
        }
      },

      matmul: other => {
        if (!tensorState) return null;
        try {
          const result = tensorState.tensor.matmul(other);
          return result;
        } catch (err) {
          setError(err);
          return null;
        }
      },
    }),
    [tensorState]
  );

  return {
    tensor: tensorState?.tensor || null,
    shape: tensorState?.shape || null,
    size: tensorState?.size || null,
    dtype: tensorState?.dtype || null,
    error,
    createTensor,
    operations,
  };
}

/**
 * Hook for performance monitoring in React applications
 * @param {boolean} enabled - Whether monitoring is enabled
 * @returns {Object} Performance monitoring state and methods
 */
export function usePerformance(enabled = true) {
  const [metrics, setMetrics] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const intervalRef = useRef(null);

  const startSession = useCallback(
    (name, metadata = {}) => {
      if (!enabled) return null;

      const id = performance.startSession(name, {
        framework: 'react',
        timestamp: new Date().toISOString(),
        ...metadata,
      });
      setSessionId(id);
      return id;
    },
    [enabled]
  );

  const endSession = useCallback(() => {
    if (!enabled || !sessionId) return null;

    const report = performance.endSession();
    setSessionId(null);
    return report;
  }, [enabled, sessionId]);

  const profileOperation = useCallback(
    async (name, operation, metadata = {}) => {
      if (!enabled) return await operation();

      return await performance.profile(name, operation, {
        framework: 'react',
        ...metadata,
      });
    },
    [enabled]
  );

  const updateMetrics = useCallback(() => {
    if (!enabled) return;

    const currentMetrics = {
      memory: performance.getMemoryUsage(),
      report: performance.getReport(),
      timestamp: Date.now(),
    };
    setMetrics(currentMetrics);
  }, [enabled]);

  useEffect(() => {
    if (enabled) {
      updateMetrics();
      intervalRef.current = setInterval(updateMetrics, 5000); // Update every 5 seconds
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (sessionId) {
        endSession();
      }
    };
  }, [enabled, updateMetrics, sessionId, endSession]);

  const cleanup = useCallback(() => {
    performance.cleanup();
    updateMetrics();
  }, [updateMetrics]);

  return {
    metrics,
    sessionId,
    startSession,
    endSession,
    profileOperation,
    updateMetrics,
    cleanup,
  };
}

/**
 * Hook for memory management in React applications
 * @returns {Object} Memory management state and methods
 */
export function useMemory() {
  const [memoryStats, setMemoryStats] = useState(null);
  const intervalRef = useRef(null);

  const updateMemoryStats = useCallback(() => {
    const stats = memory.getStats();
    setMemoryStats({
      ...stats,
      timestamp: Date.now(),
    });
  }, []);

  useEffect(() => {
    updateMemoryStats();
    intervalRef.current = setInterval(updateMemoryStats, 3000); // Update every 3 seconds

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [updateMemoryStats]);

  const cleanup = useCallback(() => {
    if (typeof window !== 'undefined' && window.gc) {
      window.gc();
    }
    performance.cleanup();
    updateMemoryStats();
  }, [updateMemoryStats]);

  return {
    memoryStats,
    updateMemoryStats,
    cleanup,
  };
}

/**
 * Hook for development tools integration
 * @param {boolean} enabled - Whether dev tools are enabled
 * @returns {Object} Development tools state and methods
 */
export function useDevTools(enabled = process.env.NODE_ENV === 'development') {
  const [debugState, setDebugState] = useState(null);
  const [isEnabled, setIsEnabled] = useState(enabled);

  const enableDebug = useCallback((options = {}) => {
    devTools.debug.enable({
      trackTensors: true,
      trackOperations: true,
      validateOperations: true,
      ...options,
    });
    setIsEnabled(true);
  }, []);

  const disableDebug = useCallback(() => {
    devTools.debug.disable();
    setIsEnabled(false);
  }, []);

  const startDebugSession = useCallback(
    (name, metadata = {}) => {
      if (!isEnabled) return null;

      return devTools.debug.startSession(name, {
        framework: 'react',
        timestamp: new Date().toISOString(),
        ...metadata,
      });
    },
    [isEnabled]
  );

  const trackTensor = useCallback(
    (tensor, operation, metadata = {}) => {
      if (!isEnabled) return;

      devTools.debug.trackTensor(tensor, operation, {
        framework: 'react',
        ...metadata,
      });
    },
    [isEnabled]
  );

  const validateOperation = useCallback(
    (operation, tensors, options = {}) => {
      if (!isEnabled) return true;

      return devTools.debug.validateOperation(operation, tensors, options);
    },
    [isEnabled]
  );

  const generateReport = useCallback(() => {
    if (!isEnabled) return null;

    const report = devTools.debug.generateReport();
    setDebugState(report);
    return report;
  }, [isEnabled]);

  const analyzeComprehensive = useCallback(
    (options = {}) => {
      if (!isEnabled) return null;

      return devTools.analyze.comprehensive({
        includeDebug: true,
        ...options,
      });
    },
    [isEnabled]
  );

  return {
    isEnabled,
    debugState,
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
 * React component for TrustformeRS initialization status
 */
export function TrustformersProvider({ children, options = {}, onInit, onError }) {
  const { isInitialized, isLoading, error, capabilities } = useTrustformersInit(options);

  useEffect(() => {
    if (isInitialized && onInit) {
      onInit(capabilities);
    }
  }, [isInitialized, capabilities, onInit]);

  useEffect(() => {
    if (error && onError) {
      onError(error);
    }
  }, [error, onError]);

  if (error) {
    return React.createElement('div', { className: 'trustformers-error' },
      React.createElement('h3', null, 'TrustformeRS Initialization Error'),
      React.createElement('p', null, error.message),
      React.createElement('pre', null, error.stack)
    );
  }

  if (isLoading) {
    return React.createElement('div', { className: 'trustformers-loading' },
      React.createElement('p', null, 'Initializing TrustformeRS...')
    );
  }

  return React.createElement('div', {
    className: 'trustformers-provider',
    'data-initialized': isInitialized
  }, children);
}

/**
 * React component for performance monitoring display
 */
export function PerformanceMonitor({ enabled = true, updateInterval = 5000 }) {
  const { metrics } = usePerformance(enabled);
  const [expanded, setExpanded] = useState(false);

  if (!enabled || !metrics) return null;

  return React.createElement('div', {
    className: 'performance-monitor',
    style: {
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
      cursor: 'pointer',
    },
    onClick: () => setExpanded(!expanded)
  },
    React.createElement('div', null, '🚀 TrustformeRS Performance'),
    metrics.memory && React.createElement('div', null, `Memory: ${(metrics.memory.used / 1024 / 1024).toFixed(1)}MB`),
    expanded && metrics.report && React.createElement('div', {
      style: { marginTop: '10px', fontSize: '10px' }
    },
      React.createElement('div', null,
        'Capabilities: ',
        Object.entries(metrics.report.capabilities)
          .filter(([, v]) => v)
          .map(([k]) => k)
          .join(', ')
      ),
      metrics.report.profiling && React.createElement('div', null, `Operations: ${metrics.report.profiling.totalOperations || 0}`)
    )
  );
}

/**
 * React component for memory usage display
 */
export function MemoryMonitor({ enabled = true }) {
  const { memoryStats } = useMemory();
  const [expanded, setExpanded] = useState(false);

  if (!enabled || !memoryStats) return null;

  const usedMB = memoryStats.used ? (memoryStats.used / 1024 / 1024).toFixed(1) : 'N/A';
  const limitMB = memoryStats.limit ? (memoryStats.limit / 1024 / 1024).toFixed(1) : 'N/A';

  return React.createElement('div', {
    className: 'memory-monitor',
    style: {
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
      cursor: 'pointer',
    },
    onClick: () => setExpanded(!expanded)
  },
    React.createElement('div', null, '💾 Memory'),
    React.createElement('div', null, `${usedMB}MB / ${limitMB}MB`),
    expanded && React.createElement('div', {
      style: { marginTop: '10px', fontSize: '10px' }
    },
      React.createElement('div', null, `Usage: ${((memoryStats.used / memoryStats.limit) * 100).toFixed(1)}%`),
      React.createElement('div', null, `Updated: ${new Date(memoryStats.timestamp).toLocaleTimeString()}`)
    )
  );
}

/**
 * Higher-order component for automatic tensor cleanup
 */
export function withTensorCleanup(WrappedComponent) {
  return function TensorCleanupWrapper(props) {
    const tensorsRef = useRef(new Set());

    const registerTensor = useCallback(tensor => {
      tensorsRef.current.add(tensor);
    }, []);

    const unregisterTensor = useCallback(tensor => {
      tensorsRef.current.delete(tensor);
    }, []);

    useEffect(
      () => () => {
        // Cleanup all registered tensors on unmount
        for (const tensor of tensorsRef.current) {
          if (tensor && tensor.free) {
            tensor.free();
          }
        }
        tensorsRef.current.clear();
      },
      []
    );

    return React.createElement(WrappedComponent, {
      ...props,
      registerTensor,
      unregisterTensor,
    });
  };
}

export default {
  useTrustformersInit,
  useModel,
  useTokenizer,
  usePipeline,
  useTensor,
  usePerformance,
  useMemory,
  useDevTools,
  TrustformersProvider,
  PerformanceMonitor,
  MemoryMonitor,
  withTensorCleanup,
};
