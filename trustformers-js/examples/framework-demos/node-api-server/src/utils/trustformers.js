/**
 * TrustformeRS Initialization and Management Utility
 * Handles model loading, pipeline management, and ML operations
 */

import logger from './logger.js';

// Global TrustformeRS instance and pipelines
let trustformersInstance = null;
let pipelines = new Map();
let modelCache = new Map();
let initializationPromise = null;

/**
 * Initialize TrustformeRS with comprehensive pipeline setup
 * @returns {Promise<void>}
 */
export async function initializeTrustformers() {
  if (initializationPromise) {
    return initializationPromise;
  }
  
  initializationPromise = _doInitialization();
  return initializationPromise;
}

async function _doInitialization() {
  const startTime = Date.now();
  
  try {
    logger.info('Initializing TrustformeRS Node.js integration...');
    
    // Dynamic import of TrustformeRS
    const { TrustformersNode } = await import('@trustformers/js/frameworks');
    
    // Initialize the Node.js integration
    trustformersInstance = await TrustformersNode.initialize({
      debug: process.env.NODE_ENV === 'development',
      cacheModels: true,
      maxCacheSize: parseInt(process.env.MAX_CACHE_SIZE) || 1024, // MB
      enablePerformanceOptimizations: true,
      devicePreference: process.env.DEVICE_PREFERENCE || 'auto', // auto, cpu, cuda, mps
    });
    
    logger.info('TrustformeRS Node.js integration initialized successfully');
    
    // Initialize commonly used pipelines
    await initializePipelines();
    
    const duration = Date.now() - startTime;
    logger.performance('trustformers_initialization', duration, {
      pipelinesLoaded: pipelines.size,
      cacheEnabled: true,
    });
    
    logger.lifecycle('trustformers_ready', {
      availablePipelines: Array.from(pipelines.keys()),
      duration: `${duration}ms`,
    });
    
  } catch (error) {
    logger.errorWithContext(error, {
      operation: 'trustformers_initialization',
      duration: Date.now() - startTime,
    });
    throw error;
  }
}

/**
 * Initialize commonly used pipelines
 */
async function initializePipelines() {
  const pipelineConfigs = [
    { name: 'sentiment-analysis', task: 'sentiment-analysis' },
    { name: 'text-classification', task: 'text-classification' },
    { name: 'text-generation', task: 'text-generation' },
    { name: 'question-answering', task: 'question-answering' },
    { name: 'token-classification', task: 'token-classification' },
    { name: 'summarization', task: 'summarization' },
    { name: 'translation', task: 'translation' },
  ];
  
  const initPromises = pipelineConfigs.map(async (config) => {
    try {
      const startTime = Date.now();
      const pipeline = await trustformersInstance.createPipeline(config.task, {
        model: config.model || 'default',
        tokenizer: config.tokenizer || 'default',
        device: config.device || 'auto',
      });
      
      pipelines.set(config.name, pipeline);
      
      const duration = Date.now() - startTime;
      logger.info(`Pipeline '${config.name}' initialized`, {
        task: config.task,
        duration: `${duration}ms`,
      });
      
      return { name: config.name, success: true, duration };
    } catch (error) {
      logger.errorWithContext(error, {
        operation: 'pipeline_initialization',
        pipeline: config.name,
        task: config.task,
      });
      return { name: config.name, success: false, error: error.message };
    }
  });
  
  const results = await Promise.allSettled(initPromises);
  const successful = results.filter(r => r.status === 'fulfilled' && r.value.success);
  
  logger.info(`Initialized ${successful.length}/${pipelineConfigs.length} pipelines`);
}

/**
 * Get or create a pipeline
 * @param {string} task - The task type
 * @param {Object} options - Pipeline options
 * @returns {Promise<Object>} Pipeline instance
 */
export async function getPipeline(task, options = {}) {
  const cacheKey = `${task}:${JSON.stringify(options)}`;
  
  // Check cache first
  if (pipelines.has(cacheKey)) {
    return pipelines.get(cacheKey);
  }
  
  // Check if we have a default pipeline for this task
  if (pipelines.has(task)) {
    return pipelines.get(task);
  }
  
  // Create new pipeline
  try {
    const startTime = Date.now();
    const pipeline = await trustformersInstance.createPipeline(task, options);
    pipelines.set(cacheKey, pipeline);
    
    const duration = Date.now() - startTime;
    logger.performance('pipeline_creation', duration, {
      task,
      options,
      cacheKey,
    });
    
    return pipeline;
  } catch (error) {
    logger.errorWithContext(error, {
      operation: 'pipeline_creation',
      task,
      options,
    });
    throw error;
  }
}

/**
 * Perform sentiment analysis
 * @param {string} text - Text to analyze
 * @param {Object} options - Analysis options
 * @returns {Promise<Object>} Sentiment analysis result
 */
export async function analyzeSentiment(text, options = {}) {
  const startTime = Date.now();
  
  try {
    const pipeline = await getPipeline('sentiment-analysis');
    const result = await pipeline.predict(text, {
      returnAllScores: options.returnAllScores || false,
      ...options,
    });
    
    const duration = Date.now() - startTime;
    logger.mlOperation('sentiment_analysis', 'default', text, result, duration, {
      textLength: text.length,
      options,
    });
    
    return {
      success: true,
      result,
      metadata: {
        duration: `${duration}ms`,
        textLength: text.length,
        model: 'default',
      },
    };
  } catch (error) {
    const duration = Date.now() - startTime;
    logger.errorWithContext(error, {
      operation: 'sentiment_analysis',
      textLength: text.length,
      duration: `${duration}ms`,
    });
    throw error;
  }
}

/**
 * Perform text classification
 * @param {string} text - Text to classify
 * @param {Array} labels - Possible labels
 * @param {Object} options - Classification options
 * @returns {Promise<Object>} Classification result
 */
export async function classifyText(text, labels = [], options = {}) {
  const startTime = Date.now();
  
  try {
    const pipeline = await getPipeline('text-classification');
    const result = await pipeline.predict(text, {
      candidateLabels: labels,
      ...options,
    });
    
    const duration = Date.now() - startTime;
    logger.mlOperation('text_classification', 'default', text, result, duration, {
      textLength: text.length,
      labelsCount: labels.length,
      options,
    });
    
    return {
      success: true,
      result,
      metadata: {
        duration: `${duration}ms`,
        textLength: text.length,
        labelsCount: labels.length,
        model: 'default',
      },
    };
  } catch (error) {
    const duration = Date.now() - startTime;
    logger.errorWithContext(error, {
      operation: 'text_classification',
      textLength: text.length,
      labelsCount: labels.length,
      duration: `${duration}ms`,
    });
    throw error;
  }
}

/**
 * Generate text
 * @param {string} prompt - Text prompt
 * @param {Object} options - Generation options
 * @returns {Promise<Object>} Generated text result
 */
export async function generateText(prompt, options = {}) {
  const startTime = Date.now();
  
  try {
    const pipeline = await getPipeline('text-generation');
    const result = await pipeline.predict(prompt, {
      maxLength: options.maxLength || 100,
      temperature: options.temperature || 0.7,
      topP: options.topP || 0.9,
      doSample: options.doSample !== false,
      ...options,
    });
    
    const duration = Date.now() - startTime;
    logger.mlOperation('text_generation', 'default', prompt, result, duration, {
      promptLength: prompt.length,
      maxLength: options.maxLength || 100,
      options,
    });
    
    return {
      success: true,
      result,
      metadata: {
        duration: `${duration}ms`,
        promptLength: prompt.length,
        model: 'default',
        options,
      },
    };
  } catch (error) {
    const duration = Date.now() - startTime;
    logger.errorWithContext(error, {
      operation: 'text_generation',
      promptLength: prompt.length,
      duration: `${duration}ms`,
    });
    throw error;
  }
}

/**
 * Answer questions based on context
 * @param {string} question - Question to answer
 * @param {string} context - Context for answering
 * @param {Object} options - QA options
 * @returns {Promise<Object>} Answer result
 */
export async function answerQuestion(question, context, options = {}) {
  const startTime = Date.now();
  
  try {
    const pipeline = await getPipeline('question-answering');
    const result = await pipeline.predict({
      question,
      context,
      ...options,
    });
    
    const duration = Date.now() - startTime;
    logger.mlOperation('question_answering', 'default', { question, context }, result, duration, {
      questionLength: question.length,
      contextLength: context.length,
      options,
    });
    
    return {
      success: true,
      result,
      metadata: {
        duration: `${duration}ms`,
        questionLength: question.length,
        contextLength: context.length,
        model: 'default',
      },
    };
  } catch (error) {
    const duration = Date.now() - startTime;
    logger.errorWithContext(error, {
      operation: 'question_answering',
      questionLength: question.length,
      contextLength: context.length,
      duration: `${duration}ms`,
    });
    throw error;
  }
}

/**
 * Summarize text
 * @param {string} text - Text to summarize
 * @param {Object} options - Summarization options
 * @returns {Promise<Object>} Summary result
 */
export async function summarizeText(text, options = {}) {
  const startTime = Date.now();
  
  try {
    const pipeline = await getPipeline('summarization');
    const result = await pipeline.predict(text, {
      maxLength: options.maxLength || 130,
      minLength: options.minLength || 30,
      ...options,
    });
    
    const duration = Date.now() - startTime;
    logger.mlOperation('summarization', 'default', text, result, duration, {
      textLength: text.length,
      maxLength: options.maxLength || 130,
      options,
    });
    
    return {
      success: true,
      result,
      metadata: {
        duration: `${duration}ms`,
        textLength: text.length,
        model: 'default',
        options,
      },
    };
  } catch (error) {
    const duration = Date.now() - startTime;
    logger.errorWithContext(error, {
      operation: 'summarization',
      textLength: text.length,
      duration: `${duration}ms`,
    });
    throw error;
  }
}

/**
 * Get system information and model status
 * @returns {Object} System information
 */
export function getSystemInfo() {
  return {
    status: trustformersInstance ? 'ready' : 'initializing',
    pipelines: {
      available: Array.from(pipelines.keys()),
      count: pipelines.size,
    },
    cache: {
      models: modelCache.size,
      size: `${JSON.stringify([...modelCache.values()]).length} bytes`,
    },
    memory: {
      used: process.memoryUsage(),
      uptime: process.uptime(),
    },
    environment: {
      nodeVersion: process.version,
      platform: process.platform,
      arch: process.arch,
    },
  };
}

/**
 * Health check for TrustformeRS components
 * @returns {Promise<Object>} Health status
 */
export async function healthCheck() {
  const checks = {
    trustformers: false,
    pipelines: false,
    memory: false,
  };
  
  try {
    // Check TrustformeRS initialization
    checks.trustformers = !!trustformersInstance;
    
    // Check pipeline availability
    checks.pipelines = pipelines.size > 0;
    
    // Check memory usage
    const memUsage = process.memoryUsage();
    const memLimit = parseInt(process.env.MEMORY_LIMIT) || 2048; // MB
    checks.memory = (memUsage.heapUsed / 1024 / 1024) < memLimit;
    
    const allHealthy = Object.values(checks).every(check => check);
    
    return {
      status: allHealthy ? 'healthy' : 'unhealthy',
      checks,
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
    };
  } catch (error) {
    logger.errorWithContext(error, { operation: 'health_check' });
    return {
      status: 'unhealthy',
      checks,
      error: error.message,
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Clear model cache
 */
export function clearCache() {
  modelCache.clear();
  pipelines.clear();
  logger.info('Model cache cleared');
}

/**
 * Get cache statistics
 * @returns {Object} Cache statistics
 */
export function getCacheStats() {
  return {
    models: modelCache.size,
    pipelines: pipelines.size,
    totalSize: JSON.stringify([...modelCache.values(), ...pipelines.values()]).length,
  };
}