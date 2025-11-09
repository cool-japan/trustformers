/**
 * TrustformeRS Pipeline Module
 * Tree-shakable pipeline utilities for common ML tasks
 */

// Re-export pipeline-related functions from main module
export {
  Pipeline,
  PipelineType,
} from '../index.js';

// Pipeline configuration presets
export const PipelinePresets = {
  textClassification: {
    task: 'text-classification',
    returnAllScores: false,
    topK: 1,
  },

  textGeneration: {
    task: 'text-generation',
    maxNewTokens: 50,
    temperature: 0.7,
    doSample: true,
    topK: 50,
    topP: 0.9,
  },

  questionAnswering: {
    task: 'question-answering',
    topK: 1,
    maxAnswerLength: 100,
  },

  summarization: {
    task: 'summarization',
    maxLength: 150,
    minLength: 30,
    lengthPenalty: 2.0,
    noRepeatNgramSize: 3,
  },

  translation: {
    task: 'translation',
    maxLength: 512,
    numBeams: 4,
    earlyStoppingLimit: true,
  },

  fillMask: {
    task: 'fill-mask',
    topK: 5,
    targetMask: '[MASK]',
  },
};

// Pipeline manager for handling multiple pipelines
export class PipelineManager {
  constructor() {
    this.pipelines = new Map();
    this.defaultConfigs = new Map();
  }

  /**
   * Register a pipeline
   */
  async register(name, task, modelPath, config = {}) {
    const pipeline = new Pipeline(task, modelPath, config);
    this.pipelines.set(name, pipeline);
    this.defaultConfigs.set(name, config);
    return pipeline;
  }

  /**
   * Get a registered pipeline
   */
  get(name) {
    return this.pipelines.get(name);
  }

  /**
   * Check if pipeline exists
   */
  has(name) {
    return this.pipelines.has(name);
  }

  /**
   * Run inference on a named pipeline
   */
  async run(name, inputs, options = {}) {
    const pipeline = this.pipelines.get(name);
    if (!pipeline) {
      throw new Error(`Pipeline '${name}' not found. Register it first.`);
    }

    return await pipeline(inputs, options);
  }

  /**
   * Remove a pipeline
   */
  async remove(name) {
    const pipeline = this.pipelines.get(name);
    if (pipeline && pipeline.dispose) {
      await pipeline.dispose();
    }
    this.pipelines.delete(name);
    this.defaultConfigs.delete(name);
  }

  /**
   * List all registered pipelines
   */
  list() {
    return Array.from(this.pipelines.keys());
  }

  /**
   * Clean up all pipelines
   */
  async dispose() {
    for (const [name, pipeline] of this.pipelines) {
      if (pipeline.dispose) {
        await pipeline.dispose();
      }
    }
    this.pipelines.clear();
    this.defaultConfigs.clear();
  }
}

// Batch processing utilities
export class BatchProcessor {
  constructor(pipeline, batchSize = 8) {
    this.pipeline = pipeline;
    this.batchSize = batchSize;
  }

  /**
   * Process inputs in batches
   */
  async *processBatches(inputs, options = {}) {
    for (let i = 0; i < inputs.length; i += this.batchSize) {
      const batch = inputs.slice(i, i + this.batchSize);
      const results = await Promise.all(batch.map(input => this.pipeline(input, options)));
      yield results;
    }
  }

  /**
   * Process all inputs and return flat results
   */
  async processAll(inputs, options = {}) {
    const results = [];
    for await (const batch of this.processBatches(inputs, options)) {
      results.push(...batch);
    }
    return results;
  }

  /**
   * Process inputs with progress tracking
   */
  async processWithProgress(inputs, options = {}, progressCallback) {
    const results = [];
    let processed = 0;

    for await (const batch of this.processBatches(inputs, options)) {
      results.push(...batch);
      processed += batch.length;

      if (progressCallback) {
        progressCallback({
          processed,
          total: inputs.length,
          progress: processed / inputs.length,
          batch: batch.length,
        });
      }
    }

    return results;
  }
}

// Pipeline composition utilities
export class PipelineComposer {
  constructor() {
    this.steps = [];
  }

  /**
   * Add a processing step
   */
  addStep(name, processor) {
    this.steps.push({ name, processor });
    return this;
  }

  /**
   * Add a pipeline step
   */
  addPipeline(name, pipeline, options = {}) {
    this.steps.push({
      name,
      processor: input => pipeline(input, options),
    });
    return this;
  }

  /**
   * Add a transformation step
   */
  addTransform(name, transform) {
    this.steps.push({ name, processor: transform });
    return this;
  }

  /**
   * Execute the composed pipeline
   */
  async execute(input) {
    let result = input;
    const stepResults = {};

    for (const step of this.steps) {
      try {
        result = await step.processor(result);
        stepResults[step.name] = result;
      } catch (error) {
        throw new Error(`Error in step '${step.name}': ${error.message}`);
      }
    }

    return {
      result,
      stepResults,
    };
  }

  /**
   * Execute with detailed timing
   */
  async executeWithTiming(input) {
    let result = input;
    const stepResults = {};
    const timing = {};

    for (const step of this.steps) {
      const startTime = performance.now();
      try {
        result = await step.processor(result);
        stepResults[step.name] = result;
        timing[step.name] = performance.now() - startTime;
      } catch (error) {
        timing[step.name] = performance.now() - startTime;
        throw new Error(`Error in step '${step.name}': ${error.message}`);
      }
    }

    return {
      result,
      stepResults,
      timing,
      totalTime: Object.values(timing).reduce((sum, time) => sum + time, 0),
    };
  }
}

// Pipeline validation utilities
export class PipelineValidator {
  /**
   * Validate pipeline task compatibility
   */
  static validateTask(task, modelType) {
    const taskModelMap = {
      'text-classification': ['bert', 'roberta', 'distilbert'],
      'text-generation': ['gpt', 'gpt2', 't5'],
      'question-answering': ['bert', 'roberta', 'distilbert'],
      summarization: ['t5', 'bart', 'pegasus'],
      translation: ['t5', 'marian', 'helsinki-nlp'],
    };

    const compatibleModels = taskModelMap[task];
    if (compatibleModels && !compatibleModels.some(model => modelType.includes(model))) {
      console.warn(`Model type '${modelType}' may not be optimal for task '${task}'`);
    }

    return true;
  }

  /**
   * Validate pipeline inputs
   */
  static validateInputs(task, inputs) {
    switch (task) {
      case 'text-classification':
      case 'text-generation':
      case 'fill-mask':
        if (typeof inputs !== 'string' && !Array.isArray(inputs)) {
          throw new Error(`Task '${task}' expects string or array inputs`);
        }
        break;

      case 'question-answering':
        if (!inputs.question || !inputs.context) {
          throw new Error('Question answering requires both question and context');
        }
        break;

      case 'translation':
        if (typeof inputs !== 'string') {
          throw new Error('Translation expects string input');
        }
        break;
    }

    return true;
  }
}

export default {
  PipelinePresets,
  PipelineManager,
  BatchProcessor,
  PipelineComposer,
  PipelineValidator,
};
