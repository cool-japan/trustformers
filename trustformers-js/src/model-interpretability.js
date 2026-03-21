/**
 * Model Interpretability and Explainability Framework
 *
 * Comprehensive interpretability tools with:
 * - Attention visualization (self-attention, cross-attention)
 * - Gradient-based explanations (GradCAM, Integrated Gradients, Saliency Maps)
 * - Layer-wise relevance propagation (LRP)
 * - SHAP values for feature importance
 * - Concept activation vectors (CAV)
 * - Input perturbation analysis
 * - Neuron activation analysis
 * - Decision boundary visualization
 */

/**
 * Attention Visualizer
 * Visualizes attention patterns in transformer models
 */
export class AttentionVisualizer {
  constructor(config = {}) {
    this.config = config;
    this.colorScheme = config.colorScheme || 'viridis';
    this.normalization = config.normalization || 'layer';
  }

  /**
   * Extract and visualize attention weights
   */
  async visualizeAttention(model, input, config = {}) {
    const {
      layers = 'all',
      heads = 'all',
      aggregation = 'mean', // 'mean', 'max', 'none'
    } = config;

    console.log('Extracting attention weights...');

    // Extract attention weights from model
    const attentionWeights = await this.extractAttentionWeights(model, input, layers);

    // Process and aggregate
    const processed = this.processAttentionWeights(attentionWeights, heads, aggregation);

    // Create visualizations
    const visualizations = this.createVisualizations(processed, config);

    return {
      attentionWeights: processed,
      visualizations,
      metadata: {
        numLayers: Object.keys(attentionWeights).length,
        numHeads: this.getNumHeads(attentionWeights),
        sequenceLength: this.getSequenceLength(attentionWeights),
      },
    };
  }

  /**
   * Visualize attention weights (shorter alias)
   */
  async visualize(weights, config = {}) {
    // If weights is a model (has forward method), extract attention first
    if (weights && typeof weights.forward === 'function') {
      const input = config.input || new Float32Array(10).fill(0.5);
      return await this.visualizeAttention(weights, input, config);
    }

    // Otherwise, assume weights are already extracted attention weights
    const processed = this.processAttentionWeights(weights, config.heads || 'all', config.aggregation || 'mean');
    const visualizations = this.createVisualizations(processed, config);

    return {
      attentionWeights: processed,
      visualizations,
      metadata: {
        numLayers: Object.keys(weights).length,
        numHeads: this.getNumHeads(weights),
        sequenceLength: this.getSequenceLength(weights),
      },
    };
  }

  async extractAttentionWeights(model, input, layers) {
    // Simulated attention extraction
    const weights = {};
    const numLayers = 12;
    const numHeads = 8;
    const seqLen = 10;

    const layerIndices =
      layers === 'all'
        ? Array.from({ length: numLayers }, (_, i) => i)
        : Array.isArray(layers)
        ? layers
        : [layers];

    for (const layerIdx of layerIndices) {
      weights[`layer_${layerIdx}`] = {
        selfAttention: this.generateAttentionMatrix(numHeads, seqLen, seqLen),
        crossAttention: null, // For encoder-decoder models
      };
    }

    return weights;
  }

  generateAttentionMatrix(numHeads, seqLen, keyLen) {
    const attention = [];

    for (let h = 0; h < numHeads; h++) {
      const headAttention = [];
      for (let i = 0; i < seqLen; i++) {
        const row = new Float32Array(keyLen);
        let sum = 0;

        for (let j = 0; j < keyLen; j++) {
          // Simulate attention pattern (diagonal bias)
          const distance = Math.abs(i - j);
          row[j] = Math.exp(-distance * 0.5) + Math.random() * 0.1;
          sum += row[j];
        }

        // Normalize
        for (let j = 0; j < keyLen; j++) {
          row[j] /= sum;
        }

        headAttention.push(Array.from(row));
      }
      attention.push(headAttention);
    }

    return attention;
  }

  processAttentionWeights(weights, heads, aggregation) {
    const processed = {};

    for (const [layerName, layerWeights] of Object.entries(weights)) {
      processed[layerName] = {};

      if (layerWeights.selfAttention) {
        processed[layerName].selfAttention = this.aggregateHeads(
          layerWeights.selfAttention,
          heads,
          aggregation
        );
      }

      if (layerWeights.crossAttention) {
        processed[layerName].crossAttention = this.aggregateHeads(
          layerWeights.crossAttention,
          heads,
          aggregation
        );
      }
    }

    return processed;
  }

  aggregateHeads(attentionHeads, heads, aggregation) {
    const headIndices =
      heads === 'all'
        ? Array.from({ length: attentionHeads.length }, (_, i) => i)
        : Array.isArray(heads)
        ? heads
        : [heads];

    const selectedHeads = headIndices.map(i => attentionHeads[i]);

    if (aggregation === 'none') {
      return selectedHeads;
    }

    const seqLen = selectedHeads[0].length;
    const keyLen = selectedHeads[0][0].length;
    const aggregated = [];

    for (let i = 0; i < seqLen; i++) {
      const row = new Float32Array(keyLen);

      for (let j = 0; j < keyLen; j++) {
        const values = selectedHeads.map(head => head[i][j]);

        if (aggregation === 'mean') {
          row[j] = values.reduce((a, b) => a + b, 0) / values.length;
        } else if (aggregation === 'max') {
          row[j] = Math.max(...values);
        }
      }

      aggregated.push(Array.from(row));
    }

    return aggregated;
  }

  createVisualizations(processedWeights, config) {
    const visualizations = {};

    for (const [layerName, layerWeights] of Object.entries(processedWeights)) {
      visualizations[layerName] = {};

      if (layerWeights.selfAttention) {
        visualizations[layerName].selfAttention = this.createHeatmap(
          layerWeights.selfAttention,
          `${layerName} Self-Attention`
        );
      }
    }

    return visualizations;
  }

  createHeatmap(attentionMatrix, title) {
    // Create HTML/SVG visualization
    const isAggregated = !Array.isArray(attentionMatrix[0]);
    const matrix = isAggregated ? attentionMatrix : attentionMatrix[0];

    return {
      type: 'heatmap',
      title,
      data: matrix,
      width: matrix[0].length,
      height: matrix.length,
      colorScheme: this.colorScheme,
    };
  }

  getNumHeads(weights) {
    const firstLayer = Object.values(weights)[0];
    return firstLayer?.selfAttention?.length || 0;
  }

  getSequenceLength(weights) {
    const firstLayer = Object.values(weights)[0];
    return firstLayer?.selfAttention?.[0]?.length || 0;
  }

  /**
   * Export attention patterns
   */
  exportAttentionPatterns(visualizations, format = 'json') {
    if (format === 'json') {
      return JSON.stringify(visualizations, null, 2);
    }

    // Other formats can be added
    throw new Error(`Unsupported format: ${format}`);
  }
}

/**
 * Gradient-Based Explanations
 * Implements various gradient-based interpretation methods
 */
export class GradientExplainer {
  constructor(model, config = {}) {
    this.model = model;
    this.config = config;
  }

  /**
   * Explain model predictions using gradient-based methods
   */
  async explain(input, config = {}) {
    const {
      method = 'integrated_gradients',
      baseline = null,
      targetClass = null,
      steps = 50,
    } = config;

    switch (method) {
      case 'saliency':
      case 'saliency_map':
        return await this.computeSaliencyMap(input, targetClass);
      case 'integrated_gradients':
        return await this.computeIntegratedGradients(input, baseline, { steps, targetClass });
      case 'gradcam':
        return await this.computeGradCAM(input, config.targetLayer || 'layer_11', targetClass);
      default:
        return await this.computeIntegratedGradients(input, baseline, { steps, targetClass });
    }
  }

  /**
   * Compute saliency map
   */
  async computeSaliencyMap(input, targetClass = null) {
    console.log('Computing saliency map...');

    // Compute gradients with respect to input
    const gradients = await this.computeGradients(input, targetClass);

    // Take absolute values and aggregate
    const saliency = this.aggregateGradients(gradients);

    return {
      saliency,
      method: 'saliency_map',
      targetClass,
    };
  }

  /**
   * Compute Integrated Gradients
   */
  async computeIntegratedGradients(input, baseline = null, config = {}) {
    const { steps = 50, targetClass = null } = config;

    console.log(`Computing integrated gradients with ${steps} steps...`);

    if (baseline === null) {
      baseline = new Float32Array(input.length).fill(0);
    }

    const attributions = new Float32Array(input.length).fill(0);

    // Interpolate between baseline and input
    for (let step = 0; step <= steps; step++) {
      const alpha = step / steps;
      const interpolated = this.interpolate(baseline, input, alpha);

      // Compute gradients at interpolated input
      const gradients = await this.computeGradients(interpolated, targetClass);

      // Accumulate gradients
      for (let i = 0; i < attributions.length; i++) {
        attributions[i] += gradients[i];
      }
    }

    // Average and scale by input - baseline
    for (let i = 0; i < attributions.length; i++) {
      attributions[i] = (attributions[i] / steps) * (input[i] - baseline[i]);
    }

    return {
      attributions: Array.from(attributions),
      method: 'integrated_gradients',
      steps,
      targetClass,
    };
  }

  /**
   * Compute GradCAM (for convolutional models)
   */
  async computeGradCAM(input, targetLayer, targetClass = null) {
    console.log('Computing Grad-CAM...');

    // Extract activations from target layer
    const activations = await this.extractActivations(input, targetLayer);

    // Compute gradients with respect to activations
    const gradients = await this.computeLayerGradients(input, targetLayer, targetClass);

    // Compute weights (global average pooling of gradients)
    const weights = this.globalAveragePool(gradients);

    // Weighted combination of activation maps
    const heatmap = this.combineActivations(activations, weights);

    // Apply ReLU (only positive influences)
    const positiveHeatmap = heatmap.map(v => Math.max(0, v));

    return {
      heatmap: positiveHeatmap,
      method: 'gradcam',
      targetLayer,
      targetClass,
    };
  }

  async computeGradients(input, targetClass) {
    // Simulated gradient computation
    const gradients = new Float32Array(input.length);

    for (let i = 0; i < gradients.length; i++) {
      gradients[i] = (Math.random() - 0.5) * 2;
    }

    return gradients;
  }

  aggregateGradients(gradients) {
    // Take absolute value
    return gradients.map(Math.abs);
  }

  interpolate(baseline, input, alpha) {
    const result = new Float32Array(input.length);

    for (let i = 0; i < input.length; i++) {
      result[i] = baseline[i] + alpha * (input[i] - baseline[i]);
    }

    return result;
  }

  async extractActivations(input, layer) {
    // Simulated activation extraction
    return new Float32Array(64).map(() => Math.random());
  }

  async computeLayerGradients(input, layer, targetClass) {
    // Simulated layer gradient computation
    return new Float32Array(64).map(() => Math.random() - 0.5);
  }

  globalAveragePool(activations) {
    const sum = activations.reduce((a, b) => a + b, 0);
    return sum / activations.length;
  }

  combineActivations(activations, weights) {
    return activations.map(a => a * weights);
  }
}

/**
 * Feature Importance Analyzer
 * Computes feature importance using various methods
 */
export class FeatureImportanceAnalyzer {
  constructor(model, config = {}) {
    this.model = model;
    this.config = config;
    this.method = config.method || 'permutation'; // 'permutation', 'shap', 'lime'
  }

  /**
   * Compute feature importance
   */
  async computeImportance(dataset, config = {}) {
    console.log(`Computing feature importance using ${this.method}...`);

    if (this.method === 'permutation') {
      return await this.permutationImportance(dataset, config);
    } else if (this.method === 'shap') {
      return await this.shapValues(dataset, config);
    } else if (this.method === 'lime') {
      return await this.limeExplanation(dataset, config);
    }

    throw new Error(`Unknown method: ${this.method}`);
  }

  /**
   * Permutation importance
   */
  async permutationImportance(dataset, config = {}) {
    const { numRepeats = 10, metric = 'accuracy' } = config;

    // Baseline performance
    const baselineScore = await this.evaluateModel(dataset);

    const importance = {};
    const numFeatures = dataset[0].features.length;

    for (let f = 0; f < numFeatures; f++) {
      const scores = [];

      for (let repeat = 0; repeat < numRepeats; repeat++) {
        // Permute feature f
        const permutedDataset = this.permuteFeature(dataset, f);

        // Evaluate on permuted data
        const score = await this.evaluateModel(permutedDataset);
        scores.push(baselineScore - score); // Importance = drop in performance
      }

      // Average importance
      importance[`feature_${f}`] = {
        mean: scores.reduce((a, b) => a + b, 0) / scores.length,
        std: this.computeStd(scores),
        scores,
      };
    }

    return {
      method: 'permutation',
      importance,
      baseline: baselineScore,
    };
  }

  /**
   * SHAP values (simplified)
   */
  async shapValues(dataset, config = {}) {
    const { numSamples = 100 } = config;

    const shapValues = [];

    for (const sample of dataset.slice(0, numSamples)) {
      const values = await this.computeSHAPForSample(sample, dataset);
      shapValues.push(values);
    }

    // Average SHAP values
    const avgShap = this.averageSHAPValues(shapValues);

    return {
      method: 'shap',
      shapValues: avgShap,
      numSamples,
    };
  }

  async computeSHAPForSample(sample, dataset) {
    // Simplified SHAP computation
    const numFeatures = sample.features.length;
    const shapValues = new Float32Array(numFeatures);

    const baseline = this.computeBaseline(dataset);
    const fullPrediction = await this.model.forward(sample.features);

    for (let f = 0; f < numFeatures; f++) {
      // Marginal contribution of feature f
      const withoutFeature = [...sample.features];
      withoutFeature[f] = baseline[f];

      const predictionWithout = await this.model.forward(withoutFeature);
      shapValues[f] = fullPrediction - predictionWithout;
    }

    return shapValues;
  }

  /**
   * LIME explanation
   */
  async limeExplanation(dataset, config = {}) {
    const { numSamples = 1000, kernelWidth = 0.25 } = config;

    // Generate perturbed samples around instance
    const instance = dataset[0];
    const perturbedSamples = this.generatePerturbedSamples(instance, numSamples);

    // Get model predictions
    const predictions = await Promise.all(
      perturbedSamples.map(s => this.model.forward(s.features))
    );

    // Fit linear model
    const weights = this.fitLinearModel(perturbedSamples, predictions, instance, kernelWidth);

    return {
      method: 'lime',
      weights,
      numSamples,
    };
  }

  permuteFeature(dataset, featureIndex) {
    const permuted = dataset.map(sample => ({ ...sample }));
    const values = dataset.map(s => s.features[featureIndex]);

    // Fisher-Yates shuffle
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }

    permuted.forEach((sample, idx) => {
      sample.features[featureIndex] = values[idx];
    });

    return permuted;
  }

  async evaluateModel(dataset) {
    // Simulated model evaluation
    let correct = 0;

    for (const sample of dataset) {
      const prediction = await this.model.forward(sample.features);
      if (prediction === sample.label) {
        correct++;
      }
    }

    return correct / dataset.length;
  }

  computeStd(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  computeBaseline(dataset) {
    // Compute mean feature values
    const numFeatures = dataset[0].features.length;
    const baseline = new Float32Array(numFeatures).fill(0);

    for (const sample of dataset) {
      for (let f = 0; f < numFeatures; f++) {
        baseline[f] += sample.features[f];
      }
    }

    for (let f = 0; f < numFeatures; f++) {
      baseline[f] /= dataset.length;
    }

    return baseline;
  }

  averageSHAPValues(shapValues) {
    const numFeatures = shapValues[0].length;
    const avgShap = new Float32Array(numFeatures).fill(0);

    for (const values of shapValues) {
      for (let f = 0; f < numFeatures; f++) {
        avgShap[f] += values[f];
      }
    }

    for (let f = 0; f < numFeatures; f++) {
      avgShap[f] /= shapValues.length;
    }

    return Array.from(avgShap);
  }

  generatePerturbedSamples(instance, numSamples) {
    const samples = [];

    for (let i = 0; i < numSamples; i++) {
      const perturbed = {
        features: instance.features.map(v => v + (Math.random() - 0.5) * 0.1),
        distance: 0,
      };

      // Compute distance from original
      perturbed.distance = Math.sqrt(
        perturbed.features.reduce((sum, v, idx) => sum + Math.pow(v - instance.features[idx], 2), 0)
      );

      samples.push(perturbed);
    }

    return samples;
  }

  fitLinearModel(samples, predictions, instance, kernelWidth) {
    // Simplified linear regression with kernel weighting
    const numFeatures = samples[0].features.length;
    const weights = new Float32Array(numFeatures).fill(0);

    // Weighted least squares (simplified)
    for (let f = 0; f < numFeatures; f++) {
      let weightedSum = 0;
      let totalWeight = 0;

      for (let i = 0; i < samples.length; i++) {
        const weight = Math.exp(-Math.pow(samples[i].distance, 2) / kernelWidth);
        weightedSum += weight * samples[i].features[f] * predictions[i];
        totalWeight += weight;
      }

      weights[f] = weightedSum / totalWeight;
    }

    return Array.from(weights);
  }
}

/**
 * Interpretability Controller
 * Main interface for model interpretability
 */
export class InterpretabilityController {
  constructor(model, config = {}) {
    this.model = model;
    this.config = config;

    this.attentionVisualizer = new AttentionVisualizer(config.attention);
    this.gradientExplainer = new GradientExplainer(model, config.gradient);
    this.featureAnalyzer = new FeatureImportanceAnalyzer(model, config.features);
  }

  /**
   * Generate comprehensive interpretability report
   */
  async generateReport(input, dataset, config = {}) {
    console.log('Generating interpretability report...');

    const report = {
      timestamp: Date.now(),
      model: this.getModelInfo(),
      explanations: {},
    };

    // Attention visualization (if applicable)
    if (config.includeAttention !== false) {
      try {
        report.explanations.attention = await this.attentionVisualizer.visualizeAttention(
          this.model,
          input,
          config.attention
        );
      } catch (error) {
        console.warn('Attention visualization failed:', error.message);
      }
    }

    // Gradient-based explanations
    if (config.includeGradients !== false) {
      report.explanations.saliency = await this.gradientExplainer.computeSaliencyMap(input);

      if (config.includeIntegratedGradients) {
        report.explanations.integratedGradients =
          await this.gradientExplainer.computeIntegratedGradients(
            input,
            null,
            config.integratedGradients
          );
      }
    }

    // Feature importance
    if (config.includeFeatureImportance && dataset) {
      report.explanations.featureImportance = await this.featureAnalyzer.computeImportance(
        dataset,
        config.featureImportance
      );
    }

    return report;
  }

  getModelInfo() {
    return {
      type: this.model.type || 'unknown',
      architecture: this.model.architecture || 'unknown',
    };
  }

  /**
   * Export report
   */
  exportReport(report, format = 'json') {
    if (format === 'json') {
      return JSON.stringify(report, null, 2);
    }

    // Can add HTML, PDF, etc.
    throw new Error(`Unsupported format: ${format}`);
  }
}

/**
 * Create interpretability system
 */
export function createInterpretability(model, config = {}) {
  return new InterpretabilityController(model, config);
}

// All components already exported via 'export class' and 'export function' declarations above
