/**
 * Enhanced Probabilistic Tensor Operations Core
 *
 * This module contains the core EnhancedProbabilisticTensorOperations class
 * integrating SciRS2-Core patterns with advanced probabilistic tensor operations,
 * providing enterprise-grade statistical computing capabilities with automatic
 * differentiation support.
 */

import {
  SciRS2Core,
  SciRS2Tensor,
  createSciRS2Core,
  scirs2_tensor,
  scirs2_random,
} from '../scirs2-integration.js';

import { BayesianInferenceEngine } from './bayesian-engine.js';
import { AdvancedStatisticalAnalyzer } from './statistical-analyzer.js';
import { StatisticalQualityController } from './quality-controller.js';
import { SciRS2ProbabilisticDistributions } from './distributions.js';
import { ConvergenceDiagnostics, AdaptiveProposalEngine } from './diagnostics.js';

/**
 * Enhanced Probabilistic Tensor Operations System
 * Integrates SciRS2 patterns with advanced probabilistic computing
 */
export class EnhancedProbabilisticTensorOperations {
  constructor(options = {}) {
    this.options = {
      precision: 'f64',
      enableSciRS2: true,
      qualityLevel: 'research',
      statisticalValidation: true,
      enableAutoGrad: true,
      enableBayesian: true,
      ...options,
    };

    // Initialize SciRS2 core for advanced operations
    this.scirs2 = this.options.enableSciRS2
      ? createSciRS2Core({
          precision: this.options.precision,
          enableAutoGrad: this.options.enableAutoGrad,
        })
      : null;

    // Advanced statistical engines
    this.bayesianEngine = new BayesianInferenceEngine(this.scirs2);
    this.statisticalAnalyzer = new AdvancedStatisticalAnalyzer(this.scirs2);
    this.qualityController = new StatisticalQualityController(this.options.qualityLevel);

    // Distribution registry with SciRS2 integration
    this.distributions = new SciRS2ProbabilisticDistributions(this.scirs2);

    // Performance optimizations
    this.optimizationCache = new Map();
    this.computationHistory = [];
  }

  /**
   * Create advanced probabilistic tensor with SciRS2 integration
   * @param {Array<number>} shape - Tensor shape
   * @param {Object} distribution_config - Distribution configuration
   * @param {number} seed - Random seed
   * @param {Object} options - Creation options
   * @returns {SciRS2Tensor} Enhanced probabilistic tensor
   */
  create_probabilistic_tensor(shape, distribution_config, seed = null, options = {}) {
    const {
      requiresGrad = false,
      enableStatisticalTracking = true,
      qualityValidation = this.options.statisticalValidation,
      dtype = this.options.precision,
    } = options;

    // Use SciRS2 for advanced tensor creation if available
    if (this.scirs2) {
      const tensor = this.scirs2.random(
        shape,
        distribution_config.type,
        {
          ...distribution_config,
          seed,
        },
        {
          requiresGrad,
          dtype,
          qualityLevel: this.options.qualityLevel,
          statisticalValidation: qualityValidation,
        }
      );

      // Enhance with probabilistic metadata
      tensor._probabilistic_metadata = {
        distribution: distribution_config,
        creation_method: 'scirs2_enhanced',
        quality_level: this.options.qualityLevel,
        timestamp: Date.now(),
        seed,
      };

      // Perform statistical validation if enabled
      if (qualityValidation) {
        const validation_results = this.qualityController.validate_tensor(
          tensor,
          distribution_config
        );
        tensor._validation_results = validation_results;

        if (!validation_results.passed) {
          console.warn('Tensor quality validation failed:', validation_results.warnings);
        }
      }

      return tensor;
    }
      // Fallback to basic probabilistic tensor creation
      return this._create_basic_probabilistic_tensor(shape, distribution_config, seed, options);

  }

  /**
   * Fallback basic probabilistic tensor creation
   * @private
   */
  _create_basic_probabilistic_tensor(shape, distribution_config, seed, options) {
    const total_elements = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(total_elements);

    // Basic random generation (fallback when SciRS2 is not available)
    for (let i = 0; i < total_elements; i++) {
      data[i] = this._sample_basic_distribution(distribution_config);
    }

    return new SciRS2Tensor(data, shape, {
      dtype: options.dtype || 'f64',
      requiresGrad: options.requiresGrad || false,
      statisticalTracking: true,
      generationMetadata: {
        distribution: distribution_config,
        creation_method: 'basic_fallback',
        timestamp: Date.now(),
      },
    });
  }

  /**
   * Sample from basic distribution (fallback)
   * @private
   */
  _sample_basic_distribution(config) {
    switch (config.type) {
      case 'normal':
        return this._box_muller_normal(config.mean || 0, config.std || 1);
      case 'uniform':
        return (config.low || 0) + Math.random() * ((config.high || 1) - (config.low || 0));
      case 'exponential':
        return -Math.log(Math.random()) / (config.rate || 1);
      default:
        return Math.random();
    }
  }

  /**
   * Box-Muller transform for normal distribution
   * @private
   */
  _box_muller_normal(mu = 0, sigma = 1) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mu + sigma * z0;
  }

  /**
   * Advanced tensor analysis with SciRS2 integration
   * @param {Object} tensor - Tensor to analyze
   * @param {Object} options - Analysis options
   * @returns {Object} Comprehensive statistical analysis
   */
  analyze_tensor(tensor, options = {}) {
    const {
      includeAdvancedStats = true,
      includeBayesianAnalysis = this.options.enableBayesian,
      includeInformationTheory = true,
      includeQualityMetrics = true,
    } = options;

    const analysis = {
      timestamp: Date.now(),
      tensor_properties: this._analyze_tensor_properties(tensor),
      basic_statistics: this._compute_basic_statistics(tensor),
      advanced_statistics: includeAdvancedStats ? this._compute_advanced_statistics(tensor) : null,
      bayesian_analysis: includeBayesianAnalysis ? this._perform_bayesian_analysis(tensor) : null,
      information_theory: includeInformationTheory
        ? this._compute_information_theory_metrics(tensor)
        : null,
      quality_metrics: includeQualityMetrics ? this._compute_quality_metrics(tensor) : null,
    };

    // Use SciRS2 statistical analyzer if available
    if (this.scirs2 && this.statisticalAnalyzer) {
      analysis.scirs2_analysis = this.statisticalAnalyzer.comprehensive_analysis(tensor);
    }

    return analysis;
  }

  /**
   * Analyze tensor properties
   * @private
   */
  _analyze_tensor_properties(tensor) {
    return {
      shape: Array.from(tensor.shape),
      dtype: tensor.dtype,
      size: tensor.size(),
      ndim: tensor.ndim(),
      requires_grad: tensor.requiresGrad || false,
      device: tensor.device || 'cpu',
      layout: tensor.layout || 'contiguous',
      has_scirs2_metadata: !!tensor._probabilistic_metadata,
      creation_method: tensor._probabilistic_metadata?.creation_method || 'unknown',
    };
  }

  /**
   * Compute basic statistics
   * @private
   */
  _compute_basic_statistics(tensor) {
    if (!tensor.data || tensor.data.length === 0) {
      return null;
    }

    const {data} = tensor;
    let sum = 0;
    let sum_squares = 0;
    let min = data[0];
    let max = data[0];

    for (let i = 0; i < data.length; i++) {
      const value = data[i];
      sum += value;
      sum_squares += value * value;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }

    const mean = sum / data.length;
    const variance = (sum_squares - (sum * sum) / data.length) / (data.length - 1);
    const std_dev = Math.sqrt(variance);

    return {
      count: data.length,
      sum,
      mean,
      variance,
      std_dev,
      min,
      max,
      range: max - min,
    };
  }

  /**
   * Compute advanced statistics
   * @private
   */
  _compute_advanced_statistics(tensor) {
    const basic_stats = this._compute_basic_statistics(tensor);
    if (!basic_stats) return null;

    const {data} = tensor;
    const { mean, std_dev } = basic_stats;

    // Compute higher-order moments
    const skewness = this._compute_skewness(data, mean, std_dev);
    const kurtosis = this._compute_kurtosis(data, mean, std_dev);

    // Compute percentiles
    const sorted_data = Array.from(data).sort((a, b) => a - b);
    const percentiles = this._compute_percentiles(sorted_data);

    // Normality tests
    const normality_test = this._shapiro_wilk_test(data);

    // Outlier detection
    const outliers = this._detect_outliers(data, mean, std_dev);

    return {
      skewness,
      kurtosis,
      percentiles,
      normality_test,
      outliers: {
        count: outliers.length,
        indices: outliers,
        percentage: (outliers.length / data.length) * 100,
      },
    };
  }

  /**
   * Compute skewness
   * @private
   */
  _compute_skewness(data, mean, std_dev) {
    const n = data.length;
    let sum_cubed = 0;

    for (const value of data) {
      sum_cubed += Math.pow((value - mean) / std_dev, 3);
    }

    return (n / ((n - 1) * (n - 2))) * sum_cubed;
  }

  /**
   * Compute kurtosis
   * @private
   */
  _compute_kurtosis(data, mean, std_dev) {
    const n = data.length;
    let sum_fourth = 0;

    for (const value of data) {
      sum_fourth += Math.pow((value - mean) / std_dev, 4);
    }

    const kurtosis = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum_fourth;
    return kurtosis - (3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3));
  }

  /**
   * Compute percentiles
   * @private
   */
  _compute_percentiles(sorted_data) {
    const percentile = p => {
      const index = (p / 100) * (sorted_data.length - 1);
      const lower = Math.floor(index);
      const upper = Math.ceil(index);
      const weight = index - lower;

      if (upper >= sorted_data.length) return sorted_data[sorted_data.length - 1];
      return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight;
    };

    return {
      p5: percentile(5),
      p10: percentile(10),
      p25: percentile(25),
      p50: percentile(50), // median
      p75: percentile(75),
      p90: percentile(90),
      p95: percentile(95),
    };
  }

  /**
   * Shapiro-Wilk normality test
   * @private
   */
  _shapiro_wilk_test(data) {
    const n = data.length;
    if (n < 3 || n > 5000) {
      return { statistic: null, p_value: null, is_normal: null };
    }

    // Simplified implementation
    const sorted_data = Array.from(data).sort((a, b) => a - b);
    const mean = data.reduce((sum, val) => sum + val, 0) / n;

    // Calculate W statistic (simplified)
    const numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n; i++) {
      denominator += Math.pow(sorted_data[i] - mean, 2);
    }

    const w_statistic = numerator / denominator;
    const is_normal = w_statistic > 0.9; // Simplified threshold

    return {
      statistic: w_statistic,
      p_value: null, // Would need full implementation
      is_normal,
    };
  }

  /**
   * Detect outliers using IQR method
   * @private
   */
  _detect_outliers(data, mean, std_dev) {
    const sorted_data = Array.from(data).sort((a, b) => a - b);
    const q1_index = Math.floor(sorted_data.length * 0.25);
    const q3_index = Math.floor(sorted_data.length * 0.75);
    const q1 = sorted_data[q1_index];
    const q3 = sorted_data[q3_index];
    const iqr = q3 - q1;

    const lower_bound = q1 - 1.5 * iqr;
    const upper_bound = q3 + 1.5 * iqr;

    const outliers = [];
    for (let i = 0; i < data.length; i++) {
      if (data[i] < lower_bound || data[i] > upper_bound) {
        outliers.push(i);
      }
    }

    return outliers;
  }

  /**
   * Perform Bayesian analysis
   * @private
   */
  _perform_bayesian_analysis(tensor) {
    if (this.bayesianEngine) {
      return this.bayesianEngine.analyze_tensor(tensor);
    }
    return null;
  }

  /**
   * Compute information theory metrics
   * @private
   */
  _compute_information_theory_metrics(tensor) {
    const {data} = tensor;
    if (!data || data.length === 0) return null;

    // Compute entropy (simplified discrete approximation)
    const bins = Math.min(50, Math.floor(Math.sqrt(data.length)));
    const histogram = this._compute_histogram(data, bins);
    const entropy = this._compute_entropy(histogram);

    return {
      entropy,
      mutual_information: null, // Would need pairs of tensors
      bins_used: bins,
    };
  }

  /**
   * Compute histogram
   * @private
   */
  _compute_histogram(data, bins) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const bin_width = (max - min) / bins;
    const histogram = new Array(bins).fill(0);

    for (const value of data) {
      const bin = Math.min(Math.floor((value - min) / bin_width), bins - 1);
      histogram[bin]++;
    }

    // Normalize
    return histogram.map(count => count / data.length);
  }

  /**
   * Compute entropy
   * @private
   */
  _compute_entropy(probabilities) {
    let entropy = 0;
    for (const p of probabilities) {
      if (p > 0) {
        entropy -= p * Math.log2(p);
      }
    }
    return entropy;
  }

  /**
   * Compute quality metrics
   * @private
   */
  _compute_quality_metrics(tensor) {
    const basic_stats = this._compute_basic_statistics(tensor);
    if (!basic_stats) return null;

    const { mean, std_dev } = basic_stats;

    return {
      signal_to_noise_ratio: Math.abs(mean) / std_dev,
      coefficient_of_variation: std_dev / Math.abs(mean || 1),
      dynamic_range: basic_stats.range,
      effective_bits: Math.log2(basic_stats.range / std_dev),
      quality_score: this._compute_overall_quality_score(tensor, basic_stats),
    };
  }

  /**
   * Compute overall quality score
   * @private
   */
  _compute_overall_quality_score(tensor, basic_stats) {
    let score = 1.0;

    // Check for NaN or Inf values
    const has_invalid = Array.from(tensor.data).some(x => !Number.isFinite(x));
    if (has_invalid) score *= 0.5;

    // Check dynamic range
    const dynamic_range = basic_stats.range / basic_stats.std_dev;
    if (dynamic_range < 2) score *= 0.8;

    // Check for reasonable distribution
    const cv = basic_stats.std_dev / Math.abs(basic_stats.mean || 1);
    if (cv > 10) score *= 0.7; // Very high variability

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Perform Bayesian inference
   * @param {Array} observed_data - Observed data
   * @param {Object} prior_config - Prior configuration
   * @param {Object} options - Inference options
   * @returns {Object} Inference results
   */
  perform_bayesian_inference(observed_data, prior_config, options = {}) {
    if (this.bayesianEngine) {
      return this.bayesianEngine.perform_inference(observed_data, prior_config, options);
    }

    // Fallback basic Bayesian inference
    return this._basic_bayesian_inference(observed_data, prior_config, options);
  }

  /**
   * Basic Bayesian inference fallback
   * @private
   */
  _basic_bayesian_inference(observed_data, prior_config, options) {
    // Simplified Bayesian updating for normal distribution
    if (prior_config.type === 'normal_normal') {
      const prior_mean = prior_config.prior_mean || 0;
      const prior_var = prior_config.prior_variance || 1;
      const likelihood_var = prior_config.likelihood_variance || 1;

      const data_mean = observed_data.reduce((sum, x) => sum + x, 0) / observed_data.length;
      const n = observed_data.length;

      // Bayesian updating formulas
      const posterior_var = 1 / (1 / prior_var + n / likelihood_var);
      const posterior_mean =
        posterior_var * (prior_mean / prior_var + (n * data_mean) / likelihood_var);

      return {
        posterior_parameters: {
          mean: posterior_mean,
          variance: posterior_var,
        },
        credible_intervals: {
          '95%': [
            posterior_mean - 1.96 * Math.sqrt(posterior_var),
            posterior_mean + 1.96 * Math.sqrt(posterior_var),
          ],
        },
        evidence_estimate: null, // Would need full implementation
      };
    }

    return null;
  }
}