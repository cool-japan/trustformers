/**
 * Enhanced Probabilistic Tensor Operations Module
 *
 * This module integrates SciRS2-Core patterns with advanced probabilistic
 * tensor operations, providing enterprise-grade statistical computing
 * capabilities with automatic differentiation support.
 */

// Core imports
import { EnhancedProbabilisticTensorOperations } from './core.js';
import { BayesianInferenceEngine } from './bayesian-engine.js';
import { AdvancedStatisticalAnalyzer } from './statistical-analyzer.js';
import { StatisticalQualityController } from './quality-controller.js';
import { SciRS2ProbabilisticDistributions } from './distributions.js';
import { ConvergenceDiagnostics, AdaptiveProposalEngine } from './diagnostics.js';

// Factory Functions
/**
 * Create enhanced probabilistic operations instance
 * @param {Object} options - Configuration options
 * @returns {EnhancedProbabilisticTensorOperations}
 */
export function create_enhanced_probabilistic_operations(options = {}) {
  return new EnhancedProbabilisticTensorOperations(options);
}

/**
 * Create advanced random tensor with distribution
 * @param {Array<number>} shape - Tensor shape
 * @param {Object} distribution_config - Distribution configuration
 * @param {number} seed - Random seed
 * @param {Object} options - Creation options
 * @returns {SciRS2Tensor}
 */
export function create_advanced_random_tensor(
  shape,
  distribution_config,
  seed = null,
  options = {}
) {
  const operations = new EnhancedProbabilisticTensorOperations();
  return operations.create_probabilistic_tensor(shape, distribution_config, seed, options);
}

/**
 * Analyze tensor statistics
 * @param {Tensor} tensor - Input tensor
 * @param {Object} options - Analysis options
 * @returns {Object} Statistical analysis results
 */
export function analyze_tensor_statistics(tensor, options = {}) {
  const operations = new EnhancedProbabilisticTensorOperations();
  return operations.analyze_tensor(tensor, options);
}

// Maintain backward compatibility with existing API
export class ProbabilisticTensorOperations extends EnhancedProbabilisticTensorOperations {
  constructor(seed = null) {
    super({ enableSciRS2: true });
    if (seed !== null) {
      this.scirs2?.rng?.setSeed(seed);
    }
  }
}

// Export individual classes for direct use
export {
  EnhancedProbabilisticTensorOperations,
  BayesianInferenceEngine,
  AdvancedStatisticalAnalyzer,
  StatisticalQualityController,
  SciRS2ProbabilisticDistributions,
  ConvergenceDiagnostics,
  AdaptiveProposalEngine,
};

// Default export for convenience
export default {
  EnhancedProbabilisticTensorOperations,
  ProbabilisticTensorOperations,
  BayesianInferenceEngine,
  AdvancedStatisticalAnalyzer,
  StatisticalQualityController,
  SciRS2ProbabilisticDistributions,
  ConvergenceDiagnostics,
  AdaptiveProposalEngine,
  create_enhanced_probabilistic_operations,
  create_advanced_random_tensor,
  analyze_tensor_statistics,
};