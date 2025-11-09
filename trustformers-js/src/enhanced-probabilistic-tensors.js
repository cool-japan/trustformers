/**
 * Enhanced Probabilistic Tensor Operations with SciRS2 Integration
 *
 * This module integrates SciRS2-Core patterns with advanced probabilistic
 * tensor operations, providing enterprise-grade statistical computing
 * capabilities with automatic differentiation support.
 *
 * This file now serves as a compatibility layer for the modularized
 * probabilistic operations. All functionality has been moved to the
 * ./probabilistic/ module for better organization and maintainability.
 */

// Re-export everything from the modular probabilistic module
export {
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
} from './probabilistic/index.js';

// Default export for backward compatibility
export { default } from './probabilistic/index.js';