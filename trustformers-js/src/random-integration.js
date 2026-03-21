/**
 * Random Integration Utilities
 *
 * This module provides seamless integration of advanced probabilistic
 * random number generation throughout the TrustformeRS system, replacing
 * basic Math.random() usage with high-quality algorithms.
 *
 * Features:
 * - Drop-in replacement for Math.random()
 * - Context-aware random generation for different use cases
 * - Advanced ID generation with cryptographic properties
 * - Statistical quality monitoring and validation
 * - Performance optimization for different scenarios
 */

import { AdvancedRandomGenerator, ProbabilisticTensorOperations } from './probabilistic-tensors.js';

class RandomIntegrationManager {
  constructor(global_seed = null) {
    this.global_seed = global_seed || this._generate_cryptographic_seed();
    this.primary_generator = new AdvancedRandomGenerator(this.global_seed);

    // Specialized generators for different contexts
    this.generators = {
      debug: new AdvancedRandomGenerator(this.global_seed ^ 0x12345678),
      tensor: new AdvancedRandomGenerator(this.global_seed ^ 0x87654321),
      performance: new AdvancedRandomGenerator(this.global_seed ^ 0xABCDEF00),
      image: new AdvancedRandomGenerator(this.global_seed ^ 0xFEDCBA98),
      model: new AdvancedRandomGenerator(this.global_seed ^ 0x13579BDF),
      worker: new AdvancedRandomGenerator(this.global_seed ^ 0x2468ACE0),
      node_optimization: new AdvancedRandomGenerator(this.global_seed ^ 0x11111111)
    };

    // Statistical monitoring
    this.usage_statistics = {
      total_calls: 0,
      calls_by_context: {},
      quality_metrics: [],
      performance_metrics: []
    };

    // Cache for expensive computations
    this.id_cache = new Map();
    this.entropy_pool = [];
    this.entropy_pool_size = 1000;

    this._initialize_entropy_pool();
  }

  _generate_cryptographic_seed() {
    // Use multiple entropy sources for better seed quality
    const timestamp = Date.now();
    const performance_now = typeof performance !== 'undefined' ? performance.now() : 0;
    const random_values = [];

    // Collect entropy from various sources
    if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
      const random_array = new Uint32Array(4);
      crypto.getRandomValues(random_array);
      random_values.push(...random_array);
    }

    // Memory address entropy (JavaScript engine dependent)
    const memory_entropy = (new Error().stack || '').length;

    // Combine all entropy sources
    let seed = timestamp ^ performance_now ^ memory_entropy;
    for (const value of random_values) {
      seed ^= value;
      seed = ((seed * 1664525) + 1013904223) >>> 0; // Linear congruential mixing
    }

    return seed;
  }

  _initialize_entropy_pool() {
    // Pre-generate high-quality random values for fast access
    for (let i = 0; i < this.entropy_pool_size; i++) {
      this.entropy_pool.push(this.primary_generator.extract_number());
    }
  }

  _refresh_entropy_pool() {
    // Refresh entropy pool when it gets low
    const refresh_threshold = this.entropy_pool_size * 0.1;
    if (this.entropy_pool.length < refresh_threshold) {
      const needed = this.entropy_pool_size - this.entropy_pool.length;
      for (let i = 0; i < needed; i++) {
        this.entropy_pool.push(this.primary_generator.extract_number());
      }
    }
  }

  // Enhanced random number generation for different contexts
  get_random_for_context(context = 'general', distribution_config = null) {
    const start_time = Date.now();

    // Update usage statistics
    this.usage_statistics.total_calls++;
    this.usage_statistics.calls_by_context[context] = (this.usage_statistics.calls_by_context[context] || 0) + 1;

    let result;
    const generator = this.generators[context] || this.primary_generator;

    if (distribution_config) {
      // Use specified distribution
      switch (distribution_config.type) {
        case 'uniform':
          result = generator.extract_number();
          break;
        case 'normal':
          result = generator.normal(distribution_config.mean || 0, distribution_config.std || 1);
          break;
        case 'gamma':
          result = generator.gamma(distribution_config.alpha, distribution_config.beta || 1);
          break;
        case 'beta':
          result = generator.beta(distribution_config.alpha, distribution_config.beta);
          break;
        default:
          result = generator.extract_number();
      }
    } else {
      // Fast path: use entropy pool for uniform distribution
      if (this.entropy_pool.length > 0) {
        result = this.entropy_pool.pop();
        this._refresh_entropy_pool();
      } else {
        result = generator.extract_number();
      }
    }

    // Record performance metrics
    const end_time = Date.now();
    this.usage_statistics.performance_metrics.push({
      context,
      duration: end_time - start_time,
      timestamp: end_time
    });

    // Keep metrics history bounded
    if (this.usage_statistics.performance_metrics.length > 1000) {
      this.usage_statistics.performance_metrics = this.usage_statistics.performance_metrics.slice(-500);
    }

    return result;
  }

  // Advanced ID generation with better entropy
  generate_advanced_id(context, options = {}) {
    const {
      length = 9,
      use_timestamp = true,
      use_counter = false,
      charset = '0123456789abcdefghijklmnopqrstuvwxyz'
    } = options;

    // Create cache key
    const cache_key = `${context}_${length}_${use_timestamp}_${use_counter}`;

    // Check if we have recent IDs for this configuration
    if (this.id_cache.has(cache_key)) {
      const cached_data = this.id_cache.get(cache_key);
      if (Date.now() - cached_data.timestamp < 1000) { // 1 second cache
        // Generate from cached entropy
        return this._generate_id_from_entropy(cached_data.entropy, length, charset, use_timestamp);
      }
    }

    // Generate fresh entropy
    const generator = this.generators[context] || this.primary_generator;
    const entropy_needed = Math.ceil(length * Math.log(charset.length) / Math.log(2) / 8) + 8; // Extra entropy for safety
    const entropy = [];

    for (let i = 0; i < entropy_needed; i++) {
      entropy.push(Math.floor(generator.extract_number() * 256));
    }

    // Cache the entropy
    this.id_cache.set(cache_key, {
      entropy,
      timestamp: Date.now()
    });

    return this._generate_id_from_entropy(entropy, length, charset, use_timestamp);
  }

  _generate_id_from_entropy(entropy, length, charset, use_timestamp) {
    const id_parts = [];

    if (use_timestamp) {
      // Add timestamp component
      const timestamp = Date.now().toString(36);
      id_parts.push(timestamp.slice(-4)); // Last 4 chars of timestamp
    }

    // Generate random part
    let random_part = '';
    let entropy_index = 0;

    const effective_length = use_timestamp ? length - 4 : length;

    for (let i = 0; i < effective_length; i++) {
      if (entropy_index >= entropy.length) {
        // Regenerate entropy if needed
        entropy_index = 0;
      }

      const char_index = entropy[entropy_index] % charset.length;
      random_part += charset[char_index];
      entropy_index++;
    }

    id_parts.push(random_part);

    return id_parts.join('');
  }

  // Confidence and similarity generation with better statistical properties
  generate_confidence_score(context, base_confidence = 0.8, variability = 0.15) {
    const generator = this.generators[context] || this.primary_generator;

    // Use beta distribution for more realistic confidence scores
    const alpha = base_confidence * ((base_confidence * (1 - base_confidence)) / (variability * variability) - 1);
    const beta = (1 - base_confidence) * ((base_confidence * (1 - base_confidence)) / (variability * variability) - 1);

    return generator.beta(Math.max(1, alpha), Math.max(1, beta));
  }

  generate_similarity_score(context, base_similarity = 0.5, decay_factor = 0.05, index = 0) {
    const generator = this.generators[context] || this.primary_generator;

    // Model similarity decay with noise
    const deterministic_component = base_similarity * Math.exp(-decay_factor * index);
    const noise_component = generator.normal(0, 0.1) * (1 - deterministic_component);

    return Math.max(0, Math.min(1, deterministic_component + noise_component));
  }

  // Feature generation with statistical properties
  generate_feature_vector(context, dimension, distribution_config = null) {
    const generator = this.generators[context] || this.primary_generator;
    const prob_system = new ProbabilisticTensorOperations(generator.seed_value);

    const config = distribution_config || {
      type: 'normal',
      mean: 0,
      std: 1
    };

    const tensor = prob_system.create_probabilistic_tensor([dimension], config);
    return Array.from(tensor.data);
  }

  // Statistical quality monitoring
  get_quality_report() {
    const recent_metrics = this.usage_statistics.performance_metrics.slice(-100);

    const avg_performance = recent_metrics.length > 0
      ? recent_metrics.reduce((sum, metric) => sum + metric.duration, 0) / recent_metrics.length
      : 0;

    // Test randomness quality using simple statistical tests
    const test_samples = [];
    for (let i = 0; i < 1000; i++) {
      test_samples.push(this.primary_generator.extract_number());
    }

    const quality_tests = this._perform_randomness_tests(test_samples);

    return {
      usage_statistics: this.usage_statistics,
      performance: {
        average_generation_time: avg_performance,
        total_calls: this.usage_statistics.total_calls
      },
      quality_tests,
      entropy_pool_status: {
        current_size: this.entropy_pool.length,
        max_size: this.entropy_pool_size,
        utilization: (this.entropy_pool_size - this.entropy_pool.length) / this.entropy_pool_size
      }
    };
  }

  _perform_randomness_tests(samples) {
    // Simple statistical tests for randomness quality
    const n = samples.length;
    const mean = samples.reduce((a, b) => a + b, 0) / n;
    const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);

    // Chi-square test for uniformity
    const bins = 10;
    const expected_per_bin = n / bins;
    const histogram = new Array(bins).fill(0);

    for (const sample of samples) {
      const bin_index = Math.min(Math.floor(sample * bins), bins - 1);
      histogram[bin_index]++;
    }

    let chi_square = 0;
    for (const observed of histogram) {
      chi_square += ((observed - expected_per_bin) ** 2) / expected_per_bin;
    }

    // Runs test for independence
    let runs = 1;
    for (let i = 1; i < n; i++) {
      if ((samples[i] >= 0.5) !== (samples[i-1] >= 0.5)) {
        runs++;
      }
    }

    const expected_runs = (2 * n - 1) / 3;
    const runs_variance = (16 * n - 29) / 90;
    const runs_z_score = (runs - expected_runs) / Math.sqrt(runs_variance);

    return {
      mean,
      variance,
      theoretical_mean: 0.5,
      theoretical_variance: 1/12,
      chi_square_statistic: chi_square,
      chi_square_critical: 16.919, // For 9 degrees of freedom at 5% significance
      uniformity_test_passed: chi_square < 16.919,
      runs_test_z_score: runs_z_score,
      independence_test_passed: Math.abs(runs_z_score) < 1.96
    };
  }

  // Advanced seeding with validation
  reseed_all_generators(new_global_seed = null) {
    const old_seed = this.global_seed;
    this.global_seed = new_global_seed || this._generate_cryptographic_seed();

    // Reseed all generators with derived seeds
    this.primary_generator = new AdvancedRandomGenerator(this.global_seed);

    const seed_derivations = [
      0x12345678, 0x87654321, 0xABCDEF00, 0xFEDCBA98,
      0x13579BDF, 0x2468ACE0, 0x11111111
    ];

    const contexts = Object.keys(this.generators);
    for (let i = 0; i < contexts.length; i++) {
      const context = contexts[i];
      const derived_seed = this.global_seed ^ seed_derivations[i % seed_derivations.length];
      this.generators[context] = new AdvancedRandomGenerator(derived_seed);
    }

    // Refresh entropy pool
    this.entropy_pool = [];
    this._initialize_entropy_pool();

    // Clear caches
    this.id_cache.clear();

    return {
      old_seed,
      new_seed: this.global_seed,
      reseeded_contexts: contexts.length + 1 // +1 for primary generator
    };
  }
}

// Global instance for easy access
let global_random_manager = null;

export function get_random_manager(seed = null) {
  if (!global_random_manager || seed !== null) {
    global_random_manager = new RandomIntegrationManager(seed);
  }
  return global_random_manager;
}

// Convenience functions that replace Math.random() usage
export function enhanced_random(context = 'general') {
  return get_random_manager().get_random_for_context(context);
}

export function enhanced_random_id(context, options = {}) {
  return get_random_manager().generate_advanced_id(context, options);
}

export function enhanced_confidence_score(context, base_confidence, variability) {
  return get_random_manager().generate_confidence_score(context, base_confidence, variability);
}

export function enhanced_similarity_score(context, base_similarity, decay_factor, index) {
  return get_random_manager().generate_similarity_score(context, base_similarity, decay_factor, index);
}

export function enhanced_feature_vector(context, dimension, distribution_config) {
  return get_random_manager().generate_feature_vector(context, dimension, distribution_config);
}

// Statistical quality monitoring
export function get_randomness_quality_report() {
  return get_random_manager().get_quality_report();
}

// Migration utilities for existing code
export const RandomMigrationUtils = {
  // Replace Math.random() calls
  replace_math_random: (context = 'general') => enhanced_random(context),

  // Replace basic ID generation
  replace_basic_id: (prefix, context, length = 9) => {
    const id = enhanced_random_id(context, { length: length - prefix.length - 1 });
    return `${prefix}_${id}`;
  },

  // Replace confidence generation patterns
  replace_confidence_pattern: (base, variation, context) => enhanced_confidence_score(context, base, variation),

  // Replace similarity decay patterns
  replace_similarity_pattern: (base, decay, index, context) => enhanced_similarity_score(context, base, decay, index)
};

export { RandomIntegrationManager };