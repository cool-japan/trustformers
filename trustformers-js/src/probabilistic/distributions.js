/**
 * SciRS2 Probabilistic Distributions
 * Provides advanced distribution sampling with SciRS2 integration
 */

export class SciRS2ProbabilisticDistributions {
  constructor(scirs2_core) {
    this.scirs2 = scirs2_core;
  }

  /**
   * Advanced distribution sampling with SciRS2 integration
   */
  sample_advanced(distribution, shape, params, options = {}) {
    if (this.scirs2) {
      // Use SciRS2's advanced random generation
      return this.scirs2.random(shape, distribution, params, options);
    }
      // Fallback to basic sampling
      return this._basic_sampling(distribution, shape, params);

  }

  /**
   * Basic sampling fallback
   * @private
   */
  _basic_sampling(distribution, shape, params) {
    const total_elements = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(total_elements);

    for (let i = 0; i < total_elements; i++) {
      data[i] = this._sample_single(distribution, params);
    }

    return data;
  }

  /**
   * Sample single value
   * @private
   */
  _sample_single(distribution, params) {
    // Basic implementation using Math.random
    switch (distribution) {
      case 'normal':
        return this._box_muller(params.mean || 0, params.std || 1);
      case 'uniform':
        return (params.low || 0) + Math.random() * ((params.high || 1) - (params.low || 0));
      case 'exponential':
        return -Math.log(Math.random()) / (params.rate || 1);
      default:
        return Math.random();
    }
  }

  /**
   * Box-Muller transform
   * @private
   */
  _box_muller(mu = 0, sigma = 1) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mu + sigma * z0;
  }
}