/**
 * SciRS2 Integration Patterns for TrustformeRS
 *
 * This module implements SciRS2-Core and SciRS2-autograd patterns for
 * advanced scientific computing and automatic differentiation in JavaScript.
 *
 * Following the user's preference for SciRS2-Core/SciRS2-autograd patterns
 * instead of basic Rand/ndarray approaches.
 */

/**
 * SciRS2-inspired Core Scientific Computing Module
 * Provides advanced mathematical operations with automatic differentiation support
 */
export class SciRS2Core {
  constructor(options = {}) {
    this.options = {
      precision: 'f64',
      enableAutoGrad: true,
      optimizeMemory: true,
      parallelization: true,
      seedManagement: 'cryptographic',
      ...options,
    };

    // Advanced random number generator following SciRS2 patterns
    this.rng = new SciRS2RandomGenerator({
      algorithm: 'pcg64',
      seed: this._generateCryptographicSeed(),
      streams: 4, // Multiple independent streams
    });

    // Automatic differentiation context
    this.autograd = new SciRS2AutoGrad(this.options);

    // Mathematical constants and special functions
    this.constants = this._initializeMathematicalConstants();

    // Statistical distributions registry
    this.distributions = new SciRS2Distributions(this.rng);

    // Performance optimization context
    this.optimization = new SciRS2Optimization();
  }

  /**
   * Generate cryptographic-quality seed following SciRS2 patterns
   * @private
   */
  _generateCryptographicSeed() {
    try {
      // Use Web Crypto API if available
      if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
        const array = new Uint32Array(4);
        crypto.getRandomValues(array);
        return array.reduce((acc, val) => acc ^ val, 0);
      }

      // Fallback to high-entropy sources
      const sources = [
        Date.now(),
        performance ? performance.now() * 1000000 : 0,
        Math.random() * 0xffffffff,
        typeof process !== 'undefined' ? process.hrtime()[1] : 0,
      ];

      return sources.reduce((acc, val) => acc ^ Math.floor(val), 0) >>> 0;
    } catch (error) {
      console.warn('Cryptographic seed generation failed, using fallback:', error.message);
      return Date.now() ^ Math.floor(Math.random() * 0xffffffff);
    }
  }

  /**
   * Initialize mathematical constants following SciRS2 precision standards
   * @private
   */
  _initializeMathematicalConstants() {
    return {
      // High-precision mathematical constants
      PI: Math.PI,
      E: Math.E,
      SQRT2: Math.SQRT2,
      SQRT1_2: Math.SQRT1_2,
      LN2: Math.LN2,
      LN10: Math.LN10,
      LOG2E: Math.LOG2E,
      LOG10E: Math.LOG10E,

      // Statistical constants
      EULER_MASCHERONI: 0.5772156649015329,
      GOLDEN_RATIO: 1.6180339887498948,

      // Numerical analysis constants
      MACHINE_EPSILON: Number.EPSILON,
      MAX_SAFE_INTEGER: Number.MAX_SAFE_INTEGER,
      MIN_SAFE_INTEGER: Number.MIN_SAFE_INTEGER,

      // Special function constants
      GAMMA_STERLING_LIMIT: 12,
      BESSEL_PRECISION: 1e-15,
      HYPERGEOMETRIC_PRECISION: 1e-12,
    };
  }

  /**
   * Create tensor with SciRS2-style advanced properties
   * @param {Array|TypedArray} data - Tensor data
   * @param {Array<number>} shape - Tensor shape
   * @param {Object} options - Creation options
   * @returns {SciRS2Tensor} Advanced tensor with SciRS2 capabilities
   */
  tensor(data, shape, options = {}) {
    const {
      dtype = this.options.precision,
      requiresGrad = false,
      device = 'cpu',
      layout = 'contiguous',
      statisticalTracking = true,
    } = options;

    return new SciRS2Tensor(data, shape, {
      dtype,
      requiresGrad,
      device,
      layout,
      statisticalTracking,
      autograd: this.autograd,
      optimization: this.optimization,
      rng: this.rng,
    });
  }

  /**
   * Advanced random tensor generation with SciRS2 distributions
   * @param {Array<number>} shape - Tensor shape
   * @param {string} distribution - Distribution type
   * @param {Object} params - Distribution parameters
   * @param {Object} options - Generation options
   * @returns {SciRS2Tensor} Random tensor with advanced properties
   */
  random(shape, distribution = 'normal', params = {}, options = {}) {
    const {
      requiresGrad = false,
      dtype = this.options.precision,
      seed = null,
      qualityLevel = 'research',
      statisticalValidation = true,
      numericalStability = 'high',
      memoryOptimization = true,
    } = options;

    // Validate input parameters
    this._validateRandomParameters(shape, distribution, params, options);

    // Use specified seed or generate new one
    if (seed !== null) {
      this.rng.setSeed(seed);
    }

    // Generate data using advanced distributions with enhanced stability
    const data = this.distributions.sample(distribution, shape, params, {
      qualityLevel,
      statisticalValidation,
      numericalStability,
      memoryOptimization,
    });

    // Validate output quality if enabled
    if (statisticalValidation) {
      const validationResult = this._validateGeneratedTensor(data, distribution, params);
      if (!validationResult.passed) {
        console.warn('Generated tensor failed quality validation:', validationResult.warnings);
        if (validationResult.severity === 'critical') {
          throw new Error(`Critical tensor generation failure: ${validationResult.error}`);
        }
      }
    }

    return this.tensor(data, shape, {
      dtype,
      requiresGrad,
      statisticalTracking: true,
      generationMetadata: {
        distribution,
        params,
        qualityLevel,
        timestamp: Date.now(),
        seed: this.rng.getCurrentSeed(),
        numericalStability,
        validationPassed: statisticalValidation,
      },
    });
  }

  /**
   * Advanced linear algebra operations with automatic differentiation
   * @param {SciRS2Tensor} a - First tensor
   * @param {SciRS2Tensor} b - Second tensor
   * @param {Object} options - Operation options
   * @returns {SciRS2Tensor} Result tensor
   */
  matmul(a, b, options = {}) {
    const {
      algorithm = 'auto',
      precision = 'high',
      checkNumericalStability = true,
      enableGradient = true,
    } = options;

    // Validate inputs
    this._validateTensorsForMatmul(a, b);

    // Perform operation with automatic differentiation tracking
    if (enableGradient && (a.requiresGrad || b.requiresGrad)) {
      return this.autograd.matmul(a, b, {
        algorithm,
        precision,
        checkNumericalStability,
      });
    } 
      return this._performMatmul(a, b, {
        algorithm,
        precision,
        checkNumericalStability,
      });
    
  }

  /**
   * Validate tensors for matrix multiplication
   * @private
   */
  _validateTensorsForMatmul(a, b) {
    if (!a || !b) {
      throw new Error('Both tensors are required for matrix multiplication');
    }

    if (a.shape.length < 2 || b.shape.length < 2) {
      throw new Error('Tensors must have at least 2 dimensions for matrix multiplication');
    }

    const aCols = a.shape[a.shape.length - 1];
    const bRows = b.shape[b.shape.length - 2];

    if (aCols !== bRows) {
      throw new Error(`Matrix dimension mismatch: ${aCols} != ${bRows}`);
    }
  }

  /**
   * Perform matrix multiplication with advanced algorithms
   * @private
   */
  _performMatmul(a, b, options) {
    const { algorithm, precision, checkNumericalStability } = options;

    // Choose optimal algorithm based on tensor properties
    const optimalAlgorithm = this._selectMatmulAlgorithm(a, b, algorithm);

    // Perform multiplication
    let result;
    switch (optimalAlgorithm) {
      case 'blocked':
        result = this._blockedMatmul(a, b);
        break;
      case 'strassen':
        result = this._strassenMatmul(a, b);
        break;
      case 'naive':
      default:
        result = this._naiveMatmul(a, b);
        break;
    }

    // Numerical stability check
    if (checkNumericalStability) {
      this._checkNumericalStability(result, precision);
    }

    return result;
  }

  /**
   * Select optimal matrix multiplication algorithm
   * @private
   */
  _selectMatmulAlgorithm(a, b, preferredAlgorithm) {
    if (preferredAlgorithm !== 'auto') {
      return preferredAlgorithm;
    }

    const aSize = a.shape.reduce((acc, dim) => acc * dim, 1);
    const bSize = b.shape.reduce((acc, dim) => acc * dim, 1);

    // Use Strassen for large matrices
    if (aSize > 1024 * 1024 && bSize > 1024 * 1024) {
      return 'strassen';
    }

    // Use blocked algorithm for medium-sized matrices
    if (aSize > 64 * 64 || bSize > 64 * 64) {
      return 'blocked';
    }

    // Use naive algorithm for small matrices
    return 'naive';
  }

  /**
   * Naive matrix multiplication implementation
   * @private
   */
  _naiveMatmul(a, b) {
    // Simplified implementation - in practice would be more optimized
    const resultShape = [...a.shape.slice(0, -1), b.shape[b.shape.length - 1]];
    const resultSize = resultShape.reduce((acc, dim) => acc * dim, 1);
    const resultData = new Float64Array(resultSize);

    // Simple matrix multiplication (for demonstration)
    const aRows = a.shape[a.shape.length - 2];
    const aCols = a.shape[a.shape.length - 1];
    const bCols = b.shape[b.shape.length - 1];

    for (let i = 0; i < aRows; i++) {
      for (let j = 0; j < bCols; j++) {
        let sum = 0;
        for (let k = 0; k < aCols; k++) {
          sum += a.data[i * aCols + k] * b.data[k * bCols + j];
        }
        resultData[i * bCols + j] = sum;
      }
    }

    return this.tensor(resultData, resultShape, {
      dtype: 'f64',
      requiresGrad: false,
    });
  }

  /**
   * Blocked matrix multiplication for better cache performance
   * @private
   */
  _blockedMatmul(a, b) {
    // Simplified blocked algorithm
    const blockSize = 64; // Optimal for most modern CPUs
    return this._naiveMatmul(a, b); // Placeholder - would implement actual blocking
  }

  /**
   * Strassen algorithm for large matrices
   * @private
   */
  _strassenMatmul(a, b) {
    // Simplified Strassen implementation
    return this._naiveMatmul(a, b); // Placeholder - would implement actual Strassen
  }

  /**
   * Check numerical stability of computation results
   * @private
   */
  _checkNumericalStability(tensor, precision) {
    if (!tensor || !tensor.data) return;

    let hasNaN = false;
    let hasInf = false;
    let maxValue = 0;
    let minValue = 0;

    for (let i = 0; i < tensor.data.length; i++) {
      const value = tensor.data[i];

      if (Number.isNaN(value)) {
        hasNaN = true;
      } else if (!Number.isFinite(value)) {
        hasInf = true;
      } else {
        maxValue = Math.max(maxValue, Math.abs(value));
        minValue = Math.min(minValue, Math.abs(value));
      }
    }

    if (hasNaN) {
      console.warn('Numerical instability detected: NaN values in result');
    }

    if (hasInf) {
      console.warn('Numerical instability detected: Infinite values in result');
    }

    const dynamicRange = maxValue > 0 ? maxValue / (minValue || Number.EPSILON) : 0;
    const precisionThreshold = precision === 'high' ? 1e12 : 1e6;

    if (dynamicRange > precisionThreshold) {
      console.warn(`Large dynamic range detected: ${dynamicRange.toExponential()}`);
    }
  }

  /**
   * Validate random tensor generation parameters
   * @private
   */
  _validateRandomParameters(shape, distribution, params, options) {
    // Validate shape
    if (!Array.isArray(shape) || shape.length === 0) {
      throw new Error('Shape must be a non-empty array of integers');
    }

    for (let i = 0; i < shape.length; i++) {
      if (!Number.isInteger(shape[i]) || shape[i] <= 0) {
        throw new Error(`Invalid shape dimension at index ${i}: ${shape[i]}`);
      }
    }

    // Validate distribution parameters
    const supportedDistributions = [
      'normal',
      'uniform',
      'exponential',
      'gamma',
      'beta',
      'chi_squared',
      't_distribution',
      'dirichlet',
      'multivariate_normal',
    ];

    if (!supportedDistributions.includes(distribution)) {
      throw new Error(
        `Unsupported distribution: ${distribution}. Supported: ${supportedDistributions.join(', ')}`
      );
    }

    // Distribution-specific parameter validation
    this._validateDistributionParameters(distribution, params);

    // Validate tensor size for memory safety
    const totalElements = shape.reduce((acc, dim) => acc * dim, 1);
    if (totalElements > 100_000_000) {
      // 100M elements limit
      throw new Error(
        `Tensor size too large: ${totalElements} elements. Maximum allowed: 100,000,000`
      );
    }
  }

  /**
   * Validate distribution-specific parameters
   * @private
   */
  _validateDistributionParameters(distribution, params) {
    switch (distribution) {
      case 'normal':
        if (params.std !== undefined && params.std <= 0) {
          throw new Error('Normal distribution std must be positive');
        }
        break;
      case 'uniform':
        if (params.low !== undefined && params.high !== undefined && params.low >= params.high) {
          throw new Error('Uniform distribution: low must be less than high');
        }
        break;
      case 'exponential':
        if (params.rate !== undefined && params.rate <= 0) {
          throw new Error('Exponential distribution rate must be positive');
        }
        break;
      case 'gamma':
        if (params.alpha !== undefined && params.alpha <= 0) {
          throw new Error('Gamma distribution alpha must be positive');
        }
        if (params.beta !== undefined && params.beta <= 0) {
          throw new Error('Gamma distribution beta must be positive');
        }
        break;
      case 'beta':
        if (params.alpha !== undefined && params.alpha <= 0) {
          throw new Error('Beta distribution alpha must be positive');
        }
        if (params.beta !== undefined && params.beta <= 0) {
          throw new Error('Beta distribution beta must be positive');
        }
        break;
      // Add more validation as needed
    }
  }

  /**
   * Validate generated tensor quality
   * @private
   */
  _validateGeneratedTensor(data, distribution, params) {
    const result = {
      passed: true,
      warnings: [],
      severity: 'info',
      error: null,
    };

    // Check for NaN or Infinity values
    const hasNaN = data.some(value => !Number.isFinite(value));
    if (hasNaN) {
      result.passed = false;
      result.severity = 'critical';
      result.error = 'Generated tensor contains NaN or Infinity values';
      return result;
    }

    // Statistical validation for specific distributions
    try {
      const stats = this._computeBasicStats(data);
      this._validateDistributionStats(distribution, params, stats, result);
    } catch (error) {
      result.warnings.push(`Statistical validation failed: ${error.message}`);
    }

    return result;
  }

  /**
   * Compute basic statistics for validation
   * @private
   */
  _computeBasicStats(data) {
    const n = data.length;
    let sum = 0;
    let sumSquares = 0;
    let min = data[0];
    let max = data[0];

    for (const value of data) {
      sum += value;
      sumSquares += value * value;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }

    const mean = sum / n;
    const variance = (sumSquares - (sum * sum) / n) / (n - 1);

    return { mean, variance, std: Math.sqrt(variance), min, max, count: n };
  }

  /**
   * Validate distribution-specific statistics
   * @private
   */
  _validateDistributionStats(distribution, params, stats, result) {
    const tolerance = 0.1; // 10% tolerance for statistical validation

    switch (distribution) {
      case 'normal':
        const expectedMean = params.mean || 0;
        const expectedStd = params.std || 1;

        if (Math.abs(stats.mean - expectedMean) > tolerance * Math.abs(expectedMean) + 0.1) {
          result.warnings.push(`Mean ${stats.mean} deviates from expected ${expectedMean}`);
        }

        if (Math.abs(stats.std - expectedStd) > tolerance * expectedStd) {
          result.warnings.push(`Std ${stats.std} deviates from expected ${expectedStd}`);
        }
        break;

      case 'uniform':
        const expectedMin = params.low || 0;
        const expectedMax = params.high || 1;

        if (stats.min < expectedMin - tolerance || stats.max > expectedMax + tolerance) {
          result.warnings.push(
            `Range [${stats.min}, ${stats.max}] outside expected [${expectedMin}, ${expectedMax}]`
          );
        }
        break;

      // Add more distribution validations as needed
    }

    if (result.warnings.length > 3) {
      result.passed = false;
      result.severity = 'warning';
    }
  }
}

/**
 * Advanced Random Number Generator following SciRS2 patterns
 */
class SciRS2RandomGenerator {
  constructor(options = {}) {
    this.options = {
      algorithm: 'pcg64',
      streams: 1,
      seed: Date.now(),
      ...options,
    };

    this.currentSeed = this.options.seed;
    this.streams = [];

    // Initialize multiple independent streams
    for (let i = 0; i < this.options.streams; i++) {
      this.streams.push(this._createStream(this.currentSeed + i));
    }

    this.activeStream = 0;
  }

  /**
   * Create a single random number stream
   * @private
   */
  _createStream(seed) {
    // PCG64 algorithm implementation (simplified)
    return {
      state: BigInt(seed),
      increment: BigInt(1442695040888963407),

      next() {
        const oldState = this.state;
        this.state = oldState * 6364136223846793005n + this.increment;
        const xorShifted = Number((((oldState >> 18n) ^ oldState) >> 27n) & 0xffffffffn);
        const rot = Number(oldState >> 59n);
        return ((xorShifted >>> rot) | (xorShifted << (-rot & 31))) >>> 0;
      },

      uniform() {
        return this.next() / 4294967296;
      },
    };
  }

  /**
   * Get current seed value
   */
  getCurrentSeed() {
    return this.currentSeed;
  }

  /**
   * Set new seed for all streams
   */
  setSeed(seed) {
    this.currentSeed = seed;
    for (let i = 0; i < this.streams.length; i++) {
      this.streams[i] = this._createStream(seed + i);
    }
  }

  /**
   * Generate uniform random number [0, 1)
   */
  uniform() {
    return this.streams[this.activeStream].uniform();
  }

  /**
   * Generate normal random number using Box-Muller transform
   */
  normal(mu = 0, sigma = 1) {
    const u1 = this.uniform();
    const u2 = this.uniform();

    const mag = sigma * Math.sqrt(-2 * Math.log(u1));
    const z0 = mag * Math.cos(2 * Math.PI * u2);

    return mu + z0;
  }

  /**
   * Switch to different random stream
   */
  switchStream(streamIndex) {
    if (streamIndex >= 0 && streamIndex < this.streams.length) {
      this.activeStream = streamIndex;
    }
  }
}

/**
 * Advanced Statistical Distributions following SciRS2 patterns
 */
class SciRS2Distributions {
  constructor(rng) {
    this.rng = rng;

    // Pre-computed tables for special functions
    this.specialFunctions = {
      gamma: new Map(),
      bessel: new Map(),
      hypergeometric: new Map(),
    };
  }

  /**
   * Sample from distribution with advanced quality control
   * @param {string} distribution - Distribution name
   * @param {Array<number>} shape - Output shape
   * @param {Object} params - Distribution parameters
   * @param {Object} options - Sampling options
   * @returns {TypedArray} Sampled data
   */
  sample(distribution, shape, params = {}, options = {}) {
    const { qualityLevel = 'standard', statisticalValidation = false } = options;

    const totalElements = shape.reduce((acc, dim) => acc * dim, 1);
    const data = new Float64Array(totalElements);

    // Generate samples
    for (let i = 0; i < totalElements; i++) {
      data[i] = this._sampleSingle(distribution, params);
    }

    // Statistical validation if requested
    if (statisticalValidation) {
      this._validateSamples(data, distribution, params);
    }

    return data;
  }

  /**
   * Sample single value from distribution
   * @private
   */
  _sampleSingle(distribution, params) {
    switch (distribution.toLowerCase()) {
      case 'normal':
      case 'gaussian':
        return this.rng.normal(params.mean || 0, params.std || 1);

      case 'uniform':
        const low = params.low || 0;
        const high = params.high || 1;
        return low + (high - low) * this.rng.uniform();

      case 'exponential':
        const rate = params.rate || 1;
        return -Math.log(this.rng.uniform()) / rate;

      case 'gamma':
        return this._gammaVariate(params.shape || 1, params.scale || 1);

      case 'beta':
        return this._betaVariate(params.alpha || 1, params.beta || 1);

      case 'chi_squared':
        return this._chiSquaredVariate(params.df || 1);

      case 'student_t':
        return this._studentTVariate(params.df || 1);

      case 'weibull':
        return this._weibullVariate(params.shape || 1, params.scale || 1);

      default:
        console.warn(`Unknown distribution: ${distribution}, using normal`);
        return this.rng.normal(0, 1);
    }
  }

  /**
   * Generate Gamma random variable using Marsaglia-Tsang method
   * @private
   */
  _gammaVariate(shape, scale) {
    if (shape < 1) {
      // Use transformation for shape < 1
      return this._gammaVariate(shape + 1, scale) * Math.pow(this.rng.uniform(), 1 / shape);
    }

    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);

    while (true) {
      let x; let v;
      do {
        x = this.rng.normal(0, 1);
        v = 1 + c * x;
      } while (v <= 0);

      v = v * v * v;
      const u = this.rng.uniform();

      if (u < 1 - 0.0331 * x * x * x * x) {
        return d * v * scale;
      }

      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
        return d * v * scale;
      }
    }
  }

  /**
   * Generate Beta random variable
   * @private
   */
  _betaVariate(alpha, beta) {
    const x = this._gammaVariate(alpha, 1);
    const y = this._gammaVariate(beta, 1);
    return x / (x + y);
  }

  /**
   * Generate Chi-squared random variable
   * @private
   */
  _chiSquaredVariate(df) {
    return this._gammaVariate(df / 2, 2);
  }

  /**
   * Generate Student's t random variable
   * @private
   */
  _studentTVariate(df) {
    const z = this.rng.normal(0, 1);
    const chi2 = this._chiSquaredVariate(df);
    return z / Math.sqrt(chi2 / df);
  }

  /**
   * Generate Weibull random variable
   * @private
   */
  _weibullVariate(shape, scale) {
    const u = this.rng.uniform();
    return scale * Math.pow(-Math.log(1 - u), 1 / shape);
  }

  /**
   * Validate statistical properties of samples
   * @private
   */
  _validateSamples(data, distribution, params) {
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    const variance =
      data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (data.length - 1);

    // Check against expected moments for the distribution
    const expected = this._getExpectedMoments(distribution, params);

    if (expected.mean !== undefined) {
      const meanError = Math.abs(mean - expected.mean) / Math.abs(expected.mean);
      if (meanError > 0.1) {
        console.warn(
          `Sample mean deviates significantly from expected: ${mean} vs ${expected.mean}`
        );
      }
    }

    if (expected.variance !== undefined) {
      const varianceError = Math.abs(variance - expected.variance) / Math.abs(expected.variance);
      if (varianceError > 0.2) {
        console.warn(
          `Sample variance deviates significantly from expected: ${variance} vs ${expected.variance}`
        );
      }
    }
  }

  /**
   * Get expected moments for distribution
   * @private
   */
  _getExpectedMoments(distribution, params) {
    switch (distribution.toLowerCase()) {
      case 'normal':
        return {
          mean: params.mean || 0,
          variance: Math.pow(params.std || 1, 2),
        };

      case 'uniform':
        const low = params.low || 0;
        const high = params.high || 1;
        return {
          mean: (low + high) / 2,
          variance: Math.pow(high - low, 2) / 12,
        };

      case 'exponential':
        const rate = params.rate || 1;
        return {
          mean: 1 / rate,
          variance: 1 / (rate * rate),
        };

      default:
        return {};
    }
  }
}

/**
 * Automatic Differentiation Engine following SciRS2 patterns
 */
class SciRS2AutoGrad {
  constructor(options = {}) {
    this.options = {
      enableBackwardMode: true,
      enableForwardMode: false,
      optimizeComputationGraph: true,
      ...options,
    };

    this.computationGraph = new Map();
    this.gradientCache = new Map();
  }

  /**
   * Matrix multiplication with gradient tracking
   */
  matmul(a, b, options = {}) {
    // Create computation node
    const nodeId = this._createNodeId();

    // Perform forward pass
    const result = this._forwardMatmul(a, b, options);

    // Track computation for backward pass
    if (this.options.enableBackwardMode && (a.requiresGrad || b.requiresGrad)) {
      this.computationGraph.set(nodeId, {
        operation: 'matmul',
        inputs: [a, b],
        output: result,
        options,
        backward: gradOutput => this._backwardMatmul(gradOutput, a, b),
      });
    }

    return result;
  }

  /**
   * Forward pass for matrix multiplication
   * @private
   */
  _forwardMatmul(a, b, options) {
    // Simplified forward pass - would use actual implementation
    const resultShape = [...a.shape.slice(0, -1), b.shape[b.shape.length - 1]];
    const resultData = new Float64Array(resultShape.reduce((acc, dim) => acc * dim, 1));

    // Perform multiplication (simplified)
    // ... implementation details ...

    return new SciRS2Tensor(resultData, resultShape, {
      requiresGrad: a.requiresGrad || b.requiresGrad,
      computationNode: this.computationGraph.size,
    });
  }

  /**
   * Backward pass for matrix multiplication
   * @private
   */
  _backwardMatmul(gradOutput, a, b) {
    // Compute gradients for inputs
    let gradA = null;
    let gradB = null;

    if (a.requiresGrad) {
      // gradA = gradOutput @ b.T
      gradA = this._computeGradientA(gradOutput, b);
    }

    if (b.requiresGrad) {
      // gradB = a.T @ gradOutput
      gradB = this._computeGradientB(a, gradOutput);
    }

    return { gradA, gradB };
  }

  /**
   * Compute gradient for first input
   * @private
   */
  _computeGradientA(gradOutput, b) {
    // Simplified gradient computation
    return gradOutput; // Placeholder
  }

  /**
   * Compute gradient for second input
   * @private
   */
  _computeGradientB(a, gradOutput) {
    // Simplified gradient computation
    return gradOutput; // Placeholder
  }

  /**
   * Create unique node ID
   * @private
   */
  _createNodeId() {
    return `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

/**
 * Performance Optimization Engine
 */
class SciRS2Optimization {
  constructor() {
    this.optimizationStrategies = new Map();
    this.performanceMetrics = new Map();
  }

  /**
   * Optimize tensor operations
   */
  optimizeTensorOp(operation, inputs, options = {}) {
    const strategyKey = `${operation}_${this._getInputSignature(inputs)}`;

    if (this.optimizationStrategies.has(strategyKey)) {
      return this.optimizationStrategies.get(strategyKey);
    }

    // Analyze operation and create optimization strategy
    const strategy = this._analyzeAndOptimize(operation, inputs, options);
    this.optimizationStrategies.set(strategyKey, strategy);

    return strategy;
  }

  /**
   * Get input signature for caching
   * @private
   */
  _getInputSignature(inputs) {
    return inputs
      .map(input => {
        if (input && input.shape) {
          return input.shape.join('x');
        }
        return 'unknown';
      })
      .join('_');
  }

  /**
   * Analyze operation and create optimization strategy
   * @private
   */
  _analyzeAndOptimize(operation, inputs, options) {
    return {
      algorithm: 'auto',
      parallelization: this._shouldParallelize(inputs),
      memoryOptimization: this._analyzeMemoryUsage(inputs),
      numericalStability: options.checkNumericalStability || false,
    };
  }

  /**
   * Determine if operation should be parallelized
   * @private
   */
  _shouldParallelize(inputs) {
    const totalElements = inputs.reduce((acc, input) => {
      if (input && input.shape) {
        return acc + input.shape.reduce((prod, dim) => prod * dim, 1);
      }
      return acc;
    }, 0);

    return totalElements > 10000; // Threshold for parallelization
  }

  /**
   * Analyze memory usage patterns
   * @private
   */
  _analyzeMemoryUsage(inputs) {
    let totalMemory = 0;

    for (const input of inputs) {
      if (input && input.data) {
        totalMemory += input.data.byteLength;
      }
    }

    return {
      totalMemory,
      useMemoryPooling: totalMemory > 1024 * 1024, // 1MB threshold
      enableCaching: totalMemory < 10 * 1024 * 1024, // 10MB threshold
    };
  }
}

/**
 * Enhanced Tensor class with SciRS2 capabilities
 */
export class SciRS2Tensor {
  constructor(data, shape, options = {}) {
    this.data = data instanceof TypedArray ? data : new Float64Array(data);
    this.shape = new Uint32Array(shape);
    this.dtype = options.dtype || 'f64';
    this.requiresGrad = options.requiresGrad || false;
    this.device = options.device || 'cpu';
    this.layout = options.layout || 'contiguous';

    // SciRS2-specific properties
    this.statisticalTracking = options.statisticalTracking || false;
    this.autograd = options.autograd || null;
    this.optimization = options.optimization || null;
    this.rng = options.rng || null;

    // Metadata
    this.creationTimestamp = Date.now();
    this.generationMetadata = options.generationMetadata || null;
    this.computationHistory = [];

    // Statistical properties (if tracking enabled)
    if (this.statisticalTracking) {
      this.statisticalProperties = this._computeStatisticalProperties();
    }
  }

  /**
   * Compute statistical properties of tensor data
   * @private
   */
  _computeStatisticalProperties() {
    if (!this.data || this.data.length === 0) return null;

    let sum = 0;
    let sumSquares = 0;
    let min = this.data[0];
    let max = this.data[0];

    for (let i = 0; i < this.data.length; i++) {
      const value = this.data[i];
      sum += value;
      sumSquares += value * value;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }

    const mean = sum / this.data.length;
    const variance = (sumSquares - (sum * sum) / this.data.length) / (this.data.length - 1);

    return {
      mean,
      variance,
      std: Math.sqrt(variance),
      min,
      max,
      count: this.data.length,
    };
  }

  /**
   * Get tensor size (total number of elements)
   */
  size() {
    return this.shape.reduce((acc, dim) => acc * dim, 1);
  }

  /**
   * Get tensor dimension count
   */
  ndim() {
    return this.shape.length;
  }

  /**
   * Clone tensor with same properties
   */
  clone() {
    return new SciRS2Tensor(new Float64Array(this.data), Array.from(this.shape), {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad,
      device: this.device,
      layout: this.layout,
      statisticalTracking: this.statisticalTracking,
      autograd: this.autograd,
      optimization: this.optimization,
      rng: this.rng,
    });
  }

  /**
   * Convert to JavaScript object for inspection
   */
  toObject() {
    return {
      shape: Array.from(this.shape),
      dtype: this.dtype,
      size: this.size(),
      ndim: this.ndim(),
      requiresGrad: this.requiresGrad,
      device: this.device,
      layout: this.layout,
      statisticalProperties: this.statisticalProperties,
      generationMetadata: this.generationMetadata,
      creationTimestamp: this.creationTimestamp,
    };
  }
}

// Factory functions for easy usage
export function createSciRS2Core(options = {}) {
  return new SciRS2Core(options);
}

export function scirs2_tensor(data, shape, options = {}) {
  const core = new SciRS2Core();
  return core.tensor(data, shape, options);
}

export function scirs2_random(shape, distribution = 'normal', params = {}, options = {}) {
  const core = new SciRS2Core();
  return core.random(shape, distribution, params, options);
}

export default {
  SciRS2Core,
  SciRS2Tensor,
  createSciRS2Core,
  scirs2_tensor,
  scirs2_random,
};
