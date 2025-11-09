/**
 * TrustformeRS Tensor Inspector
 * Advanced tensor analysis and visualization utilities
 */

/**
 * Tensor inspector class for comprehensive tensor analysis
 */
export class TensorInspector {
  constructor() {
    this.analysisCache = new Map();
    this.visualizationCache = new Map();
  }

  /**
   * Comprehensive tensor analysis
   * @param {Object} tensor - Tensor to analyze
   * @param {Object} options - Analysis options
   * @returns {Object} Analysis results
   */
  analyze(tensor, options = {}) {
    const {
      includeStatistics = true,
      includeDistribution = true,
      includeNaN = true,
      includeInfinite = true,
      includeMemory = true,
      includeGradient = false,
      cacheResults = true
    } = options;

    const tensorId = this.getTensorId(tensor);
    const cacheKey = `${tensorId}_${JSON.stringify(options)}`;
    
    if (cacheResults && this.analysisCache.has(cacheKey)) {
      return this.analysisCache.get(cacheKey);
    }

    const analysis = {
      basic: this.getBasicInfo(tensor),
      shape: this.getShapeInfo(tensor),
      dtype: this.getDTypeInfo(tensor),
      memory: includeMemory ? this.getMemoryInfo(tensor) : null,
      statistics: includeStatistics ? this.getStatistics(tensor) : null,
      distribution: includeDistribution ? this.getDistribution(tensor) : null,
      quality: {
        hasNaN: includeNaN ? this.hasNaN(tensor) : null,
        hasInfinite: includeInfinite ? this.hasInfinite(tensor) : null,
        nanCount: includeNaN ? this.getNaNCount(tensor) : null,
        infiniteCount: includeInfinite ? this.getInfiniteCount(tensor) : null
      },
      gradient: includeGradient ? this.getGradientInfo(tensor) : null,
      timestamp: performance.now()
    };

    if (cacheResults) {
      this.analysisCache.set(cacheKey, analysis);
    }

    return analysis;
  }

  /**
   * Get basic tensor information
   * @param {Object} tensor - Tensor object
   * @returns {Object} Basic info
   */
  getBasicInfo(tensor) {
    return {
      id: this.getTensorId(tensor),
      hasShape: typeof tensor.shape === 'function',
      hasDtype: typeof tensor.dtype === 'function',
      hasData: typeof tensor.data === 'function',
      isDisposed: this.isDisposed(tensor),
      constructor: tensor.constructor.name
    };
  }

  /**
   * Get shape information
   * @param {Object} tensor - Tensor object
   * @returns {Object} Shape info
   */
  getShapeInfo(tensor) {
    const shape = this.getShape(tensor);
    const ndim = shape.length;
    const size = shape.reduce((product, dim) => product * dim, 1);
    
    return {
      shape,
      ndim,
      size,
      isEmpty: size === 0,
      isScalar: ndim === 0,
      isVector: ndim === 1,
      isMatrix: ndim === 2,
      isBatch: ndim >= 3,
      strides: this.calculateStrides(shape)
    };
  }

  /**
   * Get data type information
   * @param {Object} tensor - Tensor object
   * @returns {Object} DType info
   */
  getDTypeInfo(tensor) {
    const dtype = this.getDType(tensor);
    const isFloating = ['f32', 'f64', 'float32', 'float64'].includes(dtype);
    const isInteger = ['i32', 'i64', 'u32', 'u64', 'i8', 'u8', 'int32', 'int64', 'uint32', 'uint64', 'int8', 'uint8'].includes(dtype);
    const isBoolean = dtype === 'bool';
    const isComplex = ['c64', 'c128', 'complex64', 'complex128'].includes(dtype);
    
    return {
      dtype,
      isFloating,
      isInteger,
      isBoolean,
      isComplex,
      byteSize: this.getBytesPerElement(dtype),
      precision: this.getPrecision(dtype)
    };
  }

  /**
   * Get memory information
   * @param {Object} tensor - Tensor object
   * @returns {Object} Memory info
   */
  getMemoryInfo(tensor) {
    const shape = this.getShape(tensor);
    const dtype = this.getDType(tensor);
    const elements = shape.reduce((product, dim) => product * dim, 1);
    const bytesPerElement = this.getBytesPerElement(dtype);
    const totalBytes = elements * bytesPerElement;

    return {
      elements,
      bytesPerElement,
      totalBytes,
      totalKB: totalBytes / 1024,
      totalMB: totalBytes / (1024 * 1024),
      densityRatio: 1.0, // For dense tensors
      isLarge: totalBytes > 100 * 1024 * 1024, // > 100MB
      memoryEfficiency: this.calculateMemoryEfficiency(tensor)
    };
  }

  /**
   * Get tensor statistics
   * @param {Object} tensor - Tensor object
   * @returns {Object} Statistics
   */
  getStatistics(tensor) {
    try {
      const data = this.getTensorData(tensor);
      if (!data || data.length === 0) {
        return null;
      }

      const stats = {
        count: data.length,
        min: Math.min(...data),
        max: Math.max(...data),
        mean: this.calculateMean(data),
        median: this.calculateMedian(data),
        std: this.calculateStd(data),
        variance: this.calculateVariance(data),
        sum: data.reduce((sum, val) => sum + val, 0),
        absSum: data.reduce((sum, val) => sum + Math.abs(val), 0),
        range: Math.max(...data) - Math.min(...data),
        nonZeroCount: data.filter(x => x !== 0).length,
        uniqueCount: new Set(data).size
      };

      stats.sparsity = 1 - (stats.nonZeroCount / stats.count);
      stats.meanAbsolute = stats.absSum / stats.count;

      return stats;
    } catch (error) {
      return {
        error: error.message,
        available: false
      };
    }
  }

  /**
   * Get value distribution
   * @param {Object} tensor - Tensor object
   * @param {number} bins - Number of histogram bins
   * @returns {Object} Distribution info
   */
  getDistribution(tensor, bins = 50) {
    try {
      const data = this.getTensorData(tensor);
      if (!data || data.length === 0) {
        return null;
      }

      const min = Math.min(...data);
      const max = Math.max(...data);
      const range = max - min;
      const binWidth = range / bins;
      
      const histogram = new Array(bins).fill(0);
      const binEdges = [];
      
      for (let i = 0; i <= bins; i++) {
        binEdges.push(min + i * binWidth);
      }

      data.forEach(value => {
        let binIndex = Math.floor((value - min) / binWidth);
        if (binIndex >= bins) binIndex = bins - 1;
        if (binIndex < 0) binIndex = 0;
        histogram[binIndex]++;
      });

      // Calculate percentiles
      const sortedData = [...data].sort((a, b) => a - b);
      const percentiles = {};
      [5, 10, 25, 50, 75, 90, 95, 99].forEach(p => {
        const index = Math.floor((p / 100) * sortedData.length);
        percentiles[`p${p}`] = sortedData[index];
      });

      return {
        histogram,
        binEdges,
        binWidth,
        percentiles,
        quartiles: {
          q1: percentiles.p25,
          q2: percentiles.p50,
          q3: percentiles.p75,
          iqr: percentiles.p75 - percentiles.p25
        },
        outliers: this.detectOutliers(data, percentiles.p25, percentiles.p75)
      };
    } catch (error) {
      return {
        error: error.message,
        available: false
      };
    }
  }

  /**
   * Check for NaN values
   * @param {Object} tensor - Tensor object
   * @returns {boolean} Has NaN values
   */
  hasNaN(tensor) {
    try {
      const data = this.getTensorData(tensor);
      return data.some(x => isNaN(x));
    } catch (error) {
      return null;
    }
  }

  /**
   * Check for infinite values
   * @param {Object} tensor - Tensor object
   * @returns {boolean} Has infinite values
   */
  hasInfinite(tensor) {
    try {
      const data = this.getTensorData(tensor);
      return data.some(x => !isFinite(x) && !isNaN(x));
    } catch (error) {
      return null;
    }
  }

  /**
   * Count NaN values
   * @param {Object} tensor - Tensor object
   * @returns {number} NaN count
   */
  getNaNCount(tensor) {
    try {
      const data = this.getTensorData(tensor);
      return data.filter(x => isNaN(x)).length;
    } catch (error) {
      return null;
    }
  }

  /**
   * Count infinite values
   * @param {Object} tensor - Tensor object
   * @returns {number} Infinite count
   */
  getInfiniteCount(tensor) {
    try {
      const data = this.getTensorData(tensor);
      return data.filter(x => !isFinite(x) && !isNaN(x)).length;
    } catch (error) {
      return null;
    }
  }

  /**
   * Get gradient information
   * @param {Object} tensor - Tensor object
   * @returns {Object} Gradient info
   */
  getGradientInfo(tensor) {
    return {
      requiresGrad: tensor.requires_grad || false,
      hasGrad: tensor.grad !== null && tensor.grad !== undefined,
      gradShape: tensor.grad ? this.getShape(tensor.grad) : null,
      isLeaf: tensor.is_leaf || false,
      gradFn: tensor.grad_fn ? tensor.grad_fn.constructor.name : null
    };
  }

  /**
   * Compare two tensors
   * @param {Object} tensor1 - First tensor
   * @param {Object} tensor2 - Second tensor
   * @param {Object} options - Comparison options
   * @returns {Object} Comparison results
   */
  compare(tensor1, tensor2, options = {}) {
    const {
      tolerance = 1e-6,
      includeDifference = true,
      includeStatistics = true
    } = options;

    const shape1 = this.getShape(tensor1);
    const shape2 = this.getShape(tensor2);
    const dtype1 = this.getDType(tensor1);
    const dtype2 = this.getDType(tensor2);

    const comparison = {
      shapesEqual: this.arraysEqual(shape1, shape2),
      dtypesEqual: dtype1 === dtype2,
      sizesEqual: shape1.reduce((p, d) => p * d, 1) === shape2.reduce((p, d) => p * d, 1),
      shape1,
      shape2,
      dtype1,
      dtype2
    };

    if (comparison.shapesEqual && comparison.dtypesEqual) {
      try {
        const data1 = this.getTensorData(tensor1);
        const data2 = this.getTensorData(tensor2);
        
        const differences = data1.map((val, i) => Math.abs(val - data2[i]));
        const maxDiff = Math.max(...differences);
        const meanDiff = differences.reduce((sum, diff) => sum + diff, 0) / differences.length;
        const withinTolerance = differences.every(diff => diff <= tolerance);

        comparison.valuesEqual = withinTolerance;
        comparison.maxDifference = maxDiff;
        comparison.meanDifference = meanDiff;
        comparison.tolerance = tolerance;
        comparison.differences = includeDifference ? differences : null;

        if (includeStatistics) {
          comparison.statistics = {
            tensor1: this.getStatistics(tensor1),
            tensor2: this.getStatistics(tensor2)
          };
        }
      } catch (error) {
        comparison.error = error.message;
      }
    }

    return comparison;
  }

  /**
   * Visualize tensor as text
   * @param {Object} tensor - Tensor object
   * @param {Object} options - Visualization options
   * @returns {string} Text visualization
   */
  visualizeText(tensor, options = {}) {
    const {
      maxElements = 100,
      precision = 4,
      threshold = 1000,
      linewidth = 75,
      edgeItems = 3
    } = options;

    const cacheKey = `${this.getTensorId(tensor)}_text_${JSON.stringify(options)}`;
    if (this.visualizationCache.has(cacheKey)) {
      return this.visualizationCache.get(cacheKey);
    }

    try {
      const shape = this.getShape(tensor);
      const data = this.getTensorData(tensor);
      
      let result = `Tensor(shape=${JSON.stringify(shape)}, dtype=${this.getDType(tensor)})\n`;
      
      if (data.length === 0) {
        result += '[]';
      } else if (data.length <= maxElements) {
        result += this.formatTensorData(data, shape, precision);
      } else {
        result += this.formatLargeTensorData(data, shape, precision, edgeItems);
      }

      this.visualizationCache.set(cacheKey, result);
      return result;
    } catch (error) {
      return `Error visualizing tensor: ${error.message}`;
    }
  }

  /**
   * Generate HTML visualization
   * @param {Object} tensor - Tensor object
   * @param {Object} options - Visualization options
   * @returns {string} HTML visualization
   */
  visualizeHTML(tensor, options = {}) {
    const {
      includeStatistics = true,
      includeHistogram = true,
      includeHeatmap = false,
      colorScheme = 'viridis'
    } = options;

    const analysis = this.analyze(tensor, {
      includeStatistics,
      includeDistribution: includeHistogram
    });

    let html = `
    <div class="tensor-visualization" style="font-family: 'Courier New', monospace; border: 1px solid #ccc; padding: 15px; margin: 10px; background: #f9f9f9;">
      <h3 style="margin-top: 0; color: #333;">Tensor Analysis</h3>
      <div class="basic-info" style="margin-bottom: 15px;">
        <strong>Shape:</strong> ${JSON.stringify(analysis.shape.shape)}<br>
        <strong>Data Type:</strong> ${analysis.dtype.dtype}<br>
        <strong>Size:</strong> ${analysis.shape.size.toLocaleString()} elements<br>
        <strong>Memory:</strong> ${(analysis.memory.totalBytes / 1024).toFixed(2)} KB
      </div>
    `;

    if (includeStatistics && analysis.statistics) {
      html += `
      <div class="statistics" style="margin-bottom: 15px;">
        <h4 style="margin-bottom: 10px;">Statistics</h4>
        <table style="border-collapse: collapse; width: 100%;">
          <tr><td style="padding: 2px 8px; border: 1px solid #ddd;">Min:</td><td style="padding: 2px 8px; border: 1px solid #ddd;">${analysis.statistics.min.toFixed(4)}</td></tr>
          <tr><td style="padding: 2px 8px; border: 1px solid #ddd;">Max:</td><td style="padding: 2px 8px; border: 1px solid #ddd;">${analysis.statistics.max.toFixed(4)}</td></tr>
          <tr><td style="padding: 2px 8px; border: 1px solid #ddd;">Mean:</td><td style="padding: 2px 8px; border: 1px solid #ddd;">${analysis.statistics.mean.toFixed(4)}</td></tr>
          <tr><td style="padding: 2px 8px; border: 1px solid #ddd;">Std:</td><td style="padding: 2px 8px; border: 1px solid #ddd;">${analysis.statistics.std.toFixed(4)}</td></tr>
          <tr><td style="padding: 2px 8px; border: 1px solid #ddd;">Sparsity:</td><td style="padding: 2px 8px; border: 1px solid #ddd;">${(analysis.statistics.sparsity * 100).toFixed(2)}%</td></tr>
        </table>
      </div>
      `;
    }

    if (includeHistogram && analysis.distribution) {
      html += this.generateHistogramHTML(analysis.distribution);
    }

    if (includeHeatmap && analysis.shape.isMatrix) {
      html += this.generateHeatmapHTML(tensor, colorScheme);
    }

    // Quality checks
    if (analysis.quality.hasNaN || analysis.quality.hasInfinite) {
      html += `
      <div class="quality-warnings" style="margin-top: 15px; padding: 10px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px;">
        <h4 style="margin-top: 0; color: #856404;">Quality Warnings</h4>
        ${analysis.quality.hasNaN ? `<div style="color: #d32f2f;">⚠ Contains ${analysis.quality.nanCount} NaN values</div>` : ''}
        ${analysis.quality.hasInfinite ? `<div style="color: #d32f2f;">⚠ Contains ${analysis.quality.infiniteCount} infinite values</div>` : ''}
      </div>
      `;
    }

    html += '</div>';
    return html;
  }

  /**
   * Generate summary report
   * @param {Object} tensor - Tensor object
   * @returns {Object} Summary report
   */
  summarize(tensor) {
    const analysis = this.analyze(tensor);
    
    const summary = {
      id: analysis.basic.id,
      shape: analysis.shape.shape,
      dtype: analysis.dtype.dtype,
      size: analysis.shape.size,
      memoryMB: analysis.memory.totalMB,
      hasIssues: analysis.quality.hasNaN || analysis.quality.hasInfinite,
      sparsity: analysis.statistics ? analysis.statistics.sparsity : null,
      valueRange: analysis.statistics ? [analysis.statistics.min, analysis.statistics.max] : null,
      recommendations: []
    };

    // Generate recommendations
    if (analysis.memory.isLarge) {
      summary.recommendations.push('Consider using lower precision dtype to reduce memory usage');
    }
    
    if (analysis.statistics && analysis.statistics.sparsity > 0.5) {
      summary.recommendations.push('Tensor is sparse, consider using sparse tensor formats');
    }
    
    if (analysis.quality.hasNaN) {
      summary.recommendations.push('Remove or replace NaN values before using in computations');
    }
    
    if (analysis.quality.hasInfinite) {
      summary.recommendations.push('Handle infinite values to prevent numerical instability');
    }

    return summary;
  }

  /**
   * Clear caches
   */
  clearCaches() {
    this.analysisCache.clear();
    this.visualizationCache.clear();
  }

  // Private helper methods
  getTensorId(tensor) {
    if (tensor._inspectorId) return tensor._inspectorId;
    tensor._inspectorId = `tensor_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return tensor._inspectorId;
  }

  getShape(tensor) {
    if (tensor.shape && typeof tensor.shape === 'function') {
      return Array.from(tensor.shape());
    }
    if (tensor.shape && Array.isArray(tensor.shape)) {
      return tensor.shape;
    }
    return [];
  }

  getDType(tensor) {
    if (tensor.dtype && typeof tensor.dtype === 'function') {
      return tensor.dtype();
    }
    if (tensor.dtype) {
      return tensor.dtype;
    }
    return 'unknown';
  }

  getTensorData(tensor) {
    if (tensor.data && typeof tensor.data === 'function') {
      const data = tensor.data();
      return Array.isArray(data) ? data : Array.from(data);
    }
    if (tensor.toArray && typeof tensor.toArray === 'function') {
      return tensor.toArray();
    }
    if (Array.isArray(tensor)) {
      return tensor;
    }
    throw new Error('Cannot extract data from tensor');
  }

  isDisposed(tensor) {
    return tensor.isDisposed === true || tensor._disposed === true;
  }

  getBytesPerElement(dtype) {
    const typeMap = {
      'f32': 4, 'float32': 4,
      'f64': 8, 'float64': 8,
      'i32': 4, 'int32': 4,
      'i64': 8, 'int64': 8,
      'u32': 4, 'uint32': 4,
      'i8': 1, 'int8': 1,
      'u8': 1, 'uint8': 1,
      'bool': 1,
      'c64': 8, 'complex64': 8,
      'c128': 16, 'complex128': 16
    };
    return typeMap[dtype] || 4;
  }

  getPrecision(dtype) {
    const precisionMap = {
      'f32': 'single', 'float32': 'single',
      'f64': 'double', 'float64': 'double',
      'i32': '32-bit', 'int32': '32-bit',
      'i64': '64-bit', 'int64': '64-bit',
      'u32': '32-bit unsigned', 'uint32': '32-bit unsigned',
      'i8': '8-bit', 'int8': '8-bit',
      'u8': '8-bit unsigned', 'uint8': '8-bit unsigned',
      'bool': '1-bit'
    };
    return precisionMap[dtype] || 'unknown';
  }

  calculateStrides(shape) {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  calculateMemoryEfficiency(tensor) {
    // For dense tensors, efficiency is always 1.0
    // For sparse tensors, this would calculate actual efficiency
    return 1.0;
  }

  calculateMean(data) {
    return data.reduce((sum, val) => sum + val, 0) / data.length;
  }

  calculateMedian(data) {
    const sorted = [...data].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 
      ? (sorted[mid - 1] + sorted[mid]) / 2 
      : sorted[mid];
  }

  calculateVariance(data) {
    const mean = this.calculateMean(data);
    return data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
  }

  calculateStd(data) {
    return Math.sqrt(this.calculateVariance(data));
  }

  detectOutliers(data, q1, q3) {
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    return data.filter(x => x < lowerBound || x > upperBound);
  }

  arraysEqual(arr1, arr2) {
    return arr1.length === arr2.length && arr1.every((val, i) => val === arr2[i]);
  }

  formatTensorData(data, shape, precision) {
    if (shape.length === 1) {
      return `[${data.map(x => x.toFixed(precision)).join(', ')}]`;
    } else if (shape.length === 2) {
      const [rows, cols] = shape;
      let result = '[\n';
      for (let i = 0; i < rows; i++) {
        const row = data.slice(i * cols, (i + 1) * cols);
        result += `  [${row.map(x => x.toFixed(precision)).join(', ')}]`;
        if (i < rows - 1) result += ',';
        result += '\n';
      }
      result += ']';
      return result;
    } 
      return `Tensor with ${shape.length}D shape: ${JSON.stringify(shape)}`;
    
  }

  formatLargeTensorData(data, shape, precision, edgeItems) {
    const totalElements = data.length;
    const showElements = edgeItems * 2;
    
    if (totalElements <= showElements) {
      return this.formatTensorData(data, shape, precision);
    }

    const head = data.slice(0, edgeItems);
    const tail = data.slice(-edgeItems);
    
    return `[${head.map(x => x.toFixed(precision)).join(', ')}, ..., ${tail.map(x => x.toFixed(precision)).join(', ')}]`;
  }

  generateHistogramHTML(distribution) {
    const maxHeight = 100;
    const maxCount = Math.max(...distribution.histogram);
    
    let html = `
    <div class="histogram" style="margin-bottom: 15px;">
      <h4 style="margin-bottom: 10px;">Value Distribution</h4>
      <div style="display: flex; align-items: end; height: ${maxHeight}px; margin-bottom: 5px;">
    `;

    distribution.histogram.forEach((count, i) => {
      const height = (count / maxCount) * maxHeight;
      const opacity = count === 0 ? 0.1 : 0.7;
      html += `
        <div style="
          flex: 1; 
          background: rgba(70, 130, 180, ${opacity}); 
          height: ${height}px; 
          margin-right: 1px;
          border-radius: 2px 2px 0 0;
        " title="Bin ${i}: ${count} values"></div>
      `;
    });

    html += `
      </div>
      <div style="font-size: 12px; color: #666;">
        Range: ${distribution.binEdges[0].toFixed(3)} to ${distribution.binEdges[distribution.binEdges.length - 1].toFixed(3)}
      </div>
    </div>
    `;

    return html;
  }

  generateHeatmapHTML(tensor, colorScheme) {
    // Simplified heatmap for small matrices
    const data = this.getTensorData(tensor);
    const shape = this.getShape(tensor);
    
    if (shape.length !== 2 || shape[0] > 20 || shape[1] > 20) {
      return '<div>Heatmap not available for this tensor size</div>';
    }

    const [rows, cols] = shape;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min;

    let html = `
    <div class="heatmap" style="margin-bottom: 15px;">
      <h4 style="margin-bottom: 10px;">Heatmap</h4>
      <div style="display: grid; grid-template-columns: repeat(${cols}, 20px); gap: 1px;">
    `;

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const value = data[i * cols + j];
        const normalized = range === 0 ? 0.5 : (value - min) / range;
        const intensity = Math.round(normalized * 255);
        
        html += `
          <div style="
            width: 20px; 
            height: 20px; 
            background: rgb(${255 - intensity}, ${255 - intensity}, 255);
            border: 1px solid #ccc;
            font-size: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
          " title="[${i},${j}]: ${value.toFixed(3)}"></div>
        `;
      }
    }

    html += '</div></div>';
    return html;
  }
}

// Global inspector instance
export const tensorInspector = new TensorInspector();

// Convenience functions
export const inspect = {
  analyze: (tensor, options) => tensorInspector.analyze(tensor, options),
  compare: (tensor1, tensor2, options) => tensorInspector.compare(tensor1, tensor2, options),
  visualize: (tensor, options) => tensorInspector.visualizeText(tensor, options),
  visualizeHTML: (tensor, options) => tensorInspector.visualizeHTML(tensor, options),
  summarize: (tensor) => tensorInspector.summarize(tensor),
  clearCaches: () => tensorInspector.clearCaches()
};

// Export for integration with other modules
export default TensorInspector;