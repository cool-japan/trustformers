/**
 * Input Validation Utilities for TrustformeRS
 * Provides comprehensive input validation and schema enforcement
 */

/**
 * Schema Validator
 * Validates inputs against predefined schemas
 */
export class SchemaValidator {
  constructor() {
    this.schemas = new Map();
    this.customValidators = new Map();
    this.validationCache = new Map();
  }

  /**
   * Register a validation schema
   * @param {string} name - Schema name
   * @param {Object} schema - Schema definition
   */
  registerSchema(name, schema) {
    this.schemas.set(name, this.compileSchema(schema));
  }

  /**
   * Register custom validator
   * @param {string} name - Validator name
   * @param {Function} validator - Validator function
   */
  registerValidator(name, validator) {
    this.customValidators.set(name, validator);
  }

  /**
   * Validate input against schema
   * @param {string} schemaName - Schema name
   * @param {any} input - Input to validate
   * @param {Object} options - Validation options
   * @returns {Object} Validation result
   */
  validate(schemaName, input, options = {}) {
    const schema = this.schemas.get(schemaName);
    if (!schema) {
      throw new Error(`Schema '${schemaName}' not found`);
    }

    const cacheKey = options.cache !== false ? `${schemaName}_${JSON.stringify(input)}` : null;

    if (cacheKey && this.validationCache.has(cacheKey)) {
      return this.validationCache.get(cacheKey);
    }

    const result = this.validateAgainstSchema(input, schema, options);

    if (cacheKey && result.valid) {
      this.validationCache.set(cacheKey, result);

      // Limit cache size
      if (this.validationCache.size > 1000) {
        const firstKey = this.validationCache.keys().next().value;
        this.validationCache.delete(firstKey);
      }
    }

    return result;
  }

  /**
   * Compile schema for efficient validation
   * @param {Object} schema - Raw schema
   * @returns {Object} Compiled schema
   */
  compileSchema(schema) {
    return {
      ...schema,
      compiled: true,
      compiledAt: Date.now(),
    };
  }

  /**
   * Validate input against compiled schema
   * @param {any} input - Input to validate
   * @param {Object} schema - Compiled schema
   * @param {Object} options - Validation options
   * @returns {Object} Validation result
   */
  validateAgainstSchema(input, schema, options) {
    const errors = [];
    const warnings = [];

    try {
      this.validateValue(input, schema, '', errors, warnings, options);
    } catch (error) {
      errors.push({
        path: '',
        message: error.message,
        code: 'VALIDATION_ERROR',
      });
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      input: options.sanitize ? this.sanitizeInput(input, schema) : input,
    };
  }

  /**
   * Validate individual value
   * @param {any} value - Value to validate
   * @param {Object} schema - Schema for value
   * @param {string} path - Current path
   * @param {Array} errors - Error array
   * @param {Array} warnings - Warning array
   * @param {Object} options - Validation options
   */
  validateValue(value, schema, path, errors, warnings, options) {
    // Required check
    if (schema.required && (value === undefined || value === null)) {
      errors.push({
        path,
        message: 'Required field is missing',
        code: 'REQUIRED',
      });
      return;
    }

    // Skip validation if value is not provided and not required
    if (value === undefined || value === null) {
      return;
    }

    // Type validation
    if (schema.type && !this.validateType(value, schema.type)) {
      errors.push({
        path,
        message: `Expected type ${schema.type}, got ${typeof value}`,
        code: 'TYPE_MISMATCH',
      });
      return;
    }

    // Format validation
    if (schema.format && !this.validateFormat(value, schema.format)) {
      errors.push({
        path,
        message: `Invalid format for ${schema.format}`,
        code: 'FORMAT_INVALID',
      });
    }

    // Length/range validation
    this.validateConstraints(value, schema, path, errors, warnings);

    // Pattern validation
    if (schema.pattern && typeof value === 'string') {
      const regex = new RegExp(schema.pattern);
      if (!regex.test(value)) {
        errors.push({
          path,
          message: 'Value does not match required pattern',
          code: 'PATTERN_MISMATCH',
        });
      }
    }

    // Enum validation
    if (schema.enum && !schema.enum.includes(value)) {
      errors.push({
        path,
        message: `Value must be one of: ${schema.enum.join(', ')}`,
        code: 'ENUM_INVALID',
      });
    }

    // Custom validation
    if (schema.validator) {
      const customValidator = this.customValidators.get(schema.validator);
      if (customValidator) {
        const customResult = customValidator(value, schema);
        if (!customResult.valid) {
          errors.push({
            path,
            message: customResult.message || 'Custom validation failed',
            code: 'CUSTOM_VALIDATION',
          });
        }
      }
    }

    // Object validation
    if (schema.type === 'object' && schema.properties) {
      this.validateObject(value, schema, path, errors, warnings, options);
    }

    // Array validation
    if (schema.type === 'array' && schema.items) {
      this.validateArray(value, schema, path, errors, warnings, options);
    }
  }

  /**
   * Validate type
   * @param {any} value - Value to check
   * @param {string} expectedType - Expected type
   * @returns {boolean} True if type matches
   */
  validateType(value, expectedType) {
    const actualType = Array.isArray(value) ? 'array' : typeof value;

    if (expectedType === 'integer') {
      return Number.isInteger(value);
    }

    if (expectedType === 'number') {
      return typeof value === 'number' && !isNaN(value);
    }

    return actualType === expectedType;
  }

  /**
   * Validate format
   * @param {any} value - Value to check
   * @param {string} format - Expected format
   * @returns {boolean} True if format is valid
   */
  validateFormat(value, format) {
    if (typeof value !== 'string') {
      return false;
    }

    const formatValidators = {
      email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
      url: /^https?:\/\/[^\s/$.?#].[^\s]*$/i,
      uuid: /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
      date: /^\d{4}-\d{2}-\d{2}$/,
      time: /^\d{2}:\d{2}:\d{2}$/,
      datetime: /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$/,
      ipv4: /^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/,
      ipv6: /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4})$/,
      phone: /^\+?[1-9]\d{1,14}$/,
      creditCard:
        /^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})$/,
    };

    const validator = formatValidators[format];
    return validator ? validator.test(value) : true;
  }

  /**
   * Validate constraints (length, min, max, etc.)
   * @param {any} value - Value to check
   * @param {Object} schema - Schema constraints
   * @param {string} path - Current path
   * @param {Array} errors - Error array
   * @param {Array} warnings - Warning array
   */
  validateConstraints(value, schema, path, errors, warnings) {
    // String length
    if (typeof value === 'string') {
      if (schema.minLength !== undefined && value.length < schema.minLength) {
        errors.push({
          path,
          message: `String too short. Minimum length: ${schema.minLength}`,
          code: 'MIN_LENGTH',
        });
      }

      if (schema.maxLength !== undefined && value.length > schema.maxLength) {
        errors.push({
          path,
          message: `String too long. Maximum length: ${schema.maxLength}`,
          code: 'MAX_LENGTH',
        });
      }
    }

    // Number range
    if (typeof value === 'number') {
      if (schema.minimum !== undefined && value < schema.minimum) {
        errors.push({
          path,
          message: `Value too small. Minimum: ${schema.minimum}`,
          code: 'MINIMUM',
        });
      }

      if (schema.maximum !== undefined && value > schema.maximum) {
        errors.push({
          path,
          message: `Value too large. Maximum: ${schema.maximum}`,
          code: 'MAXIMUM',
        });
      }

      if (schema.multipleOf !== undefined && value % schema.multipleOf !== 0) {
        errors.push({
          path,
          message: `Value must be multiple of ${schema.multipleOf}`,
          code: 'MULTIPLE_OF',
        });
      }
    }

    // Array length
    if (Array.isArray(value)) {
      if (schema.minItems !== undefined && value.length < schema.minItems) {
        errors.push({
          path,
          message: `Array too short. Minimum items: ${schema.minItems}`,
          code: 'MIN_ITEMS',
        });
      }

      if (schema.maxItems !== undefined && value.length > schema.maxItems) {
        errors.push({
          path,
          message: `Array too long. Maximum items: ${schema.maxItems}`,
          code: 'MAX_ITEMS',
        });
      }

      if (schema.uniqueItems && new Set(value).size !== value.length) {
        errors.push({
          path,
          message: 'Array items must be unique',
          code: 'UNIQUE_ITEMS',
        });
      }
    }
  }

  /**
   * Validate object properties
   * @param {Object} obj - Object to validate
   * @param {Object} schema - Object schema
   * @param {string} basePath - Base path
   * @param {Array} errors - Error array
   * @param {Array} warnings - Warning array
   * @param {Object} options - Validation options
   */
  validateObject(obj, schema, basePath, errors, warnings, options) {
    if (typeof obj !== 'object' || obj === null) {
      return;
    }

    // Validate required properties
    if (schema.required) {
      schema.required.forEach(prop => {
        if (!(prop in obj)) {
          errors.push({
            path: basePath ? `${basePath}.${prop}` : prop,
            message: `Required property '${prop}' is missing`,
            code: 'REQUIRED_PROPERTY',
          });
        }
      });
    }

    // Validate properties
    Object.entries(schema.properties || {}).forEach(([prop, propSchema]) => {
      const propPath = basePath ? `${basePath}.${prop}` : prop;
      if (prop in obj) {
        this.validateValue(obj[prop], propSchema, propPath, errors, warnings, options);
      }
    });

    // Check for additional properties
    if (schema.additionalProperties === false) {
      const allowedProps = new Set(Object.keys(schema.properties || {}));
      Object.keys(obj).forEach(prop => {
        if (!allowedProps.has(prop)) {
          warnings.push({
            path: basePath ? `${basePath}.${prop}` : prop,
            message: `Additional property '${prop}' is not allowed`,
            code: 'ADDITIONAL_PROPERTY',
          });
        }
      });
    }
  }

  /**
   * Validate array items
   * @param {Array} arr - Array to validate
   * @param {Object} schema - Array schema
   * @param {string} basePath - Base path
   * @param {Array} errors - Error array
   * @param {Array} warnings - Warning array
   * @param {Object} options - Validation options
   */
  validateArray(arr, schema, basePath, errors, warnings, options) {
    if (!Array.isArray(arr)) {
      return;
    }

    arr.forEach((item, index) => {
      const itemPath = `${basePath}[${index}]`;
      this.validateValue(item, schema.items, itemPath, errors, warnings, options);
    });
  }

  /**
   * Sanitize input based on schema
   * @param {any} input - Input to sanitize
   * @param {Object} schema - Schema definition
   * @returns {any} Sanitized input
   */
  sanitizeInput(input, schema) {
    if (input === null || input === undefined) {
      return input;
    }

    switch (schema.type) {
      case 'string':
        return this.sanitizeString(input, schema);
      case 'number':
      case 'integer':
        return this.sanitizeNumber(input, schema);
      case 'object':
        return this.sanitizeObject(input, schema);
      case 'array':
        return this.sanitizeArray(input, schema);
      default:
        return input;
    }
  }

  /**
   * Sanitize string value
   * @param {string} value - String value
   * @param {Object} schema - String schema
   * @returns {string} Sanitized string
   */
  sanitizeString(value, schema) {
    let sanitized = String(value);

    // Trim whitespace
    if (schema.trim !== false) {
      sanitized = sanitized.trim();
    }

    // Apply length limits
    if (schema.maxLength) {
      sanitized = sanitized.substring(0, schema.maxLength);
    }

    // Convert case
    if (schema.case === 'lower') {
      sanitized = sanitized.toLowerCase();
    } else if (schema.case === 'upper') {
      sanitized = sanitized.toUpperCase();
    }

    return sanitized;
  }

  /**
   * Sanitize number value
   * @param {number} value - Number value
   * @param {Object} schema - Number schema
   * @returns {number} Sanitized number
   */
  sanitizeNumber(value, schema) {
    let sanitized = Number(value);

    if (isNaN(sanitized)) {
      return schema.default || 0;
    }

    // Apply range limits
    if (schema.minimum !== undefined) {
      sanitized = Math.max(sanitized, schema.minimum);
    }

    if (schema.maximum !== undefined) {
      sanitized = Math.min(sanitized, schema.maximum);
    }

    // Round to integer if needed
    if (schema.type === 'integer') {
      sanitized = Math.round(sanitized);
    }

    return sanitized;
  }

  /**
   * Sanitize object value
   * @param {Object} value - Object value
   * @param {Object} schema - Object schema
   * @returns {Object} Sanitized object
   */
  sanitizeObject(value, schema) {
    if (typeof value !== 'object' || value === null) {
      return {};
    }

    const sanitized = {};
    const properties = schema.properties || {};

    Object.entries(properties).forEach(([prop, propSchema]) => {
      if (prop in value) {
        sanitized[prop] = this.sanitizeInput(value[prop], propSchema);
      }
    });

    // Include additional properties if allowed
    if (schema.additionalProperties !== false) {
      Object.keys(value).forEach(prop => {
        if (!(prop in properties)) {
          sanitized[prop] = value[prop];
        }
      });
    }

    return sanitized;
  }

  /**
   * Sanitize array value
   * @param {Array} value - Array value
   * @param {Object} schema - Array schema
   * @returns {Array} Sanitized array
   */
  sanitizeArray(value, schema) {
    if (!Array.isArray(value)) {
      return [];
    }

    let sanitized = value.map(item => this.sanitizeInput(item, schema.items || {}));

    // Apply length limits
    if (schema.maxItems) {
      sanitized = sanitized.slice(0, schema.maxItems);
    }

    // Remove duplicates if unique items required
    if (schema.uniqueItems) {
      sanitized = [...new Set(sanitized)];
    }

    return sanitized;
  }

  /**
   * Clear validation cache
   */
  clearCache() {
    this.validationCache.clear();
  }

  /**
   * Get validation statistics
   * @returns {Object} Validation statistics
   */
  getStats() {
    return {
      schemas: this.schemas.size,
      customValidators: this.customValidators.size,
      cacheSize: this.validationCache.size,
    };
  }
}

/**
 * Predefined validation schemas for common use cases
 */
export const predefinedSchemas = {
  userInput: {
    type: 'object',
    properties: {
      text: {
        type: 'string',
        maxLength: 10000,
        required: true,
        trim: true,
      },
      metadata: {
        type: 'object',
        properties: {
          timestamp: { type: 'string', format: 'datetime' },
          source: { type: 'string', maxLength: 100 },
        },
      },
    },
    required: ['text'],
  },

  modelConfig: {
    type: 'object',
    properties: {
      modelId: {
        type: 'string',
        pattern: '^[a-zA-Z0-9_-]+$',
        maxLength: 50,
        required: true,
      },
      temperature: {
        type: 'number',
        minimum: 0.1,
        maximum: 2.0,
        default: 0.7,
      },
      maxLength: {
        type: 'integer',
        minimum: 1,
        maximum: 2048,
        default: 100,
      },
      doSample: {
        type: 'boolean',
        default: true,
      },
    },
    required: ['modelId'],
  },

  apiRequest: {
    type: 'object',
    properties: {
      endpoint: {
        type: 'string',
        format: 'url',
        required: true,
      },
      method: {
        type: 'string',
        enum: ['GET', 'POST', 'PUT', 'DELETE'],
        default: 'GET',
      },
      headers: {
        type: 'object',
        additionalProperties: { type: 'string' },
      },
      body: {
        type: 'string',
        maxLength: 1000000,
      },
    },
    required: ['endpoint'],
  },

  fileUpload: {
    type: 'object',
    properties: {
      filename: {
        type: 'string',
        pattern: '^[a-zA-Z0-9._-]+$',
        maxLength: 255,
        required: true,
      },
      size: {
        type: 'integer',
        minimum: 1,
        maximum: 10485760, // 10MB
        required: true,
      },
      type: {
        type: 'string',
        enum: ['text/plain', 'application/json', 'text/markdown'],
        required: true,
      },
      content: {
        type: 'string',
        required: true,
      },
    },
    required: ['filename', 'size', 'type', 'content'],
  },
};

/**
 * Predefined custom validators
 */
export const predefinedValidators = {
  noScriptTags: value => {
    const hasScript = /<script[^>]*>.*?<\/script>/gi.test(value);
    return {
      valid: !hasScript,
      message: 'Script tags are not allowed',
    };
  },

  safePath: value => {
    const hasDotDot = /\.\./.test(value);
    const hasNullByte = /\0/.test(value);
    return {
      valid: !hasDotDot && !hasNullByte,
      message: 'Path contains dangerous characters',
    };
  },

  strongPassword: value => {
    const hasLength = value.length >= 8;
    const hasUpper = /[A-Z]/.test(value);
    const hasLower = /[a-z]/.test(value);
    const hasDigit = /\d/.test(value);
    const hasSpecial = /[!@#$%^&*(),.?":{}|<>]/.test(value);

    const score = [hasLength, hasUpper, hasLower, hasDigit, hasSpecial].reduce(
      (sum, check) => sum + (check ? 1 : 0),
      0
    );

    return {
      valid: score >= 4,
      message:
        'Password must contain at least 4 of: length 8+, uppercase, lowercase, digit, special character',
    };
  },

  noSqlInjection: value => {
    const sqlPatterns = [
      /('|\\')|(;|%3B)|(--|\\-\\-)/gi,
      /(union|select|insert|delete|update|drop|create|alter|exec|execute)\s/gi,
    ];

    const hasSqlPattern = sqlPatterns.some(pattern => pattern.test(value));
    return {
      valid: !hasSqlPattern,
      message: 'Input contains potential SQL injection patterns',
    };
  },
};

// Global validator instance
let globalValidator = null;

/**
 * Initialize global validator
 * @param {Object} options - Validator options
 * @returns {SchemaValidator} Validator instance
 */
export function initializeValidator(options = {}) {
  globalValidator = new SchemaValidator();

  // Register predefined schemas
  Object.entries(predefinedSchemas).forEach(([name, schema]) => {
    globalValidator.registerSchema(name, schema);
  });

  // Register predefined validators
  Object.entries(predefinedValidators).forEach(([name, validator]) => {
    globalValidator.registerValidator(name, validator);
  });

  // Register custom schemas and validators
  if (options.schemas) {
    Object.entries(options.schemas).forEach(([name, schema]) => {
      globalValidator.registerSchema(name, schema);
    });
  }

  if (options.validators) {
    Object.entries(options.validators).forEach(([name, validator]) => {
      globalValidator.registerValidator(name, validator);
    });
  }

  return globalValidator;
}

/**
 * Get global validator instance
 * @returns {SchemaValidator} Validator instance
 */
export function getValidator() {
  if (!globalValidator) {
    throw new Error('Validator not initialized. Call initializeValidator() first.');
  }
  return globalValidator;
}

/**
 * Quick validation function
 * @param {string} schemaName - Schema name
 * @param {any} input - Input to validate
 * @param {Object} options - Validation options
 * @returns {Object} Validation result
 */
export function validate(schemaName, input, options = {}) {
  return getValidator().validate(schemaName, input, options);
}

export default {
  SchemaValidator,
  predefinedSchemas,
  predefinedValidators,
  initializeValidator,
  getValidator,
  validate,
};
