/**
 * Security and Privacy Module for TrustformeRS
 * Provides input sanitization, CSP compliance, and privacy protection features
 */

/**
 * Input Sanitizer
 * Sanitizes user inputs to prevent injection attacks and malicious content
 */
export class InputSanitizer {
  constructor(options = {}) {
    this.options = {
      allowHtml: false,
      maxLength: 10000,
      allowedTags: ['b', 'i', 'em', 'strong', 'code', 'pre'],
      allowedAttributes: ['class', 'id'],
      enableUrlValidation: true,
      enableSqlInjectionPrevention: true,
      enableXssProtection: true,
      enableCommandInjectionPrevention: true,
      ...options,
    };

    this.dangerousPatterns = [
      // XSS patterns
      /<script[^>]*>.*?<\/script>/gi,
      /javascript:/gi,
      /vbscript:/gi,
      /on\w+\s*=/gi,
      /<iframe[^>]*>/gi,
      /<object[^>]*>/gi,
      /<embed[^>]*>/gi,
      /<form[^>]*>/gi,

      // SQL injection patterns
      /('|\\')|(;|%3B)|(--|\\-\\-)/gi,
      /(union|select|insert|delete|update|drop|create|alter|exec|execute)/gi,

      // Command injection patterns
      /(\||&|;|`|\$\(|\$\{)/gi,
      /(eval|exec|system|shell_exec|passthru)/gi,

      // Path traversal
      /(\.\.(\/|\\))+/gi,
      /(\/etc\/passwd|\/windows\/system32)/gi,
    ];

    this.htmlEntities = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#x27;',
      '/': '&#x2F;',
      '`': '&#x60;',
      '=': '&#x3D;',
    };
  }

  /**
   * Sanitize text input
   * @param {string} input - Input text to sanitize
   * @param {Object} options - Sanitization options
   * @returns {string} Sanitized text
   */
  sanitizeText(input, options = {}) {
    if (typeof input !== 'string') {
      throw new Error('Input must be a string');
    }

    const opts = { ...this.options, ...options };
    let sanitized = input;

    // Length validation
    if (sanitized.length > opts.maxLength) {
      sanitized = sanitized.substring(0, opts.maxLength);
    }

    // Remove dangerous patterns
    if (opts.enableXssProtection) {
      sanitized = this.removeXssPatterns(sanitized);
    }

    if (opts.enableSqlInjectionPrevention) {
      sanitized = this.removeSqlInjectionPatterns(sanitized);
    }

    if (opts.enableCommandInjectionPrevention) {
      sanitized = this.removeCommandInjectionPatterns(sanitized);
    }

    // HTML handling
    if (!opts.allowHtml) {
      sanitized = this.escapeHtml(sanitized);
    } else {
      sanitized = this.sanitizeHtml(sanitized, opts);
    }

    // URL validation
    if (opts.enableUrlValidation) {
      sanitized = this.sanitizeUrls(sanitized);
    }

    return sanitized.trim();
  }

  /**
   * Remove XSS patterns
   * @param {string} input - Input string
   * @returns {string} Sanitized string
   */
  removeXssPatterns(input) {
    let sanitized = input;

    // Remove script tags and content
    sanitized = sanitized.replace(/<script[^>]*>.*?<\/script>/gis, '');

    // Remove javascript: and vbscript: protocols
    sanitized = sanitized.replace(/(javascript|vbscript):/gi, '');

    // Remove event handlers
    sanitized = sanitized.replace(/on\w+\s*=\s*["'][^"']*["']/gi, '');

    // Remove dangerous tags
    const dangerousTags = ['iframe', 'object', 'embed', 'form', 'input', 'meta', 'link'];
    dangerousTags.forEach(tag => {
      const regex = new RegExp(`<${tag}[^>]*>.*?<\/${tag}>`, 'gis');
      sanitized = sanitized.replace(regex, '');

      const selfClosingRegex = new RegExp(`<${tag}[^>]*\/>`, 'gis');
      sanitized = sanitized.replace(selfClosingRegex, '');
    });

    return sanitized;
  }

  /**
   * Remove SQL injection patterns
   * @param {string} input - Input string
   * @returns {string} Sanitized string
   */
  removeSqlInjectionPatterns(input) {
    let sanitized = input;

    // Remove common SQL injection patterns
    const sqlPatterns = [
      /('|\\')|(;|%3B)|(--|\\-\\-)/gi,
      /(union|select|insert|delete|update|drop|create|alter|exec|execute)\s/gi,
      /\b(or|and)\s+1\s*=\s*1/gi,
      /\b(or|and)\s+['"]\w+['"]\s*=\s*['"]\w+['"]]/gi,
    ];

    sqlPatterns.forEach(pattern => {
      sanitized = sanitized.replace(pattern, '');
    });

    return sanitized;
  }

  /**
   * Remove command injection patterns
   * @param {string} input - Input string
   * @returns {string} Sanitized string
   */
  removeCommandInjectionPatterns(input) {
    let sanitized = input;

    // Remove command injection patterns
    const commandPatterns = [
      /(\||&|;|`|\$\(|\$\{)/gi,
      /(eval|exec|system|shell_exec|passthru|proc_open|popen|file_get_contents)/gi,
      /(\.\.(\/|\\))+/gi,
    ];

    commandPatterns.forEach(pattern => {
      sanitized = sanitized.replace(pattern, '');
    });

    return sanitized;
  }

  /**
   * Escape HTML entities
   * @param {string} input - Input string
   * @returns {string} Escaped string
   */
  escapeHtml(input) {
    return input.replace(/[&<>"'`=\/]/g, char => this.htmlEntities[char] || char);
  }

  /**
   * Sanitize HTML while preserving allowed tags
   * @param {string} input - Input HTML
   * @param {Object} options - Sanitization options
   * @returns {string} Sanitized HTML
   */
  sanitizeHtml(input, options) {
    const { allowedTags, allowedAttributes } = options;

    // Simple HTML sanitization (for production, consider using DOMPurify)
    let sanitized = input;

    // Remove all tags except allowed ones
    const tagPattern = /<\/?([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>/g;
    sanitized = sanitized.replace(tagPattern, (match, tagName) => {
      if (allowedTags.includes(tagName.toLowerCase())) {
        // Clean attributes
        return this.cleanAttributes(match, allowedAttributes);
      }
      return '';
    });

    return sanitized;
  }

  /**
   * Clean HTML attributes
   * @param {string} tag - HTML tag
   * @param {Array} allowedAttributes - Allowed attributes
   * @returns {string} Cleaned tag
   */
  cleanAttributes(tag, allowedAttributes) {
    // Simple attribute cleaning
    const attrPattern = /(\w+)\s*=\s*["'][^"']*["']/g;
    return tag.replace(attrPattern, (match, attrName) =>
      allowedAttributes.includes(attrName.toLowerCase()) ? match : ''
    );
  }

  /**
   * Sanitize URLs
   * @param {string} input - Input string containing URLs
   * @returns {string} Sanitized string
   */
  sanitizeUrls(input) {
    const urlPattern = /(https?:\/\/[^\s]+)/gi;
    return input.replace(urlPattern, url => {
      try {
        const urlObj = new URL(url);

        // Block dangerous protocols
        const allowedProtocols = ['http:', 'https:'];
        if (!allowedProtocols.includes(urlObj.protocol)) {
          return '[BLOCKED URL]';
        }

        // Block localhost and private IPs in production
        if (this.isProductionEnvironment() && this.isPrivateUrl(urlObj)) {
          return '[BLOCKED PRIVATE URL]';
        }

        return url;
      } catch (error) {
        return '[INVALID URL]';
      }
    });
  }

  /**
   * Check if URL is private/localhost
   * @param {URL} urlObj - URL object
   * @returns {boolean} True if URL is private
   */
  isPrivateUrl(urlObj) {
    const { hostname } = urlObj;

    // Check for localhost
    if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1') {
      return true;
    }

    // Check for private IP ranges
    const privateRanges = [/^10\./, /^172\.(1[6-9]|2[0-9]|3[01])\./, /^192\.168\./, /^169\.254\./];

    return privateRanges.some(range => range.test(hostname));
  }

  /**
   * Check if running in production environment
   * @returns {boolean} True if production
   */
  isProductionEnvironment() {
    return process?.env?.NODE_ENV === 'production' || window?.location?.protocol === 'https:';
  }

  /**
   * Validate and sanitize file paths
   * @param {string} path - File path
   * @returns {string} Sanitized path
   */
  sanitizeFilePath(path) {
    if (typeof path !== 'string') {
      throw new Error('Path must be a string');
    }

    let sanitized = path;

    // Remove path traversal attempts
    sanitized = sanitized.replace(/\.\.(\/|\\)/g, '');

    // Remove null bytes
    sanitized = sanitized.replace(/\0/g, '');

    // Normalize path separators
    sanitized = sanitized.replace(/\\/g, '/');

    // Remove multiple consecutive slashes
    sanitized = sanitized.replace(/\/+/g, '/');

    // Remove leading slash if not absolute path
    if (!sanitized.startsWith('/') && sanitized.includes(':')) {
      // Windows absolute path
      return sanitized;
    }

    return sanitized.replace(/^\/+/, '');
  }
}

/**
 * Content Security Policy Manager
 * Manages and enforces Content Security Policy directives
 */
export class CSPManager {
  constructor() {
    this.defaultPolicy = {
      'default-src': ["'self'"],
      'script-src': ["'self'", "'wasm-unsafe-eval'"],
      'style-src': ["'self'", "'unsafe-inline'"],
      'img-src': ["'self'", 'data:', 'blob:'],
      'font-src': ["'self'", 'data:'],
      'connect-src': ["'self'"],
      'media-src': ["'self'"],
      'object-src': ["'none'"],
      'child-src': ["'self'"],
      'frame-ancestors': ["'none'"],
      'form-action': ["'self'"],
      'base-uri': ["'self'"],
      'upgrade-insecure-requests': true,
    };

    this.trustedDomains = new Set();
    this.nonces = new Map();
    this.violationReports = [];
  }

  /**
   * Generate CSP header value
   * @param {Object} customPolicy - Custom policy directives
   * @returns {string} CSP header value
   */
  generateCSPHeader(customPolicy = {}) {
    const policy = { ...this.defaultPolicy, ...customPolicy };

    const directives = Object.entries(policy)
      .map(([directive, values]) => {
        if (typeof values === 'boolean') {
          return values ? directive : null;
        }

        if (Array.isArray(values)) {
          return `${directive} ${values.join(' ')}`;
        }

        return `${directive} ${values}`;
      })
      .filter(Boolean);

    return directives.join('; ');
  }

  /**
   * Add trusted domain
   * @param {string} domain - Domain to trust
   * @param {Array} directives - CSP directives to apply to
   */
  addTrustedDomain(domain, directives = ['connect-src']) {
    this.trustedDomains.add(domain);

    directives.forEach(directive => {
      if (!this.defaultPolicy[directive]) {
        this.defaultPolicy[directive] = [];
      }
      if (!this.defaultPolicy[directive].includes(domain)) {
        this.defaultPolicy[directive].push(domain);
      }
    });
  }

  /**
   * Generate nonce for inline scripts/styles
   * @returns {string} Generated nonce
   */
  generateNonce() {
    const nonce = this.generateRandomString(32);
    this.nonces.set(nonce, Date.now());

    // Clean old nonces (older than 1 hour)
    const oneHourAgo = Date.now() - 60 * 60 * 1000;
    for (const [nonceValue, timestamp] of this.nonces.entries()) {
      if (timestamp < oneHourAgo) {
        this.nonces.delete(nonceValue);
      }
    }

    return nonce;
  }

  /**
   * Validate nonce
   * @param {string} nonce - Nonce to validate
   * @returns {boolean} True if valid
   */
  validateNonce(nonce) {
    return this.nonces.has(nonce);
  }

  /**
   * Handle CSP violation report
   * @param {Object} report - Violation report
   */
  handleViolationReport(report) {
    const violation = {
      timestamp: new Date().toISOString(),
      directive: report['violated-directive'],
      blockedUri: report['blocked-uri'],
      sourceFile: report['source-file'],
      lineNumber: report['line-number'],
      columnNumber: report['column-number'],
      userAgent: navigator.userAgent,
    };

    this.violationReports.push(violation);

    // Keep only last 100 reports
    if (this.violationReports.length > 100) {
      this.violationReports = this.violationReports.slice(-100);
    }

    console.warn('CSP Violation:', violation);

    // Send to monitoring service if configured
    this.reportViolation(violation);
  }

  /**
   * Report violation to monitoring service
   * @param {Object} violation - Violation details
   */
  async reportViolation(violation) {
    if (typeof window !== 'undefined' && window.fetch) {
      try {
        await fetch('/api/csp-violation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(violation),
        });
      } catch (error) {
        console.error('Failed to report CSP violation:', error);
      }
    }
  }

  /**
   * Get violation reports
   * @returns {Array} Array of violation reports
   */
  getViolationReports() {
    return [...this.violationReports];
  }

  /**
   * Generate random string
   * @param {number} length - String length
   * @returns {string} Random string
   */
  generateRandomString(length) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  }
}

/**
 * Privacy Manager
 * Manages privacy settings and data protection
 */
export class PrivacyManager {
  constructor() {
    this.consentData = {
      necessary: true,
      analytics: false,
      marketing: false,
      functional: false,
    };

    this.dataRetention = {
      sessionData: 24 * 60 * 60 * 1000, // 24 hours
      userPreferences: 30 * 24 * 60 * 60 * 1000, // 30 days
      analyticsData: 7 * 24 * 60 * 60 * 1000, // 7 days
      crashReports: 14 * 24 * 60 * 60 * 1000, // 14 days
    };

    this.personalDataTypes = [
      'ip_address',
      'user_agent',
      'session_id',
      'device_id',
      'location_data',
      'usage_data',
      'preference_data',
    ];

    this.dataProcessingLog = [];
    this.encryptionKey = null;
  }

  /**
   * Initialize privacy settings
   * @param {Object} options - Privacy options
   */
  initialize(options = {}) {
    this.loadConsent();
    this.setupDataCleanup();
    this.initializeEncryption();

    if (options.enableCookieConsent) {
      this.showCookieConsent();
    }
  }

  /**
   * Load user consent preferences
   */
  loadConsent() {
    try {
      const saved = localStorage.getItem('trustformers_privacy_consent');
      if (saved) {
        this.consentData = { ...this.consentData, ...JSON.parse(saved) };
      }
    } catch (error) {
      console.error('Failed to load consent data:', error);
    }
  }

  /**
   * Save consent preferences
   * @param {Object} consent - Consent preferences
   */
  saveConsent(consent) {
    this.consentData = { ...this.consentData, ...consent };

    try {
      localStorage.setItem('trustformers_privacy_consent', JSON.stringify(this.consentData));
      this.logDataProcessing('consent_updated', consent);
    } catch (error) {
      console.error('Failed to save consent data:', error);
    }
  }

  /**
   * Check if specific consent is granted
   * @param {string} type - Consent type
   * @returns {boolean} True if consent granted
   */
  hasConsent(type) {
    return this.consentData[type] === true;
  }

  /**
   * Anonymize personal data
   * @param {Object} data - Data to anonymize
   * @returns {Object} Anonymized data
   */
  anonymizeData(data) {
    const anonymized = { ...data };

    // Remove or hash personal identifiers
    if (anonymized.ip_address) {
      anonymized.ip_address = this.hashValue(anonymized.ip_address);
    }

    if (anonymized.user_agent) {
      // Keep only browser family and version
      anonymized.user_agent = this.simplifyUserAgent(anonymized.user_agent);
    }

    if (anonymized.session_id) {
      anonymized.session_id = this.hashValue(anonymized.session_id);
    }

    // Remove precise location data
    if (anonymized.location) {
      delete anonymized.location.precise_coordinates;
      // Keep only country/region level
      anonymized.location = {
        country: anonymized.location.country,
        region: anonymized.location.region,
      };
    }

    return anonymized;
  }

  /**
   * Hash sensitive values
   * @param {string} value - Value to hash
   * @returns {string} Hashed value
   */
  hashValue(value) {
    // Simple hash function (in production, use crypto.subtle.digest)
    let hash = 0;
    for (let i = 0; i < value.length; i++) {
      const char = value.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return `hash_${Math.abs(hash).toString(36)}`;
  }

  /**
   * Simplify user agent string
   * @param {string} userAgent - Full user agent
   * @returns {string} Simplified user agent
   */
  simplifyUserAgent(userAgent) {
    // Extract basic browser info
    const browserPatterns = [
      { name: 'Chrome', pattern: /Chrome\/(\d+)/ },
      { name: 'Firefox', pattern: /Firefox\/(\d+)/ },
      { name: 'Safari', pattern: /Safari\/(\d+)/ },
      { name: 'Edge', pattern: /Edge\/(\d+)/ },
    ];

    for (const browser of browserPatterns) {
      const match = userAgent.match(browser.pattern);
      if (match) {
        return `${browser.name}/${match[1].split('.')[0]}`;
      }
    }

    return 'Unknown Browser';
  }

  /**
   * Encrypt sensitive data
   * @param {string} data - Data to encrypt
   * @returns {string} Encrypted data
   */
  async encryptData(data) {
    if (!this.encryptionKey) {
      return data; // Fallback if encryption not available
    }

    try {
      if (typeof window !== 'undefined' && window.crypto && window.crypto.subtle) {
        const encoder = new TextEncoder();
        const dataBuffer = encoder.encode(data);
        const encrypted = await window.crypto.subtle.encrypt(
          { name: 'AES-GCM', iv: new Uint8Array(12) },
          this.encryptionKey,
          dataBuffer
        );
        return btoa(String.fromCharCode(...new Uint8Array(encrypted)));
      }
    } catch (error) {
      console.error('Encryption failed:', error);
    }

    return data;
  }

  /**
   * Initialize encryption
   */
  async initializeEncryption() {
    if (typeof window !== 'undefined' && window.crypto && window.crypto.subtle) {
      try {
        this.encryptionKey = await window.crypto.subtle.generateKey(
          { name: 'AES-GCM', length: 256 },
          false,
          ['encrypt', 'decrypt']
        );
      } catch (error) {
        console.error('Failed to initialize encryption:', error);
      }
    }
  }

  /**
   * Setup automatic data cleanup
   */
  setupDataCleanup() {
    // Clean up expired data every hour
    setInterval(() => {
      this.cleanupExpiredData();
    }, 60 * 60 * 1000);
  }

  /**
   * Clean up expired data
   */
  cleanupExpiredData() {
    const now = Date.now();

    // Clean up session storage
    if (typeof sessionStorage !== 'undefined') {
      for (let i = sessionStorage.length - 1; i >= 0; i--) {
        const key = sessionStorage.key(i);
        if (key && key.startsWith('trustformers_')) {
          try {
            const item = JSON.parse(sessionStorage.getItem(key));
            if (item.timestamp && now - item.timestamp > this.dataRetention.sessionData) {
              sessionStorage.removeItem(key);
            }
          } catch (error) {
            // Invalid JSON, remove item
            sessionStorage.removeItem(key);
          }
        }
      }
    }

    // Clean up processing log
    this.dataProcessingLog = this.dataProcessingLog.filter(
      entry => now - entry.timestamp < this.dataRetention.analyticsData
    );

    this.logDataProcessing('data_cleanup', { cleaned_at: new Date().toISOString() });
  }

  /**
   * Log data processing activity
   * @param {string} action - Action performed
   * @param {Object} details - Action details
   */
  logDataProcessing(action, details = {}) {
    const entry = {
      timestamp: Date.now(),
      action,
      details: this.anonymizeData(details),
      id: this.generateUniqueId(),
    };

    this.dataProcessingLog.push(entry);

    // Keep only last 1000 entries
    if (this.dataProcessingLog.length > 1000) {
      this.dataProcessingLog = this.dataProcessingLog.slice(-1000);
    }
  }

  /**
   * Generate unique ID
   * @returns {string} Unique identifier
   */
  generateUniqueId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  }

  /**
   * Export user data (GDPR compliance)
   * @returns {Object} User data export
   */
  exportUserData() {
    return {
      consent: this.consentData,
      processing_log: this.dataProcessingLog,
      export_date: new Date().toISOString(),
      data_types: this.personalDataTypes,
    };
  }

  /**
   * Delete all user data (GDPR compliance)
   */
  deleteAllUserData() {
    // Clear local storage
    if (typeof localStorage !== 'undefined') {
      const keys = Object.keys(localStorage);
      keys.forEach(key => {
        if (key.startsWith('trustformers_')) {
          localStorage.removeItem(key);
        }
      });
    }

    // Clear session storage
    if (typeof sessionStorage !== 'undefined') {
      const keys = Object.keys(sessionStorage);
      keys.forEach(key => {
        if (key.startsWith('trustformers_')) {
          sessionStorage.removeItem(key);
        }
      });
    }

    // Clear processing log
    this.dataProcessingLog = [];

    // Reset consent
    this.consentData = {
      necessary: true,
      analytics: false,
      marketing: false,
      functional: false,
    };

    this.logDataProcessing('all_data_deleted', { deletion_date: new Date().toISOString() });
  }

  /**
   * Show cookie consent banner
   */
  showCookieConsent() {
    if (
      this.hasConsent('necessary') &&
      (this.hasConsent('analytics') ||
        this.hasConsent('marketing') ||
        this.hasConsent('functional'))
    ) {
      return; // Consent already given
    }

    // Create consent banner (simplified version)
    const banner = document.createElement('div');
    banner.style.cssText = `
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      background: #333;
      color: white;
      padding: 20px;
      z-index: 10000;
      text-align: center;
    `;

    banner.innerHTML = `
      <p>We use cookies to improve your experience. You can choose which types of cookies to accept.</p>
      <button onclick="acceptAllCookies()" style="margin: 5px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px;">Accept All</button>
      <button onclick="acceptNecessaryCookies()" style="margin: 5px; padding: 10px 20px; background: #6c757d; color: white; border: none; border-radius: 5px;">Necessary Only</button>
      <button onclick="customizeCookies()" style="margin: 5px; padding: 10px 20px; background: transparent; color: white; border: 1px solid white; border-radius: 5px;">Customize</button>
    `;

    document.body.appendChild(banner);

    // Global functions for banner
    window.acceptAllCookies = () => {
      this.saveConsent({
        necessary: true,
        analytics: true,
        marketing: true,
        functional: true,
      });
      document.body.removeChild(banner);
    };

    window.acceptNecessaryCookies = () => {
      this.saveConsent({
        necessary: true,
        analytics: false,
        marketing: false,
        functional: false,
      });
      document.body.removeChild(banner);
    };

    window.customizeCookies = () => {
      // Show detailed consent form
      this.showDetailedConsentForm(banner);
    };
  }

  /**
   * Show detailed consent form
   * @param {HTMLElement} banner - Banner element to replace
   */
  showDetailedConsentForm(banner) {
    banner.innerHTML = `
      <div style="max-width: 600px; margin: 0 auto; text-align: left;">
        <h3>Cookie Preferences</h3>
        <label style="display: block; margin: 10px 0;">
          <input type="checkbox" checked disabled> Necessary Cookies (Required)
          <br><small>Essential for basic website functionality</small>
        </label>
        <label style="display: block; margin: 10px 0;">
          <input type="checkbox" id="analytics-consent"> Analytics Cookies
          <br><small>Help us understand how you use our site</small>
        </label>
        <label style="display: block; margin: 10px 0;">
          <input type="checkbox" id="marketing-consent"> Marketing Cookies
          <br><small>Used to show relevant advertisements</small>
        </label>
        <label style="display: block; margin: 10px 0;">
          <input type="checkbox" id="functional-consent"> Functional Cookies
          <br><small>Remember your preferences and settings</small>
        </label>
        <div style="text-align: center; margin-top: 20px;">
          <button onclick="saveCustomConsent()" style="margin: 5px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px;">Save Preferences</button>
        </div>
      </div>
    `;

    window.saveCustomConsent = () => {
      this.saveConsent({
        necessary: true,
        analytics: document.getElementById('analytics-consent').checked,
        marketing: document.getElementById('marketing-consent').checked,
        functional: document.getElementById('functional-consent').checked,
      });
      document.body.removeChild(banner);
    };
  }
}

/**
 * Security Monitor
 * Monitors for security threats and anomalies
 */
export class SecurityMonitor {
  constructor() {
    this.threats = [];
    this.alertThresholds = {
      requestRate: 100, // requests per minute
      errorRate: 10, // errors per minute
      suspiciousPatterns: 5, // suspicious patterns per hour
    };
    this.metrics = {
      requests: 0,
      errors: 0,
      suspiciousActivity: 0,
      startTime: Date.now(),
    };
    this.blacklistedIPs = new Set();
    this.rateLimits = new Map();
  }

  /**
   * Start monitoring
   */
  startMonitoring() {
    // Reset metrics every minute
    setInterval(() => {
      this.resetMetrics();
    }, 60 * 1000);

    // Analyze threats every 5 minutes
    setInterval(() => {
      this.analyzeThreatPatterns();
    }, 5 * 60 * 1000);
  }

  /**
   * Record security event
   * @param {string} type - Event type
   * @param {Object} details - Event details
   */
  recordEvent(type, details = {}) {
    const event = {
      timestamp: Date.now(),
      type,
      details,
      severity: this.calculateSeverity(type, details),
      id: this.generateEventId(),
    };

    this.threats.push(event);
    this.updateMetrics(type);

    // Keep only last 1000 events
    if (this.threats.length > 1000) {
      this.threats = this.threats.slice(-1000);
    }

    // Check for immediate threats
    if (event.severity === 'high') {
      this.handleHighSeverityThreat(event);
    }
  }

  /**
   * Calculate event severity
   * @param {string} type - Event type
   * @param {Object} details - Event details
   * @returns {string} Severity level
   */
  calculateSeverity(type, details) {
    const highSeverityTypes = [
      'xss_attempt',
      'sql_injection',
      'command_injection',
      'csp_violation',
    ];
    const mediumSeverityTypes = ['suspicious_input', 'rate_limit_exceeded', 'unusual_pattern'];

    if (highSeverityTypes.includes(type)) {
      return 'high';
    } else if (mediumSeverityTypes.includes(type)) {
      return 'medium';
    }

    return 'low';
  }

  /**
   * Update metrics
   * @param {string} eventType - Type of event
   */
  updateMetrics(eventType) {
    this.metrics.requests++;

    if (eventType.includes('error') || eventType.includes('violation')) {
      this.metrics.errors++;
    }

    if (
      eventType.includes('suspicious') ||
      eventType.includes('injection') ||
      eventType.includes('xss')
    ) {
      this.metrics.suspiciousActivity++;
    }
  }

  /**
   * Reset metrics
   */
  resetMetrics() {
    this.metrics = {
      requests: 0,
      errors: 0,
      suspiciousActivity: 0,
      startTime: Date.now(),
    };
  }

  /**
   * Handle high severity threats
   * @param {Object} event - Security event
   */
  handleHighSeverityThreat(event) {
    console.error('High severity security threat detected:', event);

    // Add IP to blacklist if available
    if (event.details.ip) {
      this.blacklistedIPs.add(event.details.ip);
    }

    // Send alert to monitoring service
    this.sendSecurityAlert(event);
  }

  /**
   * Send security alert
   * @param {Object} event - Security event
   */
  async sendSecurityAlert(event) {
    if (typeof window !== 'undefined' && window.fetch) {
      try {
        await fetch('/api/security-alert', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(event),
        });
      } catch (error) {
        console.error('Failed to send security alert:', error);
      }
    }
  }

  /**
   * Analyze threat patterns
   */
  analyzeThreatPatterns() {
    const recentThreats = this.threats.filter(
      threat => Date.now() - threat.timestamp < 60 * 60 * 1000 // Last hour
    );

    // Analyze patterns
    const patterns = this.identifyPatterns(recentThreats);

    if (patterns.length > 0) {
      console.warn('Security threat patterns detected:', patterns);
    }
  }

  /**
   * Identify threat patterns
   * @param {Array} threats - Array of threats
   * @returns {Array} Identified patterns
   */
  identifyPatterns(threats) {
    const patterns = [];

    // Group by IP
    const byIP = threats.reduce((acc, threat) => {
      const ip = threat.details.ip || 'unknown';
      if (!acc[ip]) acc[ip] = [];
      acc[ip].push(threat);
      return acc;
    }, {});

    // Check for repeated attacks from same IP
    Object.entries(byIP).forEach(([ip, ipThreats]) => {
      if (ipThreats.length > 5) {
        patterns.push({
          type: 'repeated_attacks',
          ip,
          count: ipThreats.length,
          threats: ipThreats.map(t => t.type),
        });
      }
    });

    return patterns;
  }

  /**
   * Check if IP is blacklisted
   * @param {string} ip - IP address
   * @returns {boolean} True if blacklisted
   */
  isBlacklisted(ip) {
    return this.blacklistedIPs.has(ip);
  }

  /**
   * Get security metrics
   * @returns {Object} Current security metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      uptime: Date.now() - this.metrics.startTime,
      threatCount: this.threats.length,
      blacklistedIPs: this.blacklistedIPs.size,
    };
  }

  /**
   * Generate event ID
   * @returns {string} Unique event ID
   */
  generateEventId() {
    return `sec_${Date.now().toString(36)}${Math.random().toString(36).substr(2)}`;
  }
}

// Global instances
let globalSanitizer = null;
let globalCSPManager = null;
let globalPrivacyManager = null;
let globalSecurityMonitor = null;

/**
 * Initialize security and privacy features
 * @param {Object} options - Configuration options
 * @returns {Object} Security managers
 */
export function initializeSecurity(options = {}) {
  globalSanitizer = new InputSanitizer(options.sanitizer);
  globalCSPManager = new CSPManager();
  globalPrivacyManager = new PrivacyManager();
  globalSecurityMonitor = new SecurityMonitor();

  // Initialize privacy manager
  globalPrivacyManager.initialize(options.privacy);

  // Start security monitoring
  if (options.enableMonitoring !== false) {
    globalSecurityMonitor.startMonitoring();
  }

  // Set up CSP if in browser environment
  if (typeof window !== 'undefined' && options.enableCSP !== false) {
    const cspHeader = globalCSPManager.generateCSPHeader(options.csp);
    console.warn('Recommended CSP header:', cspHeader);
  }

  return {
    sanitizer: globalSanitizer,
    csp: globalCSPManager,
    privacy: globalPrivacyManager,
    monitor: globalSecurityMonitor,
  };
}

/**
 * Get global security managers
 * @returns {Object} Security managers
 */
export function getSecurityManagers() {
  return {
    sanitizer: globalSanitizer,
    csp: globalCSPManager,
    privacy: globalPrivacyManager,
    monitor: globalSecurityMonitor,
  };
}

export default {
  InputSanitizer,
  CSPManager,
  PrivacyManager,
  SecurityMonitor,
  initializeSecurity,
  getSecurityManagers,
};
