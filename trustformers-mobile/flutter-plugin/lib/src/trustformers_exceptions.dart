/// Base exception class for TrustformeRS Flutter integration
class TrustformersException implements Exception {
  final String message;
  final String? code;
  final dynamic details;
  final StackTrace? stackTrace;

  const TrustformersException(
    this.message, {
    this.code,
    this.details,
    this.stackTrace,
  });

  @override
  String toString() {
    final buffer = StringBuffer();
    buffer.write('TrustformersException: $message');
    
    if (code != null) {
      buffer.write(' (Code: $code)');
    }
    
    if (details != null) {
      buffer.write('\nDetails: $details');
    }
    
    return buffer.toString();
  }

  /// Create exception from platform exception details
  factory TrustformersException.fromPlatformException(
    String platformCode,
    String? platformMessage,
    dynamic platformDetails,
  ) {
    return TrustformersException(
      platformMessage ?? 'Unknown platform error',
      code: platformCode,
      details: platformDetails,
    );
  }

  /// Create exception with stack trace
  factory TrustformersException.withStackTrace(
    String message, {
    String? code,
    dynamic details,
    required StackTrace stackTrace,
  }) {
    return TrustformersException(
      message,
      code: code,
      details: details,
      stackTrace: stackTrace,
    );
  }
}

/// Exception thrown when engine initialization fails
class TrustformersInitializationException extends TrustformersException {
  final Map<String, dynamic>? config;

  const TrustformersInitializationException(
    String message, {
    String? code,
    dynamic details,
    this.config,
  }) : super(message, code: code, details: details);

  @override
  String toString() {
    final buffer = StringBuffer();
    buffer.write('TrustformersInitializationException: $message');
    
    if (code != null) {
      buffer.write(' (Code: $code)');
    }
    
    if (config != null) {
      buffer.write('\nConfig: $config');
    }
    
    return buffer.toString();
  }
}

/// Exception thrown when model loading fails
class TrustformersModelException extends TrustformersException {
  final String? modelPath;
  final String? engineId;

  const TrustformersModelException(
    String message, {
    String? code,
    dynamic details,
    this.modelPath,
    this.engineId,
  }) : super(message, code: code, details: details);

  @override
  String toString() {
    final buffer = StringBuffer();
    buffer.write('TrustformersModelException: $message');
    
    if (code != null) {
      buffer.write(' (Code: $code)');
    }
    
    if (modelPath != null) {
      buffer.write('\nModel Path: $modelPath');
    }
    
    if (engineId != null) {
      buffer.write('\nEngine ID: $engineId');
    }
    
    return buffer.toString();
  }
}

/// Exception thrown when inference fails
class TrustformersInferenceException extends TrustformersException {
  final String? engineId;
  final Map<String, dynamic>? inputData;
  final double? timeoutMs;

  const TrustformersInferenceException(
    String message, {
    String? code,
    dynamic details,
    this.engineId,
    this.inputData,
    this.timeoutMs,
  }) : super(message, code: code, details: details);

  @override
  String toString() {
    final buffer = StringBuffer();
    buffer.write('TrustformersInferenceException: $message');
    
    if (code != null) {
      buffer.write(' (Code: $code)');
    }
    
    if (engineId != null) {
      buffer.write('\nEngine ID: $engineId');
    }
    
    if (timeoutMs != null) {
      buffer.write('\nTimeout: ${timeoutMs}ms');
    }
    
    return buffer.toString();
  }
}

/// Exception thrown when configuration is invalid
class TrustformersConfigurationException extends TrustformersException {
  final Map<String, dynamic>? invalidConfig;
  final List<String>? validationErrors;

  const TrustformersConfigurationException(
    String message, {
    String? code,
    dynamic details,
    this.invalidConfig,
    this.validationErrors,
  }) : super(message, code: code, details: details);

  @override
  String toString() {
    final buffer = StringBuffer();
    buffer.write('TrustformersConfigurationException: $message');
    
    if (code != null) {
      buffer.write(' (Code: $code)');
    }
    
    if (validationErrors != null && validationErrors!.isNotEmpty) {
      buffer.write('\nValidation Errors:');
      for (final error in validationErrors!) {
        buffer.write('\n  - $error');
      }
    }
    
    return buffer.toString();
  }
}

/// Exception thrown when memory constraints are exceeded
class TrustformersMemoryException extends TrustformersException {
  final int? requestedMemoryMb;
  final int? availableMemoryMb;
  final int? maxMemoryMb;

  const TrustformersMemoryException(
    String message, {
    String? code,
    dynamic details,
    this.requestedMemoryMb,
    this.availableMemoryMb,
    this.maxMemoryMb,
  }) : super(message, code: code, details: details);

  @override
  String toString() {
    final buffer = StringBuffer();
    buffer.write('TrustformersMemoryException: $message');
    
    if (code != null) {
      buffer.write(' (Code: $code)');
    }
    
    if (requestedMemoryMb != null) {
      buffer.write('\nRequested: ${requestedMemoryMb}MB');
    }
    
    if (availableMemoryMb != null) {
      buffer.write('\nAvailable: ${availableMemoryMb}MB');
    }
    
    if (maxMemoryMb != null) {
      buffer.write('\nMax Allowed: ${maxMemoryMb}MB');
    }
    
    return buffer.toString();
  }
}

/// Exception thrown when platform is not supported
class TrustformersPlatformException extends TrustformersException {
  final String? currentPlatform;
  final List<String>? supportedPlatforms;

  const TrustformersPlatformException(
    String message, {
    String? code,
    dynamic details,
    this.currentPlatform,
    this.supportedPlatforms,
  }) : super(message, code: code, details: details);

  @override
  String toString() {
    final buffer = StringBuffer();
    buffer.write('TrustformersPlatformException: $message');
    
    if (code != null) {
      buffer.write(' (Code: $code)');
    }
    
    if (currentPlatform != null) {
      buffer.write('\nCurrent Platform: $currentPlatform');
    }
    
    if (supportedPlatforms != null && supportedPlatforms!.isNotEmpty) {
      buffer.write('\nSupported Platforms: ${supportedPlatforms!.join(', ')}');
    }
    
    return buffer.toString();
  }
}

/// Exception thrown when timeout occurs
class TrustformersTimeoutException extends TrustformersException {
  final Duration timeout;
  final String operation;

  const TrustformersTimeoutException(
    String message, {
    String? code,
    dynamic details,
    required this.timeout,
    required this.operation,
  }) : super(message, code: code, details: details);

  @override
  String toString() {
    final buffer = StringBuffer();
    buffer.write('TrustformersTimeoutException: $message');
    
    if (code != null) {
      buffer.write(' (Code: $code)');
    }
    
    buffer.write('\nOperation: $operation');
    buffer.write('\nTimeout: ${timeout.inMilliseconds}ms');
    
    return buffer.toString();
  }
}

/// Exception thrown when backend is not available
class TrustformersBackendException extends TrustformersException {
  final String? requestedBackend;
  final List<String>? availableBackends;

  const TrustformersBackendException(
    String message, {
    String? code,
    dynamic details,
    this.requestedBackend,
    this.availableBackends,
  }) : super(message, code: code, details: details);

  @override
  String toString() {
    final buffer = StringBuffer();
    buffer.write('TrustformersBackendException: $message');
    
    if (code != null) {
      buffer.write(' (Code: $code)');
    }
    
    if (requestedBackend != null) {
      buffer.write('\nRequested Backend: $requestedBackend');
    }
    
    if (availableBackends != null && availableBackends!.isNotEmpty) {
      buffer.write('\nAvailable Backends: ${availableBackends!.join(', ')}');
    }
    
    return buffer.toString();
  }
}

/// Exception utility methods
class TrustformersExceptionUtils {
  /// Check if exception is retryable
  static bool isRetryable(TrustformersException exception) {
    if (exception is TrustformersTimeoutException) return true;
    if (exception is TrustformersMemoryException) return false;
    if (exception is TrustformersPlatformException) return false;
    if (exception is TrustformersConfigurationException) return false;
    
    // Check by error code
    if (exception.code != null) {
      switch (exception.code!) {
        case 'NETWORK_ERROR':
        case 'TEMPORARY_FAILURE':
        case 'RESOURCE_BUSY':
          return true;
        case 'INVALID_CONFIG':
        case 'UNSUPPORTED_PLATFORM':
        case 'INITIALIZATION_FAILED':
          return false;
        default:
          return false;
      }
    }
    
    return false;
  }

  /// Get suggested retry delay
  static Duration getRetryDelay(TrustformersException exception, int attemptNumber) {
    if (!isRetryable(exception)) {
      return Duration.zero;
    }
    
    // Exponential backoff with jitter
    final baseDelay = Duration(milliseconds: 100 * (1 << attemptNumber));
    final jitter = Duration(milliseconds: (baseDelay.inMilliseconds * 0.1).round());
    
    return baseDelay + jitter;
  }

  /// Get maximum retry attempts for exception type
  static int getMaxRetries(TrustformersException exception) {
    if (exception is TrustformersTimeoutException) return 3;
    if (exception.code == 'NETWORK_ERROR') return 5;
    if (exception.code == 'TEMPORARY_FAILURE') return 2;
    
    return 1;
  }

  /// Convert exception to user-friendly message
  static String getUserFriendlyMessage(TrustformersException exception) {
    if (exception is TrustformersInitializationException) {
      return 'Failed to initialize TrustformeRS engine. Please check your configuration.';
    }
    
    if (exception is TrustformersModelException) {
      return 'Failed to load the AI model. Please check the model file and try again.';
    }
    
    if (exception is TrustformersInferenceException) {
      return 'AI inference failed. Please try again or check your input data.';
    }
    
    if (exception is TrustformersMemoryException) {
      return 'Not enough memory available. Try using a smaller model or closing other apps.';
    }
    
    if (exception is TrustformersPlatformException) {
      return 'This device or platform is not supported by TrustformeRS.';
    }
    
    if (exception is TrustformersTimeoutException) {
      return 'Operation timed out. Please check your network connection and try again.';
    }
    
    if (exception is TrustformersBackendException) {
      return 'The requested AI acceleration backend is not available on this device.';
    }
    
    return 'An unexpected error occurred. Please try again.';
  }

  /// Check if exception indicates a configuration problem
  static bool isConfigurationError(TrustformersException exception) {
    return exception is TrustformersConfigurationException ||
           exception is TrustformersPlatformException ||
           exception is TrustformersBackendException ||
           (exception.code?.contains('CONFIG') ?? false);
  }

  /// Check if exception indicates a resource problem
  static bool isResourceError(TrustformersException exception) {
    return exception is TrustformersMemoryException ||
           exception is TrustformersTimeoutException ||
           (exception.code?.contains('RESOURCE') ?? false) ||
           (exception.code?.contains('MEMORY') ?? false);
  }
}