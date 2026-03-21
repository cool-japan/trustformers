import 'dart:io';
import 'trustformers_types.dart';

/// Configuration for TrustformeRS mobile inference engine
class TrustformersConfig {
  final String engineId;
  final String modelPath;
  final TrustformersPlatform platform;
  final TrustformersBackend backend;
  final TrustformersMemoryOptimization memoryOptimization;
  final int maxMemoryMb;
  final bool useFp16;
  final TrustformersQuantizationConfig? quantization;
  final int numThreads;
  final bool enableBatching;
  final int maxBatchSize;

  const TrustformersConfig({
    required this.engineId,
    required this.modelPath,
    this.platform = TrustformersPlatform.generic,
    this.backend = TrustformersBackend.cpu,
    this.memoryOptimization = TrustformersMemoryOptimization.balanced,
    this.maxMemoryMb = 512,
    this.useFp16 = true,
    this.quantization,
    this.numThreads = 0,
    this.enableBatching = false,
    this.maxBatchSize = 1,
  });

  /// Create iOS-optimized configuration
  factory TrustformersConfig.iosOptimized({
    required String engineId,
    required String modelPath,
    int maxMemoryMb = 1024,
    int maxBatchSize = 4,
  }) {
    return TrustformersConfig(
      engineId: engineId,
      modelPath: modelPath,
      platform: TrustformersPlatform.ios,
      backend: TrustformersBackend.coreml,
      memoryOptimization: TrustformersMemoryOptimization.balanced,
      maxMemoryMb: maxMemoryMb,
      useFp16: true,
      quantization: const TrustformersQuantizationConfig(
        scheme: TrustformersQuantizationScheme.fp16,
        dynamic: false,
        perChannel: true,
      ),
      numThreads: 0,
      enableBatching: true,
      maxBatchSize: maxBatchSize,
    );
  }

  /// Create Android-optimized configuration
  factory TrustformersConfig.androidOptimized({
    required String engineId,
    required String modelPath,
    int maxMemoryMb = 768,
  }) {
    return TrustformersConfig(
      engineId: engineId,
      modelPath: modelPath,
      platform: TrustformersPlatform.android,
      backend: TrustformersBackend.nnapi,
      memoryOptimization: TrustformersMemoryOptimization.balanced,
      maxMemoryMb: maxMemoryMb,
      useFp16: true,
      quantization: const TrustformersQuantizationConfig(
        scheme: TrustformersQuantizationScheme.int8,
        dynamic: true,
        perChannel: false,
      ),
      numThreads: 0,
      enableBatching: false,
      maxBatchSize: 1,
    );
  }

  /// Create ultra low memory configuration
  factory TrustformersConfig.ultraLowMemory({
    required String engineId,
    required String modelPath,
    int maxMemoryMb = 256,
  }) {
    return TrustformersConfig(
      engineId: engineId,
      modelPath: modelPath,
      platform: TrustformersPlatform.generic,
      backend: TrustformersBackend.cpu,
      memoryOptimization: TrustformersMemoryOptimization.maximum,
      maxMemoryMb: maxMemoryMb,
      useFp16: true,
      quantization: const TrustformersQuantizationConfig(
        scheme: TrustformersQuantizationScheme.int4,
        dynamic: true,
        perChannel: true,
      ),
      numThreads: 1,
      enableBatching: false,
      maxBatchSize: 1,
    );
  }

  /// Auto-detect optimal configuration based on platform
  factory TrustformersConfig.autoDetect({
    required String engineId,
    required String modelPath,
  }) {
    if (Platform.isIOS) {
      return TrustformersConfig.iosOptimized(
        engineId: engineId,
        modelPath: modelPath,
      );
    } else if (Platform.isAndroid) {
      return TrustformersConfig.androidOptimized(
        engineId: engineId,
        modelPath: modelPath,
      );
    } else {
      return TrustformersConfig(
        engineId: engineId,
        modelPath: modelPath,
        platform: TrustformersPlatform.generic,
        backend: TrustformersBackend.cpu,
      );
    }
  }

  /// Validate configuration
  bool validate() {
    // Check memory constraints
    if (maxMemoryMb < 64 || maxMemoryMb > 4096) {
      return false;
    }

    // Check platform-backend compatibility
    if (platform == TrustformersPlatform.ios &&
        backend == TrustformersBackend.nnapi) {
      return false;
    }

    if (platform == TrustformersPlatform.android &&
        backend == TrustformersBackend.coreml) {
      return false;
    }

    // Check batch configuration
    if (enableBatching && maxBatchSize == 0) {
      return false;
    }

    // Check thread count
    if (numThreads > 16) {
      return false;
    }

    return true;
  }

  /// Estimate memory usage for the configuration
  int estimateMemoryUsage(int modelSizeMb) {
    int totalMemory = modelSizeMb;

    // Apply quantization reduction
    if (quantization != null) {
      final reductionFactor = switch (quantization!.scheme) {
        TrustformersQuantizationScheme.int4 => 8,
        TrustformersQuantizationScheme.int8 => 4,
        TrustformersQuantizationScheme.fp16 => 2,
        TrustformersQuantizationScheme.dynamic => 3,
      };
      totalMemory = modelSizeMb ~/ reductionFactor;
    } else if (useFp16) {
      totalMemory = modelSizeMb ~/ 2;
    }

    // Add runtime overhead
    final runtimeOverhead = switch (memoryOptimization) {
      TrustformersMemoryOptimization.minimal => totalMemory ~/ 2,
      TrustformersMemoryOptimization.balanced => totalMemory ~/ 4,
      TrustformersMemoryOptimization.maximum => totalMemory ~/ 8,
    };

    return totalMemory + runtimeOverhead;
  }

  /// Get recommended thread count for the platform
  int getThreadCount() {
    if (numThreads > 0) {
      return numThreads;
    }

    // Auto-detect based on platform and optimization level
    final baseThreads = switch (platform) {
      TrustformersPlatform.ios => 4,
      TrustformersPlatform.android => 2,
      TrustformersPlatform.generic => 2,
    };

    return switch (memoryOptimization) {
      TrustformersMemoryOptimization.maximum => 1,
      TrustformersMemoryOptimization.balanced => baseThreads,
      TrustformersMemoryOptimization.minimal => baseThreads * 2,
    };
  }

  /// Copy configuration with modified parameters
  TrustformersConfig copyWith({
    String? engineId,
    String? modelPath,
    TrustformersPlatform? platform,
    TrustformersBackend? backend,
    TrustformersMemoryOptimization? memoryOptimization,
    int? maxMemoryMb,
    bool? useFp16,
    TrustformersQuantizationConfig? quantization,
    int? numThreads,
    bool? enableBatching,
    int? maxBatchSize,
  }) {
    return TrustformersConfig(
      engineId: engineId ?? this.engineId,
      modelPath: modelPath ?? this.modelPath,
      platform: platform ?? this.platform,
      backend: backend ?? this.backend,
      memoryOptimization: memoryOptimization ?? this.memoryOptimization,
      maxMemoryMb: maxMemoryMb ?? this.maxMemoryMb,
      useFp16: useFp16 ?? this.useFp16,
      quantization: quantization ?? this.quantization,
      numThreads: numThreads ?? this.numThreads,
      enableBatching: enableBatching ?? this.enableBatching,
      maxBatchSize: maxBatchSize ?? this.maxBatchSize,
    );
  }

  /// Convert to JSON for native method calls
  Map<String, dynamic> toJson() {
    return {
      'engine_id': engineId,
      'model_path': modelPath,
      'platform': platform.toString(),
      'backend': backend.toString(),
      'memory_optimization': memoryOptimization.toString(),
      'max_memory_mb': maxMemoryMb,
      'use_fp16': useFp16,
      if (quantization != null) 'quantization': quantization!.toJson(),
      'num_threads': numThreads,
      'enable_batching': enableBatching,
      'max_batch_size': maxBatchSize,
    };
  }

  /// Create from JSON response
  factory TrustformersConfig.fromJson(Map<String, dynamic> json) {
    return TrustformersConfig(
      engineId: json['engine_id'],
      modelPath: json['model_path'],
      platform: TrustformersPlatform.values.firstWhere(
        (e) => e.toString() == json['platform'],
        orElse: () => TrustformersPlatform.generic,
      ),
      backend: TrustformersBackend.values.firstWhere(
        (e) => e.toString() == json['backend'],
        orElse: () => TrustformersBackend.cpu,
      ),
      memoryOptimization: TrustformersMemoryOptimization.values.firstWhere(
        (e) => e.toString() == json['memory_optimization'],
        orElse: () => TrustformersMemoryOptimization.balanced,
      ),
      maxMemoryMb: json['max_memory_mb'] ?? 512,
      useFp16: json['use_fp16'] ?? true,
      quantization: json['quantization'] != null
          ? TrustformersQuantizationConfig.fromJson(json['quantization'])
          : null,
      numThreads: json['num_threads'] ?? 0,
      enableBatching: json['enable_batching'] ?? false,
      maxBatchSize: json['max_batch_size'] ?? 1,
    );
  }

  @override
  String toString() {
    return 'TrustformersConfig('
        'engineId: $engineId, '
        'platform: $platform, '
        'backend: $backend, '
        'memoryOptimization: $memoryOptimization, '
        'maxMemoryMb: $maxMemoryMb, '
        'useFp16: $useFp16, '
        'enableBatching: $enableBatching)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is TrustformersConfig &&
        other.engineId == engineId &&
        other.modelPath == modelPath &&
        other.platform == platform &&
        other.backend == backend &&
        other.memoryOptimization == memoryOptimization &&
        other.maxMemoryMb == maxMemoryMb &&
        other.useFp16 == useFp16 &&
        other.quantization == quantization &&
        other.numThreads == numThreads &&
        other.enableBatching == enableBatching &&
        other.maxBatchSize == maxBatchSize;
  }

  @override
  int get hashCode {
    return Object.hash(
      engineId,
      modelPath,
      platform,
      backend,
      memoryOptimization,
      maxMemoryMb,
      useFp16,
      quantization,
      numThreads,
      enableBatching,
      maxBatchSize,
    );
  }
}