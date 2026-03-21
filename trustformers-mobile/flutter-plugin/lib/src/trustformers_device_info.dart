/// Device information for TrustformeRS optimization
class TrustformersDeviceInfo {
  final String platform;
  final String model;
  final int memoryTotalMb;
  final int memoryAvailableMb;
  final int cpuCores;
  final bool gpuAvailable;
  final bool neuralEngineAvailable;
  final Map<String, dynamic>? additionalInfo;

  const TrustformersDeviceInfo({
    required this.platform,
    required this.model,
    required this.memoryTotalMb,
    required this.memoryAvailableMb,
    required this.cpuCores,
    required this.gpuAvailable,
    required this.neuralEngineAvailable,
    this.additionalInfo,
  });

  factory TrustformersDeviceInfo.fromJson(Map<String, dynamic> json) {
    return TrustformersDeviceInfo(
      platform: json['platform'] ?? 'unknown',
      model: json['model'] ?? 'unknown',
      memoryTotalMb: json['memory_total_mb'] ?? 0,
      memoryAvailableMb: json['memory_available_mb'] ?? 0,
      cpuCores: json['cpu_cores'] ?? 1,
      gpuAvailable: json['gpu_available'] ?? false,
      neuralEngineAvailable: json['neural_engine_available'] ?? false,
      additionalInfo: json['additional_info'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'platform': platform,
      'model': model,
      'memory_total_mb': memoryTotalMb,
      'memory_available_mb': memoryAvailableMb,
      'cpu_cores': cpuCores,
      'gpu_available': gpuAvailable,
      'neural_engine_available': neuralEngineAvailable,
      if (additionalInfo != null) 'additional_info': additionalInfo,
    };
  }

  /// Get memory utilization percentage
  double get memoryUtilization {
    if (memoryTotalMb <= 0) return 0.0;
    final usedMemory = memoryTotalMb - memoryAvailableMb;
    return (usedMemory / memoryTotalMb) * 100.0;
  }

  /// Check if device is low-end
  bool get isLowEndDevice {
    return memoryTotalMb < 2048 || cpuCores <= 2;
  }

  /// Check if device is high-end
  bool get isHighEndDevice {
    return memoryTotalMb >= 8192 && cpuCores >= 8 && (gpuAvailable || neuralEngineAvailable);
  }

  /// Get device tier
  DeviceTier get deviceTier {
    if (isHighEndDevice) return DeviceTier.high;
    if (isLowEndDevice) return DeviceTier.low;
    return DeviceTier.medium;
  }

  /// Get recommended configuration based on device capabilities
  Map<String, dynamic> getRecommendedConfig() {
    final config = <String, dynamic>{};

    // Memory optimization based on available memory
    if (memoryAvailableMb < 512) {
      config['memory_optimization'] = 'maximum';
      config['max_memory_mb'] = 256;
    } else if (memoryAvailableMb < 1024) {
      config['memory_optimization'] = 'balanced';
      config['max_memory_mb'] = 512;
    } else {
      config['memory_optimization'] = 'minimal';
      config['max_memory_mb'] = 1024;
    }

    // Backend selection based on platform and capabilities
    if (platform.toLowerCase() == 'ios' && neuralEngineAvailable) {
      config['backend'] = 'coreml';
    } else if (platform.toLowerCase() == 'android' && neuralEngineAvailable) {
      config['backend'] = 'nnapi';
    } else if (gpuAvailable) {
      config['backend'] = 'gpu';
    } else {
      config['backend'] = 'cpu';
    }

    // Thread count based on CPU cores
    config['num_threads'] = (cpuCores / 2).ceil().clamp(1, 8);

    // Quantization based on device tier
    switch (deviceTier) {
      case DeviceTier.low:
        config['quantization'] = {
          'scheme': 'int4',
          'dynamic': true,
          'per_channel': true,
        };
        break;
      case DeviceTier.medium:
        config['quantization'] = {
          'scheme': 'int8',
          'dynamic': true,
          'per_channel': false,
        };
        break;
      case DeviceTier.high:
        config['quantization'] = {
          'scheme': 'fp16',
          'dynamic': false,
          'per_channel': true,
        };
        break;
    }

    // Batching configuration
    config['enable_batching'] = deviceTier != DeviceTier.low;
    config['max_batch_size'] = switch (deviceTier) {
      DeviceTier.low => 1,
      DeviceTier.medium => 2,
      DeviceTier.high => 4,
    };

    return config;
  }

  /// Get device capability score (0-100)
  int get capabilityScore {
    int score = 0;

    // Memory score (0-30)
    score += (memoryTotalMb / 1024 * 10).clamp(0, 30).round();

    // CPU score (0-20)
    score += (cpuCores * 2.5).clamp(0, 20).round();

    // GPU bonus (0-25)
    if (gpuAvailable) score += 25;

    // Neural Engine bonus (0-25)
    if (neuralEngineAvailable) score += 25;

    return score.clamp(0, 100);
  }

  /// Get estimated inference performance
  InferencePerformanceEstimate get performanceEstimate {
    final score = capabilityScore;
    
    if (score >= 80) {
      return InferencePerformanceEstimate.excellent;
    } else if (score >= 60) {
      return InferencePerformanceEstimate.good;
    } else if (score >= 40) {
      return InferencePerformanceEstimate.fair;
    } else {
      return InferencePerformanceEstimate.poor;
    }
  }

  @override
  String toString() {
    return 'TrustformersDeviceInfo('
        'platform: $platform, '
        'model: $model, '
        'memory: ${memoryAvailableMb}/${memoryTotalMb}MB, '
        'cpuCores: $cpuCores, '
        'gpu: $gpuAvailable, '
        'neuralEngine: $neuralEngineAvailable, '
        'tier: $deviceTier, '
        'score: $capabilityScore)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is TrustformersDeviceInfo &&
        other.platform == platform &&
        other.model == model &&
        other.memoryTotalMb == memoryTotalMb &&
        other.memoryAvailableMb == memoryAvailableMb &&
        other.cpuCores == cpuCores &&
        other.gpuAvailable == gpuAvailable &&
        other.neuralEngineAvailable == neuralEngineAvailable;
  }

  @override
  int get hashCode {
    return Object.hash(
      platform,
      model,
      memoryTotalMb,
      memoryAvailableMb,
      cpuCores,
      gpuAvailable,
      neuralEngineAvailable,
    );
  }
}

/// Device performance tiers
enum DeviceTier {
  low,
  medium,
  high;

  @override
  String toString() {
    switch (this) {
      case DeviceTier.low:
        return 'low';
      case DeviceTier.medium:
        return 'medium';
      case DeviceTier.high:
        return 'high';
    }
  }
}

/// Estimated inference performance levels
enum InferencePerformanceEstimate {
  poor,
  fair,
  good,
  excellent;

  @override
  String toString() {
    switch (this) {
      case InferencePerformanceEstimate.poor:
        return 'poor';
      case InferencePerformanceEstimate.fair:
        return 'fair';
      case InferencePerformanceEstimate.good:
        return 'good';
      case InferencePerformanceEstimate.excellent:
        return 'excellent';
    }
  }

  /// Get performance description
  String get description {
    switch (this) {
      case InferencePerformanceEstimate.poor:
        return 'Poor performance expected. Consider using lighter models or more aggressive optimization.';
      case InferencePerformanceEstimate.fair:
        return 'Fair performance expected. Moderate optimization recommended.';
      case InferencePerformanceEstimate.good:
        return 'Good performance expected. Standard optimization should work well.';
      case InferencePerformanceEstimate.excellent:
        return 'Excellent performance expected. Can use larger models with minimal optimization.';
    }
  }

  /// Get color representation for UI
  String get colorHex {
    switch (this) {
      case InferencePerformanceEstimate.poor:
        return '#FF5252'; // Red
      case InferencePerformanceEstimate.fair:
        return '#FF9800'; // Orange
      case InferencePerformanceEstimate.good:
        return '#4CAF50'; // Green
      case InferencePerformanceEstimate.excellent:
        return '#2196F3'; // Blue
    }
  }
}