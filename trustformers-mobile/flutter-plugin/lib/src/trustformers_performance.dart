/// Performance metrics for TrustformeRS inference engine
class TrustformersPerformanceMetrics {
  final String engineId;
  final int totalInferences;
  final double avgInferenceTimeMs;
  final int peakMemoryMb;
  final int currentMemoryMb;
  final double throughputTokensPerSec;
  final DateTime timestamp;
  final Map<String, dynamic>? additionalMetrics;

  const TrustformersPerformanceMetrics({
    required this.engineId,
    required this.totalInferences,
    required this.avgInferenceTimeMs,
    required this.peakMemoryMb,
    required this.currentMemoryMb,
    required this.throughputTokensPerSec,
    required this.timestamp,
    this.additionalMetrics,
  });

  factory TrustformersPerformanceMetrics.fromJson(Map<String, dynamic> json) {
    return TrustformersPerformanceMetrics(
      engineId: json['engine_id'] ?? 'unknown',
      totalInferences: json['total_inferences'] ?? 0,
      avgInferenceTimeMs: (json['avg_inference_time_ms'] ?? 0.0).toDouble(),
      peakMemoryMb: json['peak_memory_mb'] ?? 0,
      currentMemoryMb: json['current_memory_mb'] ?? 0,
      throughputTokensPerSec: (json['throughput_tokens_per_sec'] ?? 0.0).toDouble(),
      timestamp: json['timestamp'] != null
          ? DateTime.fromMillisecondsSinceEpoch(json['timestamp'])
          : DateTime.now(),
      additionalMetrics: json['additional_metrics'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'engine_id': engineId,
      'total_inferences': totalInferences,
      'avg_inference_time_ms': avgInferenceTimeMs,
      'peak_memory_mb': peakMemoryMb,
      'current_memory_mb': currentMemoryMb,
      'throughput_tokens_per_sec': throughputTokensPerSec,
      'timestamp': timestamp.millisecondsSinceEpoch,
      if (additionalMetrics != null) 'additional_metrics': additionalMetrics,
    };
  }

  /// Get memory utilization percentage
  double get memoryUtilization {
    if (peakMemoryMb <= 0) return 0.0;
    return (currentMemoryMb / peakMemoryMb) * 100.0;
  }

  /// Get performance grade
  PerformanceGrade get performanceGrade {
    // Grade based on throughput and inference time
    if (throughputTokensPerSec >= 50 && avgInferenceTimeMs <= 50) {
      return PerformanceGrade.excellent;
    } else if (throughputTokensPerSec >= 20 && avgInferenceTimeMs <= 100) {
      return PerformanceGrade.good;
    } else if (throughputTokensPerSec >= 10 && avgInferenceTimeMs <= 200) {
      return PerformanceGrade.fair;
    } else {
      return PerformanceGrade.poor;
    }
  }

  /// Get estimated battery impact
  BatteryImpact get estimatedBatteryImpact {
    // Estimate based on memory usage and inference frequency
    final memoryFactor = currentMemoryMb / 1024.0; // GB
    final frequencyFactor = totalInferences / (DateTime.now().difference(timestamp).inMinutes + 1);
    final combinedFactor = memoryFactor * frequencyFactor;

    if (combinedFactor >= 2.0) {
      return BatteryImpact.high;
    } else if (combinedFactor >= 1.0) {
      return BatteryImpact.medium;
    } else {
      return BatteryImpact.low;
    }
  }

  /// Get performance summary
  PerformanceSummary get summary {
    return PerformanceSummary(
      grade: performanceGrade,
      batteryImpact: estimatedBatteryImpact,
      avgLatency: avgInferenceTimeMs,
      throughput: throughputTokensPerSec,
      memoryUsage: currentMemoryMb,
      inferencesCompleted: totalInferences,
    );
  }

  /// Compare with another metrics instance
  PerformanceComparison compareTo(TrustformersPerformanceMetrics other) {
    return PerformanceComparison(
      current: this,
      baseline: other,
      throughputImprovement: (throughputTokensPerSec - other.throughputTokensPerSec) / other.throughputTokensPerSec * 100,
      latencyImprovement: (other.avgInferenceTimeMs - avgInferenceTimeMs) / other.avgInferenceTimeMs * 100,
      memoryChange: currentMemoryMb - other.currentMemoryMb,
    );
  }

  /// Get performance recommendations
  List<PerformanceRecommendation> getRecommendations() {
    final recommendations = <PerformanceRecommendation>[];

    // Memory-based recommendations
    if (currentMemoryMb > 1024) {
      recommendations.add(PerformanceRecommendation(
        type: RecommendationType.memory,
        priority: RecommendationPriority.high,
        title: 'High Memory Usage',
        description: 'Consider enabling more aggressive quantization or reducing batch size.',
        action: 'Optimize memory configuration',
      ));
    }

    // Throughput-based recommendations
    if (throughputTokensPerSec < 10) {
      recommendations.add(PerformanceRecommendation(
        type: RecommendationType.performance,
        priority: RecommendationPriority.medium,
        title: 'Low Throughput',
        description: 'Consider using GPU acceleration or optimizing model configuration.',
        action: 'Enable hardware acceleration',
      ));
    }

    // Latency-based recommendations
    if (avgInferenceTimeMs > 500) {
      recommendations.add(PerformanceRecommendation(
        type: RecommendationType.latency,
        priority: RecommendationPriority.high,
        title: 'High Latency',
        description: 'Inference time is high. Consider model optimization or hardware upgrade.',
        action: 'Optimize inference pipeline',
      ));
    }

    // Battery-based recommendations
    if (estimatedBatteryImpact == BatteryImpact.high) {
      recommendations.add(PerformanceRecommendation(
        type: RecommendationType.battery,
        priority: RecommendationPriority.medium,
        title: 'High Battery Impact',
        description: 'Current settings may impact battery life significantly.',
        action: 'Enable power-saving mode',
      ));
    }

    return recommendations;
  }

  @override
  String toString() {
    return 'TrustformersPerformanceMetrics('
        'engineId: $engineId, '
        'inferences: $totalInferences, '
        'avgTime: ${avgInferenceTimeMs.toStringAsFixed(2)}ms, '
        'throughput: ${throughputTokensPerSec.toStringAsFixed(2)} tokens/s, '
        'memory: ${currentMemoryMb}MB, '
        'grade: $performanceGrade)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is TrustformersPerformanceMetrics &&
        other.engineId == engineId &&
        other.totalInferences == totalInferences &&
        other.avgInferenceTimeMs == avgInferenceTimeMs &&
        other.peakMemoryMb == peakMemoryMb &&
        other.currentMemoryMb == currentMemoryMb &&
        other.throughputTokensPerSec == throughputTokensPerSec;
  }

  @override
  int get hashCode {
    return Object.hash(
      engineId,
      totalInferences,
      avgInferenceTimeMs,
      peakMemoryMb,
      currentMemoryMb,
      throughputTokensPerSec,
    );
  }
}

/// Performance grade enumeration
enum PerformanceGrade {
  poor,
  fair,
  good,
  excellent;

  @override
  String toString() {
    switch (this) {
      case PerformanceGrade.poor:
        return 'poor';
      case PerformanceGrade.fair:
        return 'fair';
      case PerformanceGrade.good:
        return 'good';
      case PerformanceGrade.excellent:
        return 'excellent';
    }
  }

  String get description {
    switch (this) {
      case PerformanceGrade.poor:
        return 'Performance is below acceptable levels';
      case PerformanceGrade.fair:
        return 'Performance is acceptable but could be improved';
      case PerformanceGrade.good:
        return 'Performance is good for most use cases';
      case PerformanceGrade.excellent:
        return 'Performance is excellent';
    }
  }

  String get colorHex {
    switch (this) {
      case PerformanceGrade.poor:
        return '#F44336'; // Red
      case PerformanceGrade.fair:
        return '#FF9800'; // Orange
      case PerformanceGrade.good:
        return '#4CAF50'; // Green
      case PerformanceGrade.excellent:
        return '#2196F3'; // Blue
    }
  }
}

/// Battery impact levels
enum BatteryImpact {
  low,
  medium,
  high;

  @override
  String toString() {
    switch (this) {
      case BatteryImpact.low:
        return 'low';
      case BatteryImpact.medium:
        return 'medium';
      case BatteryImpact.high:
        return 'high';
    }
  }

  String get description {
    switch (this) {
      case BatteryImpact.low:
        return 'Minimal impact on battery life';
      case BatteryImpact.medium:
        return 'Moderate impact on battery life';
      case BatteryImpact.high:
        return 'Significant impact on battery life';
    }
  }
}

/// Performance summary
class PerformanceSummary {
  final PerformanceGrade grade;
  final BatteryImpact batteryImpact;
  final double avgLatency;
  final double throughput;
  final int memoryUsage;
  final int inferencesCompleted;

  const PerformanceSummary({
    required this.grade,
    required this.batteryImpact,
    required this.avgLatency,
    required this.throughput,
    required this.memoryUsage,
    required this.inferencesCompleted,
  });

  Map<String, dynamic> toJson() {
    return {
      'grade': grade.toString(),
      'battery_impact': batteryImpact.toString(),
      'avg_latency': avgLatency,
      'throughput': throughput,
      'memory_usage': memoryUsage,
      'inferences_completed': inferencesCompleted,
    };
  }
}

/// Performance comparison between two metrics
class PerformanceComparison {
  final TrustformersPerformanceMetrics current;
  final TrustformersPerformanceMetrics baseline;
  final double throughputImprovement;
  final double latencyImprovement;
  final int memoryChange;

  const PerformanceComparison({
    required this.current,
    required this.baseline,
    required this.throughputImprovement,
    required this.latencyImprovement,
    required this.memoryChange,
  });

  bool get isBetter {
    return throughputImprovement > 0 && latencyImprovement > 0 && memoryChange <= 0;
  }

  bool get isWorse {
    return throughputImprovement < -10 || latencyImprovement < -10 || memoryChange > 100;
  }

  String get summary {
    if (isBetter) {
      return 'Performance improved';
    } else if (isWorse) {
      return 'Performance degraded';
    } else {
      return 'Performance similar';
    }
  }
}

/// Performance recommendation
class PerformanceRecommendation {
  final RecommendationType type;
  final RecommendationPriority priority;
  final String title;
  final String description;
  final String action;

  const PerformanceRecommendation({
    required this.type,
    required this.priority,
    required this.title,
    required this.description,
    required this.action,
  });

  Map<String, dynamic> toJson() {
    return {
      'type': type.toString(),
      'priority': priority.toString(),
      'title': title,
      'description': description,
      'action': action,
    };
  }
}

/// Recommendation types
enum RecommendationType {
  memory,
  performance,
  latency,
  battery;

  @override
  String toString() {
    switch (this) {
      case RecommendationType.memory:
        return 'memory';
      case RecommendationType.performance:
        return 'performance';
      case RecommendationType.latency:
        return 'latency';
      case RecommendationType.battery:
        return 'battery';
    }
  }
}

/// Recommendation priorities
enum RecommendationPriority {
  low,
  medium,
  high;

  @override
  String toString() {
    switch (this) {
      case RecommendationPriority.low:
        return 'low';
      case RecommendationPriority.medium:
        return 'medium';
      case RecommendationPriority.high:
        return 'high';
    }
  }
}