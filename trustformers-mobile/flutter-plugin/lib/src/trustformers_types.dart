/// Core types and enums for TrustformeRS Flutter integration

/// Supported mobile platforms
enum TrustformersPlatform {
  ios,
  android,
  generic;

  @override
  String toString() {
    switch (this) {
      case TrustformersPlatform.ios:
        return 'ios';
      case TrustformersPlatform.android:
        return 'android';
      case TrustformersPlatform.generic:
        return 'generic';
    }
  }
}

/// Inference backends for mobile deployment
enum TrustformersBackend {
  cpu,
  coreml,
  nnapi,
  gpu,
  custom;

  @override
  String toString() {
    switch (this) {
      case TrustformersBackend.cpu:
        return 'cpu';
      case TrustformersBackend.coreml:
        return 'coreml';
      case TrustformersBackend.nnapi:
        return 'nnapi';
      case TrustformersBackend.gpu:
        return 'gpu';
      case TrustformersBackend.custom:
        return 'custom';
    }
  }
}

/// Memory optimization levels
enum TrustformersMemoryOptimization {
  minimal,
  balanced,
  maximum;

  @override
  String toString() {
    switch (this) {
      case TrustformersMemoryOptimization.minimal:
        return 'minimal';
      case TrustformersMemoryOptimization.balanced:
        return 'balanced';
      case TrustformersMemoryOptimization.maximum:
        return 'maximum';
    }
  }
}

/// Quantization schemes for mobile optimization
enum TrustformersQuantizationScheme {
  int8,
  int4,
  fp16,
  dynamic;

  @override
  String toString() {
    switch (this) {
      case TrustformersQuantizationScheme.int8:
        return 'int8';
      case TrustformersQuantizationScheme.int4:
        return 'int4';
      case TrustformersQuantizationScheme.fp16:
        return 'fp16';
      case TrustformersQuantizationScheme.dynamic:
        return 'dynamic';
    }
  }
}

/// Quantization configuration
class TrustformersQuantizationConfig {
  final TrustformersQuantizationScheme scheme;
  final bool dynamic;
  final bool perChannel;

  const TrustformersQuantizationConfig({
    required this.scheme,
    this.dynamic = true,
    this.perChannel = false,
  });

  Map<String, dynamic> toJson() {
    return {
      'scheme': scheme.toString(),
      'dynamic': dynamic,
      'per_channel': perChannel,
    };
  }

  factory TrustformersQuantizationConfig.fromJson(Map<String, dynamic> json) {
    return TrustformersQuantizationConfig(
      scheme: TrustformersQuantizationScheme.values.firstWhere(
        (e) => e.toString() == json['scheme'],
        orElse: () => TrustformersQuantizationScheme.dynamic,
      ),
      dynamic: json['dynamic'] ?? true,
      perChannel: json['per_channel'] ?? false,
    );
  }
}

/// Inference generation parameters
class TrustformersGenerationConfig {
  final int? maxLength;
  final double? temperature;
  final double? topP;
  final int? topK;
  final bool doSample;

  const TrustformersGenerationConfig({
    this.maxLength,
    this.temperature,
    this.topP,
    this.topK,
    this.doSample = false,
  });

  Map<String, dynamic> toJson() {
    return {
      if (maxLength != null) 'max_length': maxLength,
      if (temperature != null) 'temperature': temperature,
      if (topP != null) 'top_p': topP,
      if (topK != null) 'top_k': topK,
      'do_sample': doSample,
    };
  }
}

/// Event types for TrustformeRS streaming updates
enum TrustformersEventType {
  modelLoaded,
  inferenceCompleted,
  inferenceStarted,
  errorOccurred,
  performanceUpdate;

  static TrustformersEventType fromString(String value) {
    switch (value) {
      case 'model_loaded':
        return TrustformersEventType.modelLoaded;
      case 'inference_completed':
        return TrustformersEventType.inferenceCompleted;
      case 'inference_started':
        return TrustformersEventType.inferenceStarted;
      case 'error_occurred':
        return TrustformersEventType.errorOccurred;
      case 'performance_update':
        return TrustformersEventType.performanceUpdate;
      default:
        throw ArgumentError('Unknown event type: $value');
    }
  }
}

/// TrustformeRS event for streaming updates
class TrustformersEvent {
  final TrustformersEventType type;
  final String engineId;
  final Map<String, dynamic>? data;

  const TrustformersEvent({
    required this.type,
    required this.engineId,
    this.data,
  });

  factory TrustformersEvent.fromJson(Map<String, dynamic> json) {
    return TrustformersEvent(
      type: TrustformersEventType.fromString(json['type']),
      engineId: json['engine_id'],
      data: json,
    );
  }
}

/// Input data for tokenization and inference
class TrustformersInput {
  final List<int> inputIds;
  final List<int>? attentionMask;
  final List<int>? tokenTypeIds;

  const TrustformersInput({
    required this.inputIds,
    this.attentionMask,
    this.tokenTypeIds,
  });

  Map<String, dynamic> toJson() {
    return {
      'input_ids': inputIds,
      if (attentionMask != null) 'attention_mask': attentionMask,
      if (tokenTypeIds != null) 'token_type_ids': tokenTypeIds,
    };
  }

  factory TrustformersInput.fromJson(Map<String, dynamic> json) {
    return TrustformersInput(
      inputIds: List<int>.from(json['input_ids']),
      attentionMask: json['attention_mask'] != null
          ? List<int>.from(json['attention_mask'])
          : null,
      tokenTypeIds: json['token_type_ids'] != null
          ? List<int>.from(json['token_type_ids'])
          : null,
    );
  }
}

/// Batch input for multiple sequences
class TrustformersBatchInput {
  final List<TrustformersInput> inputs;
  final int? maxLength;
  final bool padToMaxLength;

  const TrustformersBatchInput({
    required this.inputs,
    this.maxLength,
    this.padToMaxLength = true,
  });

  Map<String, dynamic> toJson() {
    return {
      'inputs': inputs.map((input) => input.toJson()).toList(),
      if (maxLength != null) 'max_length': maxLength,
      'pad_to_max_length': padToMaxLength,
    };
  }
}