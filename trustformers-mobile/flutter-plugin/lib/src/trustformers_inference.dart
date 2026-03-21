import 'trustformers_types.dart';

/// Inference request for TrustformeRS engine
class TrustformersInferenceRequest {
  final TrustformersInput input;
  final TrustformersGenerationConfig? generationConfig;

  const TrustformersInferenceRequest({
    required this.input,
    this.generationConfig,
  });

  Map<String, dynamic> toJson() {
    return {
      ...input.toJson(),
      if (generationConfig != null) ...generationConfig!.toJson(),
    };
  }

  factory TrustformersInferenceRequest.fromJson(Map<String, dynamic> json) {
    return TrustformersInferenceRequest(
      input: TrustformersInput.fromJson(json),
      generationConfig: json.containsKey('max_length') ||
              json.containsKey('temperature') ||
              json.containsKey('top_p') ||
              json.containsKey('top_k') ||
              json.containsKey('do_sample')
          ? TrustformersGenerationConfig(
              maxLength: json['max_length'],
              temperature: json['temperature']?.toDouble(),
              topP: json['top_p']?.toDouble(),
              topK: json['top_k'],
              doSample: json['do_sample'] ?? false,
            )
          : null,
    );
  }

  /// Create a simple text generation request
  factory TrustformersInferenceRequest.textGeneration({
    required List<int> inputIds,
    int maxLength = 50,
    double temperature = 1.0,
    double topP = 0.9,
    int topK = 50,
    bool doSample = true,
  }) {
    return TrustformersInferenceRequest(
      input: TrustformersInput(inputIds: inputIds),
      generationConfig: TrustformersGenerationConfig(
        maxLength: maxLength,
        temperature: temperature,
        topP: topP,
        topK: topK,
        doSample: doSample,
      ),
    );
  }

  /// Create a classification request
  factory TrustformersInferenceRequest.classification({
    required List<int> inputIds,
    List<int>? attentionMask,
    List<int>? tokenTypeIds,
  }) {
    return TrustformersInferenceRequest(
      input: TrustformersInput(
        inputIds: inputIds,
        attentionMask: attentionMask,
        tokenTypeIds: tokenTypeIds,
      ),
      generationConfig: const TrustformersGenerationConfig(
        maxLength: 1,
        doSample: false,
      ),
    );
  }

  /// Create a sequence-to-sequence request
  factory TrustformersInferenceRequest.seq2seq({
    required List<int> inputIds,
    List<int>? attentionMask,
    int maxLength = 100,
    double temperature = 0.8,
    bool doSample = true,
  }) {
    return TrustformersInferenceRequest(
      input: TrustformersInput(
        inputIds: inputIds,
        attentionMask: attentionMask,
      ),
      generationConfig: TrustformersGenerationConfig(
        maxLength: maxLength,
        temperature: temperature,
        doSample: doSample,
      ),
    );
  }

  @override
  String toString() {
    return 'TrustformersInferenceRequest('
        'inputIds: ${input.inputIds.length} tokens, '
        'generationConfig: $generationConfig)';
  }
}

/// Inference result from TrustformeRS engine
class TrustformersInferenceResult {
  final List<int> tokens;
  final List<double>? logits;
  final List<List<double>>? attentionWeights;
  final double inferenceTimeMs;
  final int memoryUsageMb;
  final bool isError;
  final String? errorMessage;

  const TrustformersInferenceResult({
    required this.tokens,
    this.logits,
    this.attentionWeights,
    required this.inferenceTimeMs,
    required this.memoryUsageMb,
    this.isError = false,
    this.errorMessage,
  });

  /// Create an error result
  factory TrustformersInferenceResult.error(String message) {
    return TrustformersInferenceResult(
      tokens: [],
      inferenceTimeMs: 0.0,
      memoryUsageMb: 0,
      isError: true,
      errorMessage: message,
    );
  }

  factory TrustformersInferenceResult.fromJson(Map<String, dynamic> json) {
    return TrustformersInferenceResult(
      tokens: List<int>.from(json['tokens'] ?? []),
      logits: json['logits'] != null
          ? List<double>.from(json['logits'])
          : null,
      attentionWeights: json['attention_weights'] != null
          ? (json['attention_weights'] as List)
              .map((layer) => List<double>.from(layer))
              .toList()
          : null,
      inferenceTimeMs: (json['inference_time_ms'] ?? 0.0).toDouble(),
      memoryUsageMb: json['memory_usage_mb'] ?? 0,
      isError: json['is_error'] ?? false,
      errorMessage: json['error_message'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'tokens': tokens,
      if (logits != null) 'logits': logits,
      if (attentionWeights != null) 'attention_weights': attentionWeights,
      'inference_time_ms': inferenceTimeMs,
      'memory_usage_mb': memoryUsageMb,
      'is_error': isError,
      if (errorMessage != null) 'error_message': errorMessage,
    };
  }

  /// Get the generated text (requires tokenizer for decoding)
  String getGeneratedText([Function(List<int>)? decoder]) {
    if (isError) return errorMessage ?? 'Error occurred during inference';
    if (decoder != null) {
      return decoder(tokens).toString();
    }
    return tokens.join(' '); // Fallback representation
  }

  /// Get the most likely token IDs
  List<int> getMostLikelyTokens() {
    if (isError || tokens.isEmpty) return [];
    return tokens;
  }

  /// Get inference performance metrics
  Map<String, dynamic> getPerformanceMetrics() {
    return {
      'inference_time_ms': inferenceTimeMs,
      'memory_usage_mb': memoryUsageMb,
      'tokens_per_second': tokens.isNotEmpty
          ? (tokens.length / (inferenceTimeMs / 1000.0))
          : 0.0,
      'tokens_generated': tokens.length,
    };
  }

  /// Check if inference was successful
  bool get isSuccess => !isError;

  /// Get tokens per second throughput
  double get tokensPerSecond {
    if (inferenceTimeMs <= 0 || tokens.isEmpty) return 0.0;
    return tokens.length / (inferenceTimeMs / 1000.0);
  }

  @override
  String toString() {
    if (isError) {
      return 'TrustformersInferenceResult.error($errorMessage)';
    }
    return 'TrustformersInferenceResult('
        'tokens: ${tokens.length}, '
        'time: ${inferenceTimeMs.toStringAsFixed(2)}ms, '
        'memory: ${memoryUsageMb}MB, '
        'throughput: ${tokensPerSecond.toStringAsFixed(2)} tokens/s)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is TrustformersInferenceResult &&
        other.tokens == tokens &&
        other.logits == logits &&
        other.attentionWeights == attentionWeights &&
        other.inferenceTimeMs == inferenceTimeMs &&
        other.memoryUsageMb == memoryUsageMb &&
        other.isError == isError &&
        other.errorMessage == errorMessage;
  }

  @override
  int get hashCode {
    return Object.hash(
      tokens,
      logits,
      attentionWeights,
      inferenceTimeMs,
      memoryUsageMb,
      isError,
      errorMessage,
    );
  }
}

/// Batch inference request
class TrustformersBatchInferenceRequest {
  final List<TrustformersInferenceRequest> requests;
  final bool parallel;

  const TrustformersBatchInferenceRequest({
    required this.requests,
    this.parallel = true,
  });

  Map<String, dynamic> toJson() {
    return {
      'requests': requests.map((req) => req.toJson()).toList(),
      'parallel': parallel,
    };
  }

  factory TrustformersBatchInferenceRequest.fromJson(Map<String, dynamic> json) {
    return TrustformersBatchInferenceRequest(
      requests: (json['requests'] as List)
          .map((req) => TrustformersInferenceRequest.fromJson(req))
          .toList(),
      parallel: json['parallel'] ?? true,
    );
  }

  int get requestCount => requests.length;
}

/// Batch inference result
class TrustformersBatchInferenceResult {
  final List<TrustformersInferenceResult> results;
  final double totalTimeMs;
  final int totalMemoryMb;
  final int successCount;
  final int errorCount;

  const TrustformersBatchInferenceResult({
    required this.results,
    required this.totalTimeMs,
    required this.totalMemoryMb,
    required this.successCount,
    required this.errorCount,
  });

  factory TrustformersBatchInferenceResult.fromResults(
    List<TrustformersInferenceResult> results,
  ) {
    final totalTime = results.fold(0.0, (sum, result) => sum + result.inferenceTimeMs);
    final totalMemory = results.fold(0, (sum, result) => sum + result.memoryUsageMb);
    final successCount = results.where((result) => result.isSuccess).length;
    final errorCount = results.where((result) => result.isError).length;

    return TrustformersBatchInferenceResult(
      results: results,
      totalTimeMs: totalTime,
      totalMemoryMb: totalMemory,
      successCount: successCount,
      errorCount: errorCount,
    );
  }

  /// Get average inference time per request
  double get averageTimeMs => results.isNotEmpty ? totalTimeMs / results.length : 0.0;

  /// Get average memory usage per request
  double get averageMemoryMb => results.isNotEmpty ? totalMemoryMb / results.length : 0.0;

  /// Get batch success rate
  double get successRate => results.isNotEmpty ? successCount / results.length : 0.0;

  /// Get total tokens generated
  int get totalTokens => results.fold(0, (sum, result) => sum + result.tokens.length);

  /// Get batch throughput
  double get batchThroughput => totalTimeMs > 0 ? (totalTokens / (totalTimeMs / 1000.0)) : 0.0;

  @override
  String toString() {
    return 'TrustformersBatchInferenceResult('
        'requests: ${results.length}, '
        'success: $successCount, '
        'errors: $errorCount, '
        'avg_time: ${averageTimeMs.toStringAsFixed(2)}ms, '
        'throughput: ${batchThroughput.toStringAsFixed(2)} tokens/s)';
  }
}