import 'dart:async';
import 'dart:ffi';
import 'dart:isolate';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';

import 'trustformers_config.dart';
import 'trustformers_types.dart';
import 'trustformers_inference.dart';
import 'trustformers_device_info.dart';
import 'trustformers_performance.dart';
import 'trustformers_exceptions.dart';

/// TrustformeRS Flutter inference engine
class TrustformersEngine {
  static const MethodChannel _channel = MethodChannel('trustformers_flutter');
  static const EventChannel _eventChannel = EventChannel('trustformers_flutter_events');

  final String engineId;
  final TrustformersConfig config;
  
  bool _isInitialized = false;
  bool _isModelLoaded = false;
  StreamSubscription<dynamic>? _eventSubscription;
  final StreamController<TrustformersEvent> _eventController = StreamController<TrustformersEvent>.broadcast();

  TrustformersEngine._(this.engineId, this.config);

  /// Create a new TrustformeRS engine instance
  static Future<TrustformersEngine> create(TrustformersConfig config) async {
    final engine = TrustformersEngine._(config.engineId, config);
    await engine._initialize();
    return engine;
  }

  /// Initialize the engine
  Future<void> _initialize() async {
    if (_isInitialized) return;

    try {
      final result = await _channel.invokeMethod('initialize', config.toJson());
      if (result['status'] == 'initialized') {
        _isInitialized = true;
        _setupEventStream();
      } else {
        throw TrustformersException(
          'Initialization failed',
          code: 'INITIALIZATION_FAILED',
        );
      }
    } on PlatformException catch (e) {
      throw TrustformersException(
        'Platform initialization failed: ${e.message}',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Set up event stream for real-time updates
  void _setupEventStream() {
    _eventSubscription = _eventChannel.receiveBroadcastStream().listen(
      (dynamic event) {
        try {
          final Map<String, dynamic> eventData = Map<String, dynamic>.from(event);
          final trustformersEvent = TrustformersEvent.fromJson(eventData);
          _eventController.add(trustformersEvent);
        } catch (e) {
          // Handle malformed events gracefully
          print('Error parsing event: $e');
        }
      },
      onError: (error) {
        _eventController.addError(TrustformersException(
          'Event stream error: $error',
          code: 'EVENT_STREAM_ERROR',
        ));
      },
    );
  }

  /// Load a model from the specified path
  Future<void> loadModel(String modelPath) async {
    if (!_isInitialized) {
      throw TrustformersException(
        'Engine not initialized',
        code: 'ENGINE_NOT_INITIALIZED',
      );
    }

    try {
      final result = await _channel.invokeMethod('loadModel', {
        'engine_id': engineId,
        'model_path': modelPath,
      });

      if (result['status'] == 'model_loaded') {
        _isModelLoaded = true;
      } else {
        throw TrustformersException(
          'Model loading failed',
          code: 'MODEL_LOAD_FAILED',
        );
      }
    } on PlatformException catch (e) {
      throw TrustformersException(
        'Platform model loading failed: ${e.message}',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Perform inference with the loaded model
  Future<TrustformersInferenceResult> inference(
    TrustformersInferenceRequest request,
  ) async {
    if (!_isModelLoaded) {
      throw TrustformersException(
        'Model not loaded',
        code: 'MODEL_NOT_LOADED',
      );
    }

    try {
      final result = await _channel.invokeMethod('inference', {
        'engine_id': engineId,
        ...request.toJson(),
      });

      return TrustformersInferenceResult.fromJson(result);
    } on PlatformException catch (e) {
      throw TrustformersException(
        'Inference failed: ${e.message}',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Perform batch inference
  Future<List<TrustformersInferenceResult>> batchInference(
    List<TrustformersInferenceRequest> requests,
  ) async {
    if (!_isModelLoaded) {
      throw TrustformersException(
        'Model not loaded',
        code: 'MODEL_NOT_LOADED',
      );
    }

    try {
      final result = await _channel.invokeMethod('getBatchInference', {
        'engine_id': engineId,
        'requests': requests.map((req) => req.toJson()).toList(),
      });

      return (result as List)
          .map((json) => TrustformersInferenceResult.fromJson(json))
          .toList();
    } on PlatformException catch (e) {
      throw TrustformersException(
        'Batch inference failed: ${e.message}',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Stream-based inference for real-time processing
  Stream<TrustformersInferenceResult> inferenceStream(
    Stream<TrustformersInferenceRequest> requests,
  ) async* {
    await for (final request in requests) {
      try {
        final result = await inference(request);
        yield result;
      } catch (e) {
        // Handle individual inference errors
        yield TrustformersInferenceResult.error(e.toString());
      }
    }
  }

  /// Get device information
  Future<TrustformersDeviceInfo> getDeviceInfo() async {
    try {
      final result = await _channel.invokeMethod('getDeviceInfo');
      return TrustformersDeviceInfo.fromJson(result);
    } on PlatformException catch (e) {
      throw TrustformersException(
        'Failed to get device info: ${e.message}',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Get performance metrics
  Future<TrustformersPerformanceMetrics> getPerformanceMetrics() async {
    if (!_isInitialized) {
      throw TrustformersException(
        'Engine not initialized',
        code: 'ENGINE_NOT_INITIALIZED',
      );
    }

    try {
      final result = await _channel.invokeMethod('getPerformanceMetrics', engineId);
      return TrustformersPerformanceMetrics.fromJson(result);
    } on PlatformException catch (e) {
      throw TrustformersException(
        'Failed to get performance metrics: ${e.message}',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Get model information
  Future<Map<String, dynamic>> getModelInfo() async {
    if (!_isModelLoaded) {
      throw TrustformersException(
        'Model not loaded',
        code: 'MODEL_NOT_LOADED',
      );
    }

    try {
      final result = await _channel.invokeMethod('getModelInfo', engineId);
      return Map<String, dynamic>.from(result);
    } on PlatformException catch (e) {
      throw TrustformersException(
        'Failed to get model info: ${e.message}',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Optimize configuration for current device
  Future<TrustformersConfig> optimizeForDevice() async {
    try {
      final result = await _channel.invokeMethod('optimizeForDevice', {
        'engine_id': engineId,
        'current_config': config.toJson(),
      });
      return TrustformersConfig.fromJson(result);
    } on PlatformException catch (e) {
      throw TrustformersException(
        'Failed to optimize for device: ${e.message}',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Run inference in background isolate
  Future<TrustformersInferenceResult> inferenceInBackground(
    TrustformersInferenceRequest request,
  ) async {
    final receivePort = ReceivePort();
    final isolate = await Isolate.spawn(
      _backgroundInference,
      [receivePort.sendPort, engineId, request.toJson()],
    );

    final result = await receivePort.first;
    isolate.kill(priority: Isolate.immediate);

    if (result is Map<String, dynamic>) {
      return TrustformersInferenceResult.fromJson(result);
    } else {
      throw TrustformersException(
        'Background inference failed: $result',
        code: 'BACKGROUND_INFERENCE_FAILED',
      );
    }
  }

  /// Background inference isolate function
  static void _backgroundInference(List<dynamic> args) async {
    final SendPort sendPort = args[0];
    final String engineId = args[1];
    final Map<String, dynamic> requestJson = args[2];

    try {
      final result = await _channel.invokeMethod('inference', {
        'engine_id': engineId,
        ...requestJson,
      });
      sendPort.send(result);
    } catch (e) {
      sendPort.send(e.toString());
    }
  }

  /// Warm up the model with sample input
  Future<void> warmUp({List<int>? sampleInput}) async {
    sampleInput ??= List.filled(128, 1); // Default sample input

    final warmUpRequest = TrustformersInferenceRequest(
      input: TrustformersInput(inputIds: sampleInput),
      generationConfig: const TrustformersGenerationConfig(
        maxLength: 1,
        doSample: false,
      ),
    );

    try {
      await inference(warmUpRequest);
    } catch (e) {
      // Warm-up failures are non-critical
      print('Warm-up inference failed: $e');
    }
  }

  /// Reset performance statistics
  Future<void> resetStatistics() async {
    try {
      await _channel.invokeMethod('resetStatistics', engineId);
    } on PlatformException catch (e) {
      throw TrustformersException(
        'Failed to reset statistics: ${e.message}',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Export performance data
  Future<Map<String, dynamic>> exportPerformanceData() async {
    try {
      final result = await _channel.invokeMethod('exportPerformanceData', engineId);
      return Map<String, dynamic>.from(result);
    } on PlatformException catch (e) {
      throw TrustformersException(
        'Failed to export performance data: ${e.message}',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Get event stream for real-time updates
  Stream<TrustformersEvent> get eventStream => _eventController.stream;

  /// Check if engine is initialized
  bool get isInitialized => _isInitialized;

  /// Check if model is loaded
  bool get isModelLoaded => _isModelLoaded;

  /// Get engine configuration
  TrustformersConfig get configuration => config;

  /// Dispose the engine and free resources
  Future<void> dispose() async {
    if (!_isInitialized) return;

    try {
      await _channel.invokeMethod('dispose', engineId);
      await _eventSubscription?.cancel();
      await _eventController.close();
      _isInitialized = false;
      _isModelLoaded = false;
    } on PlatformException catch (e) {
      throw TrustformersException(
        'Failed to dispose engine: ${e.message}',
        code: e.code,
        details: e.details,
      );
    }
  }
}

/// Singleton manager for multiple TrustformeRS engines
class TrustformersEngineManager {
  static final TrustformersEngineManager _instance = TrustformersEngineManager._internal();
  factory TrustformersEngineManager() => _instance;
  TrustformersEngineManager._internal();

  final Map<String, TrustformersEngine> _engines = {};

  /// Create or get existing engine
  Future<TrustformersEngine> getEngine(TrustformersConfig config) async {
    if (_engines.containsKey(config.engineId)) {
      return _engines[config.engineId]!;
    }

    final engine = await TrustformersEngine.create(config);
    _engines[config.engineId] = engine;
    return engine;
  }

  /// Remove engine
  Future<void> removeEngine(String engineId) async {
    final engine = _engines.remove(engineId);
    if (engine != null) {
      await engine.dispose();
    }
  }

  /// Get all engine IDs
  List<String> get engineIds => _engines.keys.toList();

  /// Get engine by ID
  TrustformersEngine? getEngineById(String engineId) {
    return _engines[engineId];
  }

  /// Dispose all engines
  Future<void> disposeAll() async {
    final engines = _engines.values.toList();
    _engines.clear();
    
    for (final engine in engines) {
      await engine.dispose();
    }
  }
}