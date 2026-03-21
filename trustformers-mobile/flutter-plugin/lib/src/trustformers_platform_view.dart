import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'trustformers_performance.dart';
import 'trustformers_types.dart';

/// Platform view for TrustformeRS native UI components
class TrustformersPlatformView extends StatelessWidget {
  final TrustformersPlatformViewType viewType;
  final Map<String, dynamic> creationParams;
  final PlatformViewCreatedCallback? onPlatformViewCreated;
  final Set<Factory<OneSequenceGestureRecognizer>>? gestureRecognizers;

  const TrustformersPlatformView({
    Key? key,
    required this.viewType,
    this.creationParams = const {},
    this.onPlatformViewCreated,
    this.gestureRecognizers,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Platform-specific view creation
    if (Platform.isAndroid) {
      return PlatformViewLink(
        viewType: viewType.androidViewType,
        surfaceFactory: (context, controller) {
          return AndroidViewSurface(
            controller: controller as AndroidViewController,
            gestureRecognizers: gestureRecognizers ?? const {},
            hitTestBehavior: PlatformViewHitTestBehavior.opaque,
          );
        },
        onCreatePlatformView: (params) {
          return PlatformViewsService.initSurfaceAndroidView(
            id: params.id,
            viewType: viewType.androidViewType,
            layoutDirection: TextDirection.ltr,
            creationParams: creationParams,
            creationParamsCodec: const StandardMessageCodec(),
            onFocus: () {
              params.onFocusChanged(true);
            },
          )
            ..addOnPlatformViewCreatedListener(params.onPlatformViewCreated)
            ..addOnPlatformViewCreatedListener(onPlatformViewCreated ?? (_) {})
            ..create();
        },
      );
    } else if (Platform.isIOS) {
      return UiKitView(
        viewType: viewType.iosViewType,
        creationParams: creationParams,
        creationParamsCodec: const StandardMessageCodec(),
        onPlatformViewCreated: onPlatformViewCreated,
        gestureRecognizers: gestureRecognizers,
      );
    } else {
      return Container(
        color: Colors.grey[300],
        child: const Center(
          child: Text('Platform views not supported on this platform'),
        ),
      );
    }
  }
}

/// Performance chart platform view widget
class TrustformersPerformanceChart extends StatefulWidget {
  final String engineId;
  final TrustformersChartType chartType;
  final Duration updateInterval;
  final Map<String, dynamic> chartOptions;

  const TrustformersPerformanceChart({
    Key? key,
    required this.engineId,
    this.chartType = TrustformersChartType.realTimeMetrics,
    this.updateInterval = const Duration(seconds: 1),
    this.chartOptions = const {},
  }) : super(key: key);

  @override
  State<TrustformersPerformanceChart> createState() => _TrustformersPerformanceChartState();
}

class _TrustformersPerformanceChartState extends State<TrustformersPerformanceChart> {
  static const MethodChannel _channel = MethodChannel('trustformers_flutter/performance_chart');
  int? _viewId;

  @override
  Widget build(BuildContext context) {
    final creationParams = {
      'engine_id': widget.engineId,
      'chart_type': widget.chartType.name,
      'update_interval_ms': widget.updateInterval.inMilliseconds,
      'chart_options': widget.chartOptions,
    };

    return TrustformersPlatformView(
      viewType: TrustformersPlatformViewType.performanceChart,
      creationParams: creationParams,
      onPlatformViewCreated: (int id) {
        _viewId = id;
        _startUpdates();
      },
    );
  }

  void _startUpdates() {
    if (_viewId != null) {
      _channel.invokeMethod('startUpdates', {
        'view_id': _viewId,
        'engine_id': widget.engineId,
        'update_interval_ms': widget.updateInterval.inMilliseconds,
      });
    }
  }

  @override
  void dispose() {
    if (_viewId != null) {
      _channel.invokeMethod('stopUpdates', {'view_id': _viewId});
    }
    super.dispose();
  }
}

/// Model visualization platform view widget
class TrustformersModelVisualization extends StatefulWidget {
  final String engineId;
  final TrustformersVisualizationType visualizationType;
  final Map<String, dynamic> visualizationOptions;

  const TrustformersModelVisualization({
    Key? key,
    required this.engineId,
    this.visualizationType = TrustformersVisualizationType.modelArchitecture,
    this.visualizationOptions = const {},
  }) : super(key: key);

  @override
  State<TrustformersModelVisualization> createState() => _TrustformersModelVisualizationState();
}

class _TrustformersModelVisualizationState extends State<TrustformersModelVisualization> {
  static const MethodChannel _channel = MethodChannel('trustformers_flutter/model_visualization');
  int? _viewId;

  @override
  Widget build(BuildContext context) {
    final creationParams = {
      'engine_id': widget.engineId,
      'visualization_type': widget.visualizationType.name,
      'visualization_options': widget.visualizationOptions,
    };

    return TrustformersPlatformView(
      viewType: TrustformersPlatformViewType.modelVisualization,
      creationParams: creationParams,
      onPlatformViewCreated: (int id) {
        _viewId = id;
        _initializeVisualization();
      },
    );
  }

  void _initializeVisualization() {
    if (_viewId != null) {
      _channel.invokeMethod('initializeVisualization', {
        'view_id': _viewId,
        'engine_id': widget.engineId,
        'visualization_type': widget.visualizationType.name,
        'options': widget.visualizationOptions,
      });
    }
  }

  /// Update visualization with new data
  Future<void> updateVisualization(Map<String, dynamic> data) async {
    if (_viewId != null) {
      await _channel.invokeMethod('updateVisualization', {
        'view_id': _viewId,
        'data': data,
      });
    }
  }

  /// Capture screenshot of the visualization
  Future<Uint8List?> captureScreenshot() async {
    if (_viewId == null) return null;

    try {
      final result = await _channel.invokeMethod('captureScreenshot', {
        'view_id': _viewId,
      });
      return result as Uint8List?;
    } catch (e) {
      debugPrint('Failed to capture screenshot: $e');
      return null;
    }
  }

  @override
  void dispose() {
    if (_viewId != null) {
      _channel.invokeMethod('disposeVisualization', {'view_id': _viewId});
    }
    super.dispose();
  }
}

/// Camera preview platform view for image preprocessing
class TrustformersCameraPreview extends StatefulWidget {
  final TrustformersCameraConfig cameraConfig;
  final void Function(Uint8List imageData)? onImageCaptured;
  final void Function(String error)? onError;

  const TrustformersCameraPreview({
    Key? key,
    required this.cameraConfig,
    this.onImageCaptured,
    this.onError,
  }) : super(key: key);

  @override
  State<TrustformersCameraPreview> createState() => _TrustformersCameraPreviewState();
}

class _TrustformersCameraPreviewState extends State<TrustformersCameraPreview> {
  static const MethodChannel _channel = MethodChannel('trustformers_flutter/camera_preview');
  static const EventChannel _eventChannel = EventChannel('trustformers_flutter/camera_events');
  int? _viewId;
  StreamSubscription<dynamic>? _eventSubscription;

  @override
  Widget build(BuildContext context) {
    final creationParams = {
      'camera_config': widget.cameraConfig.toJson(),
    };

    return TrustformersPlatformView(
      viewType: TrustformersPlatformViewType.cameraPreview,
      creationParams: creationParams,
      onPlatformViewCreated: (int id) {
        _viewId = id;
        _setupCamera();
        _setupEventStream();
      },
    );
  }

  void _setupCamera() {
    if (_viewId != null) {
      _channel.invokeMethod('setupCamera', {
        'view_id': _viewId,
        'config': widget.cameraConfig.toJson(),
      });
    }
  }

  void _setupEventStream() {
    _eventSubscription = _eventChannel.receiveBroadcastStream().listen(
      (dynamic event) {
        final Map<String, dynamic> data = Map<String, dynamic>.from(event);
        
        switch (data['type']) {
          case 'image_captured':
            if (widget.onImageCaptured != null) {
              final imageData = data['image_data'] as Uint8List;
              widget.onImageCaptured!(imageData);
            }
            break;
          case 'error':
            if (widget.onError != null) {
              final error = data['message'] as String;
              widget.onError!(error);
            }
            break;
        }
      },
      onError: (error) {
        if (widget.onError != null) {
          widget.onError!('Event stream error: $error');
        }
      },
    );
  }

  /// Start capturing images
  Future<void> startCapture() async {
    if (_viewId != null) {
      await _channel.invokeMethod('startCapture', {'view_id': _viewId});
    }
  }

  /// Stop capturing images
  Future<void> stopCapture() async {
    if (_viewId != null) {
      await _channel.invokeMethod('stopCapture', {'view_id': _viewId});
    }
  }

  /// Capture a single image
  Future<Uint8List?> captureImage() async {
    if (_viewId == null) return null;

    try {
      final result = await _channel.invokeMethod('captureImage', {
        'view_id': _viewId,
      });
      return result as Uint8List?;
    } catch (e) {
      debugPrint('Failed to capture image: $e');
      return null;
    }
  }

  @override
  void dispose() {
    _eventSubscription?.cancel();
    if (_viewId != null) {
      _channel.invokeMethod('disposeCamera', {'view_id': _viewId});
    }
    super.dispose();
  }
}

/// Interactive debugging console platform view
class TrustformersDebugConsole extends StatefulWidget {
  final String engineId;
  final bool showLogs;
  final bool showMetrics;
  final bool showCommands;

  const TrustformersDebugConsole({
    Key? key,
    required this.engineId,
    this.showLogs = true,
    this.showMetrics = true,
    this.showCommands = true,
  }) : super(key: key);

  @override
  State<TrustformersDebugConsole> createState() => _TrustformersDebugConsoleState();
}

class _TrustformersDebugConsoleState extends State<TrustformersDebugConsole> {
  static const MethodChannel _channel = MethodChannel('trustformers_flutter/debug_console');
  int? _viewId;

  @override
  Widget build(BuildContext context) {
    final creationParams = {
      'engine_id': widget.engineId,
      'show_logs': widget.showLogs,
      'show_metrics': widget.showMetrics,
      'show_commands': widget.showCommands,
    };

    return TrustformersPlatformView(
      viewType: TrustformersPlatformViewType.debugConsole,
      creationParams: creationParams,
      onPlatformViewCreated: (int id) {
        _viewId = id;
        _initializeConsole();
      },
    );
  }

  void _initializeConsole() {
    if (_viewId != null) {
      _channel.invokeMethod('initializeConsole', {
        'view_id': _viewId,
        'engine_id': widget.engineId,
      });
    }
  }

  /// Execute a debug command
  Future<String?> executeCommand(String command) async {
    if (_viewId == null) return null;

    try {
      final result = await _channel.invokeMethod('executeCommand', {
        'view_id': _viewId,
        'command': command,
      });
      return result as String?;
    } catch (e) {
      debugPrint('Failed to execute command: $e');
      return 'Error: $e';
    }
  }

  /// Clear the console
  Future<void> clearConsole() async {
    if (_viewId != null) {
      await _channel.invokeMethod('clearConsole', {'view_id': _viewId});
    }
  }

  @override
  void dispose() {
    if (_viewId != null) {
      _channel.invokeMethod('disposeConsole', {'view_id': _viewId});
    }
    super.dispose();
  }
}

/// Platform view types
enum TrustformersPlatformViewType {
  performanceChart('trustformers_performance_chart', 'TrustformersPerformanceChartView'),
  modelVisualization('trustformers_model_viz', 'TrustformersModelVisualizationView'),
  cameraPreview('trustformers_camera', 'TrustformersCameraPreviewView'),
  debugConsole('trustformers_debug', 'TrustformersDebugConsoleView');

  const TrustformersPlatformViewType(this.androidViewType, this.iosViewType);

  final String androidViewType;
  final String iosViewType;
}

/// Chart types for performance visualization
enum TrustformersChartType {
  realTimeMetrics,
  inferenceHistory,
  memoryUsage,
  throughputAnalysis,
  batteryImpact,
  thermalProfile,
}

/// Visualization types for model visualization
enum TrustformersVisualizationType {
  modelArchitecture,
  attentionWeights,
  activationMaps,
  gradientFlow,
  layerOutputs,
  parameterDistribution,
}

/// Camera configuration for image preprocessing
class TrustformersCameraConfig {
  final TrustformersCameraResolution resolution;
  final TrustformersCameraFormat format;
  final bool enableAutoFocus;
  final bool enableFlash;
  final double captureRate; // fps
  final Map<String, dynamic> preprocessingOptions;

  const TrustformersCameraConfig({
    this.resolution = TrustformersCameraResolution.hd720,
    this.format = TrustformersCameraFormat.yuv420,
    this.enableAutoFocus = true,
    this.enableFlash = false,
    this.captureRate = 30.0,
    this.preprocessingOptions = const {},
  });

  Map<String, dynamic> toJson() {
    return {
      'resolution': resolution.name,
      'format': format.name,
      'enable_auto_focus': enableAutoFocus,
      'enable_flash': enableFlash,
      'capture_rate': captureRate,
      'preprocessing_options': preprocessingOptions,
    };
  }
}

/// Camera resolution options
enum TrustformersCameraResolution {
  qvga, // 320x240
  vga,  // 640x480
  hd720, // 1280x720
  hd1080, // 1920x1080
  uhd4k, // 3840x2160
}

/// Camera format options
enum TrustformersCameraFormat {
  rgb888,
  bgr888,
  yuv420,
  nv21,
  jpeg,
}

/// Platform view registry for registering custom views
class TrustformersPlatformViewRegistry {
  static const MethodChannel _channel = MethodChannel('trustformers_flutter/platform_view_registry');

  /// Register all TrustformeRS platform views
  static Future<void> registerViews() async {
    try {
      await _channel.invokeMethod('registerViews');
    } catch (e) {
      debugPrint('Failed to register platform views: $e');
    }
  }

  /// Check if platform views are supported
  static Future<bool> isSupported() async {
    try {
      final result = await _channel.invokeMethod('isSupported');
      return result as bool? ?? false;
    } catch (e) {
      debugPrint('Failed to check platform view support: $e');
      return false;
    }
  }

  /// Get available platform view types
  static Future<List<String>> getAvailableViews() async {
    try {
      final result = await _channel.invokeMethod('getAvailableViews');
      return List<String>.from(result as List? ?? []);
    } catch (e) {
      debugPrint('Failed to get available views: $e');
      return [];
    }
  }
}