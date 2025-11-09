import 'package:flutter/material.dart';
import 'package:trustformers_flutter/trustformers_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TrustformeRS Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const TrustformersDemo(),
    );
  }
}

class TrustformersDemo extends StatefulWidget {
  const TrustformersDemo({super.key});

  @override
  State<TrustformersDemo> createState() => _TrustformersDemoState();
}

class _TrustformersDemoState extends State<TrustformersDemo> {
  TrustformersEngine? _engine;
  TrustformersDeviceInfo? _deviceInfo;
  TrustformersPerformanceMetrics? _performanceMetrics;
  String _status = 'Not initialized';
  bool _isLoading = false;
  String _inferenceResult = '';
  final TextEditingController _inputController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _initializeEngine();
  }

  Future<void> _initializeEngine() async {
    setState(() {
      _isLoading = true;
      _status = 'Initializing...';
    });

    try {
      // Get device information first
      final config = TrustformersConfig.autoDetect(
        engineId: 'demo_engine',
        modelPath: '/path/to/your/model', // Replace with actual model path
      );

      // Create and initialize engine
      _engine = await TrustformersEngine.create(config);
      
      // Get device info
      _deviceInfo = await _engine!.getDeviceInfo();

      // Listen to events
      _engine!.eventStream.listen((event) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Event: ${event.type}')),
        );
      });

      setState(() {
        _status = 'Engine initialized successfully';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Initialization failed: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _loadModel() async {
    if (_engine == null) return;

    setState(() {
      _isLoading = true;
      _status = 'Loading model...';
    });

    try {
      await _engine!.loadModel('/path/to/your/model'); // Replace with actual model path
      setState(() {
        _status = 'Model loaded successfully';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Model loading failed: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _runInference() async {
    if (_engine == null || !_engine!.isModelLoaded) return;

    final inputText = _inputController.text.trim();
    if (inputText.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter some input text')),
      );
      return;
    }

    setState(() {
      _isLoading = true;
      _status = 'Running inference...';
    });

    try {
      // Convert text to token IDs (this would typically use a tokenizer)
      final inputIds = inputText.codeUnits; // Simplified for demo

      final request = TrustformersInferenceRequest.textGeneration(
        inputIds: inputIds,
        maxLength: 50,
        temperature: 0.8,
        doSample: true,
      );

      final result = await _engine!.inference(request);
      
      // Get updated performance metrics
      _performanceMetrics = await _engine!.getPerformanceMetrics();

      setState(() {
        _inferenceResult = 'Generated ${result.tokens.length} tokens in ${result.inferenceTimeMs.toStringAsFixed(2)}ms';
        _status = 'Inference completed successfully';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _inferenceResult = 'Inference failed: $e';
        _status = 'Inference failed';
        _isLoading = false;
      });
    }
  }

  Future<void> _updatePerformanceMetrics() async {
    if (_engine == null || !_engine!.isInitialized) return;

    try {
      final metrics = await _engine!.getPerformanceMetrics();
      setState(() {
        _performanceMetrics = metrics;
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to get metrics: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('TrustformeRS Flutter Demo'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _updatePerformanceMetrics,
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Status Card
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Status',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 8),
                      Text(_status),
                      if (_isLoading)
                        const Padding(
                          padding: EdgeInsets.only(top: 8.0),
                          child: LinearProgressIndicator(),
                        ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // Device Info Card
              if (_deviceInfo != null)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Device Information',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 8),
                        Text('Platform: ${_deviceInfo!.platform}'),
                        Text('Model: ${_deviceInfo!.model}'),
                        Text('Memory: ${_deviceInfo!.memoryAvailableMb}/${_deviceInfo!.memoryTotalMb} MB'),
                        Text('CPU Cores: ${_deviceInfo!.cpuCores}'),
                        Text('GPU Available: ${_deviceInfo!.gpuAvailable}'),
                        Text('Neural Engine: ${_deviceInfo!.neuralEngineAvailable}'),
                        Text('Device Tier: ${_deviceInfo!.deviceTier}'),
                        Text('Performance Estimate: ${_deviceInfo!.performanceEstimate}'),
                      ],
                    ),
                  ),
                ),
              const SizedBox(height: 16),

              // Inference Input Card
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Inference Input',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 8),
                      TextField(
                        controller: _inputController,
                        decoration: const InputDecoration(
                          hintText: 'Enter text for inference...',
                          border: OutlineInputBorder(),
                        ),
                        maxLines: 3,
                      ),
                      const SizedBox(height: 16),
                      Row(
                        children: [
                          ElevatedButton(
                            onPressed: _engine?.isInitialized == true && !_isLoading ? _loadModel : null,
                            child: const Text('Load Model'),
                          ),
                          const SizedBox(width: 16),
                          ElevatedButton(
                            onPressed: _engine?.isModelLoaded == true && !_isLoading ? _runInference : null,
                            child: const Text('Run Inference'),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // Inference Result Card
              if (_inferenceResult.isNotEmpty)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Inference Result',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 8),
                        Text(_inferenceResult),
                      ],
                    ),
                  ),
                ),

              // Performance Metrics Card
              if (_performanceMetrics != null)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Performance Metrics',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 8),
                        Text('Total Inferences: ${_performanceMetrics!.totalInferences}'),
                        Text('Avg Time: ${_performanceMetrics!.avgInferenceTimeMs.toStringAsFixed(2)} ms'),
                        Text('Throughput: ${_performanceMetrics!.throughputTokensPerSec.toStringAsFixed(2)} tokens/s'),
                        Text('Memory Usage: ${_performanceMetrics!.currentMemoryMb} MB'),
                        Text('Peak Memory: ${_performanceMetrics!.peakMemoryMb} MB'),
                        Text('Performance Grade: ${_performanceMetrics!.performanceGrade}'),
                        Text('Battery Impact: ${_performanceMetrics!.estimatedBatteryImpact}'),
                        const SizedBox(height: 8),
                        
                        // Performance Recommendations
                        if (_performanceMetrics!.getRecommendations().isNotEmpty) ...[
                          Text(
                            'Recommendations:',
                            style: Theme.of(context).textTheme.titleSmall,
                          ),
                          const SizedBox(height: 4),
                          ..._performanceMetrics!.getRecommendations().map(
                            (rec) => Padding(
                              padding: const EdgeInsets.only(left: 8.0),
                              child: Text('• ${rec.title}: ${rec.description}'),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _engine?.dispose();
    _inputController.dispose();
    super.dispose();
  }
}