import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { Picker } from '@react-native-picker/picker';

// Import TrustformeRS React Native module
import {
  TrustformersEngine,
  TrustformersConfig,
  InferenceRequest,
  InferenceResponse,
  PerformanceMetrics,
  DeviceInfo,
  ModelInfo,
  TrustformersPerformanceMonitor,
  QuantizationMode,
  MemoryOptimizationLevel,
  BatteryOptimizationLevel,
} from '@trustformers/react-native';

interface ComponentState {
  isInitialized: boolean;
  isLoading: boolean;
  availableModels: ModelInfo[];
  selectedModelId: string | null;
  deviceInfo: DeviceInfo | null;
  lastInferenceResult: InferenceResponse | null;
  performanceMetrics: PerformanceMetrics | null;
  status: string;
}

const TrustformersCompleteExample: React.FC = () => {
  const [state, setState] = useState<ComponentState>({
    isInitialized: false,
    isLoading: false,
    availableModels: [],
    selectedModelId: null,
    deviceInfo: null,
    lastInferenceResult: null,
    performanceMetrics: null,
    status: 'Initializing TrustformeRS...',
  });

  const [engine, setEngine] = useState<TrustformersEngine | null>(null);
  const [performanceMonitor, setPerformanceMonitor] = useState<TrustformersPerformanceMonitor | null>(null);

  // Initialize TrustformeRS engine
  const initializeTrustformers = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, isLoading: true, status: 'Initializing engine...' }));

      // Get device information first
      const deviceInfo = await TrustformersEngine.getDeviceInfo();
      
      // Configure the engine based on device capabilities
      const config: TrustformersConfig = {
        enablePerformanceMonitoring: true,
        enableDebugLogging: __DEV__,
        maxConcurrentInferences: deviceInfo.cpuCoreCount >= 8 ? 3 : 2,
        optimizeJsBridge: true,
        useBackgroundThread: true,
        enableResultCaching: true,
        maxCacheSizeMb: deviceInfo.totalMemoryMB > 4096 ? 256 : 128,
        
        // Mobile optimizations
        quantizationMode: deviceInfo.totalMemoryMB > 6144 
          ? QuantizationMode.FP16 
          : QuantizationMode.INT8,
        memoryOptimizationLevel: deviceInfo.totalMemoryMB > 8192 
          ? MemoryOptimizationLevel.BALANCED 
          : MemoryOptimizationLevel.AGGRESSIVE,
        batteryOptimizationLevel: BatteryOptimizationLevel.BALANCED,
        
        // Platform-specific optimizations
        enableGpuAcceleration: deviceInfo.hasGpu,
        enableCoreMLAcceleration: Platform.OS === 'ios' && deviceInfo.hasCoreML,
        enableNNAPIAcceleration: Platform.OS === 'android' && deviceInfo.hasNNAPI,
        
        // Thermal management
        enableThermalMonitoring: true,
        thermalThrottlingEnabled: true,
      };

      // Initialize engine and performance monitor
      const engineInstance = await TrustformersEngine.initialize(config);
      const monitorInstance = new TrustformersPerformanceMonitor();

      setEngine(engineInstance);
      setPerformanceMonitor(monitorInstance);

      // Load available models
      const models = await engineInstance.getAvailableModels();
      
      setState(prev => ({
        ...prev,
        isInitialized: true,
        isLoading: false,
        deviceInfo,
        availableModels: models,
        selectedModelId: models.length > 0 ? models[0].id : null,
        status: 'Ready for inference',
      }));

      console.log('TrustformeRS initialized successfully');
      console.log(`Device: ${deviceInfo.deviceModel} (${deviceInfo.platform})`);
      console.log(`Available models: ${models.length}`);
      
    } catch (error) {
      console.error('Failed to initialize TrustformeRS:', error);
      setState(prev => ({
        ...prev,
        isLoading: false,
        status: `Initialization failed: ${error.message}`,
      }));
    }
  }, []);

  // Run inference with selected model
  const runInference = useCallback(async () => {
    if (!engine || !performanceMonitor || !state.selectedModelId) {
      Alert.alert('Error', 'Engine not initialized or no model selected');
      return;
    }

    try {
      setState(prev => ({ ...prev, isLoading: true, status: 'Running inference...' }));

      // Start performance monitoring
      await performanceMonitor.startMonitoring();

      // Prepare inference request
      const request: InferenceRequest = {
        requestId: `inference_${Date.now()}`,
        modelId: state.selectedModelId,
        inputData: generateSampleInput(), // Replace with actual input
        inputShape: [1, 768], // Example shape
        configOverride: {
          maxTokens: 100,
          temperature: 0.7,
          topP: 0.9,
          enableBatching: true,
        },
        enablePreprocessing: true,
        enablePostprocessing: true,
      };

      // Run inference
      const result = await engine.runInference(request);

      // Stop monitoring and get metrics
      const metrics = await performanceMonitor.stopMonitoring();

      setState(prev => ({
        ...prev,
        isLoading: false,
        lastInferenceResult: result,
        performanceMetrics: metrics,
        status: `Inference completed in ${metrics.inferenceTimeMs}ms`,
      }));

      // Show performance summary
      showPerformanceSummary(metrics);

    } catch (error) {
      console.error('Inference failed:', error);
      setState(prev => ({
        ...prev,
        isLoading: false,
        status: `Inference failed: ${error.message}`,
      }));
      Alert.alert('Inference Error', error.message);
    }
  }, [engine, performanceMonitor, state.selectedModelId]);

  // Generate sample input data (replace with real input in production)
  const generateSampleInput = (): number[] => {
    return Array.from({ length: 768 }, (_, i) => (Math.sin(i / 100) + 1) / 2);
  };

  // Show performance metrics in alert
  const showPerformanceSummary = (metrics: PerformanceMetrics) => {
    const summary = `
Inference Time: ${metrics.inferenceTimeMs}ms
Memory Usage: ${metrics.peakMemoryUsageMB}MB
CPU Usage: ${metrics.averageCpuUsage.toFixed(1)}%
GPU Usage: ${metrics.averageGpuUsage.toFixed(1)}%
Battery Impact: ${metrics.batteryImpactLevel}
Thermal State: ${metrics.thermalState}
Network Usage: ${metrics.networkUsageKB}KB
    `.trim();

    Alert.alert('Performance Summary', summary);
  };

  // Initialize on component mount
  useEffect(() => {
    initializeTrustformers();
    
    // Cleanup on unmount
    return () => {
      engine?.dispose();
      performanceMonitor?.dispose();
    };
  }, [initializeTrustformers]);

  const renderDeviceInfo = () => {
    if (!state.deviceInfo) return null;

    const { deviceInfo } = state;
    return (
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Device Information</Text>
        <Text style={styles.infoText}>Model: {deviceInfo.deviceModel}</Text>
        <Text style={styles.infoText}>Platform: {deviceInfo.platform}</Text>
        <Text style={styles.infoText}>CPU Cores: {deviceInfo.cpuCoreCount}</Text>
        <Text style={styles.infoText}>Memory: {deviceInfo.totalMemoryMB}MB</Text>
        <Text style={styles.infoText}>GPU: {deviceInfo.hasGpu ? 'Available' : 'Not Available'}</Text>
        {Platform.OS === 'ios' && (
          <Text style={styles.infoText}>Core ML: {deviceInfo.hasCoreML ? 'Available' : 'Not Available'}</Text>
        )}
        {Platform.OS === 'android' && (
          <Text style={styles.infoText}>NNAPI: {deviceInfo.hasNNAPI ? 'Available' : 'Not Available'}</Text>
        )}
      </View>
    );
  };

  const renderModelSelector = () => {
    return (
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Model Selection</Text>
        <View style={styles.pickerContainer}>
          <Picker
            selectedValue={state.selectedModelId}
            onValueChange={(value) => setState(prev => ({ ...prev, selectedModelId: value }))}
            style={styles.picker}
          >
            {state.availableModels.map((model) => (
              <Picker.Item key={model.id} label={model.name} value={model.id} />
            ))}
          </Picker>
        </View>
        <TouchableOpacity
          style={[styles.button, state.isLoading && styles.buttonDisabled]}
          onPress={runInference}
          disabled={state.isLoading || !state.selectedModelId}
        >
          {state.isLoading ? (
            <ActivityIndicator color="white" size="small" />
          ) : (
            <Text style={styles.buttonText}>Run Inference</Text>
          )}
        </TouchableOpacity>
      </View>
    );
  };

  const renderResults = () => {
    return (
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Results</Text>
        <Text style={styles.statusText}>Status: {state.status}</Text>
        
        {state.lastInferenceResult && (
          <View style={styles.resultContainer}>
            <Text style={styles.resultTitle}>Last Inference Result:</Text>
            <ScrollView style={styles.resultScroll} horizontal>
              <Text style={styles.resultText}>
                {JSON.stringify(state.lastInferenceResult, null, 2)}
              </Text>
            </ScrollView>
          </View>
        )}
      </View>
    );
  };

  const renderPerformanceMetrics = () => {
    if (!state.performanceMetrics) {
      return (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Performance Metrics</Text>
          <Text style={styles.infoText}>No performance data available. Run an inference to see metrics.</Text>
        </View>
      );
    }

    const { performanceMetrics } = state;
    return (
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Performance Metrics</Text>
        <View style={styles.metricsGrid}>
          <View style={styles.metricCard}>
            <Text style={styles.metricValue}>{performanceMetrics.inferenceTimeMs}ms</Text>
            <Text style={styles.metricLabel}>Inference Time</Text>
          </View>
          <View style={styles.metricCard}>
            <Text style={styles.metricValue}>{performanceMetrics.peakMemoryUsageMB}MB</Text>
            <Text style={styles.metricLabel}>Memory Usage</Text>
          </View>
          <View style={styles.metricCard}>
            <Text style={styles.metricValue}>{performanceMetrics.averageCpuUsage.toFixed(1)}%</Text>
            <Text style={styles.metricLabel}>CPU Usage</Text>
          </View>
          <View style={styles.metricCard}>
            <Text style={styles.metricValue}>{performanceMetrics.batteryImpactLevel}</Text>
            <Text style={styles.metricLabel}>Battery Impact</Text>
          </View>
        </View>
      </View>
    );
  };

  if (!state.isInitialized && state.isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>{state.status}</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <Text style={styles.title}>TrustformeRS React Native Demo</Text>
      
      {renderDeviceInfo()}
      {renderModelSelector()}
      {renderResults()}
      {renderPerformanceMetrics()}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  contentContainer: {
    padding: 16,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#333',
  },
  card: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#333',
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  pickerContainer: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 4,
    marginBottom: 16,
  },
  picker: {
    height: 50,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  statusText: {
    fontSize: 14,
    color: '#333',
    marginBottom: 12,
  },
  resultContainer: {
    marginTop: 8,
  },
  resultTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  resultScroll: {
    maxHeight: 200,
    backgroundColor: '#f9f9f9',
    borderRadius: 4,
    padding: 8,
  },
  resultText: {
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
    fontSize: 12,
    color: '#333',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  metricCard: {
    width: '48%',
    backgroundColor: '#f0f8ff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  metricValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#007AFF',
  },
  metricLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
    textAlign: 'center',
  },
});

export default TrustformersCompleteExample;