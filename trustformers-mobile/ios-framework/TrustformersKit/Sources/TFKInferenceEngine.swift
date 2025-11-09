//
//  TFKInferenceEngine.swift
//  TrustformersKit
//
//  Main inference engine implementation
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import Foundation
import CoreML
import Metal
import MetalPerformanceShaders
import Accelerate

/// Main inference engine for TrustformersKit
public class TFKInferenceEngine: NSObject {
    
    // MARK: - Properties
    
    private var engineHandle: OpaquePointer?
    private let config: TFKModelConfig
    private var loadedModels: [String: TFKModel] = [:]
    private let engineQueue = DispatchQueue(label: "com.trustformers.inferenceEngine", qos: .userInitiated)
    
    /// Performance statistics
    public private(set) var performanceStats: TFKPerformanceStats
    
    /// Device information
    public private(set) var deviceInfo: TFKDeviceInfo
    
    // Memory pressure monitoring
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    private var memoryPressureHandler: ((TFKMemoryOptimization) -> Void)?
    
    // Thermal state monitoring
    private var thermalStateObserver: NSObjectProtocol?
    private var thermalThrottlingEnabled = false
    
    // MARK: - Singleton
    
    /// Shared instance
    public static let shared = TFKInferenceEngine(config: .optimizedConfig())
    
    // MARK: - Initialization
    
    /// Initialize with configuration
    public init(config: TFKModelConfig) {
        self.config = config
        self.performanceStats = TFKPerformanceStats()
        self.deviceInfo = TFKDeviceInfo.currentDevice()
        
        super.init()
        
        // Create engine handle
        var cConfig = config.toCConfig()
        engineHandle = tfk_create_engine(&cConfig)
        
        // Setup monitoring
        setupMemoryPressureMonitoring()
        setupThermalStateMonitoring()
        
        // Configure logging
        setupLogging()
    }
    
    deinit {
        // Clean up
        memoryPressureSource?.cancel()
        if let observer = thermalStateObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        
        if let handle = engineHandle {
            tfk_destroy_engine(handle)
        }
    }
    
    // MARK: - Model Management
    
    /// Load model at path
    public func loadModel(at path: String, config: TFKModelConfig) throws -> TFKModel {
        guard FileManager.default.fileExists(atPath: path) else {
            throw TFKError.modelNotFound(path: path)
        }
        
        // Validate configuration
        try config.validate()
        
        // Load through Rust engine
        guard let handle = engineHandle else {
            throw TFKError.engineNotInitialized
        }
        
        let success = path.withCString { pathPtr in
            tfk_load_model(handle, pathPtr)
        }
        
        guard success else {
            let errorMessage = String(cString: tfk_get_last_error())
            throw TFKError.modelLoadFailed(reason: errorMessage)
        }
        
        // Create model wrapper
        let model = TFKModel(path: path, config: config, engineHandle: handle)
        loadedModels[path] = model
        
        return model
    }
    
    /// Load model from bundle
    public func loadModel(bundleResource name: String, type: String, config: TFKModelConfig) throws -> TFKModel {
        guard let path = Bundle.main.path(forResource: name, ofType: type) else {
            throw TFKError.modelNotFound(path: "\(name).\(type)")
        }
        
        return try loadModel(at: path, config: config)
    }
    
    // MARK: - Inference
    
    /// Perform inference with model
    public func performInference(_ model: TFKModel, input: TFKTensor) -> TFKInferenceResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            // Check thermal state
            if thermalThrottlingEnabled && deviceInfo.thermalState.rawValue >= TFKThermalState.serious.rawValue {
                // Reduce thread count for thermal throttling
                var modifiedConfig = model.config
                modifiedConfig.numThreads = 1
                model.updateConfig(modifiedConfig)
            }
            
            // Perform inference through Rust engine
            guard let handle = engineHandle,
                  let inputHandle = input.cHandle else {
                throw TFKError.inferenceFailed(reason: "Invalid handles")
            }
            
            guard let resultPtr = tfk_inference(handle, inputHandle) else {
                let errorMessage = String(cString: tfk_get_last_error())
                throw TFKError.inferenceFailed(reason: errorMessage)
            }
            
            defer {
                tfk_free_inference_result(resultPtr)
            }
            
            let result = resultPtr.pointee
            
            // Create output tensor
            let outputData = Array(UnsafeBufferPointer(start: result.data, count: Int(result.length)))
            let outputTensor = TFKTensor(floats: outputData, shape: [result.length])
            
            let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
            
            // Update stats
            performanceStats.recordInference(time: inferenceTime, memoryUsed: Int(result.memory_used_mb))
            
            return TFKInferenceResult(
                output: outputTensor,
                inferenceTime: inferenceTime,
                memoryUsed: Int(result.memory_used_mb),
                success: result.success,
                error: result.success ? nil : TFKError.inferenceFailed(reason: String(cString: result.error_message))
            )
            
        } catch {
            let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
            return TFKInferenceResult(
                output: TFKTensor(floats: [], shape: [0]),
                inferenceTime: inferenceTime,
                memoryUsed: 0,
                success: false,
                error: error
            )
        }
    }
    
    /// Perform batch inference
    public func performBatchInference(_ model: TFKModel, inputs: [TFKTensor]) -> [TFKInferenceResult] {
        guard model.config.enableBatching else {
            // Fall back to sequential inference
            return inputs.map { performInference(model, input: $0) }
        }
        
        // Perform batched inference through Rust engine
        // Implementation depends on batch processing in Rust
        return inputs.map { performInference(model, input: $0) }
    }
    
    // MARK: - Configuration
    
    /// Set thermal throttling
    public func setThermalThrottling(_ enabled: Bool) {
        thermalThrottlingEnabled = enabled
    }
    
    /// Set memory pressure handler
    public func setMemoryPressureHandler(_ handler: @escaping (TFKMemoryOptimization) -> Void) {
        memoryPressureHandler = handler
    }
    
    /// Optimize for battery life
    public func optimizeForBatteryLife(_ enabled: Bool) {
        engineQueue.async { [weak self] in
            guard let self = self else { return }
            
            if enabled {
                // Reduce thread count
                var batteryConfig = self.config
                batteryConfig.numThreads = 1
                batteryConfig.enableBatching = false
                
                // Update all loaded models
                for model in self.loadedModels.values {
                    model.updateConfig(batteryConfig)
                }
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func setupMemoryPressureMonitoring() {
        memoryPressureSource = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: engineQueue)
        
        memoryPressureSource?.setEventHandler { [weak self] in
            guard let self = self else { return }
            
            let event = self.memoryPressureSource?.data
            
            if event?.contains(.critical) == true {
                // Critical memory pressure
                self.handleMemoryPressure(level: .maximum)
            } else if event?.contains(.warning) == true {
                // Warning memory pressure
                self.handleMemoryPressure(level: .balanced)
            }
        }
        
        memoryPressureSource?.resume()
    }
    
    private func handleMemoryPressure(level: TFKMemoryOptimization) {
        // Notify handler if set
        memoryPressureHandler?(level)
        
        // Update configuration for all models
        var pressureConfig = config
        pressureConfig.memoryOptimization = level
        
        for model in loadedModels.values {
            model.updateConfig(pressureConfig)
        }
        
        // Force memory cleanup
        if let handle = engineHandle {
            let currentUsage = tfk_get_current_memory_usage(handle)
            if currentUsage > config.maxMemoryMB {
                // Unload least recently used models
                // Implementation would track model usage
            }
        }
    }
    
    private func setupThermalStateMonitoring() {
        if #available(iOS 11.0, *) {
            thermalStateObserver = NotificationCenter.default.addObserver(
                forName: ProcessInfo.thermalStateDidChangeNotification,
                object: nil,
                queue: .main
            ) { [weak self] _ in
                self?.handleThermalStateChange()
            }
        }
    }
    
    private func handleThermalStateChange() {
        if #available(iOS 11.0, *) {
            let thermalState = ProcessInfo.processInfo.thermalState
            
            // Update device info thermal state
            deviceInfo.updateThermalState(thermalState)
            
            if thermalThrottlingEnabled {
                switch thermalState {
                case .nominal, .fair:
                    // Normal operation
                    break
                case .serious, .critical:
                    // Reduce performance to manage thermals
                    optimizeForThermalState(critical: thermalState == .critical)
                @unknown default:
                    break
                }
            }
        }
    }
    
    private func optimizeForThermalState(critical: Bool) {
        var thermalConfig = config
        
        if critical {
            // Extreme thermal throttling
            thermalConfig.numThreads = 1
            thermalConfig.memoryOptimization = .maximum
            thermalConfig.enableBatching = false
        } else {
            // Moderate thermal throttling
            thermalConfig.numThreads = config.numThreads / 2
            thermalConfig.memoryOptimization = .balanced
        }
        
        // Update all models
        for model in loadedModels.values {
            model.updateConfig(thermalConfig)
        }
    }
    
    private func setupLogging() {
        // Set up Rust logging callback
        let callback: TFKLogCallback = { level, message in
            guard let message = message else { return }
            let swiftMessage = String(cString: message)
            
            TFKLogger.log(level: TFKLogLevel(rawValue: UInt(level)) ?? .info, message: swiftMessage)
        }
        
        tfk_set_log_callback(callback)
    }
}

// MARK: - TFKError

/// TrustformersKit errors
public enum TFKError: LocalizedError {
    case engineNotInitialized
    case modelNotFound(path: String)
    case modelLoadFailed(reason: String)
    case invalidConfiguration(reason: String)
    case inferenceFailed(reason: String)
    case outOfMemory
    case unsupportedBackend(TFKModelBackend)
    case tensorShapeMismatch(expected: [Int], actual: [Int])
    case quantizationFailed(reason: String)
    case thermalThrottling
    
    public var errorDescription: String? {
        switch self {
        case .engineNotInitialized:
            return "Inference engine not initialized"
        case .modelNotFound(let path):
            return "Model not found at path: \(path)"
        case .modelLoadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .invalidConfiguration(let reason):
            return "Invalid configuration: \(reason)"
        case .inferenceFailed(let reason):
            return "Inference failed: \(reason)"
        case .outOfMemory:
            return "Out of memory"
        case .unsupportedBackend(let backend):
            return "Unsupported backend: \(backend)"
        case .tensorShapeMismatch(let expected, let actual):
            return "Tensor shape mismatch. Expected: \(expected), Actual: \(actual)"
        case .quantizationFailed(let reason):
            return "Quantization failed: \(reason)"
        case .thermalThrottling:
            return "Thermal throttling active"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .outOfMemory:
            return "Try reducing model size or using quantization"
        case .thermalThrottling:
            return "Allow device to cool down before continuing"
        case .unsupportedBackend:
            return "Use a different backend or update to a newer iOS version"
        default:
            return nil
        }
    }
}