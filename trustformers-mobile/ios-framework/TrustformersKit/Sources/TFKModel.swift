//
//  TFKModel.swift
//  TrustformersKit
//
//  Model representation and management
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import Foundation
import CoreML

/// Represents a TrustformersKit model
public class TFKModel: NSObject {
    
    // MARK: - Properties
    
    /// Model file path
    public let modelPath: String
    
    /// Model configuration
    public private(set) var config: TFKModelConfig
    
    /// Whether model is loaded
    public private(set) var isLoaded: Bool = false
    
    /// Engine handle reference
    private weak var engineHandle: OpaquePointer?
    
    /// Core ML model (if using Core ML backend)
    private var coreMLModel: MLModel?
    
    /// Model metadata
    private var metadata: [String: Any] = [:]
    
    /// Load time
    private var loadTime: TimeInterval = 0
    
    // MARK: - Initialization
    
    /// Initialize with path and configuration
    public init(path: String, config: TFKModelConfig) {
        self.modelPath = path
        self.config = config
        super.init()
    }
    
    /// Initialize with bundle resource
    public convenience init(bundleResource name: String, type: String, config: TFKModelConfig) throws {
        guard let path = Bundle.main.path(forResource: name, ofType: type) else {
            throw TFKError.modelNotFound(path: "\(name).\(type)")
        }
        
        self.init(path: path, config: config)
    }
    
    /// Initialize with Core ML model
    @available(iOS 11.0, *)
    public init(coreMLModel: MLModel, config: TFKModelConfig) {
        self.modelPath = "CoreML Model"
        self.config = config
        self.coreMLModel = coreMLModel
        self.isLoaded = true
        super.init()
    }
    
    // MARK: - Internal Methods
    
    /// Set engine handle (called by TFKInferenceEngine)
    internal func setEngineHandle(_ handle: OpaquePointer) {
        self.engineHandle = handle
    }
    
    /// Update configuration
    internal func updateConfig(_ newConfig: TFKModelConfig) {
        self.config = newConfig
        
        // If model is loaded, apply configuration changes
        if isLoaded, let handle = engineHandle {
            applyConfigurationToEngine(handle)
        }
    }
    
    // MARK: - Public Methods
    
    /// Preload model into memory
    public func preload() throws {
        guard !isLoaded else { return }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        switch config.backend {
        case .coreML:
            try loadCoreMLModel()
        case .cpu, .gpu:
            try loadTrustformersModel()
        case .custom:
            throw TFKError.unsupportedBackend(.custom)
        }
        
        loadTime = CFAbsoluteTimeGetCurrent() - startTime
        isLoaded = true
        
        // Load metadata
        loadModelMetadata()
    }
    
    /// Unload model from memory
    public func unload() {
        guard isLoaded else { return }
        
        if let handle = engineHandle {
            tfk_unload_model(handle)
        }
        
        coreMLModel = nil
        isLoaded = false
        metadata.removeAll()
    }
    
    /// Get model size in MB
    public func getModelSize() -> Double {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            return 0
        }
        
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: modelPath)
            if let fileSize = attributes[.size] as? NSNumber {
                return fileSize.doubleValue / (1024 * 1024) // Convert to MB
            }
        } catch {
            TFKLogger.log(level: .warning, message: "Failed to get model size: \(error)")
        }
        
        return 0
    }
    
    /// Get estimated memory usage
    public func getEstimatedMemoryUsage() -> Int {
        let modelSizeMB = Int(getModelSize())
        return config.estimateMemoryUsage(forModelSizeMB: modelSizeMB)
    }
    
    /// Get model metadata
    public func getMetadata() -> [String: Any] {
        return metadata
    }
    
    /// Validate model
    public func validate() throws {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw TFKError.modelNotFound(path: modelPath)
        }
        
        // Additional validation based on backend
        switch config.backend {
        case .coreML:
            try validateCoreMLModel()
        case .cpu, .gpu:
            try validateTrustformersModel()
        case .custom:
            // Custom validation
            break
        }
    }
    
    // MARK: - Private Methods
    
    private func loadCoreMLModel() throws {
        guard #available(iOS 11.0, *) else {
            throw TFKError.unsupportedBackend(.coreML)
        }
        
        let compiledModelURL = try compileModelIfNeeded()
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnitsForBackend()
        
        if #available(iOS 16.0, *) {
            configuration.modelDisplayName = "TrustformersKit Model"
        }
        
        coreMLModel = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
    }
    
    private func compileModelIfNeeded() throws -> URL {
        let modelURL = URL(fileURLWithPath: modelPath)
        
        // Check if model needs compilation
        if modelPath.hasSuffix(".mlmodel") {
            // Compile the model
            let compiledURL = try MLModel.compileModel(at: modelURL)
            return compiledURL
        } else if modelPath.hasSuffix(".mlmodelc") {
            // Already compiled
            return modelURL
        } else {
            throw TFKError.modelLoadFailed(reason: "Invalid Core ML model format")
        }
    }
    
    @available(iOS 11.0, *)
    private func computeUnitsForBackend() -> MLComputeUnits {
        switch config.backend {
        case .coreML:
            if #available(iOS 13.0, *) {
                return .all
            } else {
                return .cpuAndGPU
            }
        case .cpu:
            return .cpuOnly
        case .gpu:
            if #available(iOS 13.0, *) {
                return .cpuAndGPU
            } else {
                return .cpuOnly
            }
        case .custom:
            return .cpuOnly
        }
    }
    
    private func loadTrustformersModel() throws {
        // This is handled by the Rust engine through TFKInferenceEngine
        // We just validate the model format here
        guard modelPath.hasSuffix(".tfm") || modelPath.hasSuffix(".onnx") else {
            throw TFKError.modelLoadFailed(reason: "Unsupported model format")
        }
    }
    
    private func validateCoreMLModel() throws {
        guard #available(iOS 11.0, *) else {
            throw TFKError.unsupportedBackend(.coreML)
        }
        
        // Basic validation - check if file can be compiled
        _ = try compileModelIfNeeded()
    }
    
    private func validateTrustformersModel() throws {
        // Validate model file format
        let validExtensions = [".tfm", ".onnx", ".tflite"]
        let hasValidExtension = validExtensions.contains { modelPath.hasSuffix($0) }
        
        guard hasValidExtension else {
            throw TFKError.modelLoadFailed(reason: "Invalid model format")
        }
        
        // Check file size
        let modelSize = getModelSize()
        guard modelSize > 0 else {
            throw TFKError.modelLoadFailed(reason: "Model file is empty")
        }
        
        // Check against memory limits
        let estimatedMemory = getEstimatedMemoryUsage()
        guard estimatedMemory <= config.maxMemoryMB else {
            throw TFKError.outOfMemory
        }
    }
    
    private func loadModelMetadata() {
        // Load metadata from model file or accompanying metadata file
        let metadataPath = (modelPath as NSString).deletingPathExtension + ".json"
        
        if FileManager.default.fileExists(atPath: metadataPath) {
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: metadataPath))
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    metadata = json
                }
            } catch {
                TFKLogger.log(level: .warning, message: "Failed to load model metadata: \(error)")
            }
        }
        
        // Add runtime metadata
        metadata["loadTime"] = loadTime
        metadata["modelSize"] = getModelSize()
        metadata["estimatedMemory"] = getEstimatedMemoryUsage()
        metadata["backend"] = config.backend.rawValue
    }
    
    private func applyConfigurationToEngine(_ handle: OpaquePointer) {
        // Apply configuration changes to the Rust engine
        tfk_set_memory_limit(handle, UInt32(config.maxMemoryMB))
        
        if config.numThreads > 0 {
            // Set thread count through engine configuration
            // This would be implemented in the Rust FFI
        }
    }
}

// MARK: - Model Information

extension TFKModel {
    
    /// Model information structure
    public struct ModelInfo {
        public let name: String
        public let version: String?
        public let author: String?
        public let description: String?
        public let inputShape: [Int]?
        public let outputShape: [Int]?
        public let parameters: Int?
        public let flops: Int?
    }
    
    /// Get model information
    public func getModelInfo() -> ModelInfo {
        return ModelInfo(
            name: metadata["name"] as? String ?? "Unknown",
            version: metadata["version"] as? String,
            author: metadata["author"] as? String,
            description: metadata["description"] as? String,
            inputShape: metadata["inputShape"] as? [Int],
            outputShape: metadata["outputShape"] as? [Int],
            parameters: metadata["parameters"] as? Int,
            flops: metadata["flops"] as? Int
        )
    }
}

// MARK: - Core ML Integration

@available(iOS 11.0, *)
extension TFKModel {
    
    /// Get Core ML model if available
    public var mlModel: MLModel? {
        return coreMLModel
    }
    
    /// Perform Core ML prediction
    public func predictWithCoreML(input: MLFeatureProvider) throws -> MLFeatureProvider {
        guard let model = coreMLModel else {
            throw TFKError.modelNotFound(path: "Core ML model not loaded")
        }
        
        return try model.prediction(from: input)
    }
}