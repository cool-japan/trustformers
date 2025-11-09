//
//  TFKModelConfig.swift
//  TrustformersKit
//
//  Model configuration for TrustformersKit
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import Foundation

/// Configuration for TrustformersKit models
public class TFKModelConfig: NSObject, NSCopying {
    
    // MARK: - Properties
    
    /// Inference backend
    public var backend: TFKModelBackend = .cpu
    
    /// Memory optimization level
    public var memoryOptimization: TFKMemoryOptimization = .balanced
    
    /// Maximum memory usage in MB
    public var maxMemoryMB: Int = 512
    
    /// Use FP16 precision
    public var useFP16: Bool = true
    
    /// Enable quantization
    public var enableQuantization: Bool = true
    
    /// Quantization scheme
    public var quantizationScheme: TFKQuantizationScheme = .int8
    
    /// Number of threads (0 = auto-detect)
    public var numThreads: Int = 0
    
    /// Enable batch processing
    public var enableBatching: Bool = false
    
    /// Maximum batch size
    public var maxBatchSize: Int = 1
    
    // MARK: - Factory Methods
    
    /// Create default configuration
    public static func defaultConfig() -> TFKModelConfig {
        return TFKModelConfig()
    }
    
    /// Create optimized configuration for current device
    public static func optimizedConfig() -> TFKModelConfig {
        let config = TFKModelConfig()
        let deviceInfo = TFKDeviceInfo.currentDevice()
        
        // Optimize based on device capabilities
        if deviceInfo.hasNeuralEngine && deviceInfo.hasCoreML {
            config.backend = .coreML
        } else if deviceInfo.hasGPU {
            config.backend = .gpu
        } else {
            config.backend = .cpu
        }
        
        // Set memory limit based on available memory
        let availableMemory = deviceInfo.availableMemoryMB
        config.maxMemoryMB = min(availableMemory / 4, 1024) // Use up to 25% of available memory
        
        // Enable optimizations based on device
        if deviceInfo.hasNeuralEngine {
            config.useFP16 = true
            config.enableQuantization = true
            config.quantizationScheme = .fp16
        } else if deviceInfo.totalMemoryMB >= 4096 {
            // High-end device
            config.useFP16 = true
            config.enableBatching = true
            config.maxBatchSize = 4
        } else {
            // Lower-end device
            config.enableQuantization = true
            config.quantizationScheme = .int8
            config.memoryOptimization = .balanced
        }
        
        return config
    }
    
    /// Create ultra-low memory configuration
    public static func ultraLowMemoryConfig() -> TFKModelConfig {
        let config = TFKModelConfig()
        config.memoryOptimization = .maximum
        config.maxMemoryMB = 256
        config.useFP16 = true
        config.enableQuantization = true
        config.quantizationScheme = .int4
        config.numThreads = 1
        config.enableBatching = false
        return config
    }
    
    // MARK: - Initialization
    
    public override init() {
        super.init()
    }
    
    // MARK: - NSCopying
    
    public func copy(with zone: NSZone? = nil) -> Any {
        let copy = TFKModelConfig()
        copy.backend = backend
        copy.memoryOptimization = memoryOptimization
        copy.maxMemoryMB = maxMemoryMB
        copy.useFP16 = useFP16
        copy.enableQuantization = enableQuantization
        copy.quantizationScheme = quantizationScheme
        copy.numThreads = numThreads
        copy.enableBatching = enableBatching
        copy.maxBatchSize = maxBatchSize
        return copy
    }
    
    // MARK: - Validation
    
    /// Validate configuration
    public func validate() throws {
        // Check memory constraints
        guard maxMemoryMB >= 64 else {
            throw TFKError.invalidConfiguration(reason: "Mobile deployment requires at least 64MB memory")
        }
        
        guard maxMemoryMB <= 4096 else {
            throw TFKError.invalidConfiguration(reason: "Mobile deployment should not exceed 4GB memory")
        }
        
        // Validate platform-backend compatibility
        let deviceInfo = TFKDeviceInfo.currentDevice()
        
        switch backend {
        case .coreML:
            guard deviceInfo.hasCoreML else {
                throw TFKError.invalidConfiguration(reason: "Core ML is not available on this device")
            }
        case .gpu:
            guard deviceInfo.hasGPU else {
                throw TFKError.invalidConfiguration(reason: "GPU is not available on this device")
            }
        default:
            break
        }
        
        // Validate batch size
        if enableBatching {
            guard maxBatchSize > 0 else {
                throw TFKError.invalidConfiguration(reason: "Batch size must be > 0 when batching is enabled")
            }
            
            guard maxBatchSize <= 32 else {
                throw TFKError.invalidConfiguration(reason: "Batch size should not exceed 32 on mobile")
            }
        }
        
        // Validate thread count
        guard numThreads <= 16 else {
            throw TFKError.invalidConfiguration(reason: "Mobile deployment should not use more than 16 threads")
        }
    }
    
    // MARK: - Memory Estimation
    
    /// Estimate memory usage for model with given size
    public func estimateMemoryUsage(forModelSizeMB modelSizeMB: Int) -> Int {
        var totalMemory = modelSizeMB
        
        // Apply quantization reduction
        if enableQuantization {
            let reductionFactor: Int
            switch quantizationScheme {
            case .int4:
                reductionFactor = 8 // 4-bit = 1/8 of FP32
            case .int8:
                reductionFactor = 4 // 8-bit = 1/4 of FP32
            case .fp16:
                reductionFactor = 2 // 16-bit = 1/2 of FP32
            case .dynamic:
                reductionFactor = 3 // Approximate
            }
            totalMemory = modelSizeMB / reductionFactor
        } else if useFP16 {
            totalMemory = modelSizeMB / 2 // FP16 is half of FP32
        }
        
        // Add runtime overhead based on optimization level
        let overheadPercentage: Float
        switch memoryOptimization {
        case .minimal:
            overheadPercentage = 0.5 // 50% overhead
        case .balanced:
            overheadPercentage = 0.25 // 25% overhead
        case .maximum:
            overheadPercentage = 0.125 // 12.5% overhead
        }
        
        let overhead = Int(Float(totalMemory) * overheadPercentage)
        totalMemory += overhead
        
        // Add batch processing overhead
        if enableBatching && maxBatchSize > 1 {
            totalMemory += (totalMemory * (maxBatchSize - 1)) / 4 // Additional 25% per extra batch item
        }
        
        return totalMemory
    }
    
    // MARK: - Thread Configuration
    
    /// Get recommended thread count
    public func getThreadCount() -> Int {
        if numThreads > 0 {
            return numThreads
        }
        
        // Auto-detect based on device and optimization level
        let processorCount = ProcessInfo.processInfo.processorCount
        
        switch memoryOptimization {
        case .maximum:
            return 1 // Single-threaded for minimum memory
        case .balanced:
            return min(processorCount / 2, 4) // Half of available cores, max 4
        case .minimal:
            return min(processorCount, 8) // All cores, max 8
        }
    }
    
    // MARK: - C Interop
    
    /// Convert to C configuration structure
    internal func toCConfig() -> CTrustformersConfig {
        return CTrustformersConfig(
            platform: 0, // iOS = 0
            backend: UInt32(backend.rawValue),
            memory_optimization: UInt32(memoryOptimization.rawValue),
            max_memory_mb: UInt32(maxMemoryMB),
            use_fp16: useFP16,
            num_threads: UInt32(getThreadCount()),
            enable_batching: enableBatching,
            max_batch_size: UInt32(maxBatchSize)
        )
    }
    
    /// Create from C configuration structure
    internal static func fromCConfig(_ cConfig: CTrustformersConfig) -> TFKModelConfig {
        let config = TFKModelConfig()
        config.backend = TFKModelBackend(rawValue: Int(cConfig.backend)) ?? .cpu
        config.memoryOptimization = TFKMemoryOptimization(rawValue: Int(cConfig.memory_optimization)) ?? .balanced
        config.maxMemoryMB = Int(cConfig.max_memory_mb)
        config.useFP16 = cConfig.use_fp16
        config.numThreads = Int(cConfig.num_threads)
        config.enableBatching = cConfig.enable_batching
        config.maxBatchSize = Int(cConfig.max_batch_size)
        return config
    }
}

// MARK: - Configuration Presets

extension TFKModelConfig {
    
    /// Configuration optimized for text generation
    public static func textGenerationConfig() -> TFKModelConfig {
        let config = optimizedConfig()
        config.enableBatching = false // Sequential generation
        config.memoryOptimization = .balanced
        return config
    }
    
    /// Configuration optimized for image classification
    public static func imageClassificationConfig() -> TFKModelConfig {
        let config = optimizedConfig()
        config.enableBatching = true
        config.maxBatchSize = 4
        config.useFP16 = true
        return config
    }
    
    /// Configuration optimized for real-time inference
    public static func realTimeConfig() -> TFKModelConfig {
        let config = TFKModelConfig()
        config.memoryOptimization = .minimal
        config.numThreads = ProcessInfo.processInfo.processorCount
        config.enableBatching = false
        config.useFP16 = true
        return config
    }
    
    /// Configuration for background processing
    public static func backgroundProcessingConfig() -> TFKModelConfig {
        let config = TFKModelConfig()
        config.memoryOptimization = .maximum
        config.numThreads = 1
        config.enableQuantization = true
        config.quantizationScheme = .int8
        return config
    }
}

// MARK: - Debugging

extension TFKModelConfig {
    
    /// Get configuration summary
    public func configurationSummary() -> String {
        return """
        TrustformersKit Model Configuration:
        - Backend: \(backend)
        - Memory Optimization: \(memoryOptimization)
        - Max Memory: \(maxMemoryMB) MB
        - FP16: \(useFP16)
        - Quantization: \(enableQuantization ? "\(quantizationScheme)" : "Disabled")
        - Threads: \(getThreadCount())
        - Batching: \(enableBatching ? "Enabled (max: \(maxBatchSize))" : "Disabled")
        - Estimated Memory for 100MB model: \(estimateMemoryUsage(forModelSizeMB: 100)) MB
        """
    }
}