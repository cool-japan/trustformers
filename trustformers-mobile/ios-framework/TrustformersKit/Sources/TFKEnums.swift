//
//  TFKEnums.swift
//  TrustformersKit
//
//  Core enums and types for TrustformersKit
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import Foundation

// MARK: - Model Backend

/// Model inference backend types
@objc public enum TFKModelBackend: Int, CaseIterable {
    /// CPU-only inference
    case cpu = 0
    
    /// Core ML (iOS)
    case coreML = 1
    
    /// Neural Networks API (Android)
    case nnapi = 2
    
    /// GPU acceleration (Metal/Vulkan)
    case gpu = 3
    
    /// Custom optimized backend
    case custom = 4
    
    /// Human-readable description
    public var description: String {
        switch self {
        case .cpu:
            return "CPU"
        case .coreML:
            return "Core ML"
        case .nnapi:
            return "NNAPI"
        case .gpu:
            return "GPU"
        case .custom:
            return "Custom"
        }
    }
    
    /// Check if backend is available on current platform
    public var isAvailable: Bool {
        let deviceInfo = TFKDeviceInfo.currentDevice()
        
        switch self {
        case .cpu:
            return true // Always available
        case .coreML:
            return deviceInfo.hasCoreML
        case .nnapi:
            return false // Not available on iOS
        case .gpu:
            return deviceInfo.hasGPU
        case .custom:
            return true // Assume available
        }
    }
}

// MARK: - Memory Optimization

/// Memory optimization levels for mobile deployment
@objc public enum TFKMemoryOptimization: Int, CaseIterable {
    /// Minimal optimization (fastest inference)
    case minimal = 0
    
    /// Balanced optimization (default)
    case balanced = 1
    
    /// Maximum optimization (lowest memory usage)
    case maximum = 2
    
    /// Human-readable description
    public var description: String {
        switch self {
        case .minimal:
            return "Minimal"
        case .balanced:
            return "Balanced"
        case .maximum:
            return "Maximum"
        }
    }
    
    /// Memory overhead percentage
    public var memoryOverhead: Float {
        switch self {
        case .minimal:
            return 0.5 // 50%
        case .balanced:
            return 0.25 // 25%
        case .maximum:
            return 0.125 // 12.5%
        }
    }
    
    /// Recommended thread count multiplier
    public var threadMultiplier: Float {
        switch self {
        case .minimal:
            return 1.0
        case .balanced:
            return 0.5
        case .maximum:
            return 0.25
        }
    }
}

// MARK: - Quantization Scheme

/// Quantization schemes for model optimization
@objc public enum TFKQuantizationScheme: Int, CaseIterable {
    /// 8-bit integer quantization
    case int8 = 0
    
    /// 4-bit integer quantization (ultra-low memory)
    case int4 = 1
    
    /// Float16 quantization
    case fp16 = 2
    
    /// Dynamic quantization
    case dynamic = 3
    
    /// Human-readable description
    public var description: String {
        switch self {
        case .int8:
            return "Int8"
        case .int4:
            return "Int4"
        case .fp16:
            return "FP16"
        case .dynamic:
            return "Dynamic"
        }
    }
    
    /// Memory reduction factor compared to FP32
    public var reductionFactor: Int {
        switch self {
        case .int8:
            return 4 // 8-bit = 1/4 of FP32
        case .int4:
            return 8 // 4-bit = 1/8 of FP32
        case .fp16:
            return 2 // 16-bit = 1/2 of FP32
        case .dynamic:
            return 3 // Approximate
        }
    }
    
    /// Quality vs compression trade-off (0.0 = highest compression, 1.0 = highest quality)
    public var qualityRatio: Float {
        switch self {
        case .int4:
            return 0.6
        case .int8:
            return 0.8
        case .fp16:
            return 0.95
        case .dynamic:
            return 0.85
        }
    }
}

// MARK: - Thermal State

/// Device thermal states for throttling management
@objc public enum TFKThermalState: Int, CaseIterable {
    /// Normal operating temperature
    case nominal = 0
    
    /// Slightly elevated temperature
    case fair = 1
    
    /// High temperature, may affect performance
    case serious = 2
    
    /// Critical temperature, performance severely limited
    case critical = 3
    
    /// Human-readable description
    public var description: String {
        switch self {
        case .nominal:
            return "Nominal"
        case .fair:
            return "Fair"
        case .serious:
            return "Serious"
        case .critical:
            return "Critical"
        }
    }
    
    /// Recommended performance scaling factor
    public var performanceScale: Float {
        switch self {
        case .nominal:
            return 1.0
        case .fair:
            return 0.8
        case .serious:
            return 0.5
        case .critical:
            return 0.2
        }
    }
    
    /// Should throttle inference
    public var shouldThrottle: Bool {
        return rawValue >= TFKThermalState.serious.rawValue
    }
}

// MARK: - Log Level

/// Logging levels for TrustformersKit
@objc public enum TFKLogLevel: UInt, CaseIterable {
    /// Verbose debugging information
    case trace = 0
    
    /// Debug information
    case debug = 1
    
    /// General information
    case info = 2
    
    /// Warning messages
    case warn = 3
    
    /// Error messages
    case error = 4
    
    /// Human-readable description
    public var description: String {
        switch self {
        case .trace:
            return "TRACE"
        case .debug:
            return "DEBUG"
        case .info:
            return "INFO"
        case .warn:
            return "WARN"
        case .error:
            return "ERROR"
        }
    }
    
    /// Log level prefix for formatted output
    public var prefix: String {
        switch self {
        case .trace:
            return "🔍"
        case .debug:
            return "🐛"
        case .info:
            return "ℹ️"
        case .warn:
            return "⚠️"
        case .error:
            return "❌"
        }
    }
}

// MARK: - Platform Types

/// Mobile platform types
@objc public enum TFKPlatform: Int, CaseIterable {
    /// iOS platform
    case iOS = 0
    
    /// Android platform
    case android = 1
    
    /// Generic mobile platform
    case generic = 2
    
    /// Current platform
    public static var current: TFKPlatform {
        #if os(iOS)
        return .iOS
        #elseif os(Android)
        return .android
        #else
        return .generic
        #endif
    }
    
    /// Human-readable description
    public var description: String {
        switch self {
        case .iOS:
            return "iOS"
        case .android:
            return "Android"
        case .generic:
            return "Generic"
        }
    }
}

// MARK: - Error Types

/// TrustformersKit specific error types
@objc public enum TFKErrorType: Int, CaseIterable {
    /// Engine not initialized
    case engineNotInitialized = 0
    
    /// Model not found
    case modelNotFound = 1
    
    /// Model load failed
    case modelLoadFailed = 2
    
    /// Invalid configuration
    case invalidConfiguration = 3
    
    /// Inference failed
    case inferenceFailed = 4
    
    /// Out of memory
    case outOfMemory = 5
    
    /// Unsupported backend
    case unsupportedBackend = 6
    
    /// Tensor shape mismatch
    case tensorShapeMismatch = 7
    
    /// Quantization failed
    case quantizationFailed = 8
    
    /// Thermal throttling
    case thermalThrottling = 9
    
    /// Network error
    case networkError = 10
    
    /// File system error
    case fileSystemError = 11
    
    /// Human-readable description
    public var description: String {
        switch self {
        case .engineNotInitialized:
            return "Engine Not Initialized"
        case .modelNotFound:
            return "Model Not Found"
        case .modelLoadFailed:
            return "Model Load Failed"
        case .invalidConfiguration:
            return "Invalid Configuration"
        case .inferenceFailed:
            return "Inference Failed"
        case .outOfMemory:
            return "Out of Memory"
        case .unsupportedBackend:
            return "Unsupported Backend"
        case .tensorShapeMismatch:
            return "Tensor Shape Mismatch"
        case .quantizationFailed:
            return "Quantization Failed"
        case .thermalThrottling:
            return "Thermal Throttling"
        case .networkError:
            return "Network Error"
        case .fileSystemError:
            return "File System Error"
        }
    }
}

// MARK: - Extensions

extension TFKModelBackend: CustomStringConvertible {}
extension TFKMemoryOptimization: CustomStringConvertible {}
extension TFKQuantizationScheme: CustomStringConvertible {}
extension TFKThermalState: CustomStringConvertible {}
extension TFKLogLevel: CustomStringConvertible {}
extension TFKPlatform: CustomStringConvertible {}
extension TFKErrorType: CustomStringConvertible {}

// MARK: - Utility Functions

public extension TFKModelBackend {
    /// Get optimal backend for current device
    static func optimal() -> TFKModelBackend {
        let deviceInfo = TFKDeviceInfo.currentDevice()
        
        if deviceInfo.hasNeuralEngine && deviceInfo.hasCoreML {
            return .coreML
        } else if deviceInfo.hasGPU {
            return .gpu
        } else {
            return .cpu
        }
    }
}

public extension TFKMemoryOptimization {
    /// Get recommended optimization level based on available memory
    static func recommended(availableMemoryMB: Int) -> TFKMemoryOptimization {
        if availableMemoryMB >= 4096 {
            return .minimal
        } else if availableMemoryMB >= 2048 {
            return .balanced
        } else {
            return .maximum
        }
    }
}

public extension TFKQuantizationScheme {
    /// Get recommended quantization scheme based on device capabilities
    static func recommended(hasNeuralEngine: Bool, availableMemoryMB: Int) -> TFKQuantizationScheme {
        if hasNeuralEngine {
            return .fp16
        } else if availableMemoryMB < 1024 {
            return .int4
        } else if availableMemoryMB < 2048 {
            return .int8
        } else {
            return .fp16
        }
    }
}