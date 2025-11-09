//
//  TFKDeviceInfo.swift
//  TrustformersKit
//
//  Device information and capabilities detection
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import Foundation
import UIKit

/// Device information and capabilities
public class TFKDeviceInfo: NSObject {
    
    // MARK: - Properties
    
    /// Device model identifier
    public private(set) var deviceModel: String = ""
    
    /// iOS version
    public private(set) var systemVersion: String = ""
    
    /// Total memory in MB
    public private(set) var totalMemoryMB: Int = 0
    
    /// Available memory in MB
    public var availableMemoryMB: Int {
        return Int(getAvailableMemory() / (1024 * 1024))
    }
    
    /// Number of CPU cores
    public private(set) var cpuCores: Int = 0
    
    /// GPU availability
    public private(set) var hasGPU: Bool = false
    
    /// Core ML availability
    public private(set) var hasCoreML: Bool = false
    
    /// Neural Engine availability
    public private(set) var hasNeuralEngine: Bool = false
    
    /// Current thermal state
    public private(set) var thermalState: TFKThermalState = .nominal
    
    /// Battery level (0.0 - 1.0)
    public var batteryLevel: Float {
        UIDevice.current.isBatteryMonitoringEnabled = true
        return UIDevice.current.batteryLevel
    }
    
    /// Charging status
    public var isCharging: Bool {
        UIDevice.current.isBatteryMonitoringEnabled = true
        return UIDevice.current.batteryState == .charging || UIDevice.current.batteryState == .full
    }
    
    // MARK: - Singleton
    
    private static var _current: TFKDeviceInfo?
    
    /// Get current device info
    public static func currentDevice() -> TFKDeviceInfo {
        if _current == nil {
            _current = TFKDeviceInfo()
            _current?.detectCapabilities()
        }
        return _current!
    }
    
    // MARK: - Initialization
    
    private override init() {
        super.init()
    }
    
    // MARK: - Detection
    
    private func detectCapabilities() {
        // Device model
        deviceModel = getDeviceModel()
        
        // System version
        systemVersion = UIDevice.current.systemVersion
        
        // Memory
        totalMemoryMB = Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024))
        
        // CPU cores
        cpuCores = ProcessInfo.processInfo.processorCount
        
        // GPU availability (all iOS devices have GPU)
        hasGPU = true
        
        // Core ML availability (iOS 11+)
        if #available(iOS 11.0, *) {
            hasCoreML = true
        }
        
        // Neural Engine detection
        hasNeuralEngine = detectNeuralEngine()
        
        // Thermal state
        updateThermalState()
    }
    
    private func getDeviceModel() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let modelCode = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }
        
        return modelCode ?? "Unknown"
    }
    
    private func detectNeuralEngine() -> Bool {
        // Neural Engine is available on A11 Bionic and later
        // This is a simplified detection based on device model
        let neuralEngineModels = [
            // iPhone
            "iPhone10,3", "iPhone10,6", // iPhone X
            "iPhone11,", "iPhone12,", "iPhone13,", "iPhone14,", "iPhone15,", // iPhone XS and later
            // iPad
            "iPad8,", "iPad11,", "iPad12,", "iPad13,", "iPad14,", // iPad Pro 3rd gen and later
            "iPad7,11", "iPad7,12", // iPad 7th gen and later
        ]
        
        return neuralEngineModels.contains { deviceModel.hasPrefix($0) }
    }
    
    private func getAvailableMemory() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let usedMemory = Int64(info.resident_size)
            let totalMemory = ProcessInfo.processInfo.physicalMemory
            return Int64(totalMemory) - usedMemory
        }
        
        return 0
    }
    
    // MARK: - Public Methods
    
    /// Check if feature is supported
    public func supportsFeature(_ feature: String) -> Bool {
        switch feature.lowercased() {
        case "coreml":
            return hasCoreML
        case "neuralengine", "neural-engine", "ane":
            return hasNeuralEngine
        case "gpu", "metal":
            return hasGPU
        case "fp16":
            return true // All recent iOS devices support FP16
        case "int8":
            return true // Software support for INT8 quantization
        case "batch":
            return totalMemoryMB >= 2048 // Batching recommended for 2GB+ devices
        default:
            return false
        }
    }
    
    /// Get recommended configuration for device
    public func recommendedConfig() -> TFKModelConfig {
        let config = TFKModelConfig()
        
        // Select backend based on capabilities
        if hasNeuralEngine && hasCoreML {
            config.backend = .coreML
        } else if hasGPU {
            config.backend = .gpu
        } else {
            config.backend = .cpu
        }
        
        // Memory configuration
        let availableMemory = availableMemoryMB
        if availableMemory < 512 {
            // Low memory device
            config.memoryOptimization = .maximum
            config.maxMemoryMB = 256
            config.enableQuantization = true
            config.quantizationScheme = .int8
        } else if availableMemory < 1024 {
            // Medium memory device
            config.memoryOptimization = .balanced
            config.maxMemoryMB = 512
            config.useFP16 = true
        } else {
            // High memory device
            config.memoryOptimization = .minimal
            config.maxMemoryMB = 1024
            config.useFP16 = true
            config.enableBatching = true
            config.maxBatchSize = 4
        }
        
        // Thread configuration
        if thermalState == .critical || thermalState == .serious {
            config.numThreads = 1
        } else {
            config.numThreads = min(cpuCores / 2, 4)
        }
        
        // Battery optimization
        if !isCharging && batteryLevel < 0.2 {
            config.memoryOptimization = .maximum
            config.numThreads = 1
        }
        
        return config
    }
    
    /// Update thermal state
    internal func updateThermalState() {
        if #available(iOS 11.0, *) {
            let processInfo = ProcessInfo.processInfo
            updateThermalState(processInfo.thermalState)
        }
    }
    
    @available(iOS 11.0, *)
    internal func updateThermalState(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal:
            thermalState = .nominal
        case .fair:
            thermalState = .fair
        case .serious:
            thermalState = .serious
        case .critical:
            thermalState = .critical
        @unknown default:
            thermalState = .nominal
        }
    }
    
    // MARK: - Device Categories
    
    /// Device performance tier
    public enum PerformanceTier {
        case low
        case medium
        case high
        case flagship
    }
    
    /// Get device performance tier
    public func performanceTier() -> PerformanceTier {
        // Based on Neural Engine, memory, and device model
        if hasNeuralEngine && totalMemoryMB >= 6144 {
            return .flagship
        } else if hasNeuralEngine || totalMemoryMB >= 4096 {
            return .high
        } else if totalMemoryMB >= 2048 {
            return .medium
        } else {
            return .low
        }
    }
    
    /// Get device category description
    public func deviceCategory() -> String {
        let tier = performanceTier()
        switch tier {
        case .flagship:
            return "Flagship Device (Pro models, latest generation)"
        case .high:
            return "High-End Device (Recent models with Neural Engine)"
        case .medium:
            return "Mid-Range Device (2-4GB RAM)"
        case .low:
            return "Entry-Level Device (Limited resources)"
        }
    }
}

// MARK: - Device Info Summary

extension TFKDeviceInfo {
    
    /// Get device information summary
    public func deviceSummary() -> String {
        return """
        Device Information:
        - Model: \(deviceModel)
        - iOS Version: \(systemVersion)
        - Performance Tier: \(performanceTier())
        - Total Memory: \(totalMemoryMB) MB
        - Available Memory: \(availableMemoryMB) MB
        - CPU Cores: \(cpuCores)
        - GPU: \(hasGPU ? "Available" : "Not Available")
        - Core ML: \(hasCoreML ? "Available" : "Not Available")
        - Neural Engine: \(hasNeuralEngine ? "Available" : "Not Available")
        - Thermal State: \(thermalState)
        - Battery: \(Int(batteryLevel * 100))% \(isCharging ? "(Charging)" : "")
        """
    }
    
    /// Get capabilities summary
    public func capabilitiesSummary() -> String {
        var capabilities: [String] = []
        
        if hasGPU { capabilities.append("GPU/Metal") }
        if hasCoreML { capabilities.append("Core ML") }
        if hasNeuralEngine { capabilities.append("Neural Engine") }
        if supportsFeature("fp16") { capabilities.append("FP16") }
        if supportsFeature("int8") { capabilities.append("INT8") }
        if supportsFeature("batch") { capabilities.append("Batch Processing") }
        
        return capabilities.joined(separator: ", ")
    }
}