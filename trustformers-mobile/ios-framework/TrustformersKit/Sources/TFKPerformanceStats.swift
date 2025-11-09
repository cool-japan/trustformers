//
//  TFKPerformanceStats.swift
//  TrustformersKit
//
//  Performance statistics tracking
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import Foundation

/// Performance statistics for inference
public class TFKPerformanceStats: NSObject {
    
    // MARK: - Properties
    
    /// Total number of inferences performed
    public private(set) var totalInferences: Int = 0
    
    /// Average inference time in seconds
    public private(set) var averageInferenceTime: TimeInterval = 0
    
    /// Minimum inference time
    public private(set) var minInferenceTime: TimeInterval = Double.greatestFiniteMagnitude
    
    /// Maximum inference time
    public private(set) var maxInferenceTime: TimeInterval = 0
    
    /// Current memory usage in MB
    public private(set) var currentMemoryUsageMB: Int = 0
    
    /// Peak memory usage in MB
    public private(set) var peakMemoryUsageMB: Int = 0
    
    /// Average CPU usage percentage
    public private(set) var averageCPUUsage: Float = 0
    
    /// Average GPU usage percentage
    public private(set) var averageGPUUsage: Float = 0
    
    // Private tracking properties
    private var inferenceTimesSum: TimeInterval = 0
    private var cpuUsageSum: Float = 0
    private var gpuUsageSum: Float = 0
    private let statsQueue = DispatchQueue(label: "com.trustformers.stats", qos: .utility)
    
    // Sliding window for recent performance
    private var recentInferenceTimes: [TimeInterval] = []
    private let recentWindowSize = 100
    
    // MARK: - Initialization
    
    public override init() {
        super.init()
        startMonitoring()
    }
    
    // MARK: - Recording Methods
    
    /// Record inference completion
    public func recordInference(time: TimeInterval, memoryUsed: Int) {
        statsQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Update counters
            self.totalInferences += 1
            self.inferenceTimesSum += time
            
            // Update min/max
            self.minInferenceTime = min(self.minInferenceTime, time)
            self.maxInferenceTime = max(self.maxInferenceTime, time)
            
            // Update average
            self.averageInferenceTime = self.inferenceTimesSum / Double(self.totalInferences)
            
            // Update memory
            self.currentMemoryUsageMB = memoryUsed
            self.peakMemoryUsageMB = max(self.peakMemoryUsageMB, memoryUsed)
            
            // Track recent times for sliding window
            self.recentInferenceTimes.append(time)
            if self.recentInferenceTimes.count > self.recentWindowSize {
                self.recentInferenceTimes.removeFirst()
            }
            
            // Update CPU/GPU usage
            self.updateResourceUsage()
        }
    }
    
    /// Reset all statistics
    public func reset() {
        statsQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.totalInferences = 0
            self.averageInferenceTime = 0
            self.minInferenceTime = Double.greatestFiniteMagnitude
            self.maxInferenceTime = 0
            self.currentMemoryUsageMB = 0
            self.peakMemoryUsageMB = 0
            self.averageCPUUsage = 0
            self.averageGPUUsage = 0
            self.inferenceTimesSum = 0
            self.cpuUsageSum = 0
            self.gpuUsageSum = 0
            self.recentInferenceTimes.removeAll()
        }
    }
    
    // MARK: - Statistics
    
    /// Get performance summary
    public func performanceSummary() -> String {
        return statsQueue.sync {
            let avgTimeMs = averageInferenceTime * 1000
            let minTimeMs = minInferenceTime == Double.greatestFiniteMagnitude ? 0 : minInferenceTime * 1000
            let maxTimeMs = maxInferenceTime * 1000
            
            return """
            Performance Statistics:
            - Total Inferences: \(totalInferences)
            - Average Time: \(String(format: "%.2f", avgTimeMs)) ms
            - Min Time: \(String(format: "%.2f", minTimeMs)) ms
            - Max Time: \(String(format: "%.2f", maxTimeMs)) ms
            - Current Memory: \(currentMemoryUsageMB) MB
            - Peak Memory: \(peakMemoryUsageMB) MB
            - Average CPU: \(String(format: "%.1f", averageCPUUsage))%
            - Average GPU: \(String(format: "%.1f", averageGPUUsage))%
            - Throughput: \(String(format: "%.2f", getThroughput())) inferences/sec
            """
        }
    }
    
    /// Get recent average inference time
    public func getRecentAverageTime() -> TimeInterval {
        return statsQueue.sync {
            guard !recentInferenceTimes.isEmpty else { return 0 }
            let sum = recentInferenceTimes.reduce(0, +)
            return sum / Double(recentInferenceTimes.count)
        }
    }
    
    /// Get throughput (inferences per second)
    public func getThroughput() -> Double {
        return statsQueue.sync {
            guard averageInferenceTime > 0 else { return 0 }
            return 1.0 / averageInferenceTime
        }
    }
    
    /// Get performance percentiles
    public func getPercentiles() -> (p50: TimeInterval, p90: TimeInterval, p95: TimeInterval, p99: TimeInterval) {
        return statsQueue.sync {
            guard !recentInferenceTimes.isEmpty else {
                return (0, 0, 0, 0)
            }
            
            let sorted = recentInferenceTimes.sorted()
            let count = sorted.count
            
            let p50 = sorted[count / 2]
            let p90 = sorted[Int(Double(count) * 0.9)]
            let p95 = sorted[Int(Double(count) * 0.95)]
            let p99 = sorted[min(Int(Double(count) * 0.99), count - 1)]
            
            return (p50, p90, p95, p99)
        }
    }
    
    // MARK: - Private Methods
    
    private func startMonitoring() {
        // Start periodic resource monitoring
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateResourceUsage()
        }
    }
    
    private func updateResourceUsage() {
        // Update CPU usage
        let cpuUsage = getCurrentCPUUsage()
        cpuUsageSum += cpuUsage
        if totalInferences > 0 {
            averageCPUUsage = cpuUsageSum / Float(totalInferences)
        }
        
        // Update GPU usage (simplified - would need Metal performance counters)
        let gpuUsage = getCurrentGPUUsage()
        gpuUsageSum += gpuUsage
        if totalInferences > 0 {
            averageGPUUsage = gpuUsageSum / Float(totalInferences)
        }
        
        // Update current memory
        currentMemoryUsageMB = getCurrentMemoryUsage()
    }
    
    private func getCurrentCPUUsage() -> Float {
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
            // This is a simplified CPU usage calculation
            // In practice, you'd track thread time over wall time
            return Float(info.resident_size) / Float(ProcessInfo.processInfo.physicalMemory) * 100
        }
        
        return 0
    }
    
    private func getCurrentGPUUsage() -> Float {
        // Simplified GPU usage - would need Metal performance counters
        // Return 0 for now as proper implementation requires Metal integration
        return 0
    }
    
    private func getCurrentMemoryUsage() -> Int {
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
            return Int(info.resident_size / (1024 * 1024))
        }
        
        return 0
    }
}

// MARK: - Performance Tracking

extension TFKPerformanceStats {
    
    /// Performance snapshot at a point in time
    public struct Snapshot {
        public let timestamp: Date
        public let totalInferences: Int
        public let averageInferenceTime: TimeInterval
        public let currentMemoryUsageMB: Int
        public let cpuUsage: Float
        public let gpuUsage: Float
        
        public var description: String {
            let formatter = DateFormatter()
            formatter.dateStyle = .short
            formatter.timeStyle = .medium
            
            return """
            Snapshot at \(formatter.string(from: timestamp)):
            - Inferences: \(totalInferences)
            - Avg Time: \(String(format: "%.2f", averageInferenceTime * 1000)) ms
            - Memory: \(currentMemoryUsageMB) MB
            - CPU: \(String(format: "%.1f", cpuUsage))%
            - GPU: \(String(format: "%.1f", gpuUsage))%
            """
        }
    }
    
    /// Take a performance snapshot
    public func takeSnapshot() -> Snapshot {
        return statsQueue.sync {
            Snapshot(
                timestamp: Date(),
                totalInferences: totalInferences,
                averageInferenceTime: averageInferenceTime,
                currentMemoryUsageMB: currentMemoryUsageMB,
                cpuUsage: averageCPUUsage,
                gpuUsage: averageGPUUsage
            )
        }
    }
}

// MARK: - Export

extension TFKPerformanceStats {
    
    /// Export statistics as dictionary
    public func exportStatistics() -> [String: Any] {
        return statsQueue.sync {
            let percentiles = getPercentiles()
            
            return [
                "totalInferences": totalInferences,
                "averageInferenceTimeMs": averageInferenceTime * 1000,
                "minInferenceTimeMs": minInferenceTime * 1000,
                "maxInferenceTimeMs": maxInferenceTime * 1000,
                "p50InferenceTimeMs": percentiles.p50 * 1000,
                "p90InferenceTimeMs": percentiles.p90 * 1000,
                "p95InferenceTimeMs": percentiles.p95 * 1000,
                "p99InferenceTimeMs": percentiles.p99 * 1000,
                "currentMemoryUsageMB": currentMemoryUsageMB,
                "peakMemoryUsageMB": peakMemoryUsageMB,
                "averageCPUUsage": averageCPUUsage,
                "averageGPUUsage": averageGPUUsage,
                "throughput": getThroughput(),
                "timestamp": Date().timeIntervalSince1970
            ]
        }
    }
    
    /// Export statistics as JSON
    public func exportJSON() throws -> Data {
        let stats = exportStatistics()
        return try JSONSerialization.data(withJSONObject: stats, options: .prettyPrinted)
    }
}