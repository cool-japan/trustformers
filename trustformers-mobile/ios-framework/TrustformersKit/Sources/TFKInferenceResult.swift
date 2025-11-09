//
//  TFKInferenceResult.swift
//  TrustformersKit
//
//  Inference result representation
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import Foundation

/// Result of model inference
public class TFKInferenceResult: NSObject {
    
    // MARK: - Properties
    
    /// Output tensor
    public let output: TFKTensor
    
    /// Inference time in seconds
    public let inferenceTime: TimeInterval
    
    /// Memory used in MB
    public let memoryUsed: Int
    
    /// Success status
    public let success: Bool
    
    /// Error if inference failed
    public let error: Error?
    
    /// Additional metadata
    public private(set) var metadata: [String: Any] = [:]
    
    // MARK: - Initialization
    
    /// Initialize with output and metrics
    public init(output: TFKTensor,
                inferenceTime: TimeInterval,
                memoryUsed: Int,
                success: Bool,
                error: Error? = nil) {
        self.output = output
        self.inferenceTime = inferenceTime
        self.memoryUsed = memoryUsed
        self.success = success
        self.error = error
        
        super.init()
        
        // Add basic metadata
        metadata["timestamp"] = Date()
        metadata["inferenceTimeMs"] = inferenceTime * 1000
        metadata["memoryUsedMB"] = memoryUsed
    }
    
    // MARK: - Convenience Properties
    
    /// Inference time in milliseconds
    public var inferenceTimeMs: Double {
        return inferenceTime * 1000
    }
    
    /// Get error description if failed
    public var errorDescription: String? {
        return error?.localizedDescription
    }
    
    // MARK: - Result Analysis
    
    /// Get top K predictions (for classification tasks)
    public func topKPredictions(k: Int = 5, labels: [String]? = nil) -> [(index: Int, score: Float, label: String?)] {
        guard let scores = output.floatData() else {
            return []
        }
        
        // Create indexed array
        let indexed = scores.enumerated().map { (index: $0.offset, score: $0.element) }
        
        // Sort by score descending and take top K
        let topK = indexed.sorted { $0.score > $1.score }.prefix(k)
        
        // Map to result with optional labels
        return topK.map { item in
            let label = labels != nil && item.index < labels!.count ? labels![item.index] : nil
            return (index: item.index, score: item.score, label: label)
        }
    }
    
    /// Get classification result (single prediction)
    public func getClassification(labels: [String]? = nil) -> (index: Int, score: Float, label: String?) {
        let topOne = topKPredictions(k: 1, labels: labels)
        return topOne.first ?? (index: -1, score: 0, label: nil)
    }
    
    /// Apply softmax to output
    public func applySoftmax(axis: Int = -1) throws -> TFKInferenceResult {
        let softmaxOutput = try output.softmax(axis: axis)
        
        return TFKInferenceResult(
            output: softmaxOutput,
            inferenceTime: inferenceTime,
            memoryUsed: memoryUsed,
            success: success,
            error: error
        )
    }
    
    // MARK: - Metadata
    
    /// Add custom metadata
    public func addMetadata(key: String, value: Any) {
        metadata[key] = value
    }
    
    /// Get metadata value
    public func getMetadata(key: String) -> Any? {
        return metadata[key]
    }
    
    // MARK: - Export
    
    /// Export result as dictionary
    public func exportResult() -> [String: Any] {
        var result: [String: Any] = [
            "success": success,
            "inferenceTimeMs": inferenceTimeMs,
            "memoryUsedMB": memoryUsed,
            "outputShape": output.shape,
            "outputElementCount": output.elementCount
        ]
        
        // Add error info if present
        if let error = error {
            result["error"] = error.localizedDescription
        }
        
        // Add metadata
        result["metadata"] = metadata
        
        return result
    }
    
    /// Export result as JSON
    public func exportJSON() throws -> Data {
        let dict = exportResult()
        return try JSONSerialization.data(withJSONObject: dict, options: .prettyPrinted)
    }
}

// MARK: - Result Aggregation

extension TFKInferenceResult {
    
    /// Aggregate multiple results (for batch processing)
    public static func aggregate(_ results: [TFKInferenceResult]) -> AggregatedResult {
        guard !results.isEmpty else {
            return AggregatedResult(
                totalInferences: 0,
                successCount: 0,
                failureCount: 0,
                totalInferenceTime: 0,
                averageInferenceTime: 0,
                totalMemoryUsed: 0,
                averageMemoryUsed: 0
            )
        }
        
        let totalInferences = results.count
        let successCount = results.filter { $0.success }.count
        let failureCount = totalInferences - successCount
        
        let totalInferenceTime = results.reduce(0) { $0 + $1.inferenceTime }
        let averageInferenceTime = totalInferenceTime / Double(totalInferences)
        
        let totalMemoryUsed = results.reduce(0) { $0 + $1.memoryUsed }
        let averageMemoryUsed = totalMemoryUsed / totalInferences
        
        return AggregatedResult(
            totalInferences: totalInferences,
            successCount: successCount,
            failureCount: failureCount,
            totalInferenceTime: totalInferenceTime,
            averageInferenceTime: averageInferenceTime,
            totalMemoryUsed: totalMemoryUsed,
            averageMemoryUsed: averageMemoryUsed
        )
    }
    
    /// Aggregated result structure
    public struct AggregatedResult {
        public let totalInferences: Int
        public let successCount: Int
        public let failureCount: Int
        public let totalInferenceTime: TimeInterval
        public let averageInferenceTime: TimeInterval
        public let totalMemoryUsed: Int
        public let averageMemoryUsed: Int
        
        public var successRate: Double {
            guard totalInferences > 0 else { return 0 }
            return Double(successCount) / Double(totalInferences)
        }
        
        public var description: String {
            return """
            Aggregated Results:
            - Total Inferences: \(totalInferences)
            - Success Rate: \(String(format: "%.1f", successRate * 100))%
            - Average Time: \(String(format: "%.2f", averageInferenceTime * 1000)) ms
            - Average Memory: \(averageMemoryUsed) MB
            """
        }
    }
}

// MARK: - Result Validation

extension TFKInferenceResult {
    
    /// Validate output shape
    public func validateShape(expected: [Int]) -> Bool {
        return output.shape == expected
    }
    
    /// Validate output range
    public func validateRange(min: Float, max: Float) -> Bool {
        guard let data = output.floatData() else { return false }
        
        for value in data {
            if value < min || value > max {
                return false
            }
        }
        
        return true
    }
    
    /// Validate probability distribution (sums to 1)
    public func validateProbabilityDistribution(tolerance: Float = 0.001) -> Bool {
        guard let data = output.floatData() else { return false }
        
        let sum = data.reduce(0, +)
        return abs(sum - 1.0) < tolerance
    }
}