//
//  TFKTensor.swift
//  TrustformersKit
//
//  Tensor representation for model inputs and outputs
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

/// Tensor representation for model inputs and outputs
public class TFKTensor: NSObject {
    
    // MARK: - Properties
    
    /// Tensor shape
    public let shape: [Int]
    
    /// Total number of elements
    public var elementCount: Int {
        return shape.reduce(1, *)
    }
    
    /// Size in bytes
    public var byteSize: Int {
        return elementCount * MemoryLayout<Float>.size
    }
    
    /// Underlying data
    private let data: Data
    
    /// C handle for FFI
    internal var cHandle: OpaquePointer?
    
    // MARK: - Initialization
    
    /// Initialize with float array and shape
    public init(floats: [Float], shape: [Int]) {
        precondition(floats.count == shape.reduce(1, *), "Data count must match shape")
        
        self.shape = shape
        self.data = floats.withUnsafeBytes { Data($0) }
        
        super.init()
        
        // Create C handle
        createCHandle(from: floats)
    }
    
    /// Initialize with data and shape
    public init(data: Data, shape: [Int]) {
        let expectedBytes = shape.reduce(1, *) * MemoryLayout<Float>.size
        precondition(data.count == expectedBytes, "Data size must match shape")
        
        self.shape = shape
        self.data = data
        
        super.init()
        
        // Create C handle
        data.withUnsafeBytes { bytes in
            if let floatPointer = bytes.bindMemory(to: Float.self).baseAddress {
                let floats = Array(UnsafeBufferPointer(start: floatPointer, count: elementCount))
                createCHandle(from: floats)
            }
        }
    }
    
    /// Initialize from Core ML MLMultiArray
    @available(iOS 11.0, *)
    public convenience init(mlMultiArray: MLMultiArray) throws {
        // Extract shape
        let shape = mlMultiArray.shape.map { $0.intValue }
        
        // Convert to float array
        let count = shape.reduce(1, *)
        var floats = [Float](repeating: 0, count: count)
        
        // Copy data based on data type
        switch mlMultiArray.dataType {
        case .float32:
            let ptr = mlMultiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
            floats = Array(UnsafeBufferPointer(start: ptr, count: count))
            
        case .double:
            let ptr = mlMultiArray.dataPointer.bindMemory(to: Double.self, capacity: count)
            let doubles = Array(UnsafeBufferPointer(start: ptr, count: count))
            floats = doubles.map { Float($0) }
            
        case .int32:
            let ptr = mlMultiArray.dataPointer.bindMemory(to: Int32.self, capacity: count)
            let ints = Array(UnsafeBufferPointer(start: ptr, count: count))
            floats = ints.map { Float($0) }
            
        default:
            throw TFKError.tensorShapeMismatch(expected: shape, actual: [])
        }
        
        self.init(floats: floats, shape: shape)
    }
    
    deinit {
        if let handle = cHandle {
            tfk_destroy_tensor(handle)
        }
    }
    
    // MARK: - Public Methods
    
    /// Get float data
    public func floatData() -> [Float]? {
        return data.withUnsafeBytes { bytes in
            guard let floatPointer = bytes.bindMemory(to: Float.self).baseAddress else {
                return nil
            }
            return Array(UnsafeBufferPointer(start: floatPointer, count: elementCount))
        }
    }
    
    /// Get data representation
    public func dataRepresentation() -> Data {
        return data
    }
    
    /// Convert to Core ML MLMultiArray
    @available(iOS 11.0, *)
    public func toMLMultiArray() throws -> MLMultiArray {
        let mlShape = shape.map { NSNumber(value: $0) }
        
        guard let mlArray = try? MLMultiArray(shape: mlShape, dataType: .float32) else {
            throw TFKError.tensorShapeMismatch(expected: shape, actual: [])
        }
        
        // Copy data
        if let floats = floatData() {
            let ptr = mlArray.dataPointer.bindMemory(to: Float.self, capacity: elementCount)
            floats.withUnsafeBufferPointer { buffer in
                ptr.initialize(from: buffer.baseAddress!, count: elementCount)
            }
        }
        
        return mlArray
    }
    
    /// Reshape tensor
    public func reshape(_ newShape: [Int]) throws -> TFKTensor {
        let newCount = newShape.reduce(1, *)
        guard newCount == elementCount else {
            throw TFKError.tensorShapeMismatch(expected: newShape, actual: shape)
        }
        
        guard let floats = floatData() else {
            throw TFKError.inferenceFailed(reason: "Failed to access tensor data")
        }
        
        return TFKTensor(floats: floats, shape: newShape)
    }
    
    /// Get value at index
    public func value(at indices: [Int]) -> Float? {
        guard indices.count == shape.count else { return nil }
        
        // Calculate flat index
        var flatIndex = 0
        var stride = 1
        
        for i in (0..<shape.count).reversed() {
            guard indices[i] >= 0 && indices[i] < shape[i] else { return nil }
            flatIndex += indices[i] * stride
            stride *= shape[i]
        }
        
        return floatData()?[flatIndex]
    }
    
    /// Apply softmax
    public func softmax(axis: Int = -1) throws -> TFKTensor {
        guard let floats = floatData() else {
            throw TFKError.inferenceFailed(reason: "Failed to access tensor data")
        }
        
        let actualAxis = axis < 0 ? shape.count + axis : axis
        guard actualAxis >= 0 && actualAxis < shape.count else {
            throw TFKError.inferenceFailed(reason: "Invalid axis for softmax")
        }
        
        var result = floats
        
        // Apply softmax using Accelerate
        if shape.count == 1 || (shape.count == 2 && actualAxis == 1) {
            // Simple case: 1D or 2D with axis=1
            let length = shape.last!
            let batches = elementCount / length
            
            for b in 0..<batches {
                let offset = b * length
                var slice = Array(result[offset..<offset+length])
                
                // Find max for numerical stability
                var maxVal: Float = 0
                vDSP_maxv(&slice, 1, &maxVal, vDSP_Length(length))
                
                // Subtract max and exp
                var negMax = -maxVal
                vDSP_vsadd(&slice, 1, &negMax, &slice, 1, vDSP_Length(length))
                vvexpf(&slice, &slice, [Int32(length)])
                
                // Sum
                var sum: Float = 0
                vDSP_sve(&slice, 1, &sum, vDSP_Length(length))
                
                // Divide by sum
                vDSP_vsdiv(&slice, 1, &sum, &slice, 1, vDSP_Length(length))
                
                // Copy back
                for i in 0..<length {
                    result[offset + i] = slice[i]
                }
            }
        }
        
        return TFKTensor(floats: result, shape: shape)
    }
    
    /// Apply argmax
    public func argmax(axis: Int = -1) -> [Int] {
        guard let floats = floatData() else { return [] }
        
        let actualAxis = axis < 0 ? shape.count + axis : axis
        guard actualAxis >= 0 && actualAxis < shape.count else { return [] }
        
        if shape.count == 1 {
            // 1D tensor
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(floats, 1, &maxVal, &maxIdx, vDSP_Length(elementCount))
            return [Int(maxIdx)]
        } else if shape.count == 2 && actualAxis == 1 {
            // 2D tensor, axis=1
            let rows = shape[0]
            let cols = shape[1]
            var indices = [Int](repeating: 0, count: rows)
            
            for r in 0..<rows {
                let offset = r * cols
                let slice = Array(floats[offset..<offset+cols])
                
                var maxVal: Float = 0
                var maxIdx: vDSP_Length = 0
                vDSP_maxvi(slice, 1, &maxVal, &maxIdx, vDSP_Length(cols))
                indices[r] = Int(maxIdx)
            }
            
            return indices
        }
        
        // For more complex cases, fall back to simple implementation
        return []
    }
    
    // MARK: - Private Methods
    
    private func createCHandle(from floats: [Float]) {
        let shapeArray = shape.map { Int64($0) }
        
        floats.withUnsafeBufferPointer { floatBuffer in
            shapeArray.withUnsafeBufferPointer { shapeBuffer in
                cHandle = tfk_create_tensor(
                    floatBuffer.baseAddress,
                    shapeBuffer.baseAddress,
                    shape.count
                )
            }
        }
    }
}

// MARK: - Convenience Initializers

extension TFKTensor {
    
    /// Create zeros tensor
    public static func zeros(shape: [Int]) -> TFKTensor {
        let count = shape.reduce(1, *)
        let floats = [Float](repeating: 0, count: count)
        return TFKTensor(floats: floats, shape: shape)
    }
    
    /// Create ones tensor
    public static func ones(shape: [Int]) -> TFKTensor {
        let count = shape.reduce(1, *)
        let floats = [Float](repeating: 1, count: count)
        return TFKTensor(floats: floats, shape: shape)
    }
    
    /// Create random tensor
    public static func random(shape: [Int], min: Float = 0, max: Float = 1) -> TFKTensor {
        let count = shape.reduce(1, *)
        let range = max - min
        let floats = (0..<count).map { _ in Float.random(in: 0..<1) * range + min }
        return TFKTensor(floats: floats, shape: shape)
    }
}

// MARK: - Operators

extension TFKTensor {
    
    /// Add two tensors
    public static func +(lhs: TFKTensor, rhs: TFKTensor) throws -> TFKTensor {
        guard lhs.shape == rhs.shape else {
            throw TFKError.tensorShapeMismatch(expected: lhs.shape, actual: rhs.shape)
        }
        
        guard let lhsData = lhs.floatData(),
              let rhsData = rhs.floatData() else {
            throw TFKError.inferenceFailed(reason: "Failed to access tensor data")
        }
        
        var result = [Float](repeating: 0, count: lhs.elementCount)
        vDSP_vadd(lhsData, 1, rhsData, 1, &result, 1, vDSP_Length(lhs.elementCount))
        
        return TFKTensor(floats: result, shape: lhs.shape)
    }
    
    /// Multiply two tensors element-wise
    public static func *(lhs: TFKTensor, rhs: TFKTensor) throws -> TFKTensor {
        guard lhs.shape == rhs.shape else {
            throw TFKError.tensorShapeMismatch(expected: lhs.shape, actual: rhs.shape)
        }
        
        guard let lhsData = lhs.floatData(),
              let rhsData = rhs.floatData() else {
            throw TFKError.inferenceFailed(reason: "Failed to access tensor data")
        }
        
        var result = [Float](repeating: 0, count: lhs.elementCount)
        vDSP_vmul(lhsData, 1, rhsData, 1, &result, 1, vDSP_Length(lhs.elementCount))
        
        return TFKTensor(floats: result, shape: lhs.shape)
    }
}