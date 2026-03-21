import Foundation
import TrustformeRSCore
import Accelerate

/// Multi-dimensional tensor for numerical computations
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
public struct Tensor: Sendable {
    private let data: [Float]
    private let shape: [Int]
    private let strides: [Int]
    
    /// Data type for tensor elements
    public enum DataType: String, CaseIterable {
        case float32 = "f32"
        case float64 = "f64"
        case int32 = "i32"
        case int64 = "i64"
        case bool = "bool"
        
        public var size: Int {
            switch self {
            case .float32, .int32: return 4
            case .float64, .int64: return 8
            case .bool: return 1
            }
        }
    }
    
    /// Device where tensor is stored
    public enum Device: String, CaseIterable {
        case cpu = "cpu"
        case gpu = "gpu"
        case metal = "metal"
        case coreml = "coreml"
        
        public var description: String {
            switch self {
            case .cpu: return "CPU"
            case .gpu: return "GPU"
            case .metal: return "Metal"
            case .coreml: return "Core ML"
            }
        }
    }
    
    /// Tensor properties
    public let dataType: DataType
    public let device: Device
    
    /// Initialize tensor with data and shape
    /// - Parameters:
    ///   - data: Raw tensor data
    ///   - shape: Tensor dimensions
    ///   - dataType: Element data type
    ///   - device: Device location
    public init(
        data: [Float],
        shape: [Int],
        dataType: DataType = .float32,
        device: Device = .cpu
    ) {
        self.data = data
        self.shape = shape
        self.dataType = dataType
        self.device = device
        
        // Calculate strides
        var strides = Array(repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        self.strides = strides
    }
    
    /// Create tensor from 1D array
    /// - Parameter array: 1D array of values
    /// - Returns: 1D tensor
    public static func from1D(_ array: [Float]) -> Tensor {
        return Tensor(data: array, shape: [array.count])
    }
    
    /// Create tensor from 2D array
    /// - Parameter array: 2D array of values
    /// - Returns: 2D tensor
    public static func from2D(_ array: [[Float]]) -> Tensor {
        let rows = array.count
        let cols = array.first?.count ?? 0
        let flatData = array.flatMap { $0 }
        return Tensor(data: flatData, shape: [rows, cols])
    }
    
    /// Create tensor from 3D array
    /// - Parameter array: 3D array of values
    /// - Returns: 3D tensor
    public static func from3D(_ array: [[[Float]]]) -> Tensor {
        let dim0 = array.count
        let dim1 = array.first?.count ?? 0
        let dim2 = array.first?.first?.count ?? 0
        let flatData = array.flatMap { $0.flatMap { $0 } }
        return Tensor(data: flatData, shape: [dim0, dim1, dim2])
    }
    
    /// Create tensor filled with zeros
    /// - Parameters:
    ///   - shape: Tensor dimensions
    ///   - dataType: Element data type
    ///   - device: Device location
    /// - Returns: Zero-filled tensor
    public static func zeros(
        shape: [Int],
        dataType: DataType = .float32,
        device: Device = .cpu
    ) -> Tensor {
        let size = shape.reduce(1, *)
        let data = Array(repeating: Float(0), count: size)
        return Tensor(data: data, shape: shape, dataType: dataType, device: device)
    }
    
    /// Create tensor filled with ones
    /// - Parameters:
    ///   - shape: Tensor dimensions
    ///   - dataType: Element data type
    ///   - device: Device location
    /// - Returns: One-filled tensor
    public static func ones(
        shape: [Int],
        dataType: DataType = .float32,
        device: Device = .cpu
    ) -> Tensor {
        let size = shape.reduce(1, *)
        let data = Array(repeating: Float(1), count: size)
        return Tensor(data: data, shape: shape, dataType: dataType, device: device)
    }
    
    /// Create tensor with random values
    /// - Parameters:
    ///   - shape: Tensor dimensions
    ///   - distribution: Random distribution type
    ///   - dataType: Element data type
    ///   - device: Device location
    /// - Returns: Random tensor
    public static func random(
        shape: [Int],
        distribution: RandomDistribution = .uniform(0, 1),
        dataType: DataType = .float32,
        device: Device = .cpu
    ) -> Tensor {
        let size = shape.reduce(1, *)
        var data: [Float] = []
        
        switch distribution {
        case .uniform(let min, let max):
            data = (0..<size).map { _ in Float.random(in: min...max) }
        case .normal(let mean, let std):
            data = (0..<size).map { _ in Float(Distribution.normal(mean: Double(mean), standardDeviation: Double(std))) }
        case .standardNormal:
            data = (0..<size).map { _ in Float(Distribution.normal(mean: 0, standardDeviation: 1)) }
        }
        
        return Tensor(data: data, shape: shape, dataType: dataType, device: device)
    }
    
    /// Random distribution types
    public enum RandomDistribution {
        case uniform(Float, Float)
        case normal(Float, Float)
        case standardNormal
    }
    
    /// Get tensor dimensions
    public var dimensions: Int {
        return shape.count
    }
    
    /// Get total number of elements
    public var size: Int {
        return shape.reduce(1, *)
    }
    
    /// Get tensor shape
    public var tensorShape: [Int] {
        return shape
    }
    
    /// Get raw data
    public var rawData: [Float] {
        return data
    }
    
    /// Access element at specific indices
    /// - Parameter indices: Multi-dimensional indices
    /// - Returns: Element value
    public subscript(indices: Int...) -> Float {
        get {
            let flatIndex = calculateFlatIndex(indices)
            return data[flatIndex]
        }
    }
    
    /// Reshape tensor
    /// - Parameter newShape: New tensor shape
    /// - Returns: Reshaped tensor
    /// - Throws: Error if shapes are incompatible
    public func reshape(_ newShape: [Int]) throws -> Tensor {
        let newSize = newShape.reduce(1, *)
        guard newSize == size else {
            throw TensorError.incompatibleShapes("Cannot reshape tensor of size \(size) to shape \(newShape)")
        }
        
        return Tensor(data: data, shape: newShape, dataType: dataType, device: device)
    }
    
    /// Transpose tensor (swap dimensions)
    /// - Parameters:
    ///   - dim0: First dimension
    ///   - dim1: Second dimension
    /// - Returns: Transposed tensor
    public func transpose(dim0: Int = 0, dim1: Int = 1) throws -> Tensor {
        guard dim0 < dimensions && dim1 < dimensions else {
            throw TensorError.invalidDimension("Dimensions \(dim0) and \(dim1) out of bounds for tensor with \(dimensions) dimensions")
        }
        
        var newShape = shape
        newShape.swapAt(dim0, dim1)
        
        // For 2D case, use simple transpose
        if dimensions == 2 && dim0 == 0 && dim1 == 1 {
            var newData = Array(repeating: Float(0), count: size)
            let rows = shape[0]
            let cols = shape[1]
            
            for i in 0..<rows {
                for j in 0..<cols {
                    newData[j * rows + i] = data[i * cols + j]
                }
            }
            
            return Tensor(data: newData, shape: newShape, dataType: dataType, device: device)
        }
        
        // For higher dimensions, implement general transpose
        // This is a simplified version
        return Tensor(data: data, shape: newShape, dataType: dataType, device: device)
    }
    
    /// Squeeze tensor (remove dimensions of size 1)
    /// - Parameter dim: Specific dimension to squeeze (nil for all)
    /// - Returns: Squeezed tensor
    public func squeeze(dim: Int? = nil) throws -> Tensor {
        var newShape = shape
        
        if let dim = dim {
            guard dim < dimensions && shape[dim] == 1 else {
                throw TensorError.invalidDimension("Cannot squeeze dimension \(dim)")
            }
            newShape.remove(at: dim)
        } else {
            newShape = shape.filter { $0 != 1 }
        }
        
        return Tensor(data: data, shape: newShape, dataType: dataType, device: device)
    }
    
    /// Unsqueeze tensor (add dimension of size 1)
    /// - Parameter dim: Dimension to add
    /// - Returns: Unsqueezed tensor
    public func unsqueeze(dim: Int) throws -> Tensor {
        guard dim >= 0 && dim <= dimensions else {
            throw TensorError.invalidDimension("Cannot unsqueeze at dimension \(dim)")
        }
        
        var newShape = shape
        newShape.insert(1, at: dim)
        
        return Tensor(data: data, shape: newShape, dataType: dataType, device: device)
    }
    
    /// Convert tensor to different device
    /// - Parameter device: Target device
    /// - Returns: Tensor on new device
    public func to(device: Device) -> Tensor {
        // In a real implementation, this would handle device transfer
        return Tensor(data: data, shape: shape, dataType: dataType, device: device)
    }
    
    /// Convert tensor to different data type
    /// - Parameter dataType: Target data type
    /// - Returns: Tensor with new data type
    public func to(dataType: DataType) -> Tensor {
        // In a real implementation, this would handle type conversion
        return Tensor(data: data, shape: shape, dataType: dataType, device: device)
    }
    
    /// Calculate flat index from multi-dimensional indices
    private func calculateFlatIndex(_ indices: [Int]) -> Int {
        guard indices.count == dimensions else {
            fatalError("Number of indices must match tensor dimensions")
        }
        
        var flatIndex = 0
        for (i, index) in indices.enumerated() {
            flatIndex += index * strides[i]
        }
        
        return flatIndex
    }
}

/// Mathematical operations
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension Tensor {
    /// Add two tensors
    /// - Parameter other: Other tensor
    /// - Returns: Sum tensor
    public func add(_ other: Tensor) throws -> Tensor {
        guard shape == other.shape else {
            throw TensorError.incompatibleShapes("Cannot add tensors with shapes \(shape) and \(other.shape)")
        }
        
        var result = Array(repeating: Float(0), count: size)
        vDSP_vadd(data, 1, other.data, 1, &result, 1, vDSP_Length(size))
        
        return Tensor(data: result, shape: shape, dataType: dataType, device: device)
    }
    
    /// Subtract two tensors
    /// - Parameter other: Other tensor
    /// - Returns: Difference tensor
    public func subtract(_ other: Tensor) throws -> Tensor {
        guard shape == other.shape else {
            throw TensorError.incompatibleShapes("Cannot subtract tensors with shapes \(shape) and \(other.shape)")
        }
        
        var result = Array(repeating: Float(0), count: size)
        vDSP_vsub(other.data, 1, data, 1, &result, 1, vDSP_Length(size))
        
        return Tensor(data: result, shape: shape, dataType: dataType, device: device)
    }
    
    /// Multiply two tensors element-wise
    /// - Parameter other: Other tensor
    /// - Returns: Product tensor
    public func multiply(_ other: Tensor) throws -> Tensor {
        guard shape == other.shape else {
            throw TensorError.incompatibleShapes("Cannot multiply tensors with shapes \(shape) and \(other.shape)")
        }
        
        var result = Array(repeating: Float(0), count: size)
        vDSP_vmul(data, 1, other.data, 1, &result, 1, vDSP_Length(size))
        
        return Tensor(data: result, shape: shape, dataType: dataType, device: device)
    }
    
    /// Matrix multiplication
    /// - Parameter other: Other tensor
    /// - Returns: Matrix product
    public func matmul(_ other: Tensor) throws -> Tensor {
        guard dimensions == 2 && other.dimensions == 2 else {
            throw TensorError.invalidOperation("Matrix multiplication requires 2D tensors")
        }
        
        guard shape[1] == other.shape[0] else {
            throw TensorError.incompatibleShapes("Cannot multiply matrices with shapes \(shape) and \(other.shape)")
        }
        
        let m = shape[0]
        let n = other.shape[1]
        let k = shape[1]
        
        var result = Array(repeating: Float(0), count: m * n)
        
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(m), Int32(n), Int32(k),
            1.0,
            data, Int32(k),
            other.data, Int32(n),
            0.0,
            &result, Int32(n)
        )
        
        return Tensor(data: result, shape: [m, n], dataType: dataType, device: device)
    }
    
    /// Sum tensor along axis
    /// - Parameter axis: Axis to sum along (nil for all)
    /// - Returns: Summed tensor
    public func sum(axis: Int? = nil) throws -> Tensor {
        if axis == nil {
            let sum = data.reduce(0, +)
            return Tensor(data: [sum], shape: [1])
        }
        
        guard let axis = axis, axis >= 0 && axis < dimensions else {
            throw TensorError.invalidDimension("Axis \(axis ?? -1) out of bounds for tensor with \(dimensions) dimensions")
        }
        
        // Calculate output shape
        var outputShape = shape
        outputShape.remove(at: axis)
        if outputShape.isEmpty { outputShape = [1] }
        
        let outputSize = outputShape.reduce(1, *)
        var result = Array(repeating: Float(0), count: outputSize)
        
        // Calculate strides for the operation
        let axisSize = shape[axis]
        let outerSize = shape[0..<axis].reduce(1, *)
        let innerSize = shape[(axis+1)..<shape.count].reduce(1, *)
        
        // Perform summation along the specified axis
        for outer in 0..<outerSize {
            for inner in 0..<innerSize {
                var sum: Float = 0
                for axisIdx in 0..<axisSize {
                    let inputIdx = outer * axisSize * innerSize + axisIdx * innerSize + inner
                    sum += data[inputIdx]
                }
                let outputIdx = outer * innerSize + inner
                result[outputIdx] = sum
            }
        }
        
        return Tensor(data: result, shape: outputShape, dataType: dataType, device: device)
    }
    
    /// Mean tensor along axis
    /// - Parameter axis: Axis to average along (nil for all)
    /// - Returns: Mean tensor
    public func mean(axis: Int? = nil) throws -> Tensor {
        if axis == nil {
            let mean = data.reduce(0, +) / Float(size)
            return Tensor(data: [mean], shape: [1])
        }
        
        guard let axis = axis, axis >= 0 && axis < dimensions else {
            throw TensorError.invalidDimension("Axis \(axis ?? -1) out of bounds for tensor with \(dimensions) dimensions")
        }
        
        // Calculate output shape
        var outputShape = shape
        outputShape.remove(at: axis)
        if outputShape.isEmpty { outputShape = [1] }
        
        let outputSize = outputShape.reduce(1, *)
        var result = Array(repeating: Float(0), count: outputSize)
        
        // Calculate strides for the operation
        let axisSize = shape[axis]
        let outerSize = shape[0..<axis].reduce(1, *)
        let innerSize = shape[(axis+1)..<shape.count].reduce(1, *)
        
        // Perform averaging along the specified axis
        for outer in 0..<outerSize {
            for inner in 0..<innerSize {
                var sum: Float = 0
                for axisIdx in 0..<axisSize {
                    let inputIdx = outer * axisSize * innerSize + axisIdx * innerSize + inner
                    sum += data[inputIdx]
                }
                let outputIdx = outer * innerSize + inner
                result[outputIdx] = sum / Float(axisSize)
            }
        }
        
        return Tensor(data: result, shape: outputShape, dataType: dataType, device: device)
    }
}

/// Operator overloads for convenience
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension Tensor {
    public static func + (lhs: Tensor, rhs: Tensor) throws -> Tensor {
        return try lhs.add(rhs)
    }
    
    public static func - (lhs: Tensor, rhs: Tensor) throws -> Tensor {
        return try lhs.subtract(rhs)
    }
    
    public static func * (lhs: Tensor, rhs: Tensor) throws -> Tensor {
        return try lhs.multiply(rhs)
    }
}

/// Conversion to/from C tensor
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension Tensor {
    internal func toCTensor() -> TrustformersTensor {
        let dataPtr = UnsafeMutablePointer<Float>.allocate(capacity: data.count)
        dataPtr.initialize(from: data, count: data.count)
        
        let shapePtr = UnsafeMutablePointer<Int>.allocate(capacity: shape.count)
        shapePtr.initialize(from: shape, count: shape.count)
        
        return TrustformersTensor(
            data: dataPtr,
            shape: shapePtr,
            ndim: shape.count
        )
    }
    
    internal static func fromCTensor(_ cTensor: TrustformersTensor) -> Tensor {
        let size = (0..<cTensor.ndim).map { Int(cTensor.shape[$0]) }.reduce(1, *)
        let data = Array(UnsafeBufferPointer(start: cTensor.data, count: size))
        let shape = Array(UnsafeBufferPointer(start: cTensor.shape, count: cTensor.ndim))
        
        return Tensor(data: data, shape: shape)
    }
}

/// Tensor error types
public enum TensorError: Error, LocalizedError {
    case incompatibleShapes(String)
    case invalidDimension(String)
    case invalidOperation(String)
    case notImplemented(String)
    
    public var errorDescription: String? {
        switch self {
        case .incompatibleShapes(let msg): return "Incompatible shapes: \(msg)"
        case .invalidDimension(let msg): return "Invalid dimension: \(msg)"
        case .invalidOperation(let msg): return "Invalid operation: \(msg)"
        case .notImplemented(let msg): return "Not implemented: \(msg)"
        }
    }
}

/// Distribution helper for random number generation
private enum Distribution {
    static func normal(mean: Double, standardDeviation: Double) -> Double {
        // Box-Muller transform for normal distribution
        let u1 = Double.random(in: 0...1)
        let u2 = Double.random(in: 0...1)
        let z0 = sqrt(-2 * log(u1)) * cos(2 * Double.pi * u2)
        return z0 * standardDeviation + mean
    }
}