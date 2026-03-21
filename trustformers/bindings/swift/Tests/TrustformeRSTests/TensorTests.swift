import XCTest
@testable import TrustformeRS

@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
final class TensorTests: XCTestCase {
    
    func testTensorCreation() throws {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        let tensor = Tensor(data: data, shape: [2, 3])
        
        XCTAssertEqual(tensor.dimensions, 2)
        XCTAssertEqual(tensor.size, 6)
        XCTAssertEqual(tensor.tensorShape, [2, 3])
        XCTAssertEqual(tensor.rawData, data)
    }
    
    func testTensorFactoryMethods() throws {
        // Test 1D tensor
        let tensor1D = Tensor.from1D([1, 2, 3])
        XCTAssertEqual(tensor1D.dimensions, 1)
        XCTAssertEqual(tensor1D.tensorShape, [3])
        
        // Test 2D tensor
        let tensor2D = Tensor.from2D([[1, 2], [3, 4]])
        XCTAssertEqual(tensor2D.dimensions, 2)
        XCTAssertEqual(tensor2D.tensorShape, [2, 2])
        XCTAssertEqual(tensor2D.rawData, [1, 2, 3, 4])
        
        // Test 3D tensor
        let tensor3D = Tensor.from3D([[[1, 2]], [[3, 4]]])
        XCTAssertEqual(tensor3D.dimensions, 3)
        XCTAssertEqual(tensor3D.tensorShape, [2, 1, 2])
    }
    
    func testZerosAndOnes() throws {
        let zeros = Tensor.zeros(shape: [2, 3])
        XCTAssertEqual(zeros.size, 6)
        XCTAssertTrue(zeros.rawData.allSatisfy { $0 == 0 })
        
        let ones = Tensor.ones(shape: [2, 3])
        XCTAssertEqual(ones.size, 6)
        XCTAssertTrue(ones.rawData.allSatisfy { $0 == 1 })
    }
    
    func testRandomTensor() throws {
        let randomTensor = Tensor.random(shape: [10, 10], distribution: .uniform(0, 1))
        XCTAssertEqual(randomTensor.tensorShape, [10, 10])
        XCTAssertEqual(randomTensor.size, 100)
        XCTAssertTrue(randomTensor.rawData.allSatisfy { $0 >= 0 && $0 <= 1 })
        
        let normalTensor = Tensor.random(shape: [5, 5], distribution: .standardNormal)
        XCTAssertEqual(normalTensor.size, 25)
        // For normal distribution, values can be outside [-1, 1], so just check size
    }
    
    func testTensorIndexing() throws {
        let tensor = Tensor.from2D([[1, 2, 3], [4, 5, 6]])
        
        XCTAssertEqual(tensor[0, 0], 1)
        XCTAssertEqual(tensor[0, 1], 2)
        XCTAssertEqual(tensor[0, 2], 3)
        XCTAssertEqual(tensor[1, 0], 4)
        XCTAssertEqual(tensor[1, 1], 5)
        XCTAssertEqual(tensor[1, 2], 6)
    }
    
    func testTensorReshape() throws {
        let tensor = Tensor.from1D([1, 2, 3, 4, 5, 6])
        
        let reshaped2D = try tensor.reshape([2, 3])
        XCTAssertEqual(reshaped2D.tensorShape, [2, 3])
        XCTAssertEqual(reshaped2D.size, 6)
        
        let reshaped3D = try tensor.reshape([1, 2, 3])
        XCTAssertEqual(reshaped3D.tensorShape, [1, 2, 3])
        XCTAssertEqual(reshaped3D.size, 6)
        
        // Test incompatible reshape
        XCTAssertThrowsError(try tensor.reshape([2, 2])) { error in
            XCTAssertTrue(error is TensorError)
        }
    }
    
    func testTensorTranspose() throws {
        let tensor = Tensor.from2D([[1, 2, 3], [4, 5, 6]])
        let transposed = try tensor.transpose()
        
        XCTAssertEqual(transposed.tensorShape, [3, 2])
        XCTAssertEqual(transposed[0, 0], 1)
        XCTAssertEqual(transposed[0, 1], 4)
        XCTAssertEqual(transposed[1, 0], 2)
        XCTAssertEqual(transposed[1, 1], 5)
        XCTAssertEqual(transposed[2, 0], 3)
        XCTAssertEqual(transposed[2, 1], 6)
    }
    
    func testTensorSqueeze() throws {
        let tensor = Tensor(data: [1, 2, 3], shape: [1, 3, 1])
        
        let squeezed = try tensor.squeeze()
        XCTAssertEqual(squeezed.tensorShape, [3])
        
        let squeezedDim = try tensor.squeeze(dim: 0)
        XCTAssertEqual(squeezedDim.tensorShape, [3, 1])
        
        // Test invalid squeeze
        XCTAssertThrowsError(try tensor.squeeze(dim: 1)) { error in
            XCTAssertTrue(error is TensorError)
        }
    }
    
    func testTensorUnsqueeze() throws {
        let tensor = Tensor.from1D([1, 2, 3])
        
        let unsqueezed = try tensor.unsqueeze(dim: 0)
        XCTAssertEqual(unsqueezed.tensorShape, [1, 3])
        
        let unsqueezed2 = try tensor.unsqueeze(dim: 1)
        XCTAssertEqual(unsqueezed2.tensorShape, [3, 1])
        
        // Test invalid unsqueeze
        XCTAssertThrowsError(try tensor.unsqueeze(dim: 5)) { error in
            XCTAssertTrue(error is TensorError)
        }
    }
    
    func testTensorAddition() throws {
        let tensor1 = Tensor.from1D([1, 2, 3])
        let tensor2 = Tensor.from1D([4, 5, 6])
        
        let result = try tensor1.add(tensor2)
        XCTAssertEqual(result.rawData, [5, 7, 9])
        
        // Test incompatible shapes
        let tensor3 = Tensor.from1D([1, 2])
        XCTAssertThrowsError(try tensor1.add(tensor3)) { error in
            XCTAssertTrue(error is TensorError)
        }
    }
    
    func testTensorSubtraction() throws {
        let tensor1 = Tensor.from1D([4, 5, 6])
        let tensor2 = Tensor.from1D([1, 2, 3])
        
        let result = try tensor1.subtract(tensor2)
        XCTAssertEqual(result.rawData, [3, 3, 3])
    }
    
    func testTensorMultiplication() throws {
        let tensor1 = Tensor.from1D([2, 3, 4])
        let tensor2 = Tensor.from1D([5, 6, 7])
        
        let result = try tensor1.multiply(tensor2)
        XCTAssertEqual(result.rawData, [10, 18, 28])
    }
    
    func testMatrixMultiplication() throws {
        let tensor1 = Tensor.from2D([[1, 2], [3, 4]])
        let tensor2 = Tensor.from2D([[5, 6], [7, 8]])
        
        let result = try tensor1.matmul(tensor2)
        XCTAssertEqual(result.tensorShape, [2, 2])
        
        // Expected result: [[19, 22], [43, 50]]
        XCTAssertEqual(result[0, 0], 19, accuracy: 0.001)
        XCTAssertEqual(result[0, 1], 22, accuracy: 0.001)
        XCTAssertEqual(result[1, 0], 43, accuracy: 0.001)
        XCTAssertEqual(result[1, 1], 50, accuracy: 0.001)
        
        // Test incompatible shapes
        let tensor3 = Tensor.from2D([[1, 2, 3]])
        XCTAssertThrowsError(try tensor1.matmul(tensor3)) { error in
            XCTAssertTrue(error is TensorError)
        }
    }
    
    func testTensorSum() throws {
        let tensor = Tensor.from1D([1, 2, 3, 4, 5])
        let result = try tensor.sum()
        
        XCTAssertEqual(result.rawData, [15])
        XCTAssertEqual(result.tensorShape, [1])
    }
    
    func testTensorMean() throws {
        let tensor = Tensor.from1D([2, 4, 6, 8])
        let result = try tensor.mean()
        
        XCTAssertEqual(result.rawData, [5])
        XCTAssertEqual(result.tensorShape, [1])
    }
    
    func testTensorOperatorOverloads() throws {
        let tensor1 = Tensor.from1D([1, 2, 3])
        let tensor2 = Tensor.from1D([4, 5, 6])
        
        let sum = try tensor1 + tensor2
        XCTAssertEqual(sum.rawData, [5, 7, 9])
        
        let diff = try tensor2 - tensor1
        XCTAssertEqual(diff.rawData, [3, 3, 3])
        
        let product = try tensor1 * tensor2
        XCTAssertEqual(product.rawData, [4, 10, 18])
    }
    
    func testTensorDeviceConversion() throws {
        let tensor = Tensor.from1D([1, 2, 3])
        XCTAssertEqual(tensor.device, .cpu)
        
        let gpuTensor = tensor.to(device: .gpu)
        XCTAssertEqual(gpuTensor.device, .gpu)
        XCTAssertEqual(gpuTensor.rawData, tensor.rawData)
        
        #if os(macOS) || os(iOS)
        let metalTensor = tensor.to(device: .metal)
        XCTAssertEqual(metalTensor.device, .metal)
        
        let coremlTensor = tensor.to(device: .coreml)
        XCTAssertEqual(coremlTensor.device, .coreml)
        #endif
    }
    
    func testTensorDataTypeConversion() throws {
        let tensor = Tensor.from1D([1, 2, 3])
        XCTAssertEqual(tensor.dataType, .float32)
        
        let doubleTensor = tensor.to(dataType: .float64)
        XCTAssertEqual(doubleTensor.dataType, .float64)
        XCTAssertEqual(doubleTensor.rawData, tensor.rawData)
    }
    
    func testDataTypeSizes() throws {
        XCTAssertEqual(Tensor.DataType.float32.size, 4)
        XCTAssertEqual(Tensor.DataType.float64.size, 8)
        XCTAssertEqual(Tensor.DataType.int32.size, 4)
        XCTAssertEqual(Tensor.DataType.int64.size, 8)
        XCTAssertEqual(Tensor.DataType.bool.size, 1)
    }
}

/// Performance tests for tensor operations
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension TensorTests {
    
    func testPerformanceTensorCreation() throws {
        measure {
            for _ in 0..<100 {
                _ = Tensor.random(shape: [100, 100])
            }
        }
    }
    
    func testPerformanceTensorAddition() throws {
        let tensor1 = Tensor.random(shape: [1000, 1000])
        let tensor2 = Tensor.random(shape: [1000, 1000])
        
        measure {
            _ = try! tensor1.add(tensor2)
        }
    }
    
    func testPerformanceMatrixMultiplication() throws {
        let tensor1 = Tensor.random(shape: [100, 100])
        let tensor2 = Tensor.random(shape: [100, 100])
        
        measure {
            _ = try! tensor1.matmul(tensor2)
        }
    }
}