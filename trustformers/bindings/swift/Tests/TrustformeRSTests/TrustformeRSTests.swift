import XCTest
@testable import TrustformeRS

@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
final class TrustformeRSTests: XCTestCase {
    
    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    func testVersionInfo() throws {
        let version = TrustformeRS.version
        XCTAssertFalse(version.isEmpty, "Version should not be empty")
        print("TrustformeRS version: \(version)")
    }
    
    func testGPUAvailability() throws {
        let isGPUAvailable = TrustformeRS.isGPUAvailable
        print("GPU available: \(isGPUAvailable)")
        // This test just checks that the call doesn't crash
    }
    
    #if os(macOS) || os(iOS)
    func testMetalAvailability() throws {
        let isMetalAvailable = TrustformeRS.isMetalAvailable
        print("Metal available: \(isMetalAvailable)")
        // On actual Apple devices, Metal should be available
        #if targetEnvironment(simulator)
        // Metal may not be available on simulator
        #else
        XCTAssertTrue(isMetalAvailable, "Metal should be available on actual Apple devices")
        #endif
    }
    
    func testCoreMLAvailability() throws {
        let isCoreMLAvailable = TrustformeRS.isCoreMLAvailable
        print("CoreML available: \(isCoreMLAvailable)")
        // CoreML should be available on all Apple platforms
        XCTAssertTrue(isCoreMLAvailable, "CoreML should be available on Apple platforms")
    }
    #endif
    
    func testTrustformeRSInitialization() throws {
        let config = TrustformeRS.Configuration.default
        // In a real test, we would initialize TrustformeRS
        // For now, we just test configuration creation
        XCTAssertEqual(config.numThreads, -1)
        XCTAssertFalse(config.enableLogging)
    }
    
    func testConfigurationPresets() throws {
        let defaultConfig = TrustformeRS.Configuration.default
        let iOSConfig = TrustformeRS.Configuration.iOS
        let macOSConfig = TrustformeRS.Configuration.macOS
        
        XCTAssertTrue(iOSConfig.useGPU)
        XCTAssertEqual(iOSConfig.numThreads, 4)
        XCTAssertEqual(iOSConfig.device, "auto")
        
        XCTAssertTrue(macOSConfig.useGPU)
        XCTAssertEqual(macOSConfig.numThreads, -1)
        XCTAssertEqual(macOSConfig.device, "auto")
    }
    
    func testModelConfiguration() throws {
        let localConfig = Model.Configuration.local(path: "/path/to/model")
        XCTAssertEqual(localConfig.modelPath, "/path/to/model")
        XCTAssertEqual(localConfig.tokenizerPath, "/path/to/model")
        
        let hfConfig = Model.Configuration.huggingFace("bert-base-uncased")
        XCTAssertEqual(hfConfig.modelPath, "bert-base-uncased")
        XCTAssertNil(hfConfig.tokenizerPath)
    }
    
    func testPipelineConfiguration() throws {
        let textGenConfig = Pipeline.Configuration.textGeneration(
            modelId: "gpt2",
            temperature: 0.7,
            maxNewTokens: 100
        )
        
        XCTAssertEqual(textGenConfig.task, .textGeneration)
        XCTAssertEqual(textGenConfig.modelId, "gpt2")
        XCTAssertEqual(textGenConfig.temperature, 0.7)
        XCTAssertEqual(textGenConfig.maxNewTokens, 100)
        XCTAssertTrue(textGenConfig.doSample)
        
        let classificationConfig = Pipeline.Configuration.textClassification(modelId: "bert-base-uncased")
        XCTAssertEqual(classificationConfig.task, .textClassification)
        XCTAssertEqual(classificationConfig.modelId, "bert-base-uncased")
        XCTAssertFalse(classificationConfig.doSample)
    }
    
    func testTaskDescriptions() throws {
        XCTAssertEqual(Pipeline.Task.textGeneration.description, "Text Generation")
        XCTAssertEqual(Pipeline.Task.textClassification.description, "Text Classification")
        XCTAssertEqual(Pipeline.Task.questionAnswering.description, "Question Answering")
    }
    
    func testErrorTypes() throws {
        let error1 = TrustformersError.invalidArgument("Test message")
        XCTAssertEqual(error1.description, "Invalid argument: Test message")
        
        let error2 = TrustformersError.modelNotFound("Model not found")
        XCTAssertEqual(error2.description, "Model not found: Model not found")
        
        let error3 = TrustformersError.inferenceError("Inference failed")
        XCTAssertEqual(error3.errorDescription, "Inference error: Inference failed")
    }
}

/// Performance tests
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension TrustformeRSTests {
    
    func testPerformanceVersionCall() throws {
        measure {
            _ = TrustformeRS.version
        }
    }
    
    func testPerformanceConfigurationCreation() throws {
        measure {
            for _ in 0..<1000 {
                _ = TrustformeRS.Configuration.default
            }
        }
    }
}