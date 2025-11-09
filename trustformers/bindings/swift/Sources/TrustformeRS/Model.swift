import Foundation
import TrustformeRSCore

/// Represents a loaded transformer model
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
public final class Model: @unchecked Sendable {
    private let handle: OpaquePointer
    private let trustformers: TrustformeRS
    
    /// Configuration for model loading
    public struct Configuration {
        public let modelType: String
        public let modelPath: String
        public let tokenizerPath: String?
        public let useFastTokenizer: Bool
        public let maxLength: Int
        public let doLowerCase: Bool
        
        public init(
            modelType: String = "auto",
            modelPath: String,
            tokenizerPath: String? = nil,
            useFastTokenizer: Bool = true,
            maxLength: Int = 512,
            doLowerCase: Bool = false
        ) {
            self.modelType = modelType
            self.modelPath = modelPath
            self.tokenizerPath = tokenizerPath
            self.useFastTokenizer = useFastTokenizer
            self.maxLength = maxLength
            self.doLowerCase = doLowerCase
        }
        
        /// Create configuration for a Hugging Face model
        /// - Parameter modelId: Hugging Face model ID (e.g., "bert-base-uncased")
        /// - Returns: Model configuration
        public static func huggingFace(_ modelId: String) -> Configuration {
            return Configuration(modelPath: modelId)
        }
        
        /// Create configuration for a local model
        /// - Parameters:
        ///   - path: Local path to model files
        ///   - tokenizerPath: Optional path to tokenizer (defaults to same as model)
        /// - Returns: Model configuration
        public static func local(path: String, tokenizerPath: String? = nil) -> Configuration {
            return Configuration(
                modelPath: path,
                tokenizerPath: tokenizerPath ?? path
            )
        }
    }
    
    /// Model metadata
    public struct Metadata {
        public let modelType: String
        public let vocabSize: Int
        public let maxPositionEmbeddings: Int?
        public let hiddenSize: Int?
        public let numLayers: Int?
        public let numAttentionHeads: Int?
    }
    
    internal init(trustformers: TrustformeRS, config: Configuration) throws {
        self.trustformers = trustformers
        
        var cConfig = ModelConfig()
        cConfig.model_type = config.modelType.cString(using: .utf8)
        cConfig.model_path = config.modelPath.cString(using: .utf8)
        cConfig.tokenizer_path = config.tokenizerPath?.cString(using: .utf8)
        cConfig.use_fast_tokenizer = config.useFastTokenizer
        cConfig.max_length = Int32(config.maxLength)
        cConfig.do_lower_case = config.doLowerCase
        
        var error = TrustformersError()
        guard let handle = model_load(trustformers.cHandle, &cConfig, &error) else {
            throw TrustformersError.fromC(error)
        }
        
        self.handle = handle
    }
    
    deinit {
        model_free(handle)
    }
    
    /// Get model vocabulary size
    public var vocabSize: Int {
        return Int(model_get_vocab_size(handle))
    }
    
    /// Get model type
    public var modelType: String {
        guard let typePtr = model_get_model_type(handle) else {
            return "unknown"
        }
        defer { trustformers_free_string(typePtr) }
        return String(cString: typePtr)
    }
    
    /// Generate text from input prompt
    /// - Parameters:
    ///   - input: Input prompt
    ///   - maxLength: Maximum generation length
    /// - Returns: Generated text
    /// - Throws: TrustformersError if generation fails
    public func generate(input: String, maxLength: Int = 100) throws -> String {
        var error = TrustformersError()
        guard let resultPtr = model_generate(handle, input.cString(using: .utf8), Int32(maxLength), &error) else {
            throw TrustformersError.fromC(error)
        }
        
        defer { trustformers_free_string_array(resultPtr, 1) }
        return String(cString: resultPtr[0])
    }
    
    /// Generate text asynchronously
    /// - Parameters:
    ///   - input: Input prompt
    ///   - maxLength: Maximum generation length
    /// - Returns: Generated text
    public func generateAsync(input: String, maxLength: Int = 100) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            let context = CallbackContext(continuation: continuation)
            let userDataPtr = Unmanaged.passRetained(context).toOpaque()
            
            model_generate_async(handle, input.cString(using: .utf8), Int32(maxLength)) { userData, result, error in
                guard let userData = userData else { return }
                let context = Unmanaged<CallbackContext<String>>.fromOpaque(userData).takeRetainedValue()
                
                if let error = error, error.pointee.code != TRUSTFORMERS_OK {
                    context.continuation.resume(throwing: TrustformersError.fromC(error.pointee))
                } else if let result = result {
                    context.continuation.resume(returning: String(cString: result))
                } else {
                    context.continuation.resume(throwing: TrustformersError.unknown("No result returned"))
                }
            }
        }
    }
    
    /// Perform forward pass with tensor input
    /// - Parameter input: Input tensor
    /// - Returns: Output tensor
    /// - Throws: TrustformersError if forward pass fails
    public func forward(input: Tensor) throws -> Tensor {
        var cTensor = input.toCTensor()
        var error = TrustformersError()
        
        guard let outputPtr = model_forward(handle, &cTensor, &error) else {
            throw TrustformersError.fromC(error)
        }
        
        defer { trustformers_free_tensor(outputPtr) }
        return Tensor.fromCTensor(outputPtr.pointee)
    }
}

/// Streaming generation support
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension Model {
    /// Generate text with streaming output
    /// - Parameters:
    ///   - input: Input prompt
    ///   - maxLength: Maximum generation length
    ///   - chunkHandler: Callback for each generated chunk
    /// - Throws: TrustformersError if generation fails
    public func generateStreaming(
        input: String,
        maxLength: Int = 100,
        chunkHandler: @escaping (String) -> Void
    ) async throws {
        // This would be implemented with streaming support in the C API
        // For now, we'll simulate streaming by generating in chunks
        let result = try await generateAsync(input: input, maxLength: maxLength)
        
        // Simulate streaming by sending chunks
        let chunkSize = 10
        for i in stride(from: 0, to: result.count, by: chunkSize) {
            let start = result.index(result.startIndex, offsetBy: i)
            let end = result.index(start, offsetBy: min(chunkSize, result.count - i))
            let chunk = String(result[start..<end])
            
            chunkHandler(chunk)
            
            // Small delay to simulate streaming
            try await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }
    }
}

/// Batch processing support
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension Model {
    /// Generate text for multiple inputs
    /// - Parameters:
    ///   - inputs: Array of input prompts
    ///   - maxLength: Maximum generation length
    /// - Returns: Array of generated texts
    public func generateBatch(inputs: [String], maxLength: Int = 100) async throws -> [String] {
        return try await withThrowingTaskGroup(of: (Int, String).self) { group in
            for (index, input) in inputs.enumerated() {
                group.addTask {
                    let result = try await self.generateAsync(input: input, maxLength: maxLength)
                    return (index, result)
                }
            }
            
            var results = Array<String?>(repeating: nil, count: inputs.count)
            for try await (index, result) in group {
                results[index] = result
            }
            
            return results.compactMap { $0 }
        }
    }
}

// Helper class for async callbacks
private class CallbackContext<T> {
    let continuation: CheckedContinuation<T, Error>
    
    init(continuation: CheckedContinuation<T, Error>) {
        self.continuation = continuation
    }
}