import Foundation
import TrustformeRSCore

/// Main TrustformeRS class providing high-level API for transformer models
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
public final class TrustformeRS: @unchecked Sendable {
    private let handle: OpaquePointer
    private let config: Configuration
    
    /// Configuration for TrustformeRS initialization
    public struct Configuration {
        public let useGPU: Bool
        public let device: String?
        public let numThreads: Int
        public let enableLogging: Bool
        public let cacheDir: String?
        
        public init(
            useGPU: Bool = true,
            device: String? = nil,
            numThreads: Int = -1,
            enableLogging: Bool = false,
            cacheDir: String? = nil
        ) {
            self.useGPU = useGPU
            self.device = device
            self.numThreads = numThreads
            self.enableLogging = enableLogging
            self.cacheDir = cacheDir
        }
        
        /// Default configuration optimized for the current platform
        public static let `default` = Configuration()
        
        /// iOS optimized configuration
        public static let iOS = Configuration(
            useGPU: true,
            device: "auto",
            numThreads: 4,
            enableLogging: false
        )
        
        /// macOS optimized configuration
        public static let macOS = Configuration(
            useGPU: true,
            device: "auto",
            numThreads: -1,
            enableLogging: false
        )
    }
    
    /// Initialize TrustformeRS with configuration
    /// - Parameter config: Configuration object
    /// - Throws: TrustformersError if initialization fails
    public init(config: Configuration = .default) throws {
        self.config = config
        
        var cConfig = TrustformersConfig()
        cConfig.use_gpu = config.useGPU
        cConfig.device = config.device?.cString(using: .utf8)
        cConfig.num_threads = Int32(config.numThreads)
        cConfig.enable_logging = config.enableLogging
        cConfig.cache_dir = config.cacheDir?.cString(using: .utf8)
        
        var error = TrustformersError()
        guard let handle = trustformers_init(&cConfig, &error) else {
            throw TrustformersError.fromC(error)
        }
        
        self.handle = handle
    }
    
    deinit {
        trustformers_free(handle)
    }
    
    /// Get TrustformeRS version
    public static var version: String {
        guard let versionPtr = trustformers_get_version() else {
            return "unknown"
        }
        defer { trustformers_free_string(versionPtr) }
        return String(cString: versionPtr)
    }
    
    /// Check if GPU is available
    public static var isGPUAvailable: Bool {
        return trustformers_is_gpu_available()
    }
    
    #if os(macOS) || os(iOS)
    /// Check if Metal backend is available (iOS/macOS only)
    public static var isMetalAvailable: Bool {
        return trustformers_is_metal_available()
    }
    
    /// Check if CoreML backend is available (iOS/macOS only)
    public static var isCoreMLAvailable: Bool {
        return trustformers_is_coreml_available()
    }
    
    /// Enable Metal backend for GPU acceleration
    /// - Throws: TrustformersError if Metal cannot be enabled
    public func enableMetalBackend() throws {
        var error = TrustformersError()
        let success = trustformers_enable_metal_backend(handle, &error)
        if !success {
            throw TrustformersError.fromC(error)
        }
    }
    
    /// Enable CoreML backend for inference
    /// - Throws: TrustformersError if CoreML cannot be enabled
    public func enableCoreMLBackend() throws {
        var error = TrustformersError()
        let success = trustformers_enable_coreml_backend(handle, &error)
        if !success {
            throw TrustformersError.fromC(error)
        }
    }
    #endif
    
    /// Load a model from the specified path or model ID
    /// - Parameter config: Model configuration
    /// - Returns: A Model instance
    /// - Throws: TrustformersError if model loading fails
    public func loadModel(config: Model.Configuration) throws -> Model {
        return try Model(trustformers: self, config: config)
    }
    
    /// Load a tokenizer from the specified path
    /// - Parameter path: Path to tokenizer files
    /// - Returns: A Tokenizer instance
    /// - Throws: TrustformersError if tokenizer loading fails
    public func loadTokenizer(path: String) throws -> Tokenizer {
        return try Tokenizer(trustformers: self, path: path)
    }
    
    /// Create a pipeline for a specific task
    /// - Parameter config: Pipeline configuration
    /// - Returns: A Pipeline instance
    /// - Throws: TrustformersError if pipeline creation fails
    public func createPipeline(config: Pipeline.Configuration) throws -> Pipeline {
        return try Pipeline(trustformers: self, config: config)
    }
    
    // Internal accessor for C handle
    internal var cHandle: OpaquePointer {
        return handle
    }
}

/// Error types for TrustformeRS operations
public enum TrustformersError: Error, LocalizedError, CustomStringConvertible {
    case invalidArgument(String)
    case modelNotFound(String)
    case tokenizerError(String)
    case inferenceError(String)
    case memoryError(String)
    case ioError(String)
    case unknown(String)
    
    public var description: String {
        switch self {
        case .invalidArgument(let msg):
            return "Invalid argument: \(msg)"
        case .modelNotFound(let msg):
            return "Model not found: \(msg)"
        case .tokenizerError(let msg):
            return "Tokenizer error: \(msg)"
        case .inferenceError(let msg):
            return "Inference error: \(msg)"
        case .memoryError(let msg):
            return "Memory error: \(msg)"
        case .ioError(let msg):
            return "I/O error: \(msg)"
        case .unknown(let msg):
            return "Unknown error: \(msg)"
        }
    }
    
    public var errorDescription: String? {
        return description
    }
    
    static func fromC(_ cError: TrustformersError) -> TrustformersError {
        let message = cError.message != nil ? String(cString: cError.message) : "No message"
        defer { trustformers_free_error(UnsafeMutablePointer(mutating: &cError)) }
        
        switch cError.code {
        case TRUSTFORMERS_ERROR_INVALID_ARGUMENT:
            return .invalidArgument(message)
        case TRUSTFORMERS_ERROR_MODEL_NOT_FOUND:
            return .modelNotFound(message)
        case TRUSTFORMERS_ERROR_TOKENIZER_ERROR:
            return .tokenizerError(message)
        case TRUSTFORMERS_ERROR_INFERENCE_ERROR:
            return .inferenceError(message)
        case TRUSTFORMERS_ERROR_MEMORY_ERROR:
            return .memoryError(message)
        case TRUSTFORMERS_ERROR_IO_ERROR:
            return .ioError(message)
        default:
            return .unknown(message)
        }
    }
}

/// Extension for async operations
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension TrustformeRS {
    /// Create pipeline with async configuration loading
    /// - Parameter config: Pipeline configuration
    /// - Returns: A Pipeline instance
    public func createPipelineAsync(config: Pipeline.Configuration) async throws -> Pipeline {
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                do {
                    let pipeline = try self.createPipeline(config: config)
                    continuation.resume(returning: pipeline)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

/// Convenience extensions for common operations
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension TrustformeRS {
    /// Quick text generation with default settings
    /// - Parameters:
    ///   - modelId: Hugging Face model ID or local path
    ///   - prompt: Input prompt
    ///   - maxLength: Maximum generation length
    /// - Returns: Generated text
    public func quickGenerate(modelId: String, prompt: String, maxLength: Int = 100) async throws -> String {
        let pipelineConfig = Pipeline.Configuration(
            task: .textGeneration,
            modelId: modelId,
            maxNewTokens: maxLength
        )
        
        let pipeline = try await createPipelineAsync(config: pipelineConfig)
        return try await pipeline.generateAsync(input: prompt)
    }
    
    /// Quick text classification with default settings
    /// - Parameters:
    ///   - modelId: Hugging Face model ID or local path
    ///   - text: Input text to classify
    /// - Returns: Classification results
    public func quickClassify(modelId: String, text: String) async throws -> [Pipeline.ClassificationResult] {
        let pipelineConfig = Pipeline.Configuration(
            task: .textClassification,
            modelId: modelId
        )
        
        let pipeline = try await createPipelineAsync(config: pipelineConfig)
        return try await pipeline.classifyAsync(input: text)
    }
}