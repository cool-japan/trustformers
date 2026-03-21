import Foundation
import TrustformeRSCore

/// High-level pipeline interface for common NLP tasks
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
public final class Pipeline: @unchecked Sendable {
    private let handle: OpaquePointer
    private let trustformers: TrustformeRS
    private let config: Configuration
    
    /// Supported pipeline tasks
    public enum Task: String, CaseIterable {
        case textGeneration = "text-generation"
        case textClassification = "text-classification"
        case questionAnswering = "question-answering"
        case summarization = "summarization"
        case translation = "translation"
        case tokenClassification = "token-classification"
        case fillMask = "fill-mask"
        case sentimentAnalysis = "sentiment-analysis"
        case zeroShotClassification = "zero-shot-classification"
        case conversational = "conversational"
        
        /// Get human-readable description
        public var description: String {
            switch self {
            case .textGeneration: return "Text Generation"
            case .textClassification: return "Text Classification"
            case .questionAnswering: return "Question Answering"
            case .summarization: return "Summarization"
            case .translation: return "Translation"
            case .tokenClassification: return "Token Classification"
            case .fillMask: return "Fill Mask"
            case .sentimentAnalysis: return "Sentiment Analysis"
            case .zeroShotClassification: return "Zero-shot Classification"
            case .conversational: return "Conversational"
            }
        }
    }
    
    /// Pipeline configuration
    public struct Configuration {
        public let task: Task
        public let modelId: String
        public let device: String
        public let useAuthToken: Bool
        public let revision: String
        public let temperature: Float
        public let maxNewTokens: Int
        public let topK: Int
        public let topP: Float
        public let doSample: Bool
        
        public init(
            task: Task,
            modelId: String,
            device: String = "auto",
            useAuthToken: Bool = false,
            revision: String = "main",
            temperature: Float = 1.0,
            maxNewTokens: Int = 50,
            topK: Int = 50,
            topP: Float = 1.0,
            doSample: Bool = false
        ) {
            self.task = task
            self.modelId = modelId
            self.device = device
            self.useAuthToken = useAuthToken
            self.revision = revision
            self.temperature = temperature
            self.maxNewTokens = maxNewTokens
            self.topK = topK
            self.topP = topP
            self.doSample = doSample
        }
        
        /// Create configuration for text generation
        public static func textGeneration(
            modelId: String,
            temperature: Float = 0.7,
            maxNewTokens: Int = 100
        ) -> Configuration {
            return Configuration(
                task: .textGeneration,
                modelId: modelId,
                temperature: temperature,
                maxNewTokens: maxNewTokens,
                doSample: true
            )
        }
        
        /// Create configuration for text classification
        public static func textClassification(modelId: String) -> Configuration {
            return Configuration(task: .textClassification, modelId: modelId)
        }
        
        /// Create configuration for question answering
        public static func questionAnswering(modelId: String) -> Configuration {
            return Configuration(task: .questionAnswering, modelId: modelId)
        }
    }
    
    /// Classification result
    public struct ClassificationResult {
        public let label: String
        public let score: Float
        public let confidence: Float
        
        public init(label: String, score: Float, confidence: Float = 0.0) {
            self.label = label
            self.score = score
            self.confidence = confidence == 0.0 ? score : confidence
        }
    }
    
    /// Question answering result
    public struct QAResult {
        public let answer: String
        public let score: Float
        public let start: Int?
        public let end: Int?
        
        public init(answer: String, score: Float, start: Int? = nil, end: Int? = nil) {
            self.answer = answer
            self.score = score
            self.start = start
            self.end = end
        }
    }
    
    /// Translation result
    public struct TranslationResult {
        public let translatedText: String
        public let sourceLanguage: String?
        public let targetLanguage: String
        public let confidence: Float?
        
        public init(
            translatedText: String,
            sourceLanguage: String? = nil,
            targetLanguage: String,
            confidence: Float? = nil
        ) {
            self.translatedText = translatedText
            self.sourceLanguage = sourceLanguage
            self.targetLanguage = targetLanguage
            self.confidence = confidence
        }
    }
    
    internal init(trustformers: TrustformeRS, config: Configuration) throws {
        self.trustformers = trustformers
        self.config = config
        
        var cConfig = PipelineConfig()
        cConfig.task = config.task.rawValue.cString(using: .utf8)
        cConfig.model_id = config.modelId.cString(using: .utf8)
        cConfig.device = config.device.cString(using: .utf8)
        cConfig.use_auth_token = config.useAuthToken
        cConfig.revision = config.revision.cString(using: .utf8)
        cConfig.temperature = config.temperature
        cConfig.max_new_tokens = Int32(config.maxNewTokens)
        cConfig.top_k = Int32(config.topK)
        cConfig.top_p = config.topP
        cConfig.do_sample = config.doSample
        
        var error = TrustformersError()
        guard let handle = pipeline_create(trustformers.cHandle, &cConfig, &error) else {
            throw TrustformersError.fromC(error)
        }
        
        self.handle = handle
    }
    
    deinit {
        pipeline_free(handle)
    }
    
    /// Generate text from input prompt
    /// - Parameter input: Input prompt
    /// - Returns: Generated text
    /// - Throws: TrustformersError if generation fails
    public func generate(input: String) throws -> String {
        guard config.task == .textGeneration else {
            throw TrustformersError.invalidArgument("Pipeline task must be text-generation")
        }
        
        var error = TrustformersError()
        guard let resultPtr = pipeline_text_generation(handle, input.cString(using: .utf8), &error) else {
            throw TrustformersError.fromC(error)
        }
        
        defer { trustformers_free_string_array(resultPtr, 1) }
        return String(cString: resultPtr[0])
    }
    
    /// Generate text asynchronously
    /// - Parameter input: Input prompt
    /// - Returns: Generated text
    public func generateAsync(input: String) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                do {
                    let result = try self.generate(input: input)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Classify text
    /// - Parameter input: Input text
    /// - Returns: Classification results
    /// - Throws: TrustformersError if classification fails
    public func classify(input: String) throws -> [ClassificationResult] {
        guard config.task == .textClassification || config.task == .sentimentAnalysis else {
            throw TrustformersError.invalidArgument("Pipeline task must be text-classification or sentiment-analysis")
        }
        
        var numLabels: Int = 0
        var error = TrustformersError()
        guard let scoresPtr = pipeline_text_classification(handle, input.cString(using: .utf8), &numLabels, &error) else {
            throw TrustformersError.fromC(error)
        }
        
        defer { free(scoresPtr) }
        
        var results: [ClassificationResult] = []
        for i in 0..<numLabels {
            let score = scoresPtr[i]
            let label = "LABEL_\(i)" // In real implementation, get actual labels
            results.append(ClassificationResult(label: label, score: score))
        }
        
        return results.sorted { $0.score > $1.score }
    }
    
    /// Classify text asynchronously
    /// - Parameter input: Input text
    /// - Returns: Classification results
    public func classifyAsync(input: String) async throws -> [ClassificationResult] {
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                do {
                    let results = try self.classify(input: input)
                    continuation.resume(returning: results)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Answer a question given context
    /// - Parameters:
    ///   - question: Question to answer
    ///   - context: Context containing the answer
    /// - Returns: Question answering result
    /// - Throws: TrustformersError if QA fails
    public func answerQuestion(question: String, context: String) throws -> QAResult {
        guard config.task == .questionAnswering else {
            throw TrustformersError.invalidArgument("Pipeline task must be question-answering")
        }
        
        var error = TrustformersError()
        guard let answerPtr = pipeline_question_answering(
            handle,
            question.cString(using: .utf8),
            context.cString(using: .utf8),
            &error
        ) else {
            throw TrustformersError.fromC(error)
        }
        
        defer { trustformers_free_string(answerPtr) }
        let answer = String(cString: answerPtr)
        
        return QAResult(answer: answer, score: 1.0) // In real implementation, get actual score
    }
    
    /// Summarize text
    /// - Parameter input: Input text to summarize
    /// - Returns: Summarized text
    /// - Throws: TrustformersError if summarization fails
    public func summarize(input: String) throws -> String {
        guard config.task == .summarization else {
            throw TrustformersError.invalidArgument("Pipeline task must be summarization")
        }
        
        var error = TrustformersError()
        guard let summaryPtr = pipeline_summarization(handle, input.cString(using: .utf8), &error) else {
            throw TrustformersError.fromC(error)
        }
        
        defer { trustformers_free_string(summaryPtr) }
        return String(cString: summaryPtr)
    }
    
    /// Translate text
    /// - Parameters:
    ///   - input: Input text to translate
    ///   - targetLanguage: Target language code (e.g., "en", "fr", "es")
    /// - Returns: Translation result
    /// - Throws: TrustformersError if translation fails
    public func translate(input: String, targetLanguage: String) throws -> TranslationResult {
        guard config.task == .translation else {
            throw TrustformersError.invalidArgument("Pipeline task must be translation")
        }
        
        var error = TrustformersError()
        guard let translationPtr = pipeline_translation(
            handle,
            input.cString(using: .utf8),
            targetLanguage.cString(using: .utf8),
            &error
        ) else {
            throw TrustformersError.fromC(error)
        }
        
        defer { trustformers_free_string(translationPtr) }
        let translatedText = String(cString: translationPtr)
        
        return TranslationResult(
            translatedText: translatedText,
            targetLanguage: targetLanguage
        )
    }
}

/// Batch processing extensions
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension Pipeline {
    /// Process multiple inputs in batch
    /// - Parameter inputs: Array of input texts
    /// - Returns: Array of results (type depends on pipeline task)
    public func processBatch<T>(inputs: [String], processor: (String) async throws -> T) async throws -> [T] {
        return try await withThrowingTaskGroup(of: (Int, T).self) { group in
            for (index, input) in inputs.enumerated() {
                group.addTask {
                    let result = try await processor(input)
                    return (index, result)
                }
            }
            
            var results = Array<T?>(repeating: nil, count: inputs.count)
            for try await (index, result) in group {
                results[index] = result
            }
            
            return results.compactMap { $0 }
        }
    }
    
    /// Generate text for multiple prompts
    /// - Parameter inputs: Array of input prompts
    /// - Returns: Array of generated texts
    public func generateBatch(inputs: [String]) async throws -> [String] {
        return try await processBatch(inputs: inputs) { input in
            try await self.generateAsync(input: input)
        }
    }
    
    /// Classify multiple texts
    /// - Parameter inputs: Array of input texts
    /// - Returns: Array of classification results
    public func classifyBatch(inputs: [String]) async throws -> [[ClassificationResult]] {
        return try await processBatch(inputs: inputs) { input in
            try await self.classifyAsync(input: input)
        }
    }
}

/// Streaming support
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension Pipeline {
    /// Generate text with streaming output
    /// - Parameters:
    ///   - input: Input prompt
    ///   - chunkHandler: Callback for each generated chunk
    /// - Throws: TrustformersError if generation fails
    public func generateStreaming(
        input: String,
        chunkHandler: @escaping (String) -> Void
    ) async throws {
        guard config.task == .textGeneration else {
            throw TrustformersError.invalidArgument("Pipeline task must be text-generation")
        }
        
        // Simulate streaming by generating text and sending in chunks
        let result = try await generateAsync(input: input)
        
        let chunkSize = 5
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