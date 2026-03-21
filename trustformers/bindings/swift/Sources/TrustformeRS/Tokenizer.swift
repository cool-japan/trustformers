import Foundation
import TrustformeRSCore

/// Tokenizer for converting text to tokens and vice versa
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
public final class Tokenizer: @unchecked Sendable {
    private let handle: OpaquePointer
    private let trustformers: TrustformeRS
    
    /// Tokenization result
    public struct TokenizationResult {
        public let tokenIds: [Int32]
        public let tokens: [String]?
        public let attentionMask: [Int32]?
        public let specialTokensMask: [Int32]?
        
        public init(
            tokenIds: [Int32],
            tokens: [String]? = nil,
            attentionMask: [Int32]? = nil,
            specialTokensMask: [Int32]? = nil
        ) {
            self.tokenIds = tokenIds
            self.tokens = tokens
            self.attentionMask = attentionMask
            self.specialTokensMask = specialTokensMask
        }
    }
    
    /// Batch tokenization result
    public struct BatchTokenizationResult {
        public let tokenIds: [[Int32]]
        public let attentionMasks: [[Int32]]?
        public let maxLength: Int
        
        public init(tokenIds: [[Int32]], attentionMasks: [[Int32]]? = nil) {
            self.tokenIds = tokenIds
            self.attentionMasks = attentionMasks
            self.maxLength = tokenIds.map { $0.count }.max() ?? 0
        }
    }
    
    internal init(trustformers: TrustformeRS, path: String) throws {
        self.trustformers = trustformers
        
        var error = TrustformersError()
        guard let handle = tokenizer_load(trustformers.cHandle, path.cString(using: .utf8), &error) else {
            throw TrustformersError.fromC(error)
        }
        
        self.handle = handle
    }
    
    deinit {
        tokenizer_free(handle)
    }
    
    /// Get vocabulary size
    public var vocabSize: Int {
        return Int(tokenizer_get_vocab_size(handle))
    }
    
    /// Get full vocabulary
    public var vocabulary: [String] {
        var size: Int = 0
        guard let vocabPtr = tokenizer_get_vocab(handle, &size) else {
            return []
        }
        
        defer { trustformers_free_string_array(vocabPtr, size) }
        
        var vocab: [String] = []
        for i in 0..<size {
            if let tokenPtr = vocabPtr[i] {
                vocab.append(String(cString: tokenPtr))
            }
        }
        
        return vocab
    }
    
    /// Encode text to token IDs
    /// - Parameter text: Input text to tokenize
    /// - Returns: Tokenization result
    /// - Throws: TrustformersError if tokenization fails
    public func encode(text: String) throws -> TokenizationResult {
        var length: Int = 0
        var error = TrustformersError()
        
        guard let tokensPtr = tokenizer_encode(handle, text.cString(using: .utf8), &length, &error) else {
            throw TrustformersError.fromC(error)
        }
        
        defer { free(tokensPtr) }
        
        let tokenIds = Array(UnsafeBufferPointer(start: tokensPtr, count: length))
        
        return TokenizationResult(tokenIds: tokenIds)
    }
    
    /// Encode multiple texts
    /// - Parameters:
    ///   - texts: Array of input texts
    ///   - padding: Whether to pad sequences to same length
    ///   - truncation: Whether to truncate sequences to max length
    ///   - maxLength: Maximum sequence length
    /// - Returns: Batch tokenization result
    public func encodeBatch(
        texts: [String],
        padding: Bool = true,
        truncation: Bool = true,
        maxLength: Int? = nil
    ) throws -> BatchTokenizationResult {
        var allTokenIds: [[Int32]] = []
        
        for text in texts {
            let result = try encode(text: text)
            var tokenIds = result.tokenIds
            
            // Apply truncation if needed
            if let maxLen = maxLength, truncation && tokenIds.count > maxLen {
                tokenIds = Array(tokenIds.prefix(maxLen))
            }
            
            allTokenIds.append(tokenIds)
        }
        
        // Apply padding if needed
        if padding {
            let maxLen = maxLength ?? allTokenIds.map { $0.count }.max() ?? 0
            for i in 0..<allTokenIds.count {
                while allTokenIds[i].count < maxLen {
                    allTokenIds[i].append(0) // Pad with 0 (typically pad token)
                }
            }
            
            // Create attention masks
            let attentionMasks = allTokenIds.map { tokenIds in
                tokenIds.map { $0 == 0 ? Int32(0) : Int32(1) }
            }
            
            return BatchTokenizationResult(tokenIds: allTokenIds, attentionMasks: attentionMasks)
        }
        
        return BatchTokenizationResult(tokenIds: allTokenIds)
    }
    
    /// Decode token IDs to text
    /// - Parameter tokenIds: Token IDs to decode
    /// - Returns: Decoded text
    /// - Throws: TrustformersError if decoding fails
    public func decode(tokenIds: [Int32]) throws -> String {
        var error = TrustformersError()
        
        guard let textPtr = tokenizer_decode(handle, tokenIds, tokenIds.count, &error) else {
            throw TrustformersError.fromC(error)
        }
        
        defer { trustformers_free_string(textPtr) }
        return String(cString: textPtr)
    }
    
    /// Decode multiple sequences
    /// - Parameter tokenIdsBatch: Array of token ID sequences
    /// - Returns: Array of decoded texts
    public func decodeBatch(tokenIdsBatch: [[Int32]]) throws -> [String] {
        var results: [String] = []
        
        for tokenIds in tokenIdsBatch {
            let text = try decode(tokenIds: tokenIds)
            results.append(text)
        }
        
        return results
    }
    
    /// Get token at specific ID
    /// - Parameter tokenId: Token ID
    /// - Returns: Token string, or nil if ID is invalid
    public func idToToken(_ tokenId: Int32) -> String? {
        let vocab = vocabulary
        let index = Int(tokenId)
        return index < vocab.count ? vocab[index] : nil
    }
    
    /// Get ID for specific token
    /// - Parameter token: Token string
    /// - Returns: Token ID, or nil if token not found
    public func tokenToId(_ token: String) -> Int32? {
        let vocab = vocabulary
        return vocab.firstIndex(of: token).map { Int32($0) }
    }
}

/// Async tokenization support
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension Tokenizer {
    /// Encode text asynchronously
    /// - Parameter text: Input text
    /// - Returns: Tokenization result
    public func encodeAsync(text: String) async throws -> TokenizationResult {
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                do {
                    let result = try self.encode(text: text)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Encode multiple texts asynchronously
    /// - Parameters:
    ///   - texts: Array of input texts
    ///   - padding: Whether to pad sequences
    ///   - truncation: Whether to truncate sequences
    ///   - maxLength: Maximum sequence length
    /// - Returns: Batch tokenization result
    public func encodeBatchAsync(
        texts: [String],
        padding: Bool = true,
        truncation: Bool = true,
        maxLength: Int? = nil
    ) async throws -> BatchTokenizationResult {
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                do {
                    let result = try self.encodeBatch(
                        texts: texts,
                        padding: padding,
                        truncation: truncation,
                        maxLength: maxLength
                    )
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Decode token IDs asynchronously
    /// - Parameter tokenIds: Token IDs to decode
    /// - Returns: Decoded text
    public func decodeAsync(tokenIds: [Int32]) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                do {
                    let result = try self.decode(tokenIds: tokenIds)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

/// Utility extensions
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
extension Tokenizer {
    /// Tokenize and prepare input for model
    /// - Parameters:
    ///   - text: Input text
    ///   - maxLength: Maximum sequence length
    ///   - padding: Padding strategy
    /// - Returns: Model-ready tokenization result
    public func prepareForModel(
        text: String,
        maxLength: Int = 512,
        padding: PaddingStrategy = .maxLength
    ) throws -> ModelInput {
        let result = try encode(text: text)
        var tokenIds = result.tokenIds
        
        // Truncate if necessary
        if tokenIds.count > maxLength {
            tokenIds = Array(tokenIds.prefix(maxLength))
        }
        
        // Apply padding
        let originalLength = tokenIds.count
        switch padding {
        case .maxLength:
            while tokenIds.count < maxLength {
                tokenIds.append(0) // Pad token
            }
        case .none:
            break
        }
        
        // Create attention mask
        let attentionMask = tokenIds.enumerated().map { index, _ in
            Int32(index < originalLength ? 1 : 0)
        }
        
        return ModelInput(
            tokenIds: tokenIds,
            attentionMask: attentionMask,
            originalLength: originalLength
        )
    }
    
    /// Prepare multiple texts for model
    /// - Parameters:
    ///   - texts: Input texts
    ///   - maxLength: Maximum sequence length
    ///   - padding: Padding strategy
    /// - Returns: Batch model input
    public func prepareForModelBatch(
        texts: [String],
        maxLength: Int = 512,
        padding: PaddingStrategy = .maxLength
    ) throws -> BatchModelInput {
        let inputs = try texts.map { text in
            try prepareForModel(text: text, maxLength: maxLength, padding: padding)
        }
        
        return BatchModelInput(inputs: inputs)
    }
}

/// Padding strategies for tokenization
public enum PaddingStrategy {
    case none
    case maxLength
}

/// Model input structure
public struct ModelInput {
    public let tokenIds: [Int32]
    public let attentionMask: [Int32]
    public let originalLength: Int
    
    public init(tokenIds: [Int32], attentionMask: [Int32], originalLength: Int) {
        self.tokenIds = tokenIds
        self.attentionMask = attentionMask
        self.originalLength = originalLength
    }
}

/// Batch model input structure
public struct BatchModelInput {
    public let inputs: [ModelInput]
    public let batchSize: Int
    
    public init(inputs: [ModelInput]) {
        self.inputs = inputs
        self.batchSize = inputs.count
    }
    
    /// Get all token IDs as 2D array
    public var tokenIds: [[Int32]] {
        return inputs.map { $0.tokenIds }
    }
    
    /// Get all attention masks as 2D array
    public var attentionMasks: [[Int32]] {
        return inputs.map { $0.attentionMask }
    }
}