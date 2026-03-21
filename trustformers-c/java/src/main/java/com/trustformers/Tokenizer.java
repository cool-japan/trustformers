package com.trustformers;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Text tokenizer for processing natural language text.
 * Handles encoding text to tokens and decoding tokens back to text.
 */
public class Tokenizer implements Closeable {
    
    private static final Logger logger = LoggerFactory.getLogger(Tokenizer.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    
    private final TrustformeRS trustformers;
    private long nativeHandle;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    
    /**
     * Tokenizer information structure.
     */
    public static class TokenizerInfo {
        public final String type;
        public final int vocabSize;
        public final String modelName;
        public final JsonNode metadata;
        
        public TokenizerInfo(String type, int vocabSize, String modelName, JsonNode metadata) {
            this.type = type;
            this.vocabSize = vocabSize;
            this.modelName = modelName;
            this.metadata = metadata;
        }
        
        @Override
        public String toString() {
            return String.format("TokenizerInfo{type='%s', vocabSize=%d, modelName='%s'}",
                type, vocabSize, modelName);
        }
    }
    
    /**
     * Special tokens used by the tokenizer.
     */
    public static class SpecialTokens {
        public final Integer bos;  // Beginning of sequence
        public final Integer eos;  // End of sequence
        public final Integer unk;  // Unknown token
        public final Integer sep;  // Separator
        public final Integer pad;  // Padding
        public final Integer cls;  // Classification
        public final Integer mask; // Mask token
        
        public SpecialTokens(Integer bos, Integer eos, Integer unk, Integer sep, 
                           Integer pad, Integer cls, Integer mask) {
            this.bos = bos;
            this.eos = eos;
            this.unk = unk;
            this.sep = sep;
            this.pad = pad;
            this.cls = cls;
            this.mask = mask;
        }
        
        @Override
        public String toString() {
            return String.format("SpecialTokens{bos=%s, eos=%s, unk=%s, sep=%s, pad=%s, cls=%s, mask=%s}",
                bos, eos, unk, sep, pad, cls, mask);
        }
    }
    
    // Package-private constructor
    Tokenizer(TrustformeRS trustformers, long nativeHandle) {
        this.trustformers = trustformers;
        this.nativeHandle = nativeHandle;
    }
    
    /**
     * Load a tokenizer from Hugging Face Hub.
     * 
     * @param trustformers TrustformeRS instance
     * @param modelName name of the tokenizer to load
     * @return loaded tokenizer instance
     * @throws TrustformersException if loading fails
     */
    public static Tokenizer loadFromHub(TrustformeRS trustformers, String modelName) throws TrustformersException {
        if (modelName == null || modelName.trim().isEmpty()) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER, 
                "Model name cannot be null or empty");
        }
        
        long handle = nativeLoadFromHub(modelName);
        if (handle == 0) {
            throw new TrustformersException(TrustformersException.ErrorCode.RUNTIME_ERROR,
                "Failed to load tokenizer from Hub: " + modelName);
        }
        
        Tokenizer tokenizer = new Tokenizer(trustformers, handle);
        logger.info("Successfully loaded tokenizer from Hub: {}", modelName);
        return tokenizer;
    }
    
    /**
     * Load a tokenizer from a local path.
     * 
     * @param trustformers TrustformeRS instance
     * @param tokenizerPath path to the tokenizer files
     * @return loaded tokenizer instance
     * @throws TrustformersException if loading fails
     */
    public static Tokenizer loadFromPath(TrustformeRS trustformers, String tokenizerPath) throws TrustformersException {
        if (tokenizerPath == null || tokenizerPath.trim().isEmpty()) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER, 
                "Tokenizer path cannot be null or empty");
        }
        
        long handle = nativeLoadFromPath(tokenizerPath);
        if (handle == 0) {
            throw new TrustformersException(TrustformersException.ErrorCode.RUNTIME_ERROR,
                "Failed to load tokenizer from path: " + tokenizerPath);
        }
        
        Tokenizer tokenizer = new Tokenizer(trustformers, handle);
        logger.info("Successfully loaded tokenizer from path: {}", tokenizerPath);
        return tokenizer;
    }
    
    /**
     * Free the tokenizer and release its resources.
     * 
     * @throws TrustformersException if freeing fails
     */
    public void free() throws TrustformersException {
        if (closed.get() || nativeHandle == 0) {
            return;
        }
        
        synchronized (this) {
            if (closed.get() || nativeHandle == 0) {
                return;
            }
            
            checkError(nativeFree(nativeHandle));
            nativeHandle = 0;
            closed.set(true);
        }
        
        logger.debug("Tokenizer resources freed");
    }
    
    /**
     * Close the tokenizer and release resources.
     */
    @Override
    public void close() {
        try {
            free();
        } catch (TrustformersException e) {
            logger.warn("Error closing tokenizer", e);
        }
    }
    
    /**
     * Encode text into tokens.
     * 
     * @param text text to encode
     * @return array of token IDs
     * @throws TrustformersException if encoding fails
     */
    public int[] encode(String text) throws TrustformersException {
        checkNotClosed();
        
        if (text == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Text cannot be null");
        }
        
        return nativeEncode(nativeHandle, text);
    }
    
    /**
     * Decode tokens back to text.
     * 
     * @param tokens token IDs to decode
     * @return decoded text
     * @throws TrustformersException if decoding fails
     */
    public String decode(int[] tokens) throws TrustformersException {
        checkNotClosed();
        
        if (tokens == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Tokens cannot be null");
        }
        
        return nativeDecode(nativeHandle, tokens);
    }
    
    /**
     * Encode multiple texts in a batch for better performance.
     * 
     * @param texts list of texts to encode
     * @return array of token arrays
     * @throws TrustformersException if encoding fails
     */
    public int[][] encodeBatch(List<String> texts) throws TrustformersException {
        checkNotClosed();
        
        if (texts == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Texts cannot be null");
        }
        
        String[] textArray = texts.toArray(new String[0]);
        return nativeEncodeBatch(nativeHandle, textArray);
    }
    
    /**
     * Encode multiple texts in a batch for better performance.
     * 
     * @param texts array of texts to encode
     * @return array of token arrays
     * @throws TrustformersException if encoding fails
     */
    public int[][] encodeBatch(String[] texts) throws TrustformersException {
        checkNotClosed();
        
        if (texts == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Texts cannot be null");
        }
        
        return nativeEncodeBatch(nativeHandle, texts);
    }
    
    /**
     * Decode multiple token sequences in a batch for better performance.
     * 
     * @param tokenBatches array of token arrays to decode
     * @return array of decoded texts
     * @throws TrustformersException if decoding fails
     */
    public String[] decodeBatch(int[][] tokenBatches) throws TrustformersException {
        checkNotClosed();
        
        if (tokenBatches == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Token batches cannot be null");
        }
        
        return nativeDecodeBatch(nativeHandle, tokenBatches);
    }
    
    /**
     * Get the vocabulary size of the tokenizer.
     * 
     * @return vocabulary size
     * @throws TrustformersException if the operation fails
     */
    public int getVocabSize() throws TrustformersException {
        checkNotClosed();
        
        int[] result = new int[1];
        checkError(nativeGetVocabSize(nativeHandle, result));
        return result[0];
    }
    
    /**
     * Get special tokens used by the tokenizer.
     * 
     * @return special tokens information
     * @throws TrustformersException if the operation fails
     */
    public SpecialTokens getSpecialTokens() throws TrustformersException {
        checkNotClosed();
        
        String jsonStr = nativeGetSpecialTokens(nativeHandle);
        if (jsonStr == null || jsonStr.trim().isEmpty()) {
            return new SpecialTokens(null, null, null, null, null, null, null);
        }
        
        try {
            JsonNode json = objectMapper.readTree(jsonStr);
            return new SpecialTokens(
                getOptionalInt(json, "bos"),
                getOptionalInt(json, "eos"),
                getOptionalInt(json, "unk"),
                getOptionalInt(json, "sep"),
                getOptionalInt(json, "pad"),
                getOptionalInt(json, "cls"),
                getOptionalInt(json, "mask")
            );
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse special tokens JSON", e);
        }
    }
    
    /**
     * Add a special token to the tokenizer.
     * 
     * @param token token string
     * @param tokenId token ID
     * @throws TrustformersException if the operation fails
     */
    public void addSpecialToken(String token, int tokenId) throws TrustformersException {
        checkNotClosed();
        
        if (token == null || token.isEmpty()) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Token cannot be null or empty");
        }
        
        checkError(nativeAddSpecialToken(nativeHandle, token, tokenId));
        logger.debug("Added special token: {} -> {}", token, tokenId);
    }
    
    /**
     * Get detailed tokenizer information.
     * 
     * @return tokenizer information
     * @throws TrustformersException if the operation fails
     */
    public TokenizerInfo getInfo() throws TrustformersException {
        checkNotClosed();
        
        String jsonStr = nativeGetInfo(nativeHandle);
        if (jsonStr == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.RUNTIME_ERROR,
                "Failed to get tokenizer info");
        }
        
        try {
            JsonNode json = objectMapper.readTree(jsonStr);
            return new TokenizerInfo(
                json.path("type").asText(),
                json.path("vocab_size").asInt(),
                json.path("model_name").asText(),
                json.path("metadata")
            );
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse tokenizer info JSON", e);
        }
    }
    
    /**
     * Check if the tokenizer is loaded and not closed.
     * 
     * @return true if the tokenizer is loaded
     */
    public boolean isLoaded() {
        return !closed.get() && nativeHandle != 0;
    }
    
    /**
     * Get the native handle for this tokenizer.
     * This method is package-private and used internally by other classes.
     * 
     * @return native handle
     */
    long getNativeHandle() {
        checkNotClosed();
        return nativeHandle;
    }
    
    // Private helper methods
    
    private void checkNotClosed() {
        if (closed.get() || nativeHandle == 0) {
            throw new IllegalStateException("Tokenizer has been closed or not properly loaded");
        }
    }
    
    private void checkError(int errorCode) throws TrustformersException {
        if (errorCode != 0) {
            TrustformersException.ErrorCode code = TrustformersException.ErrorCode.fromCode(errorCode);
            throw new TrustformersException(code, "Tokenizer operation failed with error code: " + errorCode);
        }
    }
    
    private Integer getOptionalInt(JsonNode json, String field) {
        JsonNode node = json.path(field);
        return node.isMissingNode() || node.isNull() ? null : node.asInt();
    }
    
    @Override
    protected void finalize() throws Throwable {
        try {
            if (!closed.get() && nativeHandle != 0) {
                logger.warn("Tokenizer was not explicitly closed, freeing in finalizer");
                free();
            }
        } catch (Exception e) {
            logger.warn("Error in tokenizer finalizer", e);
        } finally {
            super.finalize();
        }
    }
    
    // Native method declarations
    
    private static native long nativeLoadFromHub(String modelName);
    private static native long nativeLoadFromPath(String tokenizerPath);
    private static native int nativeFree(long handle);
    private static native int[] nativeEncode(long handle, String text);
    private static native String nativeDecode(long handle, int[] tokens);
    private static native int[][] nativeEncodeBatch(long handle, String[] texts);
    private static native String[] nativeDecodeBatch(long handle, int[][] tokenBatches);
    private static native int nativeGetVocabSize(long handle, int[] result);
    private static native String nativeGetSpecialTokens(long handle);
    private static native int nativeAddSpecialToken(long handle, String token, int tokenId);
    private static native String nativeGetInfo(long handle);
}