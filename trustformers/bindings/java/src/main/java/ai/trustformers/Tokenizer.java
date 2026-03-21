package ai.trustformers;

import java.util.List;
import java.util.Map;

/**
 * TrustFormeRS Tokenizer wrapper providing text tokenization functionality
 * for preprocessing text data for transformer models.
 */
public class Tokenizer implements AutoCloseable {
    
    private long nativeHandle;
    private boolean closed = false;
    
    // Native method declarations
    private static native long loadTokenizerNative(String tokenizerPath);
    private static native long loadTokenizerFromHubNative(String tokenizerName, String revision, String authToken);
    private static native void unloadTokenizerNative(long handle);
    private static native int[] encodeNative(long handle, String text, boolean addSpecialTokens);
    private static native String decodeNative(long handle, int[] tokenIds, boolean skipSpecialTokens);
    private static native String[] tokenizeNative(long handle, String text);
    private static native int getVocabSizeNative(long handle);
    private static native String getTokenizerInfoNative(long handle);
    private static native Map<String, Object> getTokenizerConfigNative(long handle);
    private static native boolean saveTokenizerNative(long handle, String savePath);
    private static native int[] encodeBatchNative(long handle, String[] texts, boolean addSpecialTokens, int maxLength);
    private static native String[] decodeBatchNative(long handle, int[][] tokenIds, boolean skipSpecialTokens);
    
    /**
     * Load a tokenizer from a local path
     * @param tokenizerPath Path to the tokenizer directory or file
     * @throws TrustFormeRSException if tokenizer loading fails
     */
    public Tokenizer(String tokenizerPath) throws TrustFormeRSException {
        this.nativeHandle = loadTokenizerNative(tokenizerPath);
        if (this.nativeHandle == 0) {
            throw new TrustFormeRSException("Failed to load tokenizer from: " + tokenizerPath);
        }
    }
    
    /**
     * Load a tokenizer from Hugging Face Hub
     * @param tokenizerName Tokenizer name (e.g., "gpt2", "bert-base-uncased")
     * @param revision Tokenizer revision/branch (can be null for default)
     * @param authToken Authentication token (can be null for public tokenizers)
     * @throws TrustFormeRSException if tokenizer loading fails
     */
    public static Tokenizer fromHub(String tokenizerName, String revision, String authToken) throws TrustFormeRSException {
        Tokenizer tokenizer = new Tokenizer();
        tokenizer.nativeHandle = loadTokenizerFromHubNative(tokenizerName, revision, authToken);
        if (tokenizer.nativeHandle == 0) {
            throw new TrustFormeRSException("Failed to load tokenizer from Hub: " + tokenizerName);
        }
        return tokenizer;
    }
    
    /**
     * Load a tokenizer from Hugging Face Hub (default revision, no auth)
     * @param tokenizerName Tokenizer name
     * @throws TrustFormeRSException if tokenizer loading fails
     */
    public static Tokenizer fromHub(String tokenizerName) throws TrustFormeRSException {
        return fromHub(tokenizerName, null, null);
    }
    
    /**
     * Private constructor for fromHub factory methods
     */
    private Tokenizer() {
        // Used by fromHub factory methods
    }
    
    /**
     * Encode text to token IDs
     * @param text Input text to encode
     * @param addSpecialTokens Whether to add special tokens (BOS, EOS, etc.)
     * @return Array of token IDs
     * @throws TrustFormeRSException if encoding fails
     */
    public int[] encode(String text, boolean addSpecialTokens) throws TrustFormeRSException {
        checkNotClosed();
        int[] tokens = encodeNative(nativeHandle, text, addSpecialTokens);
        if (tokens == null) {
            throw new TrustFormeRSException("Failed to encode text");
        }
        return tokens;
    }
    
    /**
     * Encode text to token IDs with special tokens
     * @param text Input text to encode
     * @return Array of token IDs
     * @throws TrustFormeRSException if encoding fails
     */
    public int[] encode(String text) throws TrustFormeRSException {
        return encode(text, true);
    }
    
    /**
     * Decode token IDs to text
     * @param tokenIds Array of token IDs to decode
     * @param skipSpecialTokens Whether to skip special tokens in output
     * @return Decoded text
     * @throws TrustFormeRSException if decoding fails
     */
    public String decode(int[] tokenIds, boolean skipSpecialTokens) throws TrustFormeRSException {
        checkNotClosed();
        String text = decodeNative(nativeHandle, tokenIds, skipSpecialTokens);
        if (text == null) {
            throw new TrustFormeRSException("Failed to decode tokens");
        }
        return text;
    }
    
    /**
     * Decode token IDs to text, skipping special tokens
     * @param tokenIds Array of token IDs to decode
     * @return Decoded text
     * @throws TrustFormeRSException if decoding fails
     */
    public String decode(int[] tokenIds) throws TrustFormeRSException {
        return decode(tokenIds, true);
    }
    
    /**
     * Tokenize text into individual tokens (string representation)
     * @param text Input text to tokenize
     * @return Array of token strings
     * @throws TrustFormeRSException if tokenization fails
     */
    public String[] tokenize(String text) throws TrustFormeRSException {
        checkNotClosed();
        String[] tokens = tokenizeNative(nativeHandle, text);
        if (tokens == null) {
            throw new TrustFormeRSException("Failed to tokenize text");
        }
        return tokens;
    }
    
    /**
     * Encode multiple texts in batch
     * @param texts Array of input texts
     * @param addSpecialTokens Whether to add special tokens
     * @param maxLength Maximum sequence length (for padding/truncation)
     * @return Flattened array of token IDs (requires reshaping based on batch size)
     * @throws TrustFormeRSException if batch encoding fails
     */
    public int[] encodeBatch(String[] texts, boolean addSpecialTokens, int maxLength) throws TrustFormeRSException {
        checkNotClosed();
        int[] tokens = encodeBatchNative(nativeHandle, texts, addSpecialTokens, maxLength);
        if (tokens == null) {
            throw new TrustFormeRSException("Failed to encode text batch");
        }
        return tokens;
    }
    
    /**
     * Decode multiple token sequences in batch
     * @param tokenIds 2D array of token ID sequences
     * @param skipSpecialTokens Whether to skip special tokens in output
     * @return Array of decoded texts
     * @throws TrustFormeRSException if batch decoding fails
     */
    public String[] decodeBatch(int[][] tokenIds, boolean skipSpecialTokens) throws TrustFormeRSException {
        checkNotClosed();
        String[] texts = decodeBatchNative(nativeHandle, tokenIds, skipSpecialTokens);
        if (texts == null) {
            throw new TrustFormeRSException("Failed to decode token batch");
        }
        return texts;
    }
    
    /**
     * Get the vocabulary size of the tokenizer
     * @return Vocabulary size
     * @throws TrustFormeRSException if tokenizer is closed
     */
    public int getVocabSize() throws TrustFormeRSException {
        checkNotClosed();
        return getVocabSizeNative(nativeHandle);
    }
    
    /**
     * Get tokenizer information
     * @return Tokenizer information string
     * @throws TrustFormeRSException if tokenizer is closed
     */
    public String getTokenizerInfo() throws TrustFormeRSException {
        checkNotClosed();
        return getTokenizerInfoNative(nativeHandle);
    }
    
    /**
     * Get tokenizer configuration
     * @return Tokenizer configuration as Map
     * @throws TrustFormeRSException if tokenizer is closed
     */
    public Map<String, Object> getTokenizerConfig() throws TrustFormeRSException {
        checkNotClosed();
        return getTokenizerConfigNative(nativeHandle);
    }
    
    /**
     * Save the tokenizer to a local path
     * @param savePath Path to save the tokenizer
     * @return true if save was successful
     * @throws TrustFormeRSException if tokenizer is closed or save fails
     */
    public boolean saveTokenizer(String savePath) throws TrustFormeRSException {
        checkNotClosed();
        return saveTokenizerNative(nativeHandle, savePath);
    }
    
    /**
     * Check if the tokenizer is closed
     * @return true if the tokenizer has been closed
     */
    public boolean isClosed() {
        return closed;
    }
    
    /**
     * Close the tokenizer and free native resources
     */
    @Override
    public void close() {
        if (!closed && nativeHandle != 0) {
            unloadTokenizerNative(nativeHandle);
            nativeHandle = 0;
            closed = true;
        }
    }
    
    /**
     * Finalize method to ensure native resources are cleaned up
     */
    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }
    
    /**
     * Check that the tokenizer is not closed and throw exception if it is
     * @throws TrustFormeRSException if tokenizer is closed
     */
    private void checkNotClosed() throws TrustFormeRSException {
        if (closed) {
            throw new TrustFormeRSException("Tokenizer has been closed");
        }
    }
}