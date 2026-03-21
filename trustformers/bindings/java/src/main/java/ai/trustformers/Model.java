package ai.trustformers;

import java.util.Map;
import java.util.HashMap;

/**
 * TrustFormeRS Model wrapper providing access to transformer models
 * for text generation and other ML tasks.
 */
public class Model implements AutoCloseable {
    
    private long nativeHandle;
    private boolean closed = false;
    
    // Native method declarations
    private static native long loadModelNative(String modelPath);
    private static native long loadModelFromHubNative(String modelName, String revision, String authToken);
    private static native void unloadModelNative(long handle);
    private static native String getModelInfoNative(long handle);
    private static native String generateTextNative(long handle, String prompt, Map<String, Object> config);
    private static native float[] getEmbeddingsNative(long handle, String text);
    private static native Map<String, Object> getModelConfigNative(long handle);
    private static native boolean saveModelNative(long handle, String savePath);
    
    /**
     * Load a model from a local path
     * @param modelPath Path to the model directory or file
     * @throws TrustFormeRSException if model loading fails
     */
    public Model(String modelPath) throws TrustFormeRSException {
        this.nativeHandle = loadModelNative(modelPath);
        if (this.nativeHandle == 0) {
            throw new TrustFormeRSException("Failed to load model from: " + modelPath);
        }
    }
    
    /**
     * Load a model from Hugging Face Hub
     * @param modelName Model name (e.g., "gpt2", "bert-base-uncased")
     * @param revision Model revision/branch (can be null for default)
     * @param authToken Authentication token (can be null for public models)
     * @throws TrustFormeRSException if model loading fails
     */
    public static Model fromHub(String modelName, String revision, String authToken) throws TrustFormeRSException {
        Model model = new Model();
        model.nativeHandle = loadModelFromHubNative(modelName, revision, authToken);
        if (model.nativeHandle == 0) {
            throw new TrustFormeRSException("Failed to load model from Hub: " + modelName);
        }
        return model;
    }
    
    /**
     * Load a model from Hugging Face Hub (default revision, no auth)
     * @param modelName Model name (e.g., "gpt2", "bert-base-uncased")
     * @throws TrustFormeRSException if model loading fails
     */
    public static Model fromHub(String modelName) throws TrustFormeRSException {
        return fromHub(modelName, null, null);
    }
    
    /**
     * Private constructor for fromHub factory methods
     */
    private Model() {
        // Used by fromHub factory methods
    }
    
    /**
     * Generate text using the model
     * @param prompt Input prompt text
     * @param config Generation configuration (temperature, max_length, etc.)
     * @return Generated text
     * @throws TrustFormeRSException if generation fails
     */
    public String generateText(String prompt, Map<String, Object> config) throws TrustFormeRSException {
        checkNotClosed();
        String result = generateTextNative(nativeHandle, prompt, config);
        if (result == null) {
            throw new TrustFormeRSException("Text generation failed");
        }
        return result;
    }
    
    /**
     * Generate text with default configuration
     * @param prompt Input prompt text
     * @return Generated text
     * @throws TrustFormeRSException if generation fails
     */
    public String generateText(String prompt) throws TrustFormeRSException {
        Map<String, Object> defaultConfig = new HashMap<>();
        defaultConfig.put("max_length", 100);
        defaultConfig.put("temperature", 0.7);
        defaultConfig.put("do_sample", true);
        return generateText(prompt, defaultConfig);
    }
    
    /**
     * Get text embeddings from the model
     * @param text Input text
     * @return Embedding vector as float array
     * @throws TrustFormeRSException if embedding extraction fails
     */
    public float[] getEmbeddings(String text) throws TrustFormeRSException {
        checkNotClosed();
        float[] embeddings = getEmbeddingsNative(nativeHandle, text);
        if (embeddings == null) {
            throw new TrustFormeRSException("Failed to get embeddings");
        }
        return embeddings;
    }
    
    /**
     * Get model information
     * @return Model information string
     * @throws TrustFormeRSException if model is closed
     */
    public String getModelInfo() throws TrustFormeRSException {
        checkNotClosed();
        return getModelInfoNative(nativeHandle);
    }
    
    /**
     * Get model configuration
     * @return Model configuration as Map
     * @throws TrustFormeRSException if model is closed
     */
    public Map<String, Object> getModelConfig() throws TrustFormeRSException {
        checkNotClosed();
        return getModelConfigNative(nativeHandle);
    }
    
    /**
     * Save the model to a local path
     * @param savePath Path to save the model
     * @return true if save was successful
     * @throws TrustFormeRSException if model is closed or save fails
     */
    public boolean saveModel(String savePath) throws TrustFormeRSException {
        checkNotClosed();
        return saveModelNative(nativeHandle, savePath);
    }
    
    /**
     * Check if the model is closed
     * @return true if the model has been closed
     */
    public boolean isClosed() {
        return closed;
    }
    
    /**
     * Close the model and free native resources
     */
    @Override
    public void close() {
        if (!closed && nativeHandle != 0) {
            unloadModelNative(nativeHandle);
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
     * Check that the model is not closed and throw exception if it is
     * @throws TrustFormeRSException if model is closed
     */
    private void checkNotClosed() throws TrustFormeRSException {
        if (closed) {
            throw new TrustFormeRSException("Model has been closed");
        }
    }
}