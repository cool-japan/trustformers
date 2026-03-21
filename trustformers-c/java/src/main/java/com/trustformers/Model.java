package com.trustformers;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Represents a loaded transformer model.
 * Provides methods for model management, validation, and configuration.
 */
public class Model implements Closeable {
    
    private static final Logger logger = LoggerFactory.getLogger(Model.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    
    private final TrustformeRS trustformers;
    private long nativeHandle;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    
    /**
     * Model information structure.
     */
    public static class ModelInfo {
        public final String name;
        public final String type;
        public final String architecture;
        public final long parameters;
        public final String precision;
        public final JsonNode metadata;
        
        public ModelInfo(String name, String type, String architecture, 
                        long parameters, String precision, JsonNode metadata) {
            this.name = name;
            this.type = type;
            this.architecture = architecture;
            this.parameters = parameters;
            this.precision = precision;
            this.metadata = metadata;
        }
        
        @Override
        public String toString() {
            return String.format("ModelInfo{name='%s', type='%s', architecture='%s', parameters=%d, precision='%s'}",
                name, type, architecture, parameters, precision);
        }
    }
    
    // Package-private constructor
    Model(TrustformeRS trustformers, long nativeHandle) {
        this.trustformers = trustformers;
        this.nativeHandle = nativeHandle;
    }
    
    /**
     * Load a model from Hugging Face Hub.
     * 
     * @param trustformers TrustformeRS instance
     * @param modelName name of the model to load
     * @return loaded model instance
     * @throws TrustformersException if loading fails
     */
    public static Model loadFromHub(TrustformeRS trustformers, String modelName) throws TrustformersException {
        if (modelName == null || modelName.trim().isEmpty()) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER, 
                "Model name cannot be null or empty");
        }
        
        long handle = nativeLoadFromHub(modelName);
        if (handle == 0) {
            throw new TrustformersException(TrustformersException.ErrorCode.RUNTIME_ERROR,
                "Failed to load model from Hub: " + modelName);
        }
        
        Model model = new Model(trustformers, handle);
        logger.info("Successfully loaded model from Hub: {}", modelName);
        return model;
    }
    
    /**
     * Load a model from a local path.
     * 
     * @param trustformers TrustformeRS instance
     * @param modelPath path to the model files
     * @return loaded model instance
     * @throws TrustformersException if loading fails
     */
    public static Model loadFromPath(TrustformeRS trustformers, String modelPath) throws TrustformersException {
        if (modelPath == null || modelPath.trim().isEmpty()) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER, 
                "Model path cannot be null or empty");
        }
        
        long handle = nativeLoadFromPath(modelPath);
        if (handle == 0) {
            throw new TrustformersException(TrustformersException.ErrorCode.RUNTIME_ERROR,
                "Failed to load model from path: " + modelPath);
        }
        
        Model model = new Model(trustformers, handle);
        logger.info("Successfully loaded model from path: {}", modelPath);
        return model;
    }
    
    /**
     * Free the model and release its resources.
     * This method is called automatically when the model is garbage collected,
     * but it's recommended to call it explicitly for immediate resource cleanup.
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
        
        logger.debug("Model resources freed");
    }
    
    /**
     * Close the model and release resources.
     */
    @Override
    public void close() {
        try {
            free();
        } catch (TrustformersException e) {
            logger.warn("Error closing model", e);
        }
    }
    
    /**
     * Get detailed model information.
     * 
     * @return model information
     * @throws TrustformersException if the operation fails
     */
    public ModelInfo getInfo() throws TrustformersException {
        checkNotClosed();
        
        String jsonStr = nativeGetInfo(nativeHandle);
        if (jsonStr == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.RUNTIME_ERROR,
                "Failed to get model info");
        }
        
        try {
            JsonNode json = objectMapper.readTree(jsonStr);
            return new ModelInfo(
                json.path("name").asText(),
                json.path("type").asText(),
                json.path("architecture").asText(),
                json.path("parameters").asLong(),
                json.path("precision").asText(),
                json.path("metadata")
            );
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse model info JSON", e);
        }
    }
    
    /**
     * Set the quantization level for the model.
     * 
     * @param bits number of bits for quantization (e.g., 8, 16)
     * @throws TrustformersException if the operation fails
     */
    public void setQuantization(int bits) throws TrustformersException {
        checkNotClosed();
        
        if (bits <= 0 || bits > 32) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Quantization bits must be between 1 and 32");
        }
        
        checkError(nativeSetQuantization(nativeHandle, bits));
        logger.debug("Model quantization set to {} bits", bits);
    }
    
    /**
     * Validate the model integrity.
     * 
     * @return true if the model is valid
     * @throws TrustformersException if the validation fails
     */
    public boolean validate() throws TrustformersException {
        checkNotClosed();
        
        boolean[] result = new boolean[1];
        checkError(nativeValidate(nativeHandle, result));
        return result[0];
    }
    
    /**
     * Get model metadata as JSON.
     * 
     * @return model metadata
     * @throws TrustformersException if the operation fails
     */
    public JsonNode getMetadata() throws TrustformersException {
        checkNotClosed();
        
        String jsonStr = nativeGetMetadata(nativeHandle);
        if (jsonStr == null || jsonStr.trim().isEmpty()) {
            return objectMapper.createObjectNode();
        }
        
        try {
            return objectMapper.readTree(jsonStr);
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse metadata JSON", e);
        }
    }
    
    /**
     * Check if the model is loaded and not closed.
     * 
     * @return true if the model is loaded
     */
    public boolean isLoaded() {
        return !closed.get() && nativeHandle != 0;
    }
    
    /**
     * Get the native handle for this model.
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
            throw new IllegalStateException("Model has been closed or not properly loaded");
        }
    }
    
    private void checkError(int errorCode) throws TrustformersException {
        if (errorCode != 0) {
            TrustformersException.ErrorCode code = TrustformersException.ErrorCode.fromCode(errorCode);
            throw new TrustformersException(code, "Model operation failed with error code: " + errorCode);
        }
    }
    
    @Override
    protected void finalize() throws Throwable {
        try {
            if (!closed.get() && nativeHandle != 0) {
                logger.warn("Model was not explicitly closed, freeing in finalizer");
                free();
            }
        } catch (Exception e) {
            logger.warn("Error in model finalizer", e);
        } finally {
            super.finalize();
        }
    }
    
    // Native method declarations
    
    private static native long nativeLoadFromHub(String modelName);
    private static native long nativeLoadFromPath(String modelPath);
    private static native int nativeFree(long handle);
    private static native String nativeGetInfo(long handle);
    private static native int nativeSetQuantization(long handle, int bits);
    private static native int nativeValidate(long handle, boolean[] isValid);
    private static native String nativeGetMetadata(long handle);
}