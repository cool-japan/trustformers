package com.trustformers;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.Map;
import java.util.HashMap;

/**
 * Model representation for TrustformeRS Android
 * 
 * This class represents a loaded model and provides metadata
 * and configuration information.
 */
public class Model {
    private final String modelPath;
    private final TrustformersEngine engine;
    private final Map<String, Object> metadata;
    private ModelInfo modelInfo;
    
    /**
     * Model information
     */
    public static class ModelInfo {
        private final String name;
        private final String version;
        private final String[] inputNames;
        private final int[][] inputShapes;
        private final String[] outputNames;
        private final int[][] outputShapes;
        private final long modelSizeBytes;
        private final Map<String, String> properties;
        
        ModelInfo(String name, String version, 
                  String[] inputNames, int[][] inputShapes,
                  String[] outputNames, int[][] outputShapes,
                  long modelSizeBytes, Map<String, String> properties) {
            this.name = name;
            this.version = version;
            this.inputNames = inputNames;
            this.inputShapes = inputShapes;
            this.outputNames = outputNames;
            this.outputShapes = outputShapes;
            this.modelSizeBytes = modelSizeBytes;
            this.properties = properties;
        }
        
        public String getName() {
            return name;
        }
        
        public String getVersion() {
            return version;
        }
        
        public String[] getInputNames() {
            return inputNames.clone();
        }
        
        public int[][] getInputShapes() {
            return inputShapes.clone();
        }
        
        public String[] getOutputNames() {
            return outputNames.clone();
        }
        
        public int[][] getOutputShapes() {
            return outputShapes.clone();
        }
        
        public long getModelSizeBytes() {
            return modelSizeBytes;
        }
        
        public Map<String, String> getProperties() {
            return new HashMap<>(properties);
        }
        
        /**
         * Get estimated memory usage in MB
         * @return Estimated memory usage
         */
        public int getEstimatedMemoryMB() {
            // Rough estimate: model size + working memory
            return (int) ((modelSizeBytes / (1024 * 1024)) * 2);
        }
        
        /**
         * Check if model supports batch inference
         * @return True if batch inference is supported
         */
        public boolean supportsBatchInference() {
            // Check if first dimension of inputs is batch dimension
            for (int[] shape : inputShapes) {
                if (shape.length > 0 && shape[0] == -1) {
                    return true;
                }
            }
            return false;
        }
    }
    
    Model(@NonNull String modelPath, @NonNull TrustformersEngine engine) {
        this.modelPath = modelPath;
        this.engine = engine;
        this.metadata = new HashMap<>();
        loadModelInfo();
    }
    
    private void loadModelInfo() {
        // In a real implementation, this would query the native model
        // For now, create placeholder info
        String[] inputNames = {"input"};
        int[][] inputShapes = {{1, 768}};
        String[] outputNames = {"output"};
        int[][] outputShapes = {{1, 768}};
        
        Map<String, String> properties = new HashMap<>();
        properties.put("framework", "trustformers");
        properties.put("type", "transformer");
        
        this.modelInfo = new ModelInfo(
            "model",
            "1.0",
            inputNames,
            inputShapes,
            outputNames,
            outputShapes,
            10 * 1024 * 1024, // 10 MB
            properties
        );
    }
    
    /**
     * Get model path
     * @return Path to the model file
     */
    @NonNull
    public String getModelPath() {
        return modelPath;
    }
    
    /**
     * Get model information
     * @return Model information
     */
    @NonNull
    public ModelInfo getModelInfo() {
        return modelInfo;
    }
    
    /**
     * Get model metadata
     * @return Metadata map
     */
    @NonNull
    public Map<String, Object> getMetadata() {
        return new HashMap<>(metadata);
    }
    
    /**
     * Set model metadata
     * @param key Metadata key
     * @param value Metadata value
     */
    public void setMetadata(@NonNull String key, @Nullable Object value) {
        if (value == null) {
            metadata.remove(key);
        } else {
            metadata.put(key, value);
        }
    }
    
    /**
     * Perform inference with this model
     * @param input Input tensor
     * @return Output tensor
     */
    @NonNull
    public Tensor inference(@NonNull Tensor input) {
        return engine.inference(this, input);
    }
    
    /**
     * Validate input tensor shape
     * @param input Input tensor to validate
     * @param inputIndex Index of input (for multi-input models)
     * @return True if shape is valid
     */
    public boolean validateInputShape(@NonNull Tensor input, int inputIndex) {
        if (inputIndex < 0 || inputIndex >= modelInfo.inputShapes.length) {
            return false;
        }
        
        int[] expectedShape = modelInfo.inputShapes[inputIndex];
        int[] actualShape = input.getShape();
        
        if (expectedShape.length != actualShape.length) {
            return false;
        }
        
        for (int i = 0; i < expectedShape.length; i++) {
            // -1 indicates dynamic dimension
            if (expectedShape[i] != -1 && expectedShape[i] != actualShape[i]) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Validate input tensor shape (for single input models)
     * @param input Input tensor to validate
     * @return True if shape is valid
     */
    public boolean validateInputShape(@NonNull Tensor input) {
        return validateInputShape(input, 0);
    }
    
    /**
     * Get expected input shape
     * @param inputIndex Index of input
     * @return Expected shape array
     */
    @NonNull
    public int[] getExpectedInputShape(int inputIndex) {
        if (inputIndex < 0 || inputIndex >= modelInfo.inputShapes.length) {
            throw new IndexOutOfBoundsException("Invalid input index: " + inputIndex);
        }
        return modelInfo.inputShapes[inputIndex].clone();
    }
    
    /**
     * Get expected input shape (for single input models)
     * @return Expected shape array
     */
    @NonNull
    public int[] getExpectedInputShape() {
        return getExpectedInputShape(0);
    }
    
    /**
     * Get expected output shape
     * @param outputIndex Index of output
     * @return Expected shape array
     */
    @NonNull
    public int[] getExpectedOutputShape(int outputIndex) {
        if (outputIndex < 0 || outputIndex >= modelInfo.outputShapes.length) {
            throw new IndexOutOfBoundsException("Invalid output index: " + outputIndex);
        }
        return modelInfo.outputShapes[outputIndex].clone();
    }
    
    /**
     * Get expected output shape (for single output models)
     * @return Expected shape array
     */
    @NonNull
    public int[] getExpectedOutputShape() {
        return getExpectedOutputShape(0);
    }
    
    /**
     * Check if model is loaded and ready for inference
     * @return True if model is ready
     */
    public boolean isReady() {
        return modelInfo != null;
    }
    
    @Override
    public String toString() {
        return String.format("Model(name=%s, version=%s, path=%s)",
            modelInfo.getName(), modelInfo.getVersion(), modelPath);
    }
}