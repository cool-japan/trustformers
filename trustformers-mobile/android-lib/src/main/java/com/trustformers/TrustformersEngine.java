package com.trustformers;

import android.content.Context;
import android.os.Build;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;

import org.json.JSONObject;
import org.json.JSONException;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * TrustformersEngine - Main inference engine for TrustformeRS on Android
 * 
 * This class provides the primary interface for loading and running
 * TrustformeRS models on Android devices, with support for NNAPI,
 * CPU, and GPU backends.
 */
public class TrustformersEngine implements Closeable {
    private static final String TAG = "TrustformersEngine";
    
    static {
        try {
            System.loadLibrary("trustformers_mobile");
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException("Failed to load trustformers_mobile library", e);
        }
    }
    
    private long nativeEnginePtr = 0;
    private final Context context;
    private final EngineConfig config;
    private final Executor executor;
    private boolean isInitialized = false;
    private PerformanceMonitor performanceMonitor;
    
    /**
     * Engine configuration
     */
    public static class EngineConfig {
        public enum Backend {
            CPU("cpu"),
            GPU("gpu"), 
            NNAPI("nnapi"),
            AUTO("auto");
            
            private final String value;
            
            Backend(String value) {
                this.value = value;
            }
            
            public String getValue() {
                return value;
            }
        }
        
        public enum MemoryOptimization {
            MINIMAL("minimal"),
            BALANCED("balanced"),
            MAXIMUM("maximum");
            
            private final String value;
            
            MemoryOptimization(String value) {
                this.value = value;
            }
            
            public String getValue() {
                return value;
            }
        }
        
        public enum QuantizationScheme {
            NONE("none"),
            INT8("int8"),
            INT4("int4"), 
            FP16("fp16"),
            DYNAMIC("dynamic");
            
            private final String value;
            
            QuantizationScheme(String value) {
                this.value = value;
            }
            
            public String getValue() {
                return value;
            }
        }
        
        private Backend backend = Backend.AUTO;
        private MemoryOptimization memoryOptimization = MemoryOptimization.BALANCED;
        private QuantizationScheme quantizationScheme = QuantizationScheme.NONE;
        private int maxMemoryMB = 512;
        private int numThreads = 0; // 0 = auto-detect
        private boolean useFP16 = false;
        private boolean enableBatching = false;
        private int maxBatchSize = 1;
        private boolean enableProfiling = false;
        
        public EngineConfig() {}
        
        public EngineConfig setBackend(Backend backend) {
            this.backend = backend;
            return this;
        }
        
        public EngineConfig setMemoryOptimization(MemoryOptimization optimization) {
            this.memoryOptimization = optimization;
            return this;
        }
        
        public EngineConfig setQuantizationScheme(QuantizationScheme scheme) {
            this.quantizationScheme = scheme;
            return this;
        }
        
        public EngineConfig setMaxMemoryMB(int maxMemoryMB) {
            this.maxMemoryMB = maxMemoryMB;
            return this;
        }
        
        public EngineConfig setNumThreads(int numThreads) {
            this.numThreads = numThreads;
            return this;
        }
        
        public EngineConfig setUseFP16(boolean useFP16) {
            this.useFP16 = useFP16;
            return this;
        }
        
        public EngineConfig setEnableBatching(boolean enableBatching) {
            this.enableBatching = enableBatching;
            return this;
        }
        
        public EngineConfig setMaxBatchSize(int maxBatchSize) {
            this.maxBatchSize = maxBatchSize;
            return this;
        }
        
        public EngineConfig setEnableProfiling(boolean enableProfiling) {
            this.enableProfiling = enableProfiling;
            return this;
        }
        
        public String toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("platform", "android");
            json.put("backend", backend.getValue());
            json.put("memory_optimization", memoryOptimization.getValue());
            json.put("max_memory_mb", maxMemoryMB);
            json.put("num_threads", numThreads);
            json.put("use_fp16", useFP16);
            json.put("enable_batching", enableBatching);
            json.put("max_batch_size", maxBatchSize);
            
            if (quantizationScheme != QuantizationScheme.NONE) {
                JSONObject quantization = new JSONObject();
                quantization.put("scheme", quantizationScheme.getValue());
                quantization.put("dynamic", quantizationScheme == QuantizationScheme.DYNAMIC);
                json.put("quantization", quantization);
            }
            
            return json.toString();
        }
        
        /**
         * Create optimized config based on device capabilities
         */
        public static EngineConfig createOptimized(Context context) {
            EngineConfig config = new EngineConfig();
            DeviceInfo deviceInfo = DeviceInfo.getInstance(context);
            
            // Choose backend based on device capabilities
            if (deviceInfo.hasNNAPI() && Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                config.setBackend(Backend.NNAPI);
            } else if (deviceInfo.hasGPU()) {
                config.setBackend(Backend.GPU);
            } else {
                config.setBackend(Backend.CPU);
            }
            
            // Adjust memory based on device
            long totalMemory = deviceInfo.getTotalMemoryMB();
            if (totalMemory >= 8192) {
                config.setMaxMemoryMB(1024);
                config.setMemoryOptimization(MemoryOptimization.MINIMAL);
            } else if (totalMemory >= 4096) {
                config.setMaxMemoryMB(512);
                config.setMemoryOptimization(MemoryOptimization.BALANCED);
            } else {
                config.setMaxMemoryMB(256);
                config.setMemoryOptimization(MemoryOptimization.MAXIMUM);
            }
            
            // Enable FP16 on supported devices
            if (deviceInfo.supportsFP16()) {
                config.setUseFP16(true);
            }
            
            // Set thread count based on CPU cores
            int cpuCores = deviceInfo.getCpuCores();
            config.setNumThreads(Math.max(1, cpuCores / 2));
            
            return config;
        }
    }
    
    /**
     * Constructor
     * @param context Android application context
     * @param config Engine configuration
     */
    public TrustformersEngine(@NonNull Context context, @NonNull EngineConfig config) {
        this.context = context.getApplicationContext();
        this.config = config;
        this.executor = Executors.newSingleThreadExecutor();
        
        if (config.enableProfiling) {
            this.performanceMonitor = new PerformanceMonitor();
        }
        
        initialize();
    }
    
    /**
     * Constructor with default optimized configuration
     * @param context Android application context
     */
    public TrustformersEngine(@NonNull Context context) {
        this(context, EngineConfig.createOptimized(context));
    }
    
    private synchronized void initialize() {
        if (isInitialized) {
            return;
        }
        
        try {
            String configJson = config.toJson();
            nativeEnginePtr = createEngine(configJson);
            if (nativeEnginePtr == 0) {
                throw new RuntimeException("Failed to create native engine");
            }
            isInitialized = true;
        } catch (JSONException e) {
            throw new RuntimeException("Failed to serialize configuration", e);
        }
    }
    
    /**
     * Load a model from file path
     * @param modelPath Path to the model file
     * @return Model instance
     * @throws IOException if model loading fails
     */
    public Model loadModel(@NonNull String modelPath) throws IOException {
        if (!isInitialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        
        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            throw new IOException("Model file not found: " + modelPath);
        }
        
        boolean success = loadModel(nativeEnginePtr, modelPath);
        if (!success) {
            throw new IOException("Failed to load model: " + modelPath);
        }
        
        return new Model(modelPath, this);
    }
    
    /**
     * Load a model from assets
     * @param assetPath Path to model in assets
     * @return Model instance
     * @throws IOException if model loading fails
     */
    public Model loadModelFromAssets(@NonNull String assetPath) throws IOException {
        // Extract model from assets to cache directory
        File cacheFile = new File(context.getCacheDir(), "model_" + assetPath.hashCode() + ".tfm");
        AssetUtils.copyAssetToFile(context, assetPath, cacheFile);
        
        return loadModel(cacheFile.getAbsolutePath());
    }
    
    /**
     * Perform inference with input tensor
     * @param model Model to use for inference
     * @param input Input tensor
     * @return Output tensor
     */
    public Tensor inference(@NonNull Model model, @NonNull Tensor input) {
        if (!isInitialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        
        long startTime = System.currentTimeMillis();
        
        // Convert tensor to byte array for JNI
        byte[] inputData = tensorToBytes(input);
        
        // Perform inference through JNI
        byte[] outputData = inference(nativeEnginePtr, inputData);
        if (outputData == null) {
            throw new RuntimeException("Inference failed");
        }
        
        // Convert output bytes back to tensor
        Tensor output = bytesToTensor(outputData, input.getShape());
        
        if (performanceMonitor != null) {
            long inferenceTime = System.currentTimeMillis() - startTime;
            performanceMonitor.recordInference(inferenceTime, inputData.length);
        }
        
        return output;
    }
    
    /**
     * Perform batch inference
     * @param model Model to use for inference
     * @param inputs List of input tensors
     * @return List of output tensors
     */
    public List<Tensor> batchInference(@NonNull Model model, @NonNull List<Tensor> inputs) {
        if (!config.enableBatching) {
            throw new IllegalStateException("Batching not enabled in configuration");
        }
        
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("Input list cannot be empty");
        }
        
        if (inputs.size() > config.maxBatchSize) {
            throw new IllegalArgumentException("Batch size exceeds maximum: " + 
                inputs.size() + " > " + config.maxBatchSize);
        }
        
        if (!isInitialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        
        long startTime = System.currentTimeMillis();
        
        // Validate all input tensors have the same shape (except batch dimension)
        int[] referenceShape = inputs.get(0).getShape();
        for (int i = 1; i < inputs.size(); i++) {
            int[] currentShape = inputs.get(i).getShape();
            if (!Arrays.equals(Arrays.copyOfRange(referenceShape, 1, referenceShape.length),
                              Arrays.copyOfRange(currentShape, 1, currentShape.length))) {
                throw new IllegalArgumentException("All input tensors must have the same shape except batch dimension");
            }
        }
        
        // Serialize batch data for JNI
        byte[] batchInputData = serializeBatchInputs(inputs);
        
        // Perform batch inference through JNI
        byte[] batchOutputData = batchInference(nativeEnginePtr, batchInputData, inputs.size());
        if (batchOutputData == null) {
            throw new RuntimeException("Batch inference failed");
        }
        
        // Deserialize batch outputs
        List<Tensor> outputs = deserializeBatchOutputs(batchOutputData, inputs.size(), referenceShape);
        
        if (performanceMonitor != null) {
            long inferenceTime = System.currentTimeMillis() - startTime;
            performanceMonitor.recordBatchInference(inferenceTime, inputs.size(), batchInputData.length);
        }
        
        return outputs;
    }
    
    /**
     * Get device information
     * @return Device information
     */
    public DeviceInfo getDeviceInfo() {
        return DeviceInfo.getInstance(context);
    }
    
    /**
     * Get performance statistics
     * @return Performance statistics or null if profiling not enabled
     */
    @Nullable
    public PerformanceStats getPerformanceStats() {
        return performanceMonitor != null ? performanceMonitor.getStats() : null;
    }
    
    /**
     * Get static device information as JSON
     * @return JSON string with device information
     */
    public static String getDeviceInfoJson() {
        return getDeviceInfo();
    }
    
    /**
     * Close the engine and release resources
     */
    @Override
    public synchronized void close() {
        if (nativeEnginePtr != 0) {
            releaseEngine(nativeEnginePtr);
            nativeEnginePtr = 0;
        }
        isInitialized = false;
        executor.shutdown();
    }
    
    @Override
    protected void finalize() throws Throwable {
        close();
        super.finalize();
    }
    
    // Helper methods
    
    private byte[] tensorToBytes(Tensor tensor) {
        float[] data = tensor.getData();
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        for (float value : data) {
            buffer.putFloat(value);
        }
        
        return buffer.array();
    }
    
    private Tensor bytesToTensor(byte[] data, int[] shape) {
        ByteBuffer buffer = ByteBuffer.wrap(data);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        float[] floatData = new float[data.length / 4];
        for (int i = 0; i < floatData.length; i++) {
            floatData[i] = buffer.getFloat();
        }
        
        return new Tensor(floatData, shape);
    }
    
    private byte[] serializeBatchInputs(List<Tensor> inputs) {
        // Calculate total size needed
        int totalSize = 4; // 4 bytes for batch size
        for (Tensor tensor : inputs) {
            totalSize += 4; // 4 bytes for tensor size
            totalSize += tensor.getData().length * 4; // 4 bytes per float
        }
        
        ByteBuffer buffer = ByteBuffer.allocate(totalSize);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        // Write batch size
        buffer.putInt(inputs.size());
        
        // Write each tensor
        for (Tensor tensor : inputs) {
            float[] data = tensor.getData();
            buffer.putInt(data.length);
            for (float value : data) {
                buffer.putFloat(value);
            }
        }
        
        return buffer.array();
    }
    
    private List<Tensor> deserializeBatchOutputs(byte[] data, int batchSize, int[] referenceShape) {
        ByteBuffer buffer = ByteBuffer.wrap(data);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        List<Tensor> outputs = new java.util.ArrayList<>(batchSize);
        
        for (int i = 0; i < batchSize; i++) {
            int tensorSize = buffer.getInt();
            float[] floatData = new float[tensorSize];
            
            for (int j = 0; j < tensorSize; j++) {
                floatData[j] = buffer.getFloat();
            }
            
            outputs.add(new Tensor(floatData, referenceShape));
        }
        
        return outputs;
    }
    
    // Native methods
    
    private static native long createEngine(String configJson);
    private static native boolean loadModel(long enginePtr, String modelPath);
    private static native byte[] inference(long enginePtr, byte[] inputData);
    private static native byte[] batchInference(long enginePtr, byte[] batchInputData, int batchSize);
    private static native String getDeviceInfo();
    private static native void releaseEngine(long enginePtr);
}