package com.trustformers;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Main interface for TrustformeRS library.
 * Provides high-performance transformer models and natural language processing capabilities.
 */
public class TrustformeRS implements Closeable {
    
    private static final Logger logger = LoggerFactory.getLogger(TrustformeRS.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    
    private final AtomicBoolean initialized = new AtomicBoolean(false);
    private final AtomicBoolean closed = new AtomicBoolean(false);
    
    /**
     * Log levels supported by TrustformeRS.
     */
    public enum LogLevel {
        OFF(0),
        ERROR(1),
        WARN(2),
        INFO(3),
        DEBUG(4),
        TRACE(5);
        
        private final int level;
        
        LogLevel(int level) {
            this.level = level;
        }
        
        public int getLevel() {
            return level;
        }
    }
    
    /**
     * Memory usage statistics.
     */
    public static class MemoryUsage {
        public final long totalMemoryBytes;
        public final long peakMemoryBytes;
        public final long allocatedModels;
        public final long allocatedTokenizers;
        public final long allocatedPipelines;
        public final long allocatedTensors;
        
        public MemoryUsage(long totalMemoryBytes, long peakMemoryBytes, 
                          long allocatedModels, long allocatedTokenizers,
                          long allocatedPipelines, long allocatedTensors) {
            this.totalMemoryBytes = totalMemoryBytes;
            this.peakMemoryBytes = peakMemoryBytes;
            this.allocatedModels = allocatedModels;
            this.allocatedTokenizers = allocatedTokenizers;
            this.allocatedPipelines = allocatedPipelines;
            this.allocatedTensors = allocatedTensors;
        }
        
        @Override
        public String toString() {
            return String.format("MemoryUsage{total=%d, peak=%d, models=%d, tokenizers=%d, pipelines=%d, tensors=%d}",
                totalMemoryBytes, peakMemoryBytes, allocatedModels, allocatedTokenizers, allocatedPipelines, allocatedTensors);
        }
    }
    
    /**
     * Build information.
     */
    public static class BuildInfo {
        public final String version;
        public final String features;
        public final String buildDate;
        public final String target;
        
        public BuildInfo(String version, String features, String buildDate, String target) {
            this.version = version;
            this.features = features;
            this.buildDate = buildDate;
            this.target = target;
        }
        
        @Override
        public String toString() {
            return String.format("BuildInfo{version='%s', features='%s', buildDate='%s', target='%s'}",
                version, features, buildDate, target);
        }
    }
    
    /**
     * Optimization configuration.
     */
    public static class OptimizationConfig {
        public boolean enableTracking = true;
        public boolean enableCaching = true;
        public int cacheSizeMB = 256;
        public int numThreads = 0; // Auto-detect
        public boolean enableSIMD = true;
        public boolean optimizeBatchSize = true;
        public int memoryOptimizationLevel = 2;
        
        public OptimizationConfig() {}
        
        public OptimizationConfig(boolean enableTracking, boolean enableCaching, int cacheSizeMB,
                                int numThreads, boolean enableSIMD, boolean optimizeBatchSize,
                                int memoryOptimizationLevel) {
            this.enableTracking = enableTracking;
            this.enableCaching = enableCaching;
            this.cacheSizeMB = cacheSizeMB;
            this.numThreads = numThreads;
            this.enableSIMD = enableSIMD;
            this.optimizeBatchSize = optimizeBatchSize;
            this.memoryOptimizationLevel = memoryOptimizationLevel;
        }
        
        /**
         * Create a default optimization configuration.
         */
        public static OptimizationConfig defaultConfig() {
            return new OptimizationConfig();
        }
    }
    
    // Load the native library
    static {
        try {
            NativeLibraryLoader.loadLibrary();
        } catch (Exception e) {
            logger.error("Failed to load TrustformeRS native library", e);
            throw new RuntimeException("Failed to load TrustformeRS native library", e);
        }
    }
    
    /**
     * Create and initialize a new TrustformeRS instance.
     * 
     * @throws TrustformersException if initialization fails
     */
    public TrustformeRS() throws TrustformersException {
        init();
    }
    
    /**
     * Initialize the TrustformeRS library.
     * 
     * @throws TrustformersException if initialization fails
     */
    public void init() throws TrustformersException {
        if (initialized.get()) {
            return;
        }
        
        checkError(nativeInit());
        initialized.set(true);
        
        // Add shutdown hook to cleanup resources
        Runtime.getRuntime().addShutdownHook(new Thread(this::cleanup));
        
        logger.info("TrustformeRS initialized successfully");
    }
    
    /**
     * Cleanup and release all resources.
     */
    public void cleanup() {
        if (closed.get() || !initialized.get()) {
            return;
        }
        
        try {
            nativeCleanup();
            closed.set(true);
            logger.info("TrustformeRS cleanup completed");
        } catch (Exception e) {
            logger.warn("Error during TrustformeRS cleanup", e);
        }
    }
    
    /**
     * Close the TrustformeRS instance and release resources.
     */
    @Override
    public void close() {
        cleanup();
    }
    
    /**
     * Get the library version.
     * 
     * @return version string
     */
    public String getVersion() {
        checkInitialized();
        return nativeGetVersion();
    }
    
    /**
     * Get build information.
     * 
     * @return build information
     * @throws TrustformersException if the operation fails
     */
    public BuildInfo getBuildInfo() throws TrustformersException {
        checkInitialized();
        String[] info = new String[4];
        checkError(nativeGetBuildInfo(info));
        return new BuildInfo(info[0], info[1], info[2], info[3]);
    }
    
    /**
     * Check if a feature is available.
     * 
     * @param feature feature name to check
     * @return true if the feature is available
     */
    public boolean hasFeature(String feature) {
        checkInitialized();
        return nativeHasFeature(feature);
    }
    
    /**
     * Set the logging level.
     * 
     * @param level logging level
     * @throws TrustformersException if the operation fails
     */
    public void setLogLevel(LogLevel level) throws TrustformersException {
        checkInitialized();
        checkError(nativeSetLogLevel(level.getLevel()));
    }
    
    /**
     * Get current memory usage statistics.
     * 
     * @return memory usage information
     * @throws TrustformersException if the operation fails
     */
    public MemoryUsage getMemoryUsage() throws TrustformersException {
        checkInitialized();
        long[] usage = new long[6];
        checkError(nativeGetMemoryUsage(usage));
        return new MemoryUsage(usage[0], usage[1], usage[2], usage[3], usage[4], usage[5]);
    }
    
    /**
     * Get advanced memory usage statistics with detailed information.
     * 
     * @return advanced memory usage as JSON
     * @throws TrustformersException if the operation fails
     */
    public JsonNode getAdvancedMemoryUsage() throws TrustformersException {
        checkInitialized();
        String jsonStr = nativeGetAdvancedMemoryUsage();
        try {
            return objectMapper.readTree(jsonStr);
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse advanced memory usage JSON", e);
        }
    }
    
    /**
     * Force memory cleanup.
     * 
     * @throws TrustformersException if the operation fails
     */
    public void memoryCleanup() throws TrustformersException {
        checkInitialized();
        checkError(nativeMemoryCleanup());
    }
    
    /**
     * Set memory limits and warning thresholds.
     * 
     * @param maxMemoryMB maximum memory in MB
     * @param warningThresholdMB warning threshold in MB
     * @throws TrustformersException if the operation fails
     */
    public void setMemoryLimits(long maxMemoryMB, long warningThresholdMB) throws TrustformersException {
        checkInitialized();
        checkError(nativeSetMemoryLimits(maxMemoryMB, warningThresholdMB));
    }
    
    /**
     * Check for memory leaks and get a detailed report.
     * 
     * @return leak report as JSON
     * @throws TrustformersException if the operation fails
     */
    public JsonNode checkMemoryLeaks() throws TrustformersException {
        checkInitialized();
        String jsonStr = nativeCheckMemoryLeaks();
        if (jsonStr == null || jsonStr.trim().isEmpty()) {
            return objectMapper.createObjectNode();
        }
        try {
            return objectMapper.readTree(jsonStr);
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse memory leak report JSON", e);
        }
    }
    
    /**
     * Apply performance optimizations.
     * 
     * @param config optimization configuration
     * @throws TrustformersException if the operation fails
     */
    public void applyOptimizations(OptimizationConfig config) throws TrustformersException {
        checkInitialized();
        checkError(nativeApplyOptimizations(
            config.enableTracking, config.enableCaching, config.cacheSizeMB,
            config.numThreads, config.enableSIMD, config.optimizeBatchSize,
            config.memoryOptimizationLevel));
    }
    
    /**
     * Start a performance profiling session.
     * 
     * @throws TrustformersException if the operation fails
     */
    public void startProfiling() throws TrustformersException {
        checkInitialized();
        checkError(nativeStartProfiling());
    }
    
    /**
     * Stop profiling and get a performance report.
     * 
     * @return profiling report as JSON
     * @throws TrustformersException if the operation fails
     */
    public JsonNode stopProfiling() throws TrustformersException {
        checkInitialized();
        String jsonStr = nativeStopProfiling();
        if (jsonStr == null || jsonStr.trim().isEmpty()) {
            return objectMapper.createObjectNode();
        }
        try {
            return objectMapper.readTree(jsonStr);
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse profiling report JSON", e);
        }
    }
    
    /**
     * Load a model from Hugging Face Hub.
     * 
     * @param modelName name of the model to load
     * @return loaded model instance
     * @throws TrustformersException if loading fails
     */
    public Model loadModelFromHub(String modelName) throws TrustformersException {
        checkInitialized();
        return Model.loadFromHub(this, modelName);
    }
    
    /**
     * Load a model from a local path.
     * 
     * @param modelPath path to the model files
     * @return loaded model instance
     * @throws TrustformersException if loading fails
     */
    public Model loadModelFromPath(String modelPath) throws TrustformersException {
        checkInitialized();
        return Model.loadFromPath(this, modelPath);
    }
    
    /**
     * Load a tokenizer from Hugging Face Hub.
     * 
     * @param modelName name of the tokenizer to load
     * @return loaded tokenizer instance
     * @throws TrustformersException if loading fails
     */
    public Tokenizer loadTokenizerFromHub(String modelName) throws TrustformersException {
        checkInitialized();
        return Tokenizer.loadFromHub(this, modelName);
    }
    
    /**
     * Load a tokenizer from a local path.
     * 
     * @param tokenizerPath path to the tokenizer files
     * @return loaded tokenizer instance
     * @throws TrustformersException if loading fails
     */
    public Tokenizer loadTokenizerFromPath(String tokenizerPath) throws TrustformersException {
        checkInitialized();
        return Tokenizer.loadFromPath(this, tokenizerPath);
    }
    
    /**
     * Check if the library is initialized.
     * 
     * @return true if initialized
     */
    public boolean isInitialized() {
        return initialized.get() && !closed.get();
    }
    
    // Private helper methods
    
    private void checkInitialized() {
        if (!initialized.get() || closed.get()) {
            throw new IllegalStateException("TrustformeRS is not initialized or has been closed");
        }
    }
    
    private void checkError(int errorCode) throws TrustformersException {
        if (errorCode != 0) {
            TrustformersException.ErrorCode code = TrustformersException.ErrorCode.fromCode(errorCode);
            throw new TrustformersException(code, "Operation failed with error code: " + errorCode);
        }
    }
    
    // Native method declarations
    
    private static native int nativeInit();
    private static native int nativeCleanup();
    private static native String nativeGetVersion();
    private static native int nativeGetBuildInfo(String[] buildInfo);
    private static native boolean nativeHasFeature(String feature);
    private static native int nativeSetLogLevel(int level);
    private static native int nativeGetMemoryUsage(long[] usage);
    private static native String nativeGetAdvancedMemoryUsage();
    private static native int nativeMemoryCleanup();
    private static native int nativeSetMemoryLimits(long maxMemoryMB, long warningThresholdMB);
    private static native String nativeCheckMemoryLeaks();
    private static native int nativeApplyOptimizations(boolean enableTracking, boolean enableCaching,
                                                       int cacheSizeMB, int numThreads, boolean enableSIMD,
                                                       boolean optimizeBatchSize, int memoryOptimizationLevel);
    private static native int nativeStartProfiling();
    private static native String nativeStopProfiling();
}