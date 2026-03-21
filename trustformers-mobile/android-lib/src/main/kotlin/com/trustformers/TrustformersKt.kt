package com.trustformers

import android.content.Context
import kotlinx.coroutines.*
import java.io.Closeable
import java.io.File
import kotlin.coroutines.CoroutineContext

/**
 * Kotlin extensions and coroutine support for TrustformeRS Android
 */

/**
 * Coroutine-based TrustformersEngine wrapper
 */
class TrustformersKt(
    context: Context,
    config: TrustformersEngine.EngineConfig = TrustformersEngine.EngineConfig.createOptimized(context)
) : Closeable {
    
    private val engine = TrustformersEngine(context, config)
    private val dispatcher = Dispatchers.Default
    private val scope = CoroutineScope(SupervisorJob() + dispatcher)
    
    /**
     * Load a model asynchronously
     */
    suspend fun loadModel(modelPath: String): Model = withContext(dispatcher) {
        engine.loadModel(modelPath)
    }
    
    /**
     * Load a model from assets asynchronously
     */
    suspend fun loadModelFromAssets(assetPath: String): Model = withContext(dispatcher) {
        engine.loadModelFromAssets(assetPath)
    }
    
    /**
     * Perform inference asynchronously
     */
    suspend fun inference(model: Model, input: Tensor): Tensor = withContext(dispatcher) {
        engine.inference(model, input)
    }
    
    /**
     * Perform batch inference asynchronously
     */
    suspend fun batchInference(model: Model, inputs: List<Tensor>): List<Tensor> = withContext(dispatcher) {
        engine.batchInference(model, inputs)
    }
    
    /**
     * Get device info
     */
    val deviceInfo: DeviceInfo
        get() = engine.deviceInfo
    
    /**
     * Get performance stats
     */
    val performanceStats: PerformanceMonitor.PerformanceStats?
        get() = engine.performanceStats
    
    override fun close() {
        scope.cancel()
        engine.close()
    }
}

/**
 * Extension functions for TrustformersEngine
 */

/**
 * Use TrustformersEngine with automatic resource management
 */
inline fun <T> TrustformersEngine.use(block: (TrustformersEngine) -> T): T {
    return try {
        block(this)
    } finally {
        close()
    }
}

/**
 * Use TrustformersKt with automatic resource management
 */
inline fun <T> TrustformersKt.use(block: (TrustformersKt) -> T): T {
    return try {
        block(this)
    } finally {
        close()
    }
}

/**
 * Create and configure TrustformersEngine with DSL
 */
fun trustformersEngine(
    context: Context,
    configure: TrustformersEngine.EngineConfig.() -> Unit = {}
): TrustformersEngine {
    val config = TrustformersEngine.EngineConfig.createOptimized(context).apply(configure)
    return TrustformersEngine(context, config)
}

/**
 * Create and configure TrustformersKt with DSL
 */
fun trustformersKt(
    context: Context,
    configure: TrustformersEngine.EngineConfig.() -> Unit = {}
): TrustformersKt {
    val config = TrustformersEngine.EngineConfig.createOptimized(context).apply(configure)
    return TrustformersKt(context, config)
}

/**
 * Extension functions for Tensor
 */

/**
 * Create tensor from list
 */
fun tensorOf(vararg values: Float, shape: IntArray): Tensor {
    return Tensor(values, shape)
}

/**
 * Create 2D tensor from nested lists
 */
fun tensor2D(data: List<List<Float>>): Tensor {
    val rows = data.size
    val cols = data.firstOrNull()?.size ?: 0
    val flatData = FloatArray(rows * cols)
    
    data.forEachIndexed { i, row ->
        row.forEachIndexed { j, value ->
            flatData[i * cols + j] = value
        }
    }
    
    return Tensor(flatData, intArrayOf(rows, cols))
}

/**
 * Get tensor element using operator
 */
operator fun Tensor.get(vararg indices: Int): Float {
    return getElement(*indices)
}

/**
 * Set tensor element using operator
 */
operator fun Tensor.set(vararg indices: Int, value: Float) {
    setElement(value, *indices)
}

/**
 * Convert tensor to 2D array (if applicable)
 */
fun Tensor.to2DArray(): Array<FloatArray>? {
    if (shape.size != 2) return null
    
    val rows = shape[0]
    val cols = shape[1]
    val data = data
    
    return Array(rows) { i ->
        FloatArray(cols) { j ->
            data[i * cols + j]
        }
    }
}

/**
 * Extension functions for Model
 */

/**
 * Perform inference with validation
 */
fun Model.safeInference(input: Tensor): Result<Tensor> {
    return if (validateInputShape(input)) {
        Result.success(inference(input))
    } else {
        Result.failure(IllegalArgumentException(
            "Invalid input shape: ${input.shape.contentToString()} " +
            "expected: ${expectedInputShape.contentToString()}"
        ))
    }
}

/**
 * Flow-based inference for streaming
 */
fun Model.inferenceFlow(inputs: Flow<Tensor>): Flow<Tensor> = flow {
    inputs.collect { input ->
        emit(inference(input))
    }
}

/**
 * Utility functions
 */

/**
 * Load model with retry logic
 */
suspend fun TrustformersKt.loadModelWithRetry(
    modelPath: String,
    maxRetries: Int = 3,
    delayMs: Long = 1000
): Model {
    var lastException: Exception? = null
    
    repeat(maxRetries) { attempt ->
        try {
            return loadModel(modelPath)
        } catch (e: Exception) {
            lastException = e
            if (attempt < maxRetries - 1) {
                delay(delayMs * (attempt + 1))
            }
        }
    }
    
    throw lastException ?: Exception("Failed to load model after $maxRetries attempts")
}

/**
 * Performance monitoring DSL
 */
class PerformanceScope(private val monitor: PerformanceMonitor) {
    
    suspend fun <T> measure(block: suspend () -> T): T {
        val startTime = System.currentTimeMillis()
        val result = block()
        val duration = System.currentTimeMillis() - startTime
        
        monitor.recordInference(duration, 0)
        return result
    }
}

/**
 * Device-aware configuration
 */
fun TrustformersEngine.EngineConfig.configureForDevice(deviceInfo: DeviceInfo) {
    when (deviceInfo.performanceClass) {
        DeviceInfo.PerformanceClass.ENTRY_LEVEL -> {
            backend = TrustformersEngine.EngineConfig.Backend.CPU
            memoryOptimization = TrustformersEngine.EngineConfig.MemoryOptimization.MAXIMUM
            maxMemoryMB = 256
            numThreads = 2
        }
        DeviceInfo.PerformanceClass.MID_RANGE -> {
            backend = TrustformersEngine.EngineConfig.Backend.AUTO
            memoryOptimization = TrustformersEngine.EngineConfig.MemoryOptimization.BALANCED
            maxMemoryMB = 512
            numThreads = deviceInfo.recommendedThreadCount
        }
        DeviceInfo.PerformanceClass.HIGH_END,
        DeviceInfo.PerformanceClass.CLASS_30,
        DeviceInfo.PerformanceClass.CLASS_31,
        DeviceInfo.PerformanceClass.CLASS_33 -> {
            backend = if (deviceInfo.hasNNAPI()) {
                TrustformersEngine.EngineConfig.Backend.NNAPI
            } else {
                TrustformersEngine.EngineConfig.Backend.GPU
            }
            memoryOptimization = TrustformersEngine.EngineConfig.MemoryOptimization.MINIMAL
            maxMemoryMB = 1024
            enableBatching = true
            maxBatchSize = 4
        }
    }
    
    // Thermal adjustments
    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
        when (deviceInfo.thermalStatus) {
            DeviceInfo.ThermalStatus.CRITICAL,
            DeviceInfo.ThermalStatus.EMERGENCY -> {
                backend = TrustformersEngine.EngineConfig.Backend.CPU
                numThreads = 1
                memoryOptimization = TrustformersEngine.EngineConfig.MemoryOptimization.MAXIMUM
            }
            DeviceInfo.ThermalStatus.SEVERE -> {
                numThreads = maxOf(1, numThreads / 2)
                memoryOptimization = TrustformersEngine.EngineConfig.MemoryOptimization.MAXIMUM
            }
            DeviceInfo.ThermalStatus.MODERATE -> {
                memoryOptimization = TrustformersEngine.EngineConfig.MemoryOptimization.BALANCED
            }
            else -> {
                // No adjustments needed
            }
        }
    }
}

/**
 * Example usage functions
 */

/**
 * Simple text classification example
 */
suspend fun classifyText(
    context: Context,
    text: String,
    modelPath: String,
    tokenizer: (String) -> FloatArray
): Pair<String, Float> {
    trustformersKt(context) {
        backend = TrustformersEngine.EngineConfig.Backend.AUTO
        useFP16 = true
    }.use { engine ->
        val model = engine.loadModel(modelPath)
        val tokens = tokenizer(text)
        val input = Tensor(tokens, intArrayOf(1, tokens.size))
        
        val output = engine.inference(model, input)
        val probs = output.softmax()
        val topK = probs.topK(1)
        
        val labels = listOf("positive", "negative", "neutral")
        return labels[topK.indices[0]] to topK.values[0]
    }
}

/**
 * Batch image classification example
 */
suspend fun classifyImages(
    context: Context,
    images: List<FloatArray>,
    modelPath: String,
    labels: List<String>
): List<Pair<String, Float>> = coroutineScope {
    trustformersKt(context) {
        backend = TrustformersEngine.EngineConfig.Backend.NNAPI
        enableBatching = true
        maxBatchSize = images.size
    }.use { engine ->
        val model = engine.loadModel(modelPath)
        
        // Process images in parallel
        images.map { imageData ->
            async {
                val input = Tensor(imageData, intArrayOf(1, 3, 224, 224))
                val output = engine.inference(model, input)
                val probs = output.softmax()
                val topK = probs.topK(1)
                
                labels[topK.indices[0]] to topK.values[0]
            }
        }.awaitAll()
    }
}