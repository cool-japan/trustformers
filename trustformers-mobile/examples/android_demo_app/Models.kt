package com.trustformers.model

import android.graphics.Bitmap

/**
 * Data models for TrustformersRS Android demo
 */

// Text Classification Models
data class TextClassificationResult(
    val predictions: List<Prediction>,
    val processingTimeMs: Double
)

data class Prediction(
    val label: String,
    val confidence: Double
)

// Object Detection Models
data class ObjectDetectionResult(
    val detections: List<Detection>,
    val processingTimeMs: Double
)

data class Detection(
    val className: String,
    val confidence: Double,
    val bbox: BoundingBox
)

data class BoundingBox(
    val x: Double,
    val y: Double,
    val width: Double,
    val height: Double
)

// Pose Estimation Models
data class PoseEstimationResult(
    val poses: List<HumanPose>,
    val processingTimeMs: Double
)

data class HumanPose(
    val keypoints: List<Keypoint>,
    val confidence: Double,
    val boundingBox: BoundingBox?
)

data class Keypoint(
    val type: KeypointType,
    val x: Double,
    val y: Double,
    val confidence: Double
)

enum class KeypointType {
    NOSE,           // 0
    LEFT_EYE,       // 1
    RIGHT_EYE,      // 2
    LEFT_EAR,       // 3
    RIGHT_EAR,      // 4
    LEFT_SHOULDER,  // 5
    RIGHT_SHOULDER, // 6
    LEFT_ELBOW,     // 7
    RIGHT_ELBOW,    // 8
    LEFT_WRIST,     // 9
    RIGHT_WRIST,    // 10
    LEFT_HIP,       // 11
    RIGHT_HIP,      // 12
    LEFT_KNEE,      // 13
    RIGHT_KNEE,     // 14
    LEFT_ANKLE,     // 15
    RIGHT_ANKLE     // 16
}

// Performance Models
data class PerformanceMetrics(
    val cpuUsage: Double,
    val memoryUsageMB: Double,
    val gpuUsage: Double,
    val batteryLevel: Double,
    val thermalState: String,
    val processingQueueSize: Int
)

data class DeviceInfo(
    val deviceModel: String,
    val androidVersion: String,
    val apiLevel: Int,
    val hasNNAPI: Boolean,
    val hasGPU: Boolean,
    val hasVulkan: Boolean,
    val performanceTier: PerformanceTier
)

enum class PerformanceTier {
    LOW, MEDIUM, HIGH, EXTREME
}

data class ModelInfo(
    val name: String,
    val type: ModelType,
    val backend: InferenceBackend,
    val sizeBytes: Long,
    val isLoaded: Boolean
)

enum class ModelType {
    TEXT_CLASSIFICATION,
    OBJECT_DETECTION,
    POSE_ESTIMATION,
    CUSTOM
}

enum class InferenceBackend {
    NNAPI,
    GPU,
    VULKAN,
    CPU,
    HYBRID
}

// Configuration Models
data class TrustformersConfig(
    val enableNNAPI: Boolean,
    val enableGPU: Boolean,
    val enableVulkan: Boolean,
    val maxConcurrentInferences: Int,
    val batchSize: Int,
    val useQuantization: Boolean,
    val powerMode: PowerMode
) {
    class Builder {
        private var enableNNAPI = false
        private var enableGPU = false
        private var enableVulkan = false
        private var maxConcurrentInferences = 1
        private var batchSize = 1
        private var useQuantization = true
        private var powerMode = PowerMode.BALANCED
        
        fun enableNNAPI(enable: Boolean) = apply { this.enableNNAPI = enable }
        fun enableGPU(enable: Boolean) = apply { this.enableGPU = enable }
        fun enableVulkan(enable: Boolean) = apply { this.enableVulkan = enable }
        fun setMaxConcurrentInferences(max: Int) = apply { this.maxConcurrentInferences = max }
        fun setBatchSize(size: Int) = apply { this.batchSize = size }
        fun useQuantization(use: Boolean) = apply { this.useQuantization = use }
        fun setPowerMode(mode: PowerMode) = apply { this.powerMode = mode }
        
        fun build() = TrustformersConfig(
            enableNNAPI = enableNNAPI,
            enableGPU = enableGPU,
            enableVulkan = enableVulkan,
            maxConcurrentInferences = maxConcurrentInferences,
            batchSize = batchSize,
            useQuantization = useQuantization,
            powerMode = powerMode
        )
    }
}

enum class PowerMode {
    LOW_POWER,
    BALANCED,
    HIGH_PERFORMANCE
}

// Inference Results
sealed class InferenceResult {
    data class Success<T>(val data: T, val processingTimeMs: Double) : InferenceResult()
    data class Error(val exception: Throwable) : InferenceResult()
    object Loading : InferenceResult()
}

// Model Loading States
sealed class ModelLoadState {
    object NotLoaded : ModelLoadState()
    object Loading : ModelLoadState()
    object Loaded : ModelLoadState()
    data class Error(val message: String) : ModelLoadState()
}

// Inference Tasks
sealed class InferenceTask {
    data class TextClassification(val text: String) : InferenceTask()
    data class ObjectDetection(val bitmap: Bitmap) : InferenceTask()
    data class PoseEstimation(val bitmap: Bitmap) : InferenceTask()
}

// Runtime Statistics
data class RuntimeStats(
    val averageInferenceTimeMs: Double,
    val totalInferences: Long,
    val successRate: Double,
    val memoryUsageMB: Double,
    val activeModels: Int
)