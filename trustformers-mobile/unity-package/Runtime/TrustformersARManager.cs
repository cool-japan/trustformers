using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

#if TRUSTFORMERS_AR_FOUNDATION
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
#endif

namespace TrustformersMobile.AR
{
    /// <summary>
    /// Integrates TrustformeRS with Unity's AR Foundation for real-time AR inference
    /// </summary>
    public class TrustformersARManager : MonoBehaviour
    {
        #region Configuration Classes

        [Serializable]
        public class ARInferenceConfig
        {
            [Header("Inference Settings")]
            public float inferenceInterval = 0.1f; // 10 FPS
            public bool enableObjectDetection = true;
            public bool enablePoseEstimation = false;
            public bool enablePlaneClassification = true;
            public bool enableLightEstimation = false;

            [Header("Performance")]
            public int maxConcurrentInferences = 1;
            public bool adaptiveQuality = true;
            public bool thermalThrottling = true;
            public float targetFPS = 30.0f;

            [Header("Processing")]
            public Vector2Int inputResolution = new Vector2Int(640, 480);
            public bool useGPUMemory = true;
            public bool enableBatching = false;
        }

        [Serializable]
        public class ObjectDetectionResult
        {
            public string className;
            public float confidence;
            public Rect boundingBox;
            public Vector3 worldPosition;
            public Quaternion worldRotation;
        }

        [Serializable]
        public class PoseEstimationResult
        {
            public Vector3[] keypoints;
            public float[] confidences;
            public string personId;
            public float overallConfidence;
        }

        [Serializable]
        public class PlaneClassificationResult
        {
            public PlaneClassification classification;
            public float confidence;
            public Vector3 center;
            public Vector3 size;
        }

        #endregion

        #region Public Fields

        [Header("AR Configuration")]
        public ARInferenceConfig config = new ARInferenceConfig();

        [Header("Model Paths")]
        public string objectDetectionModelPath = "";
        public string poseEstimationModelPath = "";
        public string planeClassificationModelPath = "";

        [Header("Visualization")]
        public bool showBoundingBoxes = true;
        public bool showPoseKeypoints = true;
        public bool showPlaneLabels = true;
        public GameObject boundingBoxPrefab;
        public GameObject keypointPrefab;

        #endregion

        #region Private Fields

#if TRUSTFORMERS_AR_FOUNDATION
        private ARCameraManager arCameraManager;
        private ARPlaneManager arPlaneManager;
        private ARSessionOrigin arSessionOrigin;
#endif

        private TrustformersEngine engine;
        private Camera arCamera;
        
        private float lastInferenceTime;
        private bool isProcessing = false;
        private int activeInferences = 0;
        private readonly object lockObject = new object();

        private List<GameObject> visualizationObjects = new List<GameObject>();
        private Dictionary<string, TrustformersEngine> modelEngines = new Dictionary<string, TrustformersEngine>();

        #endregion

        #region Events

        public event Action<ObjectDetectionResult[]> OnObjectsDetected;
        public event Action<PoseEstimationResult[]> OnPosesEstimated;
        public event Action<PlaneClassificationResult[]> OnPlanesClassified;
        public event Action<string> OnARError;

        #endregion

        #region MonoBehaviour Lifecycle

        void Start()
        {
            InitializeARComponents();
            InitializeEngines();
        }

        void Update()
        {
            if (ShouldProcessFrame())
            {
                ProcessARFrame();
            }

            UpdatePerformanceAdaptation();
        }

        void OnDestroy()
        {
            CleanupVisualization();
            CleanupEngines();
        }

        #endregion

        #region Initialization

        private void InitializeARComponents()
        {
#if TRUSTFORMERS_AR_FOUNDATION
            arCameraManager = FindObjectOfType<ARCameraManager>();
            arPlaneManager = FindObjectOfType<ARPlaneManager>();
            arSessionOrigin = FindObjectOfType<ARSessionOrigin>();
            arCamera = arSessionOrigin?.camera;

            if (arCameraManager == null)
            {
                Debug.LogError("ARCameraManager not found! Please add AR Session Origin to the scene.");
                OnARError?.Invoke("ARCameraManager not found");
                return;
            }

            // Subscribe to AR events
            if (arPlaneManager != null && config.enablePlaneClassification)
            {
                arPlaneManager.planesChanged += OnPlanesChanged;
            }
#else
            Debug.LogWarning("AR Foundation not available. AR features will be disabled.");
#endif
        }

        private void InitializeEngines()
        {
            // Create engines for different model types
            if (config.enableObjectDetection && !string.IsNullOrEmpty(objectDetectionModelPath))
            {
                CreateModelEngine("object_detection", objectDetectionModelPath);
            }

            if (config.enablePoseEstimation && !string.IsNullOrEmpty(poseEstimationModelPath))
            {
                CreateModelEngine("pose_estimation", poseEstimationModelPath);
            }

            if (config.enablePlaneClassification && !string.IsNullOrEmpty(planeClassificationModelPath))
            {
                CreateModelEngine("plane_classification", planeClassificationModelPath);
            }
        }

        private void CreateModelEngine(string modelType, string modelPath)
        {
            var engineObject = new GameObject($"TrustformersEngine_{modelType}");
            engineObject.transform.SetParent(transform);
            
            var engine = engineObject.AddComponent<TrustformersEngine>();
            
            // Configure for AR performance
            engine.config = TrustformersEngine.GetRecommendedConfig();
            engine.config.memoryOptimization = TrustformersEngine.MemoryOptimization.Balanced;
            engine.config.enableBatching = config.enableBatching;
            engine.config.performance.adaptivePerformance = config.adaptiveQuality;
            engine.config.performance.thermalThrottling = config.thermalThrottling;
            engine.config.performance.targetFPS = config.targetFPS;
            
            engine.modelPath = modelPath;
            engine.loadModelOnStart = true;
            
            modelEngines[modelType] = engine;
        }

        #endregion

        #region AR Processing

        private bool ShouldProcessFrame()
        {
            if (isProcessing || activeInferences >= config.maxConcurrentInferences)
                return false;

            float timeSinceLastInference = Time.time - lastInferenceTime;
            return timeSinceLastInference >= config.inferenceInterval;
        }

        private async void ProcessARFrame()
        {
#if TRUSTFORMERS_AR_FOUNDATION
            if (arCameraManager == null || !arCameraManager.TryAcquireLatestCpuImage(out var image))
                return;

            lock (lockObject)
            {
                if (isProcessing) return;
                isProcessing = true;
                activeInferences++;
            }

            lastInferenceTime = Time.time;

            try
            {
                // Convert AR camera image to inference format
                var inputData = await ConvertImageToInputData(image);
                
                // Run concurrent inferences for enabled features
                var tasks = new List<Task>();

                if (config.enableObjectDetection && modelEngines.ContainsKey("object_detection"))
                {
                    tasks.Add(ProcessObjectDetection(inputData));
                }

                if (config.enablePoseEstimation && modelEngines.ContainsKey("pose_estimation"))
                {
                    tasks.Add(ProcessPoseEstimation(inputData));
                }

                await Task.WhenAll(tasks);
            }
            catch (Exception e)
            {
                Debug.LogError($"AR processing error: {e.Message}");
                OnARError?.Invoke($"Processing error: {e.Message}");
            }
            finally
            {
                image.Dispose();
                lock (lockObject)
                {
                    isProcessing = false;
                    activeInferences--;
                }
            }
#endif
        }

        private async Task<float[]> ConvertImageToInputData(
#if TRUSTFORMERS_AR_FOUNDATION
            XRCpuImage image
#else
            object image
#endif
        )
        {
#if TRUSTFORMERS_AR_FOUNDATION
            // Convert XRCpuImage to normalized float array
            var conversionParams = new XRCpuImage.ConversionParams
            {
                inputRect = new RectInt(0, 0, image.width, image.height),
                outputDimensions = config.inputResolution,
                outputFormat = TextureFormat.RGB24,
                transformation = XRCpuImage.Transformation.MirrorY
            };

            int size = config.inputResolution.x * config.inputResolution.y * 3;
            var rawData = new byte[size];
            
            await Task.Run(() => {
                image.Convert(conversionParams, rawData);
            });

            // Normalize to [0, 1] and convert to float array
            var floatData = new float[size];
            for (int i = 0; i < size; i++)
            {
                floatData[i] = rawData[i] / 255.0f;
            }

            return floatData;
#else
            return new float[0];
#endif
        }

        private async Task ProcessObjectDetection(float[] inputData)
        {
            try
            {
                var engine = modelEngines["object_detection"];
                var output = await engine.InferenceAsync(inputData);
                
                if (output != null)
                {
                    var detections = ParseObjectDetectionOutput(output);
                    
                    // Convert screen space to world space
                    var worldDetections = ConvertToWorldSpace(detections);
                    
                    OnObjectsDetected?.Invoke(worldDetections);
                    
                    if (showBoundingBoxes)
                    {
                        UpdateObjectVisualization(worldDetections);
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Object detection error: {e.Message}");
                OnARError?.Invoke($"Object detection failed: {e.Message}");
            }
        }

        private async Task ProcessPoseEstimation(float[] inputData)
        {
            try
            {
                var engine = modelEngines["pose_estimation"];
                var output = await engine.InferenceAsync(inputData);
                
                if (output != null)
                {
                    var poses = ParsePoseEstimationOutput(output);
                    
                    OnPosesEstimated?.Invoke(poses);
                    
                    if (showPoseKeypoints)
                    {
                        UpdatePoseVisualization(poses);
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Pose estimation error: {e.Message}");
                OnARError?.Invoke($"Pose estimation failed: {e.Message}");
            }
        }

        #endregion

        #region Output Parsing

        private ObjectDetectionResult[] ParseObjectDetectionOutput(float[] output)
        {
            var results = new List<ObjectDetectionResult>();
            
            // Simplified parsing - actual format depends on model
            // Assuming format: [class_id, confidence, x, y, width, height, ...]
            int stride = 6;
            for (int i = 0; i < output.Length - stride; i += stride)
            {
                float confidence = output[i + 1];
                if (confidence > 0.5f) // Confidence threshold
                {
                    var result = new ObjectDetectionResult
                    {
                        className = GetClassName((int)output[i]),
                        confidence = confidence,
                        boundingBox = new Rect(output[i + 2], output[i + 3], output[i + 4], output[i + 5])
                    };
                    results.Add(result);
                }
            }
            
            return results.ToArray();
        }

        private PoseEstimationResult[] ParsePoseEstimationOutput(float[] output)
        {
            var results = new List<PoseEstimationResult>();
            
            // Simplified parsing for pose estimation
            // Assuming format: [person_id, confidence, keypoint_x1, keypoint_y1, conf1, ...]
            int keypointsPerPerson = 17; // Standard human pose
            int stride = 2 + (keypointsPerPerson * 3); // id + conf + (x,y,conf) per keypoint
            
            for (int i = 0; i < output.Length - stride; i += stride)
            {
                float overallConfidence = output[i + 1];
                if (overallConfidence > 0.3f)
                {
                    var keypoints = new Vector3[keypointsPerPerson];
                    var confidences = new float[keypointsPerPerson];
                    
                    for (int j = 0; j < keypointsPerPerson; j++)
                    {
                        int offset = i + 2 + (j * 3);
                        keypoints[j] = new Vector3(output[offset], output[offset + 1], 0);
                        confidences[j] = output[offset + 2];
                    }
                    
                    var result = new PoseEstimationResult
                    {
                        personId = ((int)output[i]).ToString(),
                        overallConfidence = overallConfidence,
                        keypoints = keypoints,
                        confidences = confidences
                    };
                    results.Add(result);
                }
            }
            
            return results.ToArray();
        }

        #endregion

        #region Coordinate Conversion

        private ObjectDetectionResult[] ConvertToWorldSpace(ObjectDetectionResult[] screenDetections)
        {
#if TRUSTFORMERS_AR_FOUNDATION
            if (arCamera == null) return screenDetections;

            foreach (var detection in screenDetections)
            {
                // Convert screen space bounding box center to world position
                Vector3 screenCenter = new Vector3(
                    detection.boundingBox.center.x * Screen.width,
                    detection.boundingBox.center.y * Screen.height,
                    1.0f
                );

                Ray ray = arCamera.ScreenPointToRay(screenCenter);
                
                // Simplified world position estimation (would need depth info for accuracy)
                detection.worldPosition = ray.GetPoint(2.0f); // 2 meters from camera
                detection.worldRotation = Quaternion.LookRotation(ray.direction);
            }
#endif
            return screenDetections;
        }

        #endregion

        #region Visualization

        private void UpdateObjectVisualization(ObjectDetectionResult[] detections)
        {
            CleanupVisualization();
            
            if (boundingBoxPrefab == null) return;

            foreach (var detection in detections)
            {
                var visualObj = Instantiate(boundingBoxPrefab, detection.worldPosition, detection.worldRotation);
                visualObj.transform.SetParent(transform);
                
                // Add detection info to the visualization
                var textMesh = visualObj.GetComponentInChildren<TextMesh>();
                if (textMesh != null)
                {
                    textMesh.text = $"{detection.className}\n{detection.confidence:F2}";
                }
                
                visualizationObjects.Add(visualObj);
            }
        }

        private void UpdatePoseVisualization(PoseEstimationResult[] poses)
        {
            if (keypointPrefab == null) return;

            foreach (var pose in poses)
            {
                for (int i = 0; i < pose.keypoints.Length; i++)
                {
                    if (pose.confidences[i] > 0.5f)
                    {
                        var worldPos = arCamera.ScreenToWorldPoint(new Vector3(
                            pose.keypoints[i].x * Screen.width,
                            pose.keypoints[i].y * Screen.height,
                            2.0f
                        ));
                        
                        var visualObj = Instantiate(keypointPrefab, worldPos, Quaternion.identity);
                        visualObj.transform.SetParent(transform);
                        visualizationObjects.Add(visualObj);
                    }
                }
            }
        }

        private void CleanupVisualization()
        {
            foreach (var obj in visualizationObjects)
            {
                if (obj != null)
                {
                    DestroyImmediate(obj);
                }
            }
            visualizationObjects.Clear();
        }

        #endregion

        #region Performance Adaptation

        private void UpdatePerformanceAdaptation()
        {
            if (!config.adaptiveQuality) return;

            float currentFPS = 1.0f / Time.deltaTime;
            
            if (currentFPS < config.targetFPS * 0.8f)
            {
                // Reduce quality to improve performance
                if (config.inferenceInterval < 0.5f)
                {
                    config.inferenceInterval += 0.01f;
                }
            }
            else if (currentFPS > config.targetFPS * 1.1f)
            {
                // Increase quality if performance allows
                if (config.inferenceInterval > 0.05f)
                {
                    config.inferenceInterval -= 0.01f;
                }
            }
        }

        #endregion

        #region AR Foundation Event Handlers

#if TRUSTFORMERS_AR_FOUNDATION
        private async void OnPlanesChanged(ARPlanesChangedEventArgs eventArgs)
        {
            if (!config.enablePlaneClassification || !modelEngines.ContainsKey("plane_classification"))
                return;

            foreach (var plane in eventArgs.added)
            {
                await ClassifyPlane(plane);
            }

            foreach (var plane in eventArgs.updated)
            {
                await ClassifyPlane(plane);
            }
        }

        private async Task ClassifyPlane(ARPlane plane)
        {
            try
            {
                // Create input data from plane geometry
                var inputData = CreatePlaneInputData(plane);
                
                var engine = modelEngines["plane_classification"];
                var output = await engine.InferenceAsync(inputData);
                
                if (output != null)
                {
                    var classification = ParsePlaneClassificationOutput(output);
                    classification.center = plane.center;
                    classification.size = plane.size;
                    
                    OnPlanesClassified?.Invoke(new[] { classification });
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Plane classification error: {e.Message}");
                OnARError?.Invoke($"Plane classification failed: {e.Message}");
            }
        }
#endif

        #endregion

        #region Helper Methods

        private string GetClassName(int classId)
        {
            // This would typically come from a class names file
            var classNames = new string[]
            {
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee"
            };
            
            return classId >= 0 && classId < classNames.Length ? classNames[classId] : "unknown";
        }

        private float[] CreatePlaneInputData(
#if TRUSTFORMERS_AR_FOUNDATION
            ARPlane plane
#else
            object plane
#endif
        )
        {
#if TRUSTFORMERS_AR_FOUNDATION
            // Create simple feature vector from plane properties
            return new float[]
            {
                plane.size.x, plane.size.y,
                plane.normal.x, plane.normal.y, plane.normal.z,
                (float)plane.classification,
                plane.alignment == PlaneAlignment.HorizontalUp ? 1.0f : 0.0f,
                plane.alignment == PlaneAlignment.HorizontalDown ? 1.0f : 0.0f,
                plane.alignment == PlaneAlignment.Vertical ? 1.0f : 0.0f
            };
#else
            return new float[9];
#endif
        }

        private PlaneClassificationResult ParsePlaneClassificationOutput(float[] output)
        {
            // Simplified classification parsing
            int maxIndex = 0;
            float maxConfidence = output[0];
            
            for (int i = 1; i < output.Length; i++)
            {
                if (output[i] > maxConfidence)
                {
                    maxConfidence = output[i];
                    maxIndex = i;
                }
            }
            
            return new PlaneClassificationResult
            {
#if TRUSTFORMERS_AR_FOUNDATION
                classification = (PlaneClassification)maxIndex,
#endif
                confidence = maxConfidence
            };
        }

        private void CleanupEngines()
        {
            foreach (var engine in modelEngines.Values)
            {
                if (engine != null && engine.gameObject != null)
                {
                    DestroyImmediate(engine.gameObject);
                }
            }
            modelEngines.Clear();
        }

        #endregion

        #region Public API

        /// <summary>
        /// Enable or disable specific AR inference features
        /// </summary>
        public void SetFeatureEnabled(string feature, bool enabled)
        {
            switch (feature.ToLower())
            {
                case "object_detection":
                    config.enableObjectDetection = enabled;
                    break;
                case "pose_estimation":
                    config.enablePoseEstimation = enabled;
                    break;
                case "plane_classification":
                    config.enablePlaneClassification = enabled;
                    break;
                case "light_estimation":
                    config.enableLightEstimation = enabled;
                    break;
            }
        }

        /// <summary>
        /// Update inference interval for performance tuning
        /// </summary>
        public void SetInferenceInterval(float interval)
        {
            config.inferenceInterval = Mathf.Max(0.01f, interval);
        }

        /// <summary>
        /// Get current AR processing statistics
        /// </summary>
        public Dictionary<string, object> GetARStats()
        {
            var stats = new Dictionary<string, object>
            {
                ["active_inferences"] = activeInferences,
                ["max_concurrent"] = config.maxConcurrentInferences,
                ["inference_interval"] = config.inferenceInterval,
                ["current_fps"] = 1.0f / Time.deltaTime,
                ["target_fps"] = config.targetFPS,
                ["is_processing"] = isProcessing
            };

            foreach (var kvp in modelEngines)
            {
                if (kvp.Value != null)
                {
                    var engineStats = kvp.Value.GetStats();
                    stats[$"{kvp.Key}_memory_mb"] = engineStats.memoryUsageMB;
                    stats[$"{kvp.Key}_avg_time_ms"] = engineStats.avgInferenceTimeMs;
                    stats[$"{kvp.Key}_total_inferences"] = engineStats.totalInferences;
                }
            }

            return stats;
        }

        #endregion
    }
}