using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

namespace TrustformersMobile.Performance
{
    /// <summary>
    /// Advanced performance optimization system for TrustformeRS in Unity
    /// </summary>
    public class TrustformersPerformanceOptimizer : MonoBehaviour
    {
        #region Configuration Classes

        [Serializable]
        public class PerformanceConfig
        {
            [Header("Target Performance")]
            public float targetFPS = 60.0f;
            public float lowFPSThreshold = 45.0f;
            public float highFPSThreshold = 75.0f;
            
            [Header("Memory Management")]
            public long maxMemoryUsageMB = 512;
            public float memoryCleanupInterval = 30.0f;
            public bool aggressiveGC = false;
            
            [Header("Thermal Management")]
            public float thermalCheckInterval = 5.0f;
            public bool enableThermalThrottling = true;
            public float thermalThrottleThreshold = 0.8f;
            
            [Header("Battery Optimization")]
            public bool enableBatteryOptimization = true;
            public float lowBatteryThreshold = 0.2f;
            public bool reduceQualityOnLowBattery = true;
            
            [Header("Adaptive Quality")]
            public bool enableAdaptiveQuality = true;
            public float qualityAdjustmentSpeed = 0.1f;
            public int qualityLevels = 5;
        }

        [Serializable]
        public class PerformanceMetrics
        {
            public float currentFPS;
            public float averageFPS;
            public float minFPS;
            public float maxFPS;
            
            public long totalMemoryMB;
            public long usedMemoryMB;
            public long availableMemoryMB;
            
            public float cpuUsage;
            public float gpuUsage;
            public float thermalState;
            public float batteryLevel;
            
            public int currentQualityLevel;
            public bool isThrottling;
            public bool isLowPowerMode;
        }

        [Serializable]
        public class OptimizationAction
        {
            public string name;
            public OptimizationType type;
            public float impact;
            public bool isActive;
            public DateTime lastApplied;
        }

        public enum OptimizationType
        {
            ReduceInferenceFrequency,
            LowerModelPrecision,
            ReduceBatchSize,
            DisableNonEssentialFeatures,
            ReduceTextureQuality,
            LowerFrameRate,
            EnableMemoryOptimization,
            ReduceConcurrentInferences
        }

        #endregion

        #region Public Fields

        [Header("Configuration")]
        public PerformanceConfig config = new PerformanceConfig();
        
        [Header("Monitoring")]
        public bool enableContinuousMonitoring = true;
        public bool enableLogging = true;
        public bool showPerformanceUI = false;
        
        [Header("Target Engines")]
        public TrustformersEngine[] targetEngines;

        #endregion

        #region Private Fields

        private PerformanceMetrics currentMetrics = new PerformanceMetrics();
        private List<OptimizationAction> activeOptimizations = new List<OptimizationAction>();
        
        private float[] fpsHistory = new float[60]; // 1 second of FPS data at 60 FPS
        private int fpsHistoryIndex = 0;
        
        private float lastMemoryCleanup;
        private float lastThermalCheck;
        private float lastQualityAdjustment;
        
        private bool isInitialized = false;
        private Coroutine monitoringCoroutine;
        
        // Performance tracking
        private readonly Dictionary<string, float> performanceTimers = new Dictionary<string, float>();
        private readonly Queue<float> frameTimeHistory = new Queue<float>();
        private const int MAX_FRAME_HISTORY = 300; // 5 seconds at 60 FPS

        #endregion

        #region Events

        public event Action<PerformanceMetrics> OnMetricsUpdated;
        public event Action<OptimizationAction> OnOptimizationApplied;
        public event Action<string> OnPerformanceWarning;

        #endregion

        #region MonoBehaviour Lifecycle

        void Start()
        {
            InitializeOptimizer();
        }

        void Update()
        {
            if (isInitialized && enableContinuousMonitoring)
            {
                UpdateRealTimeMetrics();
                CheckPerformanceThresholds();
            }
        }

        void OnDestroy()
        {
            if (monitoringCoroutine != null)
            {
                StopCoroutine(monitoringCoroutine);
            }
        }

        void OnApplicationFocus(bool hasFocus)
        {
            if (hasFocus)
            {
                ResumeOptimization();
            }
            else
            {
                PauseOptimization();
            }
        }

        void OnApplicationPause(bool pauseStatus)
        {
            if (pauseStatus)
            {
                PauseOptimization();
            }
            else
            {
                ResumeOptimization();
            }
        }

        #endregion

        #region Initialization

        private void InitializeOptimizer()
        {
            if (isInitialized) return;

            try
            {
                // Auto-discover engines if not specified
                if (targetEngines == null || targetEngines.Length == 0)
                {
                    targetEngines = FindObjectsOfType<TrustformersEngine>();
                }

                // Initialize metrics
                currentMetrics = new PerformanceMetrics();
                
                // Start monitoring coroutine
                if (enableContinuousMonitoring)
                {
                    monitoringCoroutine = StartCoroutine(ContinuousMonitoring());
                }

                // Initialize optimization actions
                InitializeOptimizationActions();

                isInitialized = true;
                
                if (enableLogging)
                {
                    Debug.Log("TrustformersPerformanceOptimizer initialized successfully");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to initialize performance optimizer: {e.Message}");
            }
        }

        private void InitializeOptimizationActions()
        {
            activeOptimizations.Clear();
            
            activeOptimizations.Add(new OptimizationAction
            {
                name = "Reduce Inference Frequency",
                type = OptimizationType.ReduceInferenceFrequency,
                impact = 0.3f,
                isActive = false
            });

            activeOptimizations.Add(new OptimizationAction
            {
                name = "Lower Model Precision",
                type = OptimizationType.LowerModelPrecision,
                impact = 0.5f,
                isActive = false
            });

            activeOptimizations.Add(new OptimizationAction
            {
                name = "Reduce Batch Size",
                type = OptimizationType.ReduceBatchSize,
                impact = 0.2f,
                isActive = false
            });

            activeOptimizations.Add(new OptimizationAction
            {
                name = "Enable Memory Optimization",
                type = OptimizationType.EnableMemoryOptimization,
                impact = 0.4f,
                isActive = false
            });

            activeOptimizations.Add(new OptimizationAction
            {
                name = "Reduce Concurrent Inferences",
                type = OptimizationType.ReduceConcurrentInferences,
                impact = 0.6f,
                isActive = false
            });
        }

        #endregion

        #region Performance Monitoring

        private void UpdateRealTimeMetrics()
        {
            // Update FPS metrics
            float currentFPS = 1.0f / Time.deltaTime;
            UpdateFPSHistory(currentFPS);
            
            currentMetrics.currentFPS = currentFPS;
            currentMetrics.averageFPS = CalculateAverageFPS();
            currentMetrics.minFPS = CalculateMinFPS();
            currentMetrics.maxFPS = CalculateMaxFPS();

            // Update memory metrics
            currentMetrics.totalMemoryMB = Profiler.GetTotalAllocatedMemory(0) / (1024 * 1024);
            currentMetrics.usedMemoryMB = Profiler.GetAllocatedMemoryForGraphicsDriver() / (1024 * 1024);
            currentMetrics.availableMemoryMB = currentMetrics.totalMemoryMB - currentMetrics.usedMemoryMB;

            // Update system metrics
            UpdateSystemMetrics();
            
            // Track frame time
            TrackFrameTime(Time.deltaTime * 1000); // Convert to milliseconds
        }

        private void UpdateFPSHistory(float fps)
        {
            fpsHistory[fpsHistoryIndex] = fps;
            fpsHistoryIndex = (fpsHistoryIndex + 1) % fpsHistory.Length;
        }

        private float CalculateAverageFPS()
        {
            float sum = 0;
            for (int i = 0; i < fpsHistory.Length; i++)
            {
                sum += fpsHistory[i];
            }
            return sum / fpsHistory.Length;
        }

        private float CalculateMinFPS()
        {
            float min = float.MaxValue;
            for (int i = 0; i < fpsHistory.Length; i++)
            {
                if (fpsHistory[i] > 0 && fpsHistory[i] < min)
                {
                    min = fpsHistory[i];
                }
            }
            return min == float.MaxValue ? 0 : min;
        }

        private float CalculateMaxFPS()
        {
            float max = 0;
            for (int i = 0; i < fpsHistory.Length; i++)
            {
                if (fpsHistory[i] > max)
                {
                    max = fpsHistory[i];
                }
            }
            return max;
        }

        private void UpdateSystemMetrics()
        {
            // Battery level (mobile platforms)
            currentMetrics.batteryLevel = SystemInfo.batteryLevel;
            currentMetrics.isLowPowerMode = SystemInfo.batteryStatus == BatteryStatus.Unknown;

            // Thermal state estimation (simplified)
            if (Time.time - lastThermalCheck > config.thermalCheckInterval)
            {
                EstimateThermalState();
                lastThermalCheck = Time.time;
            }

            // CPU/GPU usage estimation
            EstimateResourceUsage();
        }

        private void EstimateThermalState()
        {
            // Simplified thermal estimation based on performance degradation
            float performanceDrop = (config.targetFPS - currentMetrics.averageFPS) / config.targetFPS;
            currentMetrics.thermalState = Mathf.Clamp01(performanceDrop * 2.0f);
            
            if (currentMetrics.thermalState > config.thermalThrottleThreshold)
            {
                currentMetrics.isThrottling = true;
                if (config.enableThermalThrottling)
                {
                    ApplyThermalThrottling();
                }
            }
            else
            {
                currentMetrics.isThrottling = false;
            }
        }

        private void EstimateResourceUsage()
        {
            // Simplified resource usage estimation
            float targetFrameTime = 1.0f / config.targetFPS;
            float actualFrameTime = 1.0f / currentMetrics.currentFPS;
            
            currentMetrics.cpuUsage = Mathf.Clamp01(actualFrameTime / targetFrameTime);
            currentMetrics.gpuUsage = currentMetrics.cpuUsage * 0.8f; // Estimate GPU usage
        }

        private void TrackFrameTime(float frameTimeMs)
        {
            frameTimeHistory.Enqueue(frameTimeMs);
            
            if (frameTimeHistory.Count > MAX_FRAME_HISTORY)
            {
                frameTimeHistory.Dequeue();
            }
        }

        #endregion

        #region Continuous Monitoring

        private IEnumerator ContinuousMonitoring()
        {
            while (true)
            {
                yield return new WaitForSeconds(1.0f);
                
                if (isInitialized)
                {
                    PerformDetailedAnalysis();
                    CheckMemoryUsage();
                    UpdateAdaptiveQuality();
                    
                    OnMetricsUpdated?.Invoke(currentMetrics);
                }
            }
        }

        private void PerformDetailedAnalysis()
        {
            // Analyze performance trends
            if (frameTimeHistory.Count > 30) // At least 30 frames
            {
                float averageFrameTime = 0;
                foreach (float frameTime in frameTimeHistory)
                {
                    averageFrameTime += frameTime;
                }
                averageFrameTime /= frameTimeHistory.Count;

                float targetFrameTime = 1000.0f / config.targetFPS; // In milliseconds
                
                if (averageFrameTime > targetFrameTime * 1.2f)
                {
                    OnPerformanceWarning?.Invoke($"Average frame time ({averageFrameTime:F2}ms) exceeds target ({targetFrameTime:F2}ms)");
                    ConsiderPerformanceOptimizations();
                }
            }
        }

        private void CheckMemoryUsage()
        {
            if (Time.time - lastMemoryCleanup > config.memoryCleanupInterval)
            {
                if (currentMetrics.usedMemoryMB > config.maxMemoryUsageMB * 0.8f)
                {
                    TriggerMemoryCleanup();
                    lastMemoryCleanup = Time.time;
                }
            }
        }

        private void UpdateAdaptiveQuality()
        {
            if (!config.enableAdaptiveQuality) return;
            
            if (Time.time - lastQualityAdjustment > 2.0f) // Check every 2 seconds
            {
                float performanceRatio = currentMetrics.averageFPS / config.targetFPS;
                
                if (performanceRatio < 0.9f && currentMetrics.currentQualityLevel > 1)
                {
                    // Reduce quality
                    currentMetrics.currentQualityLevel = Mathf.Max(1, 
                        currentMetrics.currentQualityLevel - 1);
                    ApplyQualityLevel(currentMetrics.currentQualityLevel);
                    lastQualityAdjustment = Time.time;
                }
                else if (performanceRatio > 1.1f && currentMetrics.currentQualityLevel < config.qualityLevels)
                {
                    // Increase quality
                    currentMetrics.currentQualityLevel = Mathf.Min(config.qualityLevels, 
                        currentMetrics.currentQualityLevel + 1);
                    ApplyQualityLevel(currentMetrics.currentQualityLevel);
                    lastQualityAdjustment = Time.time;
                }
            }
        }

        #endregion

        #region Performance Optimization

        private void CheckPerformanceThresholds()
        {
            // Check FPS thresholds
            if (currentMetrics.currentFPS < config.lowFPSThreshold)
            {
                if (enableLogging)
                {
                    Debug.LogWarning($"Low FPS detected: {currentMetrics.currentFPS:F1}");
                }
                TriggerPerformanceOptimization();
            }

            // Check memory thresholds
            if (currentMetrics.usedMemoryMB > config.maxMemoryUsageMB)
            {
                if (enableLogging)
                {
                    Debug.LogWarning($"Memory usage exceeds limit: {currentMetrics.usedMemoryMB}MB");
                }
                TriggerMemoryOptimization();
            }

            // Check battery optimization
            if (config.enableBatteryOptimization && 
                currentMetrics.batteryLevel < config.lowBatteryThreshold &&
                config.reduceQualityOnLowBattery)
            {
                TriggerBatteryOptimization();
            }
        }

        private void ConsiderPerformanceOptimizations()
        {
            // Sort optimizations by impact (highest first)
            activeOptimizations.Sort((a, b) => b.impact.CompareTo(a.impact));
            
            foreach (var optimization in activeOptimizations)
            {
                if (!optimization.isActive)
                {
                    ApplyOptimization(optimization);
                    break; // Apply one optimization at a time
                }
            }
        }

        private void TriggerPerformanceOptimization()
        {
            ApplyOptimization(OptimizationType.ReduceInferenceFrequency);
            ApplyOptimization(OptimizationType.ReduceConcurrentInferences);
        }

        private void TriggerMemoryOptimization()
        {
            ApplyOptimization(OptimizationType.EnableMemoryOptimization);
            TriggerMemoryCleanup();
        }

        private void TriggerBatteryOptimization()
        {
            ApplyOptimization(OptimizationType.LowerFrameRate);
            ApplyOptimization(OptimizationType.LowerModelPrecision);
            ApplyOptimization(OptimizationType.ReduceInferenceFrequency);
        }

        private void ApplyThermalThrottling()
        {
            ApplyOptimization(OptimizationType.ReduceInferenceFrequency);
            ApplyOptimization(OptimizationType.LowerModelPrecision);
            ApplyOptimization(OptimizationType.ReduceConcurrentInferences);
        }

        private void ApplyOptimization(OptimizationType type)
        {
            var optimization = activeOptimizations.Find(o => o.type == type);
            if (optimization != null && !optimization.isActive)
            {
                ApplyOptimization(optimization);
            }
        }

        private void ApplyOptimization(OptimizationAction optimization)
        {
            try
            {
                switch (optimization.type)
                {
                    case OptimizationType.ReduceInferenceFrequency:
                        ReduceInferenceFrequency();
                        break;
                    case OptimizationType.LowerModelPrecision:
                        LowerModelPrecision();
                        break;
                    case OptimizationType.ReduceBatchSize:
                        ReduceBatchSize();
                        break;
                    case OptimizationType.EnableMemoryOptimization:
                        EnableMemoryOptimization();
                        break;
                    case OptimizationType.ReduceConcurrentInferences:
                        ReduceConcurrentInferences();
                        break;
                    case OptimizationType.LowerFrameRate:
                        LowerFrameRate();
                        break;
                }

                optimization.isActive = true;
                optimization.lastApplied = DateTime.Now;
                
                OnOptimizationApplied?.Invoke(optimization);
                
                if (enableLogging)
                {
                    Debug.Log($"Applied optimization: {optimization.name}");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to apply optimization {optimization.name}: {e.Message}");
            }
        }

        #endregion

        #region Specific Optimizations

        private void ReduceInferenceFrequency()
        {
            foreach (var engine in targetEngines)
            {
                if (engine != null)
                {
                    // This would require adding a method to TrustformersEngine
                    // engine.SetInferenceInterval(engine.GetInferenceInterval() * 1.5f);
                }
            }
        }

        private void LowerModelPrecision()
        {
            foreach (var engine in targetEngines)
            {
                if (engine != null && engine.config.quantization.enabled)
                {
                    // Switch to more aggressive quantization
                    if (engine.config.quantization.scheme == TrustformersEngine.QuantizationScheme.FP16)
                    {
                        engine.config.quantization.scheme = TrustformersEngine.QuantizationScheme.Int8;
                    }
                    else if (engine.config.quantization.scheme == TrustformersEngine.QuantizationScheme.Int8)
                    {
                        engine.config.quantization.scheme = TrustformersEngine.QuantizationScheme.Int4;
                    }
                }
            }
        }

        private void ReduceBatchSize()
        {
            foreach (var engine in targetEngines)
            {
                if (engine != null && engine.config.enableBatching)
                {
                    engine.config.maxBatchSize = Mathf.Max(1, engine.config.maxBatchSize / 2);
                }
            }
        }

        private void EnableMemoryOptimization()
        {
            foreach (var engine in targetEngines)
            {
                if (engine != null)
                {
                    engine.config.memoryOptimization = TrustformersEngine.MemoryOptimization.Maximum;
                }
            }
        }

        private void ReduceConcurrentInferences()
        {
            foreach (var engine in targetEngines)
            {
                if (engine != null)
                {
                    engine.config.performance.maxConcurrentInferences = 
                        Mathf.Max(1, engine.config.performance.maxConcurrentInferences - 1);
                }
            }
        }

        private void LowerFrameRate()
        {
            Application.targetFrameRate = Mathf.Max(30, (int)(config.targetFPS * 0.75f));
        }

        private void ApplyQualityLevel(int level)
        {
            float qualityMultiplier = (float)level / config.qualityLevels;
            
            foreach (var engine in targetEngines)
            {
                if (engine != null)
                {
                    // Adjust inference frequency based on quality level
                    float baseInterval = 0.1f; // 10 FPS base
                    float targetInterval = baseInterval / qualityMultiplier;
                    
                    // This would require adding a method to TrustformersEngine
                    // engine.SetInferenceInterval(targetInterval);
                }
            }
        }

        private void TriggerMemoryCleanup()
        {
            if (config.aggressiveGC)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }
            
            // Trigger native memory cleanup
            foreach (var engine in targetEngines)
            {
                if (engine != null)
                {
                    // This would require adding a method to TrustformersEngine
                    // engine.CleanupMemory();
                }
            }
        }

        #endregion

        #region Pause/Resume

        private void PauseOptimization()
        {
            if (monitoringCoroutine != null)
            {
                StopCoroutine(monitoringCoroutine);
                monitoringCoroutine = null;
            }
            
            if (enableLogging)
            {
                Debug.Log("Performance optimization paused");
            }
        }

        private void ResumeOptimization()
        {
            if (monitoringCoroutine == null && enableContinuousMonitoring)
            {
                monitoringCoroutine = StartCoroutine(ContinuousMonitoring());
            }
            
            if (enableLogging)
            {
                Debug.Log("Performance optimization resumed");
            }
        }

        #endregion

        #region Public API

        /// <summary>
        /// Get current performance metrics
        /// </summary>
        public PerformanceMetrics GetMetrics()
        {
            return currentMetrics;
        }

        /// <summary>
        /// Force apply a specific optimization
        /// </summary>
        public void ForceApplyOptimization(OptimizationType type)
        {
            ApplyOptimization(type);
        }

        /// <summary>
        /// Reset all optimizations to default state
        /// </summary>
        public void ResetOptimizations()
        {
            foreach (var optimization in activeOptimizations)
            {
                optimization.isActive = false;
            }
            
            // Reset engine configurations to defaults
            foreach (var engine in targetEngines)
            {
                if (engine != null)
                {
                    engine.config = TrustformersEngine.GetRecommendedConfig();
                }
            }
            
            Application.targetFrameRate = (int)config.targetFPS;
            currentMetrics.currentQualityLevel = config.qualityLevels;
        }

        /// <summary>
        /// Get performance optimization report
        /// </summary>
        public string GetOptimizationReport()
        {
            var report = "TrustformeRS Performance Report:\n\n";
            
            report += $"Current FPS: {currentMetrics.currentFPS:F1}\n";
            report += $"Average FPS: {currentMetrics.averageFPS:F1}\n";
            report += $"Memory Usage: {currentMetrics.usedMemoryMB}MB / {config.maxMemoryUsageMB}MB\n";
            report += $"Quality Level: {currentMetrics.currentQualityLevel}/{config.qualityLevels}\n";
            report += $"Is Throttling: {currentMetrics.isThrottling}\n\n";
            
            report += "Active Optimizations:\n";
            foreach (var optimization in activeOptimizations)
            {
                if (optimization.isActive)
                {
                    report += $"- {optimization.name} (Impact: {optimization.impact:P})\n";
                }
            }
            
            return report;
        }

        /// <summary>
        /// Set new performance targets
        /// </summary>
        public void UpdatePerformanceTargets(float targetFPS, long maxMemoryMB)
        {
            config.targetFPS = targetFPS;
            config.maxMemoryUsageMB = maxMemoryMB;
            
            // Recalculate thresholds
            config.lowFPSThreshold = targetFPS * 0.75f;
            config.highFPSThreshold = targetFPS * 1.25f;
        }

        #endregion

        #region Debug UI

        void OnGUI()
        {
            if (!showPerformanceUI) return;
            
            GUILayout.BeginArea(new Rect(10, 10, 300, 400));
            GUILayout.BeginVertical("box");
            
            GUILayout.Label("TrustformeRS Performance", EditorGUIUtility.boldLabel);
            
            GUILayout.Label($"FPS: {currentMetrics.currentFPS:F1} (Avg: {currentMetrics.averageFPS:F1})");
            GUILayout.Label($"Memory: {currentMetrics.usedMemoryMB}MB");
            GUILayout.Label($"Quality: {currentMetrics.currentQualityLevel}/{config.qualityLevels}");
            GUILayout.Label($"Battery: {currentMetrics.batteryLevel:P}");
            GUILayout.Label($"Thermal: {currentMetrics.thermalState:P}");
            
            GUILayout.Space(10);
            GUILayout.Label("Active Optimizations:");
            
            foreach (var optimization in activeOptimizations)
            {
                if (optimization.isActive)
                {
                    GUILayout.Label($"• {optimization.name}");
                }
            }
            
            GUILayout.Space(10);
            
            if (GUILayout.Button("Reset Optimizations"))
            {
                ResetOptimizations();
            }
            
            if (GUILayout.Button("Force Memory Cleanup"))
            {
                TriggerMemoryCleanup();
            }
            
            GUILayout.EndVertical();
            GUILayout.EndArea();
        }

        #endregion
    }
}