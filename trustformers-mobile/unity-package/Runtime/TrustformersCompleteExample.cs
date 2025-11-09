using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;

namespace TrustformeRS.Mobile
{
    /// <summary>
    /// Complete Unity example demonstrating all TrustformeRS mobile features
    /// </summary>
    public class TrustformersCompleteExample : MonoBehaviour
    {
        [Header("UI References")]
        public Button initializeButton;
        public Button runInferenceButton;
        public Dropdown modelDropdown;
        public Text statusText;
        public Text deviceInfoText;
        public Text performanceText;
        public Text resultsText;
        public Slider progressSlider;
        public Toggle enableGpuToggle;
        public Toggle enableQuantizationToggle;
        public InputField customInputField;

        [Header("Configuration")]
        public bool enableDebugLogging = true;
        public bool enablePerformanceMonitoring = true;
        public int maxConcurrentInferences = 2;
        public TrustformersEngine.QuantizationMode quantizationMode = TrustformersEngine.QuantizationMode.INT8;
        public TrustformersEngine.MemoryOptimizationLevel memoryOptimization = TrustformersEngine.MemoryOptimizationLevel.Balanced;

        private TrustformersEngine engine;
        private TrustformersPerformanceOptimizer performanceOptimizer;
        private List<string> availableModels = new List<string>();
        private string selectedModelId;
        private bool isInitialized = false;
        private bool isInferenceRunning = false;

        // Performance tracking
        private float lastInferenceTime = 0f;
        private float peakMemoryUsage = 0f;
        private float averageCpuUsage = 0f;
        private string lastBatteryImpact = "Unknown";

        void Start()
        {
            InitializeUI();
            DisplayDeviceInfo();
        }

        void InitializeUI()
        {
            // Setup button callbacks
            if (initializeButton != null)
            {
                initializeButton.onClick.AddListener(InitializeTrustformers);
            }

            if (runInferenceButton != null)
            {
                runInferenceButton.onClick.AddListener(RunInference);
                runInferenceButton.interactable = false;
            }

            // Setup dropdown callback
            if (modelDropdown != null)
            {
                modelDropdown.onValueChanged.AddListener(OnModelSelectionChanged);
                modelDropdown.interactable = false;
            }

            // Setup toggle callbacks
            if (enableGpuToggle != null)
            {
                enableGpuToggle.onValueChanged.AddListener(OnGpuToggleChanged);
            }

            if (enableQuantizationToggle != null)
            {
                enableQuantizationToggle.onValueChanged.AddListener(OnQuantizationToggleChanged);
            }

            // Initialize UI state
            UpdateStatusText("Ready to initialize TrustformeRS");
            UpdateProgressSlider(0f);
        }

        void DisplayDeviceInfo()
        {
            if (deviceInfoText == null) return;

            var deviceInfo = TrustformersEngine.GetDeviceInfo();
            string infoText = $"Device: {deviceInfo.deviceModel}\n" +
                            $"Platform: {Application.platform}\n" +
                            $"CPU: {deviceInfo.cpuModel} ({deviceInfo.cpuCoreCount} cores)\n" +
                            $"Memory: {deviceInfo.totalMemoryMB} MB\n" +
                            $"GPU: {(deviceInfo.hasGpu ? deviceInfo.gpuModel : "Not Available")}\n" +
                            $"Unity Version: {Application.unityVersion}";

            deviceInfoText.text = infoText;
            
            Debug.Log($"[TrustformeRS] Device Info: {infoText}");
        }

        public void InitializeTrustformers()
        {
            if (isInitialized)
            {
                Debug.LogWarning("[TrustformeRS] Already initialized");
                return;
            }

            StartCoroutine(InitializeTrustformersCoroutine());
        }

        IEnumerator InitializeTrustformersCoroutine()
        {
            UpdateStatusText("Initializing TrustformeRS...");
            UpdateProgressSlider(0.1f);

            try
            {
                // Create configuration based on UI settings
                var config = new TrustformersEngine.Config
                {
                    enablePerformanceMonitoring = enablePerformanceMonitoring,
                    enableDebugLogging = enableDebugLogging,
                    maxConcurrentInferences = maxConcurrentInferences,
                    enableGpuAcceleration = enableGpuToggle?.isOn ?? true,
                    enableQuantization = enableQuantizationToggle?.isOn ?? true,
                    quantizationMode = quantizationMode,
                    memoryOptimizationLevel = memoryOptimization,
                    enableBatteryOptimization = Application.isMobilePlatform,
                    enableThermalThrottling = Application.isMobilePlatform
                };

                UpdateProgressSlider(0.3f);

                // Initialize engine
                engine = new TrustformersEngine();
                bool initSuccess = engine.Initialize(config);

                if (!initSuccess)
                {
                    throw new Exception("Failed to initialize TrustformeRS engine");
                }

                UpdateProgressSlider(0.6f);
                yield return new WaitForSeconds(0.1f);

                // Initialize performance optimizer
                performanceOptimizer = new TrustformersPerformanceOptimizer(engine);
                performanceOptimizer.OptimizeForUnity();

                UpdateProgressSlider(0.8f);
                yield return new WaitForSeconds(0.1f);

                // Load available models
                availableModels = engine.GetAvailableModels();
                PopulateModelDropdown();

                UpdateProgressSlider(1.0f);

                isInitialized = true;
                
                // Update UI state
                runInferenceButton.interactable = availableModels.Count > 0;
                modelDropdown.interactable = availableModels.Count > 0;
                initializeButton.interactable = false;

                UpdateStatusText($"Initialized successfully! Found {availableModels.Count} models.");
                
                Debug.Log($"[TrustformeRS] Initialization complete. Available models: {availableModels.Count}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[TrustformeRS] Initialization failed: {ex.Message}");
                UpdateStatusText($"Initialization failed: {ex.Message}");
                UpdateProgressSlider(0f);
            }
        }

        void PopulateModelDropdown()
        {
            if (modelDropdown == null) return;

            modelDropdown.ClearOptions();
            modelDropdown.AddOptions(availableModels);

            if (availableModels.Count > 0)
            {
                selectedModelId = availableModels[0];
                modelDropdown.value = 0;
            }
        }

        public void RunInference()
        {
            if (!isInitialized || isInferenceRunning || string.IsNullOrEmpty(selectedModelId))
            {
                Debug.LogWarning("[TrustformeRS] Cannot run inference - not ready");
                return;
            }

            StartCoroutine(RunInferenceCoroutine());
        }

        IEnumerator RunInferenceCoroutine()
        {
            isInferenceRunning = true;
            runInferenceButton.interactable = false;
            
            UpdateStatusText("Running inference...");
            UpdateProgressSlider(0f);

            var startTime = Time.realtimeSinceStartup;

            try
            {
                // Start performance monitoring
                if (performanceOptimizer != null)
                {
                    performanceOptimizer.StartMonitoring();
                }

                UpdateProgressSlider(0.1f);

                // Prepare input data
                float[] inputData = PrepareInputData();
                int[] inputShape = { 1, inputData.Length };

                UpdateProgressSlider(0.3f);

                // Configure inference parameters
                var inferenceConfig = new TrustformersEngine.InferenceConfig
                {
                    maxTokens = 100,
                    temperature = 0.7f,
                    topP = 0.9f,
                    enableBatching = true,
                    useGpu = enableGpuToggle?.isOn ?? true
                };

                UpdateProgressSlider(0.5f);

                // Run inference
                var result = engine.RunInference(selectedModelId, inputData, inputShape, inferenceConfig);

                UpdateProgressSlider(0.8f);

                // Stop performance monitoring and get metrics
                TrustformersEngine.PerformanceMetrics metrics = null;
                if (performanceOptimizer != null)
                {
                    metrics = performanceOptimizer.StopMonitoring();
                }

                UpdateProgressSlider(1.0f);

                // Calculate inference time
                lastInferenceTime = (Time.realtimeSinceStartup - startTime) * 1000f; // Convert to ms

                // Update performance stats
                if (metrics != null)
                {
                    peakMemoryUsage = metrics.peakMemoryUsageMB;
                    averageCpuUsage = metrics.averageCpuUsage;
                    lastBatteryImpact = metrics.batteryImpactLevel;
                }

                // Display results
                DisplayInferenceResults(result, metrics);
                UpdateStatusText($"Inference completed in {lastInferenceTime:F1}ms");

                Debug.Log($"[TrustformeRS] Inference completed successfully in {lastInferenceTime:F1}ms");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[TrustformeRS] Inference failed: {ex.Message}");
                UpdateStatusText($"Inference failed: {ex.Message}");
                UpdateProgressSlider(0f);
            }
            finally
            {
                isInferenceRunning = false;
                runInferenceButton.interactable = true;
            }
        }

        float[] PrepareInputData()
        {
            // If custom input is provided, try to parse it
            if (customInputField != null && !string.IsNullOrEmpty(customInputField.text))
            {
                try
                {
                    string[] values = customInputField.text.Split(',');
                    float[] customData = new float[values.Length];
                    for (int i = 0; i < values.Length; i++)
                    {
                        customData[i] = float.Parse(values[i].Trim());
                    }
                    return customData;
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[TrustformeRS] Failed to parse custom input: {ex.Message}. Using default data.");
                }
            }

            // Generate sample input data (replace with real data in production)
            const int inputSize = 768;
            float[] inputData = new float[inputSize];
            
            for (int i = 0; i < inputSize; i++)
            {
                // Generate a simple pattern for testing
                inputData[i] = Mathf.Sin(i * 0.1f) * 0.5f + 0.5f;
            }

            return inputData;
        }

        void DisplayInferenceResults(TrustformersEngine.InferenceResult result, TrustformersEngine.PerformanceMetrics metrics)
        {
            // Update results text
            if (resultsText != null)
            {
                string resultText = $"Output Shape: [{string.Join(", ", result.outputShape)}]\n" +
                                  $"Output Size: {result.outputData.Length}\n" +
                                  $"Sample Output: [{string.Join(", ", System.Array.ConvertAll(System.Array.ConvertAll(result.outputData, 0, Math.Min(5, result.outputData.Length)), x => x.ToString("F3")))}...]";
                
                if (result.outputData.Length > 5)
                {
                    resultText += $" (showing first 5 of {result.outputData.Length})";
                }

                resultsText.text = resultText;
            }

            // Update performance text
            if (performanceText != null && metrics != null)
            {
                string perfText = $"Inference Time: {lastInferenceTime:F1}ms\n" +
                                $"Memory Usage: {metrics.peakMemoryUsageMB:F1}MB\n" +
                                $"CPU Usage: {metrics.averageCpuUsage:F1}%\n" +
                                $"GPU Usage: {metrics.averageGpuUsage:F1}%\n" +
                                $"Battery Impact: {metrics.batteryImpactLevel}\n" +
                                $"Thermal State: {metrics.thermalState}";

                performanceText.text = perfText;
            }
        }

        void OnModelSelectionChanged(int index)
        {
            if (index >= 0 && index < availableModels.Count)
            {
                selectedModelId = availableModels[index];
                Debug.Log($"[TrustformeRS] Selected model: {selectedModelId}");
            }
        }

        void OnGpuToggleChanged(bool enabled)
        {
            Debug.Log($"[TrustformeRS] GPU acceleration: {(enabled ? "Enabled" : "Disabled")}");
        }

        void OnQuantizationToggleChanged(bool enabled)
        {
            Debug.Log($"[TrustformeRS] Quantization: {(enabled ? "Enabled" : "Disabled")}");
        }

        void UpdateStatusText(string status)
        {
            if (statusText != null)
            {
                statusText.text = $"Status: {status}";
            }
        }

        void UpdateProgressSlider(float progress)
        {
            if (progressSlider != null)
            {
                progressSlider.value = progress;
            }
        }

        // Public methods for external control (e.g., from other scripts or UI)
        public bool IsInitialized => isInitialized;
        public bool IsInferenceRunning => isInferenceRunning;
        public List<string> AvailableModels => new List<string>(availableModels);
        public string SelectedModelId => selectedModelId;
        public float LastInferenceTime => lastInferenceTime;

        // Benchmark methods
        public void RunBenchmark(int iterations = 10)
        {
            if (!isInitialized)
            {
                Debug.LogWarning("[TrustformeRS] Cannot run benchmark - not initialized");
                return;
            }

            StartCoroutine(RunBenchmarkCoroutine(iterations));
        }

        IEnumerator RunBenchmarkCoroutine(int iterations)
        {
            Debug.Log($"[TrustformeRS] Starting benchmark with {iterations} iterations");
            
            List<float> inferenceTimes = new List<float>();
            
            for (int i = 0; i < iterations; i++)
            {
                UpdateStatusText($"Benchmark iteration {i + 1}/{iterations}");
                
                yield return StartCoroutine(RunInferenceCoroutine());
                inferenceTimes.Add(lastInferenceTime);
                
                yield return new WaitForSeconds(0.1f); // Small delay between iterations
            }

            // Calculate benchmark statistics
            float avgTime = 0f;
            float minTime = float.MaxValue;
            float maxTime = float.MinValue;

            foreach (float time in inferenceTimes)
            {
                avgTime += time;
                minTime = Mathf.Min(minTime, time);
                maxTime = Mathf.Max(maxTime, time);
            }
            avgTime /= iterations;

            string benchmarkResults = $"Benchmark Results ({iterations} iterations):\n" +
                                    $"Average: {avgTime:F1}ms\n" +
                                    $"Min: {minTime:F1}ms\n" +
                                    $"Max: {maxTime:F1}ms\n" +
                                    $"Std Dev: {CalculateStandardDeviation(inferenceTimes, avgTime):F1}ms";

            Debug.Log($"[TrustformeRS] {benchmarkResults}");
            UpdateStatusText("Benchmark completed");
            
            if (performanceText != null)
            {
                performanceText.text = benchmarkResults;
            }
        }

        float CalculateStandardDeviation(List<float> values, float mean)
        {
            float sumSquaredDifferences = 0f;
            foreach (float value in values)
            {
                float difference = value - mean;
                sumSquaredDifferences += difference * difference;
            }
            return Mathf.Sqrt(sumSquaredDifferences / values.Count);
        }

        void OnDestroy()
        {
            // Cleanup resources
            try
            {
                performanceOptimizer?.Dispose();
                engine?.Dispose();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[TrustformeRS] Error during cleanup: {ex.Message}");
            }
        }

        void OnApplicationPause(bool pauseStatus)
        {
            if (pauseStatus && isInitialized)
            {
                // Pause inference and free resources when app is paused
                engine?.PauseInference();
                Debug.Log("[TrustformeRS] Inference paused due to application pause");
            }
            else if (!pauseStatus && isInitialized)
            {
                // Resume inference when app is resumed
                engine?.ResumeInference();
                Debug.Log("[TrustformeRS] Inference resumed");
            }
        }

        void OnApplicationFocus(bool hasFocus)
        {
            if (!hasFocus && isInitialized)
            {
                // Reduce performance when app loses focus
                performanceOptimizer?.SetPerformanceMode(TrustformersPerformanceOptimizer.PerformanceMode.PowerSaving);
            }
            else if (hasFocus && isInitialized)
            {
                // Restore performance when app gains focus
                performanceOptimizer?.SetPerformanceMode(TrustformersPerformanceOptimizer.PerformanceMode.Balanced);
            }
        }
    }
}