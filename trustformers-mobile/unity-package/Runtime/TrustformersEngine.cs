using System;
using System.Runtime.InteropServices;
using UnityEngine;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TrustformersMobile
{
    /// <summary>
    /// Main interface for TrustformeRS mobile inference in Unity
    /// </summary>
    public class TrustformersEngine : MonoBehaviour
    {
        #region Native Interop

#if UNITY_IOS && !UNITY_EDITOR
        const string NATIVE_LIB = "__Internal";
#elif UNITY_ANDROID && !UNITY_EDITOR
        const string NATIVE_LIB = "trustformers_mobile";
#else
        const string NATIVE_LIB = "trustformers_mobile";
#endif

        [DllImport(NATIVE_LIB)]
        private static extern IntPtr trustformers_create_engine(string config_json);

        [DllImport(NATIVE_LIB)]
        private static extern void trustformers_destroy_engine(IntPtr engine);

        [DllImport(NATIVE_LIB)]
        private static extern int trustformers_load_model(IntPtr engine, string model_path);

        [DllImport(NATIVE_LIB)]
        private static extern int trustformers_inference(IntPtr engine, float[] input_data, int input_length, 
            float[] output_data, int output_length);

        [DllImport(NATIVE_LIB)]
        private static extern int trustformers_batch_inference(IntPtr engine, float[] input_data, int batch_size, 
            int input_length, float[] output_data, int output_length);

        [DllImport(NATIVE_LIB)]
        private static extern void trustformers_get_stats(IntPtr engine, out EngineStats stats);

        [DllImport(NATIVE_LIB)]
        private static extern int trustformers_set_performance_mode(IntPtr engine, int mode);

        [DllImport(NATIVE_LIB)]
        private static extern IntPtr trustformers_get_device_info();

        [DllImport(NATIVE_LIB)]
        private static extern void trustformers_free_string(IntPtr str);

        [DllImport(NATIVE_LIB)]
        private static extern int trustformers_warm_up(IntPtr engine);

        #endregion

        #region Configuration Classes

        [Serializable]
        public class EngineConfig
        {
            public MobilePlatform platform = MobilePlatform.Auto;
            public MobileBackend backend = MobileBackend.Auto;
            public MemoryOptimization memoryOptimization = MemoryOptimization.Balanced;
            public int maxMemoryMB = 512;
            public bool useFP16 = true;
            public QuantizationConfig quantization = new QuantizationConfig();
            public int numThreads = 0; // Auto-detect
            public bool enableBatching = false;
            public int maxBatchSize = 1;
            public bool enableWarmup = true;
            public PerformanceConfig performance = new PerformanceConfig();
        }

        [Serializable]
        public class QuantizationConfig
        {
            public bool enabled = true;
            public QuantizationScheme scheme = QuantizationScheme.Int8;
            public bool dynamic = true;
            public bool perChannel = false;
        }

        [Serializable]
        public class PerformanceConfig
        {
            public bool adaptivePerformance = true;
            public float targetFPS = 60.0f;
            public bool thermalThrottling = true;
            public bool batteryOptimization = true;
            public int maxConcurrentInferences = 1;
        }

        [Serializable]
        public struct EngineStats
        {
            public int platform;
            public int backend;
            public int memoryUsageMB;
            public int peakMemoryMB;
            public float avgInferenceTimeMs;
            public int totalInferences;
            public int threadCount;
            public int quantizationEnabled;
            public int fp16Enabled;
        }

        #endregion

        #region Enums

        public enum MobilePlatform
        {
            Auto = 0,
            iOS = 1,
            Android = 2,
            Generic = 3
        }

        public enum MobileBackend
        {
            Auto = 0,
            CPU = 1,
            CoreML = 2,
            NNAPI = 3,
            GPU = 4,
            Custom = 5
        }

        public enum MemoryOptimization
        {
            Minimal = 0,
            Balanced = 1,
            Maximum = 2
        }

        public enum QuantizationScheme
        {
            Int8 = 0,
            Int4 = 1,
            FP16 = 2,
            Dynamic = 3
        }

        public enum PerformanceMode
        {
            PowerSaving = 0,
            Balanced = 1,
            HighPerformance = 2
        }

        #endregion

        #region Public Fields

        [Header("Engine Configuration")]
        public EngineConfig config = new EngineConfig();

        [Header("Model Settings")]
        public string modelPath = "";
        public bool loadModelOnStart = true;

        [Header("Performance")]
        public bool showDebugInfo = false;
        public PerformanceMode performanceMode = PerformanceMode.Balanced;

        #endregion

        #region Private Fields

        private IntPtr enginePtr = IntPtr.Zero;
        private bool isInitialized = false;
        private bool isModelLoaded = false;
        private readonly object lockObject = new object();
        private EngineStats currentStats;

        #endregion

        #region Events

        public event Action<EngineStats> OnStatsUpdated;
        public event Action<string> OnError;
        public event Action OnInitialized;
        public event Action OnModelLoaded;

        #endregion

        #region MonoBehaviour Lifecycle

        void Start()
        {
            InitializeEngine();
            if (loadModelOnStart && !string.IsNullOrEmpty(modelPath))
            {
                LoadModelAsync(modelPath);
            }
        }

        void Update()
        {
            if (isInitialized && showDebugInfo)
            {
                UpdateStats();
            }
        }

        void OnApplicationPause(bool pauseStatus)
        {
            if (pauseStatus)
            {
                SetPerformanceMode(PerformanceMode.PowerSaving);
            }
            else
            {
                SetPerformanceMode(performanceMode);
            }
        }

        void OnDestroy()
        {
            DestroyEngine();
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Initialize the TrustformeRS engine
        /// </summary>
        public bool InitializeEngine()
        {
            if (isInitialized) return true;

            try
            {
                // Auto-detect platform if needed
                if (config.platform == MobilePlatform.Auto)
                {
                    config.platform = DetectPlatform();
                }

                // Auto-detect backend if needed
                if (config.backend == MobileBackend.Auto)
                {
                    config.backend = DetectOptimalBackend();
                }

                string configJson = JsonUtility.ToJson(config);
                enginePtr = trustformers_create_engine(configJson);

                if (enginePtr == IntPtr.Zero)
                {
                    Debug.LogError("Failed to create TrustformeRS engine");
                    return false;
                }

                isInitialized = true;
                
                if (config.enableWarmup)
                {
                    WarmUp();
                }

                SetPerformanceMode(performanceMode);
                OnInitialized?.Invoke();

                Debug.Log($"TrustformeRS engine initialized successfully on {config.platform} with {config.backend} backend");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to initialize TrustformeRS engine: {e.Message}");
                OnError?.Invoke($"Initialization failed: {e.Message}");
                return false;
            }
        }

        /// <summary>
        /// Load a model from the given path
        /// </summary>
        public bool LoadModel(string path)
        {
            if (!isInitialized)
            {
                Debug.LogError("Engine not initialized");
                return false;
            }

            lock (lockObject)
            {
                try
                {
                    int result = trustformers_load_model(enginePtr, path);
                    if (result == 0)
                    {
                        isModelLoaded = true;
                        modelPath = path;
                        OnModelLoaded?.Invoke();
                        Debug.Log($"Model loaded successfully: {path}");
                        return true;
                    }
                    else
                    {
                        Debug.LogError($"Failed to load model: {path}, error code: {result}");
                        OnError?.Invoke($"Model loading failed: {result}");
                        return false;
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"Exception loading model: {e.Message}");
                    OnError?.Invoke($"Model loading exception: {e.Message}");
                    return false;
                }
            }
        }

        /// <summary>
        /// Load model asynchronously
        /// </summary>
        public async Task<bool> LoadModelAsync(string path)
        {
            return await Task.Run(() => LoadModel(path));
        }

        /// <summary>
        /// Perform inference with the loaded model
        /// </summary>
        public float[] Inference(float[] input)
        {
            if (!isModelLoaded)
            {
                Debug.LogError("No model loaded");
                return null;
            }

            lock (lockObject)
            {
                try
                {
                    // Estimate output size (this would normally come from model metadata)
                    int outputSize = input.Length; // Simplified assumption
                    float[] output = new float[outputSize];

                    int result = trustformers_inference(enginePtr, input, input.Length, output, output.Length);
                    
                    if (result == 0)
                    {
                        return output;
                    }
                    else
                    {
                        Debug.LogError($"Inference failed with error code: {result}");
                        OnError?.Invoke($"Inference failed: {result}");
                        return null;
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"Exception during inference: {e.Message}");
                    OnError?.Invoke($"Inference exception: {e.Message}");
                    return null;
                }
            }
        }

        /// <summary>
        /// Perform asynchronous inference
        /// </summary>
        public async Task<float[]> InferenceAsync(float[] input)
        {
            return await Task.Run(() => Inference(input));
        }

        /// <summary>
        /// Perform batch inference
        /// </summary>
        public float[][] BatchInference(float[][] inputs)
        {
            if (!isModelLoaded || !config.enableBatching)
            {
                // Fallback to sequential inference
                var results = new List<float[]>();
                foreach (var input in inputs)
                {
                    results.Add(Inference(input));
                }
                return results.ToArray();
            }

            lock (lockObject)
            {
                try
                {
                    int batchSize = inputs.Length;
                    int inputLength = inputs[0].Length;
                    int outputLength = inputLength; // Simplified

                    // Flatten input data
                    float[] flatInput = new float[batchSize * inputLength];
                    for (int i = 0; i < batchSize; i++)
                    {
                        Array.Copy(inputs[i], 0, flatInput, i * inputLength, inputLength);
                    }

                    float[] flatOutput = new float[batchSize * outputLength];
                    
                    int result = trustformers_batch_inference(enginePtr, flatInput, batchSize, 
                        inputLength, flatOutput, outputLength);

                    if (result == 0)
                    {
                        // Unflatten output data
                        float[][] outputs = new float[batchSize][];
                        for (int i = 0; i < batchSize; i++)
                        {
                            outputs[i] = new float[outputLength];
                            Array.Copy(flatOutput, i * outputLength, outputs[i], 0, outputLength);
                        }
                        return outputs;
                    }
                    else
                    {
                        Debug.LogError($"Batch inference failed with error code: {result}");
                        OnError?.Invoke($"Batch inference failed: {result}");
                        return null;
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"Exception during batch inference: {e.Message}");
                    OnError?.Invoke($"Batch inference exception: {e.Message}");
                    return null;
                }
            }
        }

        /// <summary>
        /// Set performance mode for the engine
        /// </summary>
        public void SetPerformanceMode(PerformanceMode mode)
        {
            if (!isInitialized) return;

            try
            {
                trustformers_set_performance_mode(enginePtr, (int)mode);
                performanceMode = mode;
                Debug.Log($"Performance mode set to: {mode}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to set performance mode: {e.Message}");
                OnError?.Invoke($"Performance mode change failed: {e.Message}");
            }
        }

        /// <summary>
        /// Get current engine statistics
        /// </summary>
        public EngineStats GetStats()
        {
            if (!isInitialized) return default;

            try
            {
                trustformers_get_stats(enginePtr, out currentStats);
                return currentStats;
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to get stats: {e.Message}");
                return default;
            }
        }

        /// <summary>
        /// Get device information
        /// </summary>
        public string GetDeviceInfo()
        {
            try
            {
                IntPtr deviceInfoPtr = trustformers_get_device_info();
                if (deviceInfoPtr != IntPtr.Zero)
                {
                    string deviceInfo = Marshal.PtrToStringAnsi(deviceInfoPtr);
                    trustformers_free_string(deviceInfoPtr);
                    return deviceInfo;
                }
                return "Device info not available";
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to get device info: {e.Message}");
                return $"Error: {e.Message}";
            }
        }

        /// <summary>
        /// Warm up the engine for optimal performance
        /// </summary>
        public void WarmUp()
        {
            if (!isInitialized) return;

            try
            {
                trustformers_warm_up(enginePtr);
                Debug.Log("Engine warmed up successfully");
            }
            catch (Exception e)
            {
                Debug.LogError($"Warmup failed: {e.Message}");
            }
        }

        #endregion

        #region Private Methods

        private void UpdateStats()
        {
            var stats = GetStats();
            OnStatsUpdated?.Invoke(stats);
        }

        private MobilePlatform DetectPlatform()
        {
#if UNITY_IOS
            return MobilePlatform.iOS;
#elif UNITY_ANDROID
            return MobilePlatform.Android;
#else
            return MobilePlatform.Generic;
#endif
        }

        private MobileBackend DetectOptimalBackend()
        {
#if UNITY_IOS
            return MobileBackend.CoreML;
#elif UNITY_ANDROID
            return MobileBackend.NNAPI;
#else
            return MobileBackend.CPU;
#endif
        }

        private void DestroyEngine()
        {
            if (enginePtr != IntPtr.Zero)
            {
                trustformers_destroy_engine(enginePtr);
                enginePtr = IntPtr.Zero;
                isInitialized = false;
                isModelLoaded = false;
                Debug.Log("TrustformeRS engine destroyed");
            }
        }

        #endregion

        #region Static Helper Methods

        /// <summary>
        /// Check if TrustformeRS is supported on the current platform
        /// </summary>
        public static bool IsSupported()
        {
#if UNITY_IOS || UNITY_ANDROID
            return true;
#else
            return false;
#endif
        }

        /// <summary>
        /// Get recommended configuration for the current platform
        /// </summary>
        public static EngineConfig GetRecommendedConfig()
        {
            var config = new EngineConfig();

#if UNITY_IOS
            config.platform = MobilePlatform.iOS;
            config.backend = MobileBackend.CoreML;
            config.maxMemoryMB = 1024;
            config.enableBatching = true;
            config.maxBatchSize = 4;
#elif UNITY_ANDROID
            config.platform = MobilePlatform.Android;
            config.backend = MobileBackend.NNAPI;
            config.maxMemoryMB = 768;
            config.enableBatching = false;
            config.maxBatchSize = 1;
#else
            config.platform = MobilePlatform.Generic;
            config.backend = MobileBackend.CPU;
            config.maxMemoryMB = 512;
#endif

            return config;
        }

        #endregion
    }
}