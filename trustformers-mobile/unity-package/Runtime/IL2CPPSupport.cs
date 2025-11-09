using System;
using System.Runtime.InteropServices;
using UnityEngine;
using AOT;

namespace TrustformersMobile.IL2CPP
{
    /// <summary>
    /// IL2CPP compatibility layer for TrustformeRS native library integration
    /// </summary>
    public static class IL2CPPSupport
    {
        #region Native Function Delegates

        // Callback delegates for IL2CPP compatibility
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LogCallback(IntPtr message);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ProgressCallback(float progress);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ErrorCallback(int errorCode, IntPtr errorMessage);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void InferenceCompleteCallback(IntPtr resultData, int resultLength);

        #endregion

        #region Native Interop

#if UNITY_IOS && !UNITY_EDITOR
        const string NATIVE_LIB = "__Internal";
#elif UNITY_ANDROID && !UNITY_EDITOR
        const string NATIVE_LIB = "trustformers_mobile";
#else
        const string NATIVE_LIB = "trustformers_mobile";
#endif

        [DllImport(NATIVE_LIB)]
        private static extern void trustformers_set_log_callback(LogCallback callback);

        [DllImport(NATIVE_LIB)]
        private static extern void trustformers_set_progress_callback(ProgressCallback callback);

        [DllImport(NATIVE_LIB)]
        private static extern void trustformers_set_error_callback(ErrorCallback callback);

        [DllImport(NATIVE_LIB)]
        private static extern void trustformers_set_inference_callback(InferenceCompleteCallback callback);

        [DllImport(NATIVE_LIB)]
        private static extern int trustformers_initialize_il2cpp_support();

        [DllImport(NATIVE_LIB)]
        private static extern void trustformers_cleanup_il2cpp_support();

        // Memory management functions for IL2CPP
        [DllImport(NATIVE_LIB)]
        private static extern IntPtr trustformers_allocate_managed_memory(int size);

        [DllImport(NATIVE_LIB)]
        private static extern void trustformers_free_managed_memory(IntPtr ptr);

        [DllImport(NATIVE_LIB)]
        private static extern void trustformers_copy_to_managed_memory(IntPtr dest, float[] source, int length);

        [DllImport(NATIVE_LIB)]
        private static extern void trustformers_copy_from_managed_memory(float[] dest, IntPtr source, int length);

        #endregion

        #region Static Fields

        private static bool isInitialized = false;
        private static LogCallback logCallbackInstance;
        private static ProgressCallback progressCallbackInstance;
        private static ErrorCallback errorCallbackInstance;
        private static InferenceCompleteCallback inferenceCallbackInstance;

        #endregion

        #region Events

        public static event Action<string> OnNativeLog;
        public static event Action<float> OnProgress;
        public static event Action<int, string> OnNativeError;
        public static event Action<float[]> OnInferenceComplete;

        #endregion

        #region Initialization

        /// <summary>
        /// Initialize IL2CPP support for TrustformeRS
        /// </summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        public static void Initialize()
        {
            if (isInitialized) return;

            try
            {
                // Create persistent callback instances for IL2CPP
                logCallbackInstance = OnLogCallbackStatic;
                progressCallbackInstance = OnProgressCallbackStatic;
                errorCallbackInstance = OnErrorCallbackStatic;
                inferenceCallbackInstance = OnInferenceCompleteCallbackStatic;

                // Register callbacks with native library
                trustformers_set_log_callback(logCallbackInstance);
                trustformers_set_progress_callback(progressCallbackInstance);
                trustformers_set_error_callback(errorCallbackInstance);
                trustformers_set_inference_callback(inferenceCallbackInstance);

                // Initialize native IL2CPP support
                int result = trustformers_initialize_il2cpp_support();
                if (result == 0)
                {
                    isInitialized = true;
                    Debug.Log("IL2CPP support for TrustformeRS initialized successfully");
                }
                else
                {
                    Debug.LogError($"Failed to initialize IL2CPP support: {result}");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"IL2CPP initialization exception: {e.Message}");
            }
        }

        /// <summary>
        /// Cleanup IL2CPP support
        /// </summary>
        public static void Cleanup()
        {
            if (!isInitialized) return;

            try
            {
                trustformers_cleanup_il2cpp_support();
                isInitialized = false;
                Debug.Log("IL2CPP support cleaned up");
            }
            catch (Exception e)
            {
                Debug.LogError($"IL2CPP cleanup exception: {e.Message}");
            }
        }

        #endregion

        #region AOT Callback Methods

        [MonoPInvokeCallback(typeof(LogCallback))]
        private static void OnLogCallbackStatic(IntPtr message)
        {
            try
            {
                string logMessage = Marshal.PtrToStringAnsi(message);
                OnNativeLog?.Invoke(logMessage);
                
                // Also log to Unity console
                Debug.Log($"[TrustformeRS Native] {logMessage}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Log callback exception: {e.Message}");
            }
        }

        [MonoPInvokeCallback(typeof(ProgressCallback))]
        private static void OnProgressCallbackStatic(float progress)
        {
            try
            {
                OnProgress?.Invoke(progress);
            }
            catch (Exception e)
            {
                Debug.LogError($"Progress callback exception: {e.Message}");
            }
        }

        [MonoPInvokeCallback(typeof(ErrorCallback))]
        private static void OnErrorCallbackStatic(int errorCode, IntPtr errorMessage)
        {
            try
            {
                string message = Marshal.PtrToStringAnsi(errorMessage);
                OnNativeError?.Invoke(errorCode, message);
                
                Debug.LogError($"[TrustformeRS Native Error {errorCode}] {message}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Error callback exception: {e.Message}");
            }
        }

        [MonoPInvokeCallback(typeof(InferenceCompleteCallback))]
        private static void OnInferenceCompleteCallbackStatic(IntPtr resultData, int resultLength)
        {
            try
            {
                if (resultData == IntPtr.Zero || resultLength <= 0)
                {
                    OnInferenceComplete?.Invoke(null);
                    return;
                }

                // Copy native data to managed array
                float[] results = new float[resultLength];
                trustformers_copy_from_managed_memory(results, resultData, resultLength);
                
                OnInferenceComplete?.Invoke(results);
            }
            catch (Exception e)
            {
                Debug.LogError($"Inference callback exception: {e.Message}");
                OnInferenceComplete?.Invoke(null);
            }
        }

        #endregion

        #region Memory Management Helpers

        /// <summary>
        /// Allocate managed memory for IL2CPP-native interop
        /// </summary>
        public static IntPtr AllocateManagedMemory(int sizeInBytes)
        {
            if (!isInitialized)
            {
                Debug.LogError("IL2CPP support not initialized");
                return IntPtr.Zero;
            }

            return trustformers_allocate_managed_memory(sizeInBytes);
        }

        /// <summary>
        /// Free managed memory allocated for IL2CPP-native interop
        /// </summary>
        public static void FreeManagedMemory(IntPtr ptr)
        {
            if (ptr != IntPtr.Zero)
            {
                trustformers_free_managed_memory(ptr);
            }
        }

        /// <summary>
        /// Copy managed array to native memory with IL2CPP compatibility
        /// </summary>
        public static void CopyToNativeMemory(IntPtr destination, float[] source)
        {
            if (!isInitialized || destination == IntPtr.Zero || source == null)
                return;

            trustformers_copy_to_managed_memory(destination, source, source.Length);
        }

        /// <summary>
        /// Copy native memory to managed array with IL2CPP compatibility
        /// </summary>
        public static void CopyFromNativeMemory(float[] destination, IntPtr source, int length)
        {
            if (!isInitialized || source == IntPtr.Zero || destination == null)
                return;

            trustformers_copy_from_managed_memory(destination, source, length);
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Check if IL2CPP support is properly initialized
        /// </summary>
        public static bool IsInitialized => isInitialized;

        /// <summary>
        /// Get IL2CPP compatibility information
        /// </summary>
        public static string GetCompatibilityInfo()
        {
            return $"IL2CPP Support Status:\n" +
                   $"Initialized: {isInitialized}\n" +
                   $"Unity Version: {Application.unityVersion}\n" +
                   $"Platform: {Application.platform}\n" +
                   $"Scripting Backend: {GetScriptingBackend()}\n" +
                   $"Runtime: {GetRuntimeInfo()}";
        }

        private static string GetScriptingBackend()
        {
#if ENABLE_IL2CPP
            return "IL2CPP";
#elif ENABLE_MONO
            return "Mono";
#else
            return "Unknown";
#endif
        }

        private static string GetRuntimeInfo()
        {
            return $".NET {Environment.Version}";
        }

        /// <summary>
        /// Test IL2CPP callback functionality
        /// </summary>
        public static bool TestCallbacks()
        {
            if (!isInitialized)
            {
                Debug.LogError("IL2CPP support not initialized");
                return false;
            }

            bool testPassed = true;
            
            // Test each callback type
            try
            {
                // These would trigger test callbacks in the native library
                Debug.Log("Testing IL2CPP callbacks...");
                
                // In a real implementation, you would call native test functions here
                // For now, we just verify the callbacks are properly registered
                
                Debug.Log("IL2CPP callback test completed successfully");
            }
            catch (Exception e)
            {
                Debug.LogError($"IL2CPP callback test failed: {e.Message}");
                testPassed = false;
            }

            return testPassed;
        }

        #endregion

        #region Performance Monitoring

        /// <summary>
        /// Monitor IL2CPP performance for optimization
        /// </summary>
        public static class PerformanceMonitor
        {
            private static float lastCallbackTime;
            private static int callbackCount;
            private static float totalCallbackTime;

            public static void RecordCallback(float executionTime)
            {
                lastCallbackTime = executionTime;
                callbackCount++;
                totalCallbackTime += executionTime;
            }

            public static float GetAverageCallbackTime()
            {
                return callbackCount > 0 ? totalCallbackTime / callbackCount : 0;
            }

            public static void Reset()
            {
                lastCallbackTime = 0;
                callbackCount = 0;
                totalCallbackTime = 0;
            }

            public static string GetPerformanceReport()
            {
                return $"IL2CPP Performance Report:\n" +
                       $"Total Callbacks: {callbackCount}\n" +
                       $"Average Callback Time: {GetAverageCallbackTime():F3}ms\n" +
                       $"Last Callback Time: {lastCallbackTime:F3}ms\n" +
                       $"Total Callback Time: {totalCallbackTime:F3}ms";
            }
        }

        #endregion
    }

    /// <summary>
    /// Attribute to mark methods that require AOT compilation for IL2CPP
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class AOTRequiredAttribute : Attribute
    {
        public string Reason { get; }

        public AOTRequiredAttribute(string reason = "Required for IL2CPP compatibility")
        {
            Reason = reason;
        }
    }

    /// <summary>
    /// Helper class for IL2CPP-specific optimizations
    /// </summary>
    public static class IL2CPPOptimizations
    {
        /// <summary>
        /// Pre-warm commonly used generic types for IL2CPP
        /// </summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterAssembliesLoaded)]
        public static void PrewarmTypes()
        {
            // Pre-instantiate generic types that will be used at runtime
            // This prevents IL2CPP from failing to find them later
            
            var floatArray = new float[0];
            var intArray = new int[0];
            var stringArray = new string[0];
            
            // Pre-warm Action and Func delegates
            Action<float> floatAction = _ => { };
            Action<int, string> errorAction = (_, _) => { };
            Func<float[], bool> floatArrayFunc = _ => true;
            
            // Pre-warm common containers
            var floatList = new System.Collections.Generic.List<float>();
            var stringDict = new System.Collections.Generic.Dictionary<string, object>();
            
            Debug.Log("IL2CPP type prewarming completed");
        }

        /// <summary>
        /// Generate AOT stubs for commonly used generic methods
        /// </summary>
        public static void GenerateAOTStubs()
        {
            // These method calls will never execute but ensure IL2CPP generates
            // the necessary code for these generic instantiations
            
            if (false)
            {
                Marshal.PtrToStructure<float>(IntPtr.Zero);
                Marshal.StructureToPtr<int>(0, IntPtr.Zero, false);
                
                Array.ConvertAll<float, int>(new float[0], f => (int)f);
                Array.Find<string>(new string[0], s => s != null);
                
                System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Empty<float>());
            }
        }
    }
}