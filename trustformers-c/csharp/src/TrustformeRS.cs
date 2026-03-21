using System;
using System.Runtime.InteropServices;
using System.Text;

namespace TrustformeRS
{
    /// <summary>
    /// Main TrustformeRS C# API wrapper providing P/Invoke bindings for the native TrustformeRS library
    /// </summary>
    public static class TrustformeRS
    {
        // Dynamic library loading based on platform
        private const string LibraryName = "trustformers_c";
        
        #if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN || NET461 || NET48 || NETCOREAPP3_1_OR_GREATER
        private const string WindowsLibrary = "trustformers_c.dll";
        #endif
        
        #if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
        private const string MacOSLibrary = "libtrusformers_c.dylib";
        #endif
        
        #if UNITY_STANDALONE_LINUX || UNITY_EDITOR_LINUX
        private const string LinuxLibrary = "libtrusformers_c.so";
        #endif

        // Core API functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_init();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void trustformers_shutdown();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_get_version(out IntPtr version);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void trustformers_free_string(IntPtr str);

        // Model loading functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_load_model(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string model_path,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string config_json,
            out IntPtr model_handle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_unload_model(IntPtr model_handle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_get_model_info(
            IntPtr model_handle,
            out IntPtr info_json);

        // Tokenizer functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_create_tokenizer(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string tokenizer_path,
            out IntPtr tokenizer_handle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_tokenizer_encode(
            IntPtr tokenizer_handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            out IntPtr token_ids,
            out int num_tokens);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_tokenizer_decode(
            IntPtr tokenizer_handle,
            IntPtr token_ids,
            int num_tokens,
            out IntPtr decoded_text);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_destroy_tokenizer(IntPtr tokenizer_handle);

        // Pipeline functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_create_pipeline(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string pipeline_type,
            IntPtr model_handle,
            IntPtr tokenizer_handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string config_json,
            out IntPtr pipeline_handle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_pipeline_text_generation(
            IntPtr pipeline_handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string generation_config,
            out IntPtr result_json);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_pipeline_text_classification(
            IntPtr pipeline_handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            out IntPtr result_json);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_destroy_pipeline(IntPtr pipeline_handle);

        // Memory management functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_get_memory_usage(out IntPtr usage_json);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_cleanup_memory();

        // Performance monitoring functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_get_performance_metrics(out IntPtr metrics_json);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_reset_performance_metrics();

        // CUDA functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_cuda_init();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int trustformers_cuda_get_device_count();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_cuda_set_device(int device_id);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_cuda_get_device_info(
            int device_id,
            out IntPtr info_json);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int trustformers_cuda_is_available();

        // ROCm functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_rocm_init();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int trustformers_rocm_get_device_count();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_rocm_set_device(int device_id);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_rocm_get_device_info(
            int device_id,
            out IntPtr info_json);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int trustformers_rocm_is_available();

        // HTTP Server functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_http_server_create(out IntPtr server_id);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_http_server_create_with_config(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string config_json,
            out IntPtr server_id);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_http_server_start(IntPtr server_id);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_http_server_stop(IntPtr server_id);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_http_server_destroy(IntPtr server_id);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_http_server_add_model(
            IntPtr server_id,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string endpoint_json);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TrustformersError trustformers_http_server_get_metrics(
            IntPtr server_id,
            out IntPtr metrics_json);

        // Utility functions
        /// <summary>
        /// Converts a C string pointer to a C# string and frees the C string
        /// </summary>
        /// <param name="ptr">Pointer to C string</param>
        /// <returns>C# string</returns>
        public static string PtrToStringAndFree(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                return null;

            try
            {
                string result = Marshal.PtrToStringUTF8(ptr);
                trustformers_free_string(ptr);
                return result;
            }
            catch
            {
                trustformers_free_string(ptr);
                throw;
            }
        }

        /// <summary>
        /// Converts a C string pointer to a C# string without freeing the C string
        /// </summary>
        /// <param name="ptr">Pointer to C string</param>
        /// <returns>C# string</returns>
        public static string PtrToString(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                return null;

            return Marshal.PtrToStringUTF8(ptr);
        }

        /// <summary>
        /// Checks if TrustformersError indicates success
        /// </summary>
        /// <param name="error">Error code to check</param>
        /// <returns>True if success, false otherwise</returns>
        public static bool IsSuccess(TrustformersError error)
        {
            return error == TrustformersError.Success;
        }

        /// <summary>
        /// Throws TrustformersException if error is not Success
        /// </summary>
        /// <param name="error">Error code to check</param>
        /// <param name="message">Optional error message</param>
        public static void ThrowIfError(TrustformersError error, string message = null)
        {
            if (error != TrustformersError.Success)
            {
                throw new TrustformersException(error, message);
            }
        }

        /// <summary>
        /// Initialize TrustformersRS native library
        /// </summary>
        public static void Initialize()
        {
            var error = trustformers_init();
            ThrowIfError(error, "Failed to initialize TrustformersRS");
        }

        /// <summary>
        /// Shutdown TrustformersRS native library
        /// </summary>
        public static void Shutdown()
        {
            trustformers_shutdown();
        }

        /// <summary>
        /// Get TrustformersRS version string
        /// </summary>
        /// <returns>Version string</returns>
        public static string GetVersion()
        {
            var error = trustformers_get_version(out IntPtr versionPtr);
            ThrowIfError(error, "Failed to get version");
            return PtrToStringAndFree(versionPtr);
        }

        /// <summary>
        /// Get memory usage information as JSON string
        /// </summary>
        /// <returns>Memory usage JSON</returns>
        public static string GetMemoryUsage()
        {
            var error = trustformers_get_memory_usage(out IntPtr usagePtr);
            ThrowIfError(error, "Failed to get memory usage");
            return PtrToStringAndFree(usagePtr);
        }

        /// <summary>
        /// Cleanup unused memory
        /// </summary>
        public static void CleanupMemory()
        {
            var error = trustformers_cleanup_memory();
            ThrowIfError(error, "Failed to cleanup memory");
        }

        /// <summary>
        /// Get performance metrics as JSON string
        /// </summary>
        /// <returns>Performance metrics JSON</returns>
        public static string GetPerformanceMetrics()
        {
            var error = trustformers_get_performance_metrics(out IntPtr metricsPtr);
            ThrowIfError(error, "Failed to get performance metrics");
            return PtrToStringAndFree(metricsPtr);
        }

        /// <summary>
        /// Reset performance metrics
        /// </summary>
        public static void ResetPerformanceMetrics()
        {
            var error = trustformers_reset_performance_metrics();
            ThrowIfError(error, "Failed to reset performance metrics");
        }

        /// <summary>
        /// Check if CUDA is available
        /// </summary>
        /// <returns>True if CUDA is available</returns>
        public static bool IsCudaAvailable()
        {
            return trustformers_cuda_is_available() != 0;
        }

        /// <summary>
        /// Check if ROCm is available
        /// </summary>
        /// <returns>True if ROCm is available</returns>
        public static bool IsRocmAvailable()
        {
            return trustformers_rocm_is_available() != 0;
        }
    }
}