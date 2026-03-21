using System;
using System.Runtime.InteropServices;

namespace TrustformersMobile.Native
{
    /// <summary>
    /// Error codes returned by the native TrustformeRS Mobile library
    /// </summary>
    public enum TrustformersMobileError
    {
        Success = 0,
        InvalidParameter = 1,
        OutOfMemory = 2,
        ModelLoadError = 3,
        InferenceError = 4,
        ConfigurationError = 5,
        PlatformNotSupported = 6,
        RuntimeError = 7,
        NullPointer = 8,
        SerializationError = 9,
    }

    /// <summary>
    /// Mobile platform enumeration
    /// </summary>
    public enum MobilePlatform
    {
        iOS = 0,
        Android = 1,
        Generic = 2,
    }

    /// <summary>
    /// Mobile inference backend enumeration
    /// </summary>
    public enum MobileBackend
    {
        CPU = 0,
        CoreML = 1,
        NNAPI = 2,
        GPU = 3,
        Metal = 4,
        Vulkan = 5,
        OpenCL = 6,
        Custom = 7,
    }

    /// <summary>
    /// Memory optimization level enumeration
    /// </summary>
    public enum MemoryOptimization
    {
        Minimal = 0,
        Balanced = 1,
        Maximum = 2,
    }

    /// <summary>
    /// Native P/Invoke bindings for TrustformeRS Mobile library
    /// </summary>
    internal static class TrustformersMobileNative
    {
        private const string LibraryName = "trustformers_mobile";

        static TrustformersMobileNative()
        {
            // Attempt to resolve the native library
            NativeLibrary.SetDllImportResolver(typeof(TrustformersMobileNative).Assembly, ImportResolver);
        }

        private static IntPtr ImportResolver(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName == LibraryName)
            {
                // Try platform-specific names
                string[] libraryNames = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                    ? new[] { "trustformers_mobile.dll" }
                    : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                        ? new[] { "libtrusformers_mobile.dylib", "trustformers_mobile.dylib" }
                        : new[] { "libtrusformers_mobile.so", "trustformers_mobile.so" };

                foreach (var name in libraryNames)
                {
                    if (NativeLibrary.TryLoad(name, out IntPtr handle))
                        return handle;
                }
            }
            return IntPtr.Zero;
        }

        // Library initialization
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_init();

        // Configuration management
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_create_default(out UIntPtr configHandle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_create_ios_optimized(out UIntPtr configHandle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_create_android_optimized(out UIntPtr configHandle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_create_ultra_low_memory(out UIntPtr configHandle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_set_platform(UIntPtr configHandle, int platform);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_set_backend(UIntPtr configHandle, int backend);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_set_memory_optimization(UIntPtr configHandle, int optimization);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_set_max_memory_mb(UIntPtr configHandle, UIntPtr maxMemoryMb);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_set_use_fp16(UIntPtr configHandle, int useFp16);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_set_num_threads(UIntPtr configHandle, UIntPtr numThreads);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_validate(UIntPtr configHandle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_config_free(UIntPtr configHandle);

        // Engine management
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern TrustformersMobileError trustformers_mobile_engine_create(
            UIntPtr configHandle,
            string modelPath,
            out UIntPtr engineHandle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_engine_inference_f32(
            UIntPtr engineHandle,
            IntPtr inputData,
            UIntPtr inputSize,
            IntPtr outputData,
            UIntPtr outputSize,
            out UIntPtr actualOutputSize);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_engine_free(UIntPtr engineHandle);

        // Device information
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern TrustformersMobileError trustformers_mobile_get_device_info(out IntPtr deviceInfoJson);

        // Utility functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void trustformers_mobile_free_string(IntPtr ptr);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr trustformers_mobile_version();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int trustformers_mobile_is_platform_supported(int platform);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int trustformers_mobile_is_backend_supported(int backend);
    }
}