using System;
using System.Runtime.InteropServices;
using System.Text.Json;
using TrustformersMobile.Native;

namespace TrustformersMobile
{
    /// <summary>
    /// Main class for TrustformeRS Mobile library operations
    /// </summary>
    public static class TrustformersMobile
    {
        private static bool _initialized = false;
        private static readonly object _initLock = new object();

        /// <summary>
        /// Initializes the TrustformeRS Mobile library
        /// </summary>
        /// <exception cref="TrustformersMobileException">Thrown if initialization fails</exception>
        public static void Initialize()
        {
            if (_initialized)
                return;

            lock (_initLock)
            {
                if (_initialized)
                    return;

                var error = TrustformersMobileNative.trustformers_mobile_init();
                if (error != TrustformersMobileError.Success)
                    throw new TrustformersMobileException(error);

                _initialized = true;
            }
        }

        /// <summary>
        /// Gets the version of the native library
        /// </summary>
        /// <returns>Version string</returns>
        public static string GetVersion()
        {
            var versionPtr = TrustformersMobileNative.trustformers_mobile_version();
            return Marshal.PtrToStringAnsi(versionPtr) ?? "Unknown";
        }

        /// <summary>
        /// Gets device information as a JSON string
        /// </summary>
        /// <returns>Device information in JSON format</returns>
        /// <exception cref="TrustformersMobileException">Thrown if device info retrieval fails</exception>
        public static string GetDeviceInfoJson()
        {
            Initialize();

            var error = TrustformersMobileNative.trustformers_mobile_get_device_info(out IntPtr deviceInfoPtr);
            if (error != TrustformersMobileError.Success)
                throw new TrustformersMobileException(error);

            try
            {
                var json = Marshal.PtrToStringAnsi(deviceInfoPtr);
                return json ?? "{}";
            }
            finally
            {
                if (deviceInfoPtr != IntPtr.Zero)
                    TrustformersMobileNative.trustformers_mobile_free_string(deviceInfoPtr);
            }
        }

        /// <summary>
        /// Gets device information as a parsed object
        /// </summary>
        /// <returns>Device information object</returns>
        /// <exception cref="TrustformersMobileException">Thrown if device info retrieval fails</exception>
        public static DeviceInfo GetDeviceInfo()
        {
            var json = GetDeviceInfoJson();
            return JsonSerializer.Deserialize<DeviceInfo>(json) ?? new DeviceInfo();
        }

        /// <summary>
        /// Checks if a platform is supported
        /// </summary>
        /// <param name="platform">Platform to check</param>
        /// <returns>True if supported, false otherwise</returns>
        public static bool IsPlatformSupported(MobilePlatform platform)
        {
            return MobileConfig.IsPlatformSupported(platform);
        }

        /// <summary>
        /// Checks if a backend is supported
        /// </summary>
        /// <param name="backend">Backend to check</param>
        /// <returns>True if supported, false otherwise</returns>
        public static bool IsBackendSupported(MobileBackend backend)
        {
            return MobileConfig.IsBackendSupported(backend);
        }

        /// <summary>
        /// Creates a quick inference session with default configuration
        /// </summary>
        /// <param name="modelPath">Path to the model file</param>
        /// <returns>Configured inference engine</returns>
        /// <exception cref="TrustformersMobileException">Thrown if engine creation fails</exception>
        public static MobileInferenceEngine CreateQuickEngine(string modelPath)
        {
            Initialize();
            
            using var config = new MobileConfig();
            return new MobileInferenceEngine(config, modelPath);
        }

        /// <summary>
        /// Creates an inference session optimized for the current platform
        /// </summary>
        /// <param name="modelPath">Path to the model file</param>
        /// <returns>Platform-optimized inference engine</returns>
        /// <exception cref="TrustformersMobileException">Thrown if engine creation fails</exception>
        public static MobileInferenceEngine CreateOptimizedEngine(string modelPath)
        {
            Initialize();

            MobileConfig config;
            
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                // Assume iOS if we're on macOS in mobile context
                config = MobileConfig.CreateiOSOptimized();
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                // Assume Android if we're on Linux in mobile context
                config = MobileConfig.CreateAndroidOptimized();
            }
            else
            {
                // Default configuration for other platforms
                config = new MobileConfig();
            }

            return new MobileInferenceEngine(config, modelPath);
        }
    }

    /// <summary>
    /// Device information structure
    /// </summary>
    public class DeviceInfo
    {
        /// <summary>
        /// Device name
        /// </summary>
        public string Name { get; set; } = "";

        /// <summary>
        /// Operating system
        /// </summary>
        public string OS { get; set; } = "";

        /// <summary>
        /// CPU architecture
        /// </summary>
        public string Architecture { get; set; } = "";

        /// <summary>
        /// Number of CPU cores
        /// </summary>
        public int CpuCores { get; set; }

        /// <summary>
        /// Total memory in MB
        /// </summary>
        public long TotalMemoryMB { get; set; }

        /// <summary>
        /// Available memory in MB
        /// </summary>
        public long AvailableMemoryMB { get; set; }

        /// <summary>
        /// GPU information
        /// </summary>
        public string GPU { get; set; } = "";

        /// <summary>
        /// Whether the device supports hardware acceleration
        /// </summary>
        public bool SupportsHardwareAcceleration { get; set; }

        /// <summary>
        /// Performance tier (Low, Medium, High)
        /// </summary>
        public string PerformanceTier { get; set; } = "";

        /// <summary>
        /// Recommended backends for this device
        /// </summary>
        public string[] RecommendedBackends { get; set; } = Array.Empty<string>();
    }
}