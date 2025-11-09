using System;
using TrustformersMobile.Native;

namespace TrustformersMobile
{
    /// <summary>
    /// Configuration class for TrustformeRS Mobile inference
    /// </summary>
    public class MobileConfig : IDisposable
    {
        private UIntPtr _handle;
        private bool _disposed = false;

        /// <summary>
        /// Internal handle to the native configuration
        /// </summary>
        internal UIntPtr Handle => _handle;

        /// <summary>
        /// Creates a new configuration with default settings
        /// </summary>
        public MobileConfig()
        {
            var error = TrustformersMobileNative.trustformers_mobile_config_create_default(out _handle);
            ThrowIfError(error);
        }

        /// <summary>
        /// Creates a configuration optimized for iOS devices
        /// </summary>
        /// <returns>iOS-optimized configuration</returns>
        public static MobileConfig CreateiOSOptimized()
        {
            var config = new MobileConfig();
            config.Dispose(); // Free the default config
            var error = TrustformersMobileNative.trustformers_mobile_config_create_ios_optimized(out config._handle);
            ThrowIfError(error);
            config._disposed = false;
            return config;
        }

        /// <summary>
        /// Creates a configuration optimized for Android devices
        /// </summary>
        /// <returns>Android-optimized configuration</returns>
        public static MobileConfig CreateAndroidOptimized()
        {
            var config = new MobileConfig();
            config.Dispose(); // Free the default config
            var error = TrustformersMobileNative.trustformers_mobile_config_create_android_optimized(out config._handle);
            ThrowIfError(error);
            config._disposed = false;
            return config;
        }

        /// <summary>
        /// Creates a configuration for ultra-low memory usage
        /// </summary>
        /// <returns>Ultra-low memory configuration</returns>
        public static MobileConfig CreateUltraLowMemory()
        {
            var config = new MobileConfig();
            config.Dispose(); // Free the default config
            var error = TrustformersMobileNative.trustformers_mobile_config_create_ultra_low_memory(out config._handle);
            ThrowIfError(error);
            config._disposed = false;
            return config;
        }

        /// <summary>
        /// Sets the target mobile platform
        /// </summary>
        /// <param name="platform">Target platform</param>
        public void SetPlatform(MobilePlatform platform)
        {
            ThrowIfDisposed();
            var error = TrustformersMobileNative.trustformers_mobile_config_set_platform(_handle, (int)platform);
            ThrowIfError(error);
        }

        /// <summary>
        /// Sets the inference backend
        /// </summary>
        /// <param name="backend">Inference backend</param>
        public void SetBackend(MobileBackend backend)
        {
            ThrowIfDisposed();
            var error = TrustformersMobileNative.trustformers_mobile_config_set_backend(_handle, (int)backend);
            ThrowIfError(error);
        }

        /// <summary>
        /// Sets the memory optimization level
        /// </summary>
        /// <param name="optimization">Memory optimization level</param>
        public void SetMemoryOptimization(MemoryOptimization optimization)
        {
            ThrowIfDisposed();
            var error = TrustformersMobileNative.trustformers_mobile_config_set_memory_optimization(_handle, (int)optimization);
            ThrowIfError(error);
        }

        /// <summary>
        /// Sets the maximum memory usage in megabytes
        /// </summary>
        /// <param name="maxMemoryMb">Maximum memory usage in MB</param>
        public void SetMaxMemoryMb(ulong maxMemoryMb)
        {
            ThrowIfDisposed();
            var error = TrustformersMobileNative.trustformers_mobile_config_set_max_memory_mb(_handle, (UIntPtr)maxMemoryMb);
            ThrowIfError(error);
        }

        /// <summary>
        /// Enables or disables FP16 precision
        /// </summary>
        /// <param name="useFp16">True to enable FP16, false to disable</param>
        public void SetUseFp16(bool useFp16)
        {
            ThrowIfDisposed();
            var error = TrustformersMobileNative.trustformers_mobile_config_set_use_fp16(_handle, useFp16 ? 1 : 0);
            ThrowIfError(error);
        }

        /// <summary>
        /// Sets the number of threads for inference (0 = auto-detect)
        /// </summary>
        /// <param name="numThreads">Number of threads</param>
        public void SetNumThreads(ulong numThreads)
        {
            ThrowIfDisposed();
            var error = TrustformersMobileNative.trustformers_mobile_config_set_num_threads(_handle, (UIntPtr)numThreads);
            ThrowIfError(error);
        }

        /// <summary>
        /// Validates the configuration
        /// </summary>
        /// <exception cref="TrustformersMobileException">Thrown if configuration is invalid</exception>
        public void Validate()
        {
            ThrowIfDisposed();
            var error = TrustformersMobileNative.trustformers_mobile_config_validate(_handle);
            ThrowIfError(error);
        }

        /// <summary>
        /// Checks if a platform is supported
        /// </summary>
        /// <param name="platform">Platform to check</param>
        /// <returns>True if supported, false otherwise</returns>
        public static bool IsPlatformSupported(MobilePlatform platform)
        {
            return TrustformersMobileNative.trustformers_mobile_is_platform_supported((int)platform) != 0;
        }

        /// <summary>
        /// Checks if a backend is supported
        /// </summary>
        /// <param name="backend">Backend to check</param>
        /// <returns>True if supported, false otherwise</returns>
        public static bool IsBackendSupported(MobileBackend backend)
        {
            return TrustformersMobileNative.trustformers_mobile_is_backend_supported((int)backend) != 0;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MobileConfig));
        }

        private static void ThrowIfError(TrustformersMobileError error)
        {
            if (error != TrustformersMobileError.Success)
                throw new TrustformersMobileException(error);
        }

        /// <summary>
        /// Disposes the configuration and frees native resources
        /// </summary>
        public void Dispose()
        {
            if (!_disposed && _handle != UIntPtr.Zero)
            {
                TrustformersMobileNative.trustformers_mobile_config_free(_handle);
                _handle = UIntPtr.Zero;
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Finalizer
        /// </summary>
        ~MobileConfig()
        {
            Dispose();
        }
    }
}