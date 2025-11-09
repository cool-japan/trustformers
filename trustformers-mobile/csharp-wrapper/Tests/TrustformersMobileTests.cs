using System;
using Xunit;
using TrustformersMobile;

namespace TrustformersMobile.Tests
{
    public class TrustformersMobileTests
    {
        [Fact]
        public void Initialize_ShouldSucceed()
        {
            // Test library initialization
            TrustformersMobile.TrustformersMobile.Initialize();
            
            // Should not throw on subsequent calls
            TrustformersMobile.TrustformersMobile.Initialize();
        }

        [Fact]
        public void GetVersion_ShouldReturnValidVersion()
        {
            var version = TrustformersMobile.TrustformersMobile.GetVersion();
            
            Assert.NotNull(version);
            Assert.NotEmpty(version);
            Assert.NotEqual("Unknown", version);
        }

        [Fact]
        public void GetDeviceInfo_ShouldReturnValidInfo()
        {
            var deviceInfo = TrustformersMobile.TrustformersMobile.GetDeviceInfo();
            
            Assert.NotNull(deviceInfo);
            Assert.NotNull(deviceInfo.Name);
            Assert.NotNull(deviceInfo.OS);
            Assert.NotNull(deviceInfo.Architecture);
            Assert.True(deviceInfo.CpuCores > 0);
            Assert.True(deviceInfo.TotalMemoryMB > 0);
        }

        [Fact]
        public void GetDeviceInfoJson_ShouldReturnValidJson()
        {
            var json = TrustformersMobile.TrustformersMobile.GetDeviceInfoJson();
            
            Assert.NotNull(json);
            Assert.NotEmpty(json);
            Assert.StartsWith("{", json);
            Assert.EndsWith("}", json);
        }

        [Fact]
        public void MobileConfig_DefaultConstruction_ShouldSucceed()
        {
            using var config = new MobileConfig();
            
            // Should be able to validate default configuration
            config.Validate();
        }

        [Fact]
        public void MobileConfig_iOSOptimized_ShouldSucceed()
        {
            using var config = MobileConfig.CreateiOSOptimized();
            
            // Should be able to validate iOS configuration
            config.Validate();
        }

        [Fact]
        public void MobileConfig_AndroidOptimized_ShouldSucceed()
        {
            using var config = MobileConfig.CreateAndroidOptimized();
            
            // Should be able to validate Android configuration
            config.Validate();
        }

        [Fact]
        public void MobileConfig_UltraLowMemory_ShouldSucceed()
        {
            using var config = MobileConfig.CreateUltraLowMemory();
            
            // Should be able to validate ultra low memory configuration
            config.Validate();
        }

        [Fact]
        public void MobileConfig_SetPlatform_ShouldSucceed()
        {
            using var config = new MobileConfig();
            
            config.SetPlatform(MobilePlatform.iOS);
            config.SetPlatform(MobilePlatform.Android);
            config.SetPlatform(MobilePlatform.Generic);
            
            // Should still be valid after setting platform
            config.Validate();
        }

        [Fact]
        public void MobileConfig_SetBackend_ShouldSucceed()
        {
            using var config = new MobileConfig();
            
            config.SetBackend(MobileBackend.CPU);
            config.SetBackend(MobileBackend.GPU);
            
            // Should still be valid after setting backend
            config.Validate();
        }

        [Fact]
        public void MobileConfig_SetMemoryOptimization_ShouldSucceed()
        {
            using var config = new MobileConfig();
            
            config.SetMemoryOptimization(MemoryOptimization.Minimal);
            config.SetMemoryOptimization(MemoryOptimization.Balanced);
            config.SetMemoryOptimization(MemoryOptimization.Maximum);
            
            // Should still be valid after setting memory optimization
            config.Validate();
        }

        [Fact]
        public void MobileConfig_SetMaxMemoryMb_ShouldSucceed()
        {
            using var config = new MobileConfig();
            
            config.SetMaxMemoryMb(256);
            config.SetMaxMemoryMb(512);
            config.SetMaxMemoryMb(1024);
            
            // Should still be valid after setting memory limit
            config.Validate();
        }

        [Fact]
        public void MobileConfig_SetUseFp16_ShouldSucceed()
        {
            using var config = new MobileConfig();
            
            config.SetUseFp16(true);
            config.SetUseFp16(false);
            
            // Should still be valid after setting FP16
            config.Validate();
        }

        [Fact]
        public void MobileConfig_SetNumThreads_ShouldSucceed()
        {
            using var config = new MobileConfig();
            
            config.SetNumThreads(1);
            config.SetNumThreads(4);
            config.SetNumThreads(0); // Auto-detect
            
            // Should still be valid after setting thread count
            config.Validate();
        }

        [Theory]
        [InlineData(MobilePlatform.iOS)]
        [InlineData(MobilePlatform.Android)]
        [InlineData(MobilePlatform.Generic)]
        public void IsPlatformSupported_ShouldReturnBoolean(MobilePlatform platform)
        {
            var isSupported = TrustformersMobile.TrustformersMobile.IsPlatformSupported(platform);
            
            // Generic should always be supported
            if (platform == MobilePlatform.Generic)
            {
                Assert.True(isSupported);
            }
        }

        [Theory]
        [InlineData(MobileBackend.CPU)]
        [InlineData(MobileBackend.CoreML)]
        [InlineData(MobileBackend.NNAPI)]
        [InlineData(MobileBackend.GPU)]
        [InlineData(MobileBackend.Metal)]
        [InlineData(MobileBackend.Vulkan)]
        [InlineData(MobileBackend.OpenCL)]
        [InlineData(MobileBackend.Custom)]
        public void IsBackendSupported_ShouldReturnBoolean(MobileBackend backend)
        {
            var isSupported = TrustformersMobile.TrustformersMobile.IsBackendSupported(backend);
            
            // CPU should always be supported
            if (backend == MobileBackend.CPU)
            {
                Assert.True(isSupported);
            }
        }

        [Fact]
        public void MobileConfig_InvalidMemory_ShouldThrow()
        {
            using var config = new MobileConfig();
            
            // Set invalid memory (too low)
            config.SetMaxMemoryMb(32);
            
            // Should throw when validating
            Assert.Throws<TrustformersMobileException>(() => config.Validate());
        }

        [Fact]
        public void TrustformersMobileException_ShouldContainErrorCode()
        {
            var exception = new TrustformersMobileException(TrustformersMobileError.InvalidParameter);
            
            Assert.Equal(TrustformersMobileError.InvalidParameter, exception.ErrorCode);
            Assert.Contains("Invalid parameter", exception.Message);
        }

        [Fact]
        public void TrustformersMobileException_WithCustomMessage_ShouldUseCustomMessage()
        {
            var customMessage = "Custom error message";
            var exception = new TrustformersMobileException(TrustformersMobileError.RuntimeError, customMessage);
            
            Assert.Equal(TrustformersMobileError.RuntimeError, exception.ErrorCode);
            Assert.Equal(customMessage, exception.Message);
        }

        [Fact]
        public void MobileConfig_DisposeTwice_ShouldNotThrow()
        {
            var config = new MobileConfig();
            
            config.Dispose();
            
            // Should not throw on second dispose
            config.Dispose();
        }

        [Fact]
        public void MobileConfig_UseAfterDispose_ShouldThrow()
        {
            var config = new MobileConfig();
            config.Dispose();
            
            // Should throw when using disposed object
            Assert.Throws<ObjectDisposedException>(() => config.SetPlatform(MobilePlatform.iOS));
        }
    }
}