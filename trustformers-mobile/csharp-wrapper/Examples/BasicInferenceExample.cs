using System;
using TrustformersMobile;

namespace TrustformersMobile.Examples
{
    /// <summary>
    /// Basic example demonstrating TrustformeRS Mobile inference
    /// </summary>
    public class BasicInferenceExample
    {
        public static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("TrustformeRS Mobile C# Example");
                Console.WriteLine("==============================");

                // Initialize the library
                Console.WriteLine("Initializing TrustformeRS Mobile...");
                TrustformersMobile.TrustformersMobile.Initialize();
                
                // Get library version
                var version = TrustformersMobile.TrustformersMobile.GetVersion();
                Console.WriteLine($"Library version: {version}");

                // Get device information
                Console.WriteLine("\nDevice Information:");
                var deviceInfo = TrustformersMobile.TrustformersMobile.GetDeviceInfo();
                Console.WriteLine($"  Device: {deviceInfo.Name}");
                Console.WriteLine($"  OS: {deviceInfo.OS}");
                Console.WriteLine($"  Architecture: {deviceInfo.Architecture}");
                Console.WriteLine($"  CPU Cores: {deviceInfo.CpuCores}");
                Console.WriteLine($"  Total Memory: {deviceInfo.TotalMemoryMB} MB");
                Console.WriteLine($"  Available Memory: {deviceInfo.AvailableMemoryMB} MB");
                Console.WriteLine($"  GPU: {deviceInfo.GPU}");
                Console.WriteLine($"  Performance Tier: {deviceInfo.PerformanceTier}");
                Console.WriteLine($"  Hardware Acceleration: {deviceInfo.SupportsHardwareAcceleration}");
                
                if (deviceInfo.RecommendedBackends.Length > 0)
                {
                    Console.WriteLine($"  Recommended Backends: {string.Join(", ", deviceInfo.RecommendedBackends)}");
                }

                // Check platform and backend support
                Console.WriteLine("\nPlatform Support:");
                Console.WriteLine($"  iOS: {TrustformersMobile.TrustformersMobile.IsPlatformSupported(MobilePlatform.iOS)}");
                Console.WriteLine($"  Android: {TrustformersMobile.TrustformersMobile.IsPlatformSupported(MobilePlatform.Android)}");
                Console.WriteLine($"  Generic: {TrustformersMobile.TrustformersMobile.IsPlatformSupported(MobilePlatform.Generic)}");

                Console.WriteLine("\nBackend Support:");
                Console.WriteLine($"  CPU: {TrustformersMobile.TrustformersMobile.IsBackendSupported(MobileBackend.CPU)}");
                Console.WriteLine($"  CoreML: {TrustformersMobile.TrustformersMobile.IsBackendSupported(MobileBackend.CoreML)}");
                Console.WriteLine($"  NNAPI: {TrustformersMobile.TrustformersMobile.IsBackendSupported(MobileBackend.NNAPI)}");
                Console.WriteLine($"  GPU: {TrustformersMobile.TrustformersMobile.IsBackendSupported(MobileBackend.GPU)}");
                Console.WriteLine($"  Metal: {TrustformersMobile.TrustformersMobile.IsBackendSupported(MobileBackend.Metal)}");
                Console.WriteLine($"  Vulkan: {TrustformersMobile.TrustformersMobile.IsBackendSupported(MobileBackend.Vulkan)}");
                Console.WriteLine($"  OpenCL: {TrustformersMobile.TrustformersMobile.IsBackendSupported(MobileBackend.OpenCL)}");

                // Configuration examples
                Console.WriteLine("\nConfiguration Examples:");
                
                // Default configuration
                using (var defaultConfig = new MobileConfig())
                {
                    Console.WriteLine("✓ Created default configuration");
                    defaultConfig.Validate();
                    Console.WriteLine("✓ Default configuration is valid");
                }

                // iOS optimized configuration
                if (TrustformersMobile.TrustformersMobile.IsPlatformSupported(MobilePlatform.iOS))
                {
                    using (var iosConfig = MobileConfig.CreateiOSOptimized())
                    {
                        Console.WriteLine("✓ Created iOS optimized configuration");
                        iosConfig.Validate();
                        Console.WriteLine("✓ iOS configuration is valid");
                    }
                }

                // Android optimized configuration
                if (TrustformersMobile.TrustformersMobile.IsPlatformSupported(MobilePlatform.Android))
                {
                    using (var androidConfig = MobileConfig.CreateAndroidOptimized())
                    {
                        Console.WriteLine("✓ Created Android optimized configuration");
                        androidConfig.Validate();
                        Console.WriteLine("✓ Android configuration is valid");
                    }
                }

                // Ultra low memory configuration
                using (var lowMemConfig = MobileConfig.CreateUltraLowMemory())
                {
                    Console.WriteLine("✓ Created ultra low memory configuration");
                    lowMemConfig.Validate();
                    Console.WriteLine("✓ Ultra low memory configuration is valid");
                }

                // Custom configuration
                using (var customConfig = new MobileConfig())
                {
                    customConfig.SetPlatform(MobilePlatform.Generic);
                    customConfig.SetBackend(MobileBackend.CPU);
                    customConfig.SetMemoryOptimization(MemoryOptimization.Balanced);
                    customConfig.SetMaxMemoryMb(512);
                    customConfig.SetUseFp16(true);
                    customConfig.SetNumThreads(4);
                    
                    Console.WriteLine("✓ Created custom configuration");
                    customConfig.Validate();
                    Console.WriteLine("✓ Custom configuration is valid");
                }

                // Model inference example (if model path is provided)
                if (args.Length > 0)
                {
                    string modelPath = args[0];
                    Console.WriteLine($"\nTesting inference with model: {modelPath}");
                    
                    try
                    {
                        using var engine = TrustformersMobile.TrustformersMobile.CreateOptimizedEngine(modelPath);
                        Console.WriteLine("✓ Successfully created inference engine");

                        // Test with sample data
                        float[] sampleInput = { 1.0f, 2.0f, 3.0f, 4.0f };
                        Console.WriteLine($"Input: [{string.Join(", ", sampleInput)}]");
                        
                        float[] output = engine.InferenceF32(sampleInput);
                        Console.WriteLine($"Output: [{string.Join(", ", output)}]");
                        Console.WriteLine("✓ Inference completed successfully");
                    }
                    catch (TrustformersMobileException ex)
                    {
                        Console.WriteLine($"⚠ Inference failed: {ex.Message} (Error: {ex.ErrorCode})");
                    }
                }
                else
                {
                    Console.WriteLine("\nTo test inference, provide a model path as an argument:");
                    Console.WriteLine("  dotnet run BasicInferenceExample.cs path/to/model.bin");
                }

                Console.WriteLine("\n✅ All tests completed successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                if (ex is TrustformersMobileException mobileEx)
                {
                    Console.WriteLine($"Error Code: {mobileEx.ErrorCode}");
                }
                Environment.Exit(1);
            }
        }
    }
}