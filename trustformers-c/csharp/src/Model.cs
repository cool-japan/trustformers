using System;
using System.Text.Json;
using System.Collections.Generic;

namespace TrustformeRS
{
    /// <summary>
    /// High-level wrapper for TrustformersRS model operations
    /// </summary>
    public class Model : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed = false;

        /// <summary>
        /// Gets the native handle for this model
        /// </summary>
        public IntPtr Handle => _handle;

        /// <summary>
        /// Gets whether the model is loaded and ready for use
        /// </summary>
        public bool IsLoaded => _handle != IntPtr.Zero;

        /// <summary>
        /// Model information
        /// </summary>
        public ModelInfo Info { get; private set; }

        /// <summary>
        /// Load a model from the specified path
        /// </summary>
        /// <param name="modelPath">Path to the model file</param>
        /// <param name="config">Optional configuration JSON</param>
        /// <returns>Loaded model instance</returns>
        public static Model Load(string modelPath, string config = null)
        {
            if (string.IsNullOrEmpty(modelPath))
                throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));

            var model = new Model();
            var error = TrustformeRS.trustformers_load_model(modelPath, config, out model._handle);
            TrustformeRS.ThrowIfError(error, $"Failed to load model from {modelPath}");

            // Load model info
            model.LoadModelInfo();

            return model;
        }

        /// <summary>
        /// Load a model with specific configuration options
        /// </summary>
        /// <param name="modelPath">Path to the model file</param>
        /// <param name="options">Configuration options</param>
        /// <returns>Loaded model instance</returns>
        public static Model Load(string modelPath, ModelLoadOptions options)
        {
            var configJson = JsonSerializer.Serialize(options, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
            return Load(modelPath, configJson);
        }

        private Model()
        {
            _handle = IntPtr.Zero;
        }

        private void LoadModelInfo()
        {
            if (_handle == IntPtr.Zero)
                return;

            var error = TrustformeRS.trustformers_get_model_info(_handle, out IntPtr infoPtr);
            if (TrustformeRS.IsSuccess(error))
            {
                try
                {
                    var infoJson = TrustformeRS.PtrToStringAndFree(infoPtr);
                    if (!string.IsNullOrEmpty(infoJson))
                    {
                        Info = JsonSerializer.Deserialize<ModelInfo>(infoJson, new JsonSerializerOptions
                        {
                            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                        });
                    }
                }
                catch
                {
                    // If deserialization fails, create basic info
                    Info = new ModelInfo { Name = "Unknown Model" };
                }
            }
            else
            {
                Info = new ModelInfo { Name = "Unknown Model" };
            }
        }

        /// <summary>
        /// Reload the model information
        /// </summary>
        public void RefreshInfo()
        {
            LoadModelInfo();
        }

        /// <summary>
        /// Unload the model and free resources
        /// </summary>
        public void Unload()
        {
            if (_handle != IntPtr.Zero)
            {
                var error = TrustformeRS.trustformers_unload_model(_handle);
                _handle = IntPtr.Zero;
                Info = null;
                
                // Don't throw on unload errors during cleanup
                if (!TrustformeRS.IsSuccess(error))
                {
                    Console.WriteLine($"Warning: Error unloading model: {error}");
                }
            }
        }

        /// <summary>
        /// Dispose of the model and free resources
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                Unload();
                _disposed = true;
            }
        }

        ~Model()
        {
            Dispose(false);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Model));
        }

        private void ThrowIfNotLoaded()
        {
            ThrowIfDisposed();
            if (!IsLoaded)
                throw new InvalidOperationException("Model is not loaded");
        }

        /// <summary>
        /// Create a string representation of the model
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            if (_disposed)
                return "Model [Disposed]";
            
            if (!IsLoaded)
                return "Model [Not Loaded]";

            return Info != null ? $"Model [{Info.Name}]" : "Model [Loaded]";
        }
    }

    /// <summary>
    /// Configuration options for loading a model
    /// </summary>
    public class ModelLoadOptions
    {
        /// <summary>
        /// Device to load the model on (e.g., "cpu", "cuda:0", "rocm:0")
        /// </summary>
        public string Device { get; set; } = "cpu";

        /// <summary>
        /// Precision mode for the model (e.g., "float32", "float16", "int8")
        /// </summary>
        public string Precision { get; set; } = "float32";

        /// <summary>
        /// Maximum memory usage in MB
        /// </summary>
        public int? MaxMemoryMB { get; set; }

        /// <summary>
        /// Enable quantization
        /// </summary>
        public bool EnableQuantization { get; set; } = false;

        /// <summary>
        /// Quantization method (e.g., "dynamic", "static")
        /// </summary>
        public string QuantizationMethod { get; set; } = "dynamic";

        /// <summary>
        /// Enable model optimization
        /// </summary>
        public bool EnableOptimization { get; set; } = true;

        /// <summary>
        /// Optimization level (0-3, higher = more aggressive)
        /// </summary>
        public int OptimizationLevel { get; set; } = 1;

        /// <summary>
        /// Enable debugging mode
        /// </summary>
        public bool EnableDebugging { get; set; } = false;

        /// <summary>
        /// Custom model configuration
        /// </summary>
        public Dictionary<string, object> CustomConfig { get; set; } = new Dictionary<string, object>();
    }

    /// <summary>
    /// Information about a loaded model
    /// </summary>
    public class ModelInfo
    {
        /// <summary>
        /// Model name
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Model version
        /// </summary>
        public string Version { get; set; }

        /// <summary>
        /// Model architecture (e.g., "gpt2", "bert", "llama")
        /// </summary>
        public string Architecture { get; set; }

        /// <summary>
        /// Model parameters (in millions)
        /// </summary>
        public long? Parameters { get; set; }

        /// <summary>
        /// Model size in bytes
        /// </summary>
        public long? SizeBytes { get; set; }

        /// <summary>
        /// Vocabulary size
        /// </summary>
        public int? VocabSize { get; set; }

        /// <summary>
        /// Maximum sequence length
        /// </summary>
        public int? MaxSequenceLength { get; set; }

        /// <summary>
        /// Number of layers
        /// </summary>
        public int? NumLayers { get; set; }

        /// <summary>
        /// Hidden dimension size
        /// </summary>
        public int? HiddenSize { get; set; }

        /// <summary>
        /// Number of attention heads
        /// </summary>
        public int? NumAttentionHeads { get; set; }

        /// <summary>
        /// Device the model is loaded on
        /// </summary>
        public string Device { get; set; }

        /// <summary>
        /// Precision/data type of the model
        /// </summary>
        public string Precision { get; set; }

        /// <summary>
        /// Whether the model is quantized
        /// </summary>
        public bool? IsQuantized { get; set; }

        /// <summary>
        /// Quantization method used
        /// </summary>
        public string QuantizationMethod { get; set; }

        /// <summary>
        /// Memory usage in bytes
        /// </summary>
        public long? MemoryUsageBytes { get; set; }

        /// <summary>
        /// Model capabilities (list of supported tasks)
        /// </summary>
        public List<string> Capabilities { get; set; } = new List<string>();

        /// <summary>
        /// Additional metadata
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();

        /// <summary>
        /// Get a human-readable string representation of the model size
        /// </summary>
        /// <returns>Human-readable size string</returns>
        public string GetHumanReadableSize()
        {
            if (!SizeBytes.HasValue)
                return "Unknown";

            var size = SizeBytes.Value;
            string[] units = { "B", "KB", "MB", "GB", "TB" };
            int unitIndex = 0;
            double sizeValue = size;

            while (sizeValue >= 1024 && unitIndex < units.Length - 1)
            {
                sizeValue /= 1024;
                unitIndex++;
            }

            return $"{sizeValue:F2} {units[unitIndex]}";
        }

        /// <summary>
        /// Get a human-readable string representation of the parameter count
        /// </summary>
        /// <returns>Human-readable parameter count string</returns>
        public string GetHumanReadableParameters()
        {
            if (!Parameters.HasValue)
                return "Unknown";

            var params_ = Parameters.Value;
            if (params_ >= 1000000000)
                return $"{params_ / 1000000000.0:F1}B";
            else if (params_ >= 1000000)
                return $"{params_ / 1000000.0:F1}M";
            else if (params_ >= 1000)
                return $"{params_ / 1000.0:F1}K";
            else
                return params_.ToString();
        }
    }
}