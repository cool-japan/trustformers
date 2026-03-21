using System;
using System.Text.Json;
using System.Collections.Generic;

namespace TrustformeRS
{
    /// <summary>
    /// High-level wrapper for TrustformersRS pipeline operations
    /// </summary>
    public class Pipeline : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed = false;
        private readonly string _pipelineType;

        /// <summary>
        /// Gets the native handle for this pipeline
        /// </summary>
        public IntPtr Handle => _handle;

        /// <summary>
        /// Gets whether the pipeline is loaded and ready for use
        /// </summary>
        public bool IsLoaded => _handle != IntPtr.Zero;

        /// <summary>
        /// Gets the type of this pipeline
        /// </summary>
        public string PipelineType => _pipelineType;

        /// <summary>
        /// Create a text generation pipeline
        /// </summary>
        /// <param name="model">The model to use</param>
        /// <param name="tokenizer">The tokenizer to use</param>
        /// <param name="config">Optional configuration</param>
        /// <returns>Text generation pipeline</returns>
        public static Pipeline CreateTextGeneration(Model model, Tokenizer tokenizer, string config = null)
        {
            return Create("text-generation", model, tokenizer, config);
        }

        /// <summary>
        /// Create a text classification pipeline
        /// </summary>
        /// <param name="model">The model to use</param>
        /// <param name="tokenizer">The tokenizer to use</param>
        /// <param name="config">Optional configuration</param>
        /// <returns>Text classification pipeline</returns>
        public static Pipeline CreateTextClassification(Model model, Tokenizer tokenizer, string config = null)
        {
            return Create("text-classification", model, tokenizer, config);
        }

        /// <summary>
        /// Create a question answering pipeline
        /// </summary>
        /// <param name="model">The model to use</param>
        /// <param name="tokenizer">The tokenizer to use</param>
        /// <param name="config">Optional configuration</param>
        /// <returns>Question answering pipeline</returns>
        public static Pipeline CreateQuestionAnswering(Model model, Tokenizer tokenizer, string config = null)
        {
            return Create("question-answering", model, tokenizer, config);
        }

        /// <summary>
        /// Create a pipeline of the specified type
        /// </summary>
        /// <param name="pipelineType">Type of pipeline to create</param>
        /// <param name="model">The model to use</param>
        /// <param name="tokenizer">The tokenizer to use</param>
        /// <param name="config">Optional configuration JSON</param>
        /// <returns>Pipeline instance</returns>
        public static Pipeline Create(string pipelineType, Model model, Tokenizer tokenizer, string config = null)
        {
            if (string.IsNullOrEmpty(pipelineType))
                throw new ArgumentException("Pipeline type cannot be null or empty", nameof(pipelineType));
            
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            
            if (tokenizer == null)
                throw new ArgumentNullException(nameof(tokenizer));

            if (!model.IsLoaded)
                throw new ArgumentException("Model must be loaded", nameof(model));
            
            if (!tokenizer.IsLoaded)
                throw new ArgumentException("Tokenizer must be loaded", nameof(tokenizer));

            var pipeline = new Pipeline(pipelineType);
            var error = TrustformeRS.trustformers_create_pipeline(
                pipelineType, 
                model.Handle, 
                tokenizer.Handle, 
                config, 
                out pipeline._handle);
            
            TrustformeRS.ThrowIfError(error, $"Failed to create {pipelineType} pipeline");

            return pipeline;
        }

        private Pipeline(string pipelineType)
        {
            _pipelineType = pipelineType;
            _handle = IntPtr.Zero;
        }

        /// <summary>
        /// Generate text using this pipeline
        /// </summary>
        /// <param name="prompt">Input prompt</param>
        /// <param name="options">Generation options</param>
        /// <returns>Generated text result</returns>
        public TextGenerationResult GenerateText(string prompt, TextGenerationOptions options = null)
        {
            ThrowIfNotLoaded();
            
            if (_pipelineType != "text-generation")
                throw new InvalidOperationException("This pipeline is not configured for text generation");

            if (string.IsNullOrEmpty(prompt))
                throw new ArgumentException("Prompt cannot be null or empty", nameof(prompt));

            var config = options != null ? JsonSerializer.Serialize(options, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            }) : null;

            var error = TrustformeRS.trustformers_pipeline_text_generation(_handle, prompt, config, out IntPtr resultPtr);
            TrustformeRS.ThrowIfError(error, "Text generation failed");

            var resultJson = TrustformeRS.PtrToStringAndFree(resultPtr);
            if (string.IsNullOrEmpty(resultJson))
                throw new InvalidOperationException("No result returned from text generation");

            try
            {
                return JsonSerializer.Deserialize<TextGenerationResult>(resultJson, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });
            }
            catch (JsonException ex)
            {
                throw new TrustformersException(TrustformersError.SerializationError, 
                    "Failed to deserialize text generation result", ex);
            }
        }

        /// <summary>
        /// Classify text using this pipeline
        /// </summary>
        /// <param name="text">Text to classify</param>
        /// <returns>Classification result</returns>
        public TextClassificationResult ClassifyText(string text)
        {
            ThrowIfNotLoaded();
            
            if (_pipelineType != "text-classification")
                throw new InvalidOperationException("This pipeline is not configured for text classification");

            if (string.IsNullOrEmpty(text))
                throw new ArgumentException("Text cannot be null or empty", nameof(text));

            var error = TrustformeRS.trustformers_pipeline_text_classification(_handle, text, out IntPtr resultPtr);
            TrustformeRS.ThrowIfError(error, "Text classification failed");

            var resultJson = TrustformeRS.PtrToStringAndFree(resultPtr);
            if (string.IsNullOrEmpty(resultJson))
                throw new InvalidOperationException("No result returned from text classification");

            try
            {
                return JsonSerializer.Deserialize<TextClassificationResult>(resultJson, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });
            }
            catch (JsonException ex)
            {
                throw new TrustformersException(TrustformersError.SerializationError, 
                    "Failed to deserialize text classification result", ex);
            }
        }

        /// <summary>
        /// Destroy the pipeline and free resources
        /// </summary>
        public void Destroy()
        {
            if (_handle != IntPtr.Zero)
            {
                var error = TrustformeRS.trustformers_destroy_pipeline(_handle);
                _handle = IntPtr.Zero;
                
                // Don't throw on destroy errors during cleanup
                if (!TrustformeRS.IsSuccess(error))
                {
                    Console.WriteLine($"Warning: Error destroying pipeline: {error}");
                }
            }
        }

        /// <summary>
        /// Dispose of the pipeline and free resources
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
                Destroy();
                _disposed = true;
            }
        }

        ~Pipeline()
        {
            Dispose(false);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Pipeline));
        }

        private void ThrowIfNotLoaded()
        {
            ThrowIfDisposed();
            if (!IsLoaded)
                throw new InvalidOperationException("Pipeline is not loaded");
        }

        /// <summary>
        /// Create a string representation of the pipeline
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            if (_disposed)
                return $"Pipeline [{_pipelineType}] [Disposed]";
            
            if (!IsLoaded)
                return $"Pipeline [{_pipelineType}] [Not Loaded]";

            return $"Pipeline [{_pipelineType}] [Loaded]";
        }
    }

    /// <summary>
    /// Options for text generation
    /// </summary>
    public class TextGenerationOptions
    {
        /// <summary>
        /// Maximum length of generated text
        /// </summary>
        public int? MaxLength { get; set; }

        /// <summary>
        /// Minimum length of generated text
        /// </summary>
        public int? MinLength { get; set; }

        /// <summary>
        /// Sampling temperature (0.0 to 1.0)
        /// </summary>
        public double? Temperature { get; set; }

        /// <summary>
        /// Top-k sampling
        /// </summary>
        public int? TopK { get; set; }

        /// <summary>
        /// Top-p (nucleus) sampling
        /// </summary>
        public double? TopP { get; set; }

        /// <summary>
        /// Repetition penalty
        /// </summary>
        public double? RepetitionPenalty { get; set; }

        /// <summary>
        /// Whether to use sampling or greedy decoding
        /// </summary>
        public bool? DoSample { get; set; }

        /// <summary>
        /// Number of sequences to return
        /// </summary>
        public int? NumReturnSequences { get; set; }

        /// <summary>
        /// Random seed for reproducible generation
        /// </summary>
        public int? Seed { get; set; }

        /// <summary>
        /// Stop sequences that will end generation
        /// </summary>
        public List<string> StopSequences { get; set; } = new List<string>();

        /// <summary>
        /// Custom configuration options
        /// </summary>
        public Dictionary<string, object> CustomOptions { get; set; } = new Dictionary<string, object>();
    }

    /// <summary>
    /// Result from text generation
    /// </summary>
    public class TextGenerationResult
    {
        /// <summary>
        /// The original input prompt
        /// </summary>
        public string Prompt { get; set; }

        /// <summary>
        /// Generated text sequences
        /// </summary>
        public List<string> GeneratedText { get; set; } = new List<string>();

        /// <summary>
        /// Processing time in milliseconds
        /// </summary>
        public double ProcessingTimeMs { get; set; }

        /// <summary>
        /// Number of tokens in the prompt
        /// </summary>
        public int PromptTokens { get; set; }

        /// <summary>
        /// Number of tokens generated
        /// </summary>
        public int GeneratedTokens { get; set; }

        /// <summary>
        /// Total number of tokens processed
        /// </summary>
        public int TotalTokens => PromptTokens + GeneratedTokens;

        /// <summary>
        /// Generation speed in tokens per second
        /// </summary>
        public double TokensPerSecond => ProcessingTimeMs > 0 ? (GeneratedTokens / ProcessingTimeMs) * 1000 : 0;

        /// <summary>
        /// Model name used for generation
        /// </summary>
        public string ModelName { get; set; }

        /// <summary>
        /// Whether generation was truncated due to length limits
        /// </summary>
        public bool WasTruncated { get; set; }

        /// <summary>
        /// Additional metadata
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
    }

    /// <summary>
    /// Result from text classification
    /// </summary>
    public class TextClassificationResult
    {
        /// <summary>
        /// The original input text
        /// </summary>
        public string Text { get; set; }

        /// <summary>
        /// Classification results
        /// </summary>
        public List<ClassificationScore> Classifications { get; set; } = new List<ClassificationScore>();

        /// <summary>
        /// Processing time in milliseconds
        /// </summary>
        public double ProcessingTimeMs { get; set; }

        /// <summary>
        /// Model name used for classification
        /// </summary>
        public string ModelName { get; set; }

        /// <summary>
        /// The highest scoring classification
        /// </summary>
        public ClassificationScore TopClassification => 
            Classifications.Count > 0 ? Classifications[0] : null;

        /// <summary>
        /// Additional metadata
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
    }

    /// <summary>
    /// A single classification score
    /// </summary>
    public class ClassificationScore
    {
        /// <summary>
        /// The classification label
        /// </summary>
        public string Label { get; set; }

        /// <summary>
        /// The confidence score (0.0 to 1.0)
        /// </summary>
        public double Score { get; set; }

        /// <summary>
        /// Create a string representation
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{Label}: {Score:F4}";
        }
    }
}