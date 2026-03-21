using System;
using System.Threading.Tasks;
using TrustformeRS;

namespace TrustformeRS.Examples
{
    /// <summary>
    /// Basic usage example for TrustformersRS C# API
    /// </summary>
    class BasicUsageExample
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("TrustformersRS C# Basic Usage Example");
            Console.WriteLine("=====================================");

            try
            {
                // Initialize TrustformersRS
                Console.WriteLine("Initializing TrustformersRS...");
                TrustformeRS.Initialize();
                Console.WriteLine($"TrustformersRS version: {TrustformeRS.GetVersion()}");

                // Check hardware capabilities
                Console.WriteLine("\nHardware capabilities:");
                Console.WriteLine($"CUDA available: {TrustformeRS.IsCudaAvailable()}");
                Console.WriteLine($"ROCm available: {TrustformeRS.IsRocmAvailable()}");

                // Load model and tokenizer
                Console.WriteLine("\nLoading model and tokenizer...");
                using var model = Model.Load("path/to/model", new ModelLoadOptions
                {
                    Device = "cpu",
                    Precision = "float32",
                    EnableOptimization = true
                });
                Console.WriteLine($"Model loaded: {model.Info?.Name ?? "Unknown"}");

                using var tokenizer = Tokenizer.Create("path/to/tokenizer");
                Console.WriteLine("Tokenizer loaded successfully");

                // Create pipeline
                Console.WriteLine("\nCreating text generation pipeline...");
                using var pipeline = Pipeline.CreateTextGeneration(model, tokenizer);
                Console.WriteLine("Pipeline created successfully");

                // Generate text
                Console.WriteLine("\nGenerating text...");
                var prompt = "The future of artificial intelligence is";
                var options = new TextGenerationOptions
                {
                    MaxLength = 100,
                    Temperature = 0.7,
                    TopK = 50,
                    TopP = 0.95,
                    DoSample = true
                };

                var result = pipeline.GenerateText(prompt, options);
                Console.WriteLine($"Prompt: {result.Prompt}");
                Console.WriteLine($"Generated text:");
                foreach (var text in result.GeneratedText)
                {
                    Console.WriteLine($"  - {text}");
                }
                Console.WriteLine($"Processing time: {result.ProcessingTimeMs:F2} ms");
                Console.WriteLine($"Tokens per second: {result.TokensPerSecond:F2}");

                // Test tokenization
                Console.WriteLine("\nTesting tokenization...");
                var testText = "Hello, world! This is a test.";
                var tokens = tokenizer.Encode(testText);
                var decodedText = tokenizer.Decode(tokens);
                Console.WriteLine($"Original: {testText}");
                Console.WriteLine($"Tokens: [{string.Join(", ", tokens)}]");
                Console.WriteLine($"Decoded: {decodedText}");
                Console.WriteLine($"Token count: {tokens.Length}");

                // Test text chunking
                Console.WriteLine("\nTesting text chunking...");
                var longText = "This is a long text that will be split into chunks. " +
                              "Each chunk will have a maximum number of tokens. " +
                              "This is useful for processing long documents.";
                var chunks = tokenizer.SplitIntoChunks(longText, maxTokensPerChunk: 10, overlapTokens: 2);
                Console.WriteLine($"Original text split into {chunks.Count} chunks:");
                for (int i = 0; i < chunks.Count; i++)
                {
                    Console.WriteLine($"  Chunk {i + 1}: {chunks[i]}");
                }

                // Memory usage
                Console.WriteLine("\nMemory usage:");
                var memoryUsage = TrustformeRS.GetMemoryUsage();
                Console.WriteLine(memoryUsage);

                // Performance metrics
                Console.WriteLine("\nPerformance metrics:");
                var metrics = TrustformeRS.GetPerformanceMetrics();
                Console.WriteLine(metrics);

                Console.WriteLine("\nExample completed successfully!");
            }
            catch (TrustformersException ex)
            {
                Console.WriteLine($"TrustformersRS Error: {ex.Message}");
                Console.WriteLine($"Error Code: {ex.ErrorCode}");
                Console.WriteLine($"Severity: {TrustformersException.GetSeverity(ex.ErrorCode)}");
                Console.WriteLine($"Is recoverable: {TrustformersException.IsRecoverable(ex.ErrorCode)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Unexpected error: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
            finally
            {
                // Cleanup
                Console.WriteLine("\nCleaning up...");
                TrustformeRS.CleanupMemory();
                TrustformeRS.Shutdown();
                Console.WriteLine("Cleanup completed");
            }
        }
    }
}