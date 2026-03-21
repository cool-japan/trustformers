using System;
using System.Threading.Tasks;
using System.Threading;
using TrustformeRS;

namespace TrustformeRS.Examples
{
    /// <summary>
    /// HTTP server example for TrustformersRS C# API
    /// </summary>
    class HttpServerExample
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("TrustformersRS C# HTTP Server Example");
            Console.WriteLine("=====================================");

            try
            {
                // Initialize TrustformersRS
                Console.WriteLine("Initializing TrustformersRS...");
                TrustformeRS.Initialize();
                Console.WriteLine($"TrustformersRS version: {TrustformeRS.GetVersion()}");

                // Create HTTP server with custom configuration
                Console.WriteLine("\nCreating HTTP server...");
                var serverConfig = new HttpServerConfig
                {
                    Host = "127.0.0.1",
                    Port = 8080,
                    MaxConnections = 50,
                    RequestTimeoutMs = 30000,
                    EnableCors = true,
                    EnableMetrics = true,
                    ApiPrefix = "/api/v1"
                };

                using var server = HttpServer.Create(serverConfig);
                Console.WriteLine($"HTTP server created: {server.ServerId}");

                // Add model endpoints
                Console.WriteLine("\nConfiguring model endpoints...");
                
                // Text generation endpoint
                var textGenEndpoint = new ModelEndpoint
                {
                    Name = "gpt2-text-generation",
                    ModelPath = "path/to/gpt2/model",
                    EndpointPath = "/generate",
                    MaxBatchSize = 4,
                    TimeoutMs = 30000,
                    EnableStreaming = true,
                    RateLimit = new RateLimit
                    {
                        RequestsPerMinute = 120,
                        BurstLimit = 20
                    }
                };
                server.AddModelEndpoint(textGenEndpoint);
                Console.WriteLine("Added text generation endpoint");

                // Text classification endpoint
                var classificationEndpoint = new ModelEndpoint
                {
                    Name = "bert-sentiment-analysis",
                    ModelPath = "path/to/bert/model",
                    EndpointPath = "/classify",
                    MaxBatchSize = 8,
                    TimeoutMs = 15000,
                    EnableStreaming = false,
                    RateLimit = new RateLimit
                    {
                        RequestsPerMinute = 300,
                        BurstLimit = 50
                    }
                };
                server.AddModelEndpoint(classificationEndpoint);
                Console.WriteLine("Added text classification endpoint");

                // Start the server
                Console.WriteLine("\nStarting HTTP server...");
                server.Start();
                Console.WriteLine($"Server started and listening on {serverConfig.Host}:{serverConfig.Port}");
                Console.WriteLine("\nAvailable endpoints:");
                Console.WriteLine($"  - Health check: http://{serverConfig.Host}:{serverConfig.Port}/health");
                Console.WriteLine($"  - Metrics: http://{serverConfig.Host}:{serverConfig.Port}/metrics");
                Console.WriteLine($"  - Text generation: http://{serverConfig.Host}:{serverConfig.Port}{serverConfig.ApiPrefix}/generate");
                Console.WriteLine($"  - Text classification: http://{serverConfig.Host}:{serverConfig.Port}{serverConfig.ApiPrefix}/classify");

                // Monitor server metrics
                Console.WriteLine("\nMonitoring server (press Ctrl+C to stop)...");
                
                var cancellationTokenSource = new CancellationTokenSource();
                Console.CancelKeyPress += (sender, e) =>
                {
                    e.Cancel = true;
                    cancellationTokenSource.Cancel();
                };

                // Metrics monitoring loop
                var metricsTask = Task.Run(async () =>
                {
                    while (!cancellationTokenSource.Token.IsCancellationRequested)
                    {
                        try
                        {
                            await Task.Delay(10000, cancellationTokenSource.Token); // Update every 10 seconds
                            
                            var metrics = server.GetMetrics();
                            Console.WriteLine($"\n--- Server Metrics ({DateTime.Now:HH:mm:ss}) ---");
                            Console.WriteLine($"Total requests: {metrics.TotalRequests}");
                            Console.WriteLine($"Successful requests: {metrics.SuccessfulRequests}");
                            Console.WriteLine($"Failed requests: {metrics.FailedRequests}");
                            Console.WriteLine($"Success rate: {metrics.SuccessRate:F1}%");
                            Console.WriteLine($"Average response time: {metrics.AverageResponseTimeMs:F2} ms");
                            Console.WriteLine($"Current connections: {metrics.CurrentConnections}");
                            Console.WriteLine($"Peak connections: {metrics.PeakConnections}");
                            Console.WriteLine($"Bytes sent: {FormatBytes(metrics.BytesSent)}");
                            Console.WriteLine($"Bytes received: {FormatBytes(metrics.BytesReceived)}");

                            if (metrics.ModelRequests.Count > 0)
                            {
                                Console.WriteLine("Model request counts:");
                                foreach (var kvp in metrics.ModelRequests)
                                {
                                    Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
                                }
                            }
                        }
                        catch (OperationCanceledException)
                        {
                            break;
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error getting metrics: {ex.Message}");
                        }
                    }
                });

                // Example client requests (simulated)
                var clientTask = Task.Run(async () =>
                {
                    await Task.Delay(2000); // Wait for server to fully start
                    
                    Console.WriteLine("\nSimulating client requests...");
                    Console.WriteLine("You can test the endpoints using curl or any HTTP client:");
                    Console.WriteLine("\nText generation example:");
                    Console.WriteLine($"curl -X POST http://{serverConfig.Host}:{serverConfig.Port}{serverConfig.ApiPrefix}/generate \\");
                    Console.WriteLine("  -H \"Content-Type: application/json\" \\");
                    Console.WriteLine("  -d '{\"prompt\": \"The future of AI is\", \"max_length\": 50, \"temperature\": 0.7}'");
                    Console.WriteLine("\nText classification example:");
                    Console.WriteLine($"curl -X POST http://{serverConfig.Host}:{serverConfig.Port}{serverConfig.ApiPrefix}/classify \\");
                    Console.WriteLine("  -H \"Content-Type: application/json\" \\");
                    Console.WriteLine("  -d '{\"text\": \"I love this product!\"}'");
                    Console.WriteLine("\nHealth check:");
                    Console.WriteLine($"curl http://{serverConfig.Host}:{serverConfig.Port}/health");
                });

                // Wait for cancellation
                await Task.WhenAny(metricsTask, clientTask);
                await Task.Delay(1000); // Give some time for cleanup
                
                Console.WriteLine("\nShutting down server...");
                server.Stop();
                Console.WriteLine("Server stopped successfully");
            }
            catch (TrustformersException ex)
            {
                Console.WriteLine($"TrustformersRS Error: {ex.Message}");
                Console.WriteLine($"Error Code: {ex.ErrorCode}");
                Console.WriteLine($"Severity: {TrustformersException.GetSeverity(ex.ErrorCode)}");
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

        private static string FormatBytes(long bytes)
        {
            string[] units = { "B", "KB", "MB", "GB", "TB" };
            int unitIndex = 0;
            double value = bytes;

            while (value >= 1024 && unitIndex < units.Length - 1)
            {
                value /= 1024;
                unitIndex++;
            }

            return $"{value:F2} {units[unitIndex]}";
        }
    }
}