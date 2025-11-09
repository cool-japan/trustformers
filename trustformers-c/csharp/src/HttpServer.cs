using System;
using System.Text.Json;
using System.Collections.Generic;

namespace TrustformeRS
{
    /// <summary>
    /// High-level wrapper for TrustformersRS HTTP server operations
    /// </summary>
    public class HttpServer : IDisposable
    {
        private IntPtr _serverIdPtr;
        private string _serverId;
        private bool _disposed = false;
        private bool _isStarted = false;

        /// <summary>
        /// Gets the server ID
        /// </summary>
        public string ServerId => _serverId;

        /// <summary>
        /// Gets whether the server is started
        /// </summary>
        public bool IsStarted => _isStarted;

        /// <summary>
        /// Create an HTTP server with default configuration
        /// </summary>
        /// <returns>HTTP server instance</returns>
        public static HttpServer Create()
        {
            var server = new HttpServer();
            var error = TrustformeRS.trustformers_http_server_create(out server._serverIdPtr);
            TrustformeRS.ThrowIfError(error, "Failed to create HTTP server");

            server._serverId = TrustformeRS.PtrToStringAndFree(server._serverIdPtr);
            return server;
        }

        /// <summary>
        /// Create an HTTP server with custom configuration
        /// </summary>
        /// <param name="config">Server configuration</param>
        /// <returns>HTTP server instance</returns>
        public static HttpServer Create(HttpServerConfig config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            var configJson = JsonSerializer.Serialize(config, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

            var server = new HttpServer();
            var error = TrustformeRS.trustformers_http_server_create_with_config(configJson, out server._serverIdPtr);
            TrustformeRS.ThrowIfError(error, "Failed to create HTTP server with config");

            server._serverId = TrustformeRS.PtrToStringAndFree(server._serverIdPtr);
            return server;
        }

        private HttpServer()
        {
            _serverIdPtr = IntPtr.Zero;
        }

        /// <summary>
        /// Add a model endpoint to the server
        /// </summary>
        /// <param name="endpoint">Model endpoint configuration</param>
        public void AddModelEndpoint(ModelEndpoint endpoint)
        {
            ThrowIfDisposed();
            
            if (endpoint == null)
                throw new ArgumentNullException(nameof(endpoint));

            var endpointJson = JsonSerializer.Serialize(endpoint, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

            using var serverIdPtr = new UnmanagedString(_serverId);
            var error = TrustformeRS.trustformers_http_server_add_model(serverIdPtr.Pointer, endpointJson);
            TrustformeRS.ThrowIfError(error, "Failed to add model endpoint");
        }

        /// <summary>
        /// Start the HTTP server
        /// </summary>
        public void Start()
        {
            ThrowIfDisposed();
            
            if (_isStarted)
                throw new InvalidOperationException("Server is already started");

            using var serverIdPtr = new UnmanagedString(_serverId);
            var error = TrustformeRS.trustformers_http_server_start(serverIdPtr.Pointer);
            TrustformeRS.ThrowIfError(error, "Failed to start HTTP server");

            _isStarted = true;
        }

        /// <summary>
        /// Stop the HTTP server
        /// </summary>
        public void Stop()
        {
            ThrowIfDisposed();
            
            if (!_isStarted)
                return;

            using var serverIdPtr = new UnmanagedString(_serverId);
            var error = TrustformeRS.trustformers_http_server_stop(serverIdPtr.Pointer);
            TrustformeRS.ThrowIfError(error, "Failed to stop HTTP server");

            _isStarted = false;
        }

        /// <summary>
        /// Get server metrics
        /// </summary>
        /// <returns>Server metrics</returns>
        public HttpServerMetrics GetMetrics()
        {
            ThrowIfDisposed();

            using var serverIdPtr = new UnmanagedString(_serverId);
            var error = TrustformeRS.trustformers_http_server_get_metrics(serverIdPtr.Pointer, out IntPtr metricsPtr);
            TrustformeRS.ThrowIfError(error, "Failed to get server metrics");

            var metricsJson = TrustformeRS.PtrToStringAndFree(metricsPtr);
            if (string.IsNullOrEmpty(metricsJson))
                return new HttpServerMetrics();

            try
            {
                return JsonSerializer.Deserialize<HttpServerMetrics>(metricsJson, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });
            }
            catch (JsonException ex)
            {
                throw new TrustformersException(TrustformersError.SerializationError, 
                    "Failed to deserialize server metrics", ex);
            }
        }

        /// <summary>
        /// Dispose of the server and free resources
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
                try
                {
                    if (_isStarted)
                    {
                        Stop();
                    }
                }
                catch
                {
                    // Ignore errors during cleanup
                }

                if (!string.IsNullOrEmpty(_serverId))
                {
                    try
                    {
                        using var serverIdPtr = new UnmanagedString(_serverId);
                        TrustformeRS.trustformers_http_server_destroy(serverIdPtr.Pointer);
                    }
                    catch
                    {
                        // Ignore errors during cleanup
                    }
                }

                _disposed = true;
            }
        }

        ~HttpServer()
        {
            Dispose(false);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(HttpServer));
        }

        /// <summary>
        /// Create a string representation of the server
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            if (_disposed)
                return "HttpServer [Disposed]";
            
            var status = _isStarted ? "Started" : "Stopped";
            return $"HttpServer [{_serverId}] [{status}]";
        }
    }

    /// <summary>
    /// HTTP server configuration
    /// </summary>
    public class HttpServerConfig
    {
        /// <summary>
        /// Host address to bind to
        /// </summary>
        public string Host { get; set; } = "127.0.0.1";

        /// <summary>
        /// Port to bind to
        /// </summary>
        public int Port { get; set; } = 8080;

        /// <summary>
        /// Maximum number of concurrent connections
        /// </summary>
        public int MaxConnections { get; set; } = 100;

        /// <summary>
        /// Request timeout in milliseconds
        /// </summary>
        public int RequestTimeoutMs { get; set; } = 30000;

        /// <summary>
        /// Enable CORS
        /// </summary>
        public bool EnableCors { get; set; } = true;

        /// <summary>
        /// Allowed CORS origins
        /// </summary>
        public List<string> CorsOrigins { get; set; } = new List<string> { "*" };

        /// <summary>
        /// Enable metrics endpoint
        /// </summary>
        public bool EnableMetrics { get; set; } = true;

        /// <summary>
        /// Metrics endpoint path
        /// </summary>
        public string MetricsEndpoint { get; set; } = "/metrics";

        /// <summary>
        /// Health check endpoint path
        /// </summary>
        public string HealthEndpoint { get; set; } = "/health";

        /// <summary>
        /// API prefix for all endpoints
        /// </summary>
        public string ApiPrefix { get; set; } = "/api/v1";

        /// <summary>
        /// Enable SSL/TLS
        /// </summary>
        public bool EnableSsl { get; set; } = false;

        /// <summary>
        /// Path to SSL certificate file
        /// </summary>
        public string SslCertPath { get; set; }

        /// <summary>
        /// Path to SSL private key file
        /// </summary>
        public string SslKeyPath { get; set; }
    }

    /// <summary>
    /// Model endpoint configuration
    /// </summary>
    public class ModelEndpoint
    {
        /// <summary>
        /// Endpoint name
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Path to the model
        /// </summary>
        public string ModelPath { get; set; }

        /// <summary>
        /// HTTP endpoint path
        /// </summary>
        public string EndpointPath { get; set; }

        /// <summary>
        /// Maximum batch size for requests
        /// </summary>
        public int MaxBatchSize { get; set; } = 1;

        /// <summary>
        /// Request timeout in milliseconds
        /// </summary>
        public int TimeoutMs { get; set; } = 30000;

        /// <summary>
        /// Enable streaming responses
        /// </summary>
        public bool EnableStreaming { get; set; } = false;

        /// <summary>
        /// Rate limiting configuration
        /// </summary>
        public RateLimit RateLimit { get; set; }
    }

    /// <summary>
    /// Rate limiting configuration
    /// </summary>
    public class RateLimit
    {
        /// <summary>
        /// Number of requests allowed per minute
        /// </summary>
        public int RequestsPerMinute { get; set; } = 60;

        /// <summary>
        /// Burst limit for short-term spikes
        /// </summary>
        public int BurstLimit { get; set; } = 10;
    }

    /// <summary>
    /// HTTP server metrics
    /// </summary>
    public class HttpServerMetrics
    {
        /// <summary>
        /// Total number of requests processed
        /// </summary>
        public long TotalRequests { get; set; }

        /// <summary>
        /// Number of successful requests
        /// </summary>
        public long SuccessfulRequests { get; set; }

        /// <summary>
        /// Number of failed requests
        /// </summary>
        public long FailedRequests { get; set; }

        /// <summary>
        /// Average response time in milliseconds
        /// </summary>
        public double AverageResponseTimeMs { get; set; }

        /// <summary>
        /// Current number of active connections
        /// </summary>
        public int CurrentConnections { get; set; }

        /// <summary>
        /// Peak number of concurrent connections
        /// </summary>
        public int PeakConnections { get; set; }

        /// <summary>
        /// Total bytes sent
        /// </summary>
        public long BytesSent { get; set; }

        /// <summary>
        /// Total bytes received
        /// </summary>
        public long BytesReceived { get; set; }

        /// <summary>
        /// Request counts per model endpoint
        /// </summary>
        public Dictionary<string, long> ModelRequests { get; set; } = new Dictionary<string, long>();

        /// <summary>
        /// Success rate as a percentage
        /// </summary>
        public double SuccessRate => TotalRequests > 0 ? (double)SuccessfulRequests / TotalRequests * 100 : 0;

        /// <summary>
        /// Requests per second (estimated)
        /// </summary>
        public double RequestsPerSecond { get; set; }
    }

    /// <summary>
    /// Helper class for managing unmanaged string memory
    /// </summary>
    internal class UnmanagedString : IDisposable
    {
        public IntPtr Pointer { get; private set; }

        public UnmanagedString(string value)
        {
            if (value != null)
            {
                Pointer = System.Runtime.InteropServices.Marshal.StringToHGlobalAnsi(value);
            }
        }

        public void Dispose()
        {
            if (Pointer != IntPtr.Zero)
            {
                System.Runtime.InteropServices.Marshal.FreeHGlobal(Pointer);
                Pointer = IntPtr.Zero;
            }
        }
    }
}