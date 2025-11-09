package trustformers

// +build serving

/*
#cgo CFLAGS: -I../../
#cgo LDFLAGS: -L../../target/release -ltrustformers_c

#include <stdlib.h>

// HTTP server-related functions
extern TrustformersError trustformers_http_server_create(char** server_id);
extern TrustformersError trustformers_http_server_create_with_config(const char* config_json, char** server_id);
extern TrustformersError trustformers_http_server_add_model(const char* server_id, const char* endpoint_json);
extern TrustformersError trustformers_http_server_start(const char* server_id);
extern TrustformersError trustformers_http_server_stop(const char* server_id);
extern TrustformersError trustformers_http_server_get_metrics(const char* server_id, char** metrics_json);
extern TrustformersError trustformers_http_server_destroy(const char* server_id);
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"runtime"
	"unsafe"
)

// HttpServerConfig contains HTTP server configuration
type HttpServerConfig struct {
	Host              string   `json:"host"`
	Port              uint16   `json:"port"`
	MaxConnections    int      `json:"max_connections"`
	RequestTimeoutMs  uint64   `json:"request_timeout_ms"`
	EnableCors        bool     `json:"enable_cors"`
	CorsOrigins       []string `json:"cors_origins"`
	EnableMetrics     bool     `json:"enable_metrics"`
	MetricsEndpoint   string   `json:"metrics_endpoint"`
	HealthEndpoint    string   `json:"health_endpoint"`
	ApiPrefix         string   `json:"api_prefix"`
	EnableSSL         bool     `json:"enable_ssl"`
	SSLCertPath       *string  `json:"ssl_cert_path,omitempty"`
	SSLKeyPath        *string  `json:"ssl_key_path,omitempty"`
}

// ModelEndpoint represents a model serving endpoint
type ModelEndpoint struct {
	Name            string     `json:"name"`
	ModelPath       string     `json:"model_path"`
	EndpointPath    string     `json:"endpoint_path"`
	MaxBatchSize    int        `json:"max_batch_size"`
	TimeoutMs       uint64     `json:"timeout_ms"`
	EnableStreaming bool       `json:"enable_streaming"`
	RateLimit       *RateLimit `json:"rate_limit,omitempty"`
}

// RateLimit contains rate limiting configuration
type RateLimit struct {
	RequestsPerMinute uint32 `json:"requests_per_minute"`
	BurstLimit        uint32 `json:"burst_limit"`
}

// ServerMetrics contains HTTP server metrics
type ServerMetrics struct {
	TotalRequests         uint64            `json:"total_requests"`
	SuccessfulRequests    uint64            `json:"successful_requests"`
	FailedRequests        uint64            `json:"failed_requests"`
	AverageResponseTimeMs float64           `json:"average_response_time_ms"`
	CurrentConnections    uint32            `json:"current_connections"`
	PeakConnections       uint32            `json:"peak_connections"`
	BytesSent             uint64            `json:"bytes_sent"`
	BytesReceived         uint64            `json:"bytes_received"`
	ModelRequests         map[string]uint64 `json:"model_requests"`
}

// TextGenerationRequest represents a text generation API request
type TextGenerationRequest struct {
	Prompt              string   `json:"prompt"`
	MaxLength           *int     `json:"max_length,omitempty"`
	Temperature         *float64 `json:"temperature,omitempty"`
	TopK                *int     `json:"top_k,omitempty"`
	TopP                *float64 `json:"top_p,omitempty"`
	RepetitionPenalty   *float64 `json:"repetition_penalty,omitempty"`
	DoSample            *bool    `json:"do_sample,omitempty"`
	NumReturnSequences  *int     `json:"num_return_sequences,omitempty"`
	Stream              *bool    `json:"stream,omitempty"`
}

// TextGenerationResponse represents a text generation API response
type TextGenerationResponse struct {
	GeneratedText     []string `json:"generated_text"`
	ProcessingTimeMs  float64  `json:"processing_time_ms"`
	ModelName         string   `json:"model_name"`
	PromptTokens      int      `json:"prompt_tokens"`
	GeneratedTokens   int      `json:"generated_tokens"`
}

// TextClassificationRequest represents a text classification API request
type TextClassificationRequest struct {
	Text             string `json:"text"`
	ReturnAllScores  *bool  `json:"return_all_scores,omitempty"`
}

// TextClassificationResponse represents a text classification API response
type TextClassificationResponse struct {
	Classifications   []ClassificationResult `json:"classifications"`
	ProcessingTimeMs  float64                `json:"processing_time_ms"`
	ModelName         string                 `json:"model_name"`
}

// ClassificationResult represents a single classification result
type ClassificationResult struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

// HttpServer represents an HTTP server instance
type HttpServer struct {
	serverID string
	config   HttpServerConfig
	running  bool
}

// DefaultHttpServerConfig returns a default HTTP server configuration
func DefaultHttpServerConfig() HttpServerConfig {
	return HttpServerConfig{
		Host:             "127.0.0.1",
		Port:             8080,
		MaxConnections:   100,
		RequestTimeoutMs: 30000,
		EnableCors:       true,
		CorsOrigins:      []string{"*"},
		EnableMetrics:    true,
		MetricsEndpoint:  "/metrics",
		HealthEndpoint:   "/health",
		ApiPrefix:        "/api/v1",
		EnableSSL:        false,
	}
}

// NewHttpServer creates a new HTTP server with default configuration
func NewHttpServer() (*HttpServer, error) {
	return NewHttpServerWithConfig(DefaultHttpServerConfig())
}

// NewHttpServerWithConfig creates a new HTTP server with custom configuration
func NewHttpServerWithConfig(config HttpServerConfig) (*HttpServer, error) {
	var cServerID *C.char

	if config == (HttpServerConfig{}) {
		// Use default configuration
		err := C.trustformers_http_server_create(&cServerID)
		if err := checkError(err); err != nil {
			return nil, err
		}
	} else {
		// Use custom configuration
		configJSON, err := json.Marshal(config)
		if err != nil {
			return nil, err
		}

		cConfigJSON := C.CString(string(configJSON))
		defer C.free(unsafe.Pointer(cConfigJSON))

		err2 := C.trustformers_http_server_create_with_config(cConfigJSON, &cServerID)
		if err := checkError(err2); err != nil {
			return nil, err
		}
	}

	if cServerID == nil {
		return nil, errors.New("failed to create HTTP server")
	}

	serverID := C.GoString(cServerID)
	freeCString(cServerID)

	server := &HttpServer{
		serverID: serverID,
		config:   config,
		running:  false,
	}

	runtime.SetFinalizer(server, (*HttpServer).finalize)
	return server, nil
}

// AddModelEndpoint adds a model endpoint to the HTTP server
func (hs *HttpServer) AddModelEndpoint(endpoint ModelEndpoint) error {
	endpointJSON, err := json.Marshal(endpoint)
	if err != nil {
		return err
	}

	cServerID := C.CString(hs.serverID)
	defer C.free(unsafe.Pointer(cServerID))

	cEndpointJSON := C.CString(string(endpointJSON))
	defer C.free(unsafe.Pointer(cEndpointJSON))

	err2 := C.trustformers_http_server_add_model(cServerID, cEndpointJSON)
	return checkError(err2)
}

// AddTextGenerationEndpoint adds a text generation endpoint
func (hs *HttpServer) AddTextGenerationEndpoint(modelName, modelPath, endpointPath string, options ...EndpointOption) error {
	endpoint := ModelEndpoint{
		Name:            modelName,
		ModelPath:       modelPath,
		EndpointPath:    endpointPath,
		MaxBatchSize:    32,
		TimeoutMs:       30000,
		EnableStreaming: false,
	}

	// Apply options
	for _, option := range options {
		option(&endpoint)
	}

	return hs.AddModelEndpoint(endpoint)
}

// AddTextClassificationEndpoint adds a text classification endpoint
func (hs *HttpServer) AddTextClassificationEndpoint(modelName, modelPath, endpointPath string, options ...EndpointOption) error {
	endpoint := ModelEndpoint{
		Name:            modelName,
		ModelPath:       modelPath,
		EndpointPath:    endpointPath,
		MaxBatchSize:    64,
		TimeoutMs:       15000,
		EnableStreaming: false,
	}

	// Apply options
	for _, option := range options {
		option(&endpoint)
	}

	return hs.AddModelEndpoint(endpoint)
}

// EndpointOption is a function type for configuring model endpoints
type EndpointOption func(*ModelEndpoint)

// WithMaxBatchSize sets the maximum batch size for the endpoint
func WithMaxBatchSize(size int) EndpointOption {
	return func(endpoint *ModelEndpoint) {
		endpoint.MaxBatchSize = size
	}
}

// WithTimeout sets the timeout for the endpoint
func WithTimeout(timeoutMs uint64) EndpointOption {
	return func(endpoint *ModelEndpoint) {
		endpoint.TimeoutMs = timeoutMs
	}
}

// WithStreaming enables streaming for the endpoint
func WithStreaming(enable bool) EndpointOption {
	return func(endpoint *ModelEndpoint) {
		endpoint.EnableStreaming = enable
	}
}

// WithRateLimit sets rate limiting for the endpoint
func WithRateLimit(requestsPerMinute, burstLimit uint32) EndpointOption {
	return func(endpoint *ModelEndpoint) {
		endpoint.RateLimit = &RateLimit{
			RequestsPerMinute: requestsPerMinute,
			BurstLimit:        burstLimit,
		}
	}
}

// Start starts the HTTP server
func (hs *HttpServer) Start() error {
	if hs.running {
		return errors.New("server is already running")
	}

	cServerID := C.CString(hs.serverID)
	defer C.free(unsafe.Pointer(cServerID))

	err := C.trustformers_http_server_start(cServerID)
	if err := checkError(err); err != nil {
		return err
	}

	hs.running = true
	return nil
}

// Stop stops the HTTP server
func (hs *HttpServer) Stop() error {
	if !hs.running {
		return nil
	}

	cServerID := C.CString(hs.serverID)
	defer C.free(unsafe.Pointer(cServerID))

	err := C.trustformers_http_server_stop(cServerID)
	if err := checkError(err); err != nil {
		return err
	}

	hs.running = false
	return nil
}

// GetMetrics returns server metrics
func (hs *HttpServer) GetMetrics() (ServerMetrics, error) {
	cServerID := C.CString(hs.serverID)
	defer C.free(unsafe.Pointer(cServerID))

	var cMetricsJSON *C.char
	err := C.trustformers_http_server_get_metrics(cServerID, &cMetricsJSON)
	if err := checkError(err); err != nil {
		return ServerMetrics{}, err
	}
	defer freeCString(cMetricsJSON)

	if cMetricsJSON == nil {
		return ServerMetrics{}, errors.New("failed to get server metrics")
	}

	metricsJSON := C.GoString(cMetricsJSON)
	var metrics ServerMetrics
	if err := json.Unmarshal([]byte(metricsJSON), &metrics); err != nil {
		return ServerMetrics{}, err
	}

	return metrics, nil
}

// GetConfig returns the server configuration
func (hs *HttpServer) GetConfig() HttpServerConfig {
	return hs.config
}

// IsRunning returns whether the server is running
func (hs *HttpServer) IsRunning() bool {
	return hs.running
}

// GetServerID returns the server ID
func (hs *HttpServer) GetServerID() string {
	return hs.serverID
}

// GetAddress returns the server address
func (hs *HttpServer) GetAddress() string {
	return fmt.Sprintf("%s:%d", hs.config.Host, hs.config.Port)
}

// GetHealthEndpoint returns the health check endpoint URL
func (hs *HttpServer) GetHealthEndpoint() string {
	protocol := "http"
	if hs.config.EnableSSL {
		protocol = "https"
	}
	return fmt.Sprintf("%s://%s%s", protocol, hs.GetAddress(), hs.config.HealthEndpoint)
}

// GetMetricsEndpoint returns the metrics endpoint URL
func (hs *HttpServer) GetMetricsEndpoint() string {
	if !hs.config.EnableMetrics {
		return ""
	}
	protocol := "http"
	if hs.config.EnableSSL {
		protocol = "https"
	}
	return fmt.Sprintf("%s://%s%s", protocol, hs.GetAddress(), hs.config.MetricsEndpoint)
}

// Destroy destroys the HTTP server and frees resources
func (hs *HttpServer) Destroy() error {
	if hs.serverID == "" {
		return nil
	}

	// Stop server if running
	if hs.running {
		if err := hs.Stop(); err != nil {
			return err
		}
	}

	cServerID := C.CString(hs.serverID)
	defer C.free(unsafe.Pointer(cServerID))

	err := C.trustformers_http_server_destroy(cServerID)
	if err := checkError(err); err != nil {
		return err
	}

	hs.serverID = ""
	runtime.SetFinalizer(hs, nil)
	return nil
}

// finalize is called by the finalizer
func (hs *HttpServer) finalize() {
	if hs.serverID != "" {
		hs.Destroy()
	}
}

// Additional helper functions

// CreateDefaultTextGenerationEndpoint creates a default text generation endpoint
func CreateDefaultTextGenerationEndpoint(modelName, modelPath string) ModelEndpoint {
	return ModelEndpoint{
		Name:            modelName,
		ModelPath:       modelPath,
		EndpointPath:    "/generate",
		MaxBatchSize:    16,
		TimeoutMs:       30000,
		EnableStreaming: true,
	}
}

// CreateDefaultClassificationEndpoint creates a default classification endpoint
func CreateDefaultClassificationEndpoint(modelName, modelPath string) ModelEndpoint {
	return ModelEndpoint{
		Name:            modelName,
		ModelPath:       modelPath,
		EndpointPath:    "/classify",
		MaxBatchSize:    64,
		TimeoutMs:       15000,
		EnableStreaming: false,
	}
}