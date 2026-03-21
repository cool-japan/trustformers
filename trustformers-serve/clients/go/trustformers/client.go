// Package trustformers provides a comprehensive Go client for TrustformeRS serving infrastructure
package trustformers

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/clientcredentials"
)

// Client represents the main TrustformeRS client
type Client struct {
	baseURL    string
	httpClient *http.Client
	auth       Authenticator
	config     *ClientConfig
	retry      *RetryConfig
}

// ClientConfig holds configuration options for the client
type ClientConfig struct {
	// Timeout for requests (default: 30 seconds)
	Timeout time.Duration
	// UserAgent for requests
	UserAgent string
	// Debug mode for detailed logging
	Debug bool
	// MaxRetries for failed requests
	MaxRetries int
	// DefaultHeaders to include with all requests
	DefaultHeaders map[string]string
}

// RetryConfig configures retry behavior
type RetryConfig struct {
	MaxRetries      int
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	BackoffFactor   float64
	RetryableErrors []int // HTTP status codes to retry
}

// NewClient creates a new TrustformeRS client
func NewClient(baseURL string, options ...ClientOption) (*Client, error) {
	if baseURL == "" {
		return nil, fmt.Errorf("baseURL cannot be empty")
	}

	// Ensure baseURL doesn't end with slash
	baseURL = strings.TrimSuffix(baseURL, "/")

	// Default configuration
	config := &ClientConfig{
		Timeout:    30 * time.Second,
		UserAgent:  "trustformers-go-client/1.0.0",
		Debug:      false,
		MaxRetries: 3,
		DefaultHeaders: map[string]string{
			"Content-Type": "application/json",
			"Accept":       "application/json",
		},
	}

	// Default retry configuration
	retry := &RetryConfig{
		MaxRetries:    3,
		InitialDelay:  100 * time.Millisecond,
		MaxDelay:      5 * time.Second,
		BackoffFactor: 2.0,
		RetryableErrors: []int{
			http.StatusInternalServerError,
			http.StatusBadGateway,
			http.StatusServiceUnavailable,
			http.StatusGatewayTimeout,
			http.StatusTooManyRequests,
		},
	}

	client := &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
		config: config,
		retry:  retry,
	}

	// Apply options
	for _, option := range options {
		if err := option(client); err != nil {
			return nil, fmt.Errorf("failed to apply option: %w", err)
		}
	}

	return client, nil
}

// ClientOption is a function that configures the client
type ClientOption func(*Client) error

// WithTimeout sets the request timeout
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *Client) error {
		c.config.Timeout = timeout
		c.httpClient.Timeout = timeout
		return nil
	}
}

// WithAuth sets the authenticator
func WithAuth(auth Authenticator) ClientOption {
	return func(c *Client) error {
		c.auth = auth
		return nil
	}
}

// WithUserAgent sets the user agent
func WithUserAgent(userAgent string) ClientOption {
	return func(c *Client) error {
		c.config.UserAgent = userAgent
		return nil
	}
}

// WithDebug enables debug mode
func WithDebug(debug bool) ClientOption {
	return func(c *Client) error {
		c.config.Debug = debug
		return nil
	}
}

// WithHTTPClient sets a custom HTTP client
func WithHTTPClient(httpClient *http.Client) ClientOption {
	return func(c *Client) error {
		c.httpClient = httpClient
		return nil
	}
}

// WithRetryConfig sets retry configuration
func WithRetryConfig(retry *RetryConfig) ClientOption {
	return func(c *Client) error {
		c.retry = retry
		return nil
	}
}

// WithDefaultHeaders sets default headers
func WithDefaultHeaders(headers map[string]string) ClientOption {
	return func(c *Client) error {
		for k, v := range headers {
			c.config.DefaultHeaders[k] = v
		}
		return nil
	}
}

// InferenceRequest represents an inference request
type InferenceRequest struct {
	Input      string                 `json:"input"`
	ModelID    string                 `json:"model_id,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	Options    *InferenceOptions      `json:"options,omitempty"`
}

// InferenceOptions contains additional inference options
type InferenceOptions struct {
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	TopK        *int     `json:"top_k,omitempty"`
	Stream      bool     `json:"stream,omitempty"`
}

// InferenceResponse represents an inference response
type InferenceResponse struct {
	ID          string                 `json:"id"`
	Object      string                 `json:"object"`
	Created     int64                  `json:"created"`
	Model       string                 `json:"model"`
	Choices     []Choice               `json:"choices"`
	Usage       Usage                  `json:"usage"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	ProcessingTime float64             `json:"processing_time_ms"`
}

// Choice represents a choice in the response
type Choice struct {
	Index        int     `json:"index"`
	Text         string  `json:"text"`
	FinishReason string  `json:"finish_reason"`
	Confidence   float64 `json:"confidence,omitempty"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// BatchInferenceRequest represents a batch inference request
type BatchInferenceRequest struct {
	Requests []InferenceRequest `json:"requests"`
	Options  *BatchOptions      `json:"options,omitempty"`
}

// BatchOptions contains batch processing options
type BatchOptions struct {
	MaxBatchSize int  `json:"max_batch_size,omitempty"`
	Parallel     bool `json:"parallel,omitempty"`
}

// BatchInferenceResponse represents a batch inference response
type BatchInferenceResponse struct {
	ID        string              `json:"id"`
	Object    string              `json:"object"`
	Created   int64               `json:"created"`
	Responses []InferenceResponse `json:"responses"`
	BatchSize int                 `json:"batch_size"`
}

// HealthStatus represents the health status of the server
type HealthStatus struct {
	Status      string                 `json:"status"`
	Timestamp   time.Time              `json:"timestamp"`
	Version     string                 `json:"version"`
	Uptime      float64                `json:"uptime"`
	Details     map[string]interface{} `json:"details,omitempty"`
	Components  map[string]string      `json:"components,omitempty"`
}

// ModelInfo represents information about a model
type ModelInfo struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Version      string                 `json:"version"`
	Description  string                 `json:"description"`
	Architecture string                 `json:"architecture"`
	Parameters   int64                  `json:"parameters"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	Status       string                 `json:"status"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
}

// Inference performs a single inference request
func (c *Client) Inference(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	var resp InferenceResponse
	err := c.makeRequest(ctx, "POST", "/v1/inference", req, &resp)
	if err != nil {
		return nil, fmt.Errorf("inference request failed: %w", err)
	}

	return &resp, nil
}

// BatchInference performs a batch inference request
func (c *Client) BatchInference(ctx context.Context, req *BatchInferenceRequest) (*BatchInferenceResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	var resp BatchInferenceResponse
	err := c.makeRequest(ctx, "POST", "/v1/inference/batch", req, &resp)
	if err != nil {
		return nil, fmt.Errorf("batch inference request failed: %w", err)
	}

	return &resp, nil
}

// Health checks the health status of the server
func (c *Client) Health(ctx context.Context) (*HealthStatus, error) {
	var health HealthStatus
	err := c.makeRequest(ctx, "GET", "/health", nil, &health)
	if err != nil {
		return nil, fmt.Errorf("health check failed: %w", err)
	}

	return &health, nil
}

// DetailedHealth gets detailed health information
func (c *Client) DetailedHealth(ctx context.Context) (*HealthStatus, error) {
	var health HealthStatus
	err := c.makeRequest(ctx, "GET", "/health/detailed", nil, &health)
	if err != nil {
		return nil, fmt.Errorf("detailed health check failed: %w", err)
	}

	return &health, nil
}

// ListModels lists available models
func (c *Client) ListModels(ctx context.Context) ([]ModelInfo, error) {
	var models []ModelInfo
	err := c.makeRequest(ctx, "GET", "/v1/models", nil, &models)
	if err != nil {
		return nil, fmt.Errorf("list models failed: %w", err)
	}

	return models, nil
}

// GetModel gets information about a specific model
func (c *Client) GetModel(ctx context.Context, modelID string) (*ModelInfo, error) {
	if modelID == "" {
		return nil, fmt.Errorf("modelID cannot be empty")
	}

	var model ModelInfo
	path := fmt.Sprintf("/v1/models/%s", url.PathEscape(modelID))
	err := c.makeRequest(ctx, "GET", path, nil, &model)
	if err != nil {
		return nil, fmt.Errorf("get model failed: %w", err)
	}

	return &model, nil
}

// GetMetrics gets server metrics
func (c *Client) GetMetrics(ctx context.Context) (map[string]interface{}, error) {
	var metrics map[string]interface{}
	err := c.makeRequest(ctx, "GET", "/metrics", nil, &metrics)
	if err != nil {
		return nil, fmt.Errorf("get metrics failed: %w", err)
	}

	return metrics, nil
}

// makeRequest is the core HTTP request method with retry logic
func (c *Client) makeRequest(ctx context.Context, method, path string, body interface{}, result interface{}) error {
	url := c.baseURL + path

	// Prepare request body
	var reqBody io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewBuffer(jsonData)
		
		if c.config.Debug {
			fmt.Printf("Request body: %s\n", string(jsonData))
		}
	}

	// Retry logic
	var lastErr error
	for attempt := 0; attempt <= c.retry.MaxRetries; attempt++ {
		if attempt > 0 {
			// Calculate delay with exponential backoff
			delay := time.Duration(float64(c.retry.InitialDelay) * 
				Math.Pow(c.retry.BackoffFactor, float64(attempt-1)))
			if delay > c.retry.MaxDelay {
				delay = c.retry.MaxDelay
			}
			
			if c.config.Debug {
				fmt.Printf("Retrying request after %v (attempt %d/%d)\n", delay, attempt+1, c.retry.MaxRetries+1)
			}
			
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
		}

		// Create HTTP request
		req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
		if err != nil {
			return fmt.Errorf("failed to create request: %w", err)
		}

		// Set default headers
		for k, v := range c.config.DefaultHeaders {
			req.Header.Set(k, v)
		}

		// Set user agent
		req.Header.Set("User-Agent", c.config.UserAgent)

		// Apply authentication
		if c.auth != nil {
			if err := c.auth.Apply(req); err != nil {
				return fmt.Errorf("failed to apply authentication: %w", err)
			}
		}

		if c.config.Debug {
			fmt.Printf("Making %s request to %s\n", method, url)
		}

		// Make the request
		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("request failed: %w", err)
			continue
		}

		// Read response body
		defer resp.Body.Close()
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			lastErr = fmt.Errorf("failed to read response body: %w", err)
			continue
		}

		if c.config.Debug {
			fmt.Printf("Response status: %d\n", resp.StatusCode)
			fmt.Printf("Response body: %s\n", string(respBody))
		}

		// Check if we should retry based on status code
		shouldRetry := false
		for _, code := range c.retry.RetryableErrors {
			if resp.StatusCode == code {
				shouldRetry = true
				break
			}
		}

		if shouldRetry && attempt < c.retry.MaxRetries {
			lastErr = fmt.Errorf("retryable error: status %d", resp.StatusCode)
			continue
		}

		// Handle HTTP errors
		if resp.StatusCode >= 400 {
			var errorResp map[string]interface{}
			if err := json.Unmarshal(respBody, &errorResp); err != nil {
				return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
			}
			
			if msg, ok := errorResp["error"].(string); ok {
				return fmt.Errorf("HTTP %d: %s", resp.StatusCode, msg)
			}
			return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
		}

		// Parse successful response
		if result != nil {
			if err := json.Unmarshal(respBody, result); err != nil {
				return fmt.Errorf("failed to parse response: %w", err)
			}
		}

		return nil
	}

	return fmt.Errorf("max retries exceeded: %w", lastErr)
}

// Authenticator interface for various authentication methods
type Authenticator interface {
	Apply(req *http.Request) error
}

// APIKeyAuth implements API key authentication
type APIKeyAuth struct {
	APIKey string
	Header string // Header name, defaults to "Authorization"
	Prefix string // Prefix for the key, defaults to "Bearer "
}

// NewAPIKeyAuth creates a new API key authenticator
func NewAPIKeyAuth(apiKey string) *APIKeyAuth {
	return &APIKeyAuth{
		APIKey: apiKey,
		Header: "Authorization",
		Prefix: "Bearer ",
	}
}

// Apply applies API key authentication to the request
func (a *APIKeyAuth) Apply(req *http.Request) error {
	if a.APIKey == "" {
		return fmt.Errorf("API key is empty")
	}
	req.Header.Set(a.Header, a.Prefix+a.APIKey)
	return nil
}

// JWTAuth implements JWT token authentication
type JWTAuth struct {
	Token  string
	Header string // Header name, defaults to "Authorization"
	Prefix string // Prefix for the token, defaults to "Bearer "
}

// NewJWTAuth creates a new JWT authenticator
func NewJWTAuth(token string) *JWTAuth {
	return &JWTAuth{
		Token:  token,
		Header: "Authorization",
		Prefix: "Bearer ",
	}
}

// Apply applies JWT authentication to the request
func (j *JWTAuth) Apply(req *http.Request) error {
	if j.Token == "" {
		return fmt.Errorf("JWT token is empty")
	}
	req.Header.Set(j.Header, j.Prefix+j.Token)
	return nil
}

// OAuth2Auth implements OAuth2 client credentials authentication
type OAuth2Auth struct {
	config *clientcredentials.Config
	client *http.Client
}

// NewOAuth2Auth creates a new OAuth2 authenticator
func NewOAuth2Auth(clientID, clientSecret, tokenURL string, scopes []string) *OAuth2Auth {
	config := &clientcredentials.Config{
		ClientID:     clientID,
		ClientSecret: clientSecret,
		TokenURL:     tokenURL,
		Scopes:       scopes,
	}
	return &OAuth2Auth{
		config: config,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

// Apply applies OAuth2 authentication to the request
func (o *OAuth2Auth) Apply(req *http.Request) error {
	ctx := context.WithValue(req.Context(), oauth2.HTTPClient, o.client)
	token, err := o.config.Token(ctx)
	if err != nil {
		return fmt.Errorf("failed to get OAuth2 token: %w", err)
	}
	token.SetAuthHeader(req)
	return nil
}

// CustomAuth allows for custom authentication implementations
type CustomAuth struct {
	ApplyFunc func(*http.Request) error
}

// NewCustomAuth creates a new custom authenticator
func NewCustomAuth(applyFunc func(*http.Request) error) *CustomAuth {
	return &CustomAuth{ApplyFunc: applyFunc}
}

// Apply applies custom authentication to the request
func (c *CustomAuth) Apply(req *http.Request) error {
	if c.ApplyFunc == nil {
		return fmt.Errorf("custom auth apply function is nil")
	}
	return c.ApplyFunc(req)
}

// StreamingResponse represents a streaming response
type StreamingResponse struct {
	Reader io.ReadCloser
	cancel context.CancelFunc
}

// Close closes the streaming response
func (s *StreamingResponse) Close() error {
	if s.cancel != nil {
		s.cancel()
	}
	return s.Reader.Close()
}

// ReadChunk reads a chunk from the streaming response
func (s *StreamingResponse) ReadChunk(buffer []byte) (int, error) {
	return s.Reader.Read(buffer)
}

// StreamInference performs streaming inference
func (c *Client) StreamInference(ctx context.Context, req *InferenceRequest) (*StreamingResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Enable streaming in the request
	if req.Options == nil {
		req.Options = &InferenceOptions{}
	}
	req.Options.Stream = true

	url := c.baseURL + "/v1/inference/stream"

	// Prepare request body
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	// Create context with cancellation
	streamCtx, cancel := context.WithCancel(ctx)

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(streamCtx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	for k, v := range c.config.DefaultHeaders {
		httpReq.Header.Set(k, v)
	}
	httpReq.Header.Set("User-Agent", c.config.UserAgent)
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("Cache-Control", "no-cache")

	// Apply authentication
	if c.auth != nil {
		if err := c.auth.Apply(httpReq); err != nil {
			cancel()
			return nil, fmt.Errorf("failed to apply authentication: %w", err)
		}
	}

	if c.config.Debug {
		fmt.Printf("Making streaming POST request to %s\n", url)
	}

	// Make the request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("streaming request failed: %w", err)
	}

	// Check for HTTP errors
	if resp.StatusCode >= 400 {
		cancel()
		resp.Body.Close()
		return nil, fmt.Errorf("HTTP %d: streaming request failed", resp.StatusCode)
	}

	return &StreamingResponse{
		Reader: resp.Body,
		cancel: cancel,
	}, nil
}

// WithTLSConfig sets custom TLS configuration
func WithTLSConfig(tlsConfig *tls.Config) ClientOption {
	return func(c *Client) error {
		if transport, ok := c.httpClient.Transport.(*http.Transport); ok {
			transport.TLSClientConfig = tlsConfig
		} else {
			c.httpClient.Transport = &http.Transport{
				TLSClientConfig: tlsConfig,
			}
		}
		return nil
	}
}

// Math is a simple math utility package since Go doesn't have math.Pow for int
type Math struct{}

// Pow calculates base^exp as a float64
func (Math) Pow(base, exp float64) float64 {
	result := 1.0
	for i := 0; i < int(exp); i++ {
		result *= base
	}
	return result
}