package com.trustformers.client;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import okhttp3.*;
import okhttp3.logging.HttpLoggingInterceptor;
import okio.BufferedSource;

import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.io.IOException;
import java.security.cert.X509Certificate;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * TrustformeRS Java Client Library
 * 
 * Provides comprehensive access to TrustformeRS serving infrastructure with features including:
 * - Synchronous and asynchronous inference
 * - Batch processing
 * - Streaming support
 * - Multiple authentication methods
 * - Health monitoring
 * - Model management
 * - Retry logic with exponential backoff
 * - Request/response logging and debugging
 */
public class TrustformersClient {
    
    private final String baseUrl;
    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final ClientConfig config;
    private final Authenticator authenticator;
    
    /**
     * Client configuration options
     */
    public static class ClientConfig {
        private Duration timeout = Duration.ofSeconds(30);
        private String userAgent = "trustformers-java-client/1.0.0";
        private boolean debug = false;
        private int maxRetries = 3;
        private Duration initialRetryDelay = Duration.ofMillis(100);
        private Duration maxRetryDelay = Duration.ofSeconds(5);
        private double backoffMultiplier = 2.0;
        private Map<String, String> defaultHeaders = new HashMap<>();
        private Set<Integer> retryableStatusCodes = Set.of(500, 502, 503, 504, 429);
        
        // Getters and setters
        public Duration getTimeout() { return timeout; }
        public ClientConfig setTimeout(Duration timeout) { this.timeout = timeout; return this; }
        
        public String getUserAgent() { return userAgent; }
        public ClientConfig setUserAgent(String userAgent) { this.userAgent = userAgent; return this; }
        
        public boolean isDebug() { return debug; }
        public ClientConfig setDebug(boolean debug) { this.debug = debug; return this; }
        
        public int getMaxRetries() { return maxRetries; }
        public ClientConfig setMaxRetries(int maxRetries) { this.maxRetries = maxRetries; return this; }
        
        public Duration getInitialRetryDelay() { return initialRetryDelay; }
        public ClientConfig setInitialRetryDelay(Duration initialRetryDelay) { this.initialRetryDelay = initialRetryDelay; return this; }
        
        public Duration getMaxRetryDelay() { return maxRetryDelay; }
        public ClientConfig setMaxRetryDelay(Duration maxRetryDelay) { this.maxRetryDelay = maxRetryDelay; return this; }
        
        public double getBackoffMultiplier() { return backoffMultiplier; }
        public ClientConfig setBackoffMultiplier(double backoffMultiplier) { this.backoffMultiplier = backoffMultiplier; return this; }
        
        public Map<String, String> getDefaultHeaders() { return defaultHeaders; }
        public ClientConfig setDefaultHeaders(Map<String, String> defaultHeaders) { this.defaultHeaders = defaultHeaders; return this; }
        
        public Set<Integer> getRetryableStatusCodes() { return retryableStatusCodes; }
        public ClientConfig setRetryableStatusCodes(Set<Integer> retryableStatusCodes) { this.retryableStatusCodes = retryableStatusCodes; return this; }
    }
    
    /**
     * Authentication interface
     */
    public interface Authenticator {
        void apply(Request.Builder requestBuilder);
    }
    
    /**
     * API Key authentication
     */
    public static class ApiKeyAuth implements Authenticator {
        private final String apiKey;
        private final String header;
        private final String prefix;
        
        public ApiKeyAuth(String apiKey) {
            this(apiKey, "Authorization", "Bearer ");
        }
        
        public ApiKeyAuth(String apiKey, String header, String prefix) {
            this.apiKey = apiKey;
            this.header = header;
            this.prefix = prefix;
        }
        
        @Override
        public void apply(Request.Builder requestBuilder) {
            requestBuilder.addHeader(header, prefix + apiKey);
        }
    }
    
    /**
     * JWT token authentication
     */
    public static class JwtAuth implements Authenticator {
        private final String token;
        private final String header;
        private final String prefix;
        
        public JwtAuth(String token) {
            this(token, "Authorization", "Bearer ");
        }
        
        public JwtAuth(String token, String header, String prefix) {
            this.token = token;
            this.header = header;
            this.prefix = prefix;
        }
        
        @Override
        public void apply(Request.Builder requestBuilder) {
            requestBuilder.addHeader(header, prefix + token);
        }
    }
    
    /**
     * Custom authentication
     */
    public static class CustomAuth implements Authenticator {
        private final Consumer<Request.Builder> applyFunction;
        
        public CustomAuth(Consumer<Request.Builder> applyFunction) {
            this.applyFunction = applyFunction;
        }
        
        @Override
        public void apply(Request.Builder requestBuilder) {
            applyFunction.accept(requestBuilder);
        }
    }
    
    /**
     * Inference request
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class InferenceRequest {
        private String input;
        private String modelId;
        private Map<String, Object> parameters;
        private InferenceOptions options;
        
        // Constructors
        public InferenceRequest() {}
        
        public InferenceRequest(String input) {
            this.input = input;
        }
        
        public InferenceRequest(String input, String modelId) {
            this.input = input;
            this.modelId = modelId;
        }
        
        // Getters and setters
        public String getInput() { return input; }
        public InferenceRequest setInput(String input) { this.input = input; return this; }
        
        public String getModelId() { return modelId; }
        public InferenceRequest setModelId(String modelId) { this.modelId = modelId; return this; }
        
        public Map<String, Object> getParameters() { return parameters; }
        public InferenceRequest setParameters(Map<String, Object> parameters) { this.parameters = parameters; return this; }
        
        public InferenceOptions getOptions() { return options; }
        public InferenceRequest setOptions(InferenceOptions options) { this.options = options; return this; }
    }
    
    /**
     * Inference options
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class InferenceOptions {
        private Integer maxTokens;
        private Double temperature;
        private Double topP;
        private Integer topK;
        private Boolean stream;
        
        // Getters and setters
        public Integer getMaxTokens() { return maxTokens; }
        public InferenceOptions setMaxTokens(Integer maxTokens) { this.maxTokens = maxTokens; return this; }
        
        public Double getTemperature() { return temperature; }
        public InferenceOptions setTemperature(Double temperature) { this.temperature = temperature; return this; }
        
        public Double getTopP() { return topP; }
        public InferenceOptions setTopP(Double topP) { this.topP = topP; return this; }
        
        public Integer getTopK() { return topK; }
        public InferenceOptions setTopK(Integer topK) { this.topK = topK; return this; }
        
        public Boolean getStream() { return stream; }
        public InferenceOptions setStream(Boolean stream) { this.stream = stream; return this; }
    }
    
    /**
     * Inference response
     */
    public static class InferenceResponse {
        private String id;
        private String object;
        private long created;
        private String model;
        private List<Choice> choices;
        private Usage usage;
        private Map<String, Object> metadata;
        private double processingTimeMs;
        
        // Getters and setters
        public String getId() { return id; }
        public void setId(String id) { this.id = id; }
        
        public String getObject() { return object; }
        public void setObject(String object) { this.object = object; }
        
        public long getCreated() { return created; }
        public void setCreated(long created) { this.created = created; }
        
        public String getModel() { return model; }
        public void setModel(String model) { this.model = model; }
        
        public List<Choice> getChoices() { return choices; }
        public void setChoices(List<Choice> choices) { this.choices = choices; }
        
        public Usage getUsage() { return usage; }
        public void setUsage(Usage usage) { this.usage = usage; }
        
        public Map<String, Object> getMetadata() { return metadata; }
        public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
        
        public double getProcessingTimeMs() { return processingTimeMs; }
        public void setProcessingTimeMs(double processingTimeMs) { this.processingTimeMs = processingTimeMs; }
    }
    
    /**
     * Choice in response
     */
    public static class Choice {
        private int index;
        private String text;
        private String finishReason;
        private Double confidence;
        
        // Getters and setters
        public int getIndex() { return index; }
        public void setIndex(int index) { this.index = index; }
        
        public String getText() { return text; }
        public void setText(String text) { this.text = text; }
        
        public String getFinishReason() { return finishReason; }
        public void setFinishReason(String finishReason) { this.finishReason = finishReason; }
        
        public Double getConfidence() { return confidence; }
        public void setConfidence(Double confidence) { this.confidence = confidence; }
    }
    
    /**
     * Token usage information
     */
    public static class Usage {
        private int promptTokens;
        private int completionTokens;
        private int totalTokens;
        
        // Getters and setters
        public int getPromptTokens() { return promptTokens; }
        public void setPromptTokens(int promptTokens) { this.promptTokens = promptTokens; }
        
        public int getCompletionTokens() { return completionTokens; }
        public void setCompletionTokens(int completionTokens) { this.completionTokens = completionTokens; }
        
        public int getTotalTokens() { return totalTokens; }
        public void setTotalTokens(int totalTokens) { this.totalTokens = totalTokens; }
    }
    
    /**
     * Batch inference request
     */
    public static class BatchInferenceRequest {
        private List<InferenceRequest> requests;
        private BatchOptions options;
        
        public BatchInferenceRequest() {}
        
        public BatchInferenceRequest(List<InferenceRequest> requests) {
            this.requests = requests;
        }
        
        // Getters and setters
        public List<InferenceRequest> getRequests() { return requests; }
        public BatchInferenceRequest setRequests(List<InferenceRequest> requests) { this.requests = requests; return this; }
        
        public BatchOptions getOptions() { return options; }
        public BatchInferenceRequest setOptions(BatchOptions options) { this.options = options; return this; }
    }
    
    /**
     * Batch processing options
     */
    public static class BatchOptions {
        private Integer maxBatchSize;
        private Boolean parallel;
        
        // Getters and setters
        public Integer getMaxBatchSize() { return maxBatchSize; }
        public BatchOptions setMaxBatchSize(Integer maxBatchSize) { this.maxBatchSize = maxBatchSize; return this; }
        
        public Boolean getParallel() { return parallel; }
        public BatchOptions setParallel(Boolean parallel) { this.parallel = parallel; return this; }
    }
    
    /**
     * Batch inference response
     */
    public static class BatchInferenceResponse {
        private String id;
        private String object;
        private long created;
        private List<InferenceResponse> responses;
        private int batchSize;
        
        // Getters and setters
        public String getId() { return id; }
        public void setId(String id) { this.id = id; }
        
        public String getObject() { return object; }
        public void setObject(String object) { this.object = object; }
        
        public long getCreated() { return created; }
        public void setCreated(long created) { this.created = created; }
        
        public List<InferenceResponse> getResponses() { return responses; }
        public void setResponses(List<InferenceResponse> responses) { this.responses = responses; }
        
        public int getBatchSize() { return batchSize; }
        public void setBatchSize(int batchSize) { this.batchSize = batchSize; }
    }
    
    /**
     * Health status
     */
    public static class HealthStatus {
        private String status;
        private Instant timestamp;
        private String version;
        private double uptime;
        private Map<String, Object> details;
        private Map<String, String> components;
        
        // Getters and setters
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        
        public Instant getTimestamp() { return timestamp; }
        public void setTimestamp(Instant timestamp) { this.timestamp = timestamp; }
        
        public String getVersion() { return version; }
        public void setVersion(String version) { this.version = version; }
        
        public double getUptime() { return uptime; }
        public void setUptime(double uptime) { this.uptime = uptime; }
        
        public Map<String, Object> getDetails() { return details; }
        public void setDetails(Map<String, Object> details) { this.details = details; }
        
        public Map<String, String> getComponents() { return components; }
        public void setComponents(Map<String, String> components) { this.components = components; }
    }
    
    /**
     * Model information
     */
    public static class ModelInfo {
        private String id;
        private String name;
        private String version;
        private String description;
        private String architecture;
        private long parameters;
        private Map<String, Object> metadata;
        private String status;
        private Instant createdAt;
        private Instant updatedAt;
        
        // Getters and setters
        public String getId() { return id; }
        public void setId(String id) { this.id = id; }
        
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        
        public String getVersion() { return version; }
        public void setVersion(String version) { this.version = version; }
        
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        
        public String getArchitecture() { return architecture; }
        public void setArchitecture(String architecture) { this.architecture = architecture; }
        
        public long getParameters() { return parameters; }
        public void setParameters(long parameters) { this.parameters = parameters; }
        
        public Map<String, Object> getMetadata() { return metadata; }
        public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
        
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        
        public Instant getCreatedAt() { return createdAt; }
        public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
        
        public Instant getUpdatedAt() { return updatedAt; }
        public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
    }
    
    /**
     * Client builder
     */
    public static class Builder {
        private String baseUrl;
        private ClientConfig config = new ClientConfig();
        private Authenticator authenticator;
        private OkHttpClient.Builder httpClientBuilder = new OkHttpClient.Builder();
        
        public Builder(String baseUrl) {
            this.baseUrl = baseUrl;
        }
        
        public Builder config(ClientConfig config) {
            this.config = config;
            return this;
        }
        
        public Builder authenticator(Authenticator authenticator) {
            this.authenticator = authenticator;
            return this;
        }
        
        public Builder httpClientBuilder(OkHttpClient.Builder httpClientBuilder) {
            this.httpClientBuilder = httpClientBuilder;
            return this;
        }
        
        public Builder timeout(Duration timeout) {
            this.config.setTimeout(timeout);
            return this;
        }
        
        public Builder userAgent(String userAgent) {
            this.config.setUserAgent(userAgent);
            return this;
        }
        
        public Builder debug(boolean debug) {
            this.config.setDebug(debug);
            return this;
        }
        
        public Builder maxRetries(int maxRetries) {
            this.config.setMaxRetries(maxRetries);
            return this;
        }
        
        public Builder apiKey(String apiKey) {
            this.authenticator = new ApiKeyAuth(apiKey);
            return this;
        }
        
        public Builder jwtToken(String token) {
            this.authenticator = new JwtAuth(token);
            return this;
        }
        
        public Builder insecureTls() {
            try {
                TrustManager[] trustAllCerts = new TrustManager[] {
                    new X509TrustManager() {
                        @Override
                        public void checkClientTrusted(X509Certificate[] chain, String authType) {}\
                        
                        @Override
                        public void checkServerTrusted(X509Certificate[] chain, String authType) {}
                        
                        @Override
                        public X509Certificate[] getAcceptedIssuers() { return new X509Certificate[]{}; }
                    }
                };
                
                SSLContext sslContext = SSLContext.getInstance("SSL");
                sslContext.init(null, trustAllCerts, new java.security.SecureRandom());
                
                this.httpClientBuilder.sslSocketFactory(sslContext.getSocketFactory(), (X509TrustManager) trustAllCerts[0]);
                this.httpClientBuilder.hostnameVerifier((hostname, session) -> true);
            } catch (Exception e) {
                throw new RuntimeException("Failed to configure insecure TLS", e);
            }
            return this;
        }
        
        public TrustformersClient build() {
            return new TrustformersClient(baseUrl, config, authenticator, httpClientBuilder);
        }
    }
    
    /**
     * Create a new client builder
     */
    public static Builder builder(String baseUrl) {
        return new Builder(baseUrl);
    }
    
    /**
     * Constructor
     */
    private TrustformersClient(String baseUrl, ClientConfig config, Authenticator authenticator, OkHttpClient.Builder httpClientBuilder) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.config = config;
        this.authenticator = authenticator;
        
        // Configure HTTP client
        httpClientBuilder.connectTimeout(config.getTimeout())
                        .readTimeout(config.getTimeout())
                        .writeTimeout(config.getTimeout());
        
        // Add logging if debug is enabled
        if (config.isDebug()) {
            HttpLoggingInterceptor loggingInterceptor = new HttpLoggingInterceptor(System.out::println);
            loggingInterceptor.setLevel(HttpLoggingInterceptor.Level.BODY);
            httpClientBuilder.addInterceptor(loggingInterceptor);
        }
        
        // Add retry interceptor
        httpClientBuilder.addInterceptor(new RetryInterceptor());
        
        this.httpClient = httpClientBuilder.build();
        
        // Configure object mapper
        this.objectMapper = new ObjectMapper()
                .registerModule(new JavaTimeModule())
                .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
                .setSerializationInclusion(JsonInclude.Include.NON_NULL);
    }
    
    /**
     * Retry interceptor for handling retryable errors
     */
    private class RetryInterceptor implements Interceptor {
        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();
            Response response = null;
            IOException lastException = null;
            
            for (int attempt = 0; attempt <= config.getMaxRetries(); attempt++) {
                if (attempt > 0) {
                    // Calculate delay with exponential backoff
                    long delayMs = Math.min(
                        (long) (config.getInitialRetryDelay().toMillis() * Math.pow(config.getBackoffMultiplier(), attempt - 1)),
                        config.getMaxRetryDelay().toMillis()
                    );
                    
                    try {
                        Thread.sleep(delayMs);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new IOException("Interrupted during retry delay", e);
                    }
                }
                
                try {
                    if (response != null) {
                        response.close();
                    }
                    response = chain.proceed(request);
                    
                    // Check if we should retry based on status code
                    if (!config.getRetryableStatusCodes().contains(response.code()) || attempt == config.getMaxRetries()) {
                        return response;
                    }
                    
                } catch (IOException e) {
                    lastException = e;
                    if (attempt == config.getMaxRetries()) {
                        throw e;
                    }
                }
            }
            
            if (lastException != null) {
                throw lastException;
            }
            
            return response;
        }
    }
    
    /**
     * Perform inference
     */
    public InferenceResponse inference(InferenceRequest request) throws TrustformersException {
        try {
            String json = objectMapper.writeValueAsString(request);
            RequestBody body = RequestBody.create(json, MediaType.get("application/json; charset=utf-8"));
            
            Request httpRequest = buildRequest("/v1/inference")
                    .post(body)
                    .build();
            
            try (Response response = httpClient.newCall(httpRequest).execute()) {
                return handleResponse(response, InferenceResponse.class);
            }
        } catch (Exception e) {
            throw new TrustformersException("Inference request failed", e);
        }
    }
    
    /**
     * Perform inference asynchronously
     */
    public CompletableFuture<InferenceResponse> inferenceAsync(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return inference(request);
            } catch (TrustformersException e) {
                throw new CompletionException(e);
            }
        });
    }
    
    /**
     * Perform batch inference
     */
    public BatchInferenceResponse batchInference(BatchInferenceRequest request) throws TrustformersException {
        try {
            String json = objectMapper.writeValueAsString(request);
            RequestBody body = RequestBody.create(json, MediaType.get("application/json; charset=utf-8"));
            
            Request httpRequest = buildRequest("/v1/inference/batch")
                    .post(body)
                    .build();
            
            try (Response response = httpClient.newCall(httpRequest).execute()) {
                return handleResponse(response, BatchInferenceResponse.class);
            }
        } catch (Exception e) {
            throw new TrustformersException("Batch inference request failed", e);
        }
    }
    
    /**
     * Perform batch inference asynchronously
     */
    public CompletableFuture<BatchInferenceResponse> batchInferenceAsync(BatchInferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return batchInference(request);
            } catch (TrustformersException e) {
                throw new CompletionException(e);
            }
        });
    }
    
    /**
     * Check health status
     */
    public HealthStatus health() throws TrustformersException {
        try {
            Request httpRequest = buildRequest("/health")
                    .get()
                    .build();
            
            try (Response response = httpClient.newCall(httpRequest).execute()) {
                return handleResponse(response, HealthStatus.class);
            }
        } catch (Exception e) {
            throw new TrustformersException("Health check failed", e);
        }
    }
    
    /**
     * Get detailed health status
     */
    public HealthStatus detailedHealth() throws TrustformersException {
        try {
            Request httpRequest = buildRequest("/health/detailed")
                    .get()
                    .build();
            
            try (Response response = httpClient.newCall(httpRequest).execute()) {
                return handleResponse(response, HealthStatus.class);
            }
        } catch (Exception e) {
            throw new TrustformersException("Detailed health check failed", e);
        }
    }
    
    /**
     * List available models
     */
    public List<ModelInfo> listModels() throws TrustformersException {
        try {
            Request httpRequest = buildRequest("/v1/models")
                    .get()
                    .build();
            
            try (Response response = httpClient.newCall(httpRequest).execute()) {
                return handleResponse(response, objectMapper.getTypeFactory().constructCollectionType(List.class, ModelInfo.class));
            }
        } catch (Exception e) {
            throw new TrustformersException("List models failed", e);
        }
    }
    
    /**
     * Get model information
     */
    public ModelInfo getModel(String modelId) throws TrustformersException {
        try {
            Request httpRequest = buildRequest("/v1/models/" + modelId)
                    .get()
                    .build();
            
            try (Response response = httpClient.newCall(httpRequest).execute()) {
                return handleResponse(response, ModelInfo.class);
            }
        } catch (Exception e) {
            throw new TrustformersException("Get model failed", e);
        }
    }
    
    /**
     * Get server metrics
     */
    public Map<String, Object> getMetrics() throws TrustformersException {
        try {
            Request httpRequest = buildRequest("/metrics")
                    .get()
                    .build();
            
            try (Response response = httpClient.newCall(httpRequest).execute()) {
                return handleResponse(response, objectMapper.getTypeFactory().constructMapType(Map.class, String.class, Object.class));
            }
        } catch (Exception e) {
            throw new TrustformersException("Get metrics failed", e);
        }
    }
    
    /**
     * Streaming inference
     */
    public void streamInference(InferenceRequest request, Consumer<String> onChunk, Consumer<Throwable> onError) {
        CompletableFuture.runAsync(() -> {
            try {
                // Enable streaming
                if (request.getOptions() == null) {
                    request.setOptions(new InferenceOptions());
                }
                request.getOptions().setStream(true);
                
                String json = objectMapper.writeValueAsString(request);
                RequestBody body = RequestBody.create(json, MediaType.get("application/json; charset=utf-8"));
                
                Request httpRequest = buildRequest("/v1/inference/stream")
                        .post(body)
                        .addHeader("Accept", "text/event-stream")
                        .addHeader("Cache-Control", "no-cache")
                        .build();
                
                try (Response response = httpClient.newCall(httpRequest).execute()) {
                    if (!response.isSuccessful()) {
                        onError.accept(new TrustformersException("Streaming request failed: " + response.code()));
                        return;
                    }
                    
                    ResponseBody responseBody = response.body();
                    if (responseBody == null) {
                        onError.accept(new TrustformersException("Empty response body"));
                        return;
                    }
                    
                    BufferedSource source = responseBody.source();
                    String line;
                    while ((line = source.readUtf8Line()) != null) {
                        if (!line.trim().isEmpty()) {
                            onChunk.accept(line);
                        }
                    }
                }
            } catch (Exception e) {
                onError.accept(new TrustformersException("Streaming inference failed", e));
            }
        });
    }
    
    /**
     * Build request with common headers and authentication
     */
    private Request.Builder buildRequest(String path) {
        Request.Builder builder = new Request.Builder()
                .url(baseUrl + path)
                .addHeader("User-Agent", config.getUserAgent())
                .addHeader("Content-Type", "application/json")
                .addHeader("Accept", "application/json");
        
        // Add default headers
        config.getDefaultHeaders().forEach(builder::addHeader);
        
        // Apply authentication
        if (authenticator != null) {
            authenticator.apply(builder);
        }
        
        return builder;
    }
    
    /**
     * Handle HTTP response
     */
    private <T> T handleResponse(Response response, Class<T> responseType) throws TrustformersException, IOException {
        if (!response.isSuccessful()) {
            String errorBody = response.body() != null ? response.body().string() : "";
            throw new TrustformersException("HTTP " + response.code() + ": " + errorBody);
        }
        
        ResponseBody body = response.body();
        if (body == null) {
            throw new TrustformersException("Empty response body");
        }
        
        String json = body.string();
        try {
            return objectMapper.readValue(json, responseType);
        } catch (JsonProcessingException e) {
            throw new TrustformersException("Failed to parse response: " + json, e);
        }
    }
    
    /**
     * Handle HTTP response with type reference
     */
    private <T> T handleResponse(Response response, com.fasterxml.jackson.core.type.TypeReference<T> typeRef) throws TrustformersException, IOException {
        if (!response.isSuccessful()) {
            String errorBody = response.body() != null ? response.body().string() : "";
            throw new TrustformersException("HTTP " + response.code() + ": " + errorBody);
        }
        
        ResponseBody body = response.body();
        if (body == null) {
            throw new TrustformersException("Empty response body");
        }
        
        String json = body.string();
        try {
            return objectMapper.readValue(json, typeRef);
        } catch (JsonProcessingException e) {
            throw new TrustformersException("Failed to parse response: " + json, e);
        }
    }
    
    /**
     * Close the client and release resources
     */
    public void close() {
        httpClient.dispatcher().executorService().shutdown();
        httpClient.connectionPool().evictAll();
        
        try {
            if (!httpClient.dispatcher().executorService().awaitTermination(5, TimeUnit.SECONDS)) {
                httpClient.dispatcher().executorService().shutdownNow();
            }
        } catch (InterruptedException e) {
            httpClient.dispatcher().executorService().shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    /**
     * Client exception
     */
    public static class TrustformersException extends Exception {
        public TrustformersException(String message) {
            super(message);
        }
        
        public TrustformersException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}