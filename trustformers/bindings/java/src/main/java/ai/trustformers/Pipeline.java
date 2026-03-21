package ai.trustformers;

import java.util.Map;
import java.util.List;
import java.util.HashMap;

/**
 * TrustFormeRS Pipeline wrapper providing high-level ML pipeline functionality
 * for various tasks like text generation, classification, question answering, etc.
 */
public class Pipeline implements AutoCloseable {
    
    private long nativeHandle;
    private boolean closed = false;
    private String taskType;
    
    // Native method declarations
    private static native long createPipelineNative(String task, String modelPath, String tokenizerPath, Map<String, Object> config);
    private static native long createPipelineFromHubNative(String task, String modelName, String revision, String authToken, Map<String, Object> config);
    private static native void unloadPipelineNative(long handle);
    private static native Map<String, Object> processNative(long handle, String input, Map<String, Object> config);
    private static native List<Map<String, Object>> processBatchNative(long handle, String[] inputs, Map<String, Object> config);
    private static native String getPipelineInfoNative(long handle);
    private static native Map<String, Object> getPipelineConfigNative(long handle);
    private static native boolean savePipelineNative(long handle, String savePath);
    
    // Supported pipeline tasks
    public static final String TEXT_GENERATION = "text-generation";
    public static final String TEXT_CLASSIFICATION = "text-classification";
    public static final String QUESTION_ANSWERING = "question-answering";
    public static final String SUMMARIZATION = "summarization";
    public static final String TRANSLATION = "translation";
    public static final String FILL_MASK = "fill-mask";
    public static final String TOKEN_CLASSIFICATION = "token-classification";
    public static final String CONVERSATIONAL = "conversational";
    public static final String TEXT_TO_SPEECH = "text-to-speech";
    public static final String SPEECH_TO_TEXT = "speech-to-text";
    public static final String IMAGE_TO_TEXT = "image-to-text";
    public static final String VISUAL_QUESTION_ANSWERING = "visual-question-answering";
    
    /**
     * Create a pipeline from local model and tokenizer paths
     * @param task Pipeline task type (e.g., "text-generation", "text-classification")
     * @param modelPath Path to the model directory or file
     * @param tokenizerPath Path to the tokenizer directory or file (can be null)
     * @param config Pipeline configuration (can be null for defaults)
     * @throws TrustFormeRSException if pipeline creation fails
     */
    public Pipeline(String task, String modelPath, String tokenizerPath, Map<String, Object> config) throws TrustFormeRSException {
        this.taskType = task;
        this.nativeHandle = createPipelineNative(task, modelPath, tokenizerPath, config);
        if (this.nativeHandle == 0) {
            throw new TrustFormeRSException("Failed to create pipeline for task: " + task);
        }
    }
    
    /**
     * Create a pipeline from local model path with default configuration
     * @param task Pipeline task type
     * @param modelPath Path to the model directory or file
     * @throws TrustFormeRSException if pipeline creation fails
     */
    public Pipeline(String task, String modelPath) throws TrustFormeRSException {
        this(task, modelPath, null, null);
    }
    
    /**
     * Create a pipeline from Hugging Face Hub
     * @param task Pipeline task type
     * @param modelName Model name from Hub
     * @param revision Model revision/branch (can be null for default)
     * @param authToken Authentication token (can be null for public models)
     * @param config Pipeline configuration (can be null for defaults)
     * @throws TrustFormeRSException if pipeline creation fails
     */
    public static Pipeline fromHub(String task, String modelName, String revision, String authToken, Map<String, Object> config) throws TrustFormeRSException {
        Pipeline pipeline = new Pipeline();
        pipeline.taskType = task;
        pipeline.nativeHandle = createPipelineFromHubNative(task, modelName, revision, authToken, config);
        if (pipeline.nativeHandle == 0) {
            throw new TrustFormeRSException("Failed to create pipeline from Hub: " + modelName);
        }
        return pipeline;
    }
    
    /**
     * Create a pipeline from Hugging Face Hub with defaults
     * @param task Pipeline task type
     * @param modelName Model name from Hub
     * @throws TrustFormeRSException if pipeline creation fails
     */
    public static Pipeline fromHub(String task, String modelName) throws TrustFormeRSException {
        return fromHub(task, modelName, null, null, null);
    }
    
    /**
     * Private constructor for fromHub factory methods
     */
    private Pipeline() {
        // Used by fromHub factory methods
    }
    
    /**
     * Process a single input through the pipeline
     * @param input Input text or data
     * @param config Processing configuration (can be null for defaults)
     * @return Processing result as Map
     * @throws TrustFormeRSException if processing fails
     */
    public Map<String, Object> process(String input, Map<String, Object> config) throws TrustFormeRSException {
        checkNotClosed();
        Map<String, Object> result = processNative(nativeHandle, input, config);
        if (result == null) {
            throw new TrustFormeRSException("Pipeline processing failed");
        }
        return result;
    }
    
    /**
     * Process a single input with default configuration
     * @param input Input text or data
     * @return Processing result as Map
     * @throws TrustFormeRSException if processing fails
     */
    public Map<String, Object> process(String input) throws TrustFormeRSException {
        return process(input, null);
    }
    
    /**
     * Process multiple inputs in batch
     * @param inputs Array of input texts or data
     * @param config Processing configuration (can be null for defaults)
     * @return List of processing results
     * @throws TrustFormeRSException if batch processing fails
     */
    public List<Map<String, Object>> processBatch(String[] inputs, Map<String, Object> config) throws TrustFormeRSException {
        checkNotClosed();
        List<Map<String, Object>> results = processBatchNative(nativeHandle, inputs, config);
        if (results == null) {
            throw new TrustFormeRSException("Pipeline batch processing failed");
        }
        return results;
    }
    
    /**
     * Process multiple inputs in batch with default configuration
     * @param inputs Array of input texts or data
     * @return List of processing results
     * @throws TrustFormeRSException if batch processing fails
     */
    public List<Map<String, Object>> processBatch(String[] inputs) throws TrustFormeRSException {
        return processBatch(inputs, null);
    }
    
    // Convenience methods for specific tasks
    
    /**
     * Generate text (for text-generation pipelines)
     * @param prompt Input prompt
     * @param maxLength Maximum generation length
     * @param temperature Sampling temperature
     * @return Generated text
     * @throws TrustFormeRSException if not a text-generation pipeline or generation fails
     */
    public String generateText(String prompt, int maxLength, double temperature) throws TrustFormeRSException {
        if (!TEXT_GENERATION.equals(taskType)) {
            throw new TrustFormeRSException("generateText() can only be called on text-generation pipelines");
        }
        
        Map<String, Object> config = new HashMap<>();
        config.put("max_length", maxLength);
        config.put("temperature", temperature);
        config.put("do_sample", true);
        
        Map<String, Object> result = process(prompt, config);
        return (String) result.get("generated_text");
    }
    
    /**
     * Classify text (for text-classification pipelines)
     * @param text Input text to classify
     * @return Classification results with labels and scores
     * @throws TrustFormeRSException if not a text-classification pipeline or classification fails
     */
    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> classifyText(String text) throws TrustFormeRSException {
        if (!TEXT_CLASSIFICATION.equals(taskType)) {
            throw new TrustFormeRSException("classifyText() can only be called on text-classification pipelines");
        }
        
        Map<String, Object> result = process(text);
        return (List<Map<String, Object>>) result.get("classifications");
    }
    
    /**
     * Answer question (for question-answering pipelines)
     * @param question Question text
     * @param context Context text containing the answer
     * @return Answer result with text, score, start, and end positions
     * @throws TrustFormeRSException if not a question-answering pipeline or answering fails
     */
    public Map<String, Object> answerQuestion(String question, String context) throws TrustFormeRSException {
        if (!QUESTION_ANSWERING.equals(taskType)) {
            throw new TrustFormeRSException("answerQuestion() can only be called on question-answering pipelines");
        }
        
        Map<String, Object> config = new HashMap<>();
        config.put("question", question);
        config.put("context", context);
        
        return process(question, config);
    }
    
    /**
     * Summarize text (for summarization pipelines)
     * @param text Input text to summarize
     * @param maxLength Maximum summary length
     * @param minLength Minimum summary length
     * @return Summary text
     * @throws TrustFormeRSException if not a summarization pipeline or summarization fails
     */
    public String summarizeText(String text, int maxLength, int minLength) throws TrustFormeRSException {
        if (!SUMMARIZATION.equals(taskType)) {
            throw new TrustFormeRSException("summarizeText() can only be called on summarization pipelines");
        }
        
        Map<String, Object> config = new HashMap<>();
        config.put("max_length", maxLength);
        config.put("min_length", minLength);
        
        Map<String, Object> result = process(text, config);
        return (String) result.get("summary_text");
    }
    
    /**
     * Get the pipeline task type
     * @return Task type string
     */
    public String getTaskType() {
        return taskType;
    }
    
    /**
     * Get pipeline information
     * @return Pipeline information string
     * @throws TrustFormeRSException if pipeline is closed
     */
    public String getPipelineInfo() throws TrustFormeRSException {
        checkNotClosed();
        return getPipelineInfoNative(nativeHandle);
    }
    
    /**
     * Get pipeline configuration
     * @return Pipeline configuration as Map
     * @throws TrustFormeRSException if pipeline is closed
     */
    public Map<String, Object> getPipelineConfig() throws TrustFormeRSException {
        checkNotClosed();
        return getPipelineConfigNative(nativeHandle);
    }
    
    /**
     * Save the pipeline to a local path
     * @param savePath Path to save the pipeline
     * @return true if save was successful
     * @throws TrustFormeRSException if pipeline is closed or save fails
     */
    public boolean savePipeline(String savePath) throws TrustFormeRSException {
        checkNotClosed();
        return savePipelineNative(nativeHandle, savePath);
    }
    
    /**
     * Check if the pipeline is closed
     * @return true if the pipeline has been closed
     */
    public boolean isClosed() {
        return closed;
    }
    
    /**
     * Close the pipeline and free native resources
     */
    @Override
    public void close() {
        if (!closed && nativeHandle != 0) {
            unloadPipelineNative(nativeHandle);
            nativeHandle = 0;
            closed = true;
        }
    }
    
    /**
     * Finalize method to ensure native resources are cleaned up
     */
    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }
    
    /**
     * Check that the pipeline is not closed and throw exception if it is
     * @throws TrustFormeRSException if pipeline is closed
     */
    private void checkNotClosed() throws TrustFormeRSException {
        if (closed) {
            throw new TrustFormeRSException("Pipeline has been closed");
        }
    }
}