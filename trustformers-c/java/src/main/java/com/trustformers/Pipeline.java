package com.trustformers;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * High-level pipeline interface for various NLP tasks.
 * Provides easy-to-use methods for text generation, classification, question answering, etc.
 */
public class Pipeline implements Closeable {
    
    private static final Logger logger = LoggerFactory.getLogger(Pipeline.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    
    private final TrustformeRS trustformers;
    private final PipelineType type;
    private long nativeHandle;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    
    /**
     * Supported pipeline types.
     */
    public enum PipelineType {
        TEXT_GENERATION("text-generation"),
        TEXT_CLASSIFICATION("text-classification"),
        QUESTION_ANSWERING("question-answering"),
        CONVERSATIONAL("conversational");
        
        private final String typeName;
        
        PipelineType(String typeName) {
            this.typeName = typeName;
        }
        
        public String getTypeName() {
            return typeName;
        }
    }
    
    /**
     * Text generation options.
     */
    public static class GenerationOptions {
        public int maxLength = 50;
        public int minLength = 1;
        public double temperature = 1.0;
        public int topK = 50;
        public double topP = 1.0;
        public double repetitionPenalty = 1.0;
        public boolean doSample = false;
        public boolean earlyStopping = false;
        public int numBeams = 1;
        public int numReturnSequences = 1;
        
        public GenerationOptions() {}
        
        public GenerationOptions maxLength(int maxLength) {
            this.maxLength = maxLength;
            return this;
        }
        
        public GenerationOptions temperature(double temperature) {
            this.temperature = temperature;
            return this;
        }
        
        public GenerationOptions topK(int topK) {
            this.topK = topK;
            return this;
        }
        
        public GenerationOptions topP(double topP) {
            this.topP = topP;
            return this;
        }
        
        public GenerationOptions doSample(boolean doSample) {
            this.doSample = doSample;
            return this;
        }
        
        public String toJson() throws TrustformersException {
            try {
                return objectMapper.writeValueAsString(this);
            } catch (IOException e) {
                throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                    "Failed to serialize generation options", e);
            }
        }
    }
    
    /**
     * Text classification result.
     */
    public static class ClassificationResult {
        public final String label;
        public final double score;
        
        public ClassificationResult(String label, double score) {
            this.label = label;
            this.score = score;
        }
        
        @Override
        public String toString() {
            return String.format("ClassificationResult{label='%s', score=%.4f}", label, score);
        }
    }
    
    /**
     * Question answering result.
     */
    public static class AnswerResult {
        public final String answer;
        public final double score;
        public final int start;
        public final int end;
        
        public AnswerResult(String answer, double score, int start, int end) {
            this.answer = answer;
            this.score = score;
            this.start = start;
            this.end = end;
        }
        
        @Override
        public String toString() {
            return String.format("AnswerResult{answer='%s', score=%.4f, start=%d, end=%d}", 
                answer, score, start, end);
        }
    }
    
    /**
     * Conversation turn.
     */
    public static class ConversationTurn {
        public final String userInput;
        public final String botResponse;
        public final long timestamp;
        
        public ConversationTurn(String userInput, String botResponse, long timestamp) {
            this.userInput = userInput;
            this.botResponse = botResponse;
            this.timestamp = timestamp;
        }
        
        @Override
        public String toString() {
            return String.format("ConversationTurn{userInput='%s', botResponse='%s', timestamp=%d}",
                userInput, botResponse, timestamp);
        }
    }
    
    /**
     * Pipeline information.
     */
    public static class PipelineInfo {
        public final String type;
        public final String modelName;
        public final String tokenizerName;
        public final String[] capabilities;
        public final JsonNode metadata;
        
        public PipelineInfo(String type, String modelName, String tokenizerName, 
                          String[] capabilities, JsonNode metadata) {
            this.type = type;
            this.modelName = modelName;
            this.tokenizerName = tokenizerName;
            this.capabilities = capabilities;
            this.metadata = metadata;
        }
        
        @Override
        public String toString() {
            return String.format("PipelineInfo{type='%s', modelName='%s', tokenizerName='%s', capabilities=%s}",
                type, modelName, tokenizerName, Arrays.toString(capabilities));
        }
    }
    
    // Package-private constructor
    Pipeline(TrustformeRS trustformers, PipelineType type, long nativeHandle) {
        this.trustformers = trustformers;
        this.type = type;
        this.nativeHandle = nativeHandle;
    }
    
    /**
     * Create a text generation pipeline.
     * 
     * @param trustformers TrustformeRS instance
     * @param model loaded model
     * @param tokenizer loaded tokenizer
     * @return text generation pipeline
     * @throws TrustformersException if creation fails
     */
    public static Pipeline createTextGeneration(TrustformeRS trustformers, Model model, Tokenizer tokenizer) 
            throws TrustformersException {
        validateInputs(model, tokenizer);
        
        long handle = nativeCreateTextGeneration(model.getNativeHandle(), tokenizer.getNativeHandle());
        if (handle == 0) {
            throw new TrustformersException(TrustformersException.ErrorCode.RUNTIME_ERROR,
                "Failed to create text generation pipeline");
        }
        
        Pipeline pipeline = new Pipeline(trustformers, PipelineType.TEXT_GENERATION, handle);
        logger.info("Created text generation pipeline");
        return pipeline;
    }
    
    /**
     * Create a text classification pipeline.
     * 
     * @param trustformers TrustformeRS instance
     * @param model loaded model
     * @param tokenizer loaded tokenizer
     * @return text classification pipeline
     * @throws TrustformersException if creation fails
     */
    public static Pipeline createTextClassification(TrustformeRS trustformers, Model model, Tokenizer tokenizer) 
            throws TrustformersException {
        validateInputs(model, tokenizer);
        
        long handle = nativeCreateTextClassification(model.getNativeHandle(), tokenizer.getNativeHandle());
        if (handle == 0) {
            throw new TrustformersException(TrustformersException.ErrorCode.RUNTIME_ERROR,
                "Failed to create text classification pipeline");
        }
        
        Pipeline pipeline = new Pipeline(trustformers, PipelineType.TEXT_CLASSIFICATION, handle);
        logger.info("Created text classification pipeline");
        return pipeline;
    }
    
    /**
     * Create a question answering pipeline.
     * 
     * @param trustformers TrustformeRS instance
     * @param model loaded model
     * @param tokenizer loaded tokenizer
     * @return question answering pipeline
     * @throws TrustformersException if creation fails
     */
    public static Pipeline createQuestionAnswering(TrustformeRS trustformers, Model model, Tokenizer tokenizer) 
            throws TrustformersException {
        validateInputs(model, tokenizer);
        
        long handle = nativeCreateQuestionAnswering(model.getNativeHandle(), tokenizer.getNativeHandle());
        if (handle == 0) {
            throw new TrustformersException(TrustformersException.ErrorCode.RUNTIME_ERROR,
                "Failed to create question answering pipeline");
        }
        
        Pipeline pipeline = new Pipeline(trustformers, PipelineType.QUESTION_ANSWERING, handle);
        logger.info("Created question answering pipeline");
        return pipeline;
    }
    
    /**
     * Create a conversational pipeline.
     * 
     * @param trustformers TrustformeRS instance
     * @param model loaded model
     * @param tokenizer loaded tokenizer
     * @return conversational pipeline
     * @throws TrustformersException if creation fails
     */
    public static Pipeline createConversational(TrustformeRS trustformers, Model model, Tokenizer tokenizer) 
            throws TrustformersException {
        validateInputs(model, tokenizer);
        
        long handle = nativeCreateConversational(model.getNativeHandle(), tokenizer.getNativeHandle());
        if (handle == 0) {
            throw new TrustformersException(TrustformersException.ErrorCode.RUNTIME_ERROR,
                "Failed to create conversational pipeline");
        }
        
        Pipeline pipeline = new Pipeline(trustformers, PipelineType.CONVERSATIONAL, handle);
        logger.info("Created conversational pipeline");
        return pipeline;
    }
    
    /**
     * Free the pipeline and release its resources.
     * 
     * @throws TrustformersException if freeing fails
     */
    public void free() throws TrustformersException {
        if (closed.get() || nativeHandle == 0) {
            return;
        }
        
        synchronized (this) {
            if (closed.get() || nativeHandle == 0) {
                return;
            }
            
            checkError(nativeFree(nativeHandle));
            nativeHandle = 0;
            closed.set(true);
        }
        
        logger.debug("Pipeline resources freed");
    }
    
    /**
     * Close the pipeline and release resources.
     */
    @Override
    public void close() {
        try {
            free();
        } catch (TrustformersException e) {
            logger.warn("Error closing pipeline", e);
        }
    }
    
    /**
     * Generate text from a prompt (for text generation pipelines).
     * 
     * @param prompt input prompt
     * @return generated text
     * @throws TrustformersException if generation fails
     */
    public String generateText(String prompt) throws TrustformersException {
        checkNotClosed();
        validatePipelineType(PipelineType.TEXT_GENERATION);
        
        if (prompt == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Prompt cannot be null");
        }
        
        return nativeGenerateText(nativeHandle, prompt);
    }
    
    /**
     * Generate text with custom options (for text generation pipelines).
     * 
     * @param prompt input prompt
     * @param options generation options
     * @return generated text
     * @throws TrustformersException if generation fails
     */
    public String generateText(String prompt, GenerationOptions options) throws TrustformersException {
        checkNotClosed();
        validatePipelineType(PipelineType.TEXT_GENERATION);
        
        if (prompt == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Prompt cannot be null");
        }
        
        String optionsJson = options != null ? options.toJson() : "{}";
        return nativeGenerateTextWithOptions(nativeHandle, prompt, optionsJson);
    }
    
    /**
     * Classify text (for text classification pipelines).
     * 
     * @param text text to classify
     * @return classification results
     * @throws TrustformersException if classification fails
     */
    public ClassificationResult[] classifyText(String text) throws TrustformersException {
        checkNotClosed();
        validatePipelineType(PipelineType.TEXT_CLASSIFICATION);
        
        if (text == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Text cannot be null");
        }
        
        String jsonStr = nativeClassifyText(nativeHandle, text);
        return parseClassificationResults(jsonStr);
    }
    
    /**
     * Classify multiple texts in a batch (for text classification pipelines).
     * 
     * @param texts texts to classify
     * @return classification results for each text
     * @throws TrustformersException if classification fails
     */
    public ClassificationResult[][] classifyTextBatch(String[] texts) throws TrustformersException {
        checkNotClosed();
        validatePipelineType(PipelineType.TEXT_CLASSIFICATION);
        
        if (texts == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Texts cannot be null");
        }
        
        String jsonStr = nativeClassifyTextBatch(nativeHandle, texts);
        return parseBatchClassificationResults(jsonStr);
    }
    
    /**
     * Answer a question given context (for question answering pipelines).
     * 
     * @param context context text
     * @param question question to answer
     * @return answer result
     * @throws TrustformersException if answering fails
     */
    public AnswerResult answerQuestion(String context, String question) throws TrustformersException {
        checkNotClosed();
        validatePipelineType(PipelineType.QUESTION_ANSWERING);
        
        if (context == null || question == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Context and question cannot be null");
        }
        
        String jsonStr = nativeAnswerQuestion(nativeHandle, context, question);
        return parseAnswerResult(jsonStr);
    }
    
    /**
     * Add a conversation turn (for conversational pipelines).
     * 
     * @param userInput user input
     * @return bot response
     * @throws TrustformersException if the operation fails
     */
    public String addConversationTurn(String userInput) throws TrustformersException {
        checkNotClosed();
        validatePipelineType(PipelineType.CONVERSATIONAL);
        
        if (userInput == null) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "User input cannot be null");
        }
        
        return nativeAddConversationTurn(nativeHandle, userInput);
    }
    
    /**
     * Get conversation history (for conversational pipelines).
     * 
     * @return conversation history
     * @throws TrustformersException if the operation fails
     */
    public ConversationTurn[] getConversationHistory() throws TrustformersException {
        checkNotClosed();
        validatePipelineType(PipelineType.CONVERSATIONAL);
        
        String jsonStr = nativeGetConversationHistory(nativeHandle);
        return parseConversationHistory(jsonStr);
    }
    
    /**
     * Clear conversation history (for conversational pipelines).
     * 
     * @throws TrustformersException if the operation fails
     */
    public void clearConversation() throws TrustformersException {
        checkNotClosed();
        validatePipelineType(PipelineType.CONVERSATIONAL);
        
        checkError(nativeClearConversation(nativeHandle));
    }
    
    /**
     * Get pipeline information.
     * 
     * @return pipeline information
     * @throws TrustformersException if the operation fails
     */
    public PipelineInfo getInfo() throws TrustformersException {
        checkNotClosed();
        
        String jsonStr = nativeGetInfo(nativeHandle);
        return parsePipelineInfo(jsonStr);
    }
    
    /**
     * Get pipeline performance statistics.
     * 
     * @return performance statistics as JSON
     * @throws TrustformersException if the operation fails
     */
    public JsonNode getPerformanceStats() throws TrustformersException {
        checkNotClosed();
        
        String jsonStr = nativeGetPerformanceStats(nativeHandle);
        if (jsonStr == null || jsonStr.trim().isEmpty()) {
            return objectMapper.createObjectNode();
        }
        
        try {
            return objectMapper.readTree(jsonStr);
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse performance stats JSON", e);
        }
    }
    
    /**
     * Get the pipeline type.
     * 
     * @return pipeline type
     */
    public PipelineType getType() {
        return type;
    }
    
    /**
     * Check if the pipeline is loaded and not closed.
     * 
     * @return true if the pipeline is loaded
     */
    public boolean isLoaded() {
        return !closed.get() && nativeHandle != 0;
    }
    
    // Private helper methods
    
    private static void validateInputs(Model model, Tokenizer tokenizer) throws TrustformersException {
        if (model == null || !model.isLoaded()) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Model must be loaded");
        }
        if (tokenizer == null || !tokenizer.isLoaded()) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                "Tokenizer must be loaded");
        }
    }
    
    private void checkNotClosed() {
        if (closed.get() || nativeHandle == 0) {
            throw new IllegalStateException("Pipeline has been closed or not properly loaded");
        }
    }
    
    private void validatePipelineType(PipelineType expectedType) throws TrustformersException {
        if (type != expectedType) {
            throw new TrustformersException(TrustformersException.ErrorCode.INVALID_PARAMETER,
                String.format("This operation requires a %s pipeline, but this is a %s pipeline",
                    expectedType.getTypeName(), type.getTypeName()));
        }
    }
    
    private void checkError(int errorCode) throws TrustformersException {
        if (errorCode != 0) {
            TrustformersException.ErrorCode code = TrustformersException.ErrorCode.fromCode(errorCode);
            throw new TrustformersException(code, "Pipeline operation failed with error code: " + errorCode);
        }
    }
    
    private ClassificationResult[] parseClassificationResults(String jsonStr) throws TrustformersException {
        try {
            JsonNode[] nodes = objectMapper.readValue(jsonStr, JsonNode[].class);
            ClassificationResult[] results = new ClassificationResult[nodes.length];
            for (int i = 0; i < nodes.length; i++) {
                results[i] = new ClassificationResult(
                    nodes[i].path("label").asText(),
                    nodes[i].path("score").asDouble()
                );
            }
            return results;
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse classification results", e);
        }
    }
    
    private ClassificationResult[][] parseBatchClassificationResults(String jsonStr) throws TrustformersException {
        try {
            JsonNode[][] nodesBatch = objectMapper.readValue(jsonStr, JsonNode[][].class);
            ClassificationResult[][] results = new ClassificationResult[nodesBatch.length][];
            for (int i = 0; i < nodesBatch.length; i++) {
                results[i] = new ClassificationResult[nodesBatch[i].length];
                for (int j = 0; j < nodesBatch[i].length; j++) {
                    results[i][j] = new ClassificationResult(
                        nodesBatch[i][j].path("label").asText(),
                        nodesBatch[i][j].path("score").asDouble()
                    );
                }
            }
            return results;
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse batch classification results", e);
        }
    }
    
    private AnswerResult parseAnswerResult(String jsonStr) throws TrustformersException {
        try {
            JsonNode json = objectMapper.readTree(jsonStr);
            return new AnswerResult(
                json.path("answer").asText(),
                json.path("score").asDouble(),
                json.path("start").asInt(),
                json.path("end").asInt()
            );
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse answer result", e);
        }
    }
    
    private ConversationTurn[] parseConversationHistory(String jsonStr) throws TrustformersException {
        try {
            JsonNode[] nodes = objectMapper.readValue(jsonStr, JsonNode[].class);
            ConversationTurn[] turns = new ConversationTurn[nodes.length];
            for (int i = 0; i < nodes.length; i++) {
                turns[i] = new ConversationTurn(
                    nodes[i].path("user_input").asText(),
                    nodes[i].path("bot_response").asText(),
                    nodes[i].path("timestamp").asLong()
                );
            }
            return turns;
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse conversation history", e);
        }
    }
    
    private PipelineInfo parsePipelineInfo(String jsonStr) throws TrustformersException {
        try {
            JsonNode json = objectMapper.readTree(jsonStr);
            JsonNode capabilitiesNode = json.path("capabilities");
            String[] capabilities = new String[capabilitiesNode.size()];
            for (int i = 0; i < capabilities.length; i++) {
                capabilities[i] = capabilitiesNode.get(i).asText();
            }
            
            return new PipelineInfo(
                json.path("type").asText(),
                json.path("model_name").asText(),
                json.path("tokenizer_name").asText(),
                capabilities,
                json.path("metadata")
            );
        } catch (IOException e) {
            throw new TrustformersException(TrustformersException.ErrorCode.SERIALIZATION_ERROR,
                "Failed to parse pipeline info", e);
        }
    }
    
    @Override
    protected void finalize() throws Throwable {
        try {
            if (!closed.get() && nativeHandle != 0) {
                logger.warn("Pipeline was not explicitly closed, freeing in finalizer");
                free();
            }
        } catch (Exception e) {
            logger.warn("Error in pipeline finalizer", e);
        } finally {
            super.finalize();
        }
    }
    
    // Native method declarations
    
    private static native long nativeCreateTextGeneration(long modelHandle, long tokenizerHandle);
    private static native long nativeCreateTextClassification(long modelHandle, long tokenizerHandle);
    private static native long nativeCreateQuestionAnswering(long modelHandle, long tokenizerHandle);
    private static native long nativeCreateConversational(long modelHandle, long tokenizerHandle);
    private static native int nativeFree(long handle);
    private static native String nativeGenerateText(long handle, String prompt);
    private static native String nativeGenerateTextWithOptions(long handle, String prompt, String optionsJson);
    private static native String nativeClassifyText(long handle, String text);
    private static native String nativeClassifyTextBatch(long handle, String[] texts);
    private static native String nativeAnswerQuestion(long handle, String context, String question);
    private static native String nativeAddConversationTurn(long handle, String userInput);
    private static native String nativeGetConversationHistory(long handle);
    private static native int nativeClearConversation(long handle);
    private static native String nativeGetInfo(long handle);
    private static native String nativeGetPerformanceStats(long handle);
}