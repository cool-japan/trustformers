# frozen_string_literal: true

module TrustFormeRS
  # High-level pipeline interface for common NLP tasks
  class Pipeline
    attr_reader :client, :task, :model_id, :config

    # Supported pipeline tasks
    SUPPORTED_TASKS = %i[
      text_generation
      text_classification
      question_answering
      summarization
      translation
      token_classification
      conversational
      feature_extraction
      fill_mask
    ].freeze

    # Task-specific result structures
    TextGenerationResult = Struct.new(:generated_text, :score) do
      def initialize(generated_text:, score: nil)
        super(generated_text, score)
      end
    end

    ClassificationResult = Struct.new(:label, :score) do
      def initialize(label:, score:)
        super(label, score)
      end
    end

    QuestionAnsweringResult = Struct.new(:answer, :score, :start, :end) do
      def initialize(answer:, score:, start: nil, end: nil)
        super(answer, score, start, end)
      end
    end

    SummarizationResult = Struct.new(:summary_text, :score) do
      def initialize(summary_text:, score: nil)
        super(summary_text, score)
      end
    end

    TranslationResult = Struct.new(:translation_text, :score) do
      def initialize(translation_text:, score: nil)
        super(translation_text, score)
      end
    end

    TokenClassificationResult = Struct.new(:entity, :score, :word, :start, :end) do
      def initialize(entity:, score:, word:, start:, end:)
        super(entity, score, word, start, end)
      end
    end

    # Initialize pipeline
    # @param client [Client] TrustFormeRS client
    # @param config [Hash] Pipeline configuration
    def initialize(client, config)
      @client = client
      @config = config.dup
      @task = config[:task] || :text_generation
      @model_id = config[:model_id] || detect_default_model

      validate_task!
      initialize_pipeline!
    end

    # Perform prediction based on pipeline task
    # @param input [String, Array] Input text or array of texts
    # @param options [Hash] Task-specific options
    # @return [Object] Task-specific result
    def predict(input, **options)
      case @task
      when :text_generation
        generate(input, **options)
      when :text_classification
        classify(input, **options)
      when :question_answering
        answer_question(input, **options)
      when :summarization
        summarize(input, **options)
      when :translation
        translate(input, **options)
      when :token_classification
        classify_tokens(input, **options)
      when :conversational
        converse(input, **options)
      when :feature_extraction
        extract_features(input, **options)
      when :fill_mask
        fill_mask(input, **options)
      else
        raise UnsupportedOperationError, "Task #{@task} not supported"
      end
    end

    # Text generation
    # @param input [String] Input prompt
    # @param max_new_tokens [Integer] Maximum new tokens to generate
    # @param temperature [Float] Sampling temperature
    # @param top_p [Float] Top-p sampling parameter
    # @param top_k [Integer] Top-k sampling parameter
    # @param do_sample [Boolean] Whether to use sampling
    # @return [TextGenerationResult, Array<TextGenerationResult>] Generated text(s)
    def generate(input, max_new_tokens: 100, temperature: 0.7, top_p: 0.9, top_k: 50, do_sample: true, **options)
      generation_config = {
        max_new_tokens: max_new_tokens,
        temperature: temperature,
        top_p: top_p,
        top_k: top_k,
        do_sample: do_sample,
        **options
      }

      if input.is_a?(Array)
        input.map { |text| generate_single(text, generation_config) }
      else
        generate_single(input, generation_config)
      end
    end

    # Text classification
    # @param input [String, Array<String>] Input text(s)
    # @param top_k [Integer] Number of top results to return
    # @return [Array<ClassificationResult>] Classification results
    def classify(input, top_k: 5, **options)
      if input.is_a?(Array)
        input.map { |text| classify_single(text, top_k: top_k, **options) }
      else
        classify_single(input, top_k: top_k, **options)
      end
    end

    # Question answering
    # @param input [String, Hash] Question and context
    # @param context [String] Context text (if input is just question)
    # @return [QuestionAnsweringResult] Answer result
    def answer_question(input, context: nil, **options)
      question, ctx = extract_question_context(input, context)
      
      error = Native::TrustformersError.new
      
      # For now, use a simplified approach
      # In a full implementation, this would use specialized QA models
      combined_input = "Context: #{ctx}\nQuestion: #{question}\nAnswer:"
      result = @model.generate(combined_input, max_new_tokens: 50, temperature: 0.1)
      
      # Extract just the answer part
      answer = result.split("Answer:").last&.strip || result
      
      QuestionAnsweringResult.new(
        answer: answer,
        score: 0.9, # Placeholder score
        start: 0,
        end: answer.length
      )
    end

    # Text summarization
    # @param input [String, Array<String>] Input text(s) to summarize
    # @param max_length [Integer] Maximum summary length
    # @param min_length [Integer] Minimum summary length
    # @return [SummarizationResult, Array<SummarizationResult>] Summary result(s)
    def summarize(input, max_length: 150, min_length: 30, **options)
      if input.is_a?(Array)
        input.map { |text| summarize_single(text, max_length: max_length, min_length: min_length, **options) }
      else
        summarize_single(input, max_length: max_length, min_length: min_length, **options)
      end
    end

    # Translation
    # @param input [String, Array<String>] Input text(s) to translate
    # @param target_language [String] Target language code
    # @param source_language [String] Source language code
    # @return [TranslationResult, Array<TranslationResult>] Translation result(s)
    def translate(input, target_language:, source_language: "auto", **options)
      if input.is_a?(Array)
        input.map { |text| translate_single(text, target_language: target_language, source_language: source_language, **options) }
      else
        translate_single(input, target_language: target_language, source_language: source_language, **options)
      end
    end

    # Token classification (NER, POS tagging)
    # @param input [String, Array<String>] Input text(s)
    # @param aggregation_strategy [String] Aggregation strategy for tokens
    # @return [Array<TokenClassificationResult>] Token classification results
    def classify_tokens(input, aggregation_strategy: "simple", **options)
      if input.is_a?(Array)
        input.map { |text| classify_tokens_single(text, aggregation_strategy: aggregation_strategy, **options) }
      else
        classify_tokens_single(input, aggregation_strategy: aggregation_strategy, **options)
      end
    end

    # Conversational AI
    # @param input [String, Hash] User message or conversation history
    # @param conversation_id [String] Conversation identifier
    # @return [String] AI response
    def converse(input, conversation_id: nil, **options)
      # This would maintain conversation history and context
      # For now, treat as simple text generation
      if input.is_a?(Hash)
        # Extract the latest user message
        messages = input[:messages] || input["messages"] || []
        user_message = messages.last
        input_text = user_message.is_a?(Hash) ? (user_message[:content] || user_message["content"]) : user_message.to_s
      else
        input_text = input.to_s
      end

      result = generate(input_text, **options)
      result.is_a?(TextGenerationResult) ? result.generated_text : result
    end

    # Feature extraction (embeddings)
    # @param input [String, Array<String>] Input text(s)
    # @return [Tensor, Array<Tensor>] Feature tensors
    def extract_features(input, **options)
      if input.is_a?(Array)
        input.map { |text| @model.encode(text) }
      else
        @model.encode(input)
      end
    end

    # Fill mask (masked language modeling)
    # @param input [String] Input text with [MASK] token
    # @param top_k [Integer] Number of top predictions
    # @return [Array<Hash>] Predictions with scores
    def fill_mask(input, top_k: 5, **options)
      # This would use a masked language model
      # For now, return a placeholder
      [
        { token: "word", score: 0.9 },
        { token: "text", score: 0.8 },
        { token: "sentence", score: 0.7 }
      ][0, top_k]
    end

    # Batch processing for any task
    # @param inputs [Array] Array of inputs
    # @param batch_size [Integer] Batch size for processing
    # @param options [Hash] Task-specific options
    # @return [Array] Array of results
    def process_batch(inputs, batch_size: 8, **options)
      results = []
      
      inputs.each_slice(batch_size) do |batch|
        batch_results = batch.map { |input| predict(input, **options) }
        results.concat(batch_results)
      end
      
      results
    end

    # Get pipeline information
    # @return [Hash] Pipeline information
    def info
      {
        task: @task,
        model_id: @model_id,
        device: @model&.config&.device || @client.current_device,
        supports_batching: true,
        supported_languages: supported_languages,
        max_input_length: max_input_length
      }
    end

    # Check if pipeline supports specific feature
    # @param feature [Symbol] Feature to check
    # @return [Boolean] True if supported
    def supports?(feature)
      case feature
      when :batching
        true
      when :streaming
        @task == :text_generation
      when :multilingual
        multilingual_model?
      else
        false
      end
    end

    # Get supported languages (if applicable)
    # @return [Array<String>] Language codes
    def supported_languages
      case @task
      when :translation
        # This would be model-specific
        %w[en es fr de it pt ru zh ja ko]
      else
        %w[en] # Default to English
      end
    end

    # Get maximum input length for the model
    # @return [Integer] Maximum input length in tokens
    def max_input_length
      @config[:max_length] || 512
    end

    # Clean up resources
    def finalize
      @model&.finalize
    end

    # String representation
    # @return [String] Human-readable pipeline information
    def to_s
      "Pipeline(task: #{@task}, model: #{@model_id})"
    end

    # Detailed inspection
    # @return [String] Detailed pipeline information
    def inspect
      info_str = info.map { |k, v| "#{k}: #{v.inspect}" }.join(", ")
      "#{to_s}\nInfo: {#{info_str}}"
    end

    private

    # Initialize the pipeline with model and tokenizer
    def initialize_pipeline!
      error = Native::TrustformersError.new
      task_str = @task.to_s
      @handle = Native.pipeline_create(@client.native_handle, task_str, @model_id, error)
      
      if @handle.null?
        Native.check_error!(error)
        raise InitializationError, "Failed to create pipeline for task: #{@task}"
      end

      # Also load model and tokenizer for direct access
      @model = Model.new(@client, @model_id, @config)
      @tokenizer = Tokenizer.new(@client, @model_id, @config) if needs_tokenizer?
    end

    # Validate that the task is supported
    def validate_task!
      unless SUPPORTED_TASKS.include?(@task)
        raise ArgumentError, "Unsupported task: #{@task}. Supported tasks: #{SUPPORTED_TASKS}"
      end
    end

    # Detect default model for task if not specified
    def detect_default_model
      case @task
      when :text_generation
        "gpt2"
      when :text_classification
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
      when :question_answering
        "distilbert-base-cased-distilled-squad"
      when :summarization
        "facebook/bart-large-cnn"
      when :translation
        "Helsinki-NLP/opus-mt-en-de"
      when :token_classification
        "dbmdz/bert-large-cased-finetuned-conll03-english"
      else
        "bert-base-uncased"
      end
    end

    # Check if task needs a tokenizer
    def needs_tokenizer?
      !%i[feature_extraction].include?(@task)
    end

    # Check if model is multilingual
    def multilingual_model?
      multilingual_patterns = %w[multilingual mbert xlm-roberta opus-mt]
      multilingual_patterns.any? { |pattern| @model_id.include?(pattern) }
    end

    # Single text generation
    def generate_single(input, config)
      result = @model.generate(input, config)
      TextGenerationResult.new(generated_text: result)
    end

    # Single text classification
    def classify_single(input, top_k:, **options)
      error = Native::TrustformersError.new
      result_ptr = Native.pipeline_predict(@handle, input, error)
      
      if result_ptr.null?
        Native.check_error!(error)
        raise InferenceError, "Failed to classify text"
      end

      # Parse JSON result (simplified)
      result_json = result_ptr.read_string
      Native.trustformers_free_string(result_ptr)
      
      # For now, return mock results
      # In a real implementation, this would parse the actual model output
      [
        ClassificationResult.new(label: "POSITIVE", score: 0.8),
        ClassificationResult.new(label: "NEGATIVE", score: 0.2)
      ][0, top_k]
    end

    # Extract question and context from input
    def extract_question_context(input, context)
      if input.is_a?(Hash)
        question = input[:question] || input["question"]
        ctx = input[:context] || input["context"] || context
      else
        question = input
        ctx = context
      end

      raise ArgumentError, "Both question and context are required" if question.nil? || ctx.nil?
      
      [question, ctx]
    end

    # Single text summarization
    def summarize_single(input, max_length:, min_length:, **options)
      config = {
        max_new_tokens: max_length,
        min_length: min_length,
        num_beams: 4,
        early_stopping: true,
        **options
      }

      # Prepend summarization instruction
      prompt = "Summarize the following text:\n\n#{input}\n\nSummary:"
      result = @model.generate(prompt, config)
      
      # Extract summary from result
      summary = result.split("Summary:").last&.strip || result
      
      SummarizationResult.new(summary_text: summary)
    end

    # Single text translation
    def translate_single(input, target_language:, source_language:, **options)
      # Create translation prompt
      prompt = if source_language == "auto"
                 "Translate to #{target_language}: #{input}"
               else
                 "Translate from #{source_language} to #{target_language}: #{input}"
               end

      result = @model.generate(prompt, max_new_tokens: input.length * 2, temperature: 0.3)
      
      TranslationResult.new(translation_text: result)
    end

    # Single token classification
    def classify_tokens_single(input, aggregation_strategy:, **options)
      # This would use a token classification model
      # For now, return mock NER results
      words = input.split
      results = []
      
      words.each_with_index do |word, i|
        # Mock entity detection
        if word.match?(/\A[A-Z][a-z]+\z/) # Simple proper noun detection
          results << TokenClassificationResult.new(
            entity: "PERSON",
            score: 0.9,
            word: word,
            start: i * 5, # Simplified position
            end: (i + 1) * 5
          )
        end
      end
      
      results
    end

    # Get native handle for FFI calls
    def native_handle
      @handle
    end

    class << self
      # Create pipeline with sensible defaults for task
      # @param task [Symbol] Pipeline task
      # @param model_id [String] Model identifier
      # @param client [Client] TrustFormeRS client
      # @return [Pipeline] Configured pipeline
      def for_task(task, model_id: nil, client: nil, **options)
        client ||= TrustFormeRS.new
        config = {
          task: task,
          model_id: model_id,
          **options
        }
        new(client, config)
      end

      # Get recommended configuration for task
      # @param task [Symbol] Pipeline task
      # @return [Hash] Recommended configuration
      def recommended_config(task)
        case task
        when :text_generation
          { temperature: 0.7, max_new_tokens: 100, do_sample: true }
        when :text_classification
          { top_k: 5 }
        when :question_answering
          { max_answer_length: 50 }
        when :summarization
          { max_length: 150, min_length: 30, num_beams: 4 }
        when :translation
          { num_beams: 5, early_stopping: true }
        else
          {}
        end
      end
    end
  end
end