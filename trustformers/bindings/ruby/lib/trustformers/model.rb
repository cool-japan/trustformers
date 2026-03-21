# frozen_string_literal: true

module TrustFormeRS
  # Model class for loading and managing transformer models
  class Model
    attr_reader :client, :model_id, :config

    # Model configuration structure
    ModelConfig = Struct.new(:model_path, :tokenizer_path, :device, :precision, :cache_dir) do
      def initialize(model_path:, tokenizer_path: nil, device: "auto", precision: "float32", cache_dir: nil)
        super(model_path, tokenizer_path, device, precision, cache_dir)
      end
    end

    # Generation configuration structure
    GenerationConfig = Struct.new(
      :max_length, :max_new_tokens, :min_length, :temperature, :top_p, :top_k,
      :repetition_penalty, :do_sample, :num_beams, :early_stopping, :pad_token_id, :eos_token_id
    ) do
      def initialize(
        max_length: 50,
        max_new_tokens: -1,
        min_length: 0,
        temperature: 1.0,
        top_p: 1.0,
        top_k: 50,
        repetition_penalty: 1.0,
        do_sample: false,
        num_beams: 1,
        early_stopping: false,
        pad_token_id: nil,
        eos_token_id: nil
      )
        super(
          max_length, max_new_tokens, min_length, temperature, top_p, top_k,
          repetition_penalty, do_sample, num_beams, early_stopping, pad_token_id, eos_token_id
        )
      end

      def to_h
        {
          max_length: max_length,
          max_new_tokens: max_new_tokens,
          min_length: min_length,
          temperature: temperature,
          top_p: top_p,
          top_k: top_k,
          repetition_penalty: repetition_penalty,
          do_sample: do_sample,
          num_beams: num_beams,
          early_stopping: early_stopping,
          pad_token_id: pad_token_id,
          eos_token_id: eos_token_id
        }
      end
    end

    # Initialize model
    # @param client [Client] TrustFormeRS client
    # @param model_id [String] Model identifier or path
    # @param options [Hash] Model options
    def initialize(client, model_id, options = {})
      @client = client
      @model_id = model_id
      @options = options
      @config = create_model_config(options)

      load_model!
    end

    # Generate text from input prompt
    # @param input [String] Input prompt
    # @param generation_config [GenerationConfig, Hash] Generation configuration
    # @return [String] Generated text
    def generate(input, generation_config = {})
      config = normalize_generation_config(generation_config)
      
      error = Native::TrustformersError.new
      native_config = Native.generation_config_from_options(config.to_h)
      
      result_ptr = Native.model_generate(@handle, input, native_config, error)
      
      if result_ptr.null?
        Native.check_error!(error)
        raise InferenceError, "Failed to generate text"
      end

      result = result_ptr.read_string
      Native.trustformers_free_string(result_ptr)
      
      result
    end

    # Generate text with streaming output
    # @param input [String] Input prompt
    # @param generation_config [GenerationConfig, Hash] Generation configuration
    # @yield [String] Each generated token/chunk
    # @return [String] Complete generated text
    def generate_streaming(input, generation_config = {})
      # For now, implement as a simple wrapper that yields the complete result
      # In a full implementation, this would use streaming APIs
      result = generate(input, generation_config)
      
      if block_given?
        # Simulate streaming by yielding chunks
        chunk_size = [result.length / 10, 1].max
        result.chars.each_slice(chunk_size) do |chunk|
          yield chunk.join
        end
      end
      
      result
    end

    # Encode input text to embeddings/hidden states
    # @param input [String] Input text
    # @return [Tensor] Encoded tensor
    def encode(input)
      error = Native::TrustformersError.new
      
      native_tensor = Native.model_encode(@handle, input, error)
      
      if native_tensor.null?
        Native.check_error!(error)
        raise InferenceError, "Failed to encode input"
      end

      Tensor.from_native(native_tensor)
    end

    # Forward pass through the model
    # @param input_tensor [Tensor] Input tensor
    # @return [Tensor] Output tensor
    def forward(input_tensor)
      error = Native::TrustformersError.new
      native_input = input_tensor.to_native
      
      native_output = Native.model_forward(@handle, native_input, error)
      
      if native_output.null?
        Native.check_error!(error)
        raise InferenceError, "Failed to perform forward pass"
      end

      Tensor.from_native(native_output)
    end

    # Get model information and configuration
    # @return [Hash] Model information
    def info
      {
        model_id: @model_id,
        model_path: @config.model_path,
        tokenizer_path: @config.tokenizer_path,
        device: @config.device,
        precision: @config.precision,
        parameters: parameter_count,
        memory_usage: memory_usage
      }
    end

    # Get approximate parameter count
    # @return [Integer] Number of parameters (estimated)
    def parameter_count
      # This would be implemented by querying the native model
      # For now, return a placeholder
      @parameter_count ||= 0
    end

    # Get current memory usage
    # @return [Hash] Memory usage statistics
    def memory_usage
      @client.memory_stats
    end

    # Check if model supports a specific task
    # @param task [Symbol] Task type
    # @return [Boolean] True if task is supported
    def supports_task?(task)
      case task
      when :text_generation
        true # Most models support text generation
      when :text_classification, :question_answering, :summarization
        # Check model configuration or architecture
        true # Simplified for now
      else
        false
      end
    end

    # Get supported tasks for this model
    # @return [Array<Symbol>] List of supported tasks
    def supported_tasks
      tasks = []
      tasks << :text_generation if supports_task?(:text_generation)
      tasks << :text_classification if supports_task?(:text_classification)
      tasks << :question_answering if supports_task?(:question_answering)
      tasks << :summarization if supports_task?(:summarization)
      tasks
    end

    # Move model to different device
    # @param device [String] Target device ("cpu", "cuda", "metal", etc.)
    # @return [Boolean] True if successful
    def to_device(device)
      # This would be implemented by calling native device transfer
      @config.device = device
      true
    end

    # Change model precision
    # @param precision [String] Target precision ("float32", "float16", "int8")
    # @return [Boolean] True if successful
    def to_precision(precision)
      # This would be implemented by calling native precision conversion
      @config.precision = precision
      true
    end

    # Save model to path
    # @param path [String] Save path
    # @param format [String] Save format ("safetensors", "pytorch", "onnx")
    # @return [Boolean] True if successful
    def save(path, format: "safetensors")
      raise ArgumentError, "Model not loaded" unless @handle && !@handle.null?
      
      # Validate format
      valid_formats = %w[safetensors pytorch onnx]
      raise ArgumentError, "Invalid format '#{format}'. Must be one of: #{valid_formats.join(', ')}" unless valid_formats.include?(format)
      
      # Create directory if it doesn't exist
      dir = File.dirname(path)
      Dir.mkdir(dir) unless Dir.exist?(dir)
      
      begin
        # Call native save function
        result = Native.model_save(@handle, path, format)
        
        # Check if save was successful
        if result == 0
          # Also save configuration
          config_path = File.join(path, "config.json")
          File.write(config_path, @config.to_json) if @config
          true
        else
          false
        end
      rescue => e
        # Log error and return false
        warn "Failed to save model: #{e.message}"
        false
      end
    end

    # Clean up native resources
    def finalize
      return unless @handle && !@handle.null?
      
      Native.model_free(@handle)
      @handle = nil
    end

    # String representation
    # @return [String] Human-readable model information
    def to_s
      "Model(id: #{@model_id}, device: #{@config.device}, precision: #{@config.precision})"
    end

    # Detailed inspection
    # @return [String] Detailed model information
    def inspect
      info_str = info.map { |k, v| "#{k}: #{v.inspect}" }.join(", ")
      "#{to_s}\nInfo: {#{info_str}}"
    end

    private

    # Load the model using native API
    def load_model!
      error = Native::TrustformersError.new
      @handle = Native.model_load(@client.native_handle, @model_id, error)
      
      if @handle.null?
        Native.check_error!(error)
        raise ModelNotFoundError, "Failed to load model: #{@model_id}"
      end
    end

    # Create model configuration from options
    def create_model_config(options)
      ModelConfig.new(
        model_path: options[:model_path] || @model_id,
        tokenizer_path: options[:tokenizer_path],
        device: options[:device] || @client.config.effective_device,
        precision: options[:precision] || "float32",
        cache_dir: options[:cache_dir] || @client.config.cache_dir
      )
    end

    # Normalize generation configuration
    def normalize_generation_config(config)
      if config.is_a?(GenerationConfig)
        config
      elsif config.is_a?(Hash)
        GenerationConfig.new(**config)
      else
        raise ArgumentError, "Generation config must be GenerationConfig or Hash"
      end
    end

    # Get native handle for FFI calls
    def native_handle
      @handle
    end

    class << self
      # Create model configuration for local model
      # @param path [String] Local model path
      # @param tokenizer_path [String, nil] Tokenizer path (optional)
      # @return [Hash] Model configuration
      def local_config(path, tokenizer_path: nil)
        {
          model_path: path,
          tokenizer_path: tokenizer_path || path
        }
      end

      # Create model configuration for Hugging Face model
      # @param model_id [String] Hugging Face model ID
      # @return [Hash] Model configuration
      def huggingface_config(model_id)
        {
          model_path: model_id,
          tokenizer_path: nil # Will use same ID
        }
      end

      # Create model configuration for ONNX model
      # @param path [String] ONNX model path
      # @param tokenizer_path [String] Tokenizer path
      # @return [Hash] Model configuration
      def onnx_config(path, tokenizer_path:)
        {
          model_path: path,
          tokenizer_path: tokenizer_path,
          format: "onnx"
        }
      end

      # Get recommended generation config for different tasks
      # @param task [Symbol] Task type
      # @return [GenerationConfig] Recommended configuration
      def recommended_generation_config(task)
        case task
        when :text_generation
          GenerationConfig.new(
            max_new_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            do_sample: true
          )
        when :summarization
          GenerationConfig.new(
            max_new_tokens: 150,
            min_length: 30,
            temperature: 0.3,
            num_beams: 4,
            early_stopping: true
          )
        when :chat
          GenerationConfig.new(
            max_new_tokens: 200,
            temperature: 0.8,
            top_p: 0.9,
            repetition_penalty: 1.1,
            do_sample: true
          )
        else
          GenerationConfig.new
        end
      end
    end
  end
end