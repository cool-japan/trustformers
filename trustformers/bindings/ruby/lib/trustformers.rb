# frozen_string_literal: true

require_relative "trustformers/version"
require_relative "trustformers/native"
require_relative "trustformers/errors"
require_relative "trustformers/configuration"
require_relative "trustformers/tensor"
require_relative "trustformers/tokenizer"
require_relative "trustformers/model"
require_relative "trustformers/pipeline"

# TrustFormeRS Ruby bindings for high-performance transformer models
#
# TrustFormeRS provides efficient implementations of popular transformer architectures
# with support for text generation, classification, question answering, and other NLP tasks.
#
# @example Basic usage
#   # Initialize TrustFormeRS with default configuration
#   trustformers = TrustFormeRS.new
#
#   # Create a text generation pipeline
#   pipeline = trustformers.create_pipeline(
#     task: :text_generation,
#     model_id: "gpt2",
#     temperature: 0.7,
#     max_new_tokens: 100
#   )
#
#   # Generate text
#   result = pipeline.generate("The future of AI is")
#   puts result
#
# @example Text classification
#   pipeline = trustformers.create_pipeline(
#     task: :text_classification,
#     model_id: "cardiffnlp/twitter-roberta-base-sentiment-latest"
#   )
#
#   results = pipeline.classify("I love this new framework!")
#   results.each { |result| puts "#{result[:label]}: #{result[:score]}" }
#
module TrustFormeRS
  class << self
    # Get the version of TrustFormeRS
    # @return [String] Version string
    def version
      Native.trustformers_version
    end

    # Check if GPU is available on the system
    # @return [Boolean] True if GPU is available
    def gpu_available?
      Native.trustformers_is_gpu_available
    end

    # Check if CUDA is available on the system
    # @return [Boolean] True if CUDA is available
    def cuda_available?
      Native.trustformers_is_cuda_available
    end

    # Check if Metal is available (macOS only)
    # @return [Boolean] True if Metal is available
    def metal_available?
      return false unless RUBY_PLATFORM.include?("darwin")
      Native.trustformers_is_metal_available
    end

    # Get available devices on the system
    # @return [Array<String>] List of available devices
    def available_devices
      devices = ["cpu"]
      devices << "cuda" if cuda_available?
      devices << "metal" if metal_available?
      devices << "gpu" if gpu_available?
      devices.uniq
    end

    # Get system information
    # @return [Hash] System information including device capabilities
    def system_info
      {
        version: version,
        devices: available_devices,
        gpu_available: gpu_available?,
        cuda_available: cuda_available?,
        metal_available: metal_available?,
        ruby_version: RUBY_VERSION,
        platform: RUBY_PLATFORM
      }
    end
  end

  # Main TrustFormeRS class for creating pipelines and managing models
  class Client
    attr_reader :config

    # Initialize TrustFormeRS client
    # @param config [Configuration, Hash] Configuration options
    def initialize(config = {})
      @config = config.is_a?(Configuration) ? config : Configuration.new(config)
      @handle = Native.trustformers_init(@config.to_h)
      
      raise InitializationError, "Failed to initialize TrustFormeRS" if @handle.null?
    end

    # Create a new pipeline for a specific task
    # @param task [Symbol] Task type (:text_generation, :text_classification, etc.)
    # @param model_id [String] Model identifier (Hugging Face model ID or local path)
    # @param options [Hash] Pipeline-specific options
    # @return [Pipeline] Configured pipeline for the task
    def create_pipeline(task:, model_id:, **options)
      pipeline_config = {
        task: task,
        model_id: model_id,
        **options
      }

      Pipeline.new(self, pipeline_config)
    end

    # Load a model from a path or Hugging Face ID
    # @param model_id [String] Model identifier
    # @param options [Hash] Model loading options
    # @return [Model] Loaded model
    def load_model(model_id, **options)
      Model.new(self, model_id, options)
    end

    # Load a tokenizer from a path or Hugging Face ID
    # @param tokenizer_id [String] Tokenizer identifier
    # @param options [Hash] Tokenizer loading options
    # @return [Tokenizer] Loaded tokenizer
    def load_tokenizer(tokenizer_id, **options)
      Tokenizer.new(self, tokenizer_id, options)
    end

    # Enable GPU acceleration if available
    # @param device [String] Specific device to use ("cuda", "metal", "auto")
    # @return [Boolean] True if GPU was enabled successfully
    def enable_gpu(device = "auto")
      return false unless TrustFormeRS.gpu_available?

      device = detect_best_gpu_device if device == "auto"
      Native.trustformers_enable_gpu(@handle, device)
    end

    # Disable GPU acceleration and use CPU only
    # @return [Boolean] True if CPU mode was enabled successfully
    def disable_gpu
      Native.trustformers_disable_gpu(@handle)
    end

    # Get current device being used
    # @return [String] Current device ("cpu", "cuda", "metal", etc.)
    def current_device
      Native.trustformers_get_current_device(@handle)
    end

    # Get memory usage statistics
    # @return [Hash] Memory usage information
    def memory_stats
      stats = Native.trustformers_get_memory_stats(@handle)
      {
        total_allocated: stats[:total_allocated],
        currently_allocated: stats[:currently_allocated],
        peak_allocated: stats[:peak_allocated],
        device_memory: stats[:device_memory]
      }
    end

    # Clean up resources
    def close
      return unless @handle && !@handle.null?
      
      Native.trustformers_free(@handle)
      @handle = nil
    end

    # Automatic cleanup when object is garbage collected
    def finalize
      close
    end

    # Get the native handle (for internal use)
    # @api private
    def native_handle
      @handle
    end

    private

    def detect_best_gpu_device
      return "metal" if TrustFormeRS.metal_available?
      return "cuda" if TrustFormeRS.cuda_available?
      "gpu"
    end
  end

  # Convenience method to create a new TrustFormeRS client
  # @param config [Configuration, Hash] Configuration options
  # @return [Client] New TrustFormeRS client
  def self.new(config = {})
    Client.new(config)
  end
end