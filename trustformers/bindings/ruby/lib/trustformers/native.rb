# frozen_string_literal: true

require "ffi"

module TrustFormeRS
  # Native FFI interface to TrustFormeRS C library
  # @api private
  module Native
    extend FFI::Library

    # Determine library path based on platform
    case RUBY_PLATFORM
    when /darwin/
      ffi_lib File.expand_path("../../../target/release/libtrustformers_c.dylib", __dir__)
    when /linux/
      ffi_lib File.expand_path("../../../target/release/libtrustformers_c.so", __dir__)
    when /mswin|mingw|cygwin/
      ffi_lib File.expand_path("../../../target/release/trustformers_c.dll", __dir__)
    else
      raise LoadError, "Unsupported platform: #{RUBY_PLATFORM}"
    end

    # Error structure
    class TrustformersError < FFI::Struct
      layout :code, :int,
             :message, :string
    end

    # Configuration structure
    class TrustformersConfig < FFI::Struct
      layout :use_gpu, :bool,
             :device, :string,
             :num_threads, :int,
             :enable_logging, :bool,
             :cache_dir, :string
    end

    # Tensor structure
    class TrustformersTensor < FFI::Struct
      layout :data, :pointer,
             :shape, :pointer,
             :ndim, :int,
             :size, :int,
             :dtype, :int
    end

    # Generation configuration structure
    class TrustformersGenerationConfig < FFI::Struct
      layout :max_length, :int,
             :max_new_tokens, :int,
             :min_length, :int,
             :temperature, :float,
             :top_p, :float,
             :top_k, :int,
             :repetition_penalty, :float,
             :do_sample, :bool,
             :num_beams, :int,
             :early_stopping, :bool
    end

    # Core TrustFormeRS functions
    attach_function :trustformers_version, [], :string
    attach_function :trustformers_is_gpu_available, [], :bool
    attach_function :trustformers_is_cuda_available, [], :bool
    attach_function :trustformers_is_metal_available, [], :bool

    # Initialization and cleanup
    attach_function :trustformers_init, [TrustformersConfig.by_value], :pointer
    attach_function :trustformers_free, [:pointer], :void

    # GPU management
    attach_function :trustformers_enable_gpu, [:pointer, :string], :bool
    attach_function :trustformers_disable_gpu, [:pointer], :bool
    attach_function :trustformers_get_current_device, [:pointer], :string

    # Memory management
    attach_function :trustformers_get_memory_stats, [:pointer], :pointer
    attach_function :trustformers_free_memory_stats, [:pointer], :void

    # Model functions
    attach_function :model_load, [:pointer, :string, :pointer], :pointer
    attach_function :model_free, [:pointer], :void
    attach_function :model_generate, [:pointer, :string, TrustformersGenerationConfig.by_value, :pointer], :string
    attach_function :model_encode, [:pointer, :string, :pointer], TrustformersTensor.by_value
    attach_function :model_forward, [:pointer, TrustformersTensor.by_value, :pointer], TrustformersTensor.by_value

    # Tokenizer functions
    attach_function :tokenizer_load, [:pointer, :string, :pointer], :pointer
    attach_function :tokenizer_free, [:pointer], :void
    attach_function :tokenizer_encode, [:pointer, :string, :pointer, :pointer], :pointer
    attach_function :tokenizer_decode, [:pointer, :pointer, :int, :pointer], :string
    attach_function :tokenizer_get_vocab_size, [:pointer], :int
    attach_function :tokenizer_get_vocab, [:pointer, :pointer], :pointer

    # Pipeline functions
    attach_function :pipeline_create, [:pointer, :string, :string, :pointer], :pointer
    attach_function :pipeline_free, [:pointer], :void
    attach_function :pipeline_predict, [:pointer, :string, :pointer], :string

    # Tensor functions
    attach_function :tensor_create, [:pointer, :pointer, :int], TrustformersTensor.by_value
    attach_function :tensor_free, [TrustformersTensor.by_value], :void
    attach_function :tensor_add, [TrustformersTensor.by_value, TrustformersTensor.by_value, :pointer], TrustformersTensor.by_value
    attach_function :tensor_multiply, [TrustformersTensor.by_value, TrustformersTensor.by_value, :pointer], TrustformersTensor.by_value
    attach_function :tensor_matmul, [TrustformersTensor.by_value, TrustformersTensor.by_value, :pointer], TrustformersTensor.by_value
    attach_function :tensor_reshape, [TrustformersTensor.by_value, :pointer, :int, :pointer], TrustformersTensor.by_value

    # Utility functions
    attach_function :trustformers_free_string, [:string], :void
    attach_function :trustformers_free_string_array, [:pointer, :int], :void

    # Helper methods for memory management and error handling
    class << self
      # Check for errors after C function calls
      # @param error [TrustformersError] Error structure to check
      # @raise [TrustFormeRSError] If error occurred
      def check_error!(error)
        return unless error[:code] != 0
        
        message = error[:message] || "Unknown error"
        case error[:code]
        when 1
          raise ArgumentError, message
        when 2
          raise ModelNotFoundError, message
        when 3
          raise InferenceError, message
        when 4
          raise TokenizationError, message
        when 5
          raise InitializationError, message
        else
          raise TrustFormeRSError, message
        end
      end

      # Convert Ruby hash to C configuration struct
      # @param config [Hash] Configuration options
      # @return [TrustformersConfig] C configuration struct
      def hash_to_config(config)
        c_config = TrustformersConfig.new
        c_config[:use_gpu] = config.fetch(:use_gpu, false)
        c_config[:device] = config.fetch(:device, "cpu")
        c_config[:num_threads] = config.fetch(:num_threads, -1)
        c_config[:enable_logging] = config.fetch(:enable_logging, false)
        c_config[:cache_dir] = config.fetch(:cache_dir, nil)
        c_config
      end

      # Convert Ruby array to C tensor
      # @param data [Array<Numeric>] Tensor data
      # @param shape [Array<Integer>] Tensor shape
      # @return [TrustformersTensor] C tensor struct
      def array_to_tensor(data, shape)
        flat_data = data.flatten
        data_ptr = FFI::MemoryPointer.new(:float, flat_data.size)
        data_ptr.write_array_of_float(flat_data)

        shape_ptr = FFI::MemoryPointer.new(:int, shape.size)
        shape_ptr.write_array_of_int(shape)

        tensor_create(data_ptr, shape_ptr, shape.size)
      end

      # Convert C tensor to Ruby array
      # @param tensor [TrustformersTensor] C tensor struct
      # @return [Array] Ruby array with data and shape
      def tensor_to_array(tensor)
        data = tensor[:data].read_array_of_float(tensor[:size])
        shape = tensor[:shape].read_array_of_int(tensor[:ndim])
        [data, shape]
      end

      # Create generation configuration from options
      # @param options [Hash] Generation options
      # @return [TrustformersGenerationConfig] C generation config
      def generation_config_from_options(options)
        config = TrustformersGenerationConfig.new
        config[:max_length] = options.fetch(:max_length, 50)
        config[:max_new_tokens] = options.fetch(:max_new_tokens, -1)
        config[:min_length] = options.fetch(:min_length, 0)
        config[:temperature] = options.fetch(:temperature, 1.0)
        config[:top_p] = options.fetch(:top_p, 1.0)
        config[:top_k] = options.fetch(:top_k, 50)
        config[:repetition_penalty] = options.fetch(:repetition_penalty, 1.0)
        config[:do_sample] = options.fetch(:do_sample, false)
        config[:num_beams] = options.fetch(:num_beams, 1)
        config[:early_stopping] = options.fetch(:early_stopping, false)
        config
      end
    end
  end
end