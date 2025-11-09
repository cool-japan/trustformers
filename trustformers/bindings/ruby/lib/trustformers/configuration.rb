# frozen_string_literal: true

module TrustFormeRS
  # Configuration class for TrustFormeRS settings
  class Configuration
    DEFAULT_OPTIONS = {
      use_gpu: false,
      device: "auto",
      num_threads: -1,  # -1 means auto-detect
      enable_logging: false,
      cache_dir: nil
    }.freeze

    attr_accessor :use_gpu, :device, :num_threads, :enable_logging, :cache_dir

    # Initialize configuration with options
    # @param options [Hash] Configuration options
    def initialize(options = {})
      merged_options = DEFAULT_OPTIONS.merge(options)
      
      @use_gpu = merged_options[:use_gpu]
      @device = merged_options[:device]
      @num_threads = merged_options[:num_threads]
      @enable_logging = merged_options[:enable_logging]
      @cache_dir = merged_options[:cache_dir]

      validate!
    end

    # Create default configuration
    # @return [Configuration] Default configuration
    def self.default
      new(DEFAULT_OPTIONS)
    end

    # Create CPU-optimized configuration
    # @return [Configuration] CPU configuration
    def self.cpu
      new(
        use_gpu: false,
        device: "cpu",
        num_threads: -1,
        enable_logging: false
      )
    end

    # Create GPU-optimized configuration
    # @param device [String] GPU device type ("cuda", "metal", "auto")
    # @return [Configuration] GPU configuration
    def self.gpu(device = "auto")
      new(
        use_gpu: true,
        device: device,
        num_threads: 4,  # Fewer threads when using GPU
        enable_logging: false
      )
    end

    # Create development configuration with logging enabled
    # @return [Configuration] Development configuration
    def self.development
      new(
        use_gpu: false,
        device: "cpu",
        num_threads: 2,
        enable_logging: true,
        cache_dir: "./cache"
      )
    end

    # Create production configuration
    # @return [Configuration] Production configuration
    def self.production
      new(
        use_gpu: TrustFormeRS.gpu_available?,
        device: "auto",
        num_threads: -1,
        enable_logging: false,
        cache_dir: ENV["TRUSTFORMERS_CACHE_DIR"]
      )
    end

    # Convert configuration to hash
    # @return [Hash] Configuration as hash
    def to_h
      {
        use_gpu: @use_gpu,
        device: @device,
        num_threads: @num_threads,
        enable_logging: @enable_logging,
        cache_dir: @cache_dir
      }
    end

    # Convert configuration to JSON string
    # @return [String] Configuration as JSON
    def to_json(*args)
      require "json"
      to_h.to_json(*args)
    end

    # Validate configuration values
    # @raise [ConfigurationError] If configuration is invalid
    def validate!
      validate_device!
      validate_num_threads!
      validate_cache_dir!
    end

    # Enable GPU acceleration
    # @param device [String] GPU device type
    def enable_gpu!(device = "auto")
      unless TrustFormeRS.gpu_available?
        raise DeviceError, "GPU not available on this system"
      end

      @use_gpu = true
      @device = device
      validate_device!
    end

    # Disable GPU acceleration
    def disable_gpu!
      @use_gpu = false
      @device = "cpu"
    end

    # Set number of threads
    # @param threads [Integer] Number of threads (-1 for auto)
    def threads=(threads)
      @num_threads = threads
      validate_num_threads!
    end

    # Enable logging
    def enable_logging!
      @enable_logging = true
    end

    # Disable logging
    def disable_logging!
      @enable_logging = false
    end

    # Set cache directory
    # @param dir [String, nil] Cache directory path
    def cache_dir=(dir)
      @cache_dir = dir
      validate_cache_dir!
    end

    # Get effective device (resolves "auto" to actual device)
    # @return [String] Actual device to be used
    def effective_device
      return @device unless @device == "auto"

      if @use_gpu
        return "metal" if TrustFormeRS.metal_available?
        return "cuda" if TrustFormeRS.cuda_available?
        return "gpu" if TrustFormeRS.gpu_available?
      end

      "cpu"
    end

    # Get effective number of threads
    # @return [Integer] Actual number of threads to be used
    def effective_num_threads
      return @num_threads unless @num_threads == -1

      # Auto-detect based on system and GPU usage
      if @use_gpu
        # Use fewer threads when GPU is available
        [Etc.nprocessors / 2, 1].max
      else
        Etc.nprocessors
      end
    end

    # Check if configuration is valid for current system
    # @return [Boolean] True if configuration is valid
    def valid_for_system?
      if @use_gpu && !TrustFormeRS.gpu_available?
        return false
      end

      if @device == "cuda" && !TrustFormeRS.cuda_available?
        return false
      end

      if @device == "metal" && !TrustFormeRS.metal_available?
        return false
      end

      true
    end

    # Get configuration summary
    # @return [String] Human-readable configuration summary
    def summary
      <<~SUMMARY
        TrustFormeRS Configuration:
          Device: #{effective_device} (GPU: #{@use_gpu ? 'enabled' : 'disabled'})
          Threads: #{effective_num_threads}
          Logging: #{@enable_logging ? 'enabled' : 'disabled'}
          Cache: #{@cache_dir || 'default'}
      SUMMARY
    end

    private

    def validate_device!
      valid_devices = %w[cpu gpu cuda metal auto]
      
      unless valid_devices.include?(@device)
        raise ConfigurationError, "Invalid device '#{@device}'. Must be one of: #{valid_devices.join(', ')}"
      end

      if @use_gpu && @device == "cpu"
        raise ConfigurationError, "Cannot use GPU with CPU device"
      end

      if @device == "cuda" && !TrustFormeRS.cuda_available?
        raise DeviceError, "CUDA device requested but not available"
      end

      if @device == "metal" && !TrustFormeRS.metal_available?
        raise DeviceError, "Metal device requested but not available"
      end
    end

    def validate_num_threads!
      unless @num_threads.is_a?(Integer) && @num_threads >= -1
        raise ConfigurationError, "num_threads must be a non-negative integer or -1 for auto"
      end

      if @num_threads == 0
        raise ConfigurationError, "num_threads cannot be 0"
      end
    end

    def validate_cache_dir!
      return if @cache_dir.nil?

      unless @cache_dir.is_a?(String)
        raise ConfigurationError, "cache_dir must be a string or nil"
      end

      # Check if directory exists or can be created
      if @cache_dir && !Dir.exist?(@cache_dir)
        begin
          require "fileutils"
          FileUtils.mkdir_p(@cache_dir)
        rescue StandardError => e
          raise ConfigurationError, "Cannot create cache directory '#{@cache_dir}': #{e.message}"
        end
      end
    end
  end
end