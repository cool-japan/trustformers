# frozen_string_literal: true

module TrustFormeRS
  # Base error class for all TrustFormeRS-related errors
  class TrustFormeRSError < StandardError
    attr_reader :code

    def initialize(message = nil, code = nil)
      super(message)
      @code = code
    end
  end

  # Raised when invalid arguments are provided to TrustFormeRS functions
  class ArgumentError < TrustFormeRSError
    def initialize(message = "Invalid argument provided")
      super(message, 1)
    end
  end

  # Raised when a requested model cannot be found or loaded
  class ModelNotFoundError < TrustFormeRSError
    def initialize(message = "Model not found")
      super(message, 2)
    end
  end

  # Raised when an error occurs during model inference
  class InferenceError < TrustFormeRSError
    def initialize(message = "Inference error occurred")
      super(message, 3)
    end
  end

  # Raised when an error occurs during tokenization
  class TokenizationError < TrustFormeRSError
    def initialize(message = "Tokenization error occurred")
      super(message, 4)
    end
  end

  # Raised when TrustFormeRS fails to initialize
  class InitializationError < TrustFormeRSError
    def initialize(message = "Failed to initialize TrustFormeRS")
      super(message, 5)
    end
  end

  # Raised when tensor operations encounter incompatible shapes
  class TensorShapeError < TrustFormeRSError
    def initialize(message = "Incompatible tensor shapes")
      super(message, 6)
    end
  end

  # Raised when an unsupported operation is attempted
  class UnsupportedOperationError < TrustFormeRSError
    def initialize(message = "Unsupported operation")
      super(message, 7)
    end
  end

  # Raised when device-related operations fail
  class DeviceError < TrustFormeRSError
    def initialize(message = "Device operation failed")
      super(message, 8)
    end
  end

  # Raised when memory allocation or management fails
  class MemoryError < TrustFormeRSError
    def initialize(message = "Memory operation failed")
      super(message, 9)
    end
  end

  # Raised when configuration is invalid
  class ConfigurationError < TrustFormeRSError
    def initialize(message = "Invalid configuration")
      super(message, 10)
    end
  end
end