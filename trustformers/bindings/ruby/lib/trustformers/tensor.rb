# frozen_string_literal: true

module TrustFormeRS
  # Multi-dimensional tensor for numerical computations
  class Tensor
    attr_reader :data, :shape, :dtype, :device

    SUPPORTED_DTYPES = %i[float32 float64 int32 int64 bool].freeze
    SUPPORTED_DEVICES = %w[cpu gpu cuda metal].freeze

    # Initialize tensor with data and shape
    # @param data [Array] Tensor data (flat array)
    # @param shape [Array<Integer>] Tensor dimensions
    # @param dtype [Symbol] Data type (:float32, :float64, :int32, :int64, :bool)
    # @param device [String] Device location ("cpu", "gpu", "cuda", "metal")
    def initialize(data, shape, dtype: :float32, device: "cpu")
      @data = Array(data).flatten
      @shape = Array(shape)
      @dtype = dtype
      @device = device

      validate!
    end

    # Create tensor from 1D array
    # @param array [Array] 1D array of values
    # @param dtype [Symbol] Data type
    # @param device [String] Device location
    # @return [Tensor] 1D tensor
    def self.from_array(array, dtype: :float32, device: "cpu")
      new(array, [array.size], dtype: dtype, device: device)
    end

    # Create tensor from 2D array
    # @param array [Array<Array>] 2D array of values
    # @param dtype [Symbol] Data type
    # @param device [String] Device location
    # @return [Tensor] 2D tensor
    def self.from_2d(array, dtype: :float32, device: "cpu")
      rows = array.size
      cols = array.first&.size || 0
      flat_data = array.flatten
      new(flat_data, [rows, cols], dtype: dtype, device: device)
    end

    # Create tensor from 3D array
    # @param array [Array<Array<Array>>] 3D array of values
    # @param dtype [Symbol] Data type
    # @param device [String] Device location
    # @return [Tensor] 3D tensor
    def self.from_3d(array, dtype: :float32, device: "cpu")
      dim0 = array.size
      dim1 = array.first&.size || 0
      dim2 = array.first&.first&.size || 0
      flat_data = array.flatten
      new(flat_data, [dim0, dim1, dim2], dtype: dtype, device: device)
    end

    # Create tensor filled with zeros
    # @param shape [Array<Integer>] Tensor dimensions
    # @param dtype [Symbol] Data type
    # @param device [String] Device location
    # @return [Tensor] Zero-filled tensor
    def self.zeros(shape, dtype: :float32, device: "cpu")
      size = shape.reduce(1, :*)
      data = Array.new(size, 0.0)
      new(data, shape, dtype: dtype, device: device)
    end

    # Create tensor filled with ones
    # @param shape [Array<Integer>] Tensor dimensions
    # @param dtype [Symbol] Data type
    # @param device [String] Device location
    # @return [Tensor] One-filled tensor
    def self.ones(shape, dtype: :float32, device: "cpu")
      size = shape.reduce(1, :*)
      data = Array.new(size, 1.0)
      new(data, shape, dtype: dtype, device: device)
    end

    # Create tensor with random values
    # @param shape [Array<Integer>] Tensor dimensions
    # @param distribution [Symbol] Distribution type (:uniform, :normal)
    # @param dtype [Symbol] Data type
    # @param device [String] Device location
    # @return [Tensor] Random tensor
    def self.random(shape, distribution: :uniform, dtype: :float32, device: "cpu")
      size = shape.reduce(1, :*)
      
      data = case distribution
             when :uniform
               Array.new(size) { rand }
             when :normal
               Array.new(size) { normal_random }
             else
               raise ArgumentError, "Unsupported distribution: #{distribution}"
             end

      new(data, shape, dtype: dtype, device: device)
    end

    # Create identity matrix
    # @param size [Integer] Matrix size
    # @param dtype [Symbol] Data type
    # @param device [String] Device location
    # @return [Tensor] Identity matrix tensor
    def self.eye(size, dtype: :float32, device: "cpu")
      data = Array.new(size * size, 0.0)
      (0...size).each { |i| data[i * size + i] = 1.0 }
      new(data, [size, size], dtype: dtype, device: device)
    end

    # Get number of dimensions
    # @return [Integer] Number of dimensions
    def ndim
      @shape.size
    end

    # Get total number of elements
    # @return [Integer] Total size
    def size
      @shape.reduce(1, :*)
    end

    # Get element at specific indices
    # @param indices [Array<Integer>] Multi-dimensional indices
    # @return [Numeric] Element value
    def [](*indices)
      flat_index = calculate_flat_index(indices)
      @data[flat_index]
    end

    # Set element at specific indices
    # @param indices [Array<Integer>] Multi-dimensional indices and value
    def []=(*args)
      value = args.pop
      indices = args
      flat_index = calculate_flat_index(indices)
      @data[flat_index] = value
    end

    # Reshape tensor
    # @param new_shape [Array<Integer>] New tensor shape
    # @return [Tensor] Reshaped tensor
    def reshape(new_shape)
      new_size = new_shape.reduce(1, :*)
      
      unless new_size == size
        raise TensorShapeError, "Cannot reshape tensor of size #{size} to shape #{new_shape}"
      end

      Tensor.new(@data.dup, new_shape, dtype: @dtype, device: @device)
    end

    # Transpose tensor (swap dimensions)
    # @param dim0 [Integer] First dimension
    # @param dim1 [Integer] Second dimension
    # @return [Tensor] Transposed tensor
    def transpose(dim0 = 0, dim1 = 1)
      unless dim0 < ndim && dim1 < ndim
        raise ArgumentError, "Dimensions #{dim0} and #{dim1} out of bounds for tensor with #{ndim} dimensions"
      end

      new_shape = @shape.dup
      new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

      # For 2D case, implement actual transpose
      if ndim == 2 && dim0 == 0 && dim1 == 1
        rows, cols = @shape
        new_data = Array.new(size)
        
        (0...rows).each do |i|
          (0...cols).each do |j|
            new_data[j * rows + i] = @data[i * cols + j]
          end
        end

        return Tensor.new(new_data, new_shape, dtype: @dtype, device: @device)
      end

      # For higher dimensions, return a view with swapped shape
      # (This is a simplified implementation)
      Tensor.new(@data.dup, new_shape, dtype: @dtype, device: @device)
    end

    # Squeeze tensor (remove dimensions of size 1)
    # @param dim [Integer, nil] Specific dimension to squeeze
    # @return [Tensor] Squeezed tensor
    def squeeze(dim = nil)
      new_shape = if dim
                    unless dim < ndim && @shape[dim] == 1
                      raise ArgumentError, "Cannot squeeze dimension #{dim}"
                    end
                    @shape.dup.tap { |s| s.delete_at(dim) }
                  else
                    @shape.reject { |s| s == 1 }
                  end

      Tensor.new(@data.dup, new_shape, dtype: @dtype, device: @device)
    end

    # Unsqueeze tensor (add dimension of size 1)
    # @param dim [Integer] Dimension to add
    # @return [Tensor] Unsqueezed tensor
    def unsqueeze(dim)
      unless dim >= 0 && dim <= ndim
        raise ArgumentError, "Cannot unsqueeze at dimension #{dim}"
      end

      new_shape = @shape.dup
      new_shape.insert(dim, 1)

      Tensor.new(@data.dup, new_shape, dtype: @dtype, device: @device)
    end

    # Add two tensors
    # @param other [Tensor] Other tensor
    # @return [Tensor] Sum tensor
    def +(other)
      check_compatible_shapes!(other)
      
      result_data = @data.zip(other.data).map { |a, b| a + b }
      Tensor.new(result_data, @shape, dtype: @dtype, device: @device)
    end

    # Subtract two tensors
    # @param other [Tensor] Other tensor
    # @return [Tensor] Difference tensor
    def -(other)
      check_compatible_shapes!(other)
      
      result_data = @data.zip(other.data).map { |a, b| a - b }
      Tensor.new(result_data, @shape, dtype: @dtype, device: @device)
    end

    # Multiply two tensors element-wise
    # @param other [Tensor] Other tensor
    # @return [Tensor] Product tensor
    def *(other)
      check_compatible_shapes!(other)
      
      result_data = @data.zip(other.data).map { |a, b| a * b }
      Tensor.new(result_data, @shape, dtype: @dtype, device: @device)
    end

    # Divide two tensors element-wise
    # @param other [Tensor] Other tensor
    # @return [Tensor] Division tensor
    def /(other)
      check_compatible_shapes!(other)
      
      result_data = @data.zip(other.data).map { |a, b| a / b }
      Tensor.new(result_data, @shape, dtype: @dtype, device: @device)
    end

    # Matrix multiplication
    # @param other [Tensor] Other tensor
    # @return [Tensor] Matrix product
    def matmul(other)
      unless ndim == 2 && other.ndim == 2
        raise ArgumentError, "Matrix multiplication requires 2D tensors"
      end

      unless @shape[1] == other.shape[0]
        raise TensorShapeError, "Cannot multiply matrices with shapes #{@shape} and #{other.shape}"
      end

      m, k = @shape
      n = other.shape[1]
      result_data = Array.new(m * n, 0.0)

      (0...m).each do |i|
        (0...n).each do |j|
          sum = 0.0
          (0...k).each do |l|
            sum += @data[i * k + l] * other.data[l * n + j]
          end
          result_data[i * n + j] = sum
        end
      end

      Tensor.new(result_data, [m, n], dtype: @dtype, device: @device)
    end

    # Sum tensor elements
    # @param axis [Integer, nil] Axis to sum along
    # @return [Tensor] Sum tensor
    def sum(axis = nil)
      if axis.nil?
        total = @data.reduce(:+)
        return Tensor.new([total], [1], dtype: @dtype, device: @device)
      end

      raise ArgumentError, "Axis #{axis} out of bounds for tensor with #{@shape.size} dimensions" unless axis >= 0 && axis < @shape.size

      # Calculate output shape
      output_shape = @shape.dup
      output_shape.delete_at(axis)
      output_shape = [1] if output_shape.empty?

      output_size = output_shape.reduce(1, :*)
      result = Array.new(output_size, 0.0)

      # Calculate strides for the operation
      axis_size = @shape[axis]
      outer_size = @shape[0...axis].reduce(1, :*)
      inner_size = @shape[(axis+1)..-1].reduce(1, :*)

      # Perform summation along the specified axis
      (0...outer_size).each do |outer|
        (0...inner_size).each do |inner|
          sum = 0.0
          (0...axis_size).each do |axis_idx|
            input_idx = outer * axis_size * inner_size + axis_idx * inner_size + inner
            sum += @data[input_idx]
          end
          output_idx = outer * inner_size + inner
          result[output_idx] = sum
        end
      end

      Tensor.new(result, output_shape, dtype: @dtype, device: @device)
    end

    # Mean of tensor elements
    # @param axis [Integer, nil] Axis to average along
    # @return [Tensor] Mean tensor
    def mean(axis = nil)
      if axis.nil?
        total = @data.reduce(:+) / @data.size.to_f
        return Tensor.new([total], [1], dtype: @dtype, device: @device)
      end

      raise ArgumentError, "Axis #{axis} out of bounds for tensor with #{@shape.size} dimensions" unless axis >= 0 && axis < @shape.size

      # Calculate output shape
      output_shape = @shape.dup
      output_shape.delete_at(axis)
      output_shape = [1] if output_shape.empty?

      output_size = output_shape.reduce(1, :*)
      result = Array.new(output_size, 0.0)

      # Calculate strides for the operation
      axis_size = @shape[axis]
      outer_size = @shape[0...axis].reduce(1, :*)
      inner_size = @shape[(axis+1)..-1].reduce(1, :*)

      # Perform averaging along the specified axis
      (0...outer_size).each do |outer|
        (0...inner_size).each do |inner|
          sum = 0.0
          (0...axis_size).each do |axis_idx|
            input_idx = outer * axis_size * inner_size + axis_idx * inner_size + inner
            sum += @data[input_idx]
          end
          output_idx = outer * inner_size + inner
          result[output_idx] = sum / axis_size.to_f
        end
      end

      Tensor.new(result, output_shape, dtype: @dtype, device: @device)
    end

    # Convert tensor to different device
    # @param new_device [String] Target device
    # @return [Tensor] Tensor on new device
    def to(new_device)
      # In a real implementation, this would handle device transfer
      Tensor.new(@data.dup, @shape, dtype: @dtype, device: new_device)
    end

    # Convert tensor to different data type
    # @param new_dtype [Symbol] Target data type
    # @return [Tensor] Tensor with new data type
    def astype(new_dtype)
      # In a real implementation, this would handle type conversion
      Tensor.new(@data.dup, @shape, dtype: new_dtype, device: @device)
    end

    # Get tensor as nested array
    # @return [Array] Nested array representation
    def to_a
      return @data.dup if ndim == 1

      result = @data.dup
      @shape.reverse.each_with_index do |dim_size, i|
        next if i == 0 # Skip the last dimension
        
        result = result.each_slice(dim_size).to_a
      end
      
      result
    end

    # String representation
    # @return [String] Human-readable tensor representation
    def to_s
      "Tensor(shape: #{@shape}, dtype: #{@dtype}, device: #{@device})"
    end

    # Detailed string representation
    # @return [String] Detailed tensor information
    def inspect
      data_preview = if @data.size <= 10
                      @data.inspect
                     else
                       "#{@data.first(5).inspect}...#{@data.last(2).inspect}"
                     end

      "#{to_s}\nData: #{data_preview}"
    end

    # Convert to native C tensor for FFI calls
    # @api private
    def to_native
      Native.array_to_tensor(@data, @shape)
    end

    # Create tensor from native C tensor
    # @param native_tensor [Native::TrustformersTensor] Native tensor
    # @return [Tensor] Ruby tensor
    # @api private
    def self.from_native(native_tensor)
      data, shape = Native.tensor_to_array(native_tensor)
      new(data, shape)
    end

    private

    # Calculate flat index from multi-dimensional indices
    def calculate_flat_index(indices)
      unless indices.size == ndim
        raise ArgumentError, "Expected #{ndim} indices, got #{indices.size}"
      end

      flat_index = 0
      stride = 1
      
      (@shape.size - 1).downto(0) do |i|
        flat_index += indices[i] * stride
        stride *= @shape[i]
      end

      flat_index
    end

    # Check if shapes are compatible for element-wise operations
    def check_compatible_shapes!(other)
      unless @shape == other.shape
        raise TensorShapeError, "Incompatible shapes: #{@shape} and #{other.shape}"
      end
    end

    # Validate tensor properties
    def validate!
      unless SUPPORTED_DTYPES.include?(@dtype)
        raise ArgumentError, "Unsupported dtype: #{@dtype}"
      end

      unless SUPPORTED_DEVICES.include?(@device)
        raise ArgumentError, "Unsupported device: #{@device}"
      end

      expected_size = @shape.reduce(1, :*)
      unless @data.size == expected_size
        raise TensorShapeError, "Data size #{@data.size} doesn't match shape #{@shape} (expected #{expected_size})"
      end

      unless @shape.all? { |dim| dim > 0 }
        raise ArgumentError, "All dimensions must be positive"
      end
    end

    # Generate normal random number using Box-Muller transform
    def self.normal_random(mean = 0.0, std = 1.0)
      @spare_normal ||= nil
      
      if @spare_normal
        result = @spare_normal
        @spare_normal = nil
        return mean + std * result
      end

      u1 = rand
      u2 = rand
      
      z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math::PI * u2)
      z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math::PI * u2)
      
      @spare_normal = z1
      mean + std * z0
    end
  end
end