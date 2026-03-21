# frozen_string_literal: true

require "minitest/autorun"
require "trustformers"

class TestTensor < Minitest::Test
  def test_tensor_creation
    data = [1, 2, 3, 4, 5, 6]
    shape = [2, 3]
    tensor = TrustFormeRS::Tensor.new(data, shape)

    assert_equal data, tensor.data
    assert_equal shape, tensor.shape
    assert_equal :float32, tensor.dtype
    assert_equal "cpu", tensor.device
    assert_equal 2, tensor.ndim
    assert_equal 6, tensor.size
  end

  def test_from_array_methods
    # 1D tensor
    array_1d = [1, 2, 3]
    tensor_1d = TrustFormeRS::Tensor.from_array(array_1d)
    assert_equal [3], tensor_1d.shape
    assert_equal array_1d, tensor_1d.data

    # 2D tensor
    array_2d = [[1, 2], [3, 4]]
    tensor_2d = TrustFormeRS::Tensor.from_2d(array_2d)
    assert_equal [2, 2], tensor_2d.shape
    assert_equal [1, 2, 3, 4], tensor_2d.data

    # 3D tensor
    array_3d = [[[1, 2]], [[3, 4]]]
    tensor_3d = TrustFormeRS::Tensor.from_3d(array_3d)
    assert_equal [2, 1, 2], tensor_3d.shape
    assert_equal [1, 2, 3, 4], tensor_3d.data
  end

  def test_zeros_and_ones
    # Zeros tensor
    zeros = TrustFormeRS::Tensor.zeros([2, 3])
    assert_equal [2, 3], zeros.shape
    assert zeros.data.all? { |x| x == 0.0 }

    # Ones tensor
    ones = TrustFormeRS::Tensor.ones([2, 3])
    assert_equal [2, 3], ones.shape
    assert ones.data.all? { |x| x == 1.0 }
  end

  def test_random_tensor
    # Uniform random
    random_uniform = TrustFormeRS::Tensor.random([10, 10], distribution: :uniform)
    assert_equal [10, 10], random_uniform.shape
    assert_equal 100, random_uniform.size
    assert random_uniform.data.all? { |x| x >= 0 && x <= 1 }

    # Normal random
    random_normal = TrustFormeRS::Tensor.random([5, 5], distribution: :normal)
    assert_equal [5, 5], random_normal.shape
    assert_equal 25, random_normal.size
  end

  def test_eye_matrix
    eye = TrustFormeRS::Tensor.eye(3)
    assert_equal [3, 3], eye.shape
    
    # Check diagonal elements are 1
    (0...3).each do |i|
      assert_equal 1.0, eye[i, i]
    end

    # Check off-diagonal elements are 0
    (0...3).each do |i|
      (0...3).each do |j|
        next if i == j
        assert_equal 0.0, eye[i, j]
      end
    end
  end

  def test_tensor_indexing
    tensor = TrustFormeRS::Tensor.from_2d([[1, 2, 3], [4, 5, 6]])
    
    assert_equal 1, tensor[0, 0]
    assert_equal 2, tensor[0, 1]
    assert_equal 3, tensor[0, 2]
    assert_equal 4, tensor[1, 0]
    assert_equal 5, tensor[1, 1]
    assert_equal 6, tensor[1, 2]
  end

  def test_tensor_assignment
    tensor = TrustFormeRS::Tensor.zeros([2, 2])
    
    tensor[0, 0] = 5.0
    tensor[1, 1] = 10.0
    
    assert_equal 5.0, tensor[0, 0]
    assert_equal 10.0, tensor[1, 1]
    assert_equal 0.0, tensor[0, 1]
    assert_equal 0.0, tensor[1, 0]
  end

  def test_reshape
    tensor = TrustFormeRS::Tensor.from_array([1, 2, 3, 4, 5, 6])
    
    # Valid reshape
    reshaped = tensor.reshape([2, 3])
    assert_equal [2, 3], reshaped.shape
    assert_equal tensor.data, reshaped.data

    # Invalid reshape should raise error
    assert_raises TrustFormeRS::TensorShapeError do
      tensor.reshape([2, 2])
    end
  end

  def test_transpose
    tensor = TrustFormeRS::Tensor.from_2d([[1, 2, 3], [4, 5, 6]])
    transposed = tensor.transpose

    assert_equal [3, 2], transposed.shape
    assert_equal 1, transposed[0, 0]
    assert_equal 4, transposed[0, 1]
    assert_equal 2, transposed[1, 0]
    assert_equal 5, transposed[1, 1]
    assert_equal 3, transposed[2, 0]
    assert_equal 6, transposed[2, 1]
  end

  def test_squeeze_unsqueeze
    # Squeeze
    tensor = TrustFormeRS::Tensor.new([1, 2, 3], [1, 3, 1])
    squeezed = tensor.squeeze
    assert_equal [3], squeezed.shape

    squeezed_dim = tensor.squeeze(0)
    assert_equal [3, 1], squeezed_dim.shape

    # Unsqueeze
    tensor_1d = TrustFormeRS::Tensor.from_array([1, 2, 3])
    unsqueezed = tensor_1d.unsqueeze(0)
    assert_equal [1, 3], unsqueezed.shape

    unsqueezed2 = tensor_1d.unsqueeze(1)
    assert_equal [3, 1], unsqueezed2.shape
  end

  def test_arithmetic_operations
    tensor1 = TrustFormeRS::Tensor.from_array([1, 2, 3])
    tensor2 = TrustFormeRS::Tensor.from_array([4, 5, 6])

    # Addition
    sum_tensor = tensor1 + tensor2
    assert_equal [5, 7, 9], sum_tensor.data

    # Subtraction
    diff_tensor = tensor2 - tensor1
    assert_equal [3, 3, 3], diff_tensor.data

    # Multiplication
    mult_tensor = tensor1 * tensor2
    assert_equal [4, 10, 18], mult_tensor.data

    # Division
    div_tensor = tensor2 / tensor1
    assert_equal [4.0, 2.5, 2.0], div_tensor.data
  end

  def test_matrix_multiplication
    tensor1 = TrustFormeRS::Tensor.from_2d([[1, 2], [3, 4]])
    tensor2 = TrustFormeRS::Tensor.from_2d([[5, 6], [7, 8]])

    result = tensor1.matmul(tensor2)
    assert_equal [2, 2], result.shape

    # Expected: [[19, 22], [43, 50]]
    assert_in_delta 19, result[0, 0], 0.001
    assert_in_delta 22, result[0, 1], 0.001
    assert_in_delta 43, result[1, 0], 0.001
    assert_in_delta 50, result[1, 1], 0.001
  end

  def test_sum_and_mean
    tensor = TrustFormeRS::Tensor.from_array([1, 2, 3, 4, 5])

    # Sum
    sum_result = tensor.sum
    assert_equal [1], sum_result.shape
    assert_equal 15, sum_result.data.first

    # Mean
    mean_result = tensor.mean
    assert_equal [1], mean_result.shape
    assert_equal 3.0, mean_result.data.first
  end

  def test_device_and_dtype_conversion
    tensor = TrustFormeRS::Tensor.from_array([1, 2, 3])
    
    # Device conversion
    gpu_tensor = tensor.to("gpu")
    assert_equal "gpu", gpu_tensor.device
    assert_equal tensor.data, gpu_tensor.data

    # Data type conversion
    int_tensor = tensor.astype(:int32)
    assert_equal :int32, int_tensor.dtype
    assert_equal tensor.data, int_tensor.data
  end

  def test_to_array
    # 1D tensor
    tensor_1d = TrustFormeRS::Tensor.from_array([1, 2, 3])
    assert_equal [1, 2, 3], tensor_1d.to_a

    # 2D tensor
    tensor_2d = TrustFormeRS::Tensor.from_2d([[1, 2], [3, 4]])
    expected_2d = [[1, 2], [3, 4]]
    assert_equal expected_2d, tensor_2d.to_a
  end

  def test_tensor_validation
    # Valid tensor should not raise
    assert_silent do
      TrustFormeRS::Tensor.new([1, 2, 3, 4], [2, 2])
    end

    # Invalid shape should raise
    assert_raises TrustFormeRS::TensorShapeError do
      TrustFormeRS::Tensor.new([1, 2, 3], [2, 2])
    end

    # Invalid dtype should raise
    assert_raises ArgumentError do
      TrustFormeRS::Tensor.new([1, 2, 3], [3], dtype: :invalid_type)
    end

    # Invalid device should raise
    assert_raises ArgumentError do
      TrustFormeRS::Tensor.new([1, 2, 3], [3], device: "invalid_device")
    end

    # Negative dimensions should raise
    assert_raises ArgumentError do
      TrustFormeRS::Tensor.new([1, 2, 3], [-1, 3])
    end
  end

  def test_incompatible_shapes_error
    tensor1 = TrustFormeRS::Tensor.from_array([1, 2, 3])
    tensor2 = TrustFormeRS::Tensor.from_array([1, 2])

    assert_raises TrustFormeRS::TensorShapeError do
      tensor1 + tensor2
    end

    assert_raises TrustFormeRS::TensorShapeError do
      tensor1 - tensor2
    end

    assert_raises TrustFormeRS::TensorShapeError do
      tensor1 * tensor2
    end
  end

  def test_tensor_string_representation
    tensor = TrustFormeRS::Tensor.from_2d([[1, 2], [3, 4]])
    
    str_repr = tensor.to_s
    assert_includes str_repr, "Tensor"
    assert_includes str_repr, "shape: [2, 2]"
    assert_includes str_repr, "dtype: float32"
    assert_includes str_repr, "device: cpu"

    inspect_repr = tensor.inspect
    assert_includes inspect_repr, str_repr
    assert_includes inspect_repr, "Data:"
  end

  def test_data_type_sizes
    assert_equal 4, TrustFormeRS::Tensor::SUPPORTED_DTYPES.include?(:float32)
    assert_equal 4, TrustFormeRS::Tensor::SUPPORTED_DTYPES.include?(:int32)
    assert_equal 8, TrustFormeRS::Tensor::SUPPORTED_DTYPES.include?(:float64)
    assert_equal 8, TrustFormeRS::Tensor::SUPPORTED_DTYPES.include?(:int64)
    assert_equal 1, TrustFormeRS::Tensor::SUPPORTED_DTYPES.include?(:bool)
  end
end