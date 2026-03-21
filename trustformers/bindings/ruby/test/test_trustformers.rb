# frozen_string_literal: true

require "minitest/autorun"
require "trustformers"

class TestTrustFormeRS < Minitest::Test
  def setup
    # Skip tests if TrustFormeRS library is not available
    skip "TrustFormeRS library not available" unless library_available?
  end

  def test_version
    version = TrustFormeRS.version
    refute_nil version
    refute_empty version
    assert_match(/\d+\.\d+\.\d+/, version)
  end

  def test_system_info
    info = TrustFormeRS.system_info
    assert_kind_of Hash, info
    assert_includes info.keys, :version
    assert_includes info.keys, :devices
    assert_includes info.keys, :ruby_version
    assert_includes info.keys, :platform
  end

  def test_available_devices
    devices = TrustFormeRS.available_devices
    assert_kind_of Array, devices
    assert_includes devices, "cpu"
  end

  def test_client_initialization_default
    client = TrustFormeRS.new
    assert_kind_of TrustFormeRS::Client, client
    assert_kind_of TrustFormeRS::Configuration, client.config
  ensure
    client&.close
  end

  def test_client_initialization_with_config
    config = TrustFormeRS::Configuration.new(
      use_gpu: false,
      device: "cpu",
      num_threads: 2,
      enable_logging: true
    )
    
    client = TrustFormeRS.new(config)
    assert_equal config, client.config
    assert_equal "cpu", client.config.device
    assert_equal 2, client.config.num_threads
    assert client.config.enable_logging
  ensure
    client&.close
  end

  def test_client_initialization_with_hash
    client = TrustFormeRS.new(
      use_gpu: false,
      device: "cpu",
      num_threads: 1,
      enable_logging: false
    )
    
    assert_equal "cpu", client.config.device
    assert_equal 1, client.config.num_threads
    refute client.config.enable_logging
  ensure
    client&.close
  end

  def test_memory_stats
    client = TrustFormeRS.new
    stats = client.memory_stats
    assert_kind_of Hash, stats
    # Memory stats structure may vary, just check it's a hash
  ensure
    client&.close
  end

  def test_current_device
    client = TrustFormeRS.new(device: "cpu")
    device = client.current_device
    assert_kind_of String, device
  ensure
    client&.close
  end

  def test_gpu_methods
    # These methods should not raise errors even if GPU is not available
    assert_boolean TrustFormeRS.gpu_available?
    assert_boolean TrustFormeRS.cuda_available?
    assert_boolean TrustFormeRS.metal_available?
  end

  private

  def library_available?
    # Check if the native library is available
    TrustFormeRS.version
    true
  rescue LoadError, NoMethodError
    false
  end

  def assert_boolean(value)
    assert value == true || value == false, "Expected boolean, got #{value.class}"
  end
end