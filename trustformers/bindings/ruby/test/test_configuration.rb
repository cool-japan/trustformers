# frozen_string_literal: true

require "minitest/autorun"
require "trustformers"

class TestConfiguration < Minitest::Test
  def test_default_configuration
    config = TrustFormeRS::Configuration.default
    refute config.use_gpu
    assert_equal "auto", config.device
    assert_equal -1, config.num_threads
    refute config.enable_logging
    assert_nil config.cache_dir
  end

  def test_cpu_configuration
    config = TrustFormeRS::Configuration.cpu
    refute config.use_gpu
    assert_equal "cpu", config.device
    assert_equal -1, config.num_threads
    refute config.enable_logging
  end

  def test_gpu_configuration
    config = TrustFormeRS::Configuration.gpu
    assert config.use_gpu
    assert_equal "auto", config.device
    assert_equal 4, config.num_threads
    refute config.enable_logging
  end

  def test_development_configuration
    config = TrustFormeRS::Configuration.development
    refute config.use_gpu
    assert_equal "cpu", config.device
    assert_equal 2, config.num_threads
    assert config.enable_logging
    assert_equal "./cache", config.cache_dir
  end

  def test_custom_configuration
    config = TrustFormeRS::Configuration.new(
      use_gpu: true,
      device: "cuda",
      num_threads: 8,
      enable_logging: true,
      cache_dir: "/tmp/cache"
    )

    assert config.use_gpu
    assert_equal "cuda", config.device
    assert_equal 8, config.num_threads
    assert config.enable_logging
    assert_equal "/tmp/cache", config.cache_dir
  end

  def test_configuration_validation
    # Valid configuration should not raise
    assert_silent do
      TrustFormeRS::Configuration.new(
        use_gpu: false,
        device: "cpu",
        num_threads: 4
      )
    end

    # Invalid device should raise
    assert_raises TrustFormeRS::ConfigurationError do
      TrustFormeRS::Configuration.new(device: "invalid_device")
    end

    # Invalid num_threads should raise
    assert_raises TrustFormeRS::ConfigurationError do
      TrustFormeRS::Configuration.new(num_threads: -2)
    end

    # Conflicting GPU settings should raise
    assert_raises TrustFormeRS::ConfigurationError do
      TrustFormeRS::Configuration.new(use_gpu: true, device: "cpu")
    end
  end

  def test_to_hash
    config = TrustFormeRS::Configuration.new(
      use_gpu: true,
      device: "gpu",
      num_threads: 4,
      enable_logging: false,
      cache_dir: nil
    )

    hash = config.to_h
    expected = {
      use_gpu: true,
      device: "gpu",
      num_threads: 4,
      enable_logging: false,
      cache_dir: nil
    }

    assert_equal expected, hash
  end

  def test_to_json
    config = TrustFormeRS::Configuration.new(
      use_gpu: false,
      device: "cpu",
      num_threads: 2
    )

    json = config.to_json
    assert_kind_of String, json
    
    # Parse JSON and verify structure
    require "json"
    parsed = JSON.parse(json)
    assert_equal false, parsed["use_gpu"]
    assert_equal "cpu", parsed["device"]
    assert_equal 2, parsed["num_threads"]
  end

  def test_effective_device
    # Auto device resolution
    config = TrustFormeRS::Configuration.new(device: "auto", use_gpu: false)
    assert_equal "cpu", config.effective_device

    # Specific device
    config = TrustFormeRS::Configuration.new(device: "cpu")
    assert_equal "cpu", config.effective_device
  end

  def test_enable_disable_gpu
    config = TrustFormeRS::Configuration.new(use_gpu: false, device: "cpu")

    # This might raise if GPU is not available, so we'll handle that
    begin
      config.enable_gpu!("auto")
      assert config.use_gpu
      assert_equal "auto", config.device
    rescue TrustFormeRS::DeviceError
      # GPU not available, which is fine for testing
      skip "GPU not available for testing"
    end

    config.disable_gpu!
    refute config.use_gpu
    assert_equal "cpu", config.device
  end

  def test_threads_assignment
    config = TrustFormeRS::Configuration.new

    config.threads = 8
    assert_equal 8, config.num_threads

    assert_raises TrustFormeRS::ConfigurationError do
      config.threads = 0
    end

    assert_raises TrustFormeRS::ConfigurationError do
      config.threads = -2
    end
  end

  def test_logging_methods
    config = TrustFormeRS::Configuration.new(enable_logging: false)

    config.enable_logging!
    assert config.enable_logging

    config.disable_logging!
    refute config.enable_logging
  end

  def test_cache_dir_assignment
    config = TrustFormeRS::Configuration.new

    # Valid cache directory (will be created)
    require "tmpdir"
    temp_dir = Dir.mktmpdir
    config.cache_dir = temp_dir
    assert_equal temp_dir, config.cache_dir

    # Nil cache directory
    config.cache_dir = nil
    assert_nil config.cache_dir
  ensure
    FileUtils.rm_rf(temp_dir) if temp_dir
  end

  def test_summary
    config = TrustFormeRS::Configuration.new(
      use_gpu: false,
      device: "cpu",
      num_threads: 4,
      enable_logging: true,
      cache_dir: "/tmp"
    )

    summary = config.summary
    assert_kind_of String, summary
    assert_includes summary, "TrustFormeRS Configuration"
    assert_includes summary, "Device: cpu"
    assert_includes summary, "Threads: 4"
    assert_includes summary, "Logging: enabled"
    assert_includes summary, "Cache: /tmp"
  end

  def test_valid_for_system
    # CPU config should always be valid
    config = TrustFormeRS::Configuration.cpu
    assert config.valid_for_system?

    # GPU config validity depends on system capabilities
    gpu_config = TrustFormeRS::Configuration.gpu
    if TrustFormeRS.gpu_available?
      assert gpu_config.valid_for_system?
    else
      refute gpu_config.valid_for_system?
    end
  end
end