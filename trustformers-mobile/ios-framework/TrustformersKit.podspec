Pod::Spec.new do |spec|
  spec.name             = 'TrustformersKit'
  spec.version          = '1.0.0'
  spec.summary          = 'High-performance mobile machine learning inference framework'
  spec.description      = <<-DESC
                         TrustformersKit provides optimized on-device machine learning inference for iOS applications.
                         Features include Core ML integration, Metal Performance Shaders, privacy-preserving inference,
                         federated learning, and comprehensive performance monitoring.
                         DESC

  spec.homepage         = 'https://github.com/cool-japan/trustformers'
  spec.license          = { :type => 'MIT', :file => 'LICENSE' }
  spec.author           = { 'TrustformeRS Team' => 'team@trustformers.dev' }
  spec.source           = { :git => 'https://github.com/cool-japan/trustformers.git', :tag => spec.version.to_s }

  # Platform requirements
  spec.ios.deployment_target = '11.0'
  spec.swift_version = '5.5'

  # Source files
  spec.source_files = 'TrustformersKit/Sources/**/*.{swift,h,m}'
  spec.public_header_files = 'TrustformersKit/Headers/**/*.h'
  spec.module_map = 'TrustformersKit/Headers/module.modulemap'

  # Resources
  spec.resource_bundles = {
    'TrustformersKit' => ['TrustformersKit/Resources/**/*']
  }

  # Framework dependencies
  spec.frameworks = [
    'Foundation',
    'UIKit', 
    'CoreML',
    'Metal',
    'MetalPerformanceShaders',
    'Accelerate',
    'ARKit',
    'AVFoundation',
    'Vision',
    'CoreImage',
    'CloudKit'
  ]

  # System library dependencies
  spec.libraries = ['c++', 'z']

  # Compiler flags
  spec.compiler_flags = '-DTRUSTFORMERS_COCOAPODS=1'
  spec.pod_target_xcconfig = {
    'SWIFT_COMPILATION_MODE' => 'wholemodule',
    'SWIFT_OPTIMIZATION_LEVEL' => '-O',
    'GCC_OPTIMIZATION_LEVEL' => '3',
    'ENABLE_BITCODE' => 'YES',
    'VALID_ARCHS' => 'arm64 arm64e x86_64',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }

  # Binary framework (pre-built Rust core)
  spec.vendored_frameworks = 'TrustformersKit.xcframework'

  # Subspecs for optional features
  spec.subspec 'Core' do |core|
    core.source_files = 'TrustformersKit/Sources/Core/**/*.swift'
    core.dependency 'TrustformersKit/Headers'
  end

  spec.subspec 'Performance' do |perf|
    perf.source_files = 'TrustformersKit/Sources/Performance/**/*.swift'
    perf.dependency 'TrustformersKit/Core'
  end

  spec.subspec 'Privacy' do |privacy|
    privacy.source_files = 'TrustformersKit/Sources/Privacy/**/*.swift'
    privacy.dependency 'TrustformersKit/Core'
  end

  spec.subspec 'ARSupport' do |ar|
    ar.source_files = 'TrustformersKit/Sources/AR/**/*.swift'
    ar.dependency 'TrustformersKit/Core'
  end

  # Test spec
  spec.test_spec 'Tests' do |test_spec|
    test_spec.source_files = 'Tests/**/*.{swift,m}'
    test_spec.frameworks = ['XCTest']
  end

  # App Extension support
  spec.pod_target_xcconfig = {
    'APPLICATION_EXTENSION_API_ONLY' => 'YES'
  }

  # Minimum Xcode version
  spec.requires_arc = true
  
  # Documentation
  spec.documentation_url = 'https://docs.trustformers.dev/mobile/ios'
end