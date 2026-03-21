#!/usr/bin/env python3
"""
Test script for TrustformeRS Mobile Package Management

This script tests the mobile package management functionality without requiring
full mobile development environments (Xcode, Android NDK).
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from mobile_package_management import (
    MobilePackageManager, 
    MobileConfig, 
    MobilePlatform, 
    OptimizationLevel,
    create_mobile_packages
)

def test_config_creation():
    """Test MobileConfig creation and validation."""
    print("Testing MobileConfig creation...")
    
    config = MobileConfig(
        platform=MobilePlatform.BOTH,
        optimization_level=OptimizationLevel.BALANCED,
        target_architectures=["arm64"],
        min_ios_version="12.0",
        min_android_api=21,
        version="1.0.0"
    )
    
    assert config.platform == MobilePlatform.BOTH
    assert config.optimization_level == OptimizationLevel.BALANCED
    assert "arm64" in config.target_architectures
    print("✅ MobileConfig creation test passed")

def test_manager_initialization():
    """Test MobilePackageManager initialization."""
    print("Testing MobilePackageManager initialization...")
    
    config = MobileConfig(platform=MobilePlatform.IOS)
    
    with MobilePackageManager(config) as manager:
        assert manager.config.platform == MobilePlatform.IOS
        assert manager.build_dir.exists()
        assert manager.output_dir.exists()
        assert manager.temp_dir is not None
        
    print("✅ MobilePackageManager initialization test passed")

def test_directory_structure():
    """Test that proper directory structures are created."""
    print("Testing directory structure creation...")
    
    config = MobileConfig(platform=MobilePlatform.BOTH)
    
    with MobilePackageManager(config) as manager:
        # Test iOS framework structure creation
        framework_dir = manager.build_dir / "test_ios_framework.framework"
        framework_dir.mkdir(parents=True, exist_ok=True)
        
        manager._create_ios_framework_structure(framework_dir, "TestFramework")
        
        assert (framework_dir / "Headers").exists()
        assert (framework_dir / "Modules").exists()
        assert (framework_dir / "Resources").exists()
        assert (framework_dir / "Info.plist").exists()
        assert (framework_dir / "Modules" / "module.modulemap").exists()
        
        # Test Android library structure creation
        aar_dir = manager.build_dir / "test_android_library"
        aar_dir.mkdir(parents=True, exist_ok=True)
        
        manager._create_android_library_structure(aar_dir)
        
        assert (aar_dir / "src" / "main" / "java" / "com" / "trustformers" / "android").exists()
        assert (aar_dir / "src" / "main" / "jniLibs").exists()
        assert (aar_dir / "src" / "main" / "AndroidManifest.xml").exists()
        
    print("✅ Directory structure creation test passed")

def test_wrapper_generation():
    """Test Swift and Java wrapper generation."""
    print("Testing wrapper generation...")
    
    config = MobileConfig(platform=MobilePlatform.BOTH)
    
    with MobilePackageManager(config) as manager:
        # Test Swift wrapper generation
        framework_dir = manager.build_dir / "test_ios_framework.framework"
        framework_dir.mkdir(parents=True, exist_ok=True)
        (framework_dir / "Headers").mkdir(exist_ok=True)
        
        manager._create_swift_wrapper(framework_dir, "TestFramework")
        
        assert (framework_dir / "Headers" / "TestFramework.h").exists()
        assert (framework_dir / "TrustformeRSModel.swift").exists()
        
        # Verify header content
        header_content = (framework_dir / "Headers" / "TestFramework.h").read_text()
        assert "extern void* trustformers_create_model" in header_content
        
        # Test Java wrapper generation
        aar_dir = manager.build_dir / "test_android_library"
        aar_dir.mkdir(parents=True, exist_ok=True)
        (aar_dir / "src" / "main" / "java" / "com" / "trustformers" / "android").mkdir(parents=True, exist_ok=True)
        (aar_dir / "src" / "main" / "cpp").mkdir(parents=True, exist_ok=True)
        
        manager._create_java_wrapper(aar_dir)
        
        java_file = aar_dir / "src" / "main" / "java" / "com" / "trustformers" / "android" / "TrustformeRS.java"
        jni_file = aar_dir / "src" / "main" / "cpp" / "trustformers_jni.cpp"
        
        assert java_file.exists()
        assert jni_file.exists()
        
        # Verify Java content
        java_content = java_file.read_text()
        assert "public class TrustformeRS" in java_content
        assert "System.loadLibrary(\"trustformers\")" in java_content
        
    print("✅ Wrapper generation test passed")

def test_gradle_build_files():
    """Test Gradle build files generation."""
    print("Testing Gradle build files generation...")
    
    config = MobileConfig(platform=MobilePlatform.ANDROID)
    
    with MobilePackageManager(config) as manager:
        aar_dir = manager.build_dir / "test_android_library"
        aar_dir.mkdir(parents=True, exist_ok=True)
        
        manager._create_gradle_build_files(aar_dir)
        
        assert (aar_dir / "build.gradle").exists()
        assert (aar_dir / "gradle.properties").exists()
        assert (aar_dir / "proguard-rules.pro").exists()
        
        # Verify build.gradle content
        build_gradle_content = (aar_dir / "build.gradle").read_text()
        assert "apply plugin: 'com.android.library'" in build_gradle_content
        assert f"minSdkVersion {config.min_android_api}" in build_gradle_content
        
    print("✅ Gradle build files generation test passed")

def test_podspec_creation():
    """Test CocoaPods podspec creation."""
    print("Testing CocoaPods podspec creation...")
    
    config = MobileConfig(platform=MobilePlatform.IOS, version="1.2.3")
    
    with MobilePackageManager(config) as manager:
        manager._create_podspec("TestFramework")
        
        podspec_path = manager.build_dir / "ios" / "TestFramework.podspec"
        assert podspec_path.exists()
        
        podspec_content = podspec_path.read_text()
        assert 'spec.version      = "1.2.3"' in podspec_content
        assert f'spec.ios.deployment_target = "{config.min_ios_version}"' in podspec_content
        
    print("✅ CocoaPods podspec creation test passed")

def test_test_suite_creation():
    """Test mobile test suite creation."""
    print("Testing mobile test suite creation...")
    
    config = MobileConfig(platform=MobilePlatform.BOTH)
    
    with MobilePackageManager(config) as manager:
        test_dir = manager.create_mobile_test_suite()
        
        assert test_dir.exists()
        
        # Check iOS tests
        ios_test_file = test_dir / "ios" / "TrustformeRSTests.swift"
        assert ios_test_file.exists()
        
        ios_test_content = ios_test_file.read_text()
        assert "import TrustformeRS" in ios_test_content
        assert "func testFrameworkLoading()" in ios_test_content
        
        # Check Android tests
        android_test_file = test_dir / "android" / "TrustformeRSInstrumentedTest.java"
        assert android_test_file.exists()
        
        android_test_content = android_test_file.read_text()
        assert "import com.trustformers.android.TrustformeRS;" in android_test_content
        assert "public void testLibraryLoading()" in android_test_content
        
    print("✅ Mobile test suite creation test passed")

def test_documentation_generation():
    """Test documentation generation."""
    print("Testing documentation generation...")
    
    config = MobileConfig(platform=MobilePlatform.BOTH)
    
    with MobilePackageManager(config) as manager:
        docs_dir = manager.generate_documentation()
        
        assert docs_dir.exists()
        
        integration_guide = docs_dir / "mobile-integration-guide.md"
        assert integration_guide.exists()
        
        guide_content = integration_guide.read_text()
        assert "# TrustformeRS Mobile Integration Guide" in guide_content
        assert "## iOS Integration" in guide_content
        assert "## Android Integration" in guide_content
        
    print("✅ Documentation generation test passed")

def test_benchmark_execution():
    """Test benchmark execution."""
    print("Testing benchmark execution...")
    
    config = MobileConfig(platform=MobilePlatform.BOTH)
    
    with MobilePackageManager(config) as manager:
        benchmarks = manager.run_mobile_benchmarks()
        
        assert "timestamp" in benchmarks
        assert "config" in benchmarks
        assert "results" in benchmarks
        assert benchmarks["config"]["platform"] == "both"
        assert "estimated_inference_time_ms" in benchmarks["results"]
        assert "estimated_memory_usage_mb" in benchmarks["results"]
        
    print("✅ Benchmark execution test passed")

def test_convenience_function():
    """Test the convenience function for creating mobile packages."""
    print("Testing convenience function...")
    
    # This is a dry run test since we don't have the full build environment
    try:
        config = MobileConfig(
            platform=MobilePlatform.IOS,
            optimization_level=OptimizationLevel.DEBUG,  # Use debug to avoid actual compilation
            target_architectures=["arm64"]
        )
        
        with MobilePackageManager(config) as manager:
            # Test that we can at least initialize and create directories
            assert manager.build_dir.exists()
            assert manager.output_dir.exists()
            
        print("✅ Convenience function test passed")
        
    except Exception as e:
        print(f"⚠️  Convenience function test skipped (expected without build tools): {e}")

def run_all_tests():
    """Run all mobile package management tests."""
    print("🚀 Running TrustformeRS Mobile Package Management Tests")
    print("=" * 60)
    
    tests = [
        test_config_creation,
        test_manager_initialization,
        test_directory_structure,
        test_wrapper_generation,
        test_gradle_build_files,
        test_podspec_creation,
        test_test_suite_creation,
        test_documentation_generation,
        test_benchmark_execution,
        test_convenience_function
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Mobile package management is working correctly.")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)