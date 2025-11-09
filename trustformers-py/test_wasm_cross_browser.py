#!/usr/bin/env python3
"""
Test script for Cross-Browser WASM Testing Suite

This script tests the cross-browser WASM testing functionality without requiring
actual browser installations or WebDriver dependencies.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from cross_browser_wasm_testing import (
    CrossBrowserWASMTester,
    BrowserConfig,
    BrowserType,
    TestCategory,
    TestResult,
    run_cross_browser_tests
)

def test_browser_config_creation():
    """Test BrowserConfig creation and validation."""
    print("Testing BrowserConfig creation...")
    
    config = BrowserConfig(
        browser_type=BrowserType.CHROME,
        headless=True,
        window_size=(1920, 1080),
        enable_logging=True,
        timeout=30
    )
    
    assert config.browser_type == BrowserType.CHROME
    assert config.headless == True
    assert config.window_size == (1920, 1080)
    assert config.timeout == 30
    print("✅ BrowserConfig creation test passed")

def test_test_result_creation():
    """Test TestResult creation."""
    print("Testing TestResult creation...")
    
    result = TestResult(
        browser=BrowserType.FIREFOX,
        test_name="WASM Compatibility Test",
        category=TestCategory.COMPATIBILITY,
        passed=True,
        duration=1.5,
        performance_metrics={"load_time": 500, "inference_time": 100},
        memory_usage={"peak_mb": 85, "average_mb": 70}
    )
    
    assert result.browser == BrowserType.FIREFOX
    assert result.test_name == "WASM Compatibility Test"
    assert result.category == TestCategory.COMPATIBILITY
    assert result.passed == True
    assert result.duration == 1.5
    assert "load_time" in result.performance_metrics
    print("✅ TestResult creation test passed")

def test_tester_initialization():
    """Test CrossBrowserWASMTester initialization."""
    print("Testing CrossBrowserWASMTester initialization...")
    
    tester = CrossBrowserWASMTester()
    
    assert tester.project_root.exists()
    assert tester.test_dir.exists()
    assert tester.output_dir.exists()
    assert tester.server_port == 8888
    assert tester.server_host == "localhost"
    assert isinstance(tester.config, dict)
    assert isinstance(tester.browser_configs, dict)
    print("✅ CrossBrowserWASMTester initialization test passed")

def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    # Create temporary config file
    temp_config_path = Path("/tmp/wasm_test_config.json")
    test_config = {
        "browsers": ["chrome", "firefox"],
        "test_categories": ["compatibility", "performance"],
        "performance_thresholds": {
            "load_time_ms": 3000,
            "inference_time_ms": 500
        }
    }
    
    with open(temp_config_path, 'w') as f:
        json.dump(test_config, f)
    
    try:
        tester = CrossBrowserWASMTester(config_file=temp_config_path)
        
        assert "chrome" in tester.config["browsers"]
        assert "firefox" in tester.config["browsers"]
        assert tester.config["performance_thresholds"]["load_time_ms"] == 3000
        
        print("✅ Configuration loading test passed")
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()

def test_test_file_creation():
    """Test test file creation."""
    print("Testing test file creation...")
    
    tester = CrossBrowserWASMTester()
    tester.create_test_files()
    
    # Check if test files were created
    assert (tester.test_dir / "html" / "test.html").exists()
    assert (tester.test_dir / "js" / "wasm-test-suite.js").exists()
    assert (tester.test_dir / "html" / "mobile-test.html").exists()
    
    # Verify HTML content
    html_content = (tester.test_dir / "html" / "test.html").read_text()
    assert "TrustformeRS WASM Test Suite" in html_content
    assert "runAllTests()" in html_content
    
    # Verify JS content
    js_content = (tester.test_dir / "js" / "wasm-test-suite.js").read_text()
    assert "class WASMTestSuite" in js_content
    assert "checkWASMSupport" in js_content
    
    # Verify mobile HTML content
    mobile_content = (tester.test_dir / "html" / "mobile-test.html").read_text()
    assert "Mobile WASM Test" in mobile_content
    assert "runMobileCompatibilityTest" in mobile_content
    
    print("✅ Test file creation test passed")

def test_browser_config_creation():
    """Test browser configuration creation."""
    print("Testing browser configuration creation...")
    
    tester = CrossBrowserWASMTester()
    configs = tester._create_browser_configs()
    
    # Should have configs for all supported browsers
    assert len(configs) > 0
    
    for browser_type, config in configs.items():
        assert isinstance(browser_type, BrowserType)
        assert isinstance(config, BrowserConfig)
        assert config.browser_type == browser_type
        assert config.timeout == tester.config["test_timeout"]
    
    print("✅ Browser configuration creation test passed")

def test_html_report_generation():
    """Test HTML report generation."""
    print("Testing HTML report generation...")
    
    tester = CrossBrowserWASMTester()
    
    # Create mock test results
    mock_results = [
        TestResult(
            browser=BrowserType.CHROME,
            test_name="WASM Loading Test",
            category=TestCategory.COMPATIBILITY,
            passed=True,
            duration=1.2,
            performance_metrics={"load_time": 800}
        ),
        TestResult(
            browser=BrowserType.FIREFOX,
            test_name="Performance Test",
            category=TestCategory.PERFORMANCE,
            passed=False,
            duration=2.5,
            error_message="Timeout exceeded"
        )
    ]
    
    # Create mock benchmarks
    mock_benchmarks = {
        "timestamp": "2025-07-26T10:00:00",
        "browsers": {
            "chrome": {"passed": True, "duration": 1.2},
            "firefox": {"passed": False, "error": "Timeout"}
        },
        "summary": {
            "successful_browsers": 1,
            "average_duration": 1.2
        }
    }
    
    # Generate report
    report_path = tester.generate_test_report(mock_results, mock_benchmarks)
    
    assert report_path.exists()
    
    # Check JSON report content
    with open(report_path, 'r') as f:
        report_data = json.load(f)
    
    assert report_data["total_tests"] == 2
    assert report_data["passed_tests"] == 1
    assert report_data["failed_tests"] == 1
    assert report_data["success_rate"] == 50.0
    assert len(report_data["results"]) == 2
    
    # Check if HTML report was also generated
    html_report_path = report_path.with_suffix('.html')
    assert html_report_path.exists()
    
    html_content = html_report_path.read_text()
    assert "TrustformeRS WASM Cross-Browser Test Report" in html_content
    assert "WASM Loading Test" in html_content
    assert "Performance Test" in html_content
    
    print("✅ HTML report generation test passed")

async def test_server_configuration():
    """Test server configuration without actually starting it."""
    print("Testing server configuration...")
    
    tester = CrossBrowserWASMTester()
    
    # Test that we can configure the server app
    if hasattr(tester, '_configure_server_routes'):
        # This would test server route configuration
        pass
    
    # Test server parameters
    assert tester.server_host == "localhost"
    assert tester.server_port == 8888
    assert tester.test_dir.exists()
    
    print("✅ Server configuration test passed")

def test_mobile_device_configurations():
    """Test mobile device configurations."""
    print("Testing mobile device configurations...")
    
    tester = CrossBrowserWASMTester()
    
    # Test mobile device list from config
    mobile_devices = tester.config.get("mobile_devices", [])
    assert len(mobile_devices) > 0
    assert "iPhone 12" in mobile_devices
    assert "Samsung Galaxy S21" in mobile_devices
    
    print("✅ Mobile device configurations test passed")

def test_performance_thresholds():
    """Test performance threshold configurations."""
    print("Testing performance thresholds...")
    
    tester = CrossBrowserWASMTester()
    thresholds = tester.config["performance_thresholds"]
    
    assert "load_time_ms" in thresholds
    assert "inference_time_ms" in thresholds
    assert "memory_usage_mb" in thresholds
    
    assert thresholds["load_time_ms"] > 0
    assert thresholds["inference_time_ms"] > 0
    assert thresholds["memory_usage_mb"] > 0
    
    print("✅ Performance thresholds test passed")

@patch('cross_browser_wasm_testing.webdriver')
@patch('cross_browser_wasm_testing.SELENIUM_AVAILABLE', True)
async def test_mock_browser_test(mock_webdriver):
    """Test browser test execution with mocked WebDriver."""
    print("Testing mock browser test execution...")
    
    # Mock WebDriver
    mock_driver = Mock()
    mock_driver.get = Mock()
    mock_driver.execute_script = Mock()
    mock_driver.quit = Mock()
    
    # Mock WebDriverWait
    mock_wait = Mock()
    mock_wait.until = Mock()
    
    # Configure mocks
    mock_webdriver.Chrome.return_value = mock_driver
    mock_driver.execute_script.side_effect = [
        None,  # runAllTests()
        {   # test report
            "summary": {"failed": 0, "total": 5, "passed": 5},
            "results": []
        },
        {   # browser info
            "userAgent": "Mock Browser",
            "wasm": True,
            "webgl": True
        }
    ]
    
    with patch('cross_browser_wasm_testing.WebDriverWait', return_value=mock_wait):
        tester = CrossBrowserWASMTester()
        
        browser_config = BrowserConfig(
            browser_type=BrowserType.CHROME,
            headless=True,
            timeout=30
        )
        
        result = await tester.run_browser_test(browser_config, "http://localhost:8888/test.html")
        
        assert isinstance(result, TestResult)
        assert result.browser == BrowserType.CHROME
        assert result.test_name == "Cross-Browser WASM Test"
        
        # Verify WebDriver was called
        mock_webdriver.Chrome.assert_called_once()
        mock_driver.get.assert_called_once()
        mock_driver.quit.assert_called_once()
    
    print("✅ Mock browser test execution test passed")

async def test_convenience_function():
    """Test the convenience function for running tests."""
    print("Testing convenience function...")
    
    # This test will be mocked since we don't have real browsers
    with patch('cross_browser_wasm_testing.CrossBrowserWASMTester') as MockTester:
        mock_tester_instance = AsyncMock()
        mock_tester_instance.run_comprehensive_test_suite.return_value = {
            "total_tests": 10,
            "passed_tests": 8,
            "failed_tests": 2,
            "success_rate": 80.0,
            "browser_support": {"chrome": True, "firefox": True}
        }
        
        MockTester.return_value.__aenter__.return_value = mock_tester_instance
        
        # Test convenience function
        result = await run_cross_browser_tests(
            browsers=["chrome", "firefox"],
            include_mobile=False
        )
        
        assert result["total_tests"] == 10
        assert result["success_rate"] == 80.0
        assert "chrome" in result["browser_support"]
    
    print("✅ Convenience function test passed")

async def run_all_tests():
    """Run all cross-browser WASM testing tests."""
    print("🚀 Running Cross-Browser WASM Testing Tests")
    print("=" * 60)
    
    tests = [
        test_browser_config_creation,
        test_test_result_creation,
        test_tester_initialization,
        test_config_loading,
        test_test_file_creation,
        test_browser_config_creation,
        test_html_report_generation,
        test_server_configuration,
        test_mobile_device_configurations,
        test_performance_thresholds,
        test_mock_browser_test,
        test_convenience_function
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                await test()
            else:
                test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Cross-browser WASM testing is working correctly.")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    async def main():
        success = await run_all_tests()
        return 0 if success else 1
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)