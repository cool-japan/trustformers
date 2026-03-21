#!/usr/bin/env python3
"""
Cross-platform testing automation framework for TrustformeRS
"""

import os
import sys
import subprocess
import json
import platform
import argparse
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test-automation.log')
    ]
)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

class Platform(Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"

class Architecture(Enum):
    X86_64 = "x86_64"
    ARM64 = "arm64"
    X86 = "x86"

@dataclass
class TestResult:
    name: str
    status: TestStatus
    duration: float
    output: str
    error: Optional[str] = None
    platform: Optional[str] = None
    architecture: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TestSuite:
    name: str
    tests: List['TestCase']
    setup_commands: List[str] = None
    teardown_commands: List[str] = None
    parallel: bool = False
    timeout: int = 300  # 5 minutes default

@dataclass
class TestCase:
    name: str
    command: str
    expected_exit_code: int = 0
    timeout: int = 60
    working_directory: Optional[str] = None
    environment: Optional[Dict[str, str]] = None
    platforms: Optional[List[Platform]] = None
    architectures: Optional[List[Architecture]] = None
    skip_on_failure: bool = False
    dependencies: Optional[List[str]] = None

class TestRunner:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.results: List[TestResult] = []
        self.current_platform = self._detect_platform()
        self.current_architecture = self._detect_architecture()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load test configuration from file"""
        default_config = {
            "max_parallel_tests": 4,
            "default_timeout": 300,
            "output_directory": "test-results",
            "keep_temp_files": False,
            "coverage_enabled": True,
            "memory_testing_enabled": True,
            "performance_testing_enabled": True,
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")
        
        return default_config
    
    def _detect_platform(self) -> Platform:
        """Detect current platform"""
        system = platform.system().lower()
        if system == "windows":
            return Platform.WINDOWS
        elif system == "linux":
            return Platform.LINUX
        elif system == "darwin":
            return Platform.MACOS
        else:
            raise RuntimeError(f"Unsupported platform: {system}")
    
    def _detect_architecture(self) -> Architecture:
        """Detect current architecture"""
        machine = platform.machine().lower()
        if machine in ["x86_64", "amd64"]:
            return Architecture.X86_64
        elif machine in ["arm64", "aarch64"]:
            return Architecture.ARM64
        elif machine in ["x86", "i386", "i686"]:
            return Architecture.X86
        else:
            raise RuntimeError(f"Unsupported architecture: {machine}")
    
    def _should_run_test(self, test: TestCase) -> bool:
        """Check if test should run on current platform/architecture"""
        if test.platforms and self.current_platform not in test.platforms:
            return False
        if test.architectures and self.current_architecture not in test.architectures:
            return False
        return True
    
    def _run_command(self, command: str, cwd: Optional[str] = None, 
                    env: Optional[Dict[str, str]] = None, timeout: int = 60) -> Tuple[int, str, str]:
        """Run a shell command and return exit code, stdout, stderr"""
        try:
            # Prepare environment
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            # Run command
            logger.info(f"Running command: {command}")
            if cwd:
                logger.info(f"Working directory: {cwd}")
            
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=process_env,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            return process.returncode, stdout, stderr
            
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", str(e)
    
    def _run_test_case(self, test: TestCase) -> TestResult:
        """Run a single test case"""
        start_time = time.time()
        
        if not self._should_run_test(test):
            return TestResult(
                name=test.name,
                status=TestStatus.SKIPPED,
                duration=0,
                output="Skipped on this platform/architecture",
                platform=self.current_platform.value,
                architecture=self.current_architecture.value
            )
        
        try:
            exit_code, stdout, stderr = self._run_command(
                test.command,
                cwd=test.working_directory,
                env=test.environment,
                timeout=test.timeout
            )
            
            duration = time.time() - start_time
            
            if exit_code == test.expected_exit_code:
                status = TestStatus.PASSED
                error = None
            else:
                status = TestStatus.FAILED
                error = f"Expected exit code {test.expected_exit_code}, got {exit_code}"
            
            output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            
            return TestResult(
                name=test.name,
                status=status,
                duration=duration,
                output=output,
                error=error,
                platform=self.current_platform.value,
                architecture=self.current_architecture.value,
                metadata={"exit_code": exit_code}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=test.name,
                status=TestStatus.FAILED,
                duration=duration,
                output="",
                error=str(e),
                platform=self.current_platform.value,
                architecture=self.current_architecture.value
            )
    
    def run_test_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run a complete test suite"""
        logger.info(f"Running test suite: {suite.name}")
        results = []
        
        # Run setup commands
        if suite.setup_commands:
            logger.info("Running setup commands...")
            for cmd in suite.setup_commands:
                exit_code, stdout, stderr = self._run_command(cmd, timeout=suite.timeout)
                if exit_code != 0:
                    logger.error(f"Setup command failed: {cmd}")
                    logger.error(f"Error: {stderr}")
                    # Return failed results for all tests
                    for test in suite.tests:
                        results.append(TestResult(
                            name=test.name,
                            status=TestStatus.FAILED,
                            duration=0,
                            output="",
                            error="Setup failed",
                            platform=self.current_platform.value,
                            architecture=self.current_architecture.value
                        ))
                    return results
        
        try:
            # Run tests
            if suite.parallel:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config["max_parallel_tests"]
                ) as executor:
                    future_to_test = {
                        executor.submit(self._run_test_case, test): test 
                        for test in suite.tests
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_test):
                        result = future.result()
                        results.append(result)
                        logger.info(f"Test {result.name}: {result.status.value}")
            else:
                for test in suite.tests:
                    result = self._run_test_case(test)
                    results.append(result)
                    logger.info(f"Test {result.name}: {result.status.value}")
                    
                    # Stop on failure if configured
                    if result.status == TestStatus.FAILED and test.skip_on_failure:
                        logger.warning(f"Stopping test suite due to failure in {test.name}")
                        break
        
        finally:
            # Run teardown commands
            if suite.teardown_commands:
                logger.info("Running teardown commands...")
                for cmd in suite.teardown_commands:
                    exit_code, stdout, stderr = self._run_command(cmd, timeout=suite.timeout)
                    if exit_code != 0:
                        logger.warning(f"Teardown command failed: {cmd}")
                        logger.warning(f"Error: {stderr}")
        
        return results
    
    def generate_report(self, results: List[TestResult], output_path: str):
        """Generate test report in multiple formats"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # JSON report
        json_path = output_path + ".json"
        with open(json_path, 'w') as f:
            json.dump([asdict(result) for result in results], f, indent=2)
        
        # HTML report
        html_path = output_path + ".html"
        self._generate_html_report(results, html_path)
        
        # Text summary
        txt_path = output_path + ".txt"
        self._generate_text_report(results, txt_path)
        
        logger.info(f"Reports generated: {json_path}, {html_path}, {txt_path}")
    
    def _generate_html_report(self, results: List[TestResult], output_path: str):
        """Generate HTML test report"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped_tests = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TrustformeRS Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .test-result {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .passed {{ background: #d4edda; border-left: 5px solid #28a745; }}
        .failed {{ background: #f8d7da; border-left: 5px solid #dc3545; }}
        .skipped {{ background: #fff3cd; border-left: 5px solid #ffc107; }}
        .details {{ margin-top: 10px; background: #f8f9fa; padding: 10px; border-radius: 3px; }}
        pre {{ white-space: pre-wrap; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>TrustformeRS Test Results</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {total_tests}</p>
        <p><strong>Passed:</strong> {passed_tests}</p>
        <p><strong>Failed:</strong> {failed_tests}</p>
        <p><strong>Skipped:</strong> {skipped_tests}</p>
        <p><strong>Success Rate:</strong> {(passed_tests/total_tests*100):.1f}%</p>
        <p><strong>Platform:</strong> {self.current_platform.value}</p>
        <p><strong>Architecture:</strong> {self.current_architecture.value}</p>
    </div>
    
    <h2>Test Results</h2>
"""
        
        for result in results:
            status_class = result.status.value
            html_content += f"""
    <div class="test-result {status_class}">
        <h3>{result.name}</h3>
        <p><strong>Status:</strong> {result.status.value.upper()}</p>
        <p><strong>Duration:</strong> {result.duration:.2f}s</p>
        <p><strong>Platform:</strong> {result.platform}</p>
        <p><strong>Architecture:</strong> {result.architecture}</p>
"""
            if result.error:
                html_content += f"""
        <div class="details">
            <strong>Error:</strong>
            <pre>{result.error}</pre>
        </div>
"""
            if result.output:
                html_content += f"""
        <div class="details">
            <strong>Output:</strong>
            <pre>{result.output}</pre>
        </div>
"""
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_text_report(self, results: List[TestResult], output_path: str):
        """Generate text test report"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped_tests = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        
        with open(output_path, 'w') as f:
            f.write("TrustformeRS Test Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Summary:\n")
            f.write(f"  Total Tests: {total_tests}\n")
            f.write(f"  Passed: {passed_tests}\n")
            f.write(f"  Failed: {failed_tests}\n")
            f.write(f"  Skipped: {skipped_tests}\n")
            f.write(f"  Success Rate: {(passed_tests/total_tests*100):.1f}%\n")
            f.write(f"  Platform: {self.current_platform.value}\n")
            f.write(f"  Architecture: {self.current_architecture.value}\n\n")
            
            f.write("Test Results:\n")
            f.write("-" * 50 + "\n")
            
            for result in results:
                f.write(f"\n{result.name}:\n")
                f.write(f"  Status: {result.status.value.upper()}\n")
                f.write(f"  Duration: {result.duration:.2f}s\n")
                if result.error:
                    f.write(f"  Error: {result.error}\n")

def create_default_test_suites() -> List[TestSuite]:
    """Create default test suites for TrustformeRS"""
    
    # Core functionality tests
    core_tests = TestSuite(
        name="Core Functionality",
        setup_commands=[
            "cargo build --release",
            "cargo build --release --features cuda,rocm,metal"
        ],
        tests=[
            TestCase(
                name="Basic API Test",
                command="cargo test test_basic_api --release",
                timeout=120
            ),
            TestCase(
                name="Memory Management Test",
                command="cargo test test_memory_management --release",
                timeout=180
            ),
            TestCase(
                name="Error Handling Test",
                command="cargo test test_error_handling --release",
                timeout=60
            ),
            TestCase(
                name="C API Integration Test", 
                command="cargo test test_c_api --release",
                timeout=120
            )
        ],
        parallel=True
    )
    
    # Memory safety tests
    memory_tests = TestSuite(
        name="Memory Safety",
        setup_commands=["cargo build --release"],
        tests=[
            TestCase(
                name="Valgrind Memory Test",
                command="./scripts/valgrind_test.sh",
                timeout=300,
                platforms=[Platform.LINUX]
            ),
            TestCase(
                name="AddressSanitizer Test",
                command="./scripts/asan_test.sh", 
                timeout=300,
                platforms=[Platform.LINUX, Platform.MACOS]
            ),
            TestCase(
                name="Memory Leak Detection",
                command="cargo test test_memory_leaks --release",
                timeout=180
            )
        ]
    )
    
    # Language bindings tests
    bindings_tests = TestSuite(
        name="Language Bindings",
        tests=[
            TestCase(
                name="Python Bindings Test",
                command="python -m pytest python/tests/ -v",
                timeout=180
            ),
            TestCase(
                name="Node.js Bindings Test", 
                command="npm test",
                working_directory="nodejs",
                timeout=120
            ),
            TestCase(
                name="Go Bindings Test",
                command="go test ./...",
                working_directory="golang",
                timeout=120
            ),
            TestCase(
                name="Java Bindings Test",
                command="mvn test",
                working_directory="java",
                timeout=180
            ),
            TestCase(
                name="C# Bindings Test",
                command="dotnet test",
                working_directory="csharp",
                timeout=180,
                platforms=[Platform.WINDOWS, Platform.LINUX, Platform.MACOS]
            )
        ],
        parallel=True
    )
    
    # Performance tests
    performance_tests = TestSuite(
        name="Performance",
        tests=[
            TestCase(
                name="Inference Benchmark",
                command="cargo bench --bench inference_benchmark",
                timeout=600
            ),
            TestCase(
                name="Memory Performance Test",
                command="cargo test test_memory_performance --release",
                timeout=300
            ),
            TestCase(
                name="Throughput Test",
                command="cargo test test_throughput --release",
                timeout=300
            )
        ]
    )
    
    # Hardware acceleration tests
    hardware_tests = TestSuite(
        name="Hardware Acceleration",
        tests=[
            TestCase(
                name="CUDA Test",
                command="cargo test test_cuda --release --features cuda",
                timeout=180,
                environment={"CUDA_VISIBLE_DEVICES": "0"}
            ),
            TestCase(
                name="ROCm Test",
                command="cargo test test_rocm --release --features rocm",
                timeout=180,
                platforms=[Platform.LINUX]
            ),
            TestCase(
                name="Metal Test",
                command="cargo test test_metal --release --features metal",
                timeout=180,
                platforms=[Platform.MACOS]
            )
        ]
    )
    
    return [core_tests, memory_tests, bindings_tests, performance_tests, hardware_tests]

def main():
    parser = argparse.ArgumentParser(description="TrustformeRS Test Automation")
    parser.add_argument("--config", help="Path to test configuration file")
    parser.add_argument("--suite", help="Run specific test suite", action="append")
    parser.add_argument("--output", default="test-results/results", help="Output path for reports")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--timeout", type=int, default=300, help="Default timeout for tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize test runner
    runner = TestRunner(args.config)
    
    # Get test suites
    all_suites = create_default_test_suites()
    
    if args.suite:
        # Run specific suites
        suites_to_run = [s for s in all_suites if s.name in args.suite]
        if not suites_to_run:
            logger.error(f"No matching test suites found for: {args.suite}")
            return 1
    else:
        # Run all suites
        suites_to_run = all_suites
    
    # Run tests
    all_results = []
    start_time = time.time()
    
    for suite in suites_to_run:
        if args.parallel:
            suite.parallel = True
        if args.timeout:
            suite.timeout = args.timeout
            
        results = runner.run_test_suite(suite)
        all_results.extend(results)
    
    total_time = time.time() - start_time
    
    # Generate reports
    runner.generate_report(all_results, args.output)
    
    # Print summary
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r.status == TestStatus.PASSED)
    failed_tests = sum(1 for r in all_results if r.status == TestStatus.FAILED)
    skipped_tests = sum(1 for r in all_results if r.status == TestStatus.SKIPPED)
    
    print(f"\nTest Summary:")
    print(f"  Total: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    print(f"  Skipped: {skipped_tests}")
    print(f"  Success Rate: {(passed_tests/total_tests*100):.1f}%")
    print(f"  Total Time: {total_time:.2f}s")
    
    # Exit with error code if any tests failed
    return 1 if failed_tests > 0 else 0

if __name__ == "__main__":
    sys.exit(main())