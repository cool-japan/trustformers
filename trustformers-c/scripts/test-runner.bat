@echo off
REM Cross-platform test runner for TrustformeRS-C (Windows version)
REM Supports Windows with MSVC, MinGW, or WSL

setlocal enabledelayedexpansion

REM Configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "TEST_OUTPUT_DIR=%PROJECT_ROOT%\test-results"
set "COVERAGE_OUTPUT_DIR=%PROJECT_ROOT%\coverage"

REM Default test configuration
set "RUN_UNIT_TESTS=true"
set "RUN_INTEGRATION_TESTS=true"
set "RUN_PROPERTY_TESTS=true"
set "RUN_PERFORMANCE_TESTS=false"
set "RUN_MEMORY_TESTS=true"
set "ENABLE_COVERAGE=false"
set "BUILD_RELEASE=false"
set "PARALLEL_JOBS=4"
set "TEST_TIMEOUT=300"
set "FEATURES=default"
set "CLEAN_BUILD=false"
set "VERBOSE=false"
set "CI_MODE=false"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto setup_environment
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help
if "%~1"=="-u" set "RUN_UNIT_TESTS=true" & set "RUN_INTEGRATION_TESTS=false" & set "RUN_PROPERTY_TESTS=false" & set "RUN_PERFORMANCE_TESTS=false" & set "RUN_MEMORY_TESTS=false" & shift & goto parse_args
if "%~1"=="--unit" set "RUN_UNIT_TESTS=true" & set "RUN_INTEGRATION_TESTS=false" & set "RUN_PROPERTY_TESTS=false" & set "RUN_PERFORMANCE_TESTS=false" & set "RUN_MEMORY_TESTS=false" & shift & goto parse_args
if "%~1"=="-i" set "RUN_UNIT_TESTS=false" & set "RUN_INTEGRATION_TESTS=true" & set "RUN_PROPERTY_TESTS=false" & set "RUN_PERFORMANCE_TESTS=false" & set "RUN_MEMORY_TESTS=false" & shift & goto parse_args
if "%~1"=="--integration" set "RUN_UNIT_TESTS=false" & set "RUN_INTEGRATION_TESTS=true" & set "RUN_PROPERTY_TESTS=false" & set "RUN_PERFORMANCE_TESTS=false" & set "RUN_MEMORY_TESTS=false" & shift & goto parse_args
if "%~1"=="-r" set "RUN_UNIT_TESTS=false" & set "RUN_INTEGRATION_TESTS=false" & set "RUN_PROPERTY_TESTS=true" & set "RUN_PERFORMANCE_TESTS=false" & set "RUN_MEMORY_TESTS=false" & shift & goto parse_args
if "%~1"=="--property" set "RUN_UNIT_TESTS=false" & set "RUN_INTEGRATION_TESTS=false" & set "RUN_PROPERTY_TESTS=true" & set "RUN_PERFORMANCE_TESTS=false" & set "RUN_MEMORY_TESTS=false" & shift & goto parse_args
if "%~1"=="-m" set "RUN_UNIT_TESTS=false" & set "RUN_INTEGRATION_TESTS=false" & set "RUN_PROPERTY_TESTS=false" & set "RUN_PERFORMANCE_TESTS=false" & set "RUN_MEMORY_TESTS=true" & shift & goto parse_args
if "%~1"=="--memory" set "RUN_UNIT_TESTS=false" & set "RUN_INTEGRATION_TESTS=false" & set "RUN_PROPERTY_TESTS=false" & set "RUN_PERFORMANCE_TESTS=false" & set "RUN_MEMORY_TESTS=true" & shift & goto parse_args
if "%~1"=="-P" set "RUN_PERFORMANCE_TESTS=true" & shift & goto parse_args
if "%~1"=="--performance" set "RUN_PERFORMANCE_TESTS=true" & shift & goto parse_args
if "%~1"=="-c" set "ENABLE_COVERAGE=true" & shift & goto parse_args
if "%~1"=="--coverage" set "ENABLE_COVERAGE=true" & shift & goto parse_args
if "%~1"=="-R" set "BUILD_RELEASE=true" & shift & goto parse_args
if "%~1"=="--release" set "BUILD_RELEASE=true" & shift & goto parse_args
if "%~1"=="-j" set "PARALLEL_JOBS=%~2" & shift & shift & goto parse_args
if "%~1"=="--jobs" set "PARALLEL_JOBS=%~2" & shift & shift & goto parse_args
if "%~1"=="-t" set "TEST_TIMEOUT=%~2" & shift & shift & goto parse_args
if "%~1"=="--timeout" set "TEST_TIMEOUT=%~2" & shift & shift & goto parse_args
if "%~1"=="-f" set "FEATURES=%~2" & shift & shift & goto parse_args
if "%~1"=="--features" set "FEATURES=%~2" & shift & shift & goto parse_args
if "%~1"=="-o" set "TEST_OUTPUT_DIR=%~2" & shift & shift & goto parse_args
if "%~1"=="--output" set "TEST_OUTPUT_DIR=%~2" & shift & shift & goto parse_args
if "%~1"=="--clean" set "CLEAN_BUILD=true" & shift & goto parse_args
if "%~1"=="--verbose" set "VERBOSE=true" & shift & goto parse_args
if "%~1"=="--ci" set "CI_MODE=true" & shift & goto parse_args
echo Unknown option: %~1
goto show_help

:show_help
echo TrustformeRS-C Cross-Platform Test Runner (Windows)
echo.
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo     -h, --help              Show this help message
echo     -u, --unit              Run unit tests only
echo     -i, --integration       Run integration tests only
echo     -r, --property          Run property-based tests only
echo     -m, --memory            Run memory safety tests only
echo     -P, --performance       Run performance tests
echo     -c, --coverage          Enable code coverage
echo     -R, --release           Build in release mode
echo     -j, --jobs N            Number of parallel jobs (default: %PARALLEL_JOBS%)
echo     -t, --timeout N         Test timeout in seconds (default: %TEST_TIMEOUT%)
echo     -f, --features FEATURES Comma-separated list of features to enable
echo     -o, --output DIR        Output directory for test results
echo     --clean                 Clean build artifacts before testing
echo     --verbose               Verbose output
echo     --ci                    CI mode (non-interactive)
echo.
echo Examples:
echo     %~nx0                   Run all tests with default configuration
echo     %~nx0 -u -c            Run unit tests with coverage
echo     %~nx0 -P -R            Run performance tests in release mode
echo     %~nx0 -f "cuda,serving" Run tests with CUDA and serving features
echo     %~nx0 --clean --ci     Clean build and run in CI mode
echo.
goto :eof

:setup_environment
echo [INFO] Setting up Windows test environment
echo [INFO] Test output directory: %TEST_OUTPUT_DIR%

REM Create output directories
if not exist "%TEST_OUTPUT_DIR%" mkdir "%TEST_OUTPUT_DIR%"
if not exist "%COVERAGE_OUTPUT_DIR%" mkdir "%COVERAGE_OUTPUT_DIR%"

REM Set environment variables
if "%RUST_LOG%"=="" set "RUST_LOG=info"
if "%RUST_BACKTRACE%"=="" set "RUST_BACKTRACE=1"
set "CARGO_TERM_PROGRESS_WHEN=never"

REM Check for required tools
call :check_command "rustc" "Rust compiler" || goto error
call :check_command "cargo" "Cargo" || goto error

REM Detect compiler
if not "%CC%"=="" goto compiler_set
where cl.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Using MSVC compiler
    set "CC=cl.exe"
    set "CXX=cl.exe"
    goto compiler_set
)
where gcc.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Using MinGW compiler
    set "CC=gcc.exe"
    set "CXX=g++.exe"
    goto compiler_set
)
echo [WARNING] No suitable C compiler found

:compiler_set
goto main_execution

:check_command
where %~1 >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] %~2 (%~1) not found
    exit /b 1
)
exit /b 0

:clean_build
if "%CLEAN_BUILD%"=="true" (
    echo [INFO] Cleaning build artifacts
    cargo clean
    if exist "%TEST_OUTPUT_DIR%" rmdir /s /q "%TEST_OUTPUT_DIR%"
    if exist "%COVERAGE_OUTPUT_DIR%" rmdir /s /q "%COVERAGE_OUTPUT_DIR%"
    mkdir "%TEST_OUTPUT_DIR%"
    mkdir "%COVERAGE_OUTPUT_DIR%"
)
goto :eof

:build_project
echo [INFO] Building project with features: %FEATURES%

set "BUILD_ARGS=--features %FEATURES%"
if "%BUILD_RELEASE%"=="true" (
    set "BUILD_ARGS=%BUILD_ARGS% --release"
    echo [INFO] Building in release mode
)
if "%VERBOSE%"=="true" set "BUILD_ARGS=%BUILD_ARGS% --verbose"

cargo build %BUILD_ARGS%
if %errorlevel% neq 0 (
    echo [ERROR] Build failed
    exit /b 1
)
echo [SUCCESS] Project built successfully
goto :eof

:run_unit_tests
if "%RUN_UNIT_TESTS%"=="false" goto :eof

echo [INFO] Running unit tests

set "TEST_ARGS=--features %FEATURES% --lib"
if "%BUILD_RELEASE%"=="true" set "TEST_ARGS=%TEST_ARGS% --release"
if "%VERBOSE%"=="true" set "TEST_ARGS=%TEST_ARGS% --verbose"

cargo test %TEST_ARGS% --jobs=%PARALLEL_JOBS% > "%TEST_OUTPUT_DIR%\unit_tests.log" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Unit tests passed
) else (
    echo [ERROR] Unit tests failed
    exit /b 1
)
goto :eof

:run_integration_tests
if "%RUN_INTEGRATION_TESTS%"=="false" goto :eof

echo [INFO] Running integration tests

set "TEST_ARGS=--features %FEATURES% --test integration_tests"
if "%BUILD_RELEASE%"=="true" set "TEST_ARGS=%TEST_ARGS% --release"

cargo test %TEST_ARGS% --jobs=%PARALLEL_JOBS% > "%TEST_OUTPUT_DIR%\integration_tests.log" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Integration tests passed
) else (
    echo [ERROR] Integration tests failed
    exit /b 1
)
goto :eof

:run_property_tests
if "%RUN_PROPERTY_TESTS%"=="false" goto :eof

echo [INFO] Running property-based tests

set "TEST_ARGS=--features %FEATURES% --test property_based_tests"
if "%BUILD_RELEASE%"=="true" set "TEST_ARGS=%TEST_ARGS% --release"

cargo test %TEST_ARGS% --jobs=%PARALLEL_JOBS% > "%TEST_OUTPUT_DIR%\property_tests.log" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Property-based tests passed
) else (
    echo [ERROR] Property-based tests failed
    exit /b 1
)
goto :eof

:run_performance_tests
if "%RUN_PERFORMANCE_TESTS%"=="false" goto :eof

echo [INFO] Running performance tests

REM Always run performance tests in release mode
set "TEST_ARGS=--features %FEATURES% --test performance_regression_tests --release"

cargo test %TEST_ARGS% --jobs=%PARALLEL_JOBS% > "%TEST_OUTPUT_DIR%\performance_tests.log" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Performance tests passed
) else (
    echo [ERROR] Performance tests failed
    exit /b 1
)
goto :eof

:run_memory_tests
if "%RUN_MEMORY_TESTS%"=="false" goto :eof

echo [INFO] Running memory safety tests

set "TEST_ARGS=--features %FEATURES% --lib memory_safety"
if "%BUILD_RELEASE%"=="true" set "TEST_ARGS=%TEST_ARGS% --release"

cargo test %TEST_ARGS% > "%TEST_OUTPUT_DIR%\memory_tests.log" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Memory safety tests passed
) else (
    echo [ERROR] Memory safety tests failed
    exit /b 1
)
goto :eof

:test_language_bindings
echo [INFO] Testing language bindings

REM Test Go bindings
if exist "golang" (
    where go.exe >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] Testing Go bindings
        pushd golang
        go mod tidy
        go test ./... > "%TEST_OUTPUT_DIR%\go_tests.log" 2>&1
        if %errorlevel% equ 0 (
            echo [SUCCESS] Go bindings tests passed
        ) else (
            echo [ERROR] Go bindings tests failed
        )
        popd
    ) else (
        echo [WARNING] Go not installed, skipping Go bindings tests
    )
) else (
    echo [WARNING] Go bindings not available
)

REM Test Python bindings
if exist "python" (
    where python.exe >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] Testing Python bindings
        pushd python
        python -m pytest tests/ -v > "%TEST_OUTPUT_DIR%\python_tests.log" 2>&1
        if %errorlevel% equ 0 (
            echo [SUCCESS] Python bindings tests passed
        ) else (
            echo [ERROR] Python bindings tests failed
        )
        popd
    ) else (
        echo [WARNING] Python not installed, skipping Python bindings tests
    )
) else (
    echo [WARNING] Python bindings not available
)

REM Test Node.js bindings
if exist "nodejs" (
    where npm.cmd >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] Testing Node.js bindings
        pushd nodejs
        npm test > "%TEST_OUTPUT_DIR%\nodejs_tests.log" 2>&1
        if %errorlevel% equ 0 (
            echo [SUCCESS] Node.js bindings tests passed
        ) else (
            echo [ERROR] Node.js bindings tests failed
        )
        popd
    ) else (
        echo [WARNING] Node.js not installed, skipping Node.js bindings tests
    )
) else (
    echo [WARNING] Node.js bindings not available
)
goto :eof

:generate_coverage_report
if "%ENABLE_COVERAGE%"=="false" goto :eof

echo [INFO] Generating coverage report

REM Check if cargo-tarpaulin is available (Windows support is limited)
cargo tarpaulin --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] cargo-tarpaulin not available on Windows, skipping coverage
    goto :eof
)

cargo tarpaulin --features %FEATURES% --out Html --output-dir "%COVERAGE_OUTPUT_DIR%" --timeout %TEST_TIMEOUT% > "%TEST_OUTPUT_DIR%\coverage.log" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Coverage report generated: %COVERAGE_OUTPUT_DIR%\tarpaulin-report.html
) else (
    echo [ERROR] Failed to generate coverage report
)
goto :eof

:generate_test_report
echo [INFO] Generating test report

set "REPORT_FILE=%TEST_OUTPUT_DIR%\test_report.html"

echo ^<!DOCTYPE html^> > "%REPORT_FILE%"
echo ^<html^> >> "%REPORT_FILE%"
echo ^<head^> >> "%REPORT_FILE%"
echo     ^<title^>TrustformeRS-C Test Report^</title^> >> "%REPORT_FILE%"
echo     ^<style^> >> "%REPORT_FILE%"
echo         body { font-family: Arial, sans-serif; margin: 20px; } >> "%REPORT_FILE%"
echo         .header { background-color: #f0f0f0; padding: 10px; } >> "%REPORT_FILE%"
echo         .success { color: green; } >> "%REPORT_FILE%"
echo         .error { color: red; } >> "%REPORT_FILE%"
echo         .warning { color: orange; } >> "%REPORT_FILE%"
echo         .section { margin: 20px 0; } >> "%REPORT_FILE%"
echo         pre { background-color: #f5f5f5; padding: 10px; overflow-x: auto; } >> "%REPORT_FILE%"
echo     ^</style^> >> "%REPORT_FILE%"
echo ^</head^> >> "%REPORT_FILE%"
echo ^<body^> >> "%REPORT_FILE%"
echo     ^<div class="header"^> >> "%REPORT_FILE%"
echo         ^<h1^>TrustformeRS-C Test Report^</h1^> >> "%REPORT_FILE%"
echo         ^<p^>Generated on: %DATE% %TIME%^</p^> >> "%REPORT_FILE%"
echo         ^<p^>Platform: Windows^</p^> >> "%REPORT_FILE%"
echo         ^<p^>Features: %FEATURES%^</p^> >> "%REPORT_FILE%"
echo     ^</div^> >> "%REPORT_FILE%"

REM Add test results
for %%T in (unit integration property performance memory) do (
    if exist "%TEST_OUTPUT_DIR%\%%T_tests.log" (
        echo     ^<div class="section"^> >> "%REPORT_FILE%"
        echo         ^<h2^>%%T Tests^</h2^> >> "%REPORT_FILE%"
        echo         ^<pre^> >> "%REPORT_FILE%"
        type "%TEST_OUTPUT_DIR%\%%T_tests.log" >> "%REPORT_FILE%"
        echo         ^</pre^> >> "%REPORT_FILE%"
        echo     ^</div^> >> "%REPORT_FILE%"
    )
)

echo ^</body^> >> "%REPORT_FILE%"
echo ^</html^> >> "%REPORT_FILE%"

echo [SUCCESS] Test report generated: %REPORT_FILE%
goto :eof

:main_execution
echo [INFO] Starting TrustformeRS-C test runner
echo [INFO] Platform: Windows
echo [INFO] Test output directory: %TEST_OUTPUT_DIR%

call :clean_build
call :build_project || goto error

set "TEST_RESULTS=0"

call :run_unit_tests
if %errorlevel% neq 0 set /a TEST_RESULTS+=1

call :run_integration_tests
if %errorlevel% neq 0 set /a TEST_RESULTS+=1

call :run_property_tests
if %errorlevel% neq 0 set /a TEST_RESULTS+=1

call :run_performance_tests
if %errorlevel% neq 0 set /a TEST_RESULTS+=1

call :run_memory_tests
if %errorlevel% neq 0 set /a TEST_RESULTS+=1

call :test_language_bindings
call :generate_coverage_report
call :generate_test_report

echo [INFO] Test execution completed

if %TEST_RESULTS% equ 0 (
    echo [SUCCESS] All tests passed!
    exit /b 0
) else (
    echo [ERROR] %TEST_RESULTS% test suite(s) failed
    exit /b 1
)

:error
echo [ERROR] Test execution failed
exit /b 1