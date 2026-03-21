# TrustformeRS-C Cross-Platform Makefile
# Supports Linux, macOS, and Windows (with MSYS2/MinGW)

# Default variables
CARGO = cargo
RUST_VERSION = stable
BUILD_TYPE = debug
FEATURES = default
PARALLEL_JOBS = 4
TEST_TIMEOUT = 300

# Platform detection
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(UNAME_S),Linux)
    PLATFORM = linux
    LIB_EXT = .so
    LIB_PREFIX = lib
    EXEC_EXT =
    SHARED_LIB_FLAG = --crate-type=cdylib
endif
ifeq ($(UNAME_S),Darwin)
    PLATFORM = macos
    LIB_EXT = .dylib
    LIB_PREFIX = lib
    EXEC_EXT =
    SHARED_LIB_FLAG = --crate-type=cdylib
endif
ifneq ($(findstring MINGW,$(UNAME_S)),)
    PLATFORM = windows
    LIB_EXT = .dll
    LIB_PREFIX =
    EXEC_EXT = .exe
    SHARED_LIB_FLAG = --crate-type=cdylib
endif
ifneq ($(findstring MSYS,$(UNAME_S)),)
    PLATFORM = windows
    LIB_EXT = .dll
    LIB_PREFIX =
    EXEC_EXT = .exe
    SHARED_LIB_FLAG = --crate-type=cdylib
endif
ifneq ($(findstring CYGWIN,$(UNAME_S)),)
    PLATFORM = windows
    LIB_EXT = .dll
    LIB_PREFIX =
    EXEC_EXT = .exe
    SHARED_LIB_FLAG = --crate-type=cdylib
endif

# Directories
BUILD_DIR = target
SRC_DIR = src
TEST_DIR = tests
SCRIPTS_DIR = scripts
OUTPUT_DIR = $(BUILD_DIR)/$(BUILD_TYPE)
DOCS_DIR = target/doc

# Library name
LIB_NAME = $(LIB_PREFIX)trustformers_c$(LIB_EXT)
LIB_PATH = $(OUTPUT_DIR)/$(LIB_NAME)

# Build flags
RUST_FLAGS = 
CARGO_FLAGS = --features $(FEATURES)

ifeq ($(BUILD_TYPE),release)
    CARGO_FLAGS += --release
    RUST_FLAGS += -C opt-level=3
endif

ifeq ($(BUILD_TYPE),debug)
    RUST_FLAGS += -C debuginfo=2
endif

# Color output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m

.PHONY: all build build-release build-debug clean test test-unit test-integration test-performance test-memory test-valgrind test-asan \
        check clippy fmt doc install uninstall help setup deps platform-info examples \
        bindings bindings-go bindings-python bindings-nodejs coverage benchmark

# Default target
all: build

# Help target
help:
	@echo "TrustformeRS-C Build System"
	@echo ""
	@echo "Usage: make [target] [variables]"
	@echo ""
	@echo "Build Targets:"
	@echo "  all                Build the library (default: debug)"
	@echo "  build              Build with current BUILD_TYPE (debug/release)"
	@echo "  build-debug        Build debug version"
	@echo "  build-release      Build release version"
	@echo "  clean              Clean build artifacts"
	@echo ""
	@echo "Test Targets:"
	@echo "  test               Run all tests"
	@echo "  test-unit          Run unit tests only"
	@echo "  test-integration   Run integration tests only"
	@echo "  test-performance   Run performance tests"
	@echo "  test-memory        Run comprehensive memory safety tests"
	@echo "  test-valgrind      Run Valgrind memory leak detection"
	@echo "  test-asan          Run AddressSanitizer memory error detection"
	@echo "  coverage           Generate code coverage report"
	@echo "  benchmark          Run benchmarks"
	@echo ""
	@echo "Quality Targets:"
	@echo "  check              Run cargo check"
	@echo "  clippy             Run clippy lints"
	@echo "  fmt                Format code"
	@echo "  doc                Generate documentation"
	@echo ""
	@echo "Language Bindings:"
	@echo "  bindings           Build all language bindings"
	@echo "  bindings-go        Build Go bindings"
	@echo "  bindings-python    Build Python bindings"
	@echo "  bindings-nodejs    Build Node.js bindings"
	@echo ""
	@echo "Utility Targets:"
	@echo "  setup              Setup development environment"
	@echo "  deps               Install dependencies"
	@echo "  platform-info      Show platform information"
	@echo "  examples           Build and run examples"
	@echo "  install            Install library system-wide"
	@echo "  uninstall          Uninstall library"
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_TYPE=debug|release    Build type (default: debug)"
	@echo "  FEATURES=<features>         Cargo features (default: default)"
	@echo "  PARALLEL_JOBS=<n>          Number of parallel jobs (default: 4)"
	@echo "  TEST_TIMEOUT=<seconds>     Test timeout (default: 300)"
	@echo ""
	@echo "Examples:"
	@echo "  make build-release FEATURES=cuda,serving"
	@echo "  make test PARALLEL_JOBS=8"
	@echo "  make coverage BUILD_TYPE=release"
	@echo ""
	@echo "Current Configuration:"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Build Type: $(BUILD_TYPE)"
	@echo "  Features: $(FEATURES)"
	@echo "  Library: $(LIB_PATH)"

# Platform information
platform-info:
	@echo "$(BLUE)[INFO]$(NC) Platform Information"
	@echo "  Platform: $(PLATFORM)"
	@echo "  OS: $(UNAME_S)"
	@echo "  Library Extension: $(LIB_EXT)"
	@echo "  Library Prefix: $(LIB_PREFIX)"
	@echo "  Executable Extension: $(EXEC_EXT)"
	@echo "  Build Directory: $(BUILD_DIR)"
	@echo "  Output Directory: $(OUTPUT_DIR)"
	@echo "  Target Library: $(LIB_PATH)"
	@echo ""
	@echo "Build Configuration:"
	@echo "  Build Type: $(BUILD_TYPE)"
	@echo "  Features: $(FEATURES)"
	@echo "  Cargo Flags: $(CARGO_FLAGS)"
	@echo "  Rust Flags: $(RUST_FLAGS)"

# Setup development environment
setup:
	@echo "$(BLUE)[INFO]$(NC) Setting up development environment"
	@rustup --version || (echo "$(RED)[ERROR]$(NC) Rustup not found. Please install Rust first." && exit 1)
	@$(CARGO) --version || (echo "$(RED)[ERROR]$(NC) Cargo not found. Please install Rust first." && exit 1)
	@rustup toolchain install $(RUST_VERSION)
	@rustup default $(RUST_VERSION)
	@rustup component add rustfmt clippy
	@echo "$(GREEN)[SUCCESS]$(NC) Development environment setup complete"

# Install dependencies
deps:
	@echo "$(BLUE)[INFO]$(NC) Installing dependencies for $(PLATFORM)"
ifeq ($(PLATFORM),linux)
	@sudo apt-get update || true
	@sudo apt-get install -y build-essential pkg-config || true
endif
ifeq ($(PLATFORM),macos)
	@brew install pkg-config || true
endif
ifeq ($(PLATFORM),windows)
	@echo "Please ensure you have a C compiler (MSVC or MinGW) installed"
endif
	@echo "$(GREEN)[SUCCESS]$(NC) Dependencies installed"

# Build targets
build:
	@echo "$(BLUE)[INFO]$(NC) Building TrustformeRS-C ($(BUILD_TYPE)) with features: $(FEATURES)"
	@RUSTFLAGS="$(RUST_FLAGS)" $(CARGO) build $(CARGO_FLAGS)
	@echo "$(GREEN)[SUCCESS]$(NC) Build completed: $(LIB_PATH)"

build-debug:
	@$(MAKE) build BUILD_TYPE=debug

build-release:
	@$(MAKE) build BUILD_TYPE=release

# Clean target
clean:
	@echo "$(BLUE)[INFO]$(NC) Cleaning build artifacts"
	@$(CARGO) clean
	@rm -rf test-results coverage
	@echo "$(GREEN)[SUCCESS]$(NC) Clean completed"

# Quality targets
check:
	@echo "$(BLUE)[INFO]$(NC) Running cargo check"
	@$(CARGO) check --all-targets $(CARGO_FLAGS)
	@echo "$(GREEN)[SUCCESS]$(NC) Check completed"

clippy:
	@echo "$(BLUE)[INFO]$(NC) Running clippy"
	@$(CARGO) clippy --all-targets $(CARGO_FLAGS) -- -D warnings
	@echo "$(GREEN)[SUCCESS]$(NC) Clippy completed"

fmt:
	@echo "$(BLUE)[INFO]$(NC) Formatting code"
	@$(CARGO) fmt --all
	@echo "$(GREEN)[SUCCESS]$(NC) Formatting completed"

doc:
	@echo "$(BLUE)[INFO]$(NC) Generating documentation"
	@$(CARGO) doc $(CARGO_FLAGS) --no-deps --document-private-items
	@echo "$(GREEN)[SUCCESS]$(NC) Documentation generated: $(DOCS_DIR)/trustformers_c/index.html"

# Test targets
test: build
	@echo "$(BLUE)[INFO]$(NC) Running all tests"
ifeq ($(PLATFORM),windows)
	@$(SCRIPTS_DIR)/test-runner.bat --ci -f $(FEATURES) -j $(PARALLEL_JOBS) -t $(TEST_TIMEOUT)
else
	@chmod +x $(SCRIPTS_DIR)/test-runner.sh
	@$(SCRIPTS_DIR)/test-runner.sh --ci -f $(FEATURES) -j $(PARALLEL_JOBS) -t $(TEST_TIMEOUT)
endif
	@echo "$(GREEN)[SUCCESS]$(NC) All tests completed"

test-unit: build
	@echo "$(BLUE)[INFO]$(NC) Running unit tests"
	@$(CARGO) test --lib $(CARGO_FLAGS) --jobs=$(PARALLEL_JOBS)
	@echo "$(GREEN)[SUCCESS]$(NC) Unit tests completed"

test-integration: build
	@echo "$(BLUE)[INFO]$(NC) Running integration tests"
	@$(CARGO) test --test integration_tests $(CARGO_FLAGS) --jobs=$(PARALLEL_JOBS)
	@echo "$(GREEN)[SUCCESS]$(NC) Integration tests completed"

test-performance: build-release
	@echo "$(BLUE)[INFO]$(NC) Running performance tests"
	@$(CARGO) test --test performance_regression_tests --release $(CARGO_FLAGS)
	@echo "$(GREEN)[SUCCESS]$(NC) Performance tests completed"

test-memory: build
	@echo "$(BLUE)[INFO]$(NC) Running comprehensive memory safety tests"
ifeq ($(PLATFORM),windows)
	@echo "$(YELLOW)[WARNING]$(NC) Comprehensive memory testing not fully supported on Windows"
	@$(SCRIPTS_DIR)/test-runner.bat --memory --ci -f $(FEATURES) || true
else
	@chmod +x $(SCRIPTS_DIR)/memory_test.sh $(SCRIPTS_DIR)/valgrind_test.sh $(SCRIPTS_DIR)/asan_test.sh
	@$(SCRIPTS_DIR)/memory_test.sh
endif
	@echo "$(GREEN)[SUCCESS]$(NC) Memory safety tests completed"

test-valgrind: build
	@echo "$(BLUE)[INFO]$(NC) Running Valgrind memory tests"
ifeq ($(PLATFORM),windows)
	@echo "$(YELLOW)[WARNING]$(NC) Valgrind not available on Windows"
else
	@chmod +x $(SCRIPTS_DIR)/valgrind_test.sh
	@$(SCRIPTS_DIR)/valgrind_test.sh
endif
	@echo "$(GREEN)[SUCCESS]$(NC) Valgrind tests completed"

test-asan: build
	@echo "$(BLUE)[INFO]$(NC) Running AddressSanitizer tests"
ifeq ($(PLATFORM),windows)
	@echo "$(YELLOW)[WARNING]$(NC) AddressSanitizer testing limited on Windows"
else
	@chmod +x $(SCRIPTS_DIR)/asan_test.sh
	@$(SCRIPTS_DIR)/asan_test.sh
endif
	@echo "$(GREEN)[SUCCESS]$(NC) AddressSanitizer tests completed"

# Coverage target
coverage: build
	@echo "$(BLUE)[INFO]$(NC) Generating code coverage"
	@command -v cargo-tarpaulin >/dev/null 2>&1 || $(CARGO) install cargo-tarpaulin
	@$(CARGO) tarpaulin $(CARGO_FLAGS) --out Html --output-dir coverage --timeout $(TEST_TIMEOUT)
	@echo "$(GREEN)[SUCCESS]$(NC) Coverage report generated: coverage/tarpaulin-report.html"

# Benchmark target
benchmark: build-release
	@echo "$(BLUE)[INFO]$(NC) Running benchmarks"
	@command -v cargo-criterion >/dev/null 2>&1 || $(CARGO) install cargo-criterion
	@$(CARGO) criterion $(CARGO_FLAGS)
	@echo "$(GREEN)[SUCCESS]$(NC) Benchmarks completed"

# Language bindings
bindings: bindings-go bindings-python bindings-nodejs

bindings-go: build
	@echo "$(BLUE)[INFO]$(NC) Building Go bindings"
	@if [ -d golang ] && command -v go >/dev/null 2>&1; then \
		cd golang && go mod tidy && go build -v ./...; \
		echo "$(GREEN)[SUCCESS]$(NC) Go bindings built"; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) Go not available or golang directory not found"; \
	fi

bindings-python: build
	@echo "$(BLUE)[INFO]$(NC) Building Python bindings"
	@if [ -d python ] && command -v python3 >/dev/null 2>&1; then \
		cd python && python3 setup.py build; \
		echo "$(GREEN)[SUCCESS]$(NC) Python bindings built"; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) Python not available or python directory not found"; \
	fi

bindings-nodejs: build
	@echo "$(BLUE)[INFO]$(NC) Building Node.js bindings"
	@if [ -d nodejs ] && command -v npm >/dev/null 2>&1; then \
		cd nodejs && npm install && npm run build; \
		echo "$(GREEN)[SUCCESS]$(NC) Node.js bindings built"; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) Node.js not available or nodejs directory not found"; \
	fi

# Examples
examples: build
	@echo "$(BLUE)[INFO]$(NC) Building and running examples"
	@if [ -d examples ]; then \
		echo "Running C examples..."; \
		for example in examples/*.c; do \
			if [ -f "$$example" ]; then \
				echo "Building $$example"; \
				$(CC) -o "$${example%.c}" "$$example" -L$(OUTPUT_DIR) -ltrustformers_c -I./; \
			fi; \
		done; \
	fi
	@$(MAKE) bindings-go
	@if [ -d golang/examples ]; then \
		echo "Running Go examples..."; \
		cd golang/examples && go run basic_usage.go || true; \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) Examples completed"

# Installation targets
install: build-release
	@echo "$(BLUE)[INFO]$(NC) Installing TrustformeRS-C system-wide"
ifeq ($(PLATFORM),linux)
	@sudo cp $(OUTPUT_DIR)/$(LIB_NAME) /usr/local/lib/
	@sudo cp src/*.h /usr/local/include/ 2>/dev/null || true
	@sudo ldconfig
endif
ifeq ($(PLATFORM),macos)
	@sudo cp $(OUTPUT_DIR)/$(LIB_NAME) /usr/local/lib/
	@sudo cp src/*.h /usr/local/include/ 2>/dev/null || true
endif
ifeq ($(PLATFORM),windows)
	@echo "Manual installation required on Windows"
	@echo "Copy $(OUTPUT_DIR)/$(LIB_NAME) to your system PATH"
endif
	@echo "$(GREEN)[SUCCESS]$(NC) Installation completed"

uninstall:
	@echo "$(BLUE)[INFO]$(NC) Uninstalling TrustformeRS-C"
ifeq ($(PLATFORM),linux)
	@sudo rm -f /usr/local/lib/$(LIB_NAME)
	@sudo rm -f /usr/local/include/trustformers*.h
	@sudo ldconfig
endif
ifeq ($(PLATFORM),macos)
	@sudo rm -f /usr/local/lib/$(LIB_NAME)
	@sudo rm -f /usr/local/include/trustformers*.h
endif
ifeq ($(PLATFORM),windows)
	@echo "Manual uninstallation required on Windows"
endif
	@echo "$(GREEN)[SUCCESS]$(NC) Uninstallation completed"

# Packaging targets
package: build-release
	@echo "$(BLUE)[INFO]$(NC) Creating package"
	@mkdir -p package/lib package/include package/bindings
	@cp $(OUTPUT_DIR)/$(LIB_NAME) package/lib/
	@cp src/*.h package/include/ 2>/dev/null || true
	@cp -r golang package/bindings/ 2>/dev/null || true
	@cp -r python package/bindings/ 2>/dev/null || true
	@cp -r nodejs package/bindings/ 2>/dev/null || true
	@tar -czf trustformers-c-$(PLATFORM).tar.gz package/
	@echo "$(GREEN)[SUCCESS]$(NC) Package created: trustformers-c-$(PLATFORM).tar.gz"

# Development convenience targets
dev-build:
	@$(MAKE) build BUILD_TYPE=debug FEATURES=default

dev-test:
	@$(MAKE) test-unit BUILD_TYPE=debug FEATURES=default

dev-check:
	@$(MAKE) check
	@$(MAKE) clippy
	@$(MAKE) fmt

# CI targets
ci-test:
	@$(MAKE) setup
	@$(MAKE) deps
	@$(MAKE) check
	@$(MAKE) clippy
	@$(MAKE) test
	@$(MAKE) test-memory

ci-build:
	@$(MAKE) setup
	@$(MAKE) deps
	@$(MAKE) build-release FEATURES=default
	@$(MAKE) build-release FEATURES=serving
	@$(MAKE) build-release FEATURES=cuda

# Maintenance targets
update:
	@echo "$(BLUE)[INFO]$(NC) Updating dependencies"
	@$(CARGO) update
	@echo "$(GREEN)[SUCCESS]$(NC) Dependencies updated"

audit:
	@echo "$(BLUE)[INFO]$(NC) Running security audit"
	@command -v cargo-audit >/dev/null 2>&1 || $(CARGO) install cargo-audit
	@$(CARGO) audit
	@echo "$(GREEN)[SUCCESS]$(NC) Security audit completed"

# Show current configuration
config:
	@echo "Current Configuration:"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Build Type: $(BUILD_TYPE)"
	@echo "  Features: $(FEATURES)"
	@echo "  Parallel Jobs: $(PARALLEL_JOBS)"
	@echo "  Test Timeout: $(TEST_TIMEOUT)"
	@echo "  Cargo: $(CARGO)"
	@echo "  Rust Version: $(RUST_VERSION)"
	@echo "  Library Path: $(LIB_PATH)"