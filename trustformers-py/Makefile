# Makefile for trustformers-py package optimization and development

# Variables
PYTHON := python3
PIP := pip3
MATURIN := maturin
PACKAGE_NAME := trustformers
PACKAGE_DIR := python/trustformers
TEST_DIR := tests
DOCS_DIR := docs

# Build profiles
PROFILE := release
FEATURES := cpu

# Platform and architecture detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    ifeq ($(UNAME_M),aarch64)
        ARCH := aarch64
        TARGET := aarch64-unknown-linux-gnu
        PLATFORM_TARGET := linux-aarch64
    else
        ARCH := x86_64
        TARGET := x86_64-unknown-linux-gnu
        PLATFORM_TARGET := linux-x86_64
    endif
endif

ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
    ifeq ($(UNAME_M),arm64)
        ARCH := aarch64
        TARGET := aarch64-apple-darwin
        PLATFORM_TARGET := macos-aarch64
    else
        ARCH := x86_64
        TARGET := x86_64-apple-darwin
        PLATFORM_TARGET := macos-x86_64
    endif
endif

ifdef COMSPEC
    PLATFORM := windows
    ifeq ($(PROCESSOR_ARCHITECTURE),ARM64)
        ARCH := aarch64
        TARGET := aarch64-pc-windows-msvc
        PLATFORM_TARGET := windows-aarch64
    else
        ARCH := x86_64
        TARGET := x86_64-pc-windows-msvc
        PLATFORM_TARGET := windows-x86_64
    endif
endif

# Directories
BUILD_DIR := target
DIST_DIR := dist
WHEELS_DIR := $(BUILD_DIR)/wheels
REPORTS_DIR := reports

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: help setup clean build test lint format check-deps install dev-install

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "$(BLUE)TrustformeRS Python Package Build System$(NC)"
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Build Profiles:$(NC)"
	@echo "  debug     - Debug build with symbols"
	@echo "  release   - Optimized release build"
	@echo "  size      - Size-optimized build"
	@echo "  speed     - Speed-optimized build"
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make build PROFILE=release FEATURES=cpu,serving"
	@echo "  make test-all"
	@echo "  make optimize-wheels"
	@echo "  make dist-all"

# Setup and installation
setup: ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install maturin
	$(PYTHON) -m pip install -e .
	@echo "$(GREEN)Setup complete!$(NC)"

dev-install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PYTHON) optimize_requirements.py --config dev --format txt --output requirements-dev.txt
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)Development dependencies installed!$(NC)"

install: ## Install package in development mode
	@echo "$(BLUE)Installing package in development mode...$(NC)"
	$(MATURIN) develop --$(PROFILE)
	@echo "$(GREEN)Package installed!$(NC)"

# Build commands
build: ## Build the package
	@echo "$(BLUE)Building package with profile: $(PROFILE), features: $(FEATURES)$(NC)"
	$(MATURIN) build --$(PROFILE) --features $(FEATURES)
	@echo "$(GREEN)Build complete!$(NC)"

build-all: ## Build for all supported platforms
	@echo "$(BLUE)Building for all platforms...$(NC)"
	$(PYTHON) setup_optimization.py --all-platforms --profile $(PROFILE)
	@echo "$(GREEN)All platform builds complete!$(NC)"

build-optimized: ## Build with optimization settings
	@echo "$(BLUE)Building optimized package...$(NC)"
	$(PYTHON) setup_optimization.py --profile size
	@echo "$(GREEN)Optimized build complete!$(NC)"

build-cpu: ## Build CPU-only version
	@echo "$(BLUE)Building CPU-only version...$(NC)"
	$(PYTHON) setup_optimization.py --cpu-only --profile size
	@echo "$(GREEN)CPU-only build complete!$(NC)"

build-cuda: ## Build CUDA version (specify CUDA_VERSION=11.8)
	@echo "$(BLUE)Building CUDA version...$(NC)"
	$(PYTHON) setup_optimization.py --cuda $(CUDA_VERSION) --profile speed
	@echo "$(GREEN)CUDA build complete!$(NC)"

build-arm64: ## Build ARM64 versions for all platforms
	@echo "$(BLUE)Building ARM64 versions...$(NC)"
	$(PYTHON) setup_optimization.py --arm64-all --profile $(PROFILE)
	@echo "$(GREEN)ARM64 builds complete!$(NC)"

build-platform: ## Build for specific platform (use PLATFORM_TARGET=linux-aarch64)
	@echo "$(BLUE)Building for platform: $(PLATFORM_TARGET)$(NC)"
	$(PYTHON) setup_optimization.py --platform $(PLATFORM_TARGET) --profile $(PROFILE)
	@echo "$(GREEN)Platform build complete!$(NC)"

build-universal-macos: ## Build universal macOS wheel (x86_64 + ARM64)
	@echo "$(BLUE)Building universal macOS wheel...$(NC)"
	$(PYTHON) setup_optimization.py --universal-macos --profile $(PROFILE)
	@echo "$(GREEN)Universal macOS build complete!$(NC)"

build-wasm: ## Build experimental WASM version
	@echo "$(BLUE)Building experimental WASM version...$(NC)"
	$(PYTHON) setup_optimization.py --wasm --profile size
	@echo "$(GREEN)WASM build complete!$(NC)"

build-matrix: ## Build complete platform matrix
	@echo "$(BLUE)Building complete platform matrix...$(NC)"
	$(PYTHON) setup_optimization.py --matrix --profile $(PROFILE)
	@echo "$(GREEN)Platform matrix build complete!$(NC)"

build-python-versions: ## Build for all Python versions
	@echo "$(BLUE)Building for all Python versions...$(NC)"
	$(PYTHON) python_version_optimization.py --all-versions --profile $(PROFILE)
	@echo "$(GREEN)Python version builds complete!$(NC)"

# Testing
test: ## Run basic tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v
	@echo "$(GREEN)Tests complete!$(NC)"

test-all: ## Run comprehensive test suite
	@echo "$(BLUE)Running comprehensive test suite...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(PACKAGE_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)Comprehensive tests complete!$(NC)"

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v -k "benchmark or performance"
	@echo "$(GREEN)Performance tests complete!$(NC)"

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v -k "integration"
	@echo "$(GREEN)Integration tests complete!$(NC)"

# Code quality
lint: ## Run linting
	@echo "$(BLUE)Running linting...$(NC)"
	ruff check $(PACKAGE_DIR) $(TEST_DIR)
	flake8 $(PACKAGE_DIR) $(TEST_DIR) --max-line-length=88 --extend-ignore=E203,W503
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black $(PACKAGE_DIR) $(TEST_DIR)
	isort $(PACKAGE_DIR) $(TEST_DIR)
	@echo "$(GREEN)Code formatting complete!$(NC)"

type-check: ## Run type checking
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy $(PACKAGE_DIR) --ignore-missing-imports
	@echo "$(GREEN)Type checking complete!$(NC)"

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r $(PACKAGE_DIR) -f json -o $(REPORTS_DIR)/bandit-report.json
	safety check --json --output $(REPORTS_DIR)/safety-report.json
	@echo "$(GREEN)Security checks complete!$(NC)"

check-all: format lint type-check security-check ## Run all code quality checks
	@echo "$(GREEN)All quality checks complete!$(NC)"

# Dependency management
check-deps: ## Check and optimize dependencies
	@echo "$(BLUE)Checking dependencies...$(NC)"
	$(PYTHON) optimize_requirements.py --analyze
	@echo "$(GREEN)Dependency check complete!$(NC)"

update-deps: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PYTHON) optimize_requirements.py --all-configs
	@echo "$(GREEN)Dependencies updated!$(NC)"

generate-requirements: ## Generate optimized requirements files
	@echo "$(BLUE)Generating optimized requirements...$(NC)"
	$(PYTHON) optimize_requirements.py --all-configs --platform $(PLATFORM)
	@echo "$(GREEN)Requirements generated!$(NC)"

# Optimization
optimize-wheels: ## Optimize existing wheels
	@echo "$(BLUE)Optimizing wheels...$(NC)"
	@for wheel in $(WHEELS_DIR)/*.whl; do \
		echo "Optimizing $$wheel"; \
		$(PYTHON) setup_optimization.py --optimize "$$wheel"; \
	done
	@echo "$(GREEN)Wheel optimization complete!$(NC)"

analyze-size: ## Analyze package size
	@echo "$(BLUE)Analyzing package size...$(NC)"
	@for wheel in $(WHEELS_DIR)/*.whl; do \
		echo "Analyzing $$wheel"; \
		$(PYTHON) setup_optimization.py --analyze "$$wheel"; \
	done
	@echo "$(GREEN)Size analysis complete!$(NC)"

# Distribution
sdist: ## Create source distribution
	@echo "$(BLUE)Creating source distribution...$(NC)"
	$(PYTHON) setup_optimization.py --source-dist
	@echo "$(GREEN)Source distribution created!$(NC)"

wheels: ## Create wheels for current platform
	@echo "$(BLUE)Creating wheels...$(NC)"
	$(MATURIN) build --$(PROFILE) --out $(DIST_DIR)
	@echo "$(GREEN)Wheels created!$(NC)"

dist-all: build-all sdist optimize-wheels ## Create all distribution files
	@echo "$(BLUE)Creating all distribution files...$(NC)"
	mkdir -p $(DIST_DIR)
	cp $(WHEELS_DIR)/*.whl $(DIST_DIR)/
	cp $(WHEELS_DIR)/*.tar.gz $(DIST_DIR)/ 2>/dev/null || true
	@echo "$(GREEN)All distribution files created!$(NC)"

# Publishing
check-dist: ## Check distribution files
	@echo "$(BLUE)Checking distribution files...$(NC)"
	$(PYTHON) -m twine check $(DIST_DIR)/*
	@echo "$(GREEN)Distribution check complete!$(NC)"

publish-test: check-dist ## Publish to test PyPI
	@echo "$(BLUE)Publishing to test PyPI...$(NC)"
	$(PYTHON) -m twine upload --repository testpypi $(DIST_DIR)/*
	@echo "$(GREEN)Published to test PyPI!$(NC)"

publish: check-dist ## Publish to PyPI
	@echo "$(YELLOW)Publishing to PyPI...$(NC)"
	@echo "$(RED)Warning: This will publish to production PyPI!$(NC)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	$(PYTHON) -m twine upload $(DIST_DIR)/*
	@echo "$(GREEN)Published to PyPI!$(NC)"

# Documentation
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	mkdir -p $(DOCS_DIR)
	$(PYTHON) -c "import trustformers; help(trustformers)" > $(DOCS_DIR)/api.txt
	@echo "$(GREEN)Documentation generated!$(NC)"

# Benchmarking
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -k benchmark --benchmark-only --benchmark-json=$(REPORTS_DIR)/benchmark.json
	@echo "$(GREEN)Benchmarks complete!$(NC)"

profile: ## Profile the package
	@echo "$(BLUE)Profiling package...$(NC)"
	$(PYTHON) -m cProfile -o $(REPORTS_DIR)/profile.stats -c "import trustformers; trustformers.Tensor([1,2,3])"
	@echo "$(GREEN)Profiling complete!$(NC)"

# Maintenance
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf $(BUILD_DIR)
	rm -rf $(DIST_DIR)
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"

clean-all: clean ## Clean everything including caches
	@echo "$(BLUE)Deep cleaning...$(NC)"
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf $(REPORTS_DIR)
	cargo clean
	@echo "$(GREEN)Deep clean complete!$(NC)"

# CI/CD helpers
ci-setup: ## Set up CI environment
	@echo "$(BLUE)Setting up CI environment...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel maturin
	$(PYTHON) optimize_requirements.py --config dev --format txt --output requirements-ci.txt
	$(PIP) install -r requirements-ci.txt
	@echo "$(GREEN)CI setup complete!$(NC)"

ci-test: ci-setup build test-all lint ## Run CI test pipeline
	@echo "$(GREEN)CI test pipeline complete!$(NC)"

ci-build: ci-setup build-all optimize-wheels ## Run CI build pipeline
	@echo "$(GREEN)CI build pipeline complete!$(NC)"

# Development helpers
dev: dev-install install ## Set up full development environment
	@echo "$(GREEN)Development environment ready!$(NC)"

watch: ## Watch for changes and rebuild
	@echo "$(BLUE)Watching for changes...$(NC)"
	@while true; do \
		inotifywait -r -e modify python/ src/ 2>/dev/null || sleep 1; \
		echo "$(YELLOW)Changes detected, rebuilding...$(NC)"; \
		make install; \
	done

serve-docs: docs ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	$(PYTHON) -m http.server 8000 -d $(DOCS_DIR)

# Information
info: ## Show build information
	@echo "$(BLUE)Build Information:$(NC)"
	@echo "Platform: $(PLATFORM)"
	@echo "Architecture: $(ARCH)"
	@echo "Platform Target: $(PLATFORM_TARGET)"
	@echo "Rust Target: $(TARGET)"
	@echo "Profile: $(PROFILE)"
	@echo "Features: $(FEATURES)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Maturin: $(shell $(MATURIN) --version)"
	@echo "Cargo: $(shell cargo --version)"
	@echo "Rustc: $(shell rustc --version)"

status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'unknown')"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
	@echo "Package version: $(shell $(PYTHON) -c 'import trustformers; print(trustformers.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "Build artifacts:"
	@ls -la $(WHEELS_DIR)/*.whl 2>/dev/null || echo "  No wheels found"
	@ls -la $(DIST_DIR)/* 2>/dev/null || echo "  No distributions found"

# Create reports directory
$(REPORTS_DIR):
	mkdir -p $(REPORTS_DIR)

# Ensure reports directory exists for targets that need it
security-check benchmark profile: | $(REPORTS_DIR)

# Special targets
.PHONY: all
all: clean dev build test-all check-all dist-all ## Run complete build pipeline
	@echo "$(GREEN)Complete build pipeline finished!$(NC)"