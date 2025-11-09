# TrustformeRS Makefile for Development Tasks

.PHONY: all test check fmt lint clean doc bench release help install-tools

# Default target
all: check test

# Help target
help:
	@echo "TrustformeRS Development Commands:"
	@echo "  make install-tools  - Install required development tools"
	@echo "  make check         - Run all checks (fmt, clippy, test, doc)"
	@echo "  make test          - Run all tests with nextest"
	@echo "  make fmt           - Format code with rustfmt"
	@echo "  make lint          - Run clippy linting"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make doc           - Build documentation"
	@echo "  make bench         - Run benchmarks"
	@echo "  make release       - Build release binaries"
	@echo "  make audit         - Run security audit"
	@echo "  make deny          - Check dependencies with cargo-deny"
	@echo "  make typos         - Check for typos"
	@echo "  make pre-commit    - Run pre-commit hooks"
	@echo "  make coverage      - Generate test coverage report"
	@echo "  make msrv          - Check minimum supported Rust version"

# Install development tools
install-tools:
	@echo "Installing development tools..."
	cargo install cargo-nextest --locked
	cargo install cargo-audit --locked
	cargo install cargo-deny --locked
	cargo install cargo-criterion --locked
	cargo install cargo-tarpaulin --locked
	cargo install cargo-readme --locked
	cargo install cargo-udeps --locked
	cargo install typos-cli --locked
	@echo "Installing pre-commit..."
	pip install pre-commit
	pre-commit install
	@echo "Development tools installed!"

# Run all checks
check: fmt-check clippy test doc

# Format code
fmt:
	@echo "Formatting code..."
	cargo fmt --all

# Check formatting
fmt-check:
	@echo "Checking code formatting..."
	cargo fmt --all -- --check

# Run clippy
clippy:
	@echo "Running clippy..."
	cargo clippy --all-targets --all-features -- -D warnings

# Run tests with nextest
test:
	@echo "Running tests with nextest..."
	cargo nextest run --all-features --no-fail-fast

# Run standard cargo test (for doctests)
test-doc:
	@echo "Running doctests..."
	cargo test --doc --all-features

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -rf Cargo.lock
	find . -name "*.orig" -type f -delete
	find . -name "*.rej" -type f -delete
	find . -name "*~" -type f -delete

# Build documentation
doc:
	@echo "Building documentation..."
	cargo doc --all-features --no-deps --open

# Build documentation without opening
doc-build:
	@echo "Building documentation..."
	cargo doc --all-features --no-deps

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	cargo criterion

# Build release binaries
release:
	@echo "Building release binaries..."
	cargo build --release --all-features

# Security audit
audit:
	@echo "Running security audit..."
	cargo audit

# Dependency check with cargo-deny
deny:
	@echo "Checking dependencies..."
	cargo deny check

# Check for typos
typos:
	@echo "Checking for typos..."
	typos

# Run pre-commit hooks
pre-commit:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files

# Generate test coverage
coverage:
	@echo "Generating test coverage..."
	cargo tarpaulin --all-features --workspace --timeout 600 --out html
	@echo "Coverage report generated in target/tarpaulin/index.html"

# Check MSRV
msrv:
	@echo "Checking minimum supported Rust version..."
	@MSRV=$$(grep "rust-version" Cargo.toml | head -1 | cut -d'"' -f2); \
	echo "MSRV: $$MSRV"; \
	cargo +$$MSRV check --all-features

# Update dependencies
update:
	@echo "Updating dependencies..."
	cargo update
	cargo audit fix

# Run all CI checks locally
ci: fmt-check clippy test test-doc audit deny typos

# Quick check for development
quick: fmt clippy test

# Watch for changes and run tests
watch:
	@echo "Watching for changes..."
	cargo watch -x "nextest run" -x "clippy -- -D warnings"

# Check unused dependencies
unused-deps:
	@echo "Checking for unused dependencies..."
	cargo +nightly udeps --all-targets --all-features

# List outdated dependencies
outdated:
	@echo "Checking for outdated dependencies..."
	cargo outdated -R

# Generate README from doc comments
readme:
	@echo "Generating README from doc comments..."
	cargo readme --check

# Run integration tests only
integration:
	@echo "Running integration tests..."
	cargo test --test '*' --features integration-tests

# Profile-guided optimization build
pgo:
	@echo "Building with profile-guided optimization..."
	@echo "Step 1: Building instrumented binary..."
	RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release
	@echo "Step 2: Running benchmarks to generate profile data..."
	./target/release/trustformers benchmark
	@echo "Step 3: Building optimized binary..."
	RUSTFLAGS="-Cprofile-use=/tmp/pgo-data" cargo build --release

# Memory profiling with heaptrack
memory-profile:
	@echo "Running memory profiling..."
	heaptrack target/release/trustformers

# CPU profiling with flamegraph
cpu-profile:
	@echo "Running CPU profiling..."
	cargo flamegraph --bench inference_bench -- --bench

# Size optimization build
size-opt:
	@echo "Building size-optimized binary..."
	RUSTFLAGS="-C opt-level=z" cargo build --release
	strip target/release/trustformers
	ls -lh target/release/trustformers