#!/bin/bash

# Docker Optimization Build Script for TrustformeRS Serve
# Provides optimized builds for different deployment scenarios

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="production"
DOCKERFILE="Dockerfile.super-optimized"
IMAGE_NAME="trustformers-serve"
TAG="latest"
RUST_VERSION="1.75"
OPTIMIZATION_LEVEL="3"
PUSH_REGISTRY=""
ENABLE_BUILDKIT="true"
ENABLE_CACHE="true"
MULTI_ARCH="false"
SECURITY_SCAN="false"
COMPRESS="true"

# Print usage
usage() {
    cat << EOF
Docker Optimization Build Script for TrustformeRS Serve

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE          Build type: production, development, debug, alpine, security-hardened (default: production)
    -d, --dockerfile FILE    Dockerfile to use (default: Dockerfile.super-optimized)
    -n, --name NAME          Image name (default: trustformers-serve)
    -g, --tag TAG            Image tag (default: latest)
    -r, --rust-version VER   Rust version (default: 1.75)
    -o, --optimization LEVEL Optimization level 0-3 (default: 3)
    -p, --push REGISTRY      Push to registry after build
    -m, --multi-arch         Build for multiple architectures
    -s, --security-scan      Run security scanning
    -c, --no-cache           Disable build cache
    -z, --no-compress        Disable image compression
    -h, --help               Show this help message

BUILD TYPES:
    production         Optimized production build with distroless base
    development        Development build with hot reload
    debug              Debug build with debugging tools
    alpine             Alpine-based production build
    security-hardened  Security-hardened production build
    benchmark          Benchmarking build with performance tools

EXAMPLES:
    $0 --type production --tag v1.0.0
    $0 --type development --name trustformers-dev
    $0 --type security-hardened --security-scan --push docker.io/myrepo
    $0 --multi-arch --type production --push ghcr.io/myorg/trustformers

EOF
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -d|--dockerfile)
                DOCKERFILE="$2"
                shift 2
                ;;
            -n|--name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            -g|--tag)
                TAG="$2"
                shift 2
                ;;
            -r|--rust-version)
                RUST_VERSION="$2"
                shift 2
                ;;
            -o|--optimization)
                OPTIMIZATION_LEVEL="$2"
                shift 2
                ;;
            -p|--push)
                PUSH_REGISTRY="$2"
                shift 2
                ;;
            -m|--multi-arch)
                MULTI_ARCH="true"
                shift
                ;;
            -s|--security-scan)
                SECURITY_SCAN="true"
                shift
                ;;
            -c|--no-cache)
                ENABLE_CACHE="false"
                shift
                ;;
            -z|--no-compress)
                COMPRESS="false"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate build type
validate_build_type() {
    case $BUILD_TYPE in
        production|development|debug|alpine|security-hardened|benchmark)
            log_info "Using build type: $BUILD_TYPE"
            ;;
        *)
            log_error "Invalid build type: $BUILD_TYPE"
            log_info "Valid types: production, development, debug, alpine, security-hardened, benchmark"
            exit 1
            ;;
    esac
}

# Setup Docker Buildx
setup_buildx() {
    if [[ "$MULTI_ARCH" == "true" ]] || [[ "$ENABLE_BUILDKIT" == "true" ]]; then
        log_info "Setting up Docker Buildx..."
        
        if ! docker buildx version >/dev/null 2>&1; then
            log_error "Docker Buildx is not available. Please install it."
            exit 1
        fi
        
        # Create builder if it doesn't exist
        if ! docker buildx inspect trustformers-builder >/dev/null 2>&1; then
            log_info "Creating buildx builder instance..."
            docker buildx create --name trustformers-builder --use --bootstrap
        else
            docker buildx use trustformers-builder
        fi
    fi
}

# Get build arguments based on build type
get_build_args() {
    local -a args=()
    
    args+=(--build-arg "RUST_VERSION=$RUST_VERSION")
    args+=(--build-arg "OPTIMIZATION_LEVEL=$OPTIMIZATION_LEVEL")
    
    case $BUILD_TYPE in
        production)
            args+=(--build-arg "BUILD_PROFILE=release")
            args+=(--build-arg 'CARGO_FEATURES=--features production')
            args+=(--target "production")
            ;;
        development)
            args+=(--build-arg "BUILD_PROFILE=dev")
            args+=(--build-arg 'CARGO_FEATURES=--features development')
            args+=(--target "development")
            ;;
        debug)
            args+=(--build-arg "BUILD_PROFILE=debug")
            args+=(--build-arg 'CARGO_FEATURES=--features debug')
            args+=(--target "debug")
            ;;
        alpine)
            args+=(--build-arg "BUILD_PROFILE=release")
            args+=(--build-arg 'CARGO_FEATURES=--features production')
            args+=(--target "alpine-production")
            ;;
        security-hardened)
            args+=(--build-arg "BUILD_PROFILE=release")
            args+=(--build-arg 'CARGO_FEATURES=--features production,security')
            args+=(--target "security-hardened")
            ;;
        benchmark)
            args+=(--build-arg "BUILD_PROFILE=release")
            args+=(--build-arg 'CARGO_FEATURES=--features benchmark')
            args+=(--target "benchmark")
            ;;
    esac
    
    echo "${args[@]}"
}

# Get platforms for multi-arch build
get_platforms() {
    if [[ "$MULTI_ARCH" == "true" ]]; then
        echo "linux/amd64,linux/arm64"
    else
        echo "linux/amd64"
    fi
}

# Build Docker image
build_image() {
    local full_image_name="$IMAGE_NAME:$TAG"
    if [[ -n "$PUSH_REGISTRY" ]]; then
        full_image_name="$PUSH_REGISTRY/$full_image_name"
    fi
    
    log_info "Building Docker image: $full_image_name"
    log_info "Build type: $BUILD_TYPE"
    log_info "Dockerfile: $DOCKERFILE"
    
    # Prepare build command
    local -a build_cmd=()
    
    if [[ "$MULTI_ARCH" == "true" ]] || [[ "$ENABLE_BUILDKIT" == "true" ]]; then
        build_cmd+=(docker buildx build)
        build_cmd+=(--platform "$(get_platforms)")
    else
        build_cmd+=(docker build)
    fi
    
    # Add build arguments
    build_cmd+=($(get_build_args))
    
    # Add cache options
    if [[ "$ENABLE_CACHE" == "true" ]]; then
        build_cmd+=(--cache-from "type=gha")
        build_cmd+=(--cache-to "type=gha,mode=max")
    else
        build_cmd+=(--no-cache)
    fi
    
    # Add labels
    build_cmd+=(--label "org.opencontainers.image.title=TrustformeRS Serve")
    build_cmd+=(--label "org.opencontainers.image.description=High-performance ML inference server")
    build_cmd+=(--label "org.opencontainers.image.version=$TAG")
    build_cmd+=(--label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)")
    build_cmd+=(--label "org.opencontainers.image.build-type=$BUILD_TYPE")
    
    # Add image name and context
    build_cmd+=(--tag "$full_image_name")
    build_cmd+=(--file "$DOCKERFILE")
    
    # Push if registry specified
    if [[ -n "$PUSH_REGISTRY" ]]; then
        build_cmd+=(--push)
    else
        build_cmd+=(--load)
    fi
    
    build_cmd+=(.)
    
    # Execute build
    log_info "Executing: ${build_cmd[*]}"
    "${build_cmd[@]}"
    
    log_success "Build completed successfully!"
}

# Run security scan
run_security_scan() {
    if [[ "$SECURITY_SCAN" == "true" ]]; then
        local full_image_name="$IMAGE_NAME:$TAG"
        if [[ -n "$PUSH_REGISTRY" ]]; then
            full_image_name="$PUSH_REGISTRY/$full_image_name"
        fi
        
        log_info "Running security scan on $full_image_name..."
        
        # Trivy security scan
        if command -v trivy >/dev/null 2>&1; then
            log_info "Running Trivy security scan..."
            trivy image --exit-code 1 --severity HIGH,CRITICAL "$full_image_name"
        else
            log_warning "Trivy not found. Install it for security scanning."
        fi
        
        # Docker security scan (if available)
        if docker scan --help >/dev/null 2>&1; then
            log_info "Running Docker security scan..."
            docker scan "$full_image_name"
        fi
    fi
}

# Compress image if enabled
compress_image() {
    if [[ "$COMPRESS" == "true" ]] && [[ "$BUILD_TYPE" == "production" ]]; then
        local full_image_name="$IMAGE_NAME:$TAG"
        if [[ -n "$PUSH_REGISTRY" ]]; then
            full_image_name="$PUSH_REGISTRY/$full_image_name"
        fi
        
        log_info "Compressing image $full_image_name..."
        
        # Use docker-slim if available
        if command -v docker-slim >/dev/null 2>&1; then
            log_info "Using docker-slim for image optimization..."
            docker-slim build --target "$full_image_name" --tag "${full_image_name}-slim"
        else
            log_warning "docker-slim not found. Skipping image compression."
        fi
    fi
}

# Show image information
show_image_info() {
    local full_image_name="$IMAGE_NAME:$TAG"
    if [[ -n "$PUSH_REGISTRY" ]]; then
        full_image_name="$PUSH_REGISTRY/$full_image_name"
    fi
    
    log_info "Image information:"
    docker image inspect "$full_image_name" --format 'table {{.RepoTags}}\t{{.Size}}\t{{.Created}}' 2>/dev/null || true
    
    # Show image size
    local size
    size=$(docker image inspect "$full_image_name" --format '{{.Size}}' 2>/dev/null || echo "0")
    if [[ "$size" -gt 0 ]]; then
        local size_mb=$((size / 1024 / 1024))
        log_info "Image size: ${size_mb}MB"
    fi
    
    # Show image layers
    log_info "Image layers:"
    docker history "$full_image_name" --no-trunc 2>/dev/null || true
}

# Main execution
main() {
    log_info "TrustformeRS Docker Optimization Build Script"
    log_info "=============================================="
    
    parse_args "$@"
    validate_build_type
    setup_buildx
    build_image
    run_security_scan
    compress_image
    show_image_info
    
    log_success "All operations completed successfully!"
    
    if [[ -n "$PUSH_REGISTRY" ]]; then
        log_info "Image pushed to registry: $PUSH_REGISTRY/$IMAGE_NAME:$TAG"
    else
        log_info "Image available locally: $IMAGE_NAME:$TAG"
    fi
}

# Run main function with all arguments
main "$@"