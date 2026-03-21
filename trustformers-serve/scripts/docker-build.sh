#!/bin/bash

# Docker Build Script for TrustformeRS Serve
# Provides optimized build commands for different scenarios

set -euo pipefail

# Configuration
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
IMAGE_NAME="${IMAGE_NAME:-trustformers-serve}"
VERSION="${VERSION:-latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.optimized}"
RUST_VERSION="${RUST_VERSION:-1.75}"
BUILDKIT_ENABLED="${BUILDKIT_ENABLED:-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Help function
show_help() {
    cat << EOF
Docker Build Script for TrustformeRS Serve

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    production      Build production image (default)
    development     Build development image
    debug          Build debug image with debugging tools
    testing        Build testing image and run tests
    all            Build all variants
    multi-arch     Build multi-architecture images
    security       Build security-hardened image
    clean          Clean up build cache and images
    push           Push images to registry
    scan           Security scan images

Options:
    -t, --tag TAG          Image tag (default: latest)
    -r, --registry REG     Docker registry prefix
    -f, --dockerfile FILE  Dockerfile to use (default: Dockerfile.optimized)
    --no-cache            Build without cache
    --parallel            Build stages in parallel
    --compress            Enable build compression
    --pull                Always pull base images
    -h, --help            Show this help

Environment Variables:
    DOCKER_REGISTRY       Docker registry prefix
    IMAGE_NAME           Image name (default: trustformers-serve)
    VERSION              Version tag (default: latest)
    DOCKERFILE           Dockerfile path (default: Dockerfile.optimized)
    RUST_VERSION         Rust version for build (default: 1.75)
    BUILDKIT_ENABLED     Enable Docker BuildKit (default: 1)

Examples:
    $0 production -t v1.0.0
    $0 development --no-cache
    $0 multi-arch -r docker.io/myorg
    $0 clean
EOF
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if [ "$BUILDKIT_ENABLED" = "1" ]; then
        export DOCKER_BUILDKIT=1
        log_info "Docker BuildKit enabled"
    fi
    
    # Check Docker version
    docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
    log_info "Docker version: $docker_version"
    
    # Check available space
    available_space=$(df /var/lib/docker 2>/dev/null | awk 'NR==2 {print $4}' || echo "unknown")
    if [ "$available_space" != "unknown" ] && [ "$available_space" -lt 5000000 ]; then
        log_warning "Low disk space available for Docker builds"
    fi
}

# Get full image name
get_image_name() {
    local target="$1"
    local tag="$2"
    
    local full_name="$IMAGE_NAME"
    if [ -n "$target" ] && [ "$target" != "production" ]; then
        full_name="$IMAGE_NAME-$target"
    fi
    
    if [ -n "$DOCKER_REGISTRY" ]; then
        full_name="$DOCKER_REGISTRY/$full_name"
    fi
    
    echo "$full_name:$tag"
}

# Build function
build_image() {
    local target="$1"
    local tag="$2"
    shift 2
    local extra_args=("$@")
    
    local image_name
    image_name=$(get_image_name "$target" "$tag")
    
    log_info "Building $target image: $image_name"
    
    local build_args=(
        "--file" "$DOCKERFILE"
        "--target" "$target"
        "--tag" "$image_name"
        "--build-arg" "RUST_VERSION=$RUST_VERSION"
        "--build-arg" "BUILDKIT_INLINE_CACHE=1"
        "--label" "org.opencontainers.image.version=$tag"
        "--label" "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        "--label" "org.opencontainers.image.source=https://github.com/cool-japan/trustformers"
        "--label" "org.opencontainers.image.target=$target"
    )
    
    # Add extra arguments
    build_args+=("${extra_args[@]}")
    
    # Add current directory as build context
    build_args+=(".")
    
    # Build the image
    if docker build "${build_args[@]}"; then
        log_success "Successfully built $image_name"
        
        # Show image size
        size=$(docker images --format "table {{.Size}}" "$image_name" | tail -n1)
        log_info "Image size: $size"
        
        return 0
    else
        log_error "Failed to build $image_name"
        return 1
    fi
}

# Build production image
build_production() {
    local tag="$1"
    shift
    local extra_args=("$@")
    
    build_image "production" "$tag" "${extra_args[@]}"
}

# Build development image
build_development() {
    local tag="$1"
    shift
    local extra_args=("$@")
    
    build_image "development" "$tag" "${extra_args[@]}"
}

# Build debug image
build_debug() {
    local tag="$1"
    shift
    local extra_args=("$@")
    
    build_image "debug" "$tag" "${extra_args[@]}"
}

# Build and run testing
build_testing() {
    local tag="$1"
    shift
    local extra_args=("$@")
    
    log_info "Building testing image and running tests..."
    build_image "testing" "$tag" "${extra_args[@]}"
}

# Build all variants
build_all() {
    local tag="$1"
    shift
    local extra_args=("$@")
    
    log_info "Building all image variants..."
    
    # Build in dependency order
    build_image "deps-builder" "$tag" "${extra_args[@]}" || return 1
    build_image "development" "$tag" "${extra_args[@]}" || return 1
    build_image "testing" "$tag" "${extra_args[@]}" || return 1
    build_image "debug" "$tag" "${extra_args[@]}" || return 1
    build_image "production" "$tag" "${extra_args[@]}" || return 1
    build_image "security-hardened" "$tag" "${extra_args[@]}" || return 1
    
    log_success "All variants built successfully"
}

# Build multi-architecture images
build_multi_arch() {
    local tag="$1"
    shift
    local extra_args=("$@")
    
    log_info "Building multi-architecture images..."
    
    # Create builder instance if needed
    if ! docker buildx inspect multiarch-builder &>/dev/null; then
        log_info "Creating multiarch builder..."
        docker buildx create --name multiarch-builder --use
    fi
    
    # Build for multiple platforms
    local platforms="linux/amd64,linux/arm64"
    local image_name
    image_name=$(get_image_name "production" "$tag")
    
    docker buildx build \
        --platform "$platforms" \
        --file "$DOCKERFILE" \
        --target production \
        --tag "$image_name" \
        --build-arg "RUST_VERSION=$RUST_VERSION" \
        --push \
        "${extra_args[@]}" \
        .
    
    log_success "Multi-architecture build completed"
}

# Security scan
security_scan() {
    local tag="$1"
    
    log_info "Running security scan..."
    
    local image_name
    image_name=$(get_image_name "production" "$tag")
    
    # Check if trivy is available
    if command -v trivy &> /dev/null; then
        log_info "Scanning with Trivy..."
        trivy image "$image_name"
    elif command -v docker &> /dev/null && docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest --version &>/dev/null; then
        log_info "Scanning with Trivy (Docker)..."
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy:latest image "$image_name"
    else
        log_warning "Trivy not available, skipping security scan"
    fi
}

# Push images
push_images() {
    local tag="$1"
    
    if [ -z "$DOCKER_REGISTRY" ]; then
        log_error "DOCKER_REGISTRY not set, cannot push images"
        return 1
    fi
    
    log_info "Pushing images to registry..."
    
    local variants=("production" "development" "debug")
    for variant in "${variants[@]}"; do
        local image_name
        image_name=$(get_image_name "$variant" "$tag")
        
        log_info "Pushing $image_name..."
        if docker push "$image_name"; then
            log_success "Pushed $image_name"
        else
            log_error "Failed to push $image_name"
        fi
    done
}

# Clean up
clean_up() {
    log_info "Cleaning up Docker build cache and unused images..."
    
    # Remove build cache
    docker builder prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove dangling images
    docker images -f "dangling=true" -q | xargs -r docker rmi
    
    log_success "Cleanup completed"
}

# Main function
main() {
    local command="${1:-production}"
    local tag="$VERSION"
    local extra_args=()
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--tag)
                tag="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -f|--dockerfile)
                DOCKERFILE="$2"
                shift 2
                ;;
            --no-cache)
                extra_args+=("--no-cache")
                shift
                ;;
            --parallel)
                extra_args+=("--build-arg" "BUILDKIT_INLINE_CACHE=1")
                shift
                ;;
            --compress)
                extra_args+=("--compress")
                shift
                ;;
            --pull)
                extra_args+=("--pull")
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            production|development|debug|testing|all|multi-arch|security|clean|push|scan)
                command="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites
    
    # Execute command
    case $command in
        production)
            build_production "$tag" "${extra_args[@]}"
            ;;
        development)
            build_development "$tag" "${extra_args[@]}"
            ;;
        debug)
            build_debug "$tag" "${extra_args[@]}"
            ;;
        testing)
            build_testing "$tag" "${extra_args[@]}"
            ;;
        all)
            build_all "$tag" "${extra_args[@]}"
            ;;
        multi-arch)
            build_multi_arch "$tag" "${extra_args[@]}"
            ;;
        security)
            build_image "security-hardened" "$tag" "${extra_args[@]}"
            ;;
        scan)
            security_scan "$tag"
            ;;
        push)
            push_images "$tag"
            ;;
        clean)
            clean_up
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"