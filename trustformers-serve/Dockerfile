# Multi-stage build for optimized production image
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./

# Copy proto files
COPY proto/ ./proto/

# Copy source code
COPY src/ ./src/

# Build for release
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Create non-root user
RUN groupadd -r trustformers && useradd -r -g trustformers -u 1000 trustformers

# Create app directories
RUN mkdir -p /app/config /app/models /app/logs \
    && chown -R trustformers:trustformers /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/trustformers-serve /usr/local/bin/trustformers-serve

# Copy config files
COPY k8s/deployment.yaml /app/config/default.yaml

# Switch to non-root user
USER trustformers

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 8080 9090 9091

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables
ENV RUST_LOG=info
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080
ENV GRPC_PORT=9090
ENV METRICS_PORT=9091

# Run the application
CMD ["trustformers-serve"]