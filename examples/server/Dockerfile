# Build stage
FROM rust:1.78-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml ./
COPY trustformers/Cargo.toml ./trustformers/
COPY trustformers-core/Cargo.toml ./trustformers-core/
COPY trustformers-models/Cargo.toml ./trustformers-models/
COPY trustformers-tokenizers/Cargo.toml ./trustformers-tokenizers/
COPY trustformers-optim/Cargo.toml ./trustformers-optim/
COPY trustformers-training/Cargo.toml ./trustformers-training/
COPY examples/server/Cargo.toml ./examples/server/

# Create dummy source files to cache dependencies
RUN mkdir -p trustformers/src trustformers-core/src trustformers-models/src \
    trustformers-tokenizers/src trustformers-optim/src trustformers-training/src \
    examples/server/src && \
    echo "fn main() {}" > examples/server/src/main.rs && \
    touch trustformers/src/lib.rs trustformers-core/src/lib.rs \
    trustformers-models/src/lib.rs trustformers-tokenizers/src/lib.rs \
    trustformers-optim/src/lib.rs trustformers-training/src/lib.rs

# Build dependencies
WORKDIR /app/examples/server
RUN cargo build --release

# Copy actual source code
WORKDIR /app
COPY . .

# Build the server
WORKDIR /app/examples/server
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 trustformers

# Copy binary from builder
COPY --from=builder /app/examples/server/target/release/trustformers-server /usr/local/bin/

# Create directories for models and cache
RUN mkdir -p /var/cache/trustformers && \
    chown -R trustformers:trustformers /var/cache/trustformers

# Switch to non-root user
USER trustformers

# Environment variables
ENV RUST_LOG=info
ENV MODEL_CACHE_DIR=/var/cache/trustformers
ENV PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the server
CMD ["trustformers-server"]