# Build stage
FROM rust:1.83-bookworm as builder

# Install protobuf compiler
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY trustformers/Cargo.toml ./trustformers/
COPY trustformers-core/Cargo.toml ./trustformers-core/
COPY trustformers-tokenizers/Cargo.toml ./trustformers-tokenizers/
COPY trustformers-models/Cargo.toml ./trustformers-models/
COPY trustformers-optim/Cargo.toml ./trustformers-optim/
COPY trustformers-training/Cargo.toml ./trustformers-training/
COPY trustformers-wasm/Cargo.toml ./trustformers-wasm/
COPY trustformers-py/Cargo.toml ./trustformers-py/
COPY examples/grpc-server/Cargo.toml ./examples/grpc-server/

# Create dummy source files to cache dependencies
RUN mkdir -p trustformers/src && echo "fn main() {}" > trustformers/src/lib.rs
RUN mkdir -p trustformers-core/src && echo "fn main() {}" > trustformers-core/src/lib.rs
RUN mkdir -p trustformers-tokenizers/src && echo "fn main() {}" > trustformers-tokenizers/src/lib.rs
RUN mkdir -p trustformers-models/src && echo "fn main() {}" > trustformers-models/src/lib.rs
RUN mkdir -p trustformers-optim/src && echo "fn main() {}" > trustformers-optim/src/lib.rs
RUN mkdir -p trustformers-training/src && echo "fn main() {}" > trustformers-training/src/lib.rs
RUN mkdir -p trustformers-wasm/src && echo "fn main() {}" > trustformers-wasm/src/lib.rs
RUN mkdir -p trustformers-py/src && echo "fn main() {}" > trustformers-py/src/lib.rs
RUN mkdir -p examples/grpc-server/src && echo "fn main() {}" > examples/grpc-server/src/main.rs

# Build dependencies
RUN cd examples/grpc-server && cargo build --release

# Copy actual source code
COPY . .

# Build the application
RUN cd examples/grpc-server && \
    touch src/main.rs && \
    cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 trustformers

# Copy binary from builder
COPY --from=builder /app/examples/grpc-server/target/release/trustformers-grpc-server /usr/local/bin/

# Create model cache directory
RUN mkdir -p /models && chown trustformers:trustformers /models

# Switch to non-root user
USER trustformers

# Set environment variables
ENV RUST_LOG=info
ENV GRPC_PORT=50051

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/bin/sh", "-c", "echo 'health check' | nc -z localhost 50051 || exit 1"]

# Run the server
CMD ["trustformers-grpc-server"]