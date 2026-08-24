# Build stage.
#
# `examples/grpc-server` is excluded from the repository's cargo workspace and
# carries its own `[workspace]` table, so it is built from inside its own
# directory and its artifacts land in `examples/grpc-server/target`.
#
# The whole repository is copied in one step on purpose. The previous version of
# this file tried to pre-warm the dependency cache by copying a hand-maintained
# subset of member manifests plus dummy `src/lib.rs` files; that list had gone
# stale (it named `trustformers-py`, which is not a member, and omitted
# `trustformers-mobile`, `trustformers-debug` and `trustformers-serve`, which
# are), so the image could not build at all.
FROM rust:1.89-bookworm AS builder

# `prost-build` shells out to `protoc` to parse `proto/inference.proto`.
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

WORKDIR /app/examples/grpc-server
RUN cargo build --release

# Runtime stage.
FROM debian:bookworm-slim

# `ca-certificates` for outbound TLS (model downloads); `netcat-openbsd` for the
# TCP liveness probe below. No `libssl3`: the binary links no OpenSSL - the
# TrustformeRS stack is pure Rust.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 trustformers

COPY --from=builder \
    /app/examples/grpc-server/target/release/trustformers-grpc-server \
    /usr/local/bin/trustformers-grpc-server

RUN mkdir -p /models && chown trustformers:trustformers /models

USER trustformers

ENV RUST_LOG=info
ENV GRPC_PORT=50051

EXPOSE 50051

# A TCP liveness probe: it proves the listener is accepting connections. The
# server also implements `grpc.health.v1.Health`, which `grpc_health_probe`
# queries properly if you add that binary to the image.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD nc -z localhost 50051 || exit 1

CMD ["trustformers-grpc-server"]
