# Build stage.
#
# `examples/server` is excluded from the repository's cargo workspace and
# carries its own `[workspace]` table, so it is built from inside its own
# directory and its artifacts land in `examples/server/target`.
#
# The whole repository is copied in one step on purpose, mirroring
# `examples/grpc-server/Dockerfile`. An earlier version of this file tried to
# pre-warm the dependency cache by copying a hand-maintained subset of member
# manifests plus dummy `src/lib.rs` files; that kind of list goes stale
# silently (`trustformers`'s own path/workspace dependencies on
# `trustformers-models` / `trustformers-tokenizers` / `trustformers-optim`
# aren't even literal `path = ".."` entries in `trustformers/Cargo.toml` —
# they resolve through the root `[workspace.dependencies]` table — so a
# manually maintained copy list cannot be verified against the manifest by
# inspection alone) and was not verified against a real build in this
# environment (no Docker daemon is available here — see `README.md`).
FROM rust:1.89-bookworm AS builder

WORKDIR /app
COPY . .

WORKDIR /app/examples/server
RUN cargo build --release

# Runtime stage.
FROM debian:bookworm-slim

# `ca-certificates` for outbound TLS: `AutoConfig::from_pretrained` falls back
# to a real model-hub network request when `config.json` isn't found locally
# (see README.md). `curl` for the HEALTHCHECK below. No `libssl3`: the
# TrustformeRS stack is pure Rust and the binary links no OpenSSL.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 trustformers

COPY --from=builder \
    /app/examples/server/target/release/trustformers-server \
    /usr/local/bin/trustformers-server

# Matches the default `MODEL_CACHE_DIR` below and `kubernetes/pvc.yaml`'s
# mount path: a relative `model_name` in `POST /models` resolves under here.
RUN mkdir -p /var/cache/trustformers && chown trustformers:trustformers /var/cache/trustformers

USER trustformers

ENV RUST_LOG=info
ENV PORT=8080
ENV MODEL_CACHE_DIR=/var/cache/trustformers

EXPOSE 8080

# `/health` is a real route (see `src/main.rs`) that returns 200 with the
# running binary's version as soon as the router is serving — it does not
# depend on any model being loaded.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["trustformers-server"]
