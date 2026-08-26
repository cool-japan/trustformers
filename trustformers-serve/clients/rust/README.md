# TrustformeRS Rust Client Library

[![Crates.io](https://img.shields.io/crates/v/trustformers-client.svg)](https://crates.io/crates/trustformers-client)
[![Documentation](https://docs.rs/trustformers-client/badge.svg)](https://docs.rs/trustformers-client)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](../../LICENSE)

A Rust client library for TrustformeRS serving infrastructure.

> **Every snippet below is written against the API this crate actually exposes.**
> Up to 0.2.0 this file documented a different client — `TrustformersClient::new`,
> `client.infer`, `AuthConfig`, `StreamingRequest`, `BatchRequest`,
> `ClientError::Network` — none of which exist here, alongside two example files
> that were never written and a test feature that was never declared. That
> documentation was rewritten in 0.2.1 against the real surface.

## Features

- **Async/await API** built on `tokio`
- **Typed requests and responses** via `serde`
- **Streaming inference** over Server-Sent Events
- **Batch inference**
- **Health and readiness endpoints**, model listing and server metrics
- **Authentication**: API key, JWT, OAuth2 client credentials, or your own
  `Authenticator` implementation
- **Retry logic** with exponential backoff
- **Connection pooling** through `reqwest`

## Installation

```toml
[dependencies]
trustformers-client = "0.2.2"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

The `oauth2` feature is enabled by default; disable default features if you do
not need it.

## Quick Start

Clients are constructed through `TrustformersClient::builder`; there is no
`new` constructor, because the base URL alone is not enough to build one.

```rust,no_run
use trustformers_client::{InferenceRequest, TrustformersClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TrustformersClient::builder("http://localhost:8080")
        .api_key("your-api-key")
        .build()?;

    let request = InferenceRequest::new("Hello, world!").model_id("gpt2");

    let response = client.inference(request).await?;
    for choice in &response.choices {
        println!("{}", choice.text);
    }
    println!("{} tokens", response.usage.total_tokens);

    Ok(())
}
```

### Streaming Inference

`stream_inference` turns streaming on for you, so the request does not have to
set it.

```rust,no_run
use futures_util::StreamExt;
use trustformers_client::{InferenceOptions, InferenceRequest, TrustformersClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TrustformersClient::builder("http://localhost:8080").build()?;

    let request = InferenceRequest::new("Once upon a time")
        .model_id("gpt2")
        .options(InferenceOptions::new().max_tokens(100));

    let mut stream = client.stream_inference(request).await?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                print!("{content}");
            }
        }
    }

    Ok(())
}
```

A chunk that carries no `data:` payload — a keep-alive, for instance — is
reported as `ClientError::Streaming` rather than silently dropped, so a caller
that wants to ignore keep-alives should match on that variant.

### Batch Inference

```rust,no_run
use trustformers_client::{BatchInferenceRequest, BatchOptions, InferenceRequest, TrustformersClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TrustformersClient::builder("http://localhost:8080").build()?;

    let batch = BatchInferenceRequest::new(vec![
        InferenceRequest::new("First text").model_id("gpt2"),
        InferenceRequest::new("Second text").model_id("gpt2"),
    ])
    .options(BatchOptions::new().max_batch_size(8).parallel(true));

    let response = client.batch_inference(batch).await?;
    println!("{} responses in this batch", response.batch_size);

    Ok(())
}
```

### Health, Models and Metrics

```rust,no_run
use trustformers_client::TrustformersClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TrustformersClient::builder("http://localhost:8080").build()?;

    let health = client.health().await?;
    println!("status: {} (up {}s)", health.status, health.uptime);

    // Per-component detail from the same shape, on a different endpoint.
    let detailed = client.detailed_health().await?;
    for (component, status) in &detailed.components {
        println!("  {component}: {status}");
    }

    for model in client.list_models().await? {
        println!("{} ({} parameters)", model.id, model.parameters);
    }

    let metrics = client.get_metrics().await?;
    println!("{} metrics reported", metrics.len());

    Ok(())
}
```

## Authentication

Four authenticators ship with the crate, and the builder has a shortcut for
three of them:

```rust,no_run
use trustformers_client::TrustformersClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // API key (sent as `Authorization: Bearer …` by default)
    let client = TrustformersClient::builder("http://localhost:8080")
        .api_key("your-api-key")
        .build()?;

    // JWT bearer token
    let client = TrustformersClient::builder("http://localhost:8080")
        .jwt_token("your-jwt")
        .build()?;

    // OAuth2 client credentials (requires the default `oauth2` feature)
    let client = TrustformersClient::builder("http://localhost:8080")
        .oauth2("client-id", "client-secret", "https://auth.example.com/token")?
        .build()?;
    Ok(())
}
```

`ApiKeyAuth::with_header` and `JwtAuth::with_header` cover servers that expect a
different header or prefix; `CustomAuth::new` takes a closure over the
`reqwest::RequestBuilder` for anything else. Pass any of them to
`ClientBuilder::authenticator`.

The OAuth2 authenticator implements the client-credentials grant of RFC 6749
section 4.4 directly on this crate's `reqwest`, without the `oauth2` crate: it
fetches a token on first use and caches it until 30 seconds before it expires. A
token response with no `expires_in` is used until the server rejects it, and an
RFC 6749 section 5.2 error body is surfaced as `ClientError::Authentication`
carrying the server's error code and description. Use `with_scopes` to request
scopes:

```rust,no_run
use trustformers_client::{OAuth2Auth, TrustformersClient};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let auth = OAuth2Auth::new("client-id", "client-secret", "https://auth.example.com/token")?
        .with_scopes(["inference.read", "inference.write"]);

    let client = TrustformersClient::builder("http://localhost:8080")
        .authenticator(auth)
        .build()?;
    Ok(())
}
```

## Configuration

```rust,no_run
use std::time::Duration;
use trustformers_client::{ClientConfig, TrustformersClient};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ClientConfig {
        timeout: Duration::from_secs(30),
        max_retries: 3,
        initial_retry_delay: Duration::from_millis(100),
        max_retry_delay: Duration::from_secs(5),
        backoff_multiplier: 2.0,
        retryable_status_codes: vec![429, 500, 502, 503, 504],
        ..Default::default()
    };

    let client = TrustformersClient::builder("http://localhost:8080")
        .config(config)
        .build()?;
    Ok(())
}
```

Connection pooling is configured on the underlying HTTP client, which the
builder accepts directly. This snippet needs a matching `reqwest = "0.13"` in
your own `Cargo.toml`, because the builder takes a `reqwest::ClientBuilder` and
this crate does not re-export it:

```rust,no_run
use std::time::Duration;
use trustformers_client::TrustformersClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TrustformersClient::builder("http://localhost:8080")
        .http_client_builder(
            reqwest::Client::builder()
                .pool_idle_timeout(Duration::from_secs(90))
                .pool_max_idle_per_host(32),
        )
        .build()?;
    Ok(())
}
```

## Error Handling

```rust,no_run
use trustformers_client::{ClientError, InferenceRequest, TrustformersClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TrustformersClient::builder("http://localhost:8080").build()?;

    match client.inference(InferenceRequest::new("hello")).await {
        Ok(response) => println!("{}", response.choices[0].text),
        Err(ClientError::Request(error)) => eprintln!("transport error: {error}"),
        Err(ClientError::Http { status, message }) => {
            eprintln!("server returned {status}: {message}")
        }
        Err(ClientError::Authentication(message)) => eprintln!("auth failed: {message}"),
        Err(ClientError::Timeout) => eprintln!("request timed out"),
        Err(ClientError::MaxRetriesExceeded) => eprintln!("gave up after the retry budget"),
        Err(error) => eprintln!("other error: {error}"),
    }

    Ok(())
}
```

## Examples

- [`basic_client.rs`](examples/basic_client.rs) — health checks, model listing,
  single and batch inference, streaming, and server metrics against a running
  server.

```bash
cargo run --example basic_client
```

## Requirements

- Rust 1.89 or newer (`rust-version` in `Cargo.toml`)
- A `tokio` runtime

## Testing

```bash
cargo test
```

The test suite is hermetic: the OAuth2 tests drive a token endpoint bound to
`127.0.0.1:0` in-process, so no test contacts the network or needs a running
TrustformeRS server.

## Documentation

Full API documentation is available at
[docs.rs/trustformers-client](https://docs.rs/trustformers-client).

## License

Licensed under Apache-2.0.

## Related Projects

- [TrustformeRS](https://github.com/cool-japan/trustformers) — main transformer library
- [TrustformeRS Serve](../../) — serving infrastructure

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md)
for guidelines.
