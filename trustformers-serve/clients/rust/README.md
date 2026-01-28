# TrustformeRS Rust Client Library

[![Crates.io](https://img.shields.io/crates/v/trustformers-client.svg)](https://crates.io/crates/trustformers-client)
[![Documentation](https://docs.rs/trustformers-client/badge.svg)](https://docs.rs/trustformers-client)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-green.svg)](../../LICENSE)

A comprehensive Rust client library for TrustformeRS serving infrastructure.

## Features

- **Async/Await API**: Modern async Rust with `tokio`
- **Type-safe requests**: Strongly-typed request/response types
- **Streaming support**: Real-time streaming inference
- **Batch operations**: Efficient batched requests
- **Health checks**: Built-in health and readiness probes
- **OAuth2 authentication**: Optional OAuth2 support
- **Retry logic**: Automatic retry with exponential backoff
- **Connection pooling**: Efficient HTTP connection reuse

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
trustformers-client = "0.1.0-alpha.2"
tokio = { version = "1.0", features = ["full"] }
```

## Quick Start

### Basic Inference

```rust
use trustformers_client::{TrustformersClient, InferenceRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client
    let client = TrustformersClient::new("http://localhost:8080")?;

    // Prepare request
    let request = InferenceRequest {
        model: "bert-base-uncased".to_string(),
        inputs: vec!["Hello, world!".to_string()],
        max_length: Some(512),
        ..Default::default()
    };

    // Run inference
    let response = client.infer(&request).await?;
    println!("Predictions: {:?}", response.predictions);

    Ok(())
}
```

### Streaming Inference

```rust
use trustformers_client::{TrustformersClient, StreamingRequest};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TrustformersClient::new("http://localhost:8080")?;

    let request = StreamingRequest {
        model: "gpt2".to_string(),
        prompt: "Once upon a time".to_string(),
        max_tokens: Some(100),
        ..Default::default()
    };

    // Stream tokens as they're generated
    let mut stream = client.stream(&request).await?;
    while let Some(token) = stream.next().await {
        let token = token?;
        print!("{}", token.text);
    }

    Ok(())
}
```

### Batch Inference

```rust
use trustformers_client::{TrustformersClient, BatchRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TrustformersClient::new("http://localhost:8080")?;

    let requests = vec![
        InferenceRequest {
            model: "bert-base-uncased".to_string(),
            inputs: vec!["First text".to_string()],
            ..Default::default()
        },
        InferenceRequest {
            model: "bert-base-uncased".to_string(),
            inputs: vec!["Second text".to_string()],
            ..Default::default()
        },
    ];

    let batch = BatchRequest { requests };
    let responses = client.batch_infer(&batch).await?;

    for response in responses {
        println!("Predictions: {:?}", response.predictions);
    }

    Ok(())
}
```

### Health Checks

```rust
use trustformers_client::TrustformersClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TrustformersClient::new("http://localhost:8080")?;

    // Check server health
    let health = client.health().await?;
    println!("Server status: {}", health.status);

    // Check readiness
    let ready = client.ready().await?;
    if ready {
        println!("Server is ready to accept requests");
    }

    Ok(())
}
```

## Authentication

Enable OAuth2 authentication by adding the `oauth2` feature:

```toml
[dependencies]
trustformers-client = { version = "0.1.0-alpha.2", features = ["oauth2"] }
```

Then use authenticated client:

```rust
use trustformers_client::{TrustformersClient, AuthConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let auth = AuthConfig::OAuth2 {
        client_id: "your-client-id".to_string(),
        client_secret: "your-secret".to_string(),
        token_url: "https://auth.example.com/token".to_string(),
    };

    let client = TrustformersClient::with_auth(
        "http://localhost:8080",
        auth,
    ).await?;

    // Client automatically handles token refresh
    let response = client.infer(&request).await?;

    Ok(())
}
```

## Configuration

```rust
use trustformers_client::{TrustformersClient, ClientConfig};
use std::time::Duration;

let config = ClientConfig {
    timeout: Duration::from_secs(30),
    max_retries: 3,
    retry_delay: Duration::from_millis(100),
    pool_idle_timeout: Some(Duration::from_secs(90)),
    pool_max_idle_per_host: Some(32),
    ..Default::default()
};

let client = TrustformersClient::with_config(
    "http://localhost:8080",
    config,
)?;
```

## Error Handling

```rust
use trustformers_client::{TrustformersClient, ClientError};

#[tokio::main]
async fn main() {
    let client = TrustformersClient::new("http://localhost:8080").unwrap();

    match client.infer(&request).await {
        Ok(response) => println!("Success: {:?}", response),
        Err(ClientError::Network(e)) => eprintln!("Network error: {}", e),
        Err(ClientError::Server { status, message }) => {
            eprintln!("Server error {}: {}", status, message)
        }
        Err(ClientError::Timeout) => eprintln!("Request timed out"),
        Err(e) => eprintln!("Other error: {}", e),
    }
}
```

## Examples

See the [`examples/`](examples/) directory for more comprehensive examples:

- [`basic_client.rs`](examples/basic_client.rs) - Basic inference requests
- [`streaming_client.rs`](examples/streaming_client.rs) - Streaming token generation
- [`batch_client.rs`](examples/batch_client.rs) - Batch processing

## Requirements

- Rust 1.75 or higher
- Tokio runtime for async operations

## Testing

Run tests:

```bash
cargo test
```

Run with integration tests (requires running server):

```bash
cargo test --features integration-tests
```

## Documentation

Full API documentation is available at [docs.rs/trustformers-client](https://docs.rs/trustformers-client).

## License

Dual-licensed under MIT OR Apache-2.0, at your option.

## Related Projects

- [TrustformeRS](https://github.com/cool-japan/trustformers) - Main transformer library
- [TrustformeRS Serve](../../) - Serving infrastructure

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.
