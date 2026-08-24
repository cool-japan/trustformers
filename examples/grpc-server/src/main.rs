//! gRPC front end for TrustformeRS checkpoints.
//!
//! Configuration comes from the environment, which is what
//! `docker-compose.yml` sets:
//!
//! | variable         | default        | meaning                                    |
//! |------------------|----------------|--------------------------------------------|
//! | `GRPC_ADDR`      | `[::]:$PORT`   | full listen address; overrides `GRPC_PORT`  |
//! | `GRPC_PORT`      | `50051`        | port to listen on                           |
//! | `MAX_MODELS`     | `10`           | how many checkpoints may be loaded at once  |
//! | `DEFAULT_DEVICE` | `cpu`          | device used when a request names none       |

mod error;
mod model_manager;
mod service;

use std::env;
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;

use tonic::transport::Server;
use tracing::{info, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::model_manager::ModelManager;
use crate::service::{
    inference::inference_service_server::InferenceServiceServer, InferenceServiceImpl,
};

/// Port used when neither `GRPC_ADDR` nor `GRPC_PORT` is set.
const DEFAULT_PORT: u16 = 50051;
/// Concurrent checkpoint limit used when `MAX_MODELS` is not set.
const DEFAULT_MAX_MODELS: usize = 10;
/// Device used when neither the request nor `DEFAULT_DEVICE` names one.
const DEFAULT_DEVICE: &str = "cpu";

/// Reads `name` from the environment, parsing it into `T`.
///
/// An unset variable yields `default`; a set-but-unparseable one is a hard
/// error, because silently falling back to the default would hide a typo in a
/// deployment's configuration.
fn env_or<T>(name: &str, default: T) -> Result<T, Box<dyn std::error::Error>>
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Display,
{
    match env::var(name) {
        Ok(raw) => raw
            .trim()
            .parse::<T>()
            .map_err(|e| format!("environment variable {name}=`{raw}` is invalid: {e}").into()),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(format!("environment variable {name} could not be read: {e}").into()),
    }
}

/// Resolves the listen address from `GRPC_ADDR`, or from `GRPC_PORT` on all
/// interfaces (which is what the bundled `docker-compose.yml` port mapping
/// needs).
fn listen_addr() -> Result<SocketAddr, Box<dyn std::error::Error>> {
    if let Ok(raw) = env::var("GRPC_ADDR") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return trimmed
                .parse::<SocketAddr>()
                .map_err(|e| format!("environment variable GRPC_ADDR=`{raw}` is invalid: {e}").into());
        }
    }
    let port = env_or("GRPC_PORT", DEFAULT_PORT)?;
    Ok(SocketAddr::from(([0, 0, 0, 0], port)))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::builder()
                .with_default_directive(Level::INFO.into())
                .from_env_lossy(),
        )
        .init();

    info!("starting TrustformeRS gRPC server");

    let addr = listen_addr()?;
    let max_models = env_or("MAX_MODELS", DEFAULT_MAX_MODELS)?;
    let default_device = match env::var("DEFAULT_DEVICE") {
        Ok(device) if !device.trim().is_empty() => device.trim().to_string(),
        _ => DEFAULT_DEVICE.to_string(),
    };

    // `ModelManager::new` rejects a `MAX_MODELS` of 0 and a `DEFAULT_DEVICE`
    // this build cannot run on, so a misconfigured deployment fails at startup
    // instead of failing every request later with a misleading message.
    let model_manager = Arc::new(ModelManager::new(max_models, default_device.clone())?);
    let inference_service = InferenceServiceImpl::new(model_manager);

    // gRPC health checking (`grpc.health.v1.Health`).
    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<InferenceServiceServer<InferenceServiceImpl>>()
        .await;

    // Server reflection for grpcurl / grpcui. tonic-reflection 0.14 replaced
    // `build()` with the explicit `build_v1()` / `build_v1alpha()` pair; both
    // are served so older reflection clients keep working.
    let descriptor_set = include_bytes!(concat!(env!("OUT_DIR"), "/inference_descriptor.bin"));
    let reflection_v1 = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(descriptor_set)
        .build_v1()?;
    let reflection_v1alpha = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(descriptor_set)
        .build_v1alpha()?;

    info!(%addr, max_models, default_device, "server listening");

    Server::builder()
        .add_service(InferenceServiceServer::new(inference_service))
        .add_service(health_service)
        .add_service(reflection_v1)
        .add_service(reflection_v1alpha)
        .serve(addr)
        .await?;

    Ok(())
}
