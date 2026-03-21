mod error;
#![allow(unused_variables)]
mod model_manager;
mod service;

use std::net::SocketAddr;
use std::sync::Arc;

use tonic::transport::Server;
use tonic_health::server::HealthReporter;
use tracing::{info, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::model_manager::ModelManager;
use crate::service::{
    inference::inference_service_server::InferenceServiceServer, InferenceServiceImpl,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::builder()
                .with_default_directive(Level::INFO.into())
                .from_env_lossy(),
        )
        .init();

    info!("Starting TrustformeRS gRPC server");

    // Configuration
    let addr = "[::1]:50051".parse::<SocketAddr>()?;
    let max_models = 10;
    let default_device = "cpu".to_string();

    // Initialize model manager
    let model_manager = Arc::new(ModelManager::new(max_models, default_device));

    // Create service
    let inference_service = InferenceServiceImpl::new(model_manager);

    // Health service
    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<InferenceServiceServer<InferenceServiceImpl>>()
        .await;

    // Reflection service for grpcurl/grpcui
    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/inference_descriptor.bin"
        )))
        .build()?;

    info!("Server listening on {}", addr);

    // Start server
    Server::builder()
        .add_service(InferenceServiceServer::new(inference_service))
        .add_service(health_service)
        .add_service(reflection_service)
        .serve(addr)
        .await?;

    Ok(())
}