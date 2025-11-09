//! Container Orchestration and Optimization for TrustformeRS C API
//!
//! This module provides comprehensive container deployment capabilities including Docker containerization,
//! Kubernetes orchestration, and serverless container optimization for cloud-native deployments.
//!
//! The module is organized into several sub-modules for better maintainability:
//! - `types`: All type definitions and configuration structures
//! - `deployment`: Container deployment management functionality
//! - `optimization`: Performance and resource optimization recommendations
//! - `docker`: Docker-specific functionality (existing module)
//! - `kubernetes`: Kubernetes-specific functionality (existing module)

pub mod docker;
pub mod kubernetes;

// Refactored modules for better organization
pub mod types;
pub mod deployment;
pub mod optimization;

// Re-export key types and functions for backward compatibility
pub use types::*;
pub use deployment::{ContainerDeploymentManager, DeploymentArtifacts, DeploymentStatus, DeploymentPhase};
pub use optimization::{ContainerOptimizer, OptimizationRecommendation, OptimizationCategory,
                      ImplementationEffort, ContainerMetrics};

// Re-export from existing modules
pub use docker::{DockerImageBuilder, DockerImageConfig, DockerOptimizer};
pub use kubernetes::{KubernetesConfig, KubernetesGenerator};

// C API Functions for container operations

use crate::error::{TrustformersError, TrustformersResult};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

/// C API handle for container deployment manager
pub type ContainerDeploymentHandle = *mut ContainerDeploymentManager;

/// Create a new container deployment manager
#[no_mangle]
pub extern "C" fn trustformers_container_deployment_create(
    config_json: *const c_char,
) -> ContainerDeploymentHandle {
    if config_json.is_null() {
        return ptr::null_mut();
    }

    let config_str = match unsafe { CStr::from_ptr(config_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let config: ContainerDeploymentConfig = match serde_json::from_str(config_str) {
        Ok(c) => c,
        Err(_) => return ptr::null_mut(),
    };

    let manager = Box::new(ContainerDeploymentManager::new(config));
    Box::into_raw(manager)
}

/// Generate deployment artifacts
#[no_mangle]
pub extern "C" fn trustformers_container_generate_artifacts(
    handle: ContainerDeploymentHandle,
) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let manager = unsafe { &*handle };

    match manager.generate_deployment_artifacts() {
        Ok(artifacts) => {
            match serde_json::to_string(&artifacts) {
                Ok(json) => match CString::new(json) {
                    Ok(c_str) => c_str.into_raw(),
                    Err(_) => ptr::null_mut(),
                },
                Err(_) => ptr::null_mut(),
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Deploy container using the manager
#[no_mangle]
pub extern "C" fn trustformers_container_deploy(
    handle: ContainerDeploymentHandle,
) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let manager = unsafe { &*handle };

    match manager.deploy() {
        Ok(status) => {
            match serde_json::to_string(&status) {
                Ok(json) => match CString::new(json) {
                    Ok(c_str) => c_str.into_raw(),
                    Err(_) => ptr::null_mut(),
                },
                Err(_) => ptr::null_mut(),
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Get deployment status
#[no_mangle]
pub extern "C" fn trustformers_container_get_status(
    handle: ContainerDeploymentHandle,
) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let manager = unsafe { &*handle };

    match manager.get_deployment_status() {
        Ok(status) => {
            match serde_json::to_string(&status) {
                Ok(json) => match CString::new(json) {
                    Ok(c_str) => c_str.into_raw(),
                    Err(_) => ptr::null_mut(),
                },
                Err(_) => ptr::null_mut(),
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Scale deployment
#[no_mangle]
pub extern "C" fn trustformers_container_scale(
    handle: ContainerDeploymentHandle,
    replicas: c_int,
) -> c_int {
    if handle.is_null() || replicas < 0 {
        return -1;
    }

    let manager = unsafe { &*handle };

    match manager.scale_deployment(replicas as u32) {
        Ok(()) => 0, // Success
        Err(_) => -1, // Error
    }
}

/// Delete deployment
#[no_mangle]
pub extern "C" fn trustformers_container_delete(
    handle: ContainerDeploymentHandle,
) -> c_int {
    if handle.is_null() {
        return -1;
    }

    let manager = unsafe { &*handle };

    match manager.delete_deployment() {
        Ok(()) => 0, // Success
        Err(_) => -1, // Error
    }
}

/// Get optimization recommendations
#[no_mangle]
pub extern "C" fn trustformers_container_get_optimizations(
    config_json: *const c_char,
) -> *mut c_char {
    if config_json.is_null() {
        return ptr::null_mut();
    }

    let config_str = match unsafe { CStr::from_ptr(config_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let config: ContainerDeploymentConfig = match serde_json::from_str(config_str) {
        Ok(c) => c,
        Err(_) => return ptr::null_mut(),
    };

    match ContainerOptimizer::optimize_configuration(&config) {
        Ok(recommendations) => {
            match serde_json::to_string(&recommendations) {
                Ok(json) => match CString::new(json) {
                    Ok(c_str) => c_str.into_raw(),
                    Err(_) => ptr::null_mut(),
                },
                Err(_) => ptr::null_mut(),
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Analyze container performance
#[no_mangle]
pub extern "C" fn trustformers_container_analyze_performance(
    metrics_json: *const c_char,
) -> *mut c_char {
    if metrics_json.is_null() {
        return ptr::null_mut();
    }

    let metrics_str = match unsafe { CStr::from_ptr(metrics_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let metrics: ContainerMetrics = match serde_json::from_str(metrics_str) {
        Ok(m) => m,
        Err(_) => return ptr::null_mut(),
    };

    let recommendations = ContainerOptimizer::analyze_performance(&metrics);

    match serde_json::to_string(&recommendations) {
        Ok(json) => match CString::new(json) {
            Ok(c_str) => c_str.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        Err(_) => ptr::null_mut(),
    }
}

/// Generate optimization report
#[no_mangle]
pub extern "C" fn trustformers_container_optimization_report(
    config_json: *const c_char,
    recommendations_json: *const c_char,
) -> *mut c_char {
    if config_json.is_null() || recommendations_json.is_null() {
        return ptr::null_mut();
    }

    let config_str = match unsafe { CStr::from_ptr(config_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let recommendations_str = match unsafe { CStr::from_ptr(recommendations_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let config: ContainerDeploymentConfig = match serde_json::from_str(config_str) {
        Ok(c) => c,
        Err(_) => return ptr::null_mut(),
    };

    let recommendations: Vec<OptimizationRecommendation> = match serde_json::from_str(recommendations_str) {
        Ok(r) => r,
        Err(_) => return ptr::null_mut(),
    };

    let report = ContainerOptimizer::generate_optimization_report(&config, &recommendations);

    match CString::new(report) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free container deployment manager
#[no_mangle]
pub extern "C" fn trustformers_container_deployment_free(handle: ContainerDeploymentHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

/// Free C string allocated by container functions
#[no_mangle]
pub extern "C" fn trustformers_container_free_string(str_ptr: *mut c_char) {
    if !str_ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(str_ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_deployment_creation() {
        let config = ContainerDeploymentConfig {
            platform: ContainerPlatform::Docker,
            app_name: "test-app".to_string(),
            environment: "test".to_string(),
            image_config: DockerImageConfig {
                base_image: "nginx".to_string(),
                tag: "latest".to_string(),
                build_context: ".".to_string(),
                dockerfile_path: "Dockerfile".to_string(),
                build_args: std::collections::HashMap::new(),
                env_vars: std::collections::HashMap::new(),
                exposed_ports: vec![80],
                volumes: vec![],
            },
            orchestration: OrchestrationConfig::DockerSwarm(DockerSwarmConfig {
                replicas: 3,
                update_config: SwarmUpdateConfig {
                    parallelism: 1,
                    delay: "10s".to_string(),
                    failure_action: SwarmFailureAction::Pause,
                    order: SwarmUpdateOrder::StopFirst,
                },
                resources: SwarmResourceLimits {
                    cpu_limit: 1.0,
                    memory_limit: 512,
                    cpu_reservation: None,
                    memory_reservation: None,
                },
                constraints: vec![],
                labels: std::collections::HashMap::new(),
                networks: vec!["overlay".to_string()],
            }),
            serverless: None,
            monitoring: MonitoringConfig {
                enabled: true,
                metrics_interval: 30,
                log_aggregation: LogAggregationConfig {
                    enabled: true,
                    log_level: "info".to_string(),
                    format: "json".to_string(),
                    destination: "stdout".to_string(),
                },
                health_checks: HealthCheckConfig {
                    enabled: true,
                    endpoint: "/health".to_string(),
                    interval: 30,
                    timeout: 5,
                    failure_threshold: 3,
                },
                alerting: AlertingConfig {
                    enabled: false,
                    rules: vec![],
                    notification_channels: vec![],
                },
            },
            auto_scaling: None,
            security: SecurityConfig {
                security_scanning: true,
                image_scanning: ImageScanningConfig {
                    enabled: true,
                    scan_on_push: true,
                    vulnerability_thresholds: VulnerabilityThresholds {
                        critical: 0,
                        high: 5,
                        medium: 10,
                    },
                },
                runtime_security: RuntimeSecurityConfig {
                    enabled: true,
                    policies: vec![],
                    compliance_checks: vec![],
                },
                network_policies: vec![],
                secret_management: SecretManagementConfig {
                    store_type: "kubernetes".to_string(),
                    encryption: EncryptionConfig {
                        algorithm: "AES256".to_string(),
                        key_management: "kms".to_string(),
                    },
                    rotation_policy: RotationPolicy {
                        interval: "90d".to_string(),
                        automatic: true,
                    },
                },
            },
            network: NetworkConfig {
                network_mode: "bridge".to_string(),
                port_mappings: vec![PortMapping {
                    host_port: 8080,
                    container_port: 80,
                    protocol: "tcp".to_string(),
                }],
                load_balancer: None,
                service_mesh: None,
            },
            storage: StorageConfig {
                persistent_volumes: vec![],
                temp_storage: TempStorageConfig {
                    size_limit: "1Gi".to_string(),
                    cleanup_policy: "always".to_string(),
                },
                backup: None,
            },
        };

        let manager = ContainerDeploymentManager::new(config);
        assert_eq!(manager.config.app_name, "test-app");
    }

    #[test]
    fn test_optimization_recommendations() {
        let config = ContainerDeploymentConfig {
            platform: ContainerPlatform::Kubernetes,
            app_name: "test-app".to_string(),
            environment: "prod".to_string(),
            image_config: DockerImageConfig {
                base_image: "nginx".to_string(),
                tag: "latest".to_string(),
                build_context: ".".to_string(),
                dockerfile_path: "Dockerfile".to_string(),
                build_args: std::collections::HashMap::new(),
                env_vars: std::collections::HashMap::new(),
                exposed_ports: vec![80],
                volumes: vec![],
            },
            orchestration: OrchestrationConfig::DockerSwarm(DockerSwarmConfig {
                replicas: 1,
                update_config: SwarmUpdateConfig {
                    parallelism: 1,
                    delay: "10s".to_string(),
                    failure_action: SwarmFailureAction::Pause,
                    order: SwarmUpdateOrder::StopFirst,
                },
                resources: SwarmResourceLimits {
                    cpu_limit: 0.5,
                    memory_limit: 256,
                    cpu_reservation: None,
                    memory_reservation: None,
                },
                constraints: vec![],
                labels: std::collections::HashMap::new(),
                networks: vec![],
            }),
            serverless: None,
            monitoring: MonitoringConfig {
                enabled: false,
                metrics_interval: 60,
                log_aggregation: LogAggregationConfig {
                    enabled: false,
                    log_level: "warn".to_string(),
                    format: "text".to_string(),
                    destination: "file".to_string(),
                },
                health_checks: HealthCheckConfig {
                    enabled: false,
                    endpoint: "/".to_string(),
                    interval: 60,
                    timeout: 10,
                    failure_threshold: 5,
                },
                alerting: AlertingConfig {
                    enabled: false,
                    rules: vec![],
                    notification_channels: vec![],
                },
            },
            auto_scaling: None,
            security: SecurityConfig {
                security_scanning: false,
                image_scanning: ImageScanningConfig {
                    enabled: false,
                    scan_on_push: false,
                    vulnerability_thresholds: VulnerabilityThresholds {
                        critical: 10,
                        high: 20,
                        medium: 50,
                    },
                },
                runtime_security: RuntimeSecurityConfig {
                    enabled: false,
                    policies: vec![],
                    compliance_checks: vec![],
                },
                network_policies: vec![],
                secret_management: SecretManagementConfig {
                    store_type: "file".to_string(),
                    encryption: EncryptionConfig {
                        algorithm: "none".to_string(),
                        key_management: "manual".to_string(),
                    },
                    rotation_policy: RotationPolicy {
                        interval: "never".to_string(),
                        automatic: false,
                    },
                },
            },
            network: NetworkConfig {
                network_mode: "host".to_string(),
                port_mappings: vec![],
                load_balancer: None,
                service_mesh: None,
            },
            storage: StorageConfig {
                persistent_volumes: vec![],
                temp_storage: TempStorageConfig {
                    size_limit: "100Mi".to_string(),
                    cleanup_policy: "never".to_string(),
                },
                backup: None,
            },
        };

        let recommendations = ContainerOptimizer::optimize_configuration(&config).unwrap();
        assert!(!recommendations.is_empty());

        // Should recommend enabling monitoring
        let has_monitoring_rec = recommendations.iter()
            .any(|r| r.title.contains("Enable Monitoring"));
        assert!(has_monitoring_rec);
    }

    #[test]
    fn test_performance_analysis() {
        let metrics = ContainerMetrics {
            cpu_utilization: 85.0,
            memory_utilization: 90.0,
            network_io: 1000000,
            disk_io: 500000,
            response_time: 1500.0,
            throughput: 100.0,
            error_rate: 8.0,
        };

        let recommendations = ContainerOptimizer::analyze_performance(&metrics);
        assert!(!recommendations.is_empty());

        // Should have recommendations for high CPU, memory, response time, and error rate
        assert!(recommendations.iter().any(|r| r.title.contains("High CPU")));
        assert!(recommendations.iter().any(|r| r.title.contains("High Memory")));
        assert!(recommendations.iter().any(|r| r.title.contains("High Response Time")));
        assert!(recommendations.iter().any(|r| r.title.contains("High Error Rate")));
    }
}