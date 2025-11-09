//! Docker Container Optimization for TrustformeRS C API
//!
//! This module provides comprehensive Docker containerization support with multi-stage builds,
//! image optimization, security hardening, and performance tuning for TrustformeRS applications.

use crate::error::{TrustformersError, TrustformersResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Docker image configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerImageConfig {
    /// Base image
    pub base_image: BaseImage,
    /// Build configuration
    pub build_config: BuildConfig,
    /// Runtime configuration
    pub runtime_config: RuntimeConfig,
    /// Security configuration
    pub security_config: SecurityConfig,
    /// Optimization settings
    pub optimization: OptimizationConfig,
    /// Health check configuration
    pub health_check: Option<HealthCheckConfig>,
}

/// Base image options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaseImage {
    /// Debian-based images
    Debian {
        /// Debian version (bullseye, bookworm, etc.)
        version: String,
        /// Use slim variant
        slim: bool,
    },
    /// Ubuntu-based images
    Ubuntu {
        /// Ubuntu version
        version: String,
        /// Use minimal variant
        minimal: bool,
    },
    /// Alpine Linux (minimal)
    Alpine {
        /// Alpine version
        version: String,
    },
    /// Google Distroless
    Distroless {
        /// Distroless variant (cc, static, etc.)
        variant: DistrolessVariant,
    },
    /// Red Hat Universal Base Image
    UBI {
        /// UBI version
        version: String,
        /// Use minimal variant
        minimal: bool,
    },
    /// Scratch (empty base)
    Scratch,
}

/// Distroless variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistrolessVariant {
    /// CC (C/C++ runtime)
    CC,
    /// Static (static binaries only)
    Static,
    /// Base (with shell)
    Base,
}

/// Build configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Enable multi-stage build
    pub multi_stage: bool,
    /// Target architecture
    pub target_arch: TargetArchitecture,
    /// Build arguments
    pub build_args: HashMap<String, String>,
    /// Environment variables
    pub env_vars: HashMap<String, String>,
    /// Working directory
    pub workdir: String,
    /// Build dependencies
    pub build_dependencies: Vec<String>,
    /// Runtime dependencies
    pub runtime_dependencies: Vec<String>,
}

/// Target architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetArchitecture {
    /// AMD64/x86_64
    AMD64,
    /// ARM64/aarch64
    ARM64,
    /// ARM v7
    ARMv7,
    /// Multi-architecture build
    MultiArch(Vec<TargetArchitecture>),
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Exposed ports
    pub ports: Vec<u16>,
    /// Volume mounts
    pub volumes: Vec<VolumeMount>,
    /// Resource limits
    pub resources: ResourceLimits,
    /// User configuration
    pub user: UserConfig,
    /// Entrypoint configuration
    pub entrypoint: EntrypointConfig,
}

/// Volume mount configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMount {
    /// Mount path
    pub path: String,
    /// Mount type
    pub mount_type: MountType,
    /// Read-only flag
    pub read_only: bool,
}

/// Mount types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MountType {
    /// Named volume
    Volume,
    /// Bind mount
    Bind,
    /// tmpfs mount
    Tmpfs,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Memory limit
    pub memory: Option<String>,
    /// CPU limit
    pub cpu: Option<String>,
    /// Swap limit
    pub swap: Option<String>,
    /// PID limit
    pub pids: Option<u32>,
}

/// User configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConfig {
    /// User ID
    pub uid: u32,
    /// Group ID
    pub gid: u32,
    /// Username
    pub username: Option<String>,
    /// Group name
    pub groupname: Option<String>,
}

/// Entrypoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntrypointConfig {
    /// Entrypoint command
    pub command: Vec<String>,
    /// Default arguments
    pub args: Vec<String>,
    /// Signal handling
    pub signal_handling: bool,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Run as non-root
    pub non_root: bool,
    /// Read-only root filesystem
    pub read_only_root: bool,
    /// Security options
    pub security_opts: Vec<String>,
    /// Capabilities to drop
    pub drop_capabilities: Vec<String>,
    /// Capabilities to add
    pub add_capabilities: Vec<String>,
    /// AppArmor/SELinux profile
    pub security_profile: Option<String>,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable layer caching
    pub layer_caching: bool,
    /// Minimize layers
    pub minimize_layers: bool,
    /// Strip debug symbols
    pub strip_debug: bool,
    /// Compress with upx
    pub compress_binary: bool,
    /// Remove package caches
    pub clean_package_cache: bool,
    /// Remove development tools
    pub remove_dev_tools: bool,
    /// Image scanning
    pub enable_scanning: bool,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check command
    pub command: Vec<String>,
    /// Check interval
    pub interval: String,
    /// Timeout
    pub timeout: String,
    /// Start period
    pub start_period: Option<String>,
    /// Retries
    pub retries: u32,
}

/// Docker image builder
pub struct DockerImageBuilder {
    /// Configuration
    config: DockerImageConfig,
}

impl DockerImageBuilder {
    /// Create new Docker image builder
    pub fn new(config: DockerImageConfig) -> Self {
        Self { config }
    }

    /// Generate Dockerfile
    pub fn generate_dockerfile(&self) -> TrustformersResult<String> {
        let mut dockerfile = String::new();

        // Multi-stage build setup
        if self.config.build_config.multi_stage {
            dockerfile.push_str(&self.generate_multi_stage_dockerfile()?);
        } else {
            dockerfile.push_str(&self.generate_single_stage_dockerfile()?);
        }

        Ok(dockerfile)
    }

    /// Generate multi-stage Dockerfile
    fn generate_multi_stage_dockerfile(&self) -> TrustformersResult<String> {
        let mut dockerfile = String::new();

        // Build stage
        dockerfile.push_str(&format!(
            "# Build stage\nFROM {} AS builder\n\n",
            self.get_base_image_name(&BaseImage::Debian {
                version: "bullseye".to_string(),
                slim: false,
            })
        ));

        // Install build dependencies
        dockerfile.push_str("# Install build dependencies\n");
        dockerfile.push_str("RUN apt-get update && apt-get install -y \\\n");
        for dep in &self.config.build_config.build_dependencies {
            dockerfile.push_str(&format!("    {} \\\n", dep));
        }
        dockerfile.push_str("    && rm -rf /var/lib/apt/lists/*\n\n");

        // Install Rust toolchain
        dockerfile.push_str(&self.generate_rust_installation());

        // Set build arguments
        for (key, value) in &self.config.build_config.build_args {
            dockerfile.push_str(&format!("ARG {}={}\n", key, value));
        }
        dockerfile.push('\n');

        // Set working directory
        dockerfile.push_str(&format!("WORKDIR {}\n\n", self.config.build_config.workdir));

        // Copy source code
        dockerfile.push_str("# Copy source code\n");
        dockerfile.push_str("COPY . .\n\n");

        // Build the application
        dockerfile.push_str(&self.generate_build_commands());

        // Runtime stage
        dockerfile.push_str(&format!(
            "\n# Runtime stage\nFROM {}\n\n",
            self.get_base_image_name(&self.config.base_image)
        ));

        // Install runtime dependencies
        if !self.config.build_config.runtime_dependencies.is_empty() {
            dockerfile.push_str("# Install runtime dependencies\n");
            dockerfile.push_str(&self.generate_runtime_deps_installation());
        }

        // Create user
        if self.config.security_config.non_root {
            dockerfile.push_str(&self.generate_user_creation());
        }

        // Copy binary from build stage
        dockerfile.push_str("# Copy binary from build stage\n");
        dockerfile.push_str(&format!(
            "COPY --from=builder {}/target/release/trustformers-c /usr/local/bin/trustformers-c\n",
            self.config.build_config.workdir
        ));
        dockerfile.push_str("RUN chmod +x /usr/local/bin/trustformers-c\n\n");

        // Configure runtime
        dockerfile.push_str(&self.generate_runtime_configuration());

        Ok(dockerfile)
    }

    /// Generate single-stage Dockerfile
    fn generate_single_stage_dockerfile(&self) -> TrustformersResult<String> {
        let mut dockerfile = String::new();

        dockerfile.push_str(&format!(
            "FROM {}\n\n",
            self.get_base_image_name(&self.config.base_image)
        ));

        // Install all dependencies
        dockerfile.push_str("# Install dependencies\n");
        dockerfile.push_str("RUN apt-get update && apt-get install -y \\\n");

        let mut all_deps = self.config.build_config.build_dependencies.clone();
        all_deps.extend(self.config.build_config.runtime_dependencies.clone());

        for dep in &all_deps {
            dockerfile.push_str(&format!("    {} \\\n", dep));
        }
        dockerfile.push_str("    && rm -rf /var/lib/apt/lists/*\n\n");

        // Install Rust
        dockerfile.push_str(&self.generate_rust_installation());

        // Set working directory
        dockerfile.push_str(&format!("WORKDIR {}\n\n", self.config.build_config.workdir));

        // Copy and build
        dockerfile.push_str("COPY . .\n");
        dockerfile.push_str(&self.generate_build_commands());

        // Clean up build artifacts if optimizing
        if self.config.optimization.remove_dev_tools {
            dockerfile.push_str(&self.generate_cleanup_commands());
        }

        // Configure runtime
        dockerfile.push_str(&self.generate_runtime_configuration());

        Ok(dockerfile)
    }

    /// Get base image name
    fn get_base_image_name(&self, base_image: &BaseImage) -> String {
        match base_image {
            BaseImage::Debian { version, slim } => {
                if *slim {
                    format!("debian:{}-slim", version)
                } else {
                    format!("debian:{}", version)
                }
            },
            BaseImage::Ubuntu { version, minimal } => {
                if *minimal {
                    format!("ubuntu:{}-minimal", version)
                } else {
                    format!("ubuntu:{}", version)
                }
            },
            BaseImage::Alpine { version } => format!("alpine:{}", version),
            BaseImage::Distroless { variant } => match variant {
                DistrolessVariant::CC => "gcr.io/distroless/cc-debian11".to_string(),
                DistrolessVariant::Static => "gcr.io/distroless/static-debian11".to_string(),
                DistrolessVariant::Base => "gcr.io/distroless/base-debian11".to_string(),
            },
            BaseImage::UBI { version, minimal } => {
                if *minimal {
                    format!("registry.access.redhat.com/ubi{}/ubi-minimal", version)
                } else {
                    format!("registry.access.redhat.com/ubi{}/ubi", version)
                }
            },
            BaseImage::Scratch => "scratch".to_string(),
        }
    }

    /// Generate Rust installation commands
    fn generate_rust_installation(&self) -> String {
        let mut commands = String::new();

        commands.push_str("# Install Rust\n");
        commands.push_str("ENV RUSTUP_HOME=/usr/local/rustup \\\n");
        commands.push_str("    CARGO_HOME=/usr/local/cargo \\\n");
        commands.push_str("    PATH=/usr/local/cargo/bin:$PATH\n\n");

        commands.push_str("RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path && \\\n");
        commands.push_str("    chmod -R a+w $RUSTUP_HOME $CARGO_HOME\n\n");

        // Add target architecture if cross-compiling
        match &self.config.build_config.target_arch {
            TargetArchitecture::ARM64 => {
                commands.push_str("RUN rustup target add aarch64-unknown-linux-gnu\n\n");
            },
            TargetArchitecture::ARMv7 => {
                commands.push_str("RUN rustup target add armv7-unknown-linux-gnueabihf\n\n");
            },
            _ => {},
        }

        commands
    }

    /// Generate build commands
    fn generate_build_commands(&self) -> String {
        let mut commands = String::new();

        commands.push_str("# Build the application\n");

        // Set environment variables
        for (key, value) in &self.config.build_config.env_vars {
            commands.push_str(&format!("ENV {}={}\n", key, value));
        }
        commands.push('\n');

        // Build command
        let target_flag = match &self.config.build_config.target_arch {
            TargetArchitecture::ARM64 => " --target aarch64-unknown-linux-gnu",
            TargetArchitecture::ARMv7 => " --target armv7-unknown-linux-gnueabihf",
            _ => "",
        };

        commands.push_str(&format!("RUN cargo build --release{} && \\\n", target_flag));

        if self.config.optimization.strip_debug {
            commands.push_str("    strip target/release/trustformers-c && \\\n");
        }

        if self.config.optimization.compress_binary {
            commands.push_str("    upx --best target/release/trustformers-c && \\\n");
        }

        commands.push_str("    ls -la target/release/\n\n");

        commands
    }

    /// Generate runtime dependencies installation
    fn generate_runtime_deps_installation(&self) -> String {
        let mut commands = String::new();

        match &self.config.base_image {
            BaseImage::Debian { .. } | BaseImage::Ubuntu { .. } => {
                commands.push_str("RUN apt-get update && apt-get install -y \\\n");
                for dep in &self.config.build_config.runtime_dependencies {
                    commands.push_str(&format!("    {} \\\n", dep));
                }
                commands.push_str("    && rm -rf /var/lib/apt/lists/*\n\n");
            },
            BaseImage::Alpine { .. } => {
                commands.push_str("RUN apk add --no-cache \\\n");
                for dep in &self.config.build_config.runtime_dependencies {
                    commands.push_str(&format!("    {} \\\n", dep));
                }
                commands.push('\n');
            },
            BaseImage::UBI { .. } => {
                commands.push_str("RUN dnf install -y \\\n");
                for dep in &self.config.build_config.runtime_dependencies {
                    commands.push_str(&format!("    {} \\\n", dep));
                }
                commands.push_str("    && dnf clean all\n\n");
            },
            _ => {
                // No package manager available
            },
        }

        commands
    }

    /// Generate user creation commands
    fn generate_user_creation(&self) -> String {
        let mut commands = String::new();

        commands.push_str("# Create non-root user\n");

        let username =
            self.config.runtime_config.user.username.as_deref().unwrap_or("trustformers");
        let groupname =
            self.config.runtime_config.user.groupname.as_deref().unwrap_or("trustformers");

        commands.push_str(&format!(
            "RUN groupadd -g {} {} && \\\n",
            self.config.runtime_config.user.gid, groupname
        ));
        commands.push_str(&format!(
            "    useradd -u {} -g {} -M -s /bin/false {} && \\\n",
            self.config.runtime_config.user.uid, self.config.runtime_config.user.gid, username
        ));
        commands.push_str("    mkdir -p /home/trustformers && \\\n");
        commands.push_str(&format!(
            "    chown {}:{} /home/trustformers\n\n",
            username, groupname
        ));

        commands
    }

    /// Generate runtime configuration
    fn generate_runtime_configuration(&self) -> String {
        let mut config = String::new();

        // Set user
        if self.config.security_config.non_root {
            let username =
                self.config.runtime_config.user.username.as_deref().unwrap_or("trustformers");
            config.push_str(&format!("USER {}\n\n", username));
        }

        // Set working directory
        config.push_str(&format!("WORKDIR {}\n\n", self.config.build_config.workdir));

        // Expose ports
        for port in &self.config.runtime_config.ports {
            config.push_str(&format!("EXPOSE {}\n", port));
        }
        if !self.config.runtime_config.ports.is_empty() {
            config.push('\n');
        }

        // Add volumes
        for volume in &self.config.runtime_config.volumes {
            config.push_str(&format!("VOLUME [\"{}\"]\n", volume.path));
        }
        if !self.config.runtime_config.volumes.is_empty() {
            config.push('\n');
        }

        // Health check
        if let Some(health_check) = &self.config.health_check {
            config.push_str(&format!(
                "HEALTHCHECK --interval={} --timeout={} --retries={} \\\n",
                health_check.interval, health_check.timeout, health_check.retries
            ));

            if let Some(start_period) = &health_check.start_period {
                config.push_str(&format!("    --start-period={} \\\n", start_period));
            }

            config.push_str(&format!("    CMD {}\n\n", health_check.command.join(" ")));
        }

        // Entrypoint and CMD
        config.push_str(&format!(
            "ENTRYPOINT [{}]\n",
            self.config
                .runtime_config
                .entrypoint
                .command
                .iter()
                .map(|s| format!("\"{}\"", s))
                .collect::<Vec<_>>()
                .join(", ")
        ));

        if !self.config.runtime_config.entrypoint.args.is_empty() {
            config.push_str(&format!(
                "CMD [{}]\n",
                self.config
                    .runtime_config
                    .entrypoint
                    .args
                    .iter()
                    .map(|s| format!("\"{}\"", s))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        config
    }

    /// Generate cleanup commands
    fn generate_cleanup_commands(&self) -> String {
        let mut commands = String::new();

        commands.push_str("# Clean up build artifacts\n");
        commands.push_str("RUN cargo clean && \\\n");
        commands.push_str("    rustup self uninstall -y && \\\n");
        commands.push_str("    apt-get remove -y \\\n");

        for dep in &self.config.build_config.build_dependencies {
            if !self.config.build_config.runtime_dependencies.contains(dep) {
                commands.push_str(&format!("        {} \\\n", dep));
            }
        }

        commands.push_str("    && apt-get autoremove -y && \\\n");
        commands.push_str("    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*\n\n");

        commands
    }

    /// Generate docker-compose.yml
    pub fn generate_docker_compose(&self, service_name: &str) -> TrustformersResult<String> {
        let compose = format!(
            r#"
version: '3.8'

services:
  {}:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        RUST_LOG: info
    image: trustformers-c:latest
    container_name: trustformers-c-container
    restart: unless-stopped
    ports:
{}
    environment:
      - RUST_LOG=info
      - TRUSTFORMERS_CACHE_DIR=/tmp/models
    volumes:
      - ./models:/tmp/models:ro
      - ./data:/app/data
    security_opt:
{}
    cap_drop:
{}
    cap_add:
{}
    read_only: {}
    user: "{}:{}"
    mem_limit: {}
    cpus: {}
    healthcheck:
{}
    networks:
      - trustformers-network

networks:
  trustformers-network:
    driver: bridge

volumes:
  trustformers-data:
    driver: local
"#,
            service_name,
            self.config
                .runtime_config
                .ports
                .iter()
                .map(|p| format!("      - \"{}:{}\"", p, p))
                .collect::<Vec<_>>()
                .join("\n"),
            self.config
                .security_config
                .security_opts
                .iter()
                .map(|opt| format!("      - {}", opt))
                .collect::<Vec<_>>()
                .join("\n"),
            self.config
                .security_config
                .drop_capabilities
                .iter()
                .map(|cap| format!("      - {}", cap))
                .collect::<Vec<_>>()
                .join("\n"),
            self.config
                .security_config
                .add_capabilities
                .iter()
                .map(|cap| format!("      - {}", cap))
                .collect::<Vec<_>>()
                .join("\n"),
            self.config.security_config.read_only_root,
            self.config.runtime_config.user.uid,
            self.config.runtime_config.user.gid,
            self.config.runtime_config.resources.memory.as_deref().unwrap_or("512m"),
            self.config.runtime_config.resources.cpu.as_deref().unwrap_or("0.5"),
            if let Some(health_check) = &self.config.health_check {
                format!(
                    "      test: [\"CMD\", {}]\n      interval: {}\n      timeout: {}\n      retries: {}",
                    health_check.command.iter()
                        .map(|s| format!("\"{}\"", s))
                        .collect::<Vec<_>>()
                        .join(", "),
                    health_check.interval,
                    health_check.timeout,
                    health_check.retries
                )
            } else {
                "      test: [\"CMD\", \"curl\", \"-f\", \"http://localhost:8080/health\"]\n      interval: 30s\n      timeout: 10s\n      retries: 3".to_string()
            }
        );

        Ok(compose)
    }

    /// Generate .dockerignore
    pub fn generate_dockerignore(&self) -> String {
        r#"
# Git
.git
.gitignore
.gitattributes

# Documentation
*.md
docs/
examples/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Build artifacts
target/
Cargo.lock
.cargo/

# Test files
tests/
test-data/

# CI/CD
.github/
.gitlab-ci.yml
.travis.yml
circle.yml

# Deployment files
docker-compose*.yml
kubernetes/
helm/

# Log files
*.log
logs/

# Temporary files
tmp/
temp/
.tmp/

# Environment files
.env
.env.local
.env.development
.env.test
.env.production

# Cache directories
node_modules/
.cache/
.npm/

# Package manager files
package-lock.json
yarn.lock
"#
        .to_string()
    }

    /// Generate build script
    pub fn generate_build_script(&self) -> String {
        format!(
            r#"#!/bin/bash
set -e

echo "Building TrustformeRS Docker image..."

# Configuration
IMAGE_NAME="trustformers-c"
IMAGE_TAG="latest"
PLATFORMS="{}"

# Build arguments
BUILD_ARGS=""
{}

# Build the image
if [ "${{PLATFORMS}}" = "multi" ]; then
    echo "Building multi-architecture image..."
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --tag ${{IMAGE_NAME}}:${{IMAGE_TAG}} \
        ${{BUILD_ARGS}} \
        --push \
        .
else
    echo "Building single-architecture image..."
    docker build \
        --tag ${{IMAGE_NAME}}:${{IMAGE_TAG}} \
        ${{BUILD_ARGS}} \
        .
fi

# Security scan
if command -v trivy &> /dev/null; then
    echo "Running security scan..."
    trivy image ${{IMAGE_NAME}}:${{IMAGE_TAG}}
fi

# Image analysis
echo "Image information:"
docker images ${{IMAGE_NAME}}:${{IMAGE_TAG}}
docker history ${{IMAGE_NAME}}:${{IMAGE_TAG}}

echo "Build completed successfully!"
"#,
            match &self.config.build_config.target_arch {
                TargetArchitecture::MultiArch(_) => "multi".to_string(),
                TargetArchitecture::AMD64 => "linux/amd64".to_string(),
                TargetArchitecture::ARM64 => "linux/arm64".to_string(),
                TargetArchitecture::ARMv7 => "linux/arm/v7".to_string(),
            },
            self.config
                .build_config
                .build_args
                .iter()
                .map(|(k, v)| format!("BUILD_ARGS=\"${{BUILD_ARGS}} --build-arg {}={}\"", k, v))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

/// Docker optimization utilities
pub struct DockerOptimizer;

impl DockerOptimizer {
    /// Analyze image layers and suggest optimizations
    pub fn analyze_image(image_name: &str) -> TrustformersResult<ImageAnalysis> {
        // This would use Docker API to analyze the image
        // For now, return a mock analysis
        Ok(ImageAnalysis {
            total_size: "256MB".to_string(),
            layer_count: 12,
            large_layers: vec![LayerInfo {
                id: "sha256:abc123".to_string(),
                size: "64MB".to_string(),
                command: "RUN apt-get install".to_string(),
            }],
            optimization_suggestions: vec![
                "Combine RUN commands to reduce layers".to_string(),
                "Use multi-stage build to reduce final image size".to_string(),
                "Remove package cache after installation".to_string(),
            ],
            security_issues: vec![
                "Running as root user".to_string(),
                "Outdated base image detected".to_string(),
            ],
        })
    }

    /// Generate optimization recommendations
    pub fn recommend_optimizations(config: &DockerImageConfig) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !config.build_config.multi_stage {
            recommendations.push("Enable multi-stage build to reduce final image size".to_string());
        }

        if !config.security_config.non_root {
            recommendations.push("Run as non-root user for better security".to_string());
        }

        if !config.optimization.strip_debug {
            recommendations.push("Strip debug symbols to reduce binary size".to_string());
        }

        if !config.optimization.clean_package_cache {
            recommendations.push("Clean package cache to reduce image size".to_string());
        }

        if config.health_check.is_none() {
            recommendations.push("Add health check for better container orchestration".to_string());
        }

        recommendations
    }
}

/// Image analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAnalysis {
    /// Total image size
    pub total_size: String,
    /// Number of layers
    pub layer_count: u32,
    /// Large layers (>10MB)
    pub large_layers: Vec<LayerInfo>,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<String>,
    /// Security issues
    pub security_issues: Vec<String>,
}

/// Layer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// Layer ID
    pub id: String,
    /// Layer size
    pub size: String,
    /// Command that created the layer
    pub command: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docker_config_creation() {
        let config = DockerImageConfig {
            base_image: BaseImage::Debian {
                version: "bullseye".to_string(),
                slim: true,
            },
            build_config: BuildConfig {
                multi_stage: true,
                target_arch: TargetArchitecture::AMD64,
                build_args: HashMap::new(),
                env_vars: HashMap::new(),
                workdir: "/app".to_string(),
                build_dependencies: vec!["build-essential".to_string()],
                runtime_dependencies: vec!["ca-certificates".to_string()],
            },
            runtime_config: RuntimeConfig {
                ports: vec![8080],
                volumes: vec![],
                resources: ResourceLimits {
                    memory: Some("512m".to_string()),
                    cpu: Some("0.5".to_string()),
                    swap: None,
                    pids: None,
                },
                user: UserConfig {
                    uid: 1000,
                    gid: 1000,
                    username: Some("trustformers".to_string()),
                    groupname: Some("trustformers".to_string()),
                },
                entrypoint: EntrypointConfig {
                    command: vec!["/usr/local/bin/trustformers-c".to_string()],
                    args: vec![],
                    signal_handling: true,
                },
            },
            security_config: SecurityConfig {
                non_root: true,
                read_only_root: true,
                security_opts: vec!["no-new-privileges:true".to_string()],
                drop_capabilities: vec!["ALL".to_string()],
                add_capabilities: vec![],
                security_profile: None,
            },
            optimization: OptimizationConfig {
                layer_caching: true,
                minimize_layers: true,
                strip_debug: true,
                compress_binary: false,
                clean_package_cache: true,
                remove_dev_tools: true,
                enable_scanning: true,
            },
            health_check: Some(HealthCheckConfig {
                command: vec![
                    "curl".to_string(),
                    "-f".to_string(),
                    "http://localhost:8080/health".to_string(),
                ],
                interval: "30s".to_string(),
                timeout: "10s".to_string(),
                start_period: Some("5s".to_string()),
                retries: 3,
            }),
        };

        let builder = DockerImageBuilder::new(config);
        let dockerfile = builder.generate_dockerfile().unwrap();

        assert!(dockerfile.contains("FROM debian:bullseye-slim"));
        assert!(dockerfile.contains("USER trustformers"));
        assert!(dockerfile.contains("HEALTHCHECK"));
    }
}
