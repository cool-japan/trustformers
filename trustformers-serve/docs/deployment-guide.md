# TrustformeRS Serve Deployment Guide

This guide provides comprehensive instructions for deploying TrustformeRS Serve in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployments](#cloud-deployments)
6. [Serverless Deployment](#serverless-deployment)
7. [Edge Deployment](#edge-deployment)
8. [Multi-Cloud Deployment](#multi-cloud-deployment)
9. [Configuration Management](#configuration-management)
10. [Security Considerations](#security-considerations)

## Prerequisites

### System Requirements

- **CPU**: 8+ cores recommended
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ available space
- **GPU**: Optional, CUDA-compatible for GPU acceleration
- **Network**: High-bandwidth connection for model downloads

### Software Dependencies

- **Rust**: 1.70.0 or later
- **Docker**: 20.10.0 or later
- **Kubernetes**: 1.20.0 or later (for K8s deployment)
- **Terraform**: 1.0.0 or later (for infrastructure deployment)
- **Ansible**: 2.9.0 or later (for configuration management)

## Local Development

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/trustformers/trustformers-serve.git
   cd trustformers-serve
   ```

2. **Build the project**:
   ```bash
   cargo build --release
   ```

3. **Run the server**:
   ```bash
   ./target/release/trustformers-serve
   ```

### Development Configuration

Create a `config.toml` file:

```toml
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[batching]
max_batch_size = 32
max_wait_time = "50ms"
enable_adaptive_batching = true

[models]
model_path = "./models"
cache_size = 1000

[security]
enable_auth = false
api_key = "your-api-key"

[logging]
level = "info"
format = "json"
```

## Docker Deployment

### Basic Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t trustformers-serve .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8080:8080 \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/config.toml:/app/config.toml \
     trustformers-serve
   ```

### Docker Compose

Use the provided `docker-compose.yml`:

```yaml
version: '3.8'
services:
  trustformers-serve:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
      - ./config.toml:/app/config.toml
    environment:
      - RUST_LOG=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with:
```bash
docker-compose up -d
```

### Production Docker Deployment

For production, use the optimized Dockerfile:

```bash
docker build -f Dockerfile.super-optimized -t trustformers-serve:optimized .
```

## Kubernetes Deployment

### Basic Kubernetes Deployment

1. **Apply the manifests**:
   ```bash
   kubectl apply -f k8s/
   ```

2. **Check deployment status**:
   ```bash
   kubectl get pods -l app=trustformers-serve
   kubectl get svc trustformers-serve
   ```

### Helm Deployment

1. **Install with Helm**:
   ```bash
   helm install trustformers-serve ./helm/
   ```

2. **Upgrade configuration**:
   ```bash
   helm upgrade trustformers-serve ./helm/ --values values-production.yaml
   ```

### Production Kubernetes Configuration

Key considerations for production:

- **Resource Limits**: Set appropriate CPU and memory limits
- **Horizontal Pod Autoscaler**: Enable HPA for auto-scaling
- **Persistent Volumes**: Use PVs for model storage
- **Service Mesh**: Consider Istio for advanced traffic management
- **Monitoring**: Deploy Prometheus and Grafana for monitoring

Example production deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trustformers-serve
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trustformers-serve
  template:
    metadata:
      labels:
        app: trustformers-serve
    spec:
      containers:
      - name: trustformers-serve
        image: trustformers-serve:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 2
            memory: 4Gi
          limits:
            cpu: 4
            memory: 8Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Cloud Deployments

### AWS EKS Deployment

1. **Create EKS cluster using Terraform**:
   ```bash
   cd terraform/modules/aws-eks-complete
   terraform init
   terraform plan
   terraform apply
   ```

2. **Deploy to EKS**:
   ```bash
   kubectl apply -f k8s/production-deployment.yaml
   ```

### Google Cloud GKE Deployment

1. **Create GKE cluster**:
   ```bash
   cd terraform/modules/multi-cloud-deploy/gcp-gke
   terraform init
   terraform apply
   ```

2. **Deploy application**:
   ```bash
   kubectl apply -f k8s/
   ```

### Azure AKS Deployment

1. **Create AKS cluster**:
   ```bash
   cd terraform/modules/multi-cloud-deploy/azure-aks
   terraform init
   terraform apply
   ```

2. **Configure kubectl**:
   ```bash
   az aks get-credentials --resource-group rg-trustformers --name aks-trustformers
   ```

3. **Deploy application**:
   ```bash
   kubectl apply -f k8s/
   ```

## Serverless Deployment

### AWS Lambda Deployment

1. **Build serverless package**:
   ```bash
   cargo build --release --target x86_64-unknown-linux-musl
   ```

2. **Deploy with Terraform**:
   ```bash
   cd terraform/modules/trustformers-serve
   terraform init
   terraform apply -var="deployment_type=serverless"
   ```

### Google Cloud Functions

1. **Package for Cloud Functions**:
   ```bash
   # Create deployment package
   zip -r function.zip target/lambda/
   ```

2. **Deploy function**:
   ```bash
   gcloud functions deploy trustformers-serve \
     --runtime rust \
     --trigger-http \
     --allow-unauthenticated \
     --source .
   ```

### Azure Functions

1. **Create function app**:
   ```bash
   az functionapp create \
     --resource-group rg-trustformers \
     --consumption-plan-location westus2 \
     --name trustformers-func \
     --runtime rust
   ```

2. **Deploy function**:
   ```bash
   func azure functionapp publish trustformers-func
   ```

## Edge Deployment

### Edge Node Setup

1. **Install on edge device**:
   ```bash
   # For ARM64 devices
   cargo build --release --target aarch64-unknown-linux-gnu
   ```

2. **Configure for edge**:
   ```toml
   [edge]
   enabled = true
   sync_interval = "300s"
   offline_mode = true
   bandwidth_limit = "1MB/s"
   
   [models]
   local_cache_size = 500
   preload_models = ["small-model-v1"]
   ```

3. **Deploy with Ansible**:
   ```bash
   cd ansible
   ansible-playbook -i inventory/production.yml playbooks/deploy-trustformers-serve.yml
   ```

### Edge Orchestration

For managing multiple edge nodes:

1. **Setup edge orchestrator**:
   ```bash
   # Configure central orchestrator
   trustformers-serve --mode edge-orchestrator
   ```

2. **Register edge nodes**:
   ```bash
   # On each edge node
   trustformers-serve --mode edge-node --orchestrator-url http://central:8080
   ```

## Multi-Cloud Deployment

### Global Load Balancer Setup

1. **Deploy infrastructure**:
   ```bash
   cd terraform/modules/multi-cloud-deploy
   terraform init
   terraform apply
   ```

2. **Configure DNS**:
   ```bash
   # Set up global DNS routing
   terraform apply -var="enable_global_dns=true"
   ```

### Cross-Cloud Networking

Configure secure connections between clouds:

```hcl
# terraform/modules/multi-cloud-deploy/main.tf
resource "google_compute_vpn_gateway" "aws_to_gcp" {
  name    = "aws-to-gcp-vpn"
  network = google_compute_network.main.self_link
}

resource "aws_vpn_gateway" "gcp_to_aws" {
  vpc_id = aws_vpc.main.id
  tags = {
    Name = "gcp-to-aws-vpn"
  }
}
```

## Configuration Management

### Environment-Specific Configuration

Create configuration files for each environment:

**Development** (`config/dev.toml`):
```toml
[server]
host = "127.0.0.1"
port = 8080
debug = true

[logging]
level = "debug"
```

**Production** (`config/prod.toml`):
```toml
[server]
host = "0.0.0.0"
port = 8080
workers = 16

[security]
enable_auth = true
enable_tls = true
cert_path = "/etc/ssl/certs/server.pem"
key_path = "/etc/ssl/private/server.key"

[logging]
level = "info"
format = "json"
```

### Configuration Validation

Validate configuration before deployment:

```bash
# Validate configuration
trustformers-serve --config config/prod.toml --validate

# Test configuration
trustformers-serve --config config/prod.toml --test
```

### Secret Management

Use appropriate secret management for each environment:

- **Development**: Local environment variables
- **Kubernetes**: Kubernetes Secrets
- **AWS**: AWS Secrets Manager
- **GCP**: Google Secret Manager
- **Azure**: Azure Key Vault

Example Kubernetes secret:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: trustformers-secrets
type: Opaque
data:
  api-key: <base64-encoded-api-key>
  jwt-secret: <base64-encoded-jwt-secret>
```

## Security Considerations

### Network Security

1. **TLS/SSL Configuration**:
   ```toml
   [security]
   enable_tls = true
   cert_path = "/etc/ssl/certs/server.pem"
   key_path = "/etc/ssl/private/server.key"
   min_tls_version = "1.2"
   ```

2. **Firewall Rules**:
   ```bash
   # Allow only necessary ports
   ufw allow 8080/tcp
   ufw allow 443/tcp
   ufw enable
   ```

### Authentication and Authorization

1. **Enable JWT Authentication**:
   ```toml
   [auth]
   enable_jwt = true
   jwt_secret = "your-secret-key"
   jwt_expiration = "24h"
   ```

2. **API Key Management**:
   ```bash
   # Generate API key
   trustformers-serve --generate-api-key
   ```

### Data Protection

1. **Enable encryption at rest**:
   ```toml
   [encryption]
   enable_encryption = true
   encryption_key = "your-encryption-key"
   algorithm = "AES-256-GCM"
   ```

2. **GDPR Compliance**:
   ```toml
   [gdpr]
   enable_gdpr_compliance = true
   data_retention_days = 90
   enable_right_to_deletion = true
   ```

### Monitoring and Auditing

1. **Enable audit logging**:
   ```toml
   [audit]
   enable_audit_log = true
   audit_log_path = "/var/log/trustformers/audit.log"
   log_level = "info"
   ```

2. **Set up monitoring**:
   ```bash
   # Deploy monitoring stack
   kubectl apply -f k8s/monitoring.yaml
   ```

## Health Checks and Monitoring

### Health Check Endpoints

- `/health` - Basic health check
- `/health/ready` - Readiness probe
- `/health/live` - Liveness probe
- `/health/detailed` - Detailed health information

### Monitoring Setup

1. **Prometheus Configuration**:
   ```yaml
   scrape_configs:
     - job_name: 'trustformers-serve'
       static_configs:
         - targets: ['localhost:8080']
       metrics_path: '/metrics'
   ```

2. **Grafana Dashboard**:
   Import the provided dashboard from `static/dashboard.html`

### Alerting Rules

```yaml
groups:
- name: trustformers-serve
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
```

## Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Find process using port
   netstat -tulpn | grep :8080
   # Kill process
   kill -9 <pid>
   ```

2. **Model loading failures**:
   ```bash
   # Check model path and permissions
   ls -la /path/to/models/
   # Verify model format
   trustformers-serve --validate-model /path/to/model.onnx
   ```

3. **Memory issues**:
   ```bash
   # Monitor memory usage
   htop
   # Check for memory leaks
   valgrind --tool=memcheck ./target/release/trustformers-serve
   ```

### Log Analysis

1. **View logs**:
   ```bash
   # Docker logs
   docker logs trustformers-serve
   
   # Kubernetes logs
   kubectl logs -l app=trustformers-serve
   
   # System logs
   journalctl -u trustformers-serve
   ```

2. **Log aggregation**:
   ```bash
   # Send logs to ELK stack
   filebeat -e -c filebeat.yml
   ```

## Migration

### Version Migration

Use the migration tool for version upgrades:

```bash
# Check migration info
trustformers-migrate info --from 1.0.0 --to 2.0.0

# Perform migration
trustformers-migrate version --from 1.0.0 --to 2.0.0 --path /opt/trustformers

# Validate migration
trustformers-migrate validate --path /opt/trustformers --version 2.0.0
```

### Data Migration

```bash
# Migrate data files
trustformers-migrate data --from 1.0.0 --to 2.0.0 --data-path /opt/trustformers/data

# Migrate models
trustformers-migrate models --from 1.0.0 --to 2.0.0 --models-path /opt/trustformers/models --optimize
```

## Support

For deployment issues:

1. Check the [troubleshooting guide](troubleshooting-guide.md)
2. Review the [best practices](best-practices.md)
3. Search existing issues on GitHub
4. Create a new issue with deployment details

## Next Steps

After successful deployment:

1. Configure monitoring and alerting
2. Set up backup and disaster recovery
3. Implement CI/CD pipeline
4. Scale according to load requirements
5. Review and update security configurations