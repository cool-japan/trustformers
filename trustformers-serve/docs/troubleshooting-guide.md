# TrustformeRS Serve Troubleshooting Guide

This guide provides solutions to common issues encountered when deploying and operating TrustformeRS Serve.

## Table of Contents

1. [General Troubleshooting](#general-troubleshooting)
2. [Installation and Setup Issues](#installation-and-setup-issues)
3. [Performance Issues](#performance-issues)
4. [Memory and Resource Problems](#memory-and-resource-problems)
5. [Network and Connectivity Issues](#network-and-connectivity-issues)
6. [Authentication and Authorization Issues](#authentication-and-authorization-issues)
7. [Model Loading and Inference Issues](#model-loading-and-inference-issues)
8. [Kubernetes-Specific Issues](#kubernetes-specific-issues)
9. [Database and Caching Issues](#database-and-caching-issues)
10. [Monitoring and Observability Issues](#monitoring-and-observability-issues)
11. [Disaster Recovery and Emergency Procedures](#disaster-recovery-and-emergency-procedures)
12. [Support and Escalation](#support-and-escalation)

## General Troubleshooting

### Diagnostic Commands

**Check Server Status**:
```bash
# Check if server is running
curl -f http://localhost:8080/health

# Detailed health check
curl -f http://localhost:8080/health/detailed

# Check readiness
curl -f http://localhost:8080/health/ready

# Check liveness
curl -f http://localhost:8080/health/live
```

**View Logs**:
```bash
# Docker logs
docker logs trustformers-serve --tail 100 -f

# Kubernetes logs
kubectl logs -l app=trustformers-serve --tail 100 -f

# System logs
journalctl -u trustformers-serve --tail 100 -f

# Application logs
tail -f /var/log/trustformers/server.log
```

**Check Resource Usage**:
```bash
# CPU and memory usage
top -p $(pgrep trustformers-serve)

# Detailed process information
ps aux | grep trustformers-serve

# Memory usage
free -h

# Disk usage
df -h
lsof | grep trustformers-serve
```

### Configuration Validation

**Validate Configuration**:
```bash
# Test configuration syntax
trustformers-serve --config config.toml --validate

# Test configuration with dry run
trustformers-serve --config config.toml --dry-run

# Check configuration values
trustformers-serve --config config.toml --print-config
```

## Installation and Setup Issues

### Common Build Issues

**Issue**: Compilation errors with dependencies
```bash
error: failed to compile `trustformers-serve v1.0.0`
```

**Solution**:
```bash
# Update Rust toolchain
rustup update

# Clean and rebuild
cargo clean
cargo build --release

# Check for conflicting dependencies
cargo tree --duplicates

# Update dependencies
cargo update
```

**Issue**: Missing system dependencies
```bash
error: linking with `cc` failed: exit status: 1
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# CentOS/RHEL
sudo yum install gcc openssl-devel

# macOS
xcode-select --install
```

### Docker Issues

**Issue**: Docker build fails
```bash
ERROR: failed to solve: process "/bin/sh -c cargo build --release" did not complete successfully
```

**Solution**:
```bash
# Build with verbose output
docker build --no-cache --progress=plain -t trustformers-serve .

# Check disk space
docker system df
docker system prune -a

# Use multi-stage build
docker build -f Dockerfile.optimized -t trustformers-serve .
```

**Issue**: Container exits immediately
```bash
docker run trustformers-serve
# Container exits with code 1
```

**Solution**:
```bash
# Check container logs
docker logs <container_id>

# Run interactively
docker run -it trustformers-serve /bin/bash

# Check configuration
docker run -v $(pwd)/config.toml:/app/config.toml trustformers-serve --validate
```

### Kubernetes Deployment Issues

**Issue**: Pods stuck in Pending state
```bash
kubectl get pods
NAME                     READY   STATUS    RESTARTS   AGE
trustformers-serve-xxx   0/1     Pending   0          5m
```

**Solution**:
```bash
# Check pod events
kubectl describe pod trustformers-serve-xxx

# Check resource availability
kubectl top nodes
kubectl describe nodes

# Check PVC status
kubectl get pvc
```

**Issue**: Image pull errors
```bash
Failed to pull image "trustformers-serve:latest": rpc error: code = Unknown desc = Error response from daemon: pull access denied
```

**Solution**:
```bash
# Check image exists
docker images | grep trustformers-serve

# Create image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=<registry-url> \
  --docker-username=<username> \
  --docker-password=<password>

# Update deployment to use secret
kubectl patch deployment trustformers-serve \
  -p '{"spec":{"template":{"spec":{"imagePullSecrets":[{"name":"regcred"}]}}}}'
```

## Performance Issues

### High Latency

**Issue**: Response times are consistently high (>500ms)

**Symptoms**:
- High P95/P99 latency metrics
- Slow response times
- Client timeouts

**Diagnosis**:
```bash
# Check request profiling
curl -X POST http://localhost:8080/v1/inference \
  -H "Content-Type: application/json" \
  -H "X-Enable-Profiling: true" \
  -d '{"model_id": "test", "input": "test input"}'

# Check batching metrics
curl http://localhost:8080/metrics | grep batch

# Monitor GPU utilization
nvidia-smi -l 1
```

**Solutions**:
1. **Optimize Batching**:
   ```toml
   [batching]
   max_batch_size = 32
   max_wait_time = "10ms"
   enable_adaptive_batching = true
   ```

2. **Enable Caching**:
   ```toml
   [caching]
   enable_result_cache = true
   cache_size = 1000
   ttl = "1h"
   ```

3. **GPU Optimization**:
   ```toml
   [gpu]
   enable_mixed_precision = true
   enable_tensor_rt = true
   memory_fraction = 0.9
   ```

### High CPU Usage

**Issue**: CPU usage consistently above 80%

**Symptoms**:
- High CPU utilization
- Slow request processing
- System responsiveness issues

**Diagnosis**:
```bash
# Check CPU usage by process
top -p $(pgrep trustformers-serve)

# Profile CPU usage
perf top -p $(pgrep trustformers-serve)

# Check for CPU-intensive operations
strace -p $(pgrep trustformers-serve) -f -e trace=cpu
```

**Solutions**:
1. **Optimize Worker Configuration**:
   ```toml
   [server]
   workers = 8  # Adjust based on CPU cores
   max_connections = 1000
   ```

2. **Enable Async Processing**:
   ```toml
   [async]
   enable_async_inference = true
   async_queue_size = 1000
   ```

3. **Scale Horizontally**:
   ```bash
   # Kubernetes scaling
   kubectl scale deployment trustformers-serve --replicas=5
   ```

### GPU Performance Issues

**Issue**: GPU utilization is low despite high load

**Symptoms**:
- Low GPU utilization (<50%)
- High latency despite available GPU resources
- GPU memory underutilization

**Diagnosis**:
```bash
# Monitor GPU metrics
nvidia-smi dmon -s pucvmet -d 1

# Check GPU memory usage
nvidia-smi -q -d MEMORY

# Profile GPU operations
nvprof ./trustformers-serve
```

**Solutions**:
1. **Optimize Batch Size**:
   ```toml
   [batching]
   max_batch_size = 64  # Increase for better GPU utilization
   enable_sequence_bucketing = true
   ```

2. **Enable GPU Scheduling**:
   ```toml
   [gpu]
   enable_gpu_scheduling = true
   scheduling_algorithm = "load_balanced"
   ```

3. **Multi-GPU Setup**:
   ```toml
   [gpu]
   num_gpus = 4
   enable_multi_gpu = true
   distribution_strategy = "model_parallel"
   ```

## Memory and Resource Problems

### Out of Memory Errors

**Issue**: Server crashes with OOM errors

**Symptoms**:
- Process killed by OOM killer
- Memory usage continuously increasing
- Allocation failures

**Diagnosis**:
```bash
# Check memory usage
free -h
cat /proc/meminfo

# Monitor memory usage over time
watch -n 1 'free -h'

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full ./trustformers-serve

# Check OOM killer logs
dmesg | grep -i "killed process"
```

**Solutions**:
1. **Configure Memory Limits**:
   ```toml
   [server]
   max_memory = "8GB"
   
   [batching]
   memory_limit = "4GB"
   enable_memory_tracking = true
   ```

2. **Enable Memory Pressure Handling**:
   ```toml
   [memory]
   enable_memory_pressure_handling = true
   warning_threshold = 0.8
   critical_threshold = 0.95
   ```

3. **Optimize Caching**:
   ```toml
   [caching]
   cache_size = 500  # Reduce cache size
   eviction_policy = "lru"
   ```

### Memory Leaks

**Issue**: Memory usage continuously increases over time

**Symptoms**:
- Gradual memory increase
- Performance degradation over time
- Eventually leads to OOM

**Diagnosis**:
```bash
# Monitor memory usage
ps -o pid,ppid,rss,vsz,pmem,pcpu,cmd -p $(pgrep trustformers-serve)

# Use memory profiler
heaptrack ./trustformers-serve
heaptrack_gui heaptrack.trustformers-serve.*.gz

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full ./trustformers-serve
```

**Solutions**:
1. **Enable Memory Monitoring**:
   ```rust
   let memory_monitor = MemoryMonitor::new();
   memory_monitor.start_monitoring(Duration::from_secs(30));
   ```

2. **Implement Memory Cleanup**:
   ```rust
   // Periodic cleanup
   tokio::spawn(async move {
       let mut interval = tokio::time::interval(Duration::from_secs(300));
       loop {
           interval.tick().await;
           cleanup_unused_resources().await;
       }
   });
   ```

### Disk Space Issues

**Issue**: Disk space running out

**Symptoms**:
- Disk usage above 90%
- Cannot write logs or cache files
- Performance degradation

**Diagnosis**:
```bash
# Check disk usage
df -h

# Find large files
find /var/log -type f -size +100M -exec ls -lh {} \;

# Check log sizes
du -sh /var/log/trustformers/*

# Monitor disk usage
watch -n 5 'df -h'
```

**Solutions**:
1. **Configure Log Rotation**:
   ```bash
   # /etc/logrotate.d/trustformers-serve
   /var/log/trustformers/*.log {
       daily
       rotate 7
       compress
       missingok
       notifempty
       create 0644 trustformers trustformers
   }
   ```

2. **Clean Up Old Files**:
   ```bash
   # Remove old cache files
   find /var/cache/trustformers -name "*.cache" -mtime +7 -delete

   # Clean up temporary files
   find /tmp -name "trustformers*" -mtime +1 -delete
   ```

## Network and Connectivity Issues

### DNS Resolution Problems

**Issue**: Cannot resolve hostnames

**Symptoms**:
- DNS resolution failures
- Connection timeouts to external services
- Service discovery issues

**Diagnosis**:
```bash
# Test DNS resolution
nslookup api.example.com
dig api.example.com

# Check DNS configuration
cat /etc/resolv.conf

# Test connectivity
ping api.example.com
telnet api.example.com 443
```

**Solutions**:
1. **Configure DNS Servers**:
   ```bash
   # Update /etc/resolv.conf
   nameserver 8.8.8.8
   nameserver 8.8.4.4
   ```

2. **Use IP Addresses**:
   ```toml
   [external_services]
   model_registry_url = "https://192.168.1.100:8080"
   ```

3. **Configure Kubernetes DNS**:
   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: coredns
     namespace: kube-system
   data:
     Corefile: |
       .:53 {
           errors
           health
           ready
           kubernetes cluster.local in-addr.arpa ip6.arpa {
               pods insecure
               upstream
           }
           prometheus :9153
           forward . 8.8.8.8 8.8.4.4
           cache 30
           reload
       }
   ```

### Load Balancer Issues

**Issue**: Requests not distributed evenly

**Symptoms**:
- Uneven load distribution
- Some instances overloaded
- Health check failures

**Diagnosis**:
```bash
# Check load balancer status
curl -s http://load-balancer/status

# Monitor backend health
for i in {1..5}; do
  curl -s http://backend-$i:8080/health
done

# Check connection distribution
netstat -an | grep :8080 | wc -l
```

**Solutions**:
1. **Configure Load Balancer**:
   ```nginx
   upstream backend {
       least_conn;
       server backend1:8080 max_fails=3 fail_timeout=30s;
       server backend2:8080 max_fails=3 fail_timeout=30s;
       server backend3:8080 max_fails=3 fail_timeout=30s;
   }
   ```

2. **Implement Health Checks**:
   ```rust
   let health_check = HealthCheck::new()
       .with_path("/health")
       .with_interval(Duration::from_secs(30))
       .with_timeout(Duration::from_secs(5));
   ```

### Certificate Issues

**Issue**: SSL/TLS certificate problems

**Symptoms**:
- Certificate validation errors
- Handshake failures
- Browser warnings

**Diagnosis**:
```bash
# Check certificate
openssl s_client -connect localhost:8080 -servername localhost

# Verify certificate
openssl x509 -in certificate.pem -text -noout

# Check certificate expiration
openssl x509 -in certificate.pem -noout -dates
```

**Solutions**:
1. **Renew Certificate**:
   ```bash
   # Using Let's Encrypt
   certbot renew --dry-run
   certbot renew
   ```

2. **Configure Certificate**:
   ```toml
   [tls]
   cert_file = "/etc/ssl/certs/server.pem"
   key_file = "/etc/ssl/private/server.key"
   ca_file = "/etc/ssl/certs/ca.pem"
   ```

## Authentication and Authorization Issues

### JWT Token Problems

**Issue**: JWT authentication failures

**Symptoms**:
- 401 Unauthorized errors
- Token validation failures
- Authentication timeouts

**Diagnosis**:
```bash
# Decode JWT token
echo "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." | base64 -d

# Test token validation
curl -H "Authorization: Bearer <token>" http://localhost:8080/v1/inference

# Check token expiration
jwt-cli decode <token>
```

**Solutions**:
1. **Check Token Configuration**:
   ```toml
   [auth]
   jwt_secret = "your-secret-key"
   jwt_expiration = "24h"
   jwt_issuer = "trustformers-serve"
   jwt_audience = "api-users"
   ```

2. **Implement Token Refresh**:
   ```rust
   async fn refresh_token(token: &str) -> Result<String> {
       if is_token_near_expiry(token) {
           return generate_new_token().await;
       }
       Ok(token.to_string())
   }
   ```

### API Key Issues

**Issue**: API key authentication failures

**Symptoms**:
- API key validation errors
- Rate limiting issues
- Access denied errors

**Diagnosis**:
```bash
# Test API key
curl -H "X-API-Key: your-api-key" http://localhost:8080/v1/inference

# Check API key status
curl -H "X-API-Key: your-api-key" http://localhost:8080/v1/api-keys/status

# Verify key permissions
curl -H "X-API-Key: your-api-key" http://localhost:8080/v1/api-keys/permissions
```

**Solutions**:
1. **Regenerate API Key**:
   ```bash
   trustformers-serve --generate-api-key
   ```

2. **Configure API Key**:
   ```toml
   [api_keys]
   default_rate_limit = 1000
   enable_key_rotation = true
   key_expiration = "90d"
   ```

### OAuth2 Issues

**Issue**: OAuth2 authentication problems

**Symptoms**:
- OAuth2 flow failures
- Token exchange errors
- Scope validation issues

**Diagnosis**:
```bash
# Test OAuth2 flow
curl -X POST http://localhost:8080/auth/oauth2/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=test&client_secret=secret"

# Check token info
curl -H "Authorization: Bearer <token>" http://localhost:8080/auth/oauth2/token-info
```

**Solutions**:
1. **Configure OAuth2 Provider**:
   ```toml
   [oauth2]
   client_id = "your-client-id"
   client_secret = "your-client-secret"
   authorization_url = "https://auth.example.com/oauth2/authorize"
   token_url = "https://auth.example.com/oauth2/token"
   ```

2. **Update Scopes**:
   ```toml
   [oauth2]
   default_scopes = ["read", "write"]
   scope_mapping = { "inference" = "read", "admin" = "write" }
   ```

## Model Loading and Inference Issues

### Model Loading Failures

**Issue**: Models fail to load

**Symptoms**:
- Model loading errors
- Missing model files
- Format compatibility issues

**Diagnosis**:
```bash
# Check model file
ls -la /path/to/models/

# Validate model format
trustformers-serve --validate-model /path/to/model.onnx

# Check model metadata
trustformers-serve --model-info /path/to/model.onnx

# Test model loading
trustformers-serve --test-model /path/to/model.onnx
```

**Solutions**:
1. **Verify Model Path**:
   ```toml
   [models]
   model_path = "/opt/models"
   model_cache_path = "/tmp/model_cache"
   ```

2. **Check Model Format**:
   ```bash
   # Convert model format
   python convert_model.py --input model.pt --output model.onnx

   # Validate converted model
   onnx-checker model.onnx
   ```

3. **Update Model Registry**:
   ```bash
   # Register model
   trustformers-serve --register-model /path/to/model.onnx --name "my-model"

   # List registered models
   trustformers-serve --list-models
   ```

### Inference Errors

**Issue**: Inference requests fail

**Symptoms**:
- Inference failures
- Unexpected outputs
- Processing errors

**Diagnosis**:
```bash
# Test inference
curl -X POST http://localhost:8080/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"model_id": "test", "input": "test input"}'

# Check model status
curl http://localhost:8080/v1/models/test/status

# Enable debug logging
export RUST_LOG=debug
trustformers-serve
```

**Solutions**:
1. **Validate Input Format**:
   ```rust
   let validation_config = ValidationConfig {
       max_text_length: 1000,
       allowed_formats: vec!["text", "json"],
       enable_preprocessing: true,
   };
   ```

2. **Check Model Compatibility**:
   ```bash
   # Test with different input
   curl -X POST http://localhost:8080/v1/inference \
     -H "Content-Type: application/json" \
     -d '{"model_id": "test", "input": "simple test"}'
   ```

3. **Enable Model Debugging**:
   ```toml
   [models]
   enable_debug_mode = true
   log_predictions = true
   save_intermediate_results = true
   ```

### Model Version Issues

**Issue**: Model version conflicts

**Symptoms**:
- Version mismatch errors
- Incompatible model formats
- Deployment failures

**Diagnosis**:
```bash
# Check model versions
trustformers-serve --list-models --verbose

# Check version compatibility
trustformers-serve --check-compatibility --model model.onnx --version 2.0.0

# View model metadata
trustformers-serve --model-info model.onnx
```

**Solutions**:
1. **Model Migration**:
   ```bash
   # Migrate model
   trustformers-migrate models --from 1.0.0 --to 2.0.0 --models-path /opt/models
   ```

2. **Version Management**:
   ```toml
   [models]
   enable_versioning = true
   version_strategy = "semantic"
   auto_migration = true
   ```

## Kubernetes-Specific Issues

### Pod Crash Loops

**Issue**: Pods continuously crash and restart

**Symptoms**:
- CrashLoopBackOff status
- High restart count
- Application startup failures

**Diagnosis**:
```bash
# Check pod status
kubectl get pods -l app=trustformers-serve

# View pod logs
kubectl logs trustformers-serve-xxx --previous

# Describe pod events
kubectl describe pod trustformers-serve-xxx

# Check resource constraints
kubectl top pods
```

**Solutions**:
1. **Increase Resource Limits**:
   ```yaml
   resources:
     requests:
       cpu: 1
       memory: 2Gi
     limits:
       cpu: 2
       memory: 4Gi
   ```

2. **Fix Health Checks**:
   ```yaml
   livenessProbe:
     httpGet:
       path: /health
       port: 8080
     initialDelaySeconds: 60
     periodSeconds: 10
     timeoutSeconds: 5
     failureThreshold: 3
   ```

3. **Check Init Containers**:
   ```yaml
   initContainers:
   - name: model-downloader
     image: busybox
     command: ['sh', '-c', 'wget -O /models/model.onnx http://model-repo/model.onnx']
   ```

### Service Discovery Issues

**Issue**: Services cannot find each other

**Symptoms**:
- Service connection failures
- DNS resolution issues
- Network policies blocking traffic

**Diagnosis**:
```bash
# Check service endpoints
kubectl get endpoints

# Test service connectivity
kubectl exec -it test-pod -- curl http://trustformers-serve:8080/health

# Check network policies
kubectl get networkpolicies
```

**Solutions**:
1. **Configure Service Discovery**:
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: trustformers-serve
   spec:
     selector:
       app: trustformers-serve
     ports:
     - port: 8080
       targetPort: 8080
   ```

2. **Update Network Policies**:
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: allow-trustformers
   spec:
     podSelector:
       matchLabels:
         app: trustformers-serve
     policyTypes:
     - Ingress
     - Egress
     ingress:
     - from: []
       ports:
       - protocol: TCP
         port: 8080
   ```

### Persistent Volume Issues

**Issue**: Persistent volume problems

**Symptoms**:
- Volume mount failures
- Data persistence issues
- Storage class problems

**Diagnosis**:
```bash
# Check PVC status
kubectl get pvc

# Check storage class
kubectl get storageclass

# Check volume mounts
kubectl describe pod trustformers-serve-xxx
```

**Solutions**:
1. **Configure PVC**:
   ```yaml
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: model-storage
   spec:
     accessModes:
     - ReadWriteOnce
     resources:
       requests:
         storage: 100Gi
     storageClassName: ssd
   ```

2. **Check Storage Class**:
   ```yaml
   apiVersion: storage.k8s.io/v1
   kind: StorageClass
   metadata:
     name: ssd
   provisioner: kubernetes.io/aws-ebs
   parameters:
     type: gp3
     fsType: ext4
   ```

## Database and Caching Issues

### Redis Connection Issues

**Issue**: Cannot connect to Redis

**Symptoms**:
- Redis connection failures
- Caching not working
- Performance degradation

**Diagnosis**:
```bash
# Test Redis connection
redis-cli ping

# Check Redis status
redis-cli info

# Test from application
telnet redis-server 6379
```

**Solutions**:
1. **Configure Redis Connection**:
   ```toml
   [redis]
   url = "redis://localhost:6379"
   max_connections = 100
   connection_timeout = "5s"
   ```

2. **Check Redis Configuration**:
   ```bash
   # Check Redis config
   redis-cli config get "*"

   # Set memory policy
   redis-cli config set maxmemory-policy allkeys-lru
   ```

### Database Connection Pool Issues

**Issue**: Database connection problems

**Symptoms**:
- Connection pool exhaustion
- Database timeouts
- Performance issues

**Diagnosis**:
```bash
# Check database connections
psql -h localhost -U user -d database -c "SELECT * FROM pg_stat_activity;"

# Monitor connection pool
curl http://localhost:8080/metrics | grep db_connections
```

**Solutions**:
1. **Configure Connection Pool**:
   ```toml
   [database]
   max_connections = 100
   min_connections = 10
   connection_timeout = "30s"
   idle_timeout = "600s"
   ```

2. **Optimize Queries**:
   ```sql
   -- Add indexes
   CREATE INDEX idx_model_id ON inferences(model_id);
   
   -- Analyze query performance
   EXPLAIN ANALYZE SELECT * FROM inferences WHERE model_id = 'test';
   ```

## Monitoring and Observability Issues

### Prometheus Scraping Issues

**Issue**: Prometheus not scraping metrics

**Symptoms**:
- Missing metrics in Prometheus
- Scraping failures
- Monitoring dashboards empty

**Diagnosis**:
```bash
# Check metrics endpoint
curl http://localhost:8080/metrics

# Check Prometheus targets
curl http://prometheus:9090/api/v1/targets

# Check scraping config
kubectl get configmap prometheus-config -o yaml
```

**Solutions**:
1. **Configure Prometheus Scraping**:
   ```yaml
   scrape_configs:
   - job_name: 'trustformers-serve'
     static_configs:
     - targets: ['localhost:8080']
     metrics_path: '/metrics'
     scrape_interval: 15s
   ```

2. **Check Service Monitor**:
   ```yaml
   apiVersion: monitoring.coreos.com/v1
   kind: ServiceMonitor
   metadata:
     name: trustformers-serve
   spec:
     selector:
       matchLabels:
         app: trustformers-serve
     endpoints:
     - port: http
       path: /metrics
   ```

### Grafana Dashboard Issues

**Issue**: Grafana dashboards not showing data

**Symptoms**:
- Empty dashboard panels
- No data points
- Query errors

**Diagnosis**:
```bash
# Check Grafana datasource
curl -u admin:admin http://grafana:3000/api/datasources

# Test Prometheus query
curl -G 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=up{job="trustformers-serve"}'
```

**Solutions**:
1. **Configure Datasource**:
   ```json
   {
     "name": "Prometheus",
     "type": "prometheus",
     "url": "http://prometheus:9090",
     "access": "proxy",
     "isDefault": true
   }
   ```

2. **Import Dashboard**:
   ```bash
   # Import dashboard
   curl -X POST http://admin:admin@grafana:3000/api/dashboards/db \
     -H "Content-Type: application/json" \
     -d @dashboard.json
   ```

### Logging Issues

**Issue**: Logs not being collected

**Symptoms**:
- Missing log entries
- Log aggregation failures
- Incomplete log data

**Diagnosis**:
```bash
# Check log files
ls -la /var/log/trustformers/

# Test log output
logger -t trustformers-serve "test message"

# Check log aggregation
curl http://elasticsearch:9200/_cat/indices
```

**Solutions**:
1. **Configure Log Shipping**:
   ```yaml
   # filebeat.yml
   filebeat.inputs:
   - type: log
     enabled: true
     paths:
       - /var/log/trustformers/*.log
   
   output.elasticsearch:
     hosts: ["elasticsearch:9200"]
   ```

2. **Check Log Permissions**:
   ```bash
   # Fix log permissions
   chown -R trustformers:trustformers /var/log/trustformers/
   chmod 644 /var/log/trustformers/*.log
   ```

## Disaster Recovery and Emergency Procedures

### Service Outage Response

**Issue**: Complete service outage

**Immediate Actions**:
1. **Check Service Status**:
   ```bash
   # Check all instances
   kubectl get pods -l app=trustformers-serve
   
   # Check load balancer
   curl -f http://load-balancer/health
   ```

2. **Escalate if Needed**:
   ```bash
   # Page on-call engineer
   curl -X POST "https://events.pagerduty.com/v2/enqueue" \
     -H "Authorization: Token <token>" \
     -d '{"routing_key": "<key>", "event_action": "trigger"}'
   ```

3. **Implement Workaround**:
   ```bash
   # Redirect traffic to backup region
   kubectl patch service trustformers-serve \
     -p '{"spec":{"selector":{"app":"trustformers-serve-backup"}}}'
   ```

### Data Recovery Procedures

**Issue**: Data loss or corruption

**Recovery Steps**:
1. **Stop Services**:
   ```bash
   kubectl scale deployment trustformers-serve --replicas=0
   ```

2. **Restore from Backup**:
   ```bash
   # Restore database
   pg_restore -h localhost -U user -d database backup.dump
   
   # Restore model files
   aws s3 sync s3://backup-bucket/models/ /opt/models/
   ```

3. **Validate Recovery**:
   ```bash
   # Test data integrity
   trustformers-serve --validate-data
   
   # Test model loading
   trustformers-serve --test-models
   ```

### Rollback Procedures

**Issue**: Deployment causes issues

**Rollback Steps**:
1. **Kubernetes Rollback**:
   ```bash
   # Check rollout history
   kubectl rollout history deployment/trustformers-serve
   
   # Rollback to previous version
   kubectl rollout undo deployment/trustformers-serve
   
   # Rollback to specific revision
   kubectl rollout undo deployment/trustformers-serve --to-revision=2
   ```

2. **Configuration Rollback**:
   ```bash
   # Restore previous configuration
   kubectl apply -f previous-config.yaml
   
   # Restart services
   kubectl rollout restart deployment/trustformers-serve
   ```

## Support and Escalation

### Information to Collect

When reporting issues, collect:

1. **System Information**:
   ```bash
   uname -a
   cat /etc/os-release
   rustc --version
   docker --version
   kubectl version
   ```

2. **Service Information**:
   ```bash
   trustformers-serve --version
   ps aux | grep trustformers-serve
   netstat -tulpn | grep 8080
   ```

3. **Logs**:
   ```bash
   # Recent logs
   journalctl -u trustformers-serve --since "1 hour ago"
   
   # Error logs
   grep -i error /var/log/trustformers/server.log
   ```

4. **Configuration**:
   ```bash
   # Configuration (remove sensitive data)
   cat config.toml | grep -v password | grep -v secret
   ```

### Support Channels

1. **Self-Service**:
   - Check this troubleshooting guide
   - Search GitHub issues
   - Review documentation

2. **Community Support**:
   - GitHub Discussions
   - Stack Overflow with `trustformers-serve` tag
   - Discord community

3. **Enterprise Support**:
   - Email: support@trustformers.com
   - Phone: +1-800-TRUSTFORMERS
   - Support portal: https://support.trustformers.com

### Escalation Matrix

| Severity | Response Time | Escalation |
|----------|---------------|------------|
| Critical | 15 minutes | Immediate page |
| High | 1 hour | Phone call |
| Medium | 4 hours | Email |
| Low | 24 hours | Ticket |

### Emergency Contacts

- **On-Call Engineer**: +1-800-EMERGENCY
- **Technical Lead**: tech-lead@trustformers.com
- **Product Manager**: pm@trustformers.com
- **Security Team**: security@trustformers.com

## Prevention and Best Practices

### Proactive Monitoring

1. **Set Up Alerts**:
   ```yaml
   - alert: HighErrorRate
     expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
     for: 2m
     labels:
       severity: critical
   ```

2. **Health Checks**:
   ```rust
   let health_check = HealthCheck::new()
       .add_check("database", database_health)
       .add_check("redis", redis_health)
       .add_check("models", model_health);
   ```

3. **Capacity Planning**:
   ```bash
   # Monitor resource trends
   kubectl top nodes
   kubectl top pods
   ```

### Regular Maintenance

1. **Update Dependencies**:
   ```bash
   # Update Rust dependencies
   cargo update
   
   # Update container images
   docker pull trustformers-serve:latest
   ```

2. **Security Updates**:
   ```bash
   # Update system packages
   apt-get update && apt-get upgrade
   
   # Scan for vulnerabilities
   cargo audit
   ```

3. **Performance Tuning**:
   ```bash
   # Run performance tests
   cargo run --bin load_test
   
   # Profile application
   perf record -g ./trustformers-serve
   ```

Remember: Prevention is better than cure. Regular monitoring, maintenance, and testing can prevent most issues from occurring in production.