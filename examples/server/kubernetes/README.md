# Kubernetes Deployment Guide for TrustformeRS Server

This guide covers deploying the TrustformeRS REST API server on Kubernetes.

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Docker registry access (or local registry)
- Helm (optional, for advanced deployments)

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build the image
docker build -t your-registry/trustformers-server:latest -f examples/server/Dockerfile .

# Push to registry
docker push your-registry/trustformers-server:latest
```

### 2. Apply Kubernetes Manifests

```bash
# Create namespace
kubectl create namespace trustformers

# Apply all manifests
kubectl apply -f examples/server/kubernetes/ -n trustformers

# Check deployment status
kubectl get pods -n trustformers
kubectl get svc -n trustformers
```

### 3. Access the Service

```bash
# Port forward for testing
kubectl port-forward -n trustformers svc/trustformers-server 8080:8080

# Or get the LoadBalancer IP (if using cloud provider)
kubectl get svc -n trustformers trustformers-server
```

## Configuration

### Environment Variables

Configure the deployment through environment variables in `deployment.yaml`:

```yaml
env:
  - name: RUST_LOG
    value: "info"
  - name: MODEL_CACHE_DIR
    value: "/var/cache/trustformers"
  - name: MAX_MODELS
    value: "10"
```

### Resource Limits

Adjust resources based on your model requirements:

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

### Persistent Storage

For model caching, use a PersistentVolumeClaim:

```yaml
volumeMounts:
  - name: model-cache
    mountPath: /var/cache/trustformers
volumes:
  - name: model-cache
    persistentVolumeClaim:
      claimName: trustformers-cache
```

## Scaling

### Horizontal Pod Autoscaling

```bash
# Enable HPA
kubectl autoscale deployment trustformers-server \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n trustformers
```

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment trustformers-server --replicas=5 -n trustformers
```

## Monitoring

### Prometheus Metrics

The server exposes metrics at `/metrics`. Configure Prometheus ServiceMonitor:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: trustformers-server
spec:
  selector:
    matchLabels:
      app: trustformers-server
  endpoints:
  - port: http
    path: /metrics
```

### Health Checks

The deployment includes liveness and readiness probes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 10
```

## Production Considerations

### 1. Security

- Use NetworkPolicies to restrict traffic
- Enable RBAC for service accounts
- Use secrets for sensitive configuration
- Enable TLS with cert-manager

### 2. High Availability

- Deploy across multiple availability zones
- Use pod anti-affinity rules
- Configure pod disruption budgets
- Enable session affinity for stateful operations

### 3. Performance

- Use node selectors for GPU nodes (when available)
- Configure resource requests/limits appropriately
- Enable response compression
- Use CDN for static assets

### 4. Observability

- Centralized logging with Fluentd/Elasticsearch
- Distributed tracing with Jaeger
- Custom dashboards in Grafana
- Alert rules for SLOs

## Troubleshooting

### Common Issues

1. **OOMKilled Pods**
   - Increase memory limits
   - Reduce MAX_MODELS
   - Enable model unloading

2. **Slow Startup**
   - Increase initialDelaySeconds
   - Pre-load models in init containers
   - Use readiness gates

3. **High Latency**
   - Check node placement
   - Review resource allocation
   - Enable request batching

### Debugging Commands

```bash
# Check pod logs
kubectl logs -n trustformers deployment/trustformers-server

# Describe pod for events
kubectl describe pod -n trustformers <pod-name>

# Execute into pod
kubectl exec -it -n trustformers <pod-name> -- /bin/bash

# Check resource usage
kubectl top pods -n trustformers
```

## Advanced Deployments

### Using Helm

```bash
# Install with Helm
helm install trustformers ./helm/trustformers-server \
  --namespace trustformers \
  --create-namespace \
  --values values.yaml

# Upgrade deployment
helm upgrade trustformers ./helm/trustformers-server \
  --namespace trustformers \
  --values values.yaml
```

### GitOps with ArgoCD

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: trustformers-server
spec:
  source:
    repoURL: https://github.com/cool-japan/trustformers
    path: examples/server/kubernetes
    targetRevision: main
  destination:
    server: https://kubernetes.default.svc
    namespace: trustformers
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

## Cost Optimization

1. **Spot Instances**: Use spot/preemptible nodes for non-critical workloads
2. **Cluster Autoscaling**: Enable cluster autoscaler for dynamic scaling
3. **Resource Optimization**: Right-size pods based on actual usage
4. **Model Caching**: Share model cache across pods with ReadWriteMany PVC