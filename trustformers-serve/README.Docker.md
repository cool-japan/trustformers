# Docker Deployment Guide for TrustformeRS Serve

This guide covers Docker-based deployment options for TrustformeRS Serve, including development, testing, and production scenarios.

## Quick Start

### Prerequisites

- Docker 20.10+ with BuildKit support
- Docker Compose 2.0+
- 8GB+ RAM available for Docker
- 10GB+ free disk space

### Development Environment

```bash
# Build and start development environment
make quick-dev

# Or manually
make build-dev
make up-dev

# Access the application
curl http://localhost:8081/health
```

### Production Environment

```bash
# Build production image
make build

# Start production stack
make up

# Access the application
curl http://localhost:8080/health
```

## Docker Images

### Available Variants

| Variant | Purpose | Base Image | Size* | Use Case |
|---------|---------|------------|-------|----------|
| `production` | Production deployment | `gcr.io/distroless/cc-debian12` | ~50MB | Production workloads |
| `development` | Development/debugging | `rust:1.75-slim` | ~2GB | Local development |
| `debug` | Troubleshooting | `debian:bookworm-slim` | ~200MB | Production debugging |
| `testing` | CI/CD testing | `rust:1.75-slim` | ~2GB | Automated testing |
| `security-hardened` | High-security environments | `gcr.io/distroless/cc-debian12` | ~50MB | Security-critical deployments |

*Approximate sizes may vary

### Build Targets

```bash
# Production image (default)
make build

# Development image with tools
make build-dev

# Debug image with debugging tools
make build-debug

# Testing image (runs tests during build)
make build-test

# Security-hardened image
make build-security

# All variants
make build-all
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level (trace, debug, info, warn, error) |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8080` | HTTP API port |
| `GRPC_PORT` | `9090` | gRPC service port |
| `METRICS_PORT` | `9091` | Prometheus metrics port |
| `DATABASE_URL` | - | PostgreSQL connection string |
| `REDIS_URL` | - | Redis connection string |

### Volume Mounts

| Volume | Purpose | Path |
|--------|---------|------|
| `models` | Model files and weights | `/app/models` |
| `config` | Configuration files | `/app/config` |
| `logs` | Application logs | `/app/logs` |

## Docker Compose Profiles

### Development Profile

```bash
# Start development environment
docker-compose --profile development up -d

# Or use make
make up-dev
```

Services included:
- `trustformers-dev` - Development application with hot reload
- `postgres-dev` - Development database
- `redis-dev` - Development cache

### Debug Profile

```bash
# Start debug environment
docker-compose --profile debug up -d

# Or use make
make up-debug
```

Additional capabilities:
- Debugging tools (gdb, strace, valgrind)
- Extended privileges for profiling
- Debug symbols included

### Testing Profile

```bash
# Run tests
docker-compose --profile testing up

# Or use make
make up-test
```

Features:
- Isolated test database
- Test data fixtures
- Coverage reporting

### Monitoring Profile

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Or use make
make up-monitor
```

Monitoring stack:
- Prometheus (metrics collection)
- Grafana (dashboards)
- Jaeger (distributed tracing)

## Build Optimization

### Multi-stage Builds

The optimized Dockerfile uses multi-stage builds for:

1. **Dependency Caching**: Dependencies are built separately and cached
2. **Parallel Builds**: Multiple stages can build in parallel
3. **Size Optimization**: Final image contains only runtime artifacts
4. **Security**: Minimal attack surface with distroless base

### Build Cache

```bash
# Use build cache
docker build --cache-from trustformers-serve:latest .

# Or with BuildKit
DOCKER_BUILDKIT=1 docker build .
```

### Multi-architecture Support

```bash
# Build for multiple architectures
make build-multi-arch

# Or manually
docker buildx build --platform linux/amd64,linux/arm64 .
```

## Security Features

### Image Security

- **Distroless base**: Minimal attack surface
- **Non-root user**: Runs as unprivileged user (65532)
- **Security scanning**: Integrated Trivy scanning
- **Minimal packages**: Only essential runtime dependencies

### Security Scanning

```bash
# Scan images for vulnerabilities
make scan

# Manual scan with Trivy
trivy image trustformers-serve:latest
```

### Hardened Deployment

```bash
# Build security-hardened image
make build-security

# Use security-hardened profile
docker-compose --profile security up -d
```

## Performance Optimization

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

### Build Performance

```bash
# Parallel builds
make build --parallel

# No cache for clean builds
make build --no-cache

# Compressed builds
make build --compress
```

### Runtime Performance

- **JIT compilation**: Rust optimizations enabled
- **Memory management**: Efficient memory allocation
- **Connection pooling**: Database and cache connections
- **Load balancing**: Nginx upstream configuration

## Networking

### Port Mapping

| Service | Internal Port | External Port | Purpose |
|---------|---------------|---------------|---------|
| HTTP API | 8080 | 8080 | REST API endpoints |
| gRPC | 9090 | 9090 | gRPC service |
| Metrics | 9091 | 9091 | Prometheus metrics |
| Postgres | 5432 | 5432* | Database (dev only) |
| Redis | 6379 | 6379* | Cache (dev only) |

*Only exposed in development profile

### Load Balancing

Nginx configuration provides:
- **Health checks**: Automatic backend health monitoring
- **Rate limiting**: API endpoint protection
- **SSL termination**: HTTPS support
- **WebSocket support**: Real-time communication
- **gRPC proxying**: Protocol buffer support

## Monitoring and Observability

### Health Checks

```bash
# Application health
curl http://localhost:8080/health

# Container health
docker-compose ps

# Service health
make health
```

### Metrics

- **Prometheus**: `/metrics` endpoint
- **Custom metrics**: Application-specific metrics
- **System metrics**: Container resource usage

### Logging

```bash
# View all logs
make logs

# Application logs only
make logs-app

# Development logs
make logs-dev

# Follow logs
docker-compose logs -f
```

### Tracing

Distributed tracing with Jaeger:
- **Trace collection**: OpenTelemetry integration
- **Span visualization**: Request flow analysis
- **Performance insights**: Latency analysis

## Backup and Recovery

### Database Backup

```bash
# Backup database
make db-backup

# Restore from backup
make db-restore BACKUP_FILE=backup.sql
```

### Application State

```bash
# Full backup
make backup

# Restore state
make restore BACKUP_DATE=20240716_120000
```

## Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Clean build cache
   make clean
   
   # Check Docker space
   docker system df
   ```

2. **Container Won't Start**
   ```bash
   # Check logs
   docker-compose logs trustformers-serve
   
   # Debug mode
   make up-debug
   ```

3. **Performance Issues**
   ```bash
   # Check resource usage
   docker stats
   
   # Run profiling
   make monitor
   ```

### Debug Commands

```bash
# Enter debug container
docker-compose exec trustformers-debug bash

# Check application status
docker-compose exec trustformers-serve /usr/local/bin/trustformers-serve --health-check

# View configuration
docker-compose config
```

### Log Levels

Adjust logging for debugging:

```bash
# Trace level (maximum verbosity)
RUST_LOG=trace make up-dev

# Debug level
RUST_LOG=debug make up

# Info level (default)
RUST_LOG=info make up
```

## Production Deployment

### Pre-deployment Checklist

- [ ] Configure SSL certificates
- [ ] Set up external database
- [ ] Configure Redis cluster
- [ ] Set resource limits
- [ ] Configure monitoring
- [ ] Set up log aggregation
- [ ] Configure backup strategy
- [ ] Security scan passed
- [ ] Load testing completed

### Deployment Steps

1. **Build and test**
   ```bash
   make ci
   ```

2. **Push to registry**
   ```bash
   make push DOCKER_REGISTRY=your-registry.com
   ```

3. **Deploy to production**
   ```bash
   # Update docker-compose.prod.yml
   docker-compose -f docker-compose.prod.yml up -d
   ```

4. **Verify deployment**
   ```bash
   make health
   ```

### Scaling

```bash
# Scale application
docker-compose up -d --scale trustformers-serve=3

# Update nginx upstream
# Edit nginx.conf to add new backends
```

## Development Workflow

### Local Development

```bash
# Start development environment
make quick-dev

# Make code changes...

# Rebuild and restart
make build-dev && make restart

# Run tests
make test
```

### Hot Reload

Development container supports hot reload:

```bash
# Start with file watching
make up-dev

# Container will automatically restart on file changes
```

### Testing

```bash
# Run all tests
make test

# Load testing
make load-test

# Performance benchmarks
make benchmark
```

## Advanced Features

### Custom Build Arguments

```bash
# Custom Rust version
make build RUST_VERSION=1.76

# Custom target architecture
make build TARGET_ARCH=aarch64-unknown-linux-gnu
```

### Multi-stage Development

```bash
# Build specific stage
docker build --target development .

# Build with custom context
docker build -f Dockerfile.optimized .
```

### Registry Integration

```bash
# Configure registry
export DOCKER_REGISTRY=your-registry.com/project

# Build and push
make deploy
```

## Support and Resources

- **Documentation**: [Main README](README.md)
- **Issues**: [GitHub Issues](https://github.com/cool-japan/trustformers/issues)
- **Docker Hub**: [TrustformeRS Images](https://hub.docker.com/r/trustformers/serve)
- **Examples**: [examples/](examples/) directory

For additional help, check the troubleshooting section or open an issue on GitHub.