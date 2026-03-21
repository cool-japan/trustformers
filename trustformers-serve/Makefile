# Makefile for TrustformeRS Serve Docker Operations

# Configuration
DOCKER_REGISTRY ?= 
IMAGE_NAME ?= trustformers-serve
VERSION ?= latest
DOCKERFILE ?= Dockerfile.optimized
RUST_VERSION ?= 1.75

# Docker Compose profiles
COMPOSE_FILE ?= docker-compose.yml
DEV_PROFILE ?= development
DEBUG_PROFILE ?= debug
TEST_PROFILE ?= testing
MONITOR_PROFILE ?= monitoring

# Build script
BUILD_SCRIPT = ./scripts/docker-build.sh

.PHONY: help build dev debug test production all clean scan push pull logs status stop down restart

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "TrustformeRS Serve Docker Operations"
	@echo ""
	@echo "Usage: make [TARGET] [VARIABLES]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables:"
	@echo "  VERSION          Image version tag (default: latest)"
	@echo "  DOCKER_REGISTRY  Docker registry prefix"
	@echo "  IMAGE_NAME       Docker image name (default: trustformers-serve)"
	@echo "  DOCKERFILE       Dockerfile to use (default: Dockerfile.optimized)"
	@echo ""
	@echo "Examples:"
	@echo "  make build VERSION=v1.0.0"
	@echo "  make dev"
	@echo "  make push DOCKER_REGISTRY=docker.io/myorg"

# Build targets
build: ## Build production Docker image
	$(BUILD_SCRIPT) production -t $(VERSION)

build-dev: ## Build development Docker image
	$(BUILD_SCRIPT) development -t $(VERSION)

build-debug: ## Build debug Docker image
	$(BUILD_SCRIPT) debug -t $(VERSION)

build-test: ## Build and run tests
	$(BUILD_SCRIPT) testing -t $(VERSION)

build-all: ## Build all image variants
	$(BUILD_SCRIPT) all -t $(VERSION)

build-multi-arch: ## Build multi-architecture images
	$(BUILD_SCRIPT) multi-arch -t $(VERSION)

build-security: ## Build security-hardened image
	$(BUILD_SCRIPT) security -t $(VERSION)

# Quick aliases
production: build ## Alias for build
dev: build-dev ## Alias for build-dev
debug: build-debug ## Alias for build-debug
test: build-test ## Alias for build-test
all: build-all ## Alias for build-all
security: build-security ## Alias for build-security

# Docker Compose operations
up: ## Start services with docker-compose
	docker-compose -f $(COMPOSE_FILE) up -d

up-dev: ## Start development environment
	docker-compose -f $(COMPOSE_FILE) --profile $(DEV_PROFILE) up -d

up-debug: ## Start debug environment
	docker-compose -f $(COMPOSE_FILE) --profile $(DEBUG_PROFILE) up -d

up-test: ## Start testing environment
	docker-compose -f $(COMPOSE_FILE) --profile $(TEST_PROFILE) up

up-monitor: ## Start with monitoring stack
	docker-compose -f $(COMPOSE_FILE) --profile $(MONITOR_PROFILE) up -d

up-all: ## Start all services and profiles
	docker-compose -f $(COMPOSE_FILE) --profile $(DEV_PROFILE) --profile $(MONITOR_PROFILE) up -d

down: ## Stop and remove all containers
	docker-compose -f $(COMPOSE_FILE) down

stop: ## Stop all running containers
	docker-compose -f $(COMPOSE_FILE) stop

restart: ## Restart all services
	docker-compose -f $(COMPOSE_FILE) restart

# Status and logging
status: ## Show status of all containers
	docker-compose -f $(COMPOSE_FILE) ps

logs: ## Show logs from all services
	docker-compose -f $(COMPOSE_FILE) logs -f

logs-app: ## Show logs from application only
	docker-compose -f $(COMPOSE_FILE) logs -f trustformers-serve

logs-dev: ## Show logs from development environment
	docker-compose -f $(COMPOSE_FILE) logs -f trustformers-dev

# Registry operations
push: ## Push images to registry
	$(BUILD_SCRIPT) push -t $(VERSION)

pull: ## Pull latest images from registry
	docker-compose -f $(COMPOSE_FILE) pull

# Security and maintenance
scan: ## Run security scan on images
	$(BUILD_SCRIPT) scan -t $(VERSION)

clean: ## Clean up Docker build cache and unused images
	$(BUILD_SCRIPT) clean

clean-all: ## Clean up everything (images, containers, volumes)
	docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans
	docker system prune -a -f
	docker volume prune -f

# Database operations
db-migrate: ## Run database migrations
	docker-compose -f $(COMPOSE_FILE) exec postgres psql -U user -d trustformers -f /docker-entrypoint-initdb.d/init.sql

db-backup: ## Backup database
	mkdir -p backups
	docker-compose -f $(COMPOSE_FILE) exec postgres pg_dump -U user trustformers > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore: ## Restore database from backup (specify BACKUP_FILE)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Please specify BACKUP_FILE=path/to/backup.sql"; exit 1; fi
	docker-compose -f $(COMPOSE_FILE) exec -T postgres psql -U user -d trustformers < $(BACKUP_FILE)

# Health checks
health: ## Check health of all services
	@echo "Checking health of services..."
	@docker-compose -f $(COMPOSE_FILE) ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "Application health:"
	@curl -s http://localhost:8080/health | jq . || echo "Application not responding"

health-dev: ## Check health of development services
	@curl -s http://localhost:8081/health | jq . || echo "Development application not responding"

# Performance testing
load-test: ## Run load tests against the application
	docker run --rm --network host \
		-v $(PWD)/examples:/examples:ro \
		loadimpact/k6:latest run /examples/load_test_config.json

benchmark: ## Run performance benchmarks
	cargo bench --all-features

# Development helpers
format: ## Format Rust code
	cargo fmt --all

lint: ## Run clippy lints
	cargo clippy --all-targets --all-features -- -D warnings

test-local: ## Run tests locally
	cargo test --all-features

watch: ## Watch for changes and rebuild (requires cargo-watch)
	cargo watch -x "run --release"

# Documentation
docs: ## Generate and serve documentation
	cargo doc --open --all-features

# Initialization
init: ## Initialize project dependencies and setup
	@echo "Initializing TrustformeRS Serve..."
	@mkdir -p models config logs backups
	@chmod +x scripts/*.sh
	@echo "Setup complete!"

# Quick development workflow
quick-dev: init build-dev up-dev ## Initialize, build and start development environment

# Production deployment workflow
deploy: build push ## Build and push production images

# Full CI workflow
ci: clean build-all test scan ## Complete CI pipeline

# Show Docker info
info: ## Show Docker system information
	@echo "Docker Info:"
	@docker info --format "Version: {{.ServerVersion}}"
	@echo ""
	@echo "Images:"
	@docker images $(IMAGE_NAME)* --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
	@echo ""
	@echo "Containers:"
	@docker ps -a --filter "ancestor=$(IMAGE_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Environment setup for different scenarios
setup-prod: ## Setup production environment files
	@echo "Creating production environment..."
	@cp docker-compose.yml docker-compose.prod.yml
	@echo "Production setup complete"

setup-dev: ## Setup development environment files
	@echo "Creating development environment..."
	@cp docker-compose.yml docker-compose.dev.yml
	@echo "Development setup complete"

# Performance monitoring
monitor: ## Start monitoring dashboard
	@echo "Starting monitoring stack..."
	$(MAKE) up-monitor
	@echo "Monitoring available at:"
	@echo "  Grafana: http://localhost:3000 (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Jaeger: http://localhost:16686"

# Backup and restore operations
backup: ## Backup entire application state
	@echo "Creating backup..."
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	$(MAKE) db-backup
	docker run --rm -v trustformers-serve_models:/data -v $(PWD)/backups:/backup alpine tar czf /backup/models_$(shell date +%Y%m%d_%H%M%S).tar.gz -C /data .
	@echo "Backup complete"

restore: ## Restore application state (specify BACKUP_DATE)
	@if [ -z "$(BACKUP_DATE)" ]; then echo "Please specify BACKUP_DATE=YYYYMMDD_HHMMSS"; exit 1; fi
	$(MAKE) db-restore BACKUP_FILE=backups/backup_$(BACKUP_DATE).sql
	docker run --rm -v trustformers-serve_models:/data -v $(PWD)/backups:/backup alpine tar xzf /backup/models_$(BACKUP_DATE).tar.gz -C /data

# Performance optimization
optimize: ## Run performance optimizations
	@echo "Running optimizations..."
	docker system prune -f
	@echo "Optimization complete"