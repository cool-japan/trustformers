# Simple TrustformeRS Serve Deployment Example
# This example deploys a minimal TrustformeRS Serve infrastructure suitable for development or small workloads

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

module "trustformers_serve_simple" {
  source = "../../modules/trustformers-serve"
  
  # Basic configuration
  project_name = "trustformers"
  environment  = "dev"
  
  # Network - smaller CIDR for development
  vpc_cidr = "10.0.0.0/20"  # Supports ~4k IPs
  enable_nat_gateway = false  # Cost optimization for dev
  enable_vpn_gateway = false
  
  # Container configuration
  container_registry = var.container_registry
  container_image   = "trustformers-serve"
  container_tag     = var.container_tag
  
  # Database - minimal configuration
  enable_rds = true
  db_instance_class = "db.t3.micro"
  db_allocated_storage = 20
  db_backup_retention_period = 1  # Minimal backups for dev
  db_deletion_protection = false  # Allow easy cleanup
  db_skip_final_snapshot = true   # Skip snapshot for dev
  
  # Redis - minimal configuration
  enable_redis = true
  redis_node_type = "cache.t3.micro"
  redis_num_cache_nodes = 1
  redis_snapshot_retention_limit = 1
  
  # Use Auto Scaling instead of EKS for simplicity
  enable_eks = false
  enable_autoscaling = true
  
  asg_instance_type = "t3.small"
  asg_min_size = 1
  asg_max_size = 3
  asg_desired_capacity = 1
  asg_key_name = var.key_name  # Optional for SSH access
  
  # Load balancer
  enable_load_balancer = true
  lb_scheme = "internet-facing"
  ssl_certificate_arn = ""  # HTTP only for development
  
  # Monitoring - basic
  enable_monitoring = true
  monitoring_sns_topic_arn = ""  # No alerts for dev
  
  # Storage
  enable_s3_storage = true
  s3_versioning_enabled = false  # Simplify for dev
  s3_encryption_enabled = true
  
  # Security - relaxed for development
  allowed_cidr_blocks = ["0.0.0.0/0"]  # Open for development
  
  tags = {
    Purpose = "development"
    AutoShutdown = "yes"
  }
}

# Outputs
output "application_url" {
  value = module.trustformers_serve_simple.application_url
}

output "database_endpoint" {
  value = module.trustformers_serve_simple.database_endpoint
  sensitive = true
}

output "deployment_info" {
  value = module.trustformers_serve_simple.deployment_info
}