# TrustformeRS Serve Infrastructure Module
# Deploys a complete TrustformeRS serving infrastructure

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.4"
    }
  }
}

# Local values for common resource naming and tagging
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  common_tags = merge(
    var.tags,
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Service     = "trustformers-serve"
    }
  )
  
  # Network configuration
  availability_zones = data.aws_availability_zones.available.names
  vpc_cidr          = var.vpc_cidr
  
  # Container configuration
  container_image = "${var.container_registry}/${var.container_image}:${var.container_tag}"
  
  # Database configuration
  db_name     = replace("${local.name_prefix}_db", "-", "_")
  db_username = var.db_username
  db_password = var.db_password != "" ? var.db_password : random_password.db_password[0].result
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# Random password for database if not provided
resource "random_password" "db_password" {
  count   = var.db_password == "" ? 1 : 0
  length  = 16
  special = true
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"
  
  name_prefix        = local.name_prefix
  vpc_cidr          = local.vpc_cidr
  availability_zones = local.availability_zones
  
  enable_nat_gateway = var.enable_nat_gateway
  enable_vpn_gateway = var.enable_vpn_gateway
  
  tags = local.common_tags
}

# Security Groups
module "security_groups" {
  source = "./modules/security"
  
  name_prefix = local.name_prefix
  vpc_id      = module.vpc.vpc_id
  
  allowed_cidr_blocks = var.allowed_cidr_blocks
  
  tags = local.common_tags
}

# RDS Database
module "database" {
  source = "./modules/database"
  count  = var.enable_rds ? 1 : 0
  
  name_prefix = local.name_prefix
  
  # Database configuration
  engine         = var.db_engine
  engine_version = var.db_engine_version
  instance_class = var.db_instance_class
  
  database_name = local.db_name
  username      = local.db_username
  password      = local.db_password
  
  # Network configuration
  vpc_id                = module.vpc.vpc_id
  subnet_ids           = module.vpc.database_subnet_ids
  security_group_ids   = [module.security_groups.database_security_group_id]
  
  # Backup and maintenance
  backup_retention_period = var.db_backup_retention_period
  backup_window          = var.db_backup_window
  maintenance_window     = var.db_maintenance_window
  
  # Performance and scaling
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type         = var.db_storage_type
  iops                 = var.db_iops
  
  # Monitoring and logging
  monitoring_interval = var.db_monitoring_interval
  enabled_cloudwatch_logs_exports = var.db_enabled_cloudwatch_logs_exports
  
  # Security
  deletion_protection = var.db_deletion_protection
  skip_final_snapshot = var.db_skip_final_snapshot
  
  tags = local.common_tags
}

# ElastiCache Redis
module "redis" {
  source = "./modules/redis"
  count  = var.enable_redis ? 1 : 0
  
  name_prefix = local.name_prefix
  
  # Redis configuration
  node_type           = var.redis_node_type
  num_cache_nodes     = var.redis_num_cache_nodes
  engine_version      = var.redis_engine_version
  parameter_group_name = var.redis_parameter_group_name
  port                = var.redis_port
  
  # Network configuration
  subnet_ids         = module.vpc.elasticache_subnet_ids
  security_group_ids = [module.security_groups.redis_security_group_id]
  
  # Backup and maintenance
  snapshot_retention_limit = var.redis_snapshot_retention_limit
  snapshot_window         = var.redis_snapshot_window
  maintenance_window      = var.redis_maintenance_window
  
  tags = local.common_tags
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"
  count  = var.enable_eks ? 1 : 0
  
  name_prefix = local.name_prefix
  
  # Cluster configuration
  kubernetes_version = var.eks_kubernetes_version
  
  # Network configuration
  vpc_id                = module.vpc.vpc_id
  subnet_ids           = module.vpc.private_subnet_ids
  endpoint_config      = var.eks_endpoint_config
  
  # Node groups
  node_groups = var.eks_node_groups
  
  # Add-ons
  addons = var.eks_addons
  
  # Security
  security_group_ids = [module.security_groups.eks_cluster_security_group_id]
  
  tags = local.common_tags
}

# Load Balancer
module "load_balancer" {
  source = "./modules/load_balancer"
  count  = var.enable_load_balancer ? 1 : 0
  
  name_prefix = local.name_prefix
  
  # Load balancer configuration
  type           = var.lb_type
  scheme         = var.lb_scheme
  ip_address_type = var.lb_ip_address_type
  
  # Network configuration
  vpc_id     = module.vpc.vpc_id
  subnet_ids = var.lb_scheme == "internal" ? module.vpc.private_subnet_ids : module.vpc.public_subnet_ids
  
  # Security
  security_group_ids = [module.security_groups.load_balancer_security_group_id]
  
  # Target groups
  target_groups = var.lb_target_groups
  
  # SSL/TLS
  certificate_arn = var.ssl_certificate_arn
  ssl_policy     = var.ssl_policy
  
  tags = local.common_tags
}

# Auto Scaling
module "autoscaling" {
  source = "./modules/autoscaling"
  count  = var.enable_autoscaling && !var.enable_eks ? 1 : 0
  
  name_prefix = local.name_prefix
  
  # Launch template configuration
  instance_type          = var.asg_instance_type
  ami_id                = var.asg_ami_id
  key_name              = var.asg_key_name
  security_group_ids    = [module.security_groups.application_security_group_id]
  user_data             = var.asg_user_data
  
  # Auto Scaling Group configuration
  min_size         = var.asg_min_size
  max_size         = var.asg_max_size
  desired_capacity = var.asg_desired_capacity
  
  # Network configuration
  vpc_zone_identifier = module.vpc.private_subnet_ids
  
  # Load balancer integration
  target_group_arns = var.enable_load_balancer ? module.load_balancer[0].target_group_arns : []
  
  # Health checks
  health_check_type         = var.asg_health_check_type
  health_check_grace_period = var.asg_health_check_grace_period
  
  # Scaling policies
  scaling_policies = var.asg_scaling_policies
  
  tags = local.common_tags
}

# CloudWatch Monitoring
module "monitoring" {
  source = "./modules/monitoring"
  count  = var.enable_monitoring ? 1 : 0
  
  name_prefix = local.name_prefix
  
  # Resource ARNs for monitoring
  load_balancer_arn = var.enable_load_balancer ? module.load_balancer[0].load_balancer_arn : ""
  rds_instance_id   = var.enable_rds ? module.database[0].db_instance_id : ""
  redis_cluster_id  = var.enable_redis ? module.redis[0].redis_cluster_id : ""
  eks_cluster_name  = var.enable_eks ? module.eks[0].cluster_name : ""
  
  # Notification configuration
  sns_topic_arn = var.monitoring_sns_topic_arn
  
  # Alarm thresholds
  alarm_thresholds = var.monitoring_alarm_thresholds
  
  tags = local.common_tags
}

# S3 Bucket for model storage
module "storage" {
  source = "./modules/storage"
  count  = var.enable_s3_storage ? 1 : 0
  
  name_prefix = local.name_prefix
  
  # Bucket configuration
  versioning_enabled = var.s3_versioning_enabled
  encryption_enabled = var.s3_encryption_enabled
  
  # Lifecycle management
  lifecycle_rules = var.s3_lifecycle_rules
  
  # Access control
  bucket_policy = var.s3_bucket_policy
  
  tags = local.common_tags
}

# Secrets Manager for sensitive configuration
resource "aws_secretsmanager_secret" "app_secrets" {
  name                    = "${local.name_prefix}-secrets"
  description            = "TrustformeRS Serve application secrets"
  recovery_window_in_days = var.secrets_recovery_window_days
  
  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  
  secret_string = jsonencode({
    database_password = local.db_password
    jwt_secret       = var.jwt_secret != "" ? var.jwt_secret : random_password.jwt_secret.result
    api_keys         = var.api_keys
  })
}

resource "random_password" "jwt_secret" {
  length  = 32
  special = true
}

# IAM roles and policies
module "iam" {
  source = "./modules/iam"
  
  name_prefix = local.name_prefix
  
  # Service roles
  create_ecs_role = var.enable_autoscaling && !var.enable_eks
  create_eks_role = var.enable_eks
  
  # S3 bucket ARN for model storage
  s3_bucket_arn = var.enable_s3_storage ? module.storage[0].bucket_arn : ""
  
  # Secrets Manager ARN
  secrets_manager_arn = aws_secretsmanager_secret.app_secrets.arn
  
  tags = local.common_tags
}