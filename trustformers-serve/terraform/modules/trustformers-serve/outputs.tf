# TrustformeRS Serve Module Outputs

# Network Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnet_ids
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = module.vpc.database_subnet_ids
}

# Security Group Outputs
output "application_security_group_id" {
  description = "ID of the application security group"
  value       = module.security_groups.application_security_group_id
}

output "database_security_group_id" {
  description = "ID of the database security group"
  value       = module.security_groups.database_security_group_id
}

output "redis_security_group_id" {
  description = "ID of the Redis security group"
  value       = module.security_groups.redis_security_group_id
}

output "load_balancer_security_group_id" {
  description = "ID of the load balancer security group"
  value       = module.security_groups.load_balancer_security_group_id
}

# Database Outputs
output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = var.enable_rds ? module.database[0].db_instance_endpoint : null
}

output "database_port" {
  description = "RDS instance port"
  value       = var.enable_rds ? module.database[0].db_instance_port : null
}

output "database_name" {
  description = "Database name"
  value       = var.enable_rds ? module.database[0].db_instance_name : null
}

output "database_username" {
  description = "Database username"
  value       = var.enable_rds ? module.database[0].db_instance_username : null
  sensitive   = true
}

output "database_instance_id" {
  description = "RDS instance ID"
  value       = var.enable_rds ? module.database[0].db_instance_id : null
}

# Redis Outputs
output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = var.enable_redis ? module.redis[0].redis_cluster_address : null
}

output "redis_port" {
  description = "Redis cluster port"
  value       = var.enable_redis ? module.redis[0].redis_cluster_port : null
}

output "redis_cluster_id" {
  description = "Redis cluster ID"
  value       = var.enable_redis ? module.redis[0].redis_cluster_id : null
}

# EKS Outputs
output "eks_cluster_id" {
  description = "EKS cluster ID"
  value       = var.enable_eks ? module.eks[0].cluster_id : null
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = var.enable_eks ? module.eks[0].cluster_name : null
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = var.enable_eks ? module.eks[0].cluster_endpoint : null
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = var.enable_eks ? module.eks[0].cluster_security_group_id : null
}

output "eks_cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = var.enable_eks ? module.eks[0].cluster_iam_role_arn : null
}

output "eks_node_groups" {
  description = "EKS node groups"
  value       = var.enable_eks ? module.eks[0].node_groups : null
}

output "eks_oidc_issuer_url" {
  description = "EKS cluster OIDC issuer URL"
  value       = var.enable_eks ? module.eks[0].cluster_oidc_issuer_url : null
}

# Load Balancer Outputs
output "load_balancer_arn" {
  description = "ARN of the load balancer"
  value       = var.enable_load_balancer ? module.load_balancer[0].load_balancer_arn : null
}

output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = var.enable_load_balancer ? module.load_balancer[0].load_balancer_dns_name : null
}

output "load_balancer_zone_id" {
  description = "Canonical hosted zone ID of the load balancer"
  value       = var.enable_load_balancer ? module.load_balancer[0].load_balancer_zone_id : null
}

output "target_group_arns" {
  description = "ARNs of the target groups"
  value       = var.enable_load_balancer ? module.load_balancer[0].target_group_arns : null
}

# Auto Scaling Outputs
output "autoscaling_group_id" {
  description = "Auto Scaling Group ID"
  value       = var.enable_autoscaling && !var.enable_eks ? module.autoscaling[0].autoscaling_group_id : null
}

output "autoscaling_group_arn" {
  description = "Auto Scaling Group ARN"
  value       = var.enable_autoscaling && !var.enable_eks ? module.autoscaling[0].autoscaling_group_arn : null
}

output "launch_template_id" {
  description = "Launch template ID"
  value       = var.enable_autoscaling && !var.enable_eks ? module.autoscaling[0].launch_template_id : null
}

# Storage Outputs
output "s3_bucket_id" {
  description = "S3 bucket ID"
  value       = var.enable_s3_storage ? module.storage[0].bucket_id : null
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = var.enable_s3_storage ? module.storage[0].bucket_arn : null
}

output "s3_bucket_domain_name" {
  description = "S3 bucket domain name"
  value       = var.enable_s3_storage ? module.storage[0].bucket_domain_name : null
}

# Monitoring Outputs
output "cloudwatch_log_group_name" {
  description = "CloudWatch log group name"
  value       = var.enable_monitoring ? module.monitoring[0].log_group_name : null
}

output "cloudwatch_log_group_arn" {
  description = "CloudWatch log group ARN"
  value       = var.enable_monitoring ? module.monitoring[0].log_group_arn : null
}

output "monitoring_dashboard_url" {
  description = "CloudWatch dashboard URL"
  value       = var.enable_monitoring ? module.monitoring[0].dashboard_url : null
}

# Security Outputs
output "secrets_manager_secret_arn" {
  description = "Secrets Manager secret ARN"
  value       = aws_secretsmanager_secret.app_secrets.arn
}

output "secrets_manager_secret_name" {
  description = "Secrets Manager secret name"
  value       = aws_secretsmanager_secret.app_secrets.name
}

# IAM Outputs
output "iam_instance_profile_name" {
  description = "IAM instance profile name for EC2 instances"
  value       = module.iam.instance_profile_name
}

output "iam_instance_profile_arn" {
  description = "IAM instance profile ARN for EC2 instances"
  value       = module.iam.instance_profile_arn
}

output "iam_role_arn" {
  description = "IAM role ARN for application services"
  value       = module.iam.role_arn
}

# Application Configuration
output "application_url" {
  description = "Application URL"
  value       = var.enable_load_balancer ? "http://${module.load_balancer[0].load_balancer_dns_name}" : null
}

output "grpc_endpoint" {
  description = "gRPC endpoint"
  value       = var.enable_load_balancer ? "${module.load_balancer[0].load_balancer_dns_name}:9090" : null
}

output "metrics_endpoint" {
  description = "Metrics endpoint"
  value       = var.enable_load_balancer ? "http://${module.load_balancer[0].load_balancer_dns_name}/metrics" : null
}

# Connection Strings (for application configuration)
output "database_connection_string" {
  description = "Database connection string (without password)"
  value = var.enable_rds ? format(
    "postgresql://%s@%s:%s/%s",
    module.database[0].db_instance_username,
    module.database[0].db_instance_endpoint,
    module.database[0].db_instance_port,
    module.database[0].db_instance_name
  ) : null
  sensitive = true
}

output "redis_connection_string" {
  description = "Redis connection string"
  value = var.enable_redis ? format(
    "redis://%s:%s",
    module.redis[0].redis_cluster_address,
    module.redis[0].redis_cluster_port
  ) : null
}

# Deployment Information
output "deployment_info" {
  description = "Deployment information summary"
  value = {
    project_name        = var.project_name
    environment        = var.environment
    region             = data.aws_region.current.name
    vpc_id             = module.vpc.vpc_id
    availability_zones = local.availability_zones
    deployment_time    = timestamp()
    
    services = {
      database_enabled      = var.enable_rds
      redis_enabled        = var.enable_redis
      eks_enabled          = var.enable_eks
      load_balancer_enabled = var.enable_load_balancer
      autoscaling_enabled   = var.enable_autoscaling
      monitoring_enabled    = var.enable_monitoring
      s3_storage_enabled   = var.enable_s3_storage
    }
    
    endpoints = {
      application_url = var.enable_load_balancer ? "http://${module.load_balancer[0].load_balancer_dns_name}" : null
      health_check   = var.enable_load_balancer ? "http://${module.load_balancer[0].load_balancer_dns_name}/health" : null
      documentation  = var.enable_load_balancer ? "http://${module.load_balancer[0].load_balancer_dns_name}/docs" : null
    }
  }
}