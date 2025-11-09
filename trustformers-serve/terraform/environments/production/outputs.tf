# Production Environment Outputs

output "application_url" {
  description = "Application URL"
  value       = module.trustformers_serve.application_url
}

output "load_balancer_dns_name" {
  description = "Load balancer DNS name"
  value       = module.trustformers_serve.load_balancer_dns_name
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.trustformers_serve.eks_cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.trustformers_serve.eks_cluster_endpoint
}

output "database_endpoint" {
  description = "Database endpoint"
  value       = module.trustformers_serve.database_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = module.trustformers_serve.redis_endpoint
}

output "s3_bucket_name" {
  description = "S3 bucket name for model storage"
  value       = module.trustformers_serve.s3_bucket_id
}

output "secrets_manager_secret_name" {
  description = "Secrets Manager secret name"
  value       = module.trustformers_serve.secrets_manager_secret_name
}

output "monitoring_dashboard_url" {
  description = "CloudWatch dashboard URL"
  value       = module.trustformers_serve.monitoring_dashboard_url
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.trustformers_serve.vpc_id
}

output "waf_web_acl_arn" {
  description = "WAF Web ACL ARN"
  value       = aws_wafv2_web_acl.main.arn
}

output "cloudtrail_arn" {
  description = "CloudTrail ARN"
  value       = aws_cloudtrail.main.arn
}

output "backup_vault_arn" {
  description = "Backup vault ARN"
  value       = aws_backup_vault.main.arn
}

output "deployment_summary" {
  description = "Deployment summary"
  value = {
    environment     = "production"
    region         = var.aws_region
    application_url = module.trustformers_serve.application_url
    cluster_name   = module.trustformers_serve.eks_cluster_name
    vpc_id         = module.trustformers_serve.vpc_id
    deployment_time = timestamp()
  }
}