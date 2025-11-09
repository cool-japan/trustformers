# Outputs for the complete AWS EKS module

# Cluster Information
output "cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.main.id
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_version" {
  description = "EKS cluster Kubernetes version"
  value       = aws_eks_cluster.main.version
}

output "cluster_platform_version" {
  description = "EKS cluster platform version"
  value       = aws_eks_cluster.main.platform_version
}

output "cluster_status" {
  description = "EKS cluster status"
  value       = aws_eks_cluster.main.status
}

# Cluster Security
output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.main.certificate_authority[0].data
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.main.arn
}

# OIDC Provider
output "oidc_provider_arn" {
  description = "ARN of the OIDC provider for the EKS cluster"
  value       = aws_iam_openid_connect_provider.eks.arn
}

output "oidc_provider_url" {
  description = "URL of the OIDC provider for the EKS cluster"
  value       = local.oidc_provider_url
}

# IAM Roles
output "cluster_iam_role_arn" {
  description = "IAM role ARN of the EKS cluster"
  value       = aws_iam_role.cluster.arn
}

output "cluster_iam_role_name" {
  description = "IAM role name of the EKS cluster"
  value       = aws_iam_role.cluster.name
}

output "node_group_iam_role_arn" {
  description = "IAM role ARN of the EKS node groups"
  value       = aws_iam_role.node_group.arn
}

output "node_group_iam_role_name" {
  description = "IAM role name of the EKS node groups"
  value       = aws_iam_role.node_group.name
}

# Service-specific IAM Roles
output "aws_load_balancer_controller_role_arn" {
  description = "IAM role ARN for AWS Load Balancer Controller"
  value       = var.enable_aws_load_balancer_controller ? aws_iam_role.aws_load_balancer_controller[0].arn : null
}

output "cluster_autoscaler_role_arn" {
  description = "IAM role ARN for Cluster Autoscaler"
  value       = var.enable_cluster_autoscaler ? aws_iam_role.cluster_autoscaler[0].arn : null
}

output "ebs_csi_driver_role_arn" {
  description = "IAM role ARN for EBS CSI Driver"
  value       = var.enable_ebs_csi_driver ? aws_iam_role.ebs_csi_driver[0].arn : null
}

# Node Groups
output "node_groups" {
  description = "Map of node group configurations and their statuses"
  value = {
    for k, v in aws_eks_node_group.main : k => {
      arn           = v.arn
      status        = v.status
      capacity_type = v.capacity_type
      instance_types = v.instance_types
      ami_type      = v.ami_type
      disk_size     = v.disk_size
      scaling_config = v.scaling_config
      remote_access = v.remote_access
      labels        = v.labels
      taints        = v.taint
    }
  }
}

# Node Group ARNs
output "node_group_arns" {
  description = "List of node group ARNs"
  value       = [for ng in aws_eks_node_group.main : ng.arn]
}

output "node_group_statuses" {
  description = "Map of node group names to their statuses"
  value       = { for k, v in aws_eks_node_group.main : k => v.status }
}

# Security Groups
output "additional_security_group_id" {
  description = "ID of additional security group created for the cluster"
  value       = length(aws_security_group.additional) > 0 ? aws_security_group.additional[0].id : null
}

# Encryption
output "kms_key_id" {
  description = "KMS key ID used for cluster encryption"
  value       = aws_kms_key.eks.id
}

output "kms_key_arn" {
  description = "KMS key ARN used for cluster encryption"
  value       = aws_kms_key.eks.arn
}

# Logging
output "cloudwatch_log_group_name" {
  description = "CloudWatch log group name for EKS cluster logs"
  value       = aws_cloudwatch_log_group.eks.name
}

output "cloudwatch_log_group_arn" {
  description = "CloudWatch log group ARN for EKS cluster logs"
  value       = aws_cloudwatch_log_group.eks.arn
}

# Add-ons
output "cluster_addons" {
  description = "Map of cluster add-ons"
  value = {
    for k, v in aws_eks_addon.main : k => {
      arn    = v.arn
      status = v.status
      version = v.addon_version
    }
  }
}

# Network Information
output "cluster_primary_security_group_id" {
  description = "The cluster primary security group ID created by EKS"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "vpc_config" {
  description = "VPC configuration of the cluster"
  value = {
    vpc_id                  = aws_eks_cluster.main.vpc_config[0].vpc_id
    subnet_ids              = aws_eks_cluster.main.vpc_config[0].subnet_ids
    endpoint_private_access = aws_eks_cluster.main.vpc_config[0].endpoint_private_access
    endpoint_public_access  = aws_eks_cluster.main.vpc_config[0].endpoint_public_access
    public_access_cidrs     = aws_eks_cluster.main.vpc_config[0].public_access_cidrs
  }
}

# kubectl Configuration
output "kubectl_config" {
  description = "kubectl configuration for connecting to the cluster"
  value = {
    cluster_name     = aws_eks_cluster.main.name
    endpoint         = aws_eks_cluster.main.endpoint
    ca_certificate   = aws_eks_cluster.main.certificate_authority[0].data
    region          = data.aws_region.current.name
    
    # Command to update kubeconfig
    update_kubeconfig_command = "aws eks update-kubeconfig --region ${data.aws_region.current.name} --name ${aws_eks_cluster.main.name}"
  }
}

# Helm Release Information
output "helm_releases" {
  description = "Information about installed Helm releases"
  value = {
    aws_load_balancer_controller = var.enable_aws_load_balancer_controller ? {
      name      = helm_release.aws_load_balancer_controller[0].name
      namespace = helm_release.aws_load_balancer_controller[0].namespace
      version   = helm_release.aws_load_balancer_controller[0].version
      status    = helm_release.aws_load_balancer_controller[0].status
    } : null
    
    cluster_autoscaler = var.enable_cluster_autoscaler ? {
      name      = helm_release.cluster_autoscaler[0].name
      namespace = helm_release.cluster_autoscaler[0].namespace
      version   = helm_release.cluster_autoscaler[0].version
      status    = helm_release.cluster_autoscaler[0].status
    } : null
    
    metrics_server = var.enable_metrics_server ? {
      name      = helm_release.metrics_server[0].name
      namespace = helm_release.metrics_server[0].namespace
      version   = helm_release.metrics_server[0].version
      status    = helm_release.metrics_server[0].status
    } : null
    
    trustformers_serve = var.install_trustformers_serve ? {
      name      = helm_release.trustformers_serve[0].name
      namespace = helm_release.trustformers_serve[0].namespace
      version   = helm_release.trustformers_serve[0].version
      status    = helm_release.trustformers_serve[0].status
    } : null
  }
}

# Cost Information
output "estimated_costs" {
  description = "Estimated monthly costs for the EKS cluster"
  value = {
    cluster_cost = "Approximately $72/month for EKS control plane"
    node_groups = {
      for k, v in var.node_groups : k => {
        instance_type = v.instance_types[0]
        min_nodes     = v.min_size
        max_nodes     = v.max_size
        estimated_cost = "Varies based on instance type and usage"
      }
    }
    note = "Actual costs depend on instance types, usage patterns, and additional AWS services"
  }
}

# Cluster Tags
output "cluster_tags" {
  description = "Tags applied to the EKS cluster"
  value       = aws_eks_cluster.main.tags
}

# Cluster Identity
output "cluster_identity" {
  description = "EKS cluster identity information"
  value = {
    oidc = aws_eks_cluster.main.identity[0].oidc[0]
  }
}

# Resource Information for Monitoring
output "monitoring_targets" {
  description = "Resources that should be monitored"
  value = {
    cluster_name         = aws_eks_cluster.main.name
    cluster_arn          = aws_eks_cluster.main.arn
    log_group_name       = aws_cloudwatch_log_group.eks.name
    node_group_arns      = [for ng in aws_eks_node_group.main : ng.arn]
    security_group_ids   = concat(
      [aws_eks_cluster.main.vpc_config[0].cluster_security_group_id],
      length(aws_security_group.additional) > 0 ? [aws_security_group.additional[0].id] : []
    )
  }
}

# Connection Information for Applications
output "connection_info" {
  description = "Connection information for applications"
  value = {
    # Internal cluster endpoint for in-cluster communication
    internal_endpoint = aws_eks_cluster.main.endpoint
    
    # External access information
    public_endpoint_enabled = aws_eks_cluster.main.vpc_config[0].endpoint_public_access
    private_endpoint_enabled = aws_eks_cluster.main.vpc_config[0].endpoint_private_access
    
    # DNS information
    cluster_dns_name = replace(aws_eks_cluster.main.endpoint, "https://", "")
    
    # Service discovery
    cluster_domain = "cluster.local"
  }
}