# Outputs for Multi-Cloud Deploy module

# General Deployment Information
output "deployment_name" {
  description = "Name of the multi-cloud deployment"
  value       = local.deployment_name
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "primary_cloud" {
  description = "Primary cloud provider"
  value       = var.primary_cloud
}

output "enabled_clouds" {
  description = "List of enabled cloud providers"
  value       = [for cloud, config in var.clouds : cloud if config.enabled]
}

# AWS Outputs
output "aws_deployment" {
  description = "AWS deployment information"
  value = local.aws_enabled ? {
    cluster_name     = module.aws_deployment[0].cluster_name
    cluster_endpoint = module.aws_deployment[0].cluster_endpoint
    cluster_version  = module.aws_deployment[0].cluster_version
    cluster_arn      = module.aws_deployment[0].cluster_arn
    region          = var.clouds.aws.region
    vpc_config      = module.aws_deployment[0].vpc_config
    node_groups     = module.aws_deployment[0].node_groups
    helm_releases   = module.aws_deployment[0].helm_releases
  } : null
}

output "aws_cluster_name" {
  description = "AWS EKS cluster name"
  value       = local.aws_enabled ? module.aws_deployment[0].cluster_name : null
}

output "aws_cluster_endpoint" {
  description = "AWS EKS cluster endpoint"
  value       = local.aws_enabled ? module.aws_deployment[0].cluster_endpoint : null
}

output "aws_cluster_arn" {
  description = "AWS EKS cluster ARN"
  value       = local.aws_enabled ? module.aws_deployment[0].cluster_arn : null
}

output "aws_kubectl_config" {
  description = "AWS EKS kubectl configuration"
  value       = local.aws_enabled ? module.aws_deployment[0].kubectl_config : null
  sensitive   = true
}

# Azure Outputs
output "azure_deployment" {
  description = "Azure deployment information"
  value = local.azure_enabled ? {
    cluster_name     = module.azure_deployment[0].cluster_name
    cluster_endpoint = module.azure_deployment[0].cluster_endpoint
    cluster_version  = module.azure_deployment[0].cluster_version
    cluster_fqdn     = module.azure_deployment[0].cluster_fqdn
    location        = var.clouds.azure.location
    resource_group  = module.azure_deployment[0].resource_group_name
    node_pools      = module.azure_deployment[0].additional_node_pools
    helm_releases   = module.azure_deployment[0].helm_releases
  } : null
}

output "azure_cluster_name" {
  description = "Azure AKS cluster name"
  value       = local.azure_enabled ? module.azure_deployment[0].cluster_name : null
}

output "azure_cluster_endpoint" {
  description = "Azure AKS cluster endpoint"
  value       = local.azure_enabled ? module.azure_deployment[0].cluster_endpoint : null
}

output "azure_cluster_fqdn" {
  description = "Azure AKS cluster FQDN"
  value       = local.azure_enabled ? module.azure_deployment[0].cluster_fqdn : null
}

output "azure_kube_config" {
  description = "Azure AKS kubeconfig"
  value       = local.azure_enabled ? module.azure_deployment[0].kube_config : null
  sensitive   = true
}

# GCP Outputs
output "gcp_deployment" {
  description = "GCP deployment information"
  value = local.gcp_enabled ? {
    cluster_name     = module.gcp_deployment[0].cluster_name
    cluster_endpoint = module.gcp_deployment[0].cluster_endpoint
    cluster_version  = module.gcp_deployment[0].cluster_version
    cluster_location = module.gcp_deployment[0].cluster_location
    project_id      = var.clouds.gcp.project_id
    region          = var.clouds.gcp.region
    node_pools      = module.gcp_deployment[0].additional_node_pools
    helm_releases   = module.gcp_deployment[0].helm_releases
  } : null
}

output "gcp_cluster_name" {
  description = "GCP GKE cluster name"
  value       = local.gcp_enabled ? module.gcp_deployment[0].cluster_name : null
}

output "gcp_cluster_endpoint" {
  description = "GCP GKE cluster endpoint"
  value       = local.gcp_enabled ? module.gcp_deployment[0].cluster_endpoint : null
}

output "gcp_cluster_location" {
  description = "GCP GKE cluster location"
  value       = local.gcp_enabled ? module.gcp_deployment[0].cluster_location : null
}

# Global Load Balancer Outputs
output "global_load_balancer" {
  description = "Global load balancer information"
  value = local.global_load_balancer_enabled ? {
    endpoint           = module.global_load_balancer[0].load_balancer_endpoint
    domain_name        = module.global_load_balancer[0].domain_name
    primary_cloud      = module.global_load_balancer[0].primary_cloud
    dns_zone_id        = module.global_load_balancer[0].dns_zone_id
    ssl_certificate_arn = module.global_load_balancer[0].ssl_certificate_arn
    backend_summary    = module.global_load_balancer[0].backend_summary
    traffic_distribution = module.global_load_balancer[0].traffic_distribution_config
  } : null
}

output "global_endpoint" {
  description = "Global load balancer endpoint"
  value       = local.global_load_balancer_enabled ? module.global_load_balancer[0].load_balancer_endpoint : null
}

output "domain_name" {
  description = "Domain name for the global load balancer"
  value       = local.global_load_balancer_enabled ? module.global_load_balancer[0].domain_name : null
}

# Cross-Cloud Networking Outputs
output "cross_cloud_networking" {
  description = "Cross-cloud networking information"
  value = var.cross_cloud_networking.enabled ? {
    enabled                 = true
    vpn_connections_enabled = var.cross_cloud_networking.enable_vpn_connections
    private_connectivity    = var.cross_cloud_networking.enable_private_connectivity
    shared_services_cidr    = var.cross_cloud_networking.shared_services_cidr
    # Additional networking details would be added by the networking module
  } : null
}

# Monitoring Outputs
output "monitoring" {
  description = "Multi-cloud monitoring information"
  value = var.monitoring.enabled ? {
    enabled              = true
    primary_cloud        = local.primary_cloud
    prometheus_enabled   = var.monitoring.prometheus_enabled
    grafana_enabled      = var.monitoring.grafana_enabled
    alertmanager_enabled = var.monitoring.alertmanager_enabled
    logging_enabled      = var.monitoring.logging_enabled
    tracing_enabled      = var.monitoring.tracing_enabled
    # Additional monitoring details would be added by the monitoring module
  } : null
}

# Disaster Recovery Outputs
output "disaster_recovery" {
  description = "Disaster recovery information"
  value = var.disaster_recovery.enabled ? {
    enabled                   = true
    backup_schedule          = var.disaster_recovery.backup_schedule
    backup_retention_days    = var.disaster_recovery.backup_retention_days
    cross_region_backup      = var.disaster_recovery.enable_cross_region_backup
    dr_testing_enabled       = var.disaster_recovery.enable_dr_testing
    rpo_hours               = var.disaster_recovery.rpo_hours
    rto_hours               = var.disaster_recovery.rto_hours
    # Additional DR details would be added by the DR module
  } : null
}

# Security Outputs
output "security" {
  description = "Security configuration information"
  value = var.security.enabled ? {
    enabled                      = true
    pod_security_standards       = var.security.enable_pod_security_standards
    network_policies            = var.security.enable_network_policies
    rbac_enabled                = var.security.enable_rbac
    compliance_frameworks       = var.security.compliance_frameworks
    external_secrets_enabled    = var.security.enable_external_secrets
    vulnerability_scanning       = var.security.enable_vulnerability_scanning
    # Additional security details would be added by the security module
  } : null
}

# Cost Optimization Outputs
output "cost_optimization" {
  description = "Cost optimization information"
  value = var.cost_optimization.enabled ? {
    enabled                        = true
    spot_instances_enabled         = var.cost_optimization.enable_spot_instances
    cluster_autoscaling_enabled    = var.cost_optimization.enable_cluster_autoscaling
    vpa_enabled                   = var.cost_optimization.enable_vertical_pod_autoscaling
    resource_recommendations      = var.cost_optimization.enable_resource_recommendations
    cost_monitoring_enabled       = var.cost_optimization.enable_cost_monitoring
    reserved_instance_coverage    = var.cost_optimization.reserved_instance_coverage
    # Additional cost optimization details would be added by the cost module
  } : null
}

# TrustformeRS Configuration Outputs
output "trustformers_config" {
  description = "TrustformeRS Serve configuration"
  value = {
    namespace         = var.trustformers_config.namespace
    helm_chart_path   = var.trustformers_config.helm_chart_path
    helm_chart_version = var.trustformers_config.helm_chart_version
    image_repository  = var.trustformers_config.image_repository
    image_tag         = var.trustformers_config.image_tag
    replicas          = var.trustformers_config.replicas
    resources         = var.trustformers_config.resources
  }
}

# Deployment Status
output "deployment_status" {
  description = "Overall deployment status"
  value = {
    total_clouds_enabled    = length([for cloud, config in var.clouds : cloud if config.enabled])
    clouds_enabled         = [for cloud, config in var.clouds : cloud if config.enabled]
    global_lb_enabled      = local.global_load_balancer_enabled
    cross_cloud_networking = var.cross_cloud_networking.enabled
    monitoring_enabled     = var.monitoring.enabled
    security_enabled       = var.security.enabled
    cost_optimization      = var.cost_optimization.enabled
    disaster_recovery      = var.disaster_recovery.enabled
  }
}

# Connection Information
output "connection_info" {
  description = "Connection information for applications"
  value = {
    # Global endpoint (if available)
    global_endpoint = local.global_load_balancer_enabled ? module.global_load_balancer[0].load_balancer_endpoint : null
    
    # Individual cloud endpoints
    aws_endpoint   = local.aws_enabled ? module.aws_deployment[0].cluster_endpoint : null
    azure_endpoint = local.azure_enabled ? module.azure_deployment[0].cluster_endpoint : null
    gcp_endpoint   = local.gcp_enabled ? module.gcp_deployment[0].cluster_endpoint : null
    
    # Kubernetes namespaces
    trustformers_namespace = var.trustformers_config.namespace
    
    # Service discovery
    service_discovery = {
      cluster_domain = "cluster.local"
      dns_suffix    = var.trustformers_config.namespace
    }
    
    # Load balancing
    load_balancing = {
      method = local.global_load_balancer_enabled ? var.global_load_balancer.traffic_distribution.method : "round-robin"
      clouds_distribution = local.global_load_balancer_enabled ? {
        aws   = var.global_load_balancer.traffic_distribution.aws_weight
        azure = var.global_load_balancer.traffic_distribution.azure_weight
        gcp   = var.global_load_balancer.traffic_distribution.gcp_weight
      } : null
    }
  }
}

# Resource Summary
output "resource_summary" {
  description = "Summary of created resources across clouds"
  value = {
    aws_resources = local.aws_enabled ? {
      eks_cluster    = 1
      node_groups   = length(var.clouds.aws.node_groups)
      helm_releases = length([
        for release, enabled in {
          aws_load_balancer_controller = var.clouds.aws.enable_load_balancer_controller
          cluster_autoscaler          = var.clouds.aws.enable_cluster_autoscaler
          metrics_server              = var.clouds.aws.enable_metrics_server
          trustformers_serve          = var.clouds.aws.install_trustformers_serve
        } : release if enabled
      ])
    } : {}
    
    azure_resources = local.azure_enabled ? {
      aks_cluster   = 1
      node_pools    = length(var.clouds.azure.additional_node_pools) + 1
      helm_releases = length([
        for release, enabled in {
          cert_manager       = true
          nginx_ingress     = true
          trustformers_serve = var.clouds.azure.install_trustformers_serve
        } : release if enabled
      ])
    } : {}
    
    gcp_resources = local.gcp_enabled ? {
      gke_cluster   = 1
      node_pools    = length(var.clouds.gcp.additional_node_pools) + 1
      helm_releases = length([
        for release, enabled in {
          cert_manager       = true
          nginx_ingress     = true
          trustformers_serve = var.clouds.gcp.install_trustformers_serve
        } : release if enabled
      ])
    } : {}
    
    global_resources = {
      load_balancer         = local.global_load_balancer_enabled ? 1 : 0
      cross_cloud_networking = var.cross_cloud_networking.enabled ? 1 : 0
      monitoring_stack      = var.monitoring.enabled ? 1 : 0
      security_policies     = var.security.enabled ? 1 : 0
    }
  }
}

# Cost Estimation
output "estimated_costs" {
  description = "Estimated monthly costs for the multi-cloud deployment"
  value = {
    aws_costs = local.aws_enabled ? {
      eks_cluster = "~$72/month for control plane"
      node_groups = "Varies by instance types and usage"
      data_transfer = "Varies by traffic volume"
    } : {}
    
    azure_costs = local.azure_enabled ? {
      aks_cluster = var.clouds.azure.sku_tier == "Standard" ? "~$73/month for Standard tier" : "Free tier"
      node_pools = "Varies by VM sizes and usage"
      data_transfer = "Varies by traffic volume"
    } : {}
    
    gcp_costs = local.gcp_enabled ? {
      gke_cluster = "~$73/month for standard cluster management"
      node_pools = "Varies by machine types and usage"
      data_transfer = "Varies by traffic volume"
    } : {}
    
    global_costs = {
      load_balancer = local.global_load_balancer_enabled ? "Varies by primary cloud provider" : "Not deployed"
      dns_costs    = local.global_load_balancer_enabled ? "~$0.50/zone + query charges" : "Not applicable"
      monitoring   = var.monitoring.enabled ? "Varies by metrics volume" : "Not deployed"
    }
    
    total_estimate = "Costs depend on instance types, traffic volume, and enabled features"
    optimization_note = var.cost_optimization.enabled ? "Cost optimization features are enabled" : "Consider enabling cost optimization"
  }
}

# Health Check Endpoints
output "health_check_endpoints" {
  description = "Health check endpoints for each deployment"
  value = {
    aws_health_check   = local.aws_enabled ? "${module.aws_deployment[0].cluster_endpoint}/health" : null
    azure_health_check = local.azure_enabled ? "${module.azure_deployment[0].cluster_endpoint}/health" : null
    gcp_health_check   = local.gcp_enabled ? "${module.gcp_deployment[0].cluster_endpoint}/health" : null
    global_health_check = local.global_load_balancer_enabled ? "${module.global_load_balancer[0].load_balancer_endpoint}/health" : null
  }
}

# Deployment Metadata
output "deployment_metadata" {
  description = "Deployment metadata and configuration"
  value = {
    deployment_id     = random_id.deployment.hex
    creation_time    = timestamp()
    terraform_version = ">=1.5"
    module_version   = "1.0.0"
    tags             = local.common_tags
    
    configuration_hash = md5(jsonencode({
      clouds                = var.clouds
      global_load_balancer  = var.global_load_balancer
      monitoring           = var.monitoring
      security             = var.security
      cost_optimization    = var.cost_optimization
    }))
  }
}