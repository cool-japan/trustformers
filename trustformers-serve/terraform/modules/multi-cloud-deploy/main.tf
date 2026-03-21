# Multi-Cloud Deployment Module for TrustformeRS Serve
# Supports AWS, Azure, and Google Cloud Platform

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
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

locals {
  deployment_name = "${var.project_name}-${var.environment}"
  
  common_tags = merge(var.tags, {
    Project      = var.project_name
    Environment  = var.environment
    ManagedBy    = "terraform"
    Service      = "trustformers-serve"
    MultiCloud   = "enabled"
  })
  
  # Cloud-specific configurations
  aws_enabled   = var.clouds.aws.enabled
  azure_enabled = var.clouds.azure.enabled
  gcp_enabled   = var.clouds.gcp.enabled
  
  # Primary cloud for DNS and global resources
  primary_cloud = var.primary_cloud
  
  # Load balancer configurations
  global_load_balancer_enabled = var.global_load_balancer.enabled
}

# Random resources for unique naming
resource "random_id" "deployment" {
  byte_length = 4
}

#############################################
# AWS Deployment
#############################################
module "aws_deployment" {
  count  = local.aws_enabled ? 1 : 0
  source = "../aws-eks-complete"
  
  project_name = var.project_name
  environment  = var.environment
  tags         = local.common_tags
  
  # VPC Configuration
  vpc_id                 = var.clouds.aws.vpc_id
  subnet_ids            = var.clouds.aws.subnet_ids
  node_subnet_ids       = var.clouds.aws.node_subnet_ids
  
  # EKS Configuration
  kubernetes_version        = var.clouds.aws.kubernetes_version
  endpoint_private_access   = var.clouds.aws.endpoint_private_access
  endpoint_public_access    = var.clouds.aws.endpoint_public_access
  public_access_cidrs       = var.clouds.aws.public_access_cidrs
  
  # Node Groups
  node_groups = var.clouds.aws.node_groups
  
  # Add-ons
  addons = var.clouds.aws.addons
  
  # Features
  enable_aws_load_balancer_controller = var.clouds.aws.enable_load_balancer_controller
  enable_cluster_autoscaler          = var.clouds.aws.enable_cluster_autoscaler
  enable_metrics_server              = var.clouds.aws.enable_metrics_server
  
  # TrustformeRS Deployment
  install_trustformers_serve    = var.clouds.aws.install_trustformers_serve
  trustformers_namespace        = var.trustformers_config.namespace
  trustformers_helm_chart_path  = var.trustformers_config.helm_chart_path
  trustformers_helm_values      = var.clouds.aws.trustformers_helm_values
}

#############################################
# Azure Deployment
#############################################
module "azure_deployment" {
  count  = local.azure_enabled ? 1 : 0
  source = "./azure-aks"
  
  project_name = var.project_name
  environment  = var.environment
  tags         = local.common_tags
  
  # Resource Group
  resource_group_name = var.clouds.azure.resource_group_name
  location           = var.clouds.azure.location
  
  # Virtual Network
  vnet_name                = var.clouds.azure.vnet_name
  subnet_name             = var.clouds.azure.subnet_name
  
  # AKS Configuration
  kubernetes_version       = var.clouds.azure.kubernetes_version
  sku_tier                = var.clouds.azure.sku_tier
  private_cluster_enabled  = var.clouds.azure.private_cluster_enabled
  
  # Node Pools
  default_node_pool = var.clouds.azure.default_node_pool
  additional_node_pools = var.clouds.azure.additional_node_pools
  
  # Features
  enable_auto_scaling       = var.clouds.azure.enable_auto_scaling
  enable_pod_security_policy = var.clouds.azure.enable_pod_security_policy
  enable_rbac              = var.clouds.azure.enable_rbac
  
  # Azure-specific features
  enable_azure_policy      = var.clouds.azure.enable_azure_policy
  enable_oms_agent        = var.clouds.azure.enable_oms_agent
  enable_ingress_application_gateway = var.clouds.azure.enable_ingress_application_gateway
  
  # TrustformeRS Deployment
  install_trustformers_serve = var.clouds.azure.install_trustformers_serve
  trustformers_helm_values   = var.clouds.azure.trustformers_helm_values
}

#############################################
# Google Cloud Platform Deployment
#############################################
module "gcp_deployment" {
  count  = local.gcp_enabled ? 1 : 0
  source = "./gcp-gke"
  
  project_name = var.project_name
  environment  = var.environment
  tags         = local.common_tags
  
  # Project Configuration
  project_id = var.clouds.gcp.project_id
  region     = var.clouds.gcp.region
  zones      = var.clouds.gcp.zones
  
  # Network Configuration
  network_name    = var.clouds.gcp.network_name
  subnet_name     = var.clouds.gcp.subnet_name
  
  # GKE Configuration
  kubernetes_version       = var.clouds.gcp.kubernetes_version
  release_channel         = var.clouds.gcp.release_channel
  private_cluster         = var.clouds.gcp.private_cluster
  master_ipv4_cidr_block  = var.clouds.gcp.master_ipv4_cidr_block
  
  # Node Pools
  default_node_pool = var.clouds.gcp.default_node_pool
  additional_node_pools = var.clouds.gcp.additional_node_pools
  
  # Features
  enable_network_policy     = var.clouds.gcp.enable_network_policy
  enable_pod_security_policy = var.clouds.gcp.enable_pod_security_policy
  enable_workload_identity  = var.clouds.gcp.enable_workload_identity
  enable_gcp_filestore_csi  = var.clouds.gcp.enable_gcp_filestore_csi
  
  # GCP-specific features
  enable_google_cloud_logging   = var.clouds.gcp.enable_google_cloud_logging
  enable_google_cloud_monitoring = var.clouds.gcp.enable_google_cloud_monitoring
  enable_istio                  = var.clouds.gcp.enable_istio
  
  # TrustformeRS Deployment
  install_trustformers_serve = var.clouds.gcp.install_trustformers_serve
  trustformers_helm_values   = var.clouds.gcp.trustformers_helm_values
}

#############################################
# Global Load Balancer (Primary Cloud)
#############################################
module "global_load_balancer" {
  count  = local.global_load_balancer_enabled ? 1 : 0
  source = "./global-load-balancer"
  
  project_name = var.project_name
  environment  = var.environment
  tags         = local.common_tags
  
  primary_cloud = local.primary_cloud
  
  # Backend configurations from each cloud
  aws_backend = local.aws_enabled ? {
    enabled    = true
    endpoint   = module.aws_deployment[0].cluster_endpoint
    region     = var.clouds.aws.region
    health_check_path = "/health"
  } : { enabled = false }
  
  azure_backend = local.azure_enabled ? {
    enabled    = true
    endpoint   = module.azure_deployment[0].cluster_endpoint
    region     = var.clouds.azure.location
    health_check_path = "/health"
  } : { enabled = false }
  
  gcp_backend = local.gcp_enabled ? {
    enabled    = true
    endpoint   = module.gcp_deployment[0].cluster_endpoint
    region     = var.clouds.gcp.region
    health_check_path = "/health"
  } : { enabled = false }
  
  # Load balancer configuration
  dns_zone_name = var.global_load_balancer.dns_zone_name
  domain_name   = var.global_load_balancer.domain_name
  ssl_policy    = var.global_load_balancer.ssl_policy
  
  # Traffic distribution
  traffic_distribution = var.global_load_balancer.traffic_distribution
  failover_policy     = var.global_load_balancer.failover_policy
}

#############################################
# Cross-Cloud Networking (if enabled)
#############################################
module "cross_cloud_networking" {
  count  = var.cross_cloud_networking.enabled ? 1 : 0
  source = "./cross-cloud-networking"
  
  project_name = var.project_name
  environment  = var.environment
  tags         = local.common_tags
  
  # Cloud configurations
  aws_config = local.aws_enabled ? {
    vpc_id     = var.clouds.aws.vpc_id
    cidr_block = var.clouds.aws.vpc_cidr
    region     = var.clouds.aws.region
  } : null
  
  azure_config = local.azure_enabled ? {
    vnet_name           = var.clouds.azure.vnet_name
    resource_group_name = var.clouds.azure.resource_group_name
    address_space       = var.clouds.azure.vnet_address_space
    location           = var.clouds.azure.location
  } : null
  
  gcp_config = local.gcp_enabled ? {
    network_name = var.clouds.gcp.network_name
    project_id   = var.clouds.gcp.project_id
    region       = var.clouds.gcp.region
  } : null
  
  # Networking configuration
  enable_vpn_connections      = var.cross_cloud_networking.enable_vpn_connections
  enable_private_connectivity = var.cross_cloud_networking.enable_private_connectivity
  shared_services_cidr       = var.cross_cloud_networking.shared_services_cidr
}

#############################################
# Multi-Cloud Monitoring
#############################################
module "multi_cloud_monitoring" {
  count  = var.monitoring.enabled ? 1 : 0
  source = "./multi-cloud-monitoring"
  
  project_name = var.project_name
  environment  = var.environment
  tags         = local.common_tags
  
  primary_cloud = local.primary_cloud
  
  # Cloud endpoints for monitoring
  aws_cluster_endpoint   = local.aws_enabled ? module.aws_deployment[0].cluster_endpoint : ""
  azure_cluster_endpoint = local.azure_enabled ? module.azure_deployment[0].cluster_endpoint : ""
  gcp_cluster_endpoint   = local.gcp_enabled ? module.gcp_deployment[0].cluster_endpoint : ""
  
  # Monitoring configuration
  prometheus_enabled    = var.monitoring.prometheus_enabled
  grafana_enabled      = var.monitoring.grafana_enabled
  alertmanager_enabled = var.monitoring.alertmanager_enabled
  
  # Centralized logging
  logging_enabled      = var.monitoring.logging_enabled
  log_aggregation_type = var.monitoring.log_aggregation_type
  
  # Distributed tracing
  tracing_enabled = var.monitoring.tracing_enabled
  jaeger_enabled  = var.monitoring.jaeger_enabled
}

#############################################
# Disaster Recovery and Backup
#############################################
module "disaster_recovery" {
  count  = var.disaster_recovery.enabled ? 1 : 0
  source = "./disaster-recovery"
  
  project_name = var.project_name
  environment  = var.environment
  tags         = local.common_tags
  
  # Cloud configurations for backup
  aws_config = local.aws_enabled ? {
    cluster_name = module.aws_deployment[0].cluster_name
    region       = var.clouds.aws.region
  } : null
  
  azure_config = local.azure_enabled ? {
    cluster_name        = module.azure_deployment[0].cluster_name
    resource_group_name = var.clouds.azure.resource_group_name
    location           = var.clouds.azure.location
  } : null
  
  gcp_config = local.gcp_enabled ? {
    cluster_name = module.gcp_deployment[0].cluster_name
    project_id   = var.clouds.gcp.project_id
    region       = var.clouds.gcp.region
  } : null
  
  # Backup configuration
  backup_schedule       = var.disaster_recovery.backup_schedule
  backup_retention_days = var.disaster_recovery.backup_retention_days
  
  # Cross-region replication
  enable_cross_region_backup = var.disaster_recovery.enable_cross_region_backup
  backup_regions            = var.disaster_recovery.backup_regions
  
  # Disaster recovery testing
  enable_dr_testing = var.disaster_recovery.enable_dr_testing
  dr_test_schedule  = var.disaster_recovery.dr_test_schedule
}

#############################################
# Security and Compliance
#############################################
module "security_compliance" {
  count  = var.security.enabled ? 1 : 0
  source = "./security-compliance"
  
  project_name = var.project_name
  environment  = var.environment
  tags         = local.common_tags
  
  # Cloud configurations
  aws_enabled   = local.aws_enabled
  azure_enabled = local.azure_enabled
  gcp_enabled   = local.gcp_enabled
  
  # Security features
  enable_pod_security_standards = var.security.enable_pod_security_standards
  enable_network_policies       = var.security.enable_network_policies
  enable_rbac                  = var.security.enable_rbac
  
  # Compliance requirements
  compliance_frameworks = var.security.compliance_frameworks
  
  # Secret management
  enable_external_secrets = var.security.enable_external_secrets
  secret_stores          = var.security.secret_stores
  
  # Runtime security
  enable_falco           = var.security.enable_falco
  enable_opa_gatekeeper  = var.security.enable_opa_gatekeeper
}

#############################################
# Cost Optimization
#############################################
module "cost_optimization" {
  count  = var.cost_optimization.enabled ? 1 : 0
  source = "./cost-optimization"
  
  project_name = var.project_name
  environment  = var.environment
  tags         = local.common_tags
  
  # Cloud configurations
  aws_enabled   = local.aws_enabled
  azure_enabled = local.azure_enabled
  gcp_enabled   = local.gcp_enabled
  
  # Cost optimization features
  enable_spot_instances        = var.cost_optimization.enable_spot_instances
  enable_cluster_autoscaling   = var.cost_optimization.enable_cluster_autoscaling
  enable_vertical_pod_autoscaling = var.cost_optimization.enable_vertical_pod_autoscaling
  
  # Resource optimization
  enable_resource_recommendations = var.cost_optimization.enable_resource_recommendations
  enable_unused_resource_cleanup  = var.cost_optimization.enable_unused_resource_cleanup
  
  # Cost monitoring
  enable_cost_monitoring = var.cost_optimization.enable_cost_monitoring
  cost_budgets          = var.cost_optimization.cost_budgets
}