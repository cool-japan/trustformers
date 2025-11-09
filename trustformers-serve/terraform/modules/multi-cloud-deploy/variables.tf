# Variables for Multi-Cloud Deploy module

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Primary Cloud Configuration
variable "primary_cloud" {
  description = "Primary cloud provider for DNS and global resources (aws, azure, gcp)"
  type        = string
  default     = "aws"
  validation {
    condition     = contains(["aws", "azure", "gcp"], var.primary_cloud)
    error_message = "Primary cloud must be one of: aws, azure, gcp."
  }
}

# Cloud Provider Configurations
variable "clouds" {
  description = "Configuration for each cloud provider"
  type = object({
    aws = object({
      enabled                        = optional(bool, false)
      region                        = optional(string, "us-east-1")
      vpc_id                        = optional(string, "")
      subnet_ids                    = optional(list(string), [])
      node_subnet_ids               = optional(list(string), [])
      vpc_cidr                      = optional(string, "10.0.0.0/16")
      kubernetes_version            = optional(string, "1.27")
      endpoint_private_access       = optional(bool, true)
      endpoint_public_access        = optional(bool, true)
      public_access_cidrs           = optional(list(string), ["0.0.0.0/0"])
      node_groups                   = optional(map(any), {})
      addons                        = optional(map(any), {})
      enable_load_balancer_controller = optional(bool, true)
      enable_cluster_autoscaler     = optional(bool, true)
      enable_metrics_server         = optional(bool, true)
      install_trustformers_serve    = optional(bool, false)
      trustformers_helm_values      = optional(string, "")
    })
    azure = object({
      enabled                           = optional(bool, false)
      location                         = optional(string, "East US")
      resource_group_name              = optional(string, "")
      vnet_name                        = optional(string, "")
      subnet_name                      = optional(string, "")
      vnet_address_space               = optional(list(string), ["10.1.0.0/16"])
      kubernetes_version               = optional(string, "1.27.7")
      sku_tier                        = optional(string, "Standard")
      private_cluster_enabled          = optional(bool, false)
      default_node_pool               = optional(map(any), {})
      additional_node_pools           = optional(map(any), {})
      enable_auto_scaling             = optional(bool, true)
      enable_pod_security_policy      = optional(bool, false)
      enable_rbac                     = optional(bool, true)
      enable_azure_policy             = optional(bool, true)
      enable_oms_agent                = optional(bool, true)
      enable_ingress_application_gateway = optional(bool, false)
      install_trustformers_serve      = optional(bool, false)
      trustformers_helm_values        = optional(string, "")
    })
    gcp = object({
      enabled                         = optional(bool, false)
      project_id                      = optional(string, "")
      region                          = optional(string, "us-central1")
      zones                          = optional(list(string), [])
      network_name                   = optional(string, "")
      subnet_name                    = optional(string, "")
      kubernetes_version             = optional(string, "1.27.7-gke.1121000")
      release_channel                = optional(string, "REGULAR")
      private_cluster                = optional(map(any), {})
      master_ipv4_cidr_block         = optional(string, "172.16.0.0/28")
      default_node_pool              = optional(map(any), {})
      additional_node_pools          = optional(map(any), {})
      enable_network_policy          = optional(bool, true)
      enable_pod_security_policy     = optional(bool, false)
      enable_workload_identity       = optional(bool, true)
      enable_gcp_filestore_csi       = optional(bool, false)
      enable_google_cloud_logging    = optional(bool, true)
      enable_google_cloud_monitoring = optional(bool, true)
      enable_istio                   = optional(bool, false)
      install_trustformers_serve     = optional(bool, false)
      trustformers_helm_values       = optional(string, "")
    })
  })
  default = {
    aws = {
      enabled = false
      region  = "us-east-1"
    }
    azure = {
      enabled  = false
      location = "East US"
    }
    gcp = {
      enabled = false
      region  = "us-central1"
    }
  }
}

# TrustformeRS Configuration
variable "trustformers_config" {
  description = "TrustformeRS Serve configuration"
  type = object({
    namespace         = optional(string, "trustformers")
    helm_chart_path   = optional(string, "../../../helm")
    helm_chart_version = optional(string, "0.1.0")
    image_repository  = optional(string, "trustformers/serve")
    image_tag         = optional(string, "latest")
    replicas          = optional(number, 3)
    resources = optional(object({
      requests = optional(object({
        cpu    = optional(string, "500m")
        memory = optional(string, "1Gi")
      }), {})
      limits = optional(object({
        cpu    = optional(string, "2")
        memory = optional(string, "4Gi")
      }), {})
    }), {})
  })
  default = {
    namespace         = "trustformers"
    helm_chart_path   = "../../../helm"
    helm_chart_version = "0.1.0"
    image_repository  = "trustformers/serve"
    image_tag         = "latest"
    replicas          = 3
    resources = {
      requests = {
        cpu    = "500m"
        memory = "1Gi"
      }
      limits = {
        cpu    = "2"
        memory = "4Gi"
      }
    }
  }
}

# Global Load Balancer Configuration
variable "global_load_balancer" {
  description = "Global load balancer configuration"
  type = object({
    enabled               = optional(bool, false)
    domain_name          = optional(string, "")
    dns_zone_name        = optional(string, "")
    ssl_policy           = optional(string, "ELBSecurityPolicy-TLS-1-2-2017-01")
    enable_waf           = optional(bool, false)
    enable_ddos_protection = optional(bool, true)
    enable_access_logging = optional(bool, true)
    traffic_distribution = optional(object({
      method              = optional(string, "Weighted")
      aws_weight         = optional(number, 33)
      azure_weight       = optional(number, 33)
      gcp_weight         = optional(number, 34)
      aws_geo_mappings   = optional(list(string), ["US"])
      azure_geo_mappings = optional(list(string), ["EU"])
      gcp_geo_mappings   = optional(list(string), ["AS"])
    }), {})
    failover_policy = optional(object({
      aws_priority    = optional(number, 1)
      azure_priority  = optional(number, 2)
      gcp_priority    = optional(number, 3)
      enable_failover = optional(bool, true)
    }), {})
  })
  default = {
    enabled = false
  }
}

# Cross-Cloud Networking Configuration
variable "cross_cloud_networking" {
  description = "Cross-cloud networking configuration"
  type = object({
    enabled                     = optional(bool, false)
    enable_vpn_connections      = optional(bool, false)
    enable_private_connectivity = optional(bool, false)
    shared_services_cidr        = optional(string, "192.168.0.0/16")
    encryption_in_transit       = optional(bool, true)
    bandwidth_mbps              = optional(number, 1000)
  })
  default = {
    enabled = false
  }
}

# Multi-Cloud Monitoring Configuration
variable "monitoring" {
  description = "Multi-cloud monitoring configuration"
  type = object({
    enabled              = optional(bool, true)
    prometheus_enabled   = optional(bool, true)
    grafana_enabled      = optional(bool, true)
    alertmanager_enabled = optional(bool, true)
    logging_enabled      = optional(bool, true)
    log_aggregation_type = optional(string, "elasticsearch")
    tracing_enabled      = optional(bool, true)
    jaeger_enabled       = optional(bool, true)
    retention_days       = optional(number, 30)
  })
  default = {
    enabled              = true
    prometheus_enabled   = true
    grafana_enabled      = true
    alertmanager_enabled = true
    logging_enabled      = true
    log_aggregation_type = "elasticsearch"
    tracing_enabled      = true
    jaeger_enabled       = true
    retention_days       = 30
  }
}

# Disaster Recovery Configuration
variable "disaster_recovery" {
  description = "Disaster recovery configuration"
  type = object({
    enabled                    = optional(bool, false)
    backup_schedule           = optional(string, "0 2 * * *")
    backup_retention_days     = optional(number, 30)
    enable_cross_region_backup = optional(bool, true)
    backup_regions            = optional(list(string), [])
    enable_dr_testing         = optional(bool, false)
    dr_test_schedule          = optional(string, "0 4 * * 0")
    rpo_hours                 = optional(number, 4)
    rto_hours                 = optional(number, 1)
  })
  default = {
    enabled = false
  }
}

# Security and Compliance Configuration
variable "security" {
  description = "Security and compliance configuration"
  type = object({
    enabled                       = optional(bool, true)
    enable_pod_security_standards = optional(bool, true)
    enable_network_policies       = optional(bool, true)
    enable_rbac                  = optional(bool, true)
    compliance_frameworks        = optional(list(string), ["SOC2", "ISO27001"])
    enable_external_secrets      = optional(bool, true)
    secret_stores               = optional(list(string), ["aws-secrets-manager", "azure-key-vault", "gcp-secret-manager"])
    enable_falco                = optional(bool, false)
    enable_opa_gatekeeper       = optional(bool, false)
    enable_vulnerability_scanning = optional(bool, true)
    enable_image_scanning        = optional(bool, true)
  })
  default = {
    enabled                       = true
    enable_pod_security_standards = true
    enable_network_policies       = true
    enable_rbac                  = true
    compliance_frameworks        = ["SOC2", "ISO27001"]
    enable_external_secrets      = true
    secret_stores               = ["aws-secrets-manager", "azure-key-vault", "gcp-secret-manager"]
    enable_falco                = false
    enable_opa_gatekeeper       = false
    enable_vulnerability_scanning = true
    enable_image_scanning        = true
  }
}

# Cost Optimization Configuration
variable "cost_optimization" {
  description = "Cost optimization configuration"
  type = object({
    enabled                         = optional(bool, true)
    enable_spot_instances           = optional(bool, false)
    enable_cluster_autoscaling      = optional(bool, true)
    enable_vertical_pod_autoscaling = optional(bool, false)
    enable_resource_recommendations = optional(bool, true)
    enable_unused_resource_cleanup  = optional(bool, true)
    enable_cost_monitoring          = optional(bool, true)
    cost_budgets = optional(list(object({
      name           = string
      amount         = number
      time_unit      = string
      threshold_type = string
      threshold_percentage = number
    })), [])
    reserved_instance_coverage = optional(number, 70)
    savings_plans_coverage     = optional(number, 80)
  })
  default = {
    enabled                         = true
    enable_spot_instances           = false
    enable_cluster_autoscaling      = true
    enable_vertical_pod_autoscaling = false
    enable_resource_recommendations = true
    enable_unused_resource_cleanup  = true
    enable_cost_monitoring          = true
    cost_budgets                   = []
    reserved_instance_coverage     = 70
    savings_plans_coverage         = 80
  }
}

# Data and Storage Configuration
variable "data_storage" {
  description = "Data and storage configuration"
  type = object({
    enable_persistent_storage = optional(bool, true)
    storage_classes          = optional(list(string), ["ssd", "standard"])
    enable_backup           = optional(bool, true)
    backup_schedule         = optional(string, "0 3 * * *")
    retention_policy        = optional(string, "30d")
    enable_encryption       = optional(bool, true)
    enable_cross_region_replication = optional(bool, false)
  })
  default = {
    enable_persistent_storage = true
    storage_classes          = ["ssd", "standard"]
    enable_backup           = true
    backup_schedule         = "0 3 * * *"
    retention_policy        = "30d"
    enable_encryption       = true
    enable_cross_region_replication = false
  }
}

# Service Mesh Configuration
variable "service_mesh" {
  description = "Service mesh configuration"
  type = object({
    enabled         = optional(bool, false)
    provider        = optional(string, "istio")
    mtls_enabled    = optional(bool, true)
    observability   = optional(bool, true)
    traffic_management = optional(bool, true)
    security_policies  = optional(bool, true)
  })
  default = {
    enabled         = false
    provider        = "istio"
    mtls_enabled    = true
    observability   = true
    traffic_management = true
    security_policies  = true
  }
}

# CI/CD Integration Configuration
variable "cicd_integration" {
  description = "CI/CD integration configuration"
  type = object({
    enabled               = optional(bool, false)
    provider             = optional(string, "github-actions")
    auto_deploy_enabled  = optional(bool, false)
    environments         = optional(list(string), ["dev", "staging", "prod"])
    approval_required    = optional(bool, true)
    rollback_enabled     = optional(bool, true)
    canary_deployment    = optional(bool, false)
    blue_green_deployment = optional(bool, false)
  })
  default = {
    enabled               = false
    provider             = "github-actions"
    auto_deploy_enabled  = false
    environments         = ["dev", "staging", "prod"]
    approval_required    = true
    rollback_enabled     = true
    canary_deployment    = false
    blue_green_deployment = false
  }
}

# External Integrations
variable "external_integrations" {
  description = "External service integrations"
  type = object({
    enable_external_dns      = optional(bool, false)
    enable_cert_manager      = optional(bool, true)
    enable_external_secrets  = optional(bool, true)
    enable_keda_autoscaling  = optional(bool, false)
    enable_argocd           = optional(bool, false)
    enable_flux             = optional(bool, false)
    enable_prometheus_operator = optional(bool, true)
  })
  default = {
    enable_external_dns      = false
    enable_cert_manager      = true
    enable_external_secrets  = true
    enable_keda_autoscaling  = false
    enable_argocd           = false
    enable_flux             = false
    enable_prometheus_operator = true
  }
}

# Performance and Scaling Configuration
variable "performance_scaling" {
  description = "Performance and scaling configuration"
  type = object({
    enable_hpa                = optional(bool, true)
    enable_vpa                = optional(bool, false)
    enable_cluster_autoscaler = optional(bool, true)
    min_nodes_per_cloud      = optional(number, 1)
    max_nodes_per_cloud      = optional(number, 10)
    target_cpu_utilization   = optional(number, 70)
    target_memory_utilization = optional(number, 80)
    scale_down_delay         = optional(string, "10m")
    scale_up_delay           = optional(string, "3m")
  })
  default = {
    enable_hpa                = true
    enable_vpa                = false
    enable_cluster_autoscaler = true
    min_nodes_per_cloud      = 1
    max_nodes_per_cloud      = 10
    target_cpu_utilization   = 70
    target_memory_utilization = 80
    scale_down_delay         = "10m"
    scale_up_delay           = "3m"
  }
}

# Advanced Features
variable "advanced_features" {
  description = "Advanced features configuration"
  type = object({
    enable_chaos_engineering = optional(bool, false)
    enable_feature_flags     = optional(bool, false)
    enable_ab_testing        = optional(bool, false)
    enable_ml_inference_optimization = optional(bool, true)
    enable_model_versioning  = optional(bool, true)
    enable_model_a_b_testing = optional(bool, false)
    enable_model_monitoring  = optional(bool, true)
  })
  default = {
    enable_chaos_engineering = false
    enable_feature_flags     = false
    enable_ab_testing        = false
    enable_ml_inference_optimization = true
    enable_model_versioning  = true
    enable_model_a_b_testing = false
    enable_model_monitoring  = true
  }
}