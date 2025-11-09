# Variables for Azure AKS module

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

# Resource Group Configuration
variable "resource_group_name" {
  description = "Name of the resource group (will create if empty)"
  type        = string
  default     = ""
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "East US"
}

# Virtual Network Configuration
variable "vnet_name" {
  description = "Name of the virtual network (will create if empty)"
  type        = string
  default     = ""
}

variable "subnet_name" {
  description = "Name of the subnet for AKS (will create if empty)"
  type        = string
  default     = ""
}

# AKS Cluster Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for the AKS cluster"
  type        = string
  default     = "1.27.7"
}

variable "sku_tier" {
  description = "SKU tier for the AKS cluster"
  type        = string
  default     = "Standard"
  validation {
    condition     = contains(["Free", "Standard"], var.sku_tier)
    error_message = "SKU tier must be either 'Free' or 'Standard'."
  }
}

variable "private_cluster_enabled" {
  description = "Enable private cluster"
  type        = bool
  default     = false
}

variable "api_server_authorized_ip_ranges" {
  description = "Authorized IP ranges for API server access"
  type        = list(string)
  default     = []
}

# Default Node Pool Configuration
variable "default_node_pool" {
  description = "Configuration for the default node pool"
  type = object({
    name            = string
    node_count      = number
    vm_size         = string
    min_count       = optional(number, 1)
    max_count       = optional(number, 5)
    max_pods        = optional(number, 110)
    os_disk_size_gb = optional(number, 128)
    os_disk_type    = optional(string, "Managed")
    node_labels     = optional(map(string), {})
    tags            = optional(map(string), {})
  })
  default = {
    name            = "default"
    node_count      = 2
    vm_size         = "Standard_D2s_v3"
    min_count       = 1
    max_count       = 5
    max_pods        = 110
    os_disk_size_gb = 128
    os_disk_type    = "Managed"
    node_labels     = {}
    tags            = {}
  }
}

# Additional Node Pools
variable "additional_node_pools" {
  description = "Map of additional node pool configurations"
  type = map(object({
    vm_size              = string
    node_count           = number
    enable_auto_scaling  = optional(bool, true)
    min_count           = optional(number, 1)
    max_count           = optional(number, 5)
    max_pods            = optional(number, 110)
    os_disk_size_gb     = optional(number, 128)
    os_disk_type        = optional(string, "Managed")
    os_type             = optional(string, "Linux")
    priority            = optional(string, "Regular")
    eviction_policy     = optional(string, "Delete")
    spot_max_price      = optional(number, -1)
    node_labels         = optional(map(string), {})
    node_taints         = optional(list(string), [])
    tags                = optional(map(string), {})
  }))
  default = {
    inference = {
      vm_size              = "Standard_D4s_v3"
      node_count           = 1
      enable_auto_scaling  = true
      min_count           = 0
      max_count           = 3
      max_pods            = 110
      os_disk_size_gb     = 256
      os_disk_type        = "Managed"
      os_type             = "Linux"
      priority            = "Regular"
      node_labels = {
        "workload-type" = "inference"
        "node-class"    = "compute-optimized"
      }
      node_taints = [
        "inference-workload=true:NoSchedule"
      ]
      tags = {}
    }
  }
}

# Auto Scaling Configuration
variable "enable_auto_scaling" {
  description = "Enable auto scaling for node pools"
  type        = bool
  default     = true
}

# Security Configuration
variable "enable_rbac" {
  description = "Enable Role-Based Access Control"
  type        = bool
  default     = true
}

variable "admin_group_object_ids" {
  description = "Azure AD group object IDs for cluster administrators"
  type        = list(string)
  default     = []
}

variable "enable_network_policy" {
  description = "Enable network policy"
  type        = bool
  default     = true
}

variable "enable_pod_security_policy" {
  description = "Enable Pod Security Policy (deprecated in K8s 1.25+)"
  type        = bool
  default     = false
}

# Azure-specific Features
variable "enable_azure_policy" {
  description = "Enable Azure Policy for AKS"
  type        = bool
  default     = true
}

variable "enable_oms_agent" {
  description = "Enable OMS Agent for monitoring"
  type        = bool
  default     = true
}

variable "enable_ingress_application_gateway" {
  description = "Enable Ingress Application Gateway"
  type        = bool
  default     = false
}

variable "enable_key_vault_secrets_provider" {
  description = "Enable Key Vault Secrets Provider"
  type        = bool
  default     = true
}

# Additional Components
variable "enable_cert_manager" {
  description = "Install cert-manager via Helm"
  type        = bool
  default     = true
}

variable "enable_nginx_ingress" {
  description = "Install NGINX Ingress Controller via Helm"
  type        = bool
  default     = true
}

# TrustformeRS Serve Deployment
variable "install_trustformers_serve" {
  description = "Install TrustformeRS Serve via Helm"
  type        = bool
  default     = false
}

variable "trustformers_namespace" {
  description = "Kubernetes namespace for TrustformeRS Serve"
  type        = string
  default     = "trustformers"
}

variable "trustformers_helm_chart_path" {
  description = "Path to TrustformeRS Serve Helm chart"
  type        = string
  default     = "../../../helm"
}

variable "trustformers_helm_chart_version" {
  description = "Version of TrustformeRS Serve Helm chart"
  type        = string
  default     = "0.1.0"
}

variable "trustformers_helm_values" {
  description = "Helm values for TrustformeRS Serve deployment"
  type        = string
  default     = ""
}

# Advanced Features
variable "enable_workload_identity" {
  description = "Enable Workload Identity for Azure AD integration"
  type        = bool
  default     = false
}

variable "enable_secret_store_csi_driver" {
  description = "Enable Secret Store CSI Driver"
  type        = bool
  default     = false
}

variable "enable_image_cleaner" {
  description = "Enable Image Cleaner to automatically clean up unused images"
  type        = bool
  default     = false
}

# Monitoring and Logging
variable "enable_container_insights" {
  description = "Enable Container Insights for monitoring"
  type        = bool
  default     = true
}

variable "log_analytics_workspace_id" {
  description = "Log Analytics Workspace ID (will create if not provided)"
  type        = string
  default     = ""
}

# Backup and Disaster Recovery
variable "enable_backup" {
  description = "Enable backup for persistent volumes"
  type        = bool
  default     = false
}

variable "backup_vault_resource_group" {
  description = "Resource group for backup vault"
  type        = string
  default     = ""
}

# Cost Optimization
variable "enable_spot_node_pools" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "maintenance_window" {
  description = "Maintenance window configuration"
  type = object({
    allowed = list(object({
      day   = string
      hours = list(number)
    }))
    not_allowed = optional(list(object({
      start = string
      end   = string
    })), [])
  })
  default = {
    allowed = [
      {
        day   = "Sunday"
        hours = [1, 2]
      }
    ]
    not_allowed = []
  }
}

# Network Configuration
variable "dns_service_ip" {
  description = "IP address within the Kubernetes service address range for cluster DNS"
  type        = string
  default     = "10.2.0.10"
}

variable "docker_bridge_cidr" {
  description = "CIDR notation IP range assigned to the Docker bridge network"
  type        = string
  default     = "172.17.0.1/16"
}

variable "service_cidr" {
  description = "CIDR notation IP range from which to assign service cluster IPs"
  type        = string
  default     = "10.2.0.0/24"
}

# Storage Configuration
variable "enable_disk_encryption_set" {
  description = "Enable disk encryption set for node pools"
  type        = bool
  default     = false
}

variable "disk_encryption_set_id" {
  description = "ID of the disk encryption set"
  type        = string
  default     = ""
}