# Variables for GCP GKE module

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "tags" {
  description = "Additional labels to apply to all resources"
  type        = map(string)
  default     = {}
}

# Project Configuration
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "zones" {
  description = "GCP zones for node pools (empty for regional clusters)"
  type        = list(string)
  default     = []
}

# Network Configuration
variable "network_name" {
  description = "Name of the VPC network (will create if empty)"
  type        = string
  default     = ""
}

variable "subnet_name" {
  description = "Name of the subnet for GKE (will create if empty)"
  type        = string
  default     = ""
}

# GKE Cluster Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for the GKE cluster"
  type        = string
  default     = "1.27.7-gke.1121000"
}

variable "release_channel" {
  description = "Release channel for GKE cluster (RAPID, REGULAR, STABLE, or empty for static version)"
  type        = string
  default     = "REGULAR"
  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE", ""], var.release_channel)
    error_message = "Release channel must be RAPID, REGULAR, STABLE, or empty."
  }
}

# Private Cluster Configuration
variable "private_cluster" {
  description = "Private cluster configuration"
  type = object({
    enable_private_endpoint        = optional(bool, false)
    master_global_access_enabled   = optional(bool, false)
  })
  default = {
    enable_private_endpoint        = false
    master_global_access_enabled   = false
  }
}

variable "master_ipv4_cidr_block" {
  description = "IPv4 CIDR block for the master network"
  type        = string
  default     = "172.16.0.0/28"
}

variable "master_authorized_networks" {
  description = "List of master authorized networks"
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}

# Default Node Pool Configuration
variable "default_node_pool" {
  description = "Configuration for the default node pool"
  type = object({
    name                = optional(string, "default-pool")
    initial_node_count  = optional(number, 1)
    autoscaling_enabled = optional(bool, true)
    min_node_count      = optional(number, 1)
    max_node_count      = optional(number, 5)
    machine_type        = optional(string, "e2-medium")
    disk_size_gb        = optional(number, 100)
    disk_type           = optional(string, "pd-standard")
    image_type          = optional(string, "COS_CONTAINERD")
    preemptible         = optional(bool, false)
    auto_repair         = optional(bool, true)
    auto_upgrade        = optional(bool, true)
    max_surge           = optional(number, 1)
    max_unavailable     = optional(number, 0)
    node_labels         = optional(map(string), {})
    node_taints = optional(list(object({
      key    = string
      value  = string
      effect = string
    })), [])
    oauth_scopes = optional(list(string), [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only"
    ])
  })
  default = {
    name                = "default-pool"
    initial_node_count  = 1
    autoscaling_enabled = true
    min_node_count      = 1
    max_node_count      = 5
    machine_type        = "e2-medium"
    disk_size_gb        = 100
    disk_type           = "pd-standard"
    image_type          = "COS_CONTAINERD"
    preemptible         = false
    auto_repair         = true
    auto_upgrade        = true
    max_surge           = 1
    max_unavailable     = 0
    node_labels         = {}
    node_taints         = []
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only"
    ]
  }
}

# Additional Node Pools
variable "additional_node_pools" {
  description = "Map of additional node pool configurations"
  type = map(object({
    initial_node_count   = optional(number, 1)
    autoscaling_enabled  = optional(bool, true)
    min_node_count       = optional(number, 1)
    max_node_count       = optional(number, 5)
    machine_type         = optional(string, "e2-medium")
    disk_size_gb         = optional(number, 100)
    disk_type            = optional(string, "pd-standard")
    image_type           = optional(string, "COS_CONTAINERD")
    preemptible          = optional(bool, false)
    spot                 = optional(bool, false)
    auto_repair          = optional(bool, true)
    auto_upgrade         = optional(bool, true)
    max_surge            = optional(number, 1)
    max_unavailable      = optional(number, 0)
    accelerator_count    = optional(number, 0)
    accelerator_type     = optional(string, "")
    node_labels          = optional(map(string), {})
    node_taints = optional(list(object({
      key    = string
      value  = string
      effect = string
    })), [])
    oauth_scopes = optional(list(string), [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only"
    ])
  }))
  default = {
    inference = {
      initial_node_count   = 1
      autoscaling_enabled  = true
      min_node_count       = 0
      max_node_count       = 3
      machine_type         = "n1-standard-4"
      disk_size_gb         = 200
      disk_type            = "pd-ssd"
      image_type           = "COS_CONTAINERD"
      preemptible          = false
      spot                 = false
      auto_repair          = true
      auto_upgrade         = true
      max_surge            = 1
      max_unavailable      = 0
      accelerator_count    = 0
      accelerator_type     = ""
      node_labels = {
        "workload-type" = "inference"
        "node-class"    = "compute-optimized"
      }
      node_taints = [
        {
          key    = "inference-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      oauth_scopes = [
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/monitoring",
        "https://www.googleapis.com/auth/devstorage.read_only"
      ]
    }
  }
}

# Security Configuration
variable "enable_network_policy" {
  description = "Enable network policy (Calico)"
  type        = bool
  default     = true
}

variable "enable_pod_security_policy" {
  description = "Enable Pod Security Policy (deprecated in K8s 1.25+)"
  type        = bool
  default     = false
}

variable "enable_workload_identity" {
  description = "Enable Workload Identity for secure pod-to-service communication"
  type        = bool
  default     = true
}

variable "enable_shielded_nodes" {
  description = "Enable Shielded GKE nodes"
  type        = bool
  default     = true
}

variable "enable_binary_authorization" {
  description = "Enable Binary Authorization"
  type        = bool
  default     = false
}

variable "database_encryption_key" {
  description = "KMS key for etcd encryption"
  type        = string
  default     = ""
}

# GCP-specific Features
variable "enable_google_cloud_logging" {
  description = "Enable Google Cloud Logging"
  type        = bool
  default     = true
}

variable "enable_google_cloud_monitoring" {
  description = "Enable Google Cloud Monitoring"
  type        = bool
  default     = true
}

variable "enable_managed_prometheus" {
  description = "Enable Managed Prometheus"
  type        = bool
  default     = true
}

variable "enable_istio" {
  description = "Enable Istio service mesh"
  type        = bool
  default     = false
}

variable "enable_cloudrun" {
  description = "Enable Cloud Run on GKE"
  type        = bool
  default     = false
}

variable "enable_http_load_balancing" {
  description = "Enable HTTP Load Balancing addon"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaling" {
  description = "Enable Horizontal Pod Autoscaling addon"
  type        = bool
  default     = true
}

variable "enable_dns_cache" {
  description = "Enable DNS cache addon"
  type        = bool
  default     = true
}

variable "enable_gcp_filestore_csi" {
  description = "Enable GCP Filestore CSI driver"
  type        = bool
  default     = false
}

variable "enable_gcs_fuse_csi" {
  description = "Enable GCS FUSE CSI driver"
  type        = bool
  default     = false
}

# Autopilot Mode
variable "enable_autopilot" {
  description = "Enable GKE Autopilot mode (serverless Kubernetes)"
  type        = bool
  default     = false
}

# Maintenance Configuration
variable "maintenance_window" {
  description = "Maintenance window configuration"
  type = object({
    start_time = string
  })
  default = {
    start_time = "02:00"
  }
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

# Cost Management
variable "enable_cost_management" {
  description = "Enable cost management and optimization features"
  type        = bool
  default     = false
}

variable "enable_resource_usage_export" {
  description = "Enable resource usage export to BigQuery"
  type        = bool
  default     = false
}

variable "resource_usage_bigquery_dataset" {
  description = "BigQuery dataset for resource usage export"
  type        = string
  default     = ""
}

# Notification Configuration
variable "notification_config" {
  description = "Notification configuration for cluster events"
  type = object({
    pubsub_topic = string
  })
  default = null
}

# Advanced Features
variable "enable_intranode_visibility" {
  description = "Enable intranode visibility"
  type        = bool
  default     = false
}

variable "enable_l4_ilb_subsetting" {
  description = "Enable L4 ILB subsetting"
  type        = bool
  default     = false
}

variable "enable_vertical_pod_autoscaling" {
  description = "Enable Vertical Pod Autoscaling"
  type        = bool
  default     = false
}

variable "enable_cluster_autoscaling" {
  description = "Enable cluster-level autoscaling"
  type        = bool
  default     = true
}

# Backup and Disaster Recovery
variable "enable_backup_for_gke" {
  description = "Enable Backup for GKE"
  type        = bool
  default     = false
}

variable "backup_plan_config" {
  description = "Backup plan configuration"
  type = object({
    backup_schedule              = optional(string, "0 2 * * *")
    backup_retention_days        = optional(number, 30)
    include_volume_data          = optional(bool, true)
    include_secrets             = optional(bool, false)
    selected_namespaces         = optional(list(string), [])
    selected_applications       = optional(list(string), [])
  })
  default = {
    backup_schedule              = "0 2 * * *"
    backup_retention_days        = 30
    include_volume_data          = true
    include_secrets             = false
    selected_namespaces         = []
    selected_applications       = []
  }
}

# GPU Support
variable "enable_gpu_sharing" {
  description = "Enable GPU sharing on nodes"
  type        = bool
  default     = false
}

variable "gpu_partition_size" {
  description = "GPU partition size for multi-instance GPU"
  type        = string
  default     = ""
}

# Security Posture
variable "enable_security_posture" {
  description = "Enable Security Posture Management"
  type        = bool
  default     = false
}

variable "security_posture_mode" {
  description = "Security Posture mode (BASIC or ENTERPRISE)"
  type        = string
  default     = "BASIC"
  validation {
    condition     = contains(["BASIC", "ENTERPRISE"], var.security_posture_mode)
    error_message = "Security posture mode must be BASIC or ENTERPRISE."
  }
}

variable "vulnerability_mode" {
  description = "Vulnerability mode (VULNERABILITY_DISABLED, VULNERABILITY_BASIC, VULNERABILITY_ENTERPRISE)"
  type        = string
  default     = "VULNERABILITY_BASIC"
  validation {
    condition     = contains(["VULNERABILITY_DISABLED", "VULNERABILITY_BASIC", "VULNERABILITY_ENTERPRISE"], var.vulnerability_mode)
    error_message = "Vulnerability mode must be VULNERABILITY_DISABLED, VULNERABILITY_BASIC, or VULNERABILITY_ENTERPRISE."
  }
}