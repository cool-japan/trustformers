# Variables for the complete AWS EKS module

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

# VPC Configuration
variable "vpc_id" {
  description = "VPC ID where the EKS cluster will be deployed"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for the EKS cluster"
  type        = list(string)
}

variable "node_subnet_ids" {
  description = "Subnet IDs for the EKS node groups (defaults to subnet_ids if not specified)"
  type        = list(string)
  default     = []
}

# EKS Cluster Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.27"
}

variable "endpoint_private_access" {
  description = "Enable private access to the EKS cluster endpoint"
  type        = bool
  default     = true
}

variable "endpoint_public_access" {
  description = "Enable public access to the EKS cluster endpoint"
  type        = bool
  default     = true
}

variable "public_access_cidrs" {
  description = "CIDR blocks that can access the public EKS cluster endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "additional_security_group_ids" {
  description = "Additional security group IDs to attach to the EKS cluster"
  type        = list(string)
  default     = []
}

variable "enabled_cluster_log_types" {
  description = "List of cluster log types to enable"
  type        = list(string)
  default     = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
}

variable "log_retention_days" {
  description = "Number of days to retain cluster logs"
  type        = number
  default     = 7
}

# Node Groups Configuration
variable "node_groups" {
  description = "Map of EKS node group configurations"
  type = map(object({
    instance_types               = list(string)
    capacity_type               = string
    desired_size                = number
    max_size                    = number
    min_size                    = number
    disk_size                   = number
    ami_type                    = string
    max_unavailable_percentage  = number
    key_name                    = string
    source_security_group_ids   = list(string)
    labels                      = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
    tags = map(string)
  }))
  default = {
    general = {
      instance_types              = ["t3.medium"]
      capacity_type              = "ON_DEMAND"
      desired_size               = 2
      max_size                   = 10
      min_size                   = 1
      disk_size                  = 50
      ami_type                   = "AL2_x86_64"
      max_unavailable_percentage = 25
      key_name                   = ""
      source_security_group_ids  = []
      labels                     = {}
      taints                     = []
      tags                       = {}
    }
    inference = {
      instance_types              = ["m5.xlarge", "m5.2xlarge"]
      capacity_type              = "ON_DEMAND"
      desired_size               = 1
      max_size                   = 5
      min_size                   = 0
      disk_size                  = 100
      ami_type                   = "AL2_x86_64"
      max_unavailable_percentage = 25
      key_name                   = ""
      source_security_group_ids  = []
      labels = {
        "workload-type" = "inference"
        "node-class"    = "compute-optimized"
      }
      taints = [
        {
          key    = "inference-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      tags = {
        "k8s.io/cluster-autoscaler/node-template/label/workload-type" = "inference"
      }
    }
  }
}

# EKS Addons Configuration
variable "addons" {
  description = "Map of EKS addons to install"
  type = map(object({
    version                  = string
    configuration_values     = optional(string)
    service_account_role_arn = optional(string)
  }))
  default = {
    "vpc-cni" = {
      version = "v1.13.4-eksbuild.1"
    }
    "coredns" = {
      version = "v1.10.1-eksbuild.1"
    }
    "kube-proxy" = {
      version = "v1.27.3-eksbuild.1"
    }
    "aws-ebs-csi-driver" = {
      version = "v1.20.0-eksbuild.1"
    }
  }
}

# Security Groups
variable "additional_security_group_rules" {
  description = "Additional security group rules to create"
  type = list(object({
    description = string
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_blocks = list(string)
  }))
  default = []
}

# AWS Load Balancer Controller
variable "enable_aws_load_balancer_controller" {
  description = "Enable AWS Load Balancer Controller"
  type        = bool
  default     = true
}

variable "aws_load_balancer_controller_version" {
  description = "Version of AWS Load Balancer Controller Helm chart"
  type        = string
  default     = "1.5.4"
}

# Cluster Autoscaler
variable "enable_cluster_autoscaler" {
  description = "Enable Cluster Autoscaler"
  type        = bool
  default     = true
}

variable "cluster_autoscaler_version" {
  description = "Version of Cluster Autoscaler Helm chart"
  type        = string
  default     = "9.29.0"
}

# Metrics Server
variable "enable_metrics_server" {
  description = "Enable Metrics Server"
  type        = bool
  default     = true
}

variable "metrics_server_version" {
  description = "Version of Metrics Server Helm chart"
  type        = string
  default     = "3.11.0"
}

# EBS CSI Driver
variable "enable_ebs_csi_driver" {
  description = "Enable EBS CSI Driver IAM role"
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

# Monitoring and Observability
variable "enable_monitoring_stack" {
  description = "Enable monitoring stack (Prometheus, Grafana, etc.)"
  type        = bool
  default     = false
}

variable "enable_logging_stack" {
  description = "Enable logging stack (Fluent Bit, Elasticsearch, etc.)"
  type        = bool
  default     = false
}

variable "enable_service_mesh" {
  description = "Enable service mesh (Istio)"
  type        = bool
  default     = false
}

# Backup and Disaster Recovery
variable "enable_velero" {
  description = "Enable Velero for backup and disaster recovery"
  type        = bool
  default     = false
}

variable "velero_s3_bucket" {
  description = "S3 bucket for Velero backups"
  type        = string
  default     = ""
}

# Network Policies
variable "enable_network_policies" {
  description = "Enable Calico for network policies"
  type        = bool
  default     = false
}

# GPU Support
variable "enable_gpu_support" {
  description = "Enable NVIDIA device plugin for GPU support"
  type        = bool
  default     = false
}

# Cost Management
variable "enable_cost_monitoring" {
  description = "Enable KubeCost for cost monitoring"
  type        = bool
  default     = false
}

# Security
variable "enable_pod_security_standards" {
  description = "Enable Pod Security Standards"
  type        = bool
  default     = true
}

variable "enable_falco" {
  description = "Enable Falco for runtime security monitoring"
  type        = bool
  default     = false
}

# Advanced Features
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_pools" {
  description = "Number of spot instance pools"
  type        = number
  default     = 3
}

variable "enable_mixed_instances" {
  description = "Enable mixed instances policy for node groups"
  type        = bool
  default     = false
}

variable "mixed_instances_policy" {
  description = "Mixed instances policy configuration"
  type = object({
    instances_distribution = object({
      on_demand_base_capacity                  = number
      on_demand_percentage_above_base_capacity = number
      spot_allocation_strategy                 = string
      spot_instance_pools                      = number
      spot_max_price                          = string
    })
    override = list(object({
      instance_type     = string
      weighted_capacity = number
    }))
  })
  default = {
    instances_distribution = {
      on_demand_base_capacity                  = 1
      on_demand_percentage_above_base_capacity = 25
      spot_allocation_strategy                 = "capacity-optimized"
      spot_instance_pools                      = 3
      spot_max_price                          = ""
    }
    override = [
      {
        instance_type     = "m5.large"
        weighted_capacity = 1
      },
      {
        instance_type     = "m5.xlarge"
        weighted_capacity = 2
      },
      {
        instance_type     = "m5a.large"
        weighted_capacity = 1
      },
      {
        instance_type     = "m5a.xlarge"
        weighted_capacity = 2
      }
    ]
  }
}

# Fargate Configuration
variable "enable_fargate" {
  description = "Enable Fargate profiles"
  type        = bool
  default     = false
}

variable "fargate_profiles" {
  description = "Map of Fargate profile configurations"
  type = map(object({
    subnet_ids = list(string)
    selectors = list(object({
      namespace = string
      labels    = map(string)
    }))
  }))
  default = {}
}

# IRSA (IAM Roles for Service Accounts) Configuration
variable "irsa_roles" {
  description = "Map of IRSA role configurations"
  type = map(object({
    namespace           = string
    service_account     = string
    policy_documents    = list(string)
    policy_arns         = list(string)
  }))
  default = {}
}

# Windows Support
variable "enable_windows_support" {
  description = "Enable Windows node groups"
  type        = bool
  default     = false
}

variable "windows_node_groups" {
  description = "Map of Windows node group configurations"
  type = map(object({
    instance_types = list(string)
    capacity_type  = string
    desired_size   = number
    max_size       = number
    min_size       = number
    disk_size      = number
    ami_type       = string
    labels         = map(string)
    tags           = map(string)
  }))
  default = {}
}