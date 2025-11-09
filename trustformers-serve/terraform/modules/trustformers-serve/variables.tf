# TrustformeRS Serve Module Variables

# General Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "trustformers"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_vpn_gateway" {
  description = "Enable VPN Gateway"
  type        = bool
  default     = false
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the infrastructure"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Container Configuration
variable "container_registry" {
  description = "Container registry URL"
  type        = string
  default     = "public.ecr.aws/trustformers"
}

variable "container_image" {
  description = "Container image name"
  type        = string
  default     = "trustformers-serve"
}

variable "container_tag" {
  description = "Container image tag"
  type        = string
  default     = "latest"
}

# Database Configuration
variable "enable_rds" {
  description = "Enable RDS PostgreSQL database"
  type        = bool
  default     = true
}

variable "db_engine" {
  description = "Database engine"
  type        = string
  default     = "postgres"
}

variable "db_engine_version" {
  description = "Database engine version"
  type        = string
  default     = "15.4"
}

variable "db_instance_class" {
  description = "Database instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_username" {
  description = "Database master username"
  type        = string
  default     = "trustformers"
}

variable "db_password" {
  description = "Database master password (leave empty to generate)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "db_allocated_storage" {
  description = "Database allocated storage in GB"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Database maximum allocated storage in GB"
  type        = number
  default     = 100
}

variable "db_storage_type" {
  description = "Database storage type"
  type        = string
  default     = "gp3"
}

variable "db_iops" {
  description = "Database IOPS (for io1/io2 storage types)"
  type        = number
  default     = null
}

variable "db_backup_retention_period" {
  description = "Database backup retention period in days"
  type        = number
  default     = 7
}

variable "db_backup_window" {
  description = "Database backup window"
  type        = string
  default     = "03:00-04:00"
}

variable "db_maintenance_window" {
  description = "Database maintenance window"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "db_monitoring_interval" {
  description = "Database monitoring interval in seconds"
  type        = number
  default     = 60
}

variable "db_enabled_cloudwatch_logs_exports" {
  description = "List of log types to export to CloudWatch"
  type        = list(string)
  default     = ["postgresql"]
}

variable "db_deletion_protection" {
  description = "Enable deletion protection for the database"
  type        = bool
  default     = true
}

variable "db_skip_final_snapshot" {
  description = "Skip final snapshot when deleting the database"
  type        = bool
  default     = false
}

# Redis Configuration
variable "enable_redis" {
  description = "Enable ElastiCache Redis"
  type        = bool
  default     = true
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 1
}

variable "redis_engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

variable "redis_parameter_group_name" {
  description = "Redis parameter group name"
  type        = string
  default     = "default.redis7"
}

variable "redis_port" {
  description = "Redis port"
  type        = number
  default     = 6379
}

variable "redis_snapshot_retention_limit" {
  description = "Redis snapshot retention limit in days"
  type        = number
  default     = 5
}

variable "redis_snapshot_window" {
  description = "Redis snapshot window"
  type        = string
  default     = "03:00-05:00"
}

variable "redis_maintenance_window" {
  description = "Redis maintenance window"
  type        = string
  default     = "sun:05:00-sun:07:00"
}

# EKS Configuration
variable "enable_eks" {
  description = "Enable EKS cluster"
  type        = bool
  default     = false
}

variable "eks_kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.27"
}

variable "eks_endpoint_config" {
  description = "EKS cluster endpoint configuration"
  type = object({
    private_access = bool
    public_access  = bool
    public_access_cidrs = list(string)
  })
  default = {
    private_access      = true
    public_access       = true
    public_access_cidrs = ["0.0.0.0/0"]
  }
}

variable "eks_node_groups" {
  description = "EKS node groups configuration"
  type = map(object({
    instance_types = list(string)
    capacity_type  = string
    min_size      = number
    max_size      = number
    desired_size  = number
    disk_size     = number
    ami_type      = string
    labels        = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    main = {
      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
      min_size      = 1
      max_size      = 3
      desired_size  = 2
      disk_size     = 50
      ami_type      = "AL2_x86_64"
      labels        = {}
      taints        = []
    }
  }
}

variable "eks_addons" {
  description = "EKS cluster addons"
  type = map(object({
    version = string
    configuration_values = optional(string)
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
  }
}

# Load Balancer Configuration
variable "enable_load_balancer" {
  description = "Enable Application Load Balancer"
  type        = bool
  default     = true
}

variable "lb_type" {
  description = "Load balancer type"
  type        = string
  default     = "application"
  
  validation {
    condition     = contains(["application", "network"], var.lb_type)
    error_message = "Load balancer type must be 'application' or 'network'."
  }
}

variable "lb_scheme" {
  description = "Load balancer scheme"
  type        = string
  default     = "internet-facing"
  
  validation {
    condition     = contains(["internet-facing", "internal"], var.lb_scheme)
    error_message = "Load balancer scheme must be 'internet-facing' or 'internal'."
  }
}

variable "lb_ip_address_type" {
  description = "Load balancer IP address type"
  type        = string
  default     = "ipv4"
}

variable "lb_target_groups" {
  description = "Load balancer target groups"
  type = map(object({
    port              = number
    protocol          = string
    target_type       = string
    health_check_path = string
    health_check_port = string
    matcher           = string
  }))
  default = {
    http = {
      port              = 8080
      protocol          = "HTTP"
      target_type       = "instance"
      health_check_path = "/health"
      health_check_port = "8080"
      matcher           = "200"
    }
    grpc = {
      port              = 9090
      protocol          = "HTTP"
      target_type       = "instance"
      health_check_path = "/health"
      health_check_port = "8080"
      matcher           = "200"
    }
  }
}

variable "ssl_certificate_arn" {
  description = "SSL certificate ARN for HTTPS listeners"
  type        = string
  default     = ""
}

variable "ssl_policy" {
  description = "SSL policy for HTTPS listeners"
  type        = string
  default     = "ELBSecurityPolicy-TLS-1-2-2017-01"
}

# Auto Scaling Configuration
variable "enable_autoscaling" {
  description = "Enable Auto Scaling Group (used when EKS is disabled)"
  type        = bool
  default     = true
}

variable "asg_instance_type" {
  description = "EC2 instance type for Auto Scaling Group"
  type        = string
  default     = "t3.medium"
}

variable "asg_ami_id" {
  description = "AMI ID for Auto Scaling Group instances"
  type        = string
  default     = ""
}

variable "asg_key_name" {
  description = "EC2 Key Pair name for Auto Scaling Group instances"
  type        = string
  default     = ""
}

variable "asg_user_data" {
  description = "User data script for Auto Scaling Group instances"
  type        = string
  default     = ""
}

variable "asg_min_size" {
  description = "Minimum size of Auto Scaling Group"
  type        = number
  default     = 1
}

variable "asg_max_size" {
  description = "Maximum size of Auto Scaling Group"
  type        = number
  default     = 5
}

variable "asg_desired_capacity" {
  description = "Desired capacity of Auto Scaling Group"
  type        = number
  default     = 2
}

variable "asg_health_check_type" {
  description = "Health check type for Auto Scaling Group"
  type        = string
  default     = "ELB"
}

variable "asg_health_check_grace_period" {
  description = "Health check grace period for Auto Scaling Group"
  type        = number
  default     = 300
}

variable "asg_scaling_policies" {
  description = "Auto Scaling policies"
  type = map(object({
    adjustment_type         = string
    scaling_adjustment      = number
    cooldown               = number
    metric_aggregation_type = string
    policy_type            = string
    target_value           = optional(number)
    metric_name            = string
    namespace              = string
    statistic              = string
  }))
  default = {
    scale_up = {
      adjustment_type         = "ChangeInCapacity"
      scaling_adjustment      = 1
      cooldown               = 300
      metric_aggregation_type = "Average"
      policy_type            = "SimpleScaling"
      metric_name            = "CPUUtilization"
      namespace              = "AWS/EC2"
      statistic              = "Average"
    }
    scale_down = {
      adjustment_type         = "ChangeInCapacity"
      scaling_adjustment      = -1
      cooldown               = 300
      metric_aggregation_type = "Average"
      policy_type            = "SimpleScaling"
      metric_name            = "CPUUtilization"
      namespace              = "AWS/EC2"
      statistic              = "Average"
    }
  }
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring and alarms"
  type        = bool
  default     = true
}

variable "monitoring_sns_topic_arn" {
  description = "SNS topic ARN for monitoring alerts"
  type        = string
  default     = ""
}

variable "monitoring_alarm_thresholds" {
  description = "Monitoring alarm thresholds"
  type = object({
    cpu_high_threshold        = number
    memory_high_threshold     = number
    disk_usage_threshold      = number
    response_time_threshold   = number
    error_rate_threshold      = number
    database_cpu_threshold    = number
    database_memory_threshold = number
    redis_cpu_threshold       = number
    redis_memory_threshold    = number
  })
  default = {
    cpu_high_threshold        = 80
    memory_high_threshold     = 85
    disk_usage_threshold      = 90
    response_time_threshold   = 5
    error_rate_threshold      = 5
    database_cpu_threshold    = 80
    database_memory_threshold = 85
    redis_cpu_threshold       = 80
    redis_memory_threshold    = 85
  }
}

# Storage Configuration
variable "enable_s3_storage" {
  description = "Enable S3 bucket for model storage"
  type        = bool
  default     = true
}

variable "s3_versioning_enabled" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

variable "s3_encryption_enabled" {
  description = "Enable S3 bucket encryption"
  type        = bool
  default     = true
}

variable "s3_lifecycle_rules" {
  description = "S3 bucket lifecycle rules"
  type = list(object({
    id      = string
    enabled = bool
    expiration_days = number
    transition_days = number
    transition_storage_class = string
  }))
  default = [
    {
      id      = "model_lifecycle"
      enabled = true
      expiration_days = 365
      transition_days = 30
      transition_storage_class = "STANDARD_IA"
    }
  ]
}

variable "s3_bucket_policy" {
  description = "S3 bucket policy JSON"
  type        = string
  default     = ""
}

# Security Configuration
variable "secrets_recovery_window_days" {
  description = "Number of days AWS Secrets Manager waits before deleting a secret"
  type        = number
  default     = 7
}

variable "jwt_secret" {
  description = "JWT secret key (leave empty to generate)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "api_keys" {
  description = "Map of API keys for application access"
  type        = map(string)
  default     = {}
  sensitive   = true
}