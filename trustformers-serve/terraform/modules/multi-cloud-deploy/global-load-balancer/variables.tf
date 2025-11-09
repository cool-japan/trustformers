# Variables for Global Load Balancer module

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "tags" {
  description = "Additional tags/labels to apply to all resources"
  type        = map(string)
  default     = {}
}

# Primary Cloud Configuration
variable "primary_cloud" {
  description = "Primary cloud provider for DNS and global resources (aws, azure, gcp)"
  type        = string
  validation {
    condition     = contains(["aws", "azure", "gcp"], var.primary_cloud)
    error_message = "Primary cloud must be one of: aws, azure, gcp."
  }
}

# Domain and SSL Configuration
variable "domain_name" {
  description = "Domain name for the global load balancer"
  type        = string
}

variable "dns_zone_name" {
  description = "Name of the DNS zone (if different from domain_name)"
  type        = string
  default     = ""
}

variable "ssl_policy" {
  description = "SSL/TLS policy for HTTPS listeners"
  type        = string
  default     = "ELBSecurityPolicy-TLS-1-2-2017-01"
}

# Backend Configurations
variable "aws_backend" {
  description = "AWS backend configuration"
  type = object({
    enabled           = bool
    endpoint          = optional(string, "")
    region            = optional(string, "")
    health_check_path = optional(string, "/health")
  })
  default = {
    enabled           = false
    endpoint          = ""
    region            = ""
    health_check_path = "/health"
  }
}

variable "azure_backend" {
  description = "Azure backend configuration"
  type = object({
    enabled           = bool
    endpoint          = optional(string, "")
    region            = optional(string, "")
    health_check_path = optional(string, "/health")
  })
  default = {
    enabled           = false
    endpoint          = ""
    region            = ""
    health_check_path = "/health"
  }
}

variable "gcp_backend" {
  description = "GCP backend configuration"
  type = object({
    enabled           = bool
    endpoint          = optional(string, "")
    region            = optional(string, "")
    health_check_path = optional(string, "/health")
  })
  default = {
    enabled           = false
    endpoint          = ""
    region            = ""
    health_check_path = "/health"
  }
}

# Traffic Distribution Configuration
variable "traffic_distribution" {
  description = "Traffic distribution configuration across clouds"
  type = object({
    method               = optional(string, "Weighted")
    aws_weight          = optional(number, 33)
    azure_weight        = optional(number, 33)
    gcp_weight          = optional(number, 34)
    aws_geo_mappings    = optional(list(string), ["US"])
    azure_geo_mappings  = optional(list(string), ["EU"])
    gcp_geo_mappings    = optional(list(string), ["AS"])
  })
  default = {
    method               = "Weighted"
    aws_weight          = 33
    azure_weight        = 33
    gcp_weight          = 34
    aws_geo_mappings    = ["US"]
    azure_geo_mappings  = ["EU"]
    gcp_geo_mappings    = ["AS"]
  }
}

# Failover Policy Configuration
variable "failover_policy" {
  description = "Failover policy configuration"
  type = object({
    aws_priority   = optional(number, 1)
    azure_priority = optional(number, 2)
    gcp_priority   = optional(number, 3)
    enable_failover = optional(bool, true)
  })
  default = {
    aws_priority   = 1
    azure_priority = 2
    gcp_priority   = 3
    enable_failover = true
  }
}

# AWS-specific Configuration (when AWS is primary)
variable "aws_vpc_id" {
  description = "AWS VPC ID for load balancer (when AWS is primary)"
  type        = string
  default     = ""
}

variable "aws_subnet_ids" {
  description = "AWS subnet IDs for load balancer (when AWS is primary)"
  type        = list(string)
  default     = []
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for AWS ALB"
  type        = bool
  default     = true
}

# Azure-specific Configuration (when Azure is primary)
variable "azure_resource_group_name" {
  description = "Azure resource group name (when Azure is primary)"
  type        = string
  default     = ""
}

variable "azure_location" {
  description = "Azure location (when Azure is primary)"
  type        = string
  default     = "East US"
}

# GCP-specific Configuration (when GCP is primary)
variable "gcp_project_id" {
  description = "GCP project ID (when GCP is primary)"
  type        = string
  default     = ""
}

variable "gcp_region" {
  description = "GCP region (when GCP is primary)"
  type        = string
  default     = "us-central1"
}

variable "enable_cdn" {
  description = "Enable CDN for GCP global load balancer"
  type        = bool
  default     = true
}

# Cloudflare Configuration (Optional)
variable "enable_cloudflare_dns" {
  description = "Enable Cloudflare DNS for additional redundancy"
  type        = bool
  default     = false
}

variable "cloudflare_zone_id" {
  description = "Cloudflare zone ID"
  type        = string
  default     = ""
}

variable "cloudflare_proxied" {
  description = "Enable Cloudflare proxy (orange cloud)"
  type        = bool
  default     = false
}

# Health Check Configuration
variable "health_check_config" {
  description = "Health check configuration"
  type = object({
    interval_seconds        = optional(number, 30)
    timeout_seconds        = optional(number, 5)
    healthy_threshold      = optional(number, 2)
    unhealthy_threshold    = optional(number, 3)
    path                   = optional(string, "/health")
    port                   = optional(number, 443)
    protocol              = optional(string, "HTTPS")
    success_codes         = optional(string, "200")
  })
  default = {
    interval_seconds        = 30
    timeout_seconds        = 5
    healthy_threshold      = 2
    unhealthy_threshold    = 3
    path                   = "/health"
    port                   = 443
    protocol              = "HTTPS"
    success_codes         = "200"
  }
}

# Monitoring and Alerting Configuration
variable "sns_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarms (AWS primary)"
  type        = string
  default     = ""
}

variable "enable_monitoring" {
  description = "Enable monitoring and alerting"
  type        = bool
  default     = true
}

variable "monitoring_config" {
  description = "Monitoring configuration"
  type = object({
    response_time_threshold   = optional(number, 1.0)
    error_rate_threshold     = optional(number, 5.0)
    enable_detailed_metrics  = optional(bool, true)
    log_requests            = optional(bool, false)
  })
  default = {
    response_time_threshold   = 1.0
    error_rate_threshold     = 5.0
    enable_detailed_metrics  = true
    log_requests            = false
  }
}

# Security Configuration
variable "enable_waf" {
  description = "Enable Web Application Firewall"
  type        = bool
  default     = false
}

variable "waf_config" {
  description = "WAF configuration"
  type = object({
    managed_rules = optional(list(string), [
      "AWSManagedRulesCommonRuleSet",
      "AWSManagedRulesKnownBadInputsRuleSet"
    ])
    rate_limit          = optional(number, 10000)
    enable_geo_blocking = optional(bool, false)
    blocked_countries   = optional(list(string), [])
  })
  default = {
    managed_rules = [
      "AWSManagedRulesCommonRuleSet",
      "AWSManagedRulesKnownBadInputsRuleSet"
    ]
    rate_limit          = 10000
    enable_geo_blocking = false
    blocked_countries   = []
  }
}

# DDoS Protection
variable "enable_ddos_protection" {
  description = "Enable DDoS protection"
  type        = bool
  default     = true
}

# Access Logging
variable "enable_access_logging" {
  description = "Enable access logging"
  type        = bool
  default     = true
}

variable "access_log_config" {
  description = "Access log configuration"
  type = object({
    s3_bucket_name     = optional(string, "")
    s3_bucket_prefix   = optional(string, "access-logs")
    retention_days     = optional(number, 7)
    include_cookies    = optional(bool, false)
  })
  default = {
    s3_bucket_name     = ""
    s3_bucket_prefix   = "access-logs"
    retention_days     = 7
    include_cookies    = false
  }
}

# Caching Configuration
variable "caching_config" {
  description = "Caching configuration"
  type = object({
    enable_caching        = optional(bool, true)
    default_ttl          = optional(number, 86400)
    max_ttl              = optional(number, 31536000)
    compress             = optional(bool, true)
    cache_key_headers    = optional(list(string), ["Host"])
    cache_key_query_strings = optional(list(string), [])
  })
  default = {
    enable_caching        = true
    default_ttl          = 86400
    max_ttl              = 31536000
    compress             = true
    cache_key_headers    = ["Host"]
    cache_key_query_strings = []
  }
}

# Advanced Routing Configuration
variable "routing_rules" {
  description = "Advanced routing rules"
  type = list(object({
    priority            = number
    host_header         = optional(string, "")
    path_pattern        = optional(string, "")
    http_header_name    = optional(string, "")
    http_header_values  = optional(list(string), [])
    query_string        = optional(map(string), {})
    target_cloud        = string
    weight              = optional(number, 100)
  }))
  default = []
}

# Session Stickiness
variable "session_stickiness" {
  description = "Session stickiness configuration"
  type = object({
    enabled         = optional(bool, false)
    cookie_duration = optional(number, 86400)
    cookie_name     = optional(string, "LB")
  })
  default = {
    enabled         = false
    cookie_duration = 86400
    cookie_name     = "LB"
  }
}

# Blue/Green Deployment Support
variable "blue_green_config" {
  description = "Blue/Green deployment configuration"
  type = object({
    enabled            = optional(bool, false)
    blue_weight       = optional(number, 100)
    green_weight      = optional(number, 0)
    enable_canary     = optional(bool, false)
    canary_percentage = optional(number, 10)
  })
  default = {
    enabled            = false
    blue_weight       = 100
    green_weight      = 0
    enable_canary     = false
    canary_percentage = 10
  }
}

# Circuit Breaker Configuration
variable "circuit_breaker" {
  description = "Circuit breaker configuration"
  type = object({
    enabled                = optional(bool, false)
    failure_threshold     = optional(number, 50)
    success_threshold     = optional(number, 10)
    timeout_seconds       = optional(number, 30)
    recovery_timeout      = optional(number, 60)
  })
  default = {
    enabled                = false
    failure_threshold     = 50
    success_threshold     = 10
    timeout_seconds       = 30
    recovery_timeout      = 60
  }
}