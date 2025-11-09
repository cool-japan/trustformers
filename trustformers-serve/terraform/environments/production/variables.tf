# Production Environment Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the infrastructure"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "container_registry" {
  description = "Container registry URL"
  type        = string
  default     = "public.ecr.aws/trustformers"
}

variable "container_tag" {
  description = "Container image tag to deploy"
  type        = string
  default     = "latest"
}

variable "ssl_certificate_arn" {
  description = "SSL certificate ARN for HTTPS listeners"
  type        = string
  default     = ""
}

variable "monitoring_sns_topic_arn" {
  description = "SNS topic ARN for monitoring alerts"
  type        = string
  default     = ""
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

variable "blocked_country_codes" {
  description = "List of country codes to block (ISO 3166-1 alpha-2)"
  type        = list(string)
  default     = []
}