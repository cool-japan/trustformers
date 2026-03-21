# Simple Example Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "container_registry" {
  description = "Container registry URL"
  type        = string
  default     = "public.ecr.aws/trustformers"
}

variable "container_tag" {
  description = "Container image tag"
  type        = string
  default     = "latest"
}

variable "key_name" {
  description = "EC2 Key Pair name for SSH access (optional)"
  type        = string
  default     = ""
}