variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-2"
}

variable "aws_profile" {
  description = "AWS CLI profile to use"
  type        = string
  default     = "vijay-dev"
}

variable "environment" {
  description = "Deployment environment tag"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Short name used as a prefix for all resource names"
  type        = string
  default     = "energy-intelligence-agent"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "ec2_instance_type" {
  description = "EC2 instance type for the SSM tunnel host"
  type        = string
  default     = "t3.micro"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "Name of the initial PostgreSQL database"
  type        = string
  default     = "energy_db"
}

variable "db_username" {
  description = "Master username for RDS PostgreSQL"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Master password for RDS PostgreSQL (min 8 chars)"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.db_password) >= 8
    error_message = "db_password must be at least 8 characters."
  }
}
