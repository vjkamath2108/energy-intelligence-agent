output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of the two public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the two private subnets"
  value       = aws_subnet.private[*].id
}

output "ec2_instance_id" {
  description = "Instance ID of the SSM tunnel host"
  value       = aws_instance.tunnel.id
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint (host:port)"
  value       = "${aws_db_instance.postgres.address}:${aws_db_instance.postgres.port}"
}

output "rds_address" {
  description = "RDS PostgreSQL hostname (without port)"
  value       = aws_db_instance.postgres.address
}

output "rds_secret_arn" {
  description = "ARN of the Secrets Manager secret holding RDS credentials"
  value       = aws_secretsmanager_secret.rds.arn
}

output "ssm_start_session_command" {
  description = "AWS CLI command to open an SSM session to the tunnel host"
  value       = "aws ssm start-session --target ${aws_instance.tunnel.id} --region ${var.aws_region}"
}
