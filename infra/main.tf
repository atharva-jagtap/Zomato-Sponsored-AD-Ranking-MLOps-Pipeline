# ─── VPC + EC2 + ALB for Zomato Ads Ranking Service ─────────────────────────
#
# What this creates:
#   - VPC with public + private subnets across 2 AZs
#   - EC2 instances (Blue + Green) for blue/green deployment
#   - ALB with target groups for traffic shifting
#   - Security groups: ALB → EC2 only (no direct public access to EC2)
#
# Usage:
#   terraform init
#   terraform plan -var="key_pair_name=your-keypair"
#   terraform apply -var="key_pair_name=your-keypair"

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "zomato-ads-mlops-688927627267"
    key    = "infra/terraform.tfstate"
    region = "ap-south-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# ─── Variables ────────────────────────────────────────────────────────────────

variable "aws_region" { default = "ap-south-1" }
variable "key_pair_name" { description = "EC2 key pair for SSH access" }
variable "instance_type" { default = "t3.medium" } # 2 vCPU, 4GB RAM — sufficient for XGBoost serving
variable "project" { default = "zomato-ads-mlops" }
variable "mlflow_tracking_uri" {
  description = "Reachable MLflow tracking URI for EC2 serving instances"
}

# ─── VPC ──────────────────────────────────────────────────────────────────────

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags                 = { Name = "${var.project}-vpc" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project}-igw" }
}

# Public subnets (ALB lives here)
resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.project}-public-a" }
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.project}-public-b" }
}

# Private subnets (EC2 serving instances live here)
resource "aws_subnet" "private_a" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.10.0/24"
  availability_zone = "${var.aws_region}a"
  tags              = { Name = "${var.project}-private-a" }
}

resource "aws_subnet" "private_b" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.11.0/24"
  availability_zone = "${var.aws_region}b"
  tags              = { Name = "${var.project}-private-b" }
}

# Route table: public subnets → internet gateway
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "${var.project}-public-rt" }
}

resource "aws_route_table_association" "public_a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "${var.project}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public_a.id
  tags          = { Name = "${var.project}-nat" }

  depends_on = [aws_internet_gateway.main]
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
  tags = { Name = "${var.project}-private-rt" }
}

resource "aws_route_table_association" "private_a" {
  subnet_id      = aws_subnet.private_a.id
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "private_b" {
  subnet_id      = aws_subnet.private_b.id
  route_table_id = aws_route_table.private.id
}

# ─── Security Groups ──────────────────────────────────────────────────────────

# ALB: accept HTTPS from internet
resource "aws_security_group" "alb" {
  name   = "${var.project}-alb-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS from internet"
  }
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP (redirect to HTTPS)"
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "${var.project}-alb-sg" }
}

# EC2: only accept traffic from ALB on port 8080
resource "aws_security_group" "ec2" {
  name   = "${var.project}-ec2-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "Ranking API from ALB only"
  }
  # Prometheus scrape from internal monitoring EC2
  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "Prometheus scrape from VPC"
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "${var.project}-ec2-sg" }
}

# ─── ALB ─────────────────────────────────────────────────────────────────────

resource "aws_lb" "main" {
  name               = "${var.project}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]
  tags               = { Name = "${var.project}-alb" }
}

# Blue target group (current production)
resource "aws_lb_target_group" "blue" {
  name        = "${var.project}-blue-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "instance"

  health_check {
    path                = "/health"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
  tags = { Name = "${var.project}-blue-tg" }
}

# Green target group (new version — receives traffic during deployment)
resource "aws_lb_target_group" "green" {
  name        = "${var.project}-green-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "instance"

  health_check {
    path                = "/health"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
  tags = { Name = "${var.project}-green-tg" }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  # Default: 100% traffic to Blue
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.blue.arn
  }
}

# ─── EC2 Instances ────────────────────────────────────────────────────────────

# Latest Amazon Linux 2023 AMI
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

# IAM role for EC2: pull from ECR, read from S3, describe MLflow
resource "aws_iam_role" "ec2_role" {
  name = "${var.project}-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_readonly" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "s3_readonly" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# User data: pull Docker image from ECR and start serving container
locals {
  user_data = base64encode(<<-EOF
    #!/bin/bash
    yum update -y
    yum install -y docker
    systemctl start docker
    systemctl enable docker

    # Login to ECR
    aws ecr get-login-password --region ${var.aws_region} | \
      docker login --username AWS --password-stdin \
      $(aws sts get-caller-identity --query Account --output text).dkr.ecr.${var.aws_region}.amazonaws.com

    # Pull and run serving container
    ECR_URI=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${var.aws_region}.amazonaws.com/${var.project}:serving-latest

    docker pull $ECR_URI
    docker run -d \
      --name ranking-service \
      --restart unless-stopped \
      -p 8080:8080 \
      -e MLFLOW_TRACKING_URI=${var.mlflow_tracking_uri} \
      -e MODEL_STAGE=Production \
      $ECR_URI
  EOF
  )
}

# Blue EC2 instance
resource "aws_instance" "blue" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.private_a.id
  vpc_security_group_ids = [aws_security_group.ec2.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  key_name               = var.key_pair_name
  user_data              = local.user_data
  tags = {
    Name       = "${var.project}-blue"
    Deployment = "blue"
  }
}

# Green EC2 instance (initially stopped — CodeDeploy activates it)
resource "aws_instance" "green" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.private_b.id
  vpc_security_group_ids = [aws_security_group.ec2.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  key_name               = var.key_pair_name
  user_data              = local.user_data
  tags = {
    Name       = "${var.project}-green"
    Deployment = "green"
  }
}

# Register Blue instance with Blue target group
resource "aws_lb_target_group_attachment" "blue" {
  target_group_arn = aws_lb_target_group.blue.arn
  target_id        = aws_instance.blue.id
  port             = 8080
}

# ─── Outputs ─────────────────────────────────────────────────────────────────

output "alb_dns_name" {
  description = "ALB DNS — use this as the ranking API endpoint"
  value       = aws_lb.main.dns_name
}

output "blue_instance_id" { value = aws_instance.blue.id }
output "green_instance_id" { value = aws_instance.green.id }
output "blue_tg_arn" { value = aws_lb_target_group.blue.arn }
output "green_tg_arn" { value = aws_lb_target_group.green.arn }
output "alb_listener_arn" { value = aws_lb_listener.http.arn }
