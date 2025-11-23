# Deployment Guide

Step-by-step guide for deploying LSTM_PFD in production environments.

**Version:** 1.0.0
**Date:** November 2025

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start (Docker)](#quick-start-docker)
- [Cloud Deployment](#cloud-deployment)
- [Edge Deployment](#edge-deployment)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB
- OS: Ubuntu 20.04+ / Windows 10+ / macOS 11+

**Recommended:**
- CPU: 8 cores
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM (optional, for faster inference)
- Storage: 20GB SSD

### Software Dependencies

```bash
# Docker (recommended)
Docker Engine 20.10+
Docker Compose 2.0+

# Or Python environment
Python 3.8+
CUDA 11.8+ (for GPU support)
```

---

## Quick Start (Docker)

### 1. Clone Repository

```bash
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git
cd LSTM_PFD
```

### 2. Build Docker Image

```bash
docker build -t lstm_pfd:v1.0 -f deployment/Dockerfile .
```

### 3. Run Container

```bash
docker run -d \
  --name lstm_pfd \
  -p 8000:8000 \
  --gpus all \  # Remove if no GPU
  lstm_pfd:v1.0
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [0.1, ...],  # 102,400 values
    "return_probabilities": true
  }'
```

### 5. View API Documentation

Open browser: `http://localhost:8000/docs`

---

## Docker Compose Deployment

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    image: lstm_pfd:v1.0
    container_name: lstm_pfd_api
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/ensemble_model.onnx
      - DEVICE=cuda
      - BATCH_SIZE=32
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  prometheus:
    image: prom/prometheus:latest
    container_name: lstm_pfd_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: lstm_pfd_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana_dashboard.json:/etc/grafana/provisioning/dashboards/dashboard.json
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

### Deploy with Docker Compose

```bash
docker-compose up -d
```

### Stop Deployment

```bash
docker-compose down
```

---

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance

```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g4dn.xlarge \  # GPU instance
  --key-name your-key \
  --security-group-ids sg-xxxxxx \
  --subnet-id subnet-xxxxxx

# SSH into instance
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Deploy
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git
cd LSTM_PFD
docker-compose up -d
```

#### 2. ECS/Fargate Deployment

```bash
# Build and push to ECR
aws ecr create-repository --repository-name lstm_pfd
docker tag lstm_pfd:v1.0 123456789.dkr.ecr.us-east-1.amazonaws.com/lstm_pfd:v1.0
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/lstm_pfd:v1.0

# Create ECS task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-def.json

# Create ECS service
aws ecs create-service \
  --cluster lstm-pfd-cluster \
  --service-name lstm-pfd-service \
  --task-definition lstm-pfd:1 \
  --desired-count 2 \
  --launch-type FARGATE
```

### Google Cloud Platform (GCP)

```bash
# Deploy to Cloud Run
gcloud run deploy lstm-pfd \
  --image gcr.io/your-project/lstm_pfd:v1.0 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 8Gi \
  --cpu 4
```

### Microsoft Azure

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group lstm-pfd-rg \
  --name lstm-pfd \
  --image youracr.azurecr.io/lstm_pfd:v1.0 \
  --cpu 4 \
  --memory 8 \
  --ports 8000 \
  --ip-address Public
```

---

## Edge Deployment

### Raspberry Pi 4 (8GB)

#### 1. Install Dependencies

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python
sudo apt-get install python3.9 python3-pip -y

# Install ONNX Runtime
pip3 install onnxruntime
```

#### 2. Deploy Quantized Model

```bash
# Copy quantized model
scp models/ensemble_model_quantized_int8.onnx pi@raspberrypi:/home/pi/

# Run inference
python3 deployment/edge_inference.py \
  --model ensemble_model_quantized_int8.onnx \
  --device cpu
```

### NVIDIA Jetson Nano

```bash
# Install JetPack SDK
# Flash SD card with JetPack 4.6+

# Install PyTorch
pip3 install torch torchvision

# Deploy
python3 deployment/edge_inference.py \
  --model ensemble_model.onnx \
  --device cuda
```

---

## Monitoring & Logging

### Prometheus Configuration

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'lstm_pfd'
    static_configs:
      - targets: ['api:8000']
```

### Grafana Dashboard

Access: `http://localhost:3000`
- **Username:** admin
- **Password:** admin

**Key Metrics:**
- Request rate (req/s)
- Inference latency (ms)
- GPU utilization (%)
- Error rate (%)
- Prediction distribution

### Logging

**Configure logging:**
```python
# config/logging.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  file:
    class: logging.FileHandler
    filename: logs/lstm_pfd.log
    formatter: default
  console:
    class: logging.StreamHandler
    formatter: default
root:
  level: INFO
  handlers: [console, file]
```

**View logs:**
```bash
# Docker logs
docker logs -f lstm_pfd

# File logs
tail -f logs/lstm_pfd.log
```

---

## Troubleshooting

### Common Issues

#### 1. Model Not Loading

**Error:** `Model file not found`

**Solution:**
```bash
# Check model path
ls models/

# Download model if missing
wget https://example.com/models/ensemble_model.onnx -O models/ensemble_model.onnx
```

#### 2. CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
export BATCH_SIZE=16  # Default is 32

# Or use CPU
export DEVICE=cpu
```

#### 3. Slow Inference

**Symptoms:** >100ms latency

**Solutions:**
- Use GPU instead of CPU
- Use quantized model (INT8)
- Reduce batch size
- Check GPU utilization

#### 4. Container Fails to Start

**Check logs:**
```bash
docker logs lstm_pfd

# Common issues:
# - Port 8000 already in use
# - GPU driver mismatch
# - Insufficient memory
```

**Solutions:**
```bash
# Change port
docker run -p 8080:8000 lstm_pfd:v1.0

# Use CPU only
docker run --env DEVICE=cpu lstm_pfd:v1.0
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [...], "return_probabilities": true}'
```

### Performance Tuning

```bash
# Optimize for throughput
export BATCH_SIZE=64
export NUM_WORKERS=4

# Optimize for latency
export BATCH_SIZE=1
export DEVICE=cuda

# Monitor performance
docker stats lstm_pfd
```

---

## Security Considerations

### Production Checklist

- [ ] Enable HTTPS (use reverse proxy like nginx)
- [ ] Add API authentication (API keys or OAuth)
- [ ] Rate limiting configured
- [ ] Input validation enabled
- [ ] Model file integrity checks
- [ ] Firewall rules configured
- [ ] Regular security updates
- [ ] Audit logging enabled

### Example nginx Configuration

```nginx
upstream lstm_pfd {
    server localhost:8000;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;

    location / {
        proxy_pass http://lstm_pfd;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Rate limiting
        limit_req zone=api_limit burst=20 nodelay;
    }
}
```

---

## Rollback Procedure

### Quick Rollback

```bash
# Stop current deployment
docker-compose down

# Deploy previous version
docker tag lstm_pfd:v0.9 lstm_pfd:v1.0
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

### Database Rollback

```bash
# Restore from backup
docker exec -i postgres psql -U user -d db < backup.sql
```

---

## Contact & Support

- **Documentation:** See USAGE_GUIDES/
- **Issues:** https://github.com/abbas-ahmad-cowlar/LSTM_PFD/issues
- **Email:** support@example.com

---

**End of Deployment Guide**
