# 🚀 Deployment Guide - HCL Knowledge Assistant

Complete step-by-step guide for deploying the HCL Knowledge Assistant in various environments.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)

---

## ✅ Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **RAM**: Minimum 4GB, Recommended 8GB
- **Storage**: 10GB free space
- **Network**: Internet connection for API access

### Software Requirements
- **Docker**: Version 20.10+ ([Install](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ (included with Docker Desktop)
- **Git**: For cloning repository
- **Groq API Key**: Free at [console.groq.com](https://console.groq.com)

### Verify Installation

```bash
# Check Docker
docker --version
docker-compose --version

# Check Git
git --version
```

---

## 💻 Local Development

### Step 1: Clone Repository

```bash
# Clone the project
cd /path/to/workspace
cp -r day10_capstone ./hcl-knowledge-assistant
cd hcl-knowledge-assistant
```

### Step 2: Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

**Required Environment Variables:**
```bash
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your-random-secret-key
DEBUG=True  # For development
```

### Step 3: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Run Backend

```bash
# Start backend server
cd backend
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: http://localhost:8000

### Step 5: Run Frontend

```bash
# In a new terminal
cd frontend

# Simple HTTP server
python -m http.server 8080

# Or use Node.js http-server
npx http-server -p 8080
```

Frontend will be available at: http://localhost:8080

### Step 6: Test Application

```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2025-02-12T...",
  "version": "1.0.0",
  "services": {
    "rag_system": "online",
    "agent_system": "online",
    "database": "online"
  }
}
```

---

## 🐳 Docker Deployment

### Quick Start (Recommended)

```bash
# 1. Setup environment
cp .env.example .env
nano .env  # Add your GROQ_API_KEY

# 2. Build and start
docker-compose up -d

# 3. Check status
docker-compose ps

# 4. View logs
docker-compose logs -f
```

### Step-by-Step Docker Deployment

#### 1. Build Images

```bash
# Build backend image
docker-compose build backend

# Or build all services
docker-compose build
```

#### 2. Start Services

```bash
# Start in detached mode
docker-compose up -d

# Start with logs
docker-compose up

# Start specific service
docker-compose up -d backend
```

#### 3. Verify Deployment

```bash
# Check running containers
docker-compose ps

# Expected output:
NAME                     STATUS
hcl-knowledge-backend    Up 30 seconds (healthy)
hcl-knowledge-frontend   Up 30 seconds

# Test backend
curl http://localhost:8000/api/v1/health

# Test frontend
curl http://localhost
```

#### 4. View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

#### 5. Stop Services

```bash
# Stop containers (keeps data)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove everything (including volumes)
docker-compose down -v
```

### Docker Commands Reference

```bash
# Restart service
docker-compose restart backend

# Rebuild and restart
docker-compose up -d --build backend

# Execute command in container
docker-compose exec backend bash

# View resource usage
docker stats

# Clean up
docker system prune -a
```

---

## ☁️ Cloud Deployment

### AWS EC2 Deployment

#### 1. Launch EC2 Instance

- **AMI**: Ubuntu 22.04 LTS
- **Instance Type**: t3.medium (2 vCPU, 4GB RAM)
- **Storage**: 20GB gp3
- **Security Group**: 
  - Port 22 (SSH)
  - Port 80 (HTTP)
  - Port 443 (HTTPS)
  - Port 8000 (API - for testing)

#### 2. Connect to Instance

```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

#### 3. Install Docker

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo apt install docker-compose-plugin
```

#### 4. Deploy Application

```bash
# Clone repository
git clone <your-repo-url>
cd hcl-knowledge-assistant

# Setup environment
cp .env.example .env
nano .env  # Add API keys

# Deploy
docker-compose up -d

# Check status
docker-compose ps
```

#### 5. Setup Domain (Optional)

```bash
# Install nginx
sudo apt install nginx

# Configure domain
sudo nano /etc/nginx/sites-available/hcl-knowledge

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/hcl-knowledge /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Azure Container Instances

```bash
# Login to Azure
az login

# Create resource group
az group create --name hcl-knowledge-rg --location eastus

# Deploy container
az container create \
  --resource-group hcl-knowledge-rg \
  --name hcl-knowledge \
  --image your-registry/hcl-knowledge:latest \
  --dns-name-label hcl-knowledge \
  --ports 80 8000 \
  --environment-variables GROQ_API_KEY=your_key

# Get URL
az container show --resource-group hcl-knowledge-rg --name hcl-knowledge --query ipAddress.fqdn
```

### Google Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/your-project/hcl-knowledge

# Deploy
gcloud run deploy hcl-knowledge \
  --image gcr.io/your-project/hcl-knowledge \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GROQ_API_KEY=your_key
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Application
APP_NAME=HCL Knowledge Assistant
APP_VERSION=1.0.0
DEBUG=False
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000

# Security
SECRET_KEY=<generate-with: openssl rand -hex 32>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# LLM
GROQ_API_KEY=<your-groq-api-key>

# Database
DATABASE_URL=sqlite:///./hcl_knowledge.db
VECTOR_DB_PATH=./data/vectordb

# RAG
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### Performance Tuning

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Logging Configuration

```bash
# Log to file
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOG_ROTATION=1 day
LOG_RETENTION=7 days
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8001:8000"
```

#### 2. Container Won't Start

```bash
# Check logs
docker-compose logs backend

# Common fixes:
# - Check .env file exists
# - Verify API key is set
# - Check file permissions
# - Rebuild image: docker-compose build --no-cache
```

#### 3. API Connection Failed

```bash
# Check backend is running
curl http://localhost:8000/api/v1/health

# Check firewall
sudo ufw status
sudo ufw allow 8000

# Check Docker network
docker network ls
docker network inspect hcl-network
```

#### 4. Out of Memory

```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory

# Or add to docker-compose.yml:
services:
  backend:
    mem_limit: 2g
    memswap_limit: 2g
```

### Debug Mode

```bash
# Enable debug logging
DEBUG=True
LOG_LEVEL=DEBUG

# Restart with logs
docker-compose down
docker-compose up
```

### Health Checks

```bash
# Backend health
curl http://localhost:8000/api/v1/health

# Check all services
docker-compose ps
docker-compose logs --tail=50
```

---

## 🔄 Maintenance

### Regular Updates

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup

```bash
# Backup database
docker-compose exec backend cp /app/hcl_knowledge.db /app/backup/

# Backup data directory
tar -czf backup-$(date +%Y%m%d).tar.gz data/ logs/
```

### Monitoring

```bash
# Watch logs
docker-compose logs -f --tail=100

# Monitor resources
docker stats

# Check disk usage
docker system df
```

### Cleanup

```bash
# Remove old containers
docker-compose down --remove-orphans

# Clean Docker system
docker system prune -a

# Clean volumes (⚠️ deletes data)
docker volume prune
```

---

## 📊 Production Checklist

Before deploying to production:

- [ ] Change default passwords
- [ ] Set SECRET_KEY to random value
- [ ] Set DEBUG=False
- [ ] Configure SSL/TLS certificates
- [ ] Setup backup strategy
- [ ] Configure monitoring
- [ ] Setup log rotation
- [ ] Test disaster recovery
- [ ] Document runbooks
- [ ] Setup alerts

---

## 🆘 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify environment: Check `.env` file
3. Test connectivity: `curl` health endpoint
4. Review documentation
5. Check GitHub issues

---

**Deployment Complete!** 🎉

Access your application at: http://localhost

API Documentation: http://localhost:8000/api/docs
