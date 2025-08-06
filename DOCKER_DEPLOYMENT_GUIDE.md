# DATect Docker Deployment Guide

## 🐳 Production-Ready Containerization

The DATect system is now fully containerized for easy deployment, scaling, and management across any environment.

## 🚀 Quick Start

### **Basic API Service**
```bash
# Build and start the API service
docker-compose up --build datect-api

# Access the web interface
open http://localhost:8000
```

### **Full System with Dashboard**
```bash
# Start API + Dashboard services
docker-compose --profile dashboard up --build

# Access services
# - API/Web Interface: http://localhost:8000
# - Interactive Dashboard: http://localhost:8065
```

## 📋 Available Services

### **Core Services**

#### **datect-api** (Port 8000)
- **Purpose**: Main FastAPI prediction service with web interface
- **Features**: REST API, health monitoring, prediction endpoints
- **Always Required**: Yes
- **Access**: http://localhost:8000

#### **datect-dashboard** (Port 8065)  
- **Purpose**: Interactive Dash forecasting dashboard
- **Features**: Real-time predictions, model comparisons, visualizations
- **Always Required**: No (Optional)
- **Access**: http://localhost:8065
- **Enable**: `--profile dashboard`

#### **datect-processor**
- **Purpose**: Data processing pipeline (satellite, climate, toxin data)
- **Features**: Downloads and processes all external data sources
- **Always Required**: Run periodically
- **Enable**: `--profile data`

### **Production Services**

#### **nginx** (Ports 80, 443)
- **Purpose**: Reverse proxy, load balancing, SSL termination
- **Features**: Rate limiting, security headers, static file serving
- **Enable**: `--profile production`

#### **datect-monitor** (Port 9090)
- **Purpose**: Prometheus monitoring and metrics collection
- **Features**: Performance tracking, alerting, dashboards
- **Enable**: `--profile monitoring`

## 🏗️ Deployment Scenarios

### **1. Development Environment**
```bash
# Quick local development
docker-compose up --build datect-api

# With dashboard for testing
docker-compose --profile dashboard up --build
```

### **2. Production Deployment**
```bash
# Full production stack
docker-compose --profile production --profile monitoring up -d

# Services running:
# - datect-api (behind nginx)
# - nginx (port 80/443)
# - prometheus monitoring
```

### **3. Data Processing Only**
```bash
# Run data processing pipeline
docker-compose --profile data up datect-processor

# Or on schedule (cron example):
0 2 * * * cd /path/to/datect && docker-compose --profile data up datect-processor
```

### **4. High Availability Setup**
```bash
# Scale API service
docker-compose up --build --scale datect-api=3

# With nginx load balancing
docker-compose --profile production up --build --scale datect-api=3
```

## 🔧 Configuration

### **Environment Variables**

#### **Production Configuration**
```bash
# Create .env file
cat > .env << EOF
DATECT_ENVIRONMENT=production
DATECT_API_HOST=0.0.0.0
DATECT_API_PORT=8000
DATECT_LOG_LEVEL=INFO
DATECT_SECRET_KEY=your-super-secure-secret-key
DATECT_API_KEY=your-api-access-key
EOF
```

#### **Development Configuration**
```bash
# Development environment
DATECT_ENVIRONMENT=development
DATECT_DEBUG=true
DATECT_LOG_LEVEL=DEBUG
```

### **Volume Mounts**

#### **Required Data**
- `./final_output.parquet` → Processed training data
- `./model_artifacts/` → Trained ML models
- `./da-input/` → DA measurement CSV files
- `./pn-input/` → PN measurement CSV files

#### **Persistent Storage**
- `datect-logs` → Application logs
- `prometheus-data` → Monitoring data

### **Port Configuration**
```yaml
Services:
  datect-api: 8000      # Main API service
  datect-dashboard: 8065 # Interactive dashboard
  nginx: 80, 443        # Reverse proxy
  prometheus: 9090      # Monitoring
```

## 📊 Monitoring & Health Checks

### **Health Monitoring**
```bash
# Check service health
curl http://localhost:8000/health

# Detailed system status  
curl http://localhost:8000/health/detailed

# Performance metrics
curl http://localhost:8000/metrics
```

### **Container Health**
```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs datect-api

# Follow logs in real-time
docker-compose logs -f datect-api
```

### **Prometheus Monitoring**
```bash
# Access Prometheus dashboard
open http://localhost:9090

# Key metrics to monitor:
# - datect_requests_total
# - datect_response_time_ms  
# - datect_predictions_total
# - datect_uptime_seconds
```

## 🔒 Security Considerations

### **Production Security**
1. **API Keys**: Always set `DATECT_API_KEY` in production
2. **Secret Keys**: Use strong `DATECT_SECRET_KEY`
3. **Network**: Use internal networks for service communication
4. **SSL/TLS**: Enable HTTPS with proper certificates
5. **User Permissions**: Run containers as non-root user

### **SSL Configuration**
```bash
# Create SSL directory
mkdir -p docker/ssl

# Add your certificates
cp your-cert.pem docker/ssl/
cp your-key.pem docker/ssl/

# Update nginx.conf for HTTPS
```

### **Firewall Rules**
```bash
# Only expose necessary ports
# 80/443: Web access
# 8000: API (if direct access needed)
# 9090: Monitoring (restrict to admins)
```

## 🚀 Production Deployment Steps

### **1. Server Preparation**
```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
pip install docker-compose
```

### **2. Application Setup**
```bash
# Clone repository
git clone https://github.com/your-org/DATect-Forecasting-Domoic-Acid.git
cd DATect-Forecasting-Domoic-Acid

# Prepare data files
cp /path/to/final_output.parquet .
cp -r /path/to/model_artifacts .

# Set production configuration
cp .env.example .env
# Edit .env with production values
```

### **3. Deploy Services**
```bash
# Production deployment
docker-compose --profile production --profile monitoring up -d

# Verify deployment
docker-compose ps
curl http://localhost/health
```

### **4. Setup Monitoring**
```bash
# Configure log rotation
# Setup prometheus alerts
# Configure backup procedures
```

## 🔄 Maintenance

### **Updates**
```bash
# Update application
git pull
docker-compose build --no-cache
docker-compose --profile production up -d

# Update individual service
docker-compose up -d --build datect-api
```

### **Backup**
```bash
# Backup volumes
docker run --rm -v datect-logs:/backup-source -v $(pwd):/backup alpine tar czf /backup/logs-backup.tar.gz -C /backup-source .

# Backup configuration
tar czf datect-config-backup.tar.gz .env docker/ model_artifacts/
```

### **Log Management**
```bash
# View logs
docker-compose logs datect-api

# Log rotation (configure in production)
# Use logrotate or similar for persistent log management
```

## 🐛 Troubleshooting

### **Common Issues**

#### **Container Won't Start**
```bash
# Check container status
docker-compose ps

# View error logs  
docker-compose logs datect-api

# Check resource usage
docker stats
```

#### **Health Check Failures**
```bash
# Manual health check
curl -v http://localhost:8000/health

# Check container networking
docker network ls
docker network inspect datect-network
```

#### **Permission Issues**
```bash
# Fix file permissions
sudo chown -R 1000:1000 model_artifacts/ logs/

# Check volume mounts
docker-compose config
```

#### **Performance Issues**
```bash
# Monitor resource usage
docker stats

# Scale services
docker-compose up --scale datect-api=2 -d

# Check system resources
free -h
df -h
```

### **Debug Mode**
```bash
# Run in debug mode
DATECT_DEBUG=true docker-compose up datect-api

# Interactive shell
docker-compose exec datect-api bash

# Check configuration
docker-compose exec datect-api python -c "from forecasting.core.env_config import get_config; print(get_config().get_config_summary())"
```

## 📈 Scaling

### **Horizontal Scaling**
```bash
# Scale API service
docker-compose up --scale datect-api=3 -d

# With load balancer
docker-compose --profile production up --scale datect-api=3 -d
```

### **Resource Allocation**
```yaml
# Add to docker-compose.yml
services:
  datect-api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0' 
          memory: 2G
```

### **Auto-scaling with Docker Swarm**
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml datect

# Scale service
docker service scale datect_datect-api=3
```

## 🎯 Best Practices

1. **Always use specific image tags** in production (not `latest`)
2. **Implement proper log aggregation** (ELK stack, etc.)
3. **Use secrets management** for sensitive configuration
4. **Regular backup procedures** for data and configuration
5. **Monitor resource usage** and set appropriate limits
6. **Implement CI/CD pipelines** for automated deployments
7. **Use multi-stage builds** to minimize image sizes
8. **Security scanning** of container images
9. **Network segmentation** for service isolation
10. **Disaster recovery planning** with tested procedures

The DATect system is now **production-ready** with enterprise-grade containerization supporting development, staging, and production environments!