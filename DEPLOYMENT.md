# DATect Google Cloud Deployment Guide

## Overview

This guide explains how to deploy DATect to Google Cloud with pre-computed caching to avoid expensive server-side computations. The system pre-computes all retrospective forecasts and spectral analyses locally, then bakes them into a Docker image for deployment.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Local Build   │    │  Docker Image    │    │  Google Cloud   │
│                 │    │                  │    │                 │
│ • Pre-compute   │───▶│ • Cached Data    │───▶│ • Cloud Run     │
│ • All forecasts │    │ • API Server     │    │ • Auto-scaling  │
│ • Spectral data │    │ • Frontend       │    │ • Load balancer │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Pre-Computed Components

### 1. Retrospective Forecasts
- **Classification + XGBoost**: ~2000 predictions across 10 sites
- **Classification + Linear**: ~2000 predictions across 10 sites  
- **Regression + XGBoost**: ~2000 predictions across 10 sites
- **Regression + Linear**: ~2000 predictions across 10 sites

### 2. Spectral Analysis
- **All Sites Combined**: Power spectral density, coherence analysis
- **Individual Sites**: Site-specific spectral characteristics for all 10 locations
- **XGBoost Comparisons**: Spectral analysis of model predictions vs actual data

### 3. Visualization Data
- **Correlation Matrices**: Pre-computed for all sites
- **Feature Importance**: Cached for all model combinations

## Deployment Steps

### Prerequisites

```bash
# Set environment variables
export PROJECT_ID=datect-forecasting-domoic-acid
export REGION=us-west1

# Install Google Cloud CLI
# https://cloud.google.com/sdk/docs/install
```

### Option 1: Automated Deployment (Recommended)

```bash
# One-command deployment
./deploy_gcloud.sh
```

This script will:
1. ✅ Pre-compute all expensive operations locally (~10-30 minutes)
2. ✅ Build optimized Docker image with baked cache
3. ✅ Deploy to Google Cloud Run with auto-scaling
4. ✅ Configure health checks and monitoring
5. ✅ Provide service URL and API documentation links

### Option 2: Manual Deployment

#### Step 1: Pre-compute Cache Locally

```bash
# Generate all cache files (takes 10-30 minutes)
python3 precompute_cache.py

# Verify cache generation
ls -la ./cache/
cat ./cache/manifest.json
```

#### Step 2: Build Docker Image

```bash
# Build production image with baked cache
docker build -f Dockerfile.production -t datect-production .

# Test locally (optional)
docker run -p 8000:8000 datect-production
```

#### Step 3: Deploy to Google Cloud

```bash
# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# Deploy using Cloud Build
gcloud builds submit --config=cloudbuild.yaml --substitutions=_REGION=$REGION

# Get service URL
gcloud run services describe datect-api --platform=managed --region=$REGION --format="value(status.url)"
```

## Configuration

### Environment Variables

- `CACHE_DIR=/app/cache` - Read-only cache directory in container
- `NODE_ENV=production` - Production mode for frontend
- `PYTHONPATH=/app` - Python module path

### Resource Configuration

- **CPU**: 2 vCPUs (sufficient for serving cached data)
- **Memory**: 2GB (handles multiple concurrent requests)
- **Concurrency**: 80 requests per instance
- **Auto-scaling**: 0-10 instances based on demand
- **Timeout**: 5 minutes (for any remaining compute-heavy requests)

## Performance Benefits

### Before (Server-Side Computation)
- ❌ Retrospective analysis: 2-5 minutes per request
- ❌ Spectral analysis: 30-60 seconds per site
- ❌ High CPU usage and costs
- ❌ Poor user experience with long wait times

### After (Pre-Computed Cache)
- ✅ Retrospective analysis: <100ms response time
- ✅ Spectral analysis: <50ms response time  
- ✅ Minimal server resources needed
- ✅ Excellent user experience

## Cache Size and Storage

Typical cache sizes:
- **Retrospective forecasts**: ~50-100MB (JSON format)
- **Spectral analysis**: ~20-50MB (plot data)
- **Visualizations**: ~5-10MB (correlation matrices)
- **Total cache size**: ~100-200MB

The cache is baked into the Docker image as read-only files, eliminating the need for persistent storage or database connections.

## API Endpoints

All endpoints automatically serve cached data when available:

- `GET /api/retrospective` - Serves pre-computed forecasts
- `GET /api/visualizations/spectral/{site}` - Serves cached spectral analysis
- `GET /api/cache` - Shows cache status and availability
- `GET /health` - Health check endpoint

## Monitoring and Logs

```bash
# View application logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=datect-api" --limit 50

# Monitor cache hit rates
curl https://YOUR_SERVICE_URL/api/cache

# Check system health  
curl https://YOUR_SERVICE_URL/health
```

## Cost Optimization

### Compute Savings
- Pre-computation happens once during build (no recurring costs)
- Minimal server resources needed (2GB RAM, 2 vCPU)
- Auto-scaling to zero when not in use

### Build Optimization
- Uses Cloud Build's high-CPU machines for fast cache generation
- Multi-stage Docker build for minimal final image size
- Efficient caching and layer optimization

## Updating the Deployment

### Data Updates
1. Update source data files locally
2. Run `python3 precompute_cache.py` to regenerate cache
3. Re-deploy using `./deploy_gcloud.sh`

### Code Updates  
1. Make code changes
2. Deploy using Cloud Build (cache will be preserved if data hasn't changed)

## Troubleshooting

### Cache Issues
```bash
# Check cache status
curl https://YOUR_SERVICE_URL/api/cache

# Verify cache files in container
gcloud run services describe datect-api --format=export > service.yaml
```

### Build Issues
```bash
# Check Cloud Build logs
gcloud builds list --limit=10

# View specific build
gcloud builds log [BUILD_ID]
```

### Performance Issues
```bash
# Monitor resource usage
gcloud run services describe datect-api --format="get(spec.template.spec.containers[0].resources)"

# Check auto-scaling
gcloud run revisions list --service=datect-api
```

## Security

- ✅ HTTPS enforced for all traffic
- ✅ No sensitive data stored in cache
- ✅ Read-only cache prevents tampering
- ✅ Container runs as non-root user
- ✅ Minimal attack surface (no database, external dependencies)

## Next Steps

After deployment:
1. 🔗 Test all functionality via the web interface
2. 📊 Monitor performance and resource usage
3. 🔄 Set up automated deployments via CI/CD
4. 📈 Configure alerting and monitoring
5. 🎯 Optimize cache refresh frequency based on data update patterns