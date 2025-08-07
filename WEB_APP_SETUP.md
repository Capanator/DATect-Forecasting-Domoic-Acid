# DATect Web Application Setup Guide

This guide explains how to set up and deploy the modern web application version of DATect.

## Architecture Overview

The web application consists of:
- **Backend**: FastAPI REST API (Python)
- **Frontend**: React SPA with Plotly visualizations
- **Database**: PostgreSQL (for future user management)
- **Deployment**: Docker containers with nginx

## Quick Start (Development)

### Prerequisites
- Python 3.9+
- Node.js 16+
- Docker & Docker Compose (for production)

### 1. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the API server
python main.py
# Server runs on http://localhost:8000
```

### 2. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
# Frontend runs on http://localhost:3000
```

### 3. Access the Application
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs
- API Health: http://localhost:8000/health

## Production Deployment

### Using Docker Compose
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Production Setup

#### 1. Backend Production
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### 2. Frontend Production
```bash
cd frontend

# Install dependencies
npm ci --only=production

# Build for production
npm run build

# Serve with nginx or another web server
# Built files are in the 'dist' directory
```

#### 3. Database Setup
```bash
# Start PostgreSQL
docker run -d --name datect-postgres \
  -e POSTGRES_DB=datect \
  -e POSTGRES_USER=datect_user \
  -e POSTGRES_PASSWORD=your_secure_password \
  -p 5432:5432 \
  postgres:15-alpine

# Initialize database
psql -h localhost -U datect_user -d datect -f database/init.sql
```

## Configuration

### Backend Environment Variables
```bash
# .env file for backend
DATABASE_URL=postgresql://datect_user:password@localhost:5432/datect
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
ENVIRONMENT=production
```

### Frontend Environment Variables
```bash
# .env file for frontend
VITE_API_URL=http://localhost:8000
```

## API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /api/sites` - Available monitoring sites
- `GET /api/models` - Available ML models
- `POST /api/forecast` - Generate forecast
- `GET /api/historical/{site}` - Historical data

### Example API Usage
```javascript
// Generate a forecast
const response = await fetch('http://localhost:8000/api/forecast', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    date: '2023-06-15',
    site: 'Monterey Bay',
    task: 'regression',
    model: 'xgboost'
  })
});

const forecast = await response.json();
console.log(forecast);
```

## Features

### Current Features
- ✅ Real-time forecasting interface
- ✅ Historical data visualization
- ✅ Multiple ML models (XGBoost, Ridge, Logistic)
- ✅ Feature importance analysis
- ✅ Responsive design
- ✅ REST API with automatic documentation

### Future Features (Optional Extensions)
- 🔄 User authentication and profiles
- 🔄 Forecast result caching in database
- 🔄 Email notifications for high-risk forecasts
- 🔄 Batch forecasting capabilities
- 🔄 Data export functionality
- 🔄 Advanced analytics dashboard

## Development

### Adding New Features
1. Backend changes: Modify `backend/main.py`
2. Frontend changes: Add components in `frontend/src/`
3. Database changes: Update `database/init.sql`

### Testing
```bash
# Backend testing
cd backend
python -m pytest

# Frontend testing
cd frontend
npm test

# Integration testing
python tools/testing/test_complete_pipeline.py
```

## Deployment Options

### 1. Docker Compose (Recommended)
- Single command deployment
- Includes all services
- Easy scaling and management

### 2. Cloud Platforms
- **AWS**: ECS/Fargate + RDS
- **Google Cloud**: Cloud Run + Cloud SQL
- **Azure**: Container Instances + Azure Database
- **Heroku**: Web dynos + Postgres addon

### 3. Traditional VPS
- nginx reverse proxy
- systemd services
- PostgreSQL installation

## Security Considerations

### Production Checklist
- [ ] Change default database passwords
- [ ] Use HTTPS/SSL certificates
- [ ] Configure CORS properly
- [ ] Set up firewall rules
- [ ] Enable database backups
- [ ] Monitor system resources
- [ ] Set up logging and alerting

### Environment Security
```bash
# Use environment variables for secrets
export SECRET_KEY="$(openssl rand -hex 32)"
export DATABASE_URL="postgresql://user:$(openssl rand -base64 32)@localhost/db"
```

## Monitoring and Maintenance

### Health Checks
- Backend: `GET /health`
- Database: Connection status
- Frontend: Application loading

### Logging
- Backend: FastAPI automatic request logging
- nginx: Access and error logs
- Database: PostgreSQL logs

### Performance
- Backend: Async FastAPI for high concurrency
- Frontend: Code splitting and lazy loading
- Database: Proper indexing for queries

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Check `CORS_ORIGINS` in backend configuration
   - Verify frontend API URL configuration

2. **Database Connection Failed**
   - Verify PostgreSQL is running
   - Check connection string format
   - Confirm firewall/security group settings

3. **Frontend Build Errors**
   - Clear node_modules: `rm -rf node_modules && npm install`
   - Check Node.js version compatibility

4. **Backend Import Errors**
   - Ensure all dependencies in requirements.txt
   - Verify Python path includes forecasting modules

### Getting Help
- Check API documentation: http://localhost:8000/docs
- Review application logs: `docker-compose logs`
- Validate data processing: Run temporal integrity tests

## Migration from Dash

If migrating from the existing Dash application:

1. **Data Compatibility**: No changes needed - uses same data processing pipeline
2. **Functionality**: All forecasting features preserved
3. **Performance**: Improved with async API and optimized frontend
4. **Deployment**: More flexible with containerization

The web application maintains full compatibility with your existing scientific validation and data processing systems.