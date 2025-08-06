# DATect Web Interface Guide

## 🌐 Modern Web Interface for FastAPI Service

The DATect system now includes a **professional web interface** that provides a complete dashboard for domoic acid forecasting through the FastAPI service.

## 🚀 Quick Start

### 1. Start the API Service
```bash
python run_api_service.py
```

### 2. Open the Web Interface
Visit: **http://localhost:8000**

The web interface will load automatically with:
- ✅ Real-time system health monitoring
- ✅ Interactive prediction interface  
- ✅ Performance metrics dashboard
- ✅ API documentation links

## 🎯 Web Interface Features

### **Dashboard Overview**
The web interface provides a comprehensive dashboard with multiple sections:

#### **System Health Panel** (Left Side)
- **Real-time Status**: Live system health monitoring
- **Resource Usage**: CPU, memory, disk utilization  
- **Service Uptime**: How long the system has been running
- **Health Checks**: Status of all system components

#### **Quick Stats Panel**
- **API Requests**: Total number of requests processed
- **Predictions Made**: Number of forecasts generated
- **Average Response Time**: System performance metrics
- **Auto-refresh**: Updates every 30 seconds

#### **Prediction Interface** (Main Panel)
- **Model Selection**: Choose between XGBoost and Ridge models
- **Site Selection**: All 10 Pacific Coast monitoring sites
- **Input Parameters**: 
  - Sea Surface Temperature (SST)
  - Chlorophyll concentration
  - Historical DA lag values (1-2 periods back)
- **Instant Results**: Real-time predictions with risk categorization

#### **API Documentation Links** (Bottom)
- **Interactive Docs**: Swagger UI for API testing
- **Health Endpoints**: Direct access to health checks
- **Metrics Dashboard**: Performance monitoring
- **Model Information**: Available models and versions

## 🎨 Design Features

### **Professional UI/UX**
- **Modern Design**: Built with Tailwind CSS framework
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Clean Interface**: Professional appearance suitable for operations
- **Loading States**: Smooth transitions and loading indicators

### **Real-time Updates** 
- **Live Health Monitoring**: Status updates every 30 seconds
- **Instant Predictions**: Results appear immediately
- **Dynamic Status Badges**: Color-coded system status
- **Error Handling**: User-friendly error messages

### **Interactive Elements**
- **Form Validation**: Input validation with helpful messages
- **Visual Feedback**: Color-coded risk categories
- **Hover Effects**: Interactive buttons and links
- **Responsive Design**: Adapts to different screen sizes

## 📊 Prediction Interface

### **How to Make Predictions**

1. **Select Model**: Choose XGBoost (recommended) or Ridge Regression
2. **Choose Site**: Pick from 10 monitoring locations
3. **Enter Parameters**:
   - **SST**: Sea surface temperature in Celsius
   - **Chlorophyll**: Chlorophyll-a concentration  
   - **DA Lag 1**: Previous week's DA measurement
   - **DA Lag 2**: Two weeks ago DA measurement
4. **Click "Make Prediction"**: Get instant results

### **Understanding Results**

#### **DA Concentration**
- Displayed in **μg/g** (micrograms per gram of tissue)
- Precise numerical prediction from ML model

#### **Risk Categories**
- **🟢 Low (0-5 μg/g)**: Safe for consumption
- **🟡 Moderate (5-20 μg/g)**: Caution advised  
- **🟠 High (20-40 μg/g)**: Avoid consumption
- **🔴 Extreme (>40 μg/g)**: Health hazard

#### **Model Information**
- Model name and type used for prediction
- Processing time in milliseconds
- Selected monitoring site
- Data quality indicators

## 🔧 Technical Implementation

### **Frontend Architecture**
```html
Modern Web Stack:
├── Tailwind CSS - Responsive styling framework
├── Font Awesome - Professional icons
├── Axios - HTTP client for API calls
└── Vanilla JavaScript - No framework dependencies
```

### **API Integration**
The web interface connects to these FastAPI endpoints:
- `GET /health` - System health status
- `GET /metrics` - Performance metrics
- `GET /models` - Available models
- `POST /predict` - Make predictions
- `GET /docs` - API documentation

### **Security Features**
- **Development Mode**: Uses development API key automatically
- **Input Validation**: Client and server-side validation
- **Error Handling**: Graceful error display
- **HTTPS Ready**: Secure connections in production

## 🖥️ Production Deployment

### **Environment Configuration**
For production deployment, set these environment variables:
```bash
export DATECT_ENVIRONMENT=production
export DATECT_API_KEY=your-secure-api-key
export DATECT_API_HOST=0.0.0.0
export DATECT_API_PORT=8000
```

### **Production Features**
- **Authentication**: API key requirement in production
- **Security Headers**: CORS and trusted host protection
- **Performance Monitoring**: Built-in metrics collection
- **Health Checks**: Comprehensive system monitoring
- **Static File Serving**: Efficient asset delivery

### **Reverse Proxy Setup** (Optional)
For production, use nginx or similar:
```nginx
server {
    listen 80;
    server_name datect.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📱 Mobile Responsiveness

The web interface is fully responsive and works on:
- **Desktop**: Full dashboard layout
- **Tablet**: Adapted grid layout  
- **Mobile**: Stacked single-column layout
- **All Browsers**: Chrome, Firefox, Safari, Edge

## 🔍 Troubleshooting

### **Common Issues**

#### **Web Interface Won't Load**
```bash
# Check if API service is running
curl http://localhost:8000/health

# Restart the service
python run_api_service.py
```

#### **Predictions Fail**
- Check that `final_output.parquet` exists
- Verify models are trained and available
- Check logs: `tail -f logs/datect_main.log`

#### **Health Status Shows Errors**
- Review system resources (CPU, memory, disk)
- Check model artifacts directory
- Verify configuration settings

#### **Models Don't Load**
```bash
# Check if models exist
ls -la model_artifacts/

# Train models if needed
python modular-forecast.py
```

### **Browser Compatibility**
- **Minimum Requirements**: Chrome 60+, Firefox 55+, Safari 12+
- **JavaScript**: ES6+ features required
- **CSS**: Flexbox and Grid support needed

## 🌟 Advanced Features

### **API Key Management** (Production)
```javascript
// Custom API key configuration
const API_KEY = 'your-production-key';
axios.defaults.headers.common['Authorization'] = `Bearer ${API_KEY}`;
```

### **Custom Styling**
The interface uses Tailwind CSS classes that can be customized:
```css
/* Custom theme colors */
.bg-primary { @apply bg-blue-600; }
.text-primary { @apply text-blue-600; }
.border-primary { @apply border-blue-600; }
```

### **Extended Monitoring**
Access detailed monitoring at:
- `/health/detailed` - Comprehensive health report
- `/metrics/prometheus` - Prometheus format metrics
- `/health/report` - Generate health reports

## 📈 Performance Optimization

### **Caching Strategy**
- Health status cached for 30 seconds
- Model list cached until refresh
- Static assets served efficiently

### **Loading Performance**
- CDN-hosted CSS/JS libraries
- Minimal JavaScript bundle
- Efficient API calls
- Progressive loading

## 🎯 Use Cases

### **Operational Monitoring**
- Real-time system health dashboard
- Performance metrics tracking
- Service availability monitoring

### **Scientific Research**
- Interactive prediction testing
- Model comparison interface
- Data quality validation

### **Public Health**
- Quick DA risk assessment
- Site-specific forecasts
- Historical trend analysis

The web interface transforms the DATect API into a **complete forecasting platform** suitable for operational deployment, scientific research, and public health monitoring.