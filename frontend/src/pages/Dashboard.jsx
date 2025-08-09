import React, { useState, useEffect } from 'react'
import { Calendar, MapPin, Cpu, AlertTriangle, Settings, Play, BarChart3, TrendingUp, Clock } from 'lucide-react'
import DatePicker from 'react-datepicker'
import Select from 'react-select'
import Plot from 'react-plotly.js'
import { format, subDays } from 'date-fns'
import api from '../services/api'
import 'react-datepicker/dist/react-datepicker.css'

const Dashboard = () => {
  // Workflow state
  const [currentStep, setCurrentStep] = useState('config') // 'config', 'realtime', 'retrospective', 'results'
  
  // Config state
  const [config, setConfig] = useState({
    forecast_mode: 'realtime',
    forecast_task: 'regression', 
    forecast_model: 'xgboost',
    selected_sites: [] // For retrospective site filtering
  })
  const [configLoading, setConfigLoading] = useState(false)
  
  // Data loading state
  const [sites, setSites] = useState([])
  const [models, setModels] = useState({ regression: [], classification: [] })
  const [dateRange, setDateRange] = useState({ min: null, max: null })
  
  // Realtime forecast state
  const [selectedDate, setSelectedDate] = useState(null)
  const [selectedSite, setSelectedSite] = useState(null)
  const [selectedModel, setSelectedModel] = useState('xgboost')
  const [task, setTask] = useState('regression')
  
  // Results state
  const [forecast, setForecast] = useState(null)
  const [retrospectiveResults, setRetrospectiveResults] = useState(null)
  const [filteredResults, setFilteredResults] = useState(null)
  const [selectedSiteFilter, setSelectedSiteFilter] = useState('all') // For filtering existing results
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadInitialData()
    loadConfig()
  }, [])

  const loadInitialData = async () => {
    try {
      const [sitesRes, modelsRes] = await Promise.all([
        api.get('/api/sites'),
        api.get('/api/models')
      ])
      
      setSites(sitesRes.data.sites)
      setModels(modelsRes.data.available_models)
      setDateRange(sitesRes.data.date_range)
      
      // Set defaults
      if (sitesRes.data.sites.length > 0) {
        setSelectedSite({ value: sitesRes.data.sites[0], label: sitesRes.data.sites[0] })
        // Initialize selected_sites with all sites for retrospective analysis
        setConfig(prev => ({ ...prev, selected_sites: sitesRes.data.sites }))
      }
      
      if (sitesRes.data.date_range.max) {
        setSelectedDate(subDays(new Date(sitesRes.data.date_range.max), 30))
      }
      
    } catch (err) {
      setError('Failed to load initial data')
      console.error(err)
    }
  }

  const loadConfig = async () => {
    try {
      const response = await api.get('/api/config')
      setConfig(response.data)
      setTask(response.data.forecast_task)
      setSelectedModel(response.data.forecast_model)
    } catch (err) {
      console.error('Failed to load config:', err)
    }
  }

  const applyConfig = async () => {
    if (!config.forecast_mode) {
      setError('Please select a forecast mode')
      return
    }
    
    if (config.forecast_mode === 'retrospective' && (!config.forecast_task || !config.forecast_model)) {
      setError('Please select forecast task and model for retrospective analysis')
      return
    }

    setConfigLoading(true)
    setError(null)
    
    try {
      const response = await api.post('/api/config', config)
      if (response.data.success) {
        // Update local state
        setTask(config.forecast_task)
        setSelectedModel(config.forecast_model)
        
        // Move to next step based on mode
        if (config.forecast_mode === 'realtime') {
          setCurrentStep('realtime')
        } else {
          // Start retrospective analysis immediately
          setCurrentStep('retrospective')
          await runRetrospectiveAnalysis()
        }
      }
    } catch (err) {
      setError('Failed to update configuration')
      console.error(err)
    } finally {
      setConfigLoading(false)
    }
  }

  // Filter results locally without re-running analysis
  const filterResultsBySite = (siteFilter) => {
    if (!retrospectiveResults) return
    
    setSelectedSiteFilter(siteFilter)
    
    if (siteFilter === 'all') {
      setFilteredResults(retrospectiveResults)
    } else {
      const filtered = {
        ...retrospectiveResults,
        results: retrospectiveResults.results.filter(r => r.site === siteFilter)
      }
      
      // Recalculate summary statistics for filtered data based on task type
      const isClassification = config.forecast_task === 'classification'
      
      if (isClassification) {
        // Recalculate classification metrics
        const validClassification = filtered.results.filter(r => 
          r.actual_category !== null && r.actual_category !== undefined &&
          r.predicted_category !== null && r.predicted_category !== undefined
        )
        
        if (validClassification.length > 0) {
          const correctPredictions = validClassification.filter(r => 
            r.actual_category === r.predicted_category
          ).length
          const accuracy = correctPredictions / validClassification.length
          
          filtered.summary = {
            ...filtered.summary,
            total_forecasts: filtered.results.length,
            classification_forecasts: validClassification.length,
            accuracy: accuracy
          }
        }
      } else {
        // Recalculate regression metrics
        const validRegression = filtered.results.filter(r => 
          r.actual_da !== null && r.predicted_da !== null
        )
        
        if (validRegression.length > 0) {
          const actuals = validRegression.map(r => r.actual_da)
          const predictions = validRegression.map(r => r.predicted_da)
          
          // Calculate R2
          const meanActual = actuals.reduce((a, b) => a + b, 0) / actuals.length
          const ssTotal = actuals.reduce((sum, val) => sum + Math.pow(val - meanActual, 2), 0)
          const ssResidual = actuals.reduce((sum, val, i) => 
            sum + Math.pow(val - predictions[i], 2), 0
          )
          const r2 = 1 - (ssResidual / ssTotal)
          
          // Calculate MAE
          const mae = actuals.reduce((sum, val, i) => 
            sum + Math.abs(val - predictions[i]), 0
          ) / actuals.length
          
          filtered.summary = {
            ...filtered.summary,
            total_forecasts: filtered.results.length,
            regression_forecasts: validRegression.length,
            r2_score: r2,
            mae: mae
          }
        }
      }
      
      setFilteredResults(filtered)
    }
  }

  const runRetrospectiveAnalysis = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.post('/api/retrospective', {
        selected_sites: config.selected_sites
      })
      
      setRetrospectiveResults(response.data)
      setFilteredResults(response.data)
      setSelectedSiteFilter('all')
      setCurrentStep('results')
    } catch (err) {
      setError('Failed to run retrospective analysis')
      console.error('Retrospective analysis error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleRealtimeForecast = async () => {
    if (!selectedDate || !selectedSite) {
      setError('Please select both date and site')
      return
    }

    setLoading(true)
    setError(null)
    
    try {
      console.log('Sending forecast request:', {
        date: format(selectedDate, 'yyyy-MM-dd'),
        site: selectedSite.value,
        task: task,
        model: selectedModel
      })
      
      const response = await api.post('/api/forecast/enhanced', {
        date: format(selectedDate, 'yyyy-MM-dd'),
        site: selectedSite.value,
        task: task,
        model: selectedModel
      })
      
      console.log('Received forecast response:', response.data)
      console.log('Classification data:', response.data?.classification)
      console.log('Class probabilities:', response.data?.classification?.class_probabilities)
      setForecast(response.data)
      setCurrentStep('results')
    } catch (err) {
      console.error('Forecast error details:', err)
      console.error('Error response:', err.response?.data)
      setError(`Failed to generate forecast: ${err.response?.data?.detail || err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const resetWorkflow = () => {
    setCurrentStep('config')
    setForecast(null)
    setRetrospectiveResults(null)
    setError(null)
  }

  // Create retrospective time series graph
  const createRetrospectiveTimeSeries = () => {
    if (!filteredResults?.results) return null

    const results = filteredResults.results
    const isClassification = config.forecast_task === 'classification'
    
    // Group by site for better visualization
    const sites = [...new Set(results.map(r => r.site))].slice(0, 5) // Limit to 5 sites for readability
    const isSingleSite = sites.length === 1
    
    const traces = []
    
    if (isClassification) {
      // Classification time series
      sites.forEach((site, siteIndex) => {
        const siteData = results
          .filter(r => r.site === site && r.actual_category !== null && r.actual_category !== undefined && 
                       r.predicted_category !== null && r.predicted_category !== undefined)
          .sort((a, b) => new Date(a.date) - new Date(b.date))
        
        if (siteData.length === 0) return
        
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        const siteColor = colors[siteIndex % colors.length]
        
        // Color logic: single site = blue/red, multiple sites = site-specific colors
        const actualColor = isSingleSite ? 'blue' : siteColor
        const predictedColor = isSingleSite ? 'red' : siteColor
        
        // Actual categories
        traces.push({
          x: siteData.map(d => d.date),
          y: siteData.map(d => d.actual_category),
          mode: 'lines+markers',
          name: `${site} - Actual Category`,
          line: { color: actualColor, width: 2 },
          marker: { size: 6 },
          hovertemplate: '<b>%{text}</b><br>Date: %{x}<br>Actual Category: %{customdata}<extra></extra>',
          text: siteData.map(d => site),
          customdata: siteData.map(d => {
            const categories = ['Low', 'Moderate', 'High', 'Extreme']
            return categories[d.actual_category] || `Category ${d.actual_category}`
          })
        })
        
        // Predicted categories
        traces.push({
          x: siteData.map(d => d.date),
          y: siteData.map(d => d.predicted_category),
          mode: 'lines+markers',
          name: `${site} - Predicted Category`,
          line: { color: predictedColor, width: 2, dash: 'dash' },
          marker: { size: 6, symbol: 'square' },
          hovertemplate: '<b>%{text}</b><br>Date: %{x}<br>Predicted Category: %{customdata}<extra></extra>',
          text: siteData.map(d => site),
          customdata: siteData.map(d => {
            const categories = ['Low', 'Moderate', 'High', 'Extreme']
            return categories[d.predicted_category] || `Category ${d.predicted_category}`
          })
        })
      })
      
      return {
        data: traces,
        layout: {
          title: `DA Category Forecasting Results - ${config.forecast_model} Classification`,
          xaxis: { title: 'Date' },
          yaxis: { 
            title: 'DA Risk Category',
            tickmode: 'array',
            tickvals: [0, 1, 2, 3],
            ticktext: ['Low', 'Moderate', 'High', 'Extreme'],
            range: [-0.5, 3.5]
          },
          height: 500,
          hovermode: 'closest'
        }
      }
    } else {
      // Regression time series (existing code)
      sites.forEach((site, siteIndex) => {
        const siteData = results
          .filter(r => r.site === site && r.actual_da !== null && r.predicted_da !== null)
          .sort((a, b) => new Date(a.date) - new Date(b.date))
        
        if (siteData.length === 0) return
        
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        const siteColor = colors[siteIndex % colors.length]
        
        // Color logic: single site = blue/red, multiple sites = site-specific colors
        const actualColor = isSingleSite ? 'blue' : siteColor
        const predictedColor = isSingleSite ? 'red' : siteColor
        
        // Actual values
        traces.push({
          x: siteData.map(d => d.date),
          y: siteData.map(d => d.actual_da),
          mode: 'lines+markers',
          name: `${site} - Actual`,
          line: { color: actualColor, width: 2 },
          marker: { size: 4 }
        })
        
        // Predicted values
        traces.push({
          x: siteData.map(d => d.date),
          y: siteData.map(d => d.predicted_da),
          mode: 'lines+markers',
          name: `${site} - Predicted`,
          line: { color: predictedColor, width: 2, dash: 'dash' },
          marker: { size: 4, symbol: 'square' }
        })
      })

      return {
        data: traces,
        layout: {
          title: `Retrospective Analysis: Actual vs Predicted DA Concentrations (${config.forecast_task})`,
          xaxis: { title: 'Date' },
          yaxis: { title: 'DA Concentration (μg/g)' },
          height: 500,
          hovermode: 'closest'
        }
      }
    }
  }

  // Create scatter plot for retrospective results
  const createRetrospectiveScatter = () => {
    if (!filteredResults?.results) return null
    
    const isClassification = config.forecast_task === 'classification'
    
    if (isClassification) {
      // Classification scatter plot (confusion matrix style)
      const validData = filteredResults.results.filter(r => 
        r.actual_category !== null && r.actual_category !== undefined &&
        r.predicted_category !== null && r.predicted_category !== undefined
      )
      
      if (validData.length === 0) return null
      
      // Calculate accuracy
      const correctPredictions = validData.filter(d => d.actual_category === d.predicted_category).length
      const accuracy = correctPredictions / validData.length
      
      // Add jitter to see overlapping points better
      const jitterStrength = 0.1
      const jitteredData = validData.map(d => ({
        ...d,
        x_jitter: d.actual_category + (Math.random() - 0.5) * jitterStrength * 2,
        y_jitter: d.predicted_category + (Math.random() - 0.5) * jitterStrength * 2
      }))
      
      // Group by site for colors
      const siteGroups = {}
      jitteredData.forEach(d => {
        if (!siteGroups[d.site]) {
          siteGroups[d.site] = []
        }
        siteGroups[d.site].push(d)
      })
      
      const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
      const traces = []
      
      // Add diagonal reference line for perfect predictions
      traces.push({
        x: [-0.5, 3.5],
        y: [-0.5, 3.5],
        mode: 'lines',
        line: { color: 'red', width: 2, dash: 'dash' },
        name: 'Perfect Prediction',
        hoverinfo: 'skip'
      })
      
      // Add scatter points for each site
      Object.entries(siteGroups).forEach(([site, data], index) => {
        traces.push({
          x: data.map(d => d.x_jitter),
          y: data.map(d => d.y_jitter),
          mode: 'markers',
          type: 'scatter',
          name: site,
          marker: { 
            color: colors[index % colors.length],
            size: 8,
            opacity: 0.6
          },
          hovertemplate: '<b>%{text}</b><br>Actual Category: %{customdata[0]}<br>Predicted Category: %{customdata[1]}<extra></extra>',
          text: data.map(d => `${d.site}<br>${d.date}`),
          customdata: data.map(d => {
            const categories = ['Low', 'Moderate', 'High', 'Extreme']
            return [
              categories[d.actual_category] || `Category ${d.actual_category}`,
              categories[d.predicted_category] || `Category ${d.predicted_category}`
            ]
          })
        })
      })
      
      return {
        data: traces,
        layout: {
          title: `Actual vs Predicted Category - ${config.forecast_model} (Accuracy = ${(accuracy * 100).toFixed(1)}%)`,
          xaxis: {
            title: 'Actual DA Category',
            tickmode: 'array',
            tickvals: [0, 1, 2, 3],
            ticktext: ['Low', 'Moderate', 'High', 'Extreme'],
            range: [-0.5, 3.5]
          },
          yaxis: {
            title: 'Predicted DA Category',
            tickmode: 'array',
            tickvals: [0, 1, 2, 3],
            ticktext: ['Low', 'Moderate', 'High', 'Extreme'],
            range: [-0.5, 3.5]
          },
          height: 500,
          showlegend: true,
          legend: { 
            x: 0.02, 
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.8)'
          }
        }
      }
    } else {
      // Regression scatter plot (existing code)
      const validData = filteredResults.results.filter(r => 
        r.actual_da !== null && r.predicted_da !== null
      )

      if (validData.length === 0) return null

      // Calculate range for diagonal line
      const allValues = [...validData.map(d => d.actual_da), ...validData.map(d => d.predicted_da)]
      const minVal = Math.min(...allValues)
      const maxVal = Math.max(...allValues)
      const range = [Math.max(0, minVal - 0.1), maxVal + 0.1]

      // Group data by site for different colors
      const siteGroups = {}
      validData.forEach(d => {
        if (!siteGroups[d.site]) {
          siteGroups[d.site] = []
        }
        siteGroups[d.site].push(d)
      })

      const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
      const traces = []

      // Add diagonal reference line
      traces.push({
        x: range,
        y: range,
        mode: 'lines',
        line: { color: 'red', width: 2, dash: 'dash' },
        name: 'Perfect Prediction',
        hovertemplate: 'Perfect prediction line<extra></extra>'
      })

      // Add scatter points for each site
      Object.entries(siteGroups).forEach(([site, data], index) => {
        traces.push({
          x: data.map(d => d.actual_da),
          y: data.map(d => d.predicted_da),
          mode: 'markers',
          type: 'scatter',
          name: site,
          marker: { 
            color: colors[index % colors.length],
            size: 8,
            opacity: 0.7
          },
          text: data.map(d => `${d.site}<br>${d.date}`),
          hovertemplate: '%{text}<br>Actual: %{x:.2f} μg/g<br>Predicted: %{y:.2f} μg/g<extra></extra>'
        })
      })

      return {
        data: traces,
        layout: {
          title: 'Model Performance: Actual vs Predicted DA Concentrations',
          xaxis: { 
            title: 'Actual DA Concentration (μg/g)',
            range: range
          },
          yaxis: { 
            title: 'Predicted DA Concentration (μg/g)',
            range: range
          },
          height: 500,
          showlegend: true,
          legend: { 
            x: 0.02, 
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.8)'
          }
        }
      }
    }
  }

  // Render different steps
  const renderConfigStep = () => (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="bg-white rounded-lg shadow-md p-8">
        <div className="text-center mb-6">
          <Settings className="w-12 h-12 text-blue-600 mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Configure DATect Forecasting System
          </h1>
          <p className="text-gray-600">
            Select your forecasting mode and parameters before proceeding
          </p>
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4 text-blue-800">System Configuration</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <Clock className="w-4 h-4 inline mr-1" />
                Forecast Mode *
              </label>
              <select
                value={config.forecast_mode}
                onChange={(e) => setConfig({...config, forecast_mode: e.target.value})}
                className="w-full p-3 border border-gray-300 rounded-md text-lg"
              >
                <option value="realtime">Realtime - Interactive single forecasts</option>
                <option value="retrospective">Retrospective - Historical validation analysis</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {config.forecast_mode === 'realtime' 
                  ? 'Generate forecasts for specific dates and sites'
                  : 'Run comprehensive historical analysis with actual vs predicted comparisons'
                }
              </p>
            </div>


            {/* Task and Model selection only for retrospective mode */}
            {config.forecast_mode === 'retrospective' && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <BarChart3 className="w-4 h-4 inline mr-1" />
                    Forecast Task *
                  </label>
                  <select
                    value={config.forecast_task}
                    onChange={(e) => setConfig({...config, forecast_task: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-md text-lg"
                  >
                    <option value="regression">Regression - Predict continuous DA levels (μg/g)</option>
                    <option value="classification">Classification - Predict risk categories (Low/Moderate/High/Extreme)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <Cpu className="w-4 h-4 inline mr-1" />
                    Machine Learning Model *
                  </label>
                  <select
                    value={config.forecast_model}
                    onChange={(e) => setConfig({...config, forecast_model: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-md text-lg"
                  >
                    <option value="xgboost">XGBoost - Advanced gradient boosting (Recommended)</option>
                    <option value="linear">Linear Models - Linear regression & Logistic classification</option>
                  </select>
                </div>
              </>
            )}
          </div>

          <div className="mt-6 pt-4 border-t border-blue-200">
            <button
              onClick={applyConfig}
              disabled={configLoading}
              className="w-full bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 disabled:opacity-50 text-lg font-medium flex items-center justify-center"
            >
              {configLoading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Applying Configuration...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 mr-2" />
                  Apply Configuration & Continue
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )

  const renderRealtimeStep = () => (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header with config summary */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-800 mb-2">
              Realtime Forecasting Interface
            </h1>
            <p className="text-gray-600">
              Mode: <span className="font-medium">{config.forecast_mode}</span> | 
              Task: <span className="font-medium">{config.forecast_task}</span> | 
              Model: <span className="font-medium">{config.forecast_model}</span>
            </p>
          </div>
          <button
            onClick={resetWorkflow}
            className="bg-gray-500 text-white px-4 py-2 rounded-md hover:bg-gray-600"
          >
            Change Config
          </button>
        </div>
      </div>

      {/* Forecast form */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Generate Forecast</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              Forecast Date
            </label>
            <DatePicker
              selected={selectedDate}
              onChange={setSelectedDate}
              minDate={dateRange.min ? new Date(dateRange.min) : null}
              /* Allow forecasting beyond dataset end; no hard max */
              maxDate={null}
              className="w-full p-2 border border-gray-300 rounded-md"
              dateFormat="yyyy-MM-dd"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <MapPin className="w-4 h-4 inline mr-1" />
              Monitoring Site
            </label>
            <Select
              value={selectedSite}
              onChange={setSelectedSite}
              options={sites.map(site => ({ value: site, label: site }))}
              className="text-sm"
              placeholder="Select site..."
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={handleRealtimeForecast}
              disabled={loading || !selectedDate || !selectedSite}
              className="w-full bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center justify-center"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Generating...
                </>
              ) : (
                <>
                  <TrendingUp className="w-4 h-4 mr-2" />
                  Generate Forecast
                </>
              )}
            </button>
          </div>
        </div>
        
        {/* Helpful note */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-4">
          <p className="text-sm text-blue-700">
            💡 <strong>Tip:</strong> Change the date or site above and click "Generate Forecast" to get new predictions. 
            Each forecast is specific to the selected date and monitoring location.
          </p>
        </div>
      </div>
    </div>
  )

  const renderRetrospectiveStep = () => (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-md p-8">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-2 text-center">
            Running Retrospective Analysis
          </h2>
          <p className="text-gray-600 mb-6 text-center">
            Processing historical data with {config.forecast_model} model for {config.forecast_task}...
          </p>
          
          {/* Loading spinner */}
          <div className="text-center mb-6">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          </div>
          
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-700">
              This analysis runs forecasts across historical time periods and compares 
              predicted values against actual measurements to validate model performance.
            </p>
          </div>
        </div>
      </div>
    </div>
  )

  const renderResults = () => {
    if (config.forecast_mode === 'realtime' && forecast) {
      return renderRealtimeResults()
    } else if (config.forecast_mode === 'retrospective' && retrospectiveResults) {
      return renderRetrospectiveResults()
    }
    return null
  }

  const renderRealtimeResults = () => (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Results header */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">Realtime Forecast Results</h2>
          <button
            onClick={() => setCurrentStep('realtime')}
            className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
          >
            Generate Another
          </button>
        </div>
      </div>

      {forecast && forecast.success && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {forecast.regression && (
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="text-lg font-medium text-blue-800 mb-2">
                    🎯 DA Concentration Prediction
                  </h3>
                  <div className="text-2xl font-bold text-blue-600">
                    {forecast.regression.predicted_da?.toFixed(3)} μg/g
                  </div>
                  <p className="text-sm text-gray-600 mt-2">
                    Training samples: {forecast.regression.training_samples}
                  </p>
                </div>
              )}

              {forecast.classification && (
                <div className="bg-green-50 p-4 rounded-lg">
                  <h3 className="text-lg font-medium text-green-800 mb-2">
                    📊 Risk Category Prediction
                  </h3>
                  <div className="text-2xl font-bold text-green-600">
                    {['Low', 'Moderate', 'High', 'Extreme'][forecast.classification.predicted_category] || 'Unknown'}
                  </div>
                  <p className="text-sm text-gray-600 mt-2">
                    Training samples: {forecast.classification.training_samples}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Level Range and Category Range Graphs - Match modular-forecast exactly */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Level Range Graph - Advanced Gradient Visualization */}
              {forecast.graphs && forecast.graphs.level_range && (
                <div>
                  {forecast.graphs.level_range.type === 'gradient_uncertainty' && forecast.graphs.level_range.gradient_plot ? (
                    // Use the advanced gradient plot from backend
                    <Plot
                      {...JSON.parse(forecast.graphs.level_range.gradient_plot)}
                      config={{ responsive: true }}
                      style={{ width: '100%' }}
                    />
                  ) : (
                    // Fallback to simple visualization
                    <Plot
                      data={(() => {
                        const levelData = forecast.graphs.level_range;
                        const quantiles = levelData.gradient_quantiles || {};
                        const q05 = quantiles.q05 || levelData.q05;
                        const q50 = quantiles.q50 || levelData.q50;
                        const q95 = quantiles.q95 || levelData.q95;
                        const xgb_pred = levelData.xgboost_prediction || levelData.predicted_da;
                        
                        const traces = [];
                        const n_segments = 30;
                        const range_width = q95 - q05;
                        const max_distance = Math.max(q50 - q05, q95 - q50) || 1;
                        
                        // Median line (Gradient Boosting Q50)
                        traces.push({
                          x: [q50, q50],
                          y: [0.35, 0.65],
                          mode: 'lines',
                          line: { color: 'rgb(30, 60, 90)', width: 3 },
                          name: 'GB Median (Q50)'
                        });
                        
                        // Range endpoints (GB quantiles)
                        traces.push({
                          x: [q05, q95],
                          y: [0.5, 0.5],
                          mode: 'markers',
                          marker: { size: 12, color: 'rgba(70, 130, 180, 0.4)', symbol: 'line-ns-open' },
                          name: 'GB Range (Q05-Q95)'
                        });
                        
                        // XGBoost point prediction
                        traces.push({
                          x: [xgb_pred],
                          y: [0.5],
                          mode: 'markers',
                          marker: {
                            size: 14,
                            color: 'darkorange',
                            symbol: 'diamond-tall',
                            line: { width: 2, color: 'black' }
                          },
                          name: 'XGBoost Prediction'
                        });
                        
                        return traces;
                      })()}
                      layout={{
                        title: "Advanced DA Level Forecast: Gradient Boosting Quantiles + XGBoost Point",
                        xaxis: { title: "DA Level (μg/L)" },
                        yaxis: { visible: false, range: [0, 1] },
                        showlegend: true,
                        height: 350,
                        plot_bgcolor: 'white',
                        shapes: (() => {
                          const levelData = forecast.graphs.level_range;
                          const quantiles = levelData.gradient_quantiles || {};
                          const q05 = quantiles.q05 || levelData.q05;
                          const q50 = quantiles.q50 || levelData.q50;
                          const q95 = quantiles.q95 || levelData.q95;
                          const n_segments = 30;
                          const range_width = q95 - q05;
                          const max_distance = Math.max(q50 - q05, q95 - q50) || 1;
                          
                          const shapes = [];
                          for (let i = 0; i < n_segments; i++) {
                            const x0 = q05 + (i / n_segments) * range_width;
                            const x1 = q05 + ((i + 1) / n_segments) * range_width;
                            const midpoint = (x0 + x1) / 2;
                            const distance = Math.abs(midpoint - q50);
                            const opacity = Math.max(0.1, Math.min(0.9, 1 - Math.pow(distance / max_distance, 0.5)));
                            
                            shapes.push({
                              type: 'rect',
                              x0: x0,
                              x1: x1,
                              y0: 0.35,
                              y1: 0.65,
                              fillcolor: `rgba(70, 130, 180, ${opacity})`,
                              line: { width: 0 },
                              layer: 'below'
                            });
                          }
                          return shapes;
                        })()
                      }}
                      config={{ responsive: true }}
                      style={{ width: '100%' }}
                    />
                  )}
                </div>
              )}
              
              {/* Category Range Graph - for classification */}
              {forecast.graphs && forecast.graphs.category_range && (
                <div>
                  <Plot
                    data={[{
                      x: forecast.graphs.category_range.category_labels || ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)'],
                      y: forecast.graphs.category_range.class_probabilities || [0, 0, 0, 0],
                      type: 'bar',
                      marker: {
                        color: forecast.graphs.category_range.category_labels?.map((_, i) => 
                          i === forecast.graphs.category_range.predicted_category ? '#2ca02c' : '#1f77b4'
                        ) || ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']
                      },
                      text: forecast.graphs.category_range.class_probabilities?.map(p => `${(p * 100).toFixed(1)}%`) || [],
                      textposition: 'auto'
                    }]}
                    layout={{
                      title: "Category Probability Distribution",
                      yaxis: { title: "Probability", range: [0, 1.1] },
                      xaxis: { title: "Category" },
                      showlegend: false,
                      height: 400
                    }}
                    config={{ responsive: true }}
                    style={{ width: '100%' }}
                  />
                </div>
              )}
            </div>
          </div>
          
          {/* Feature Importance Graph - if available */}
          {(forecast.regression?.feature_importance || forecast.classification?.feature_importance) && (
            <div className="bg-white rounded-lg shadow-md p-6 mt-6">
              <h3 className="text-lg font-semibold mb-4">Top Feature Importance</h3>
              <Plot
                data={[{
                  x: (forecast.regression?.feature_importance || forecast.classification?.feature_importance)
                    ?.slice(0, 15)
                    ?.map(f => f.importance) || [],
                  y: (forecast.regression?.feature_importance || forecast.classification?.feature_importance)
                    ?.slice(0, 15)
                    ?.map(f => f.feature) || [],
                  type: 'bar',
                  orientation: 'h',
                  marker: { color: 'steelblue' }
                }]}
                layout={{
                  title: "Top Feature Importance",
                  xaxis_title: "Importance Score",
                  yaxis_title: "Features",
                  height: 400,
                  yaxis: { categoryorder: 'total ascending' }
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )

  const renderRetrospectiveResults = () => {
    const timeSeriesData = createRetrospectiveTimeSeries()
    const scatterData = createRetrospectiveScatter()
    
    return (
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Results header */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-semibold">Retrospective Analysis Results</h2>
              <p className="text-gray-600">
                Model: {config.forecast_model} | Task: {config.forecast_task}
              </p>
            </div>
            <button
              onClick={resetWorkflow}
              className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
            >
              New Analysis
            </button>
          </div>
        </div>

        {/* Site filtering controls - Simple dropdown that filters existing results */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium text-gray-700">
              <MapPin className="w-4 h-4 inline mr-1" />
              Filter by Site:
            </label>
            <select
              value={selectedSiteFilter}
              onChange={(e) => filterResultsBySite(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm"
            >
              <option value="all">All Sites</option>
              {sites.map(site => (
                <option key={site} value={site}>{site}</option>
              ))}
            </select>
            <span className="text-sm text-gray-600">
              Showing {filteredResults?.results?.length || 0} forecasts
            </span>
          </div>
        </div>

        {/* Summary statistics */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Performance Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">{filteredResults?.summary?.total_forecasts || 0}</div>
              <div className="text-sm text-gray-600">Total Forecasts</div>
            </div>
            {filteredResults?.summary?.r2_score !== undefined && (
              <div className="bg-green-50 p-4 rounded-lg text-center">
                <div className="text-2xl font-bold text-green-600">{filteredResults.summary.r2_score.toFixed(3)}</div>
                <div className="text-sm text-gray-600">R² Score</div>
              </div>
            )}
            {filteredResults?.summary?.mae !== undefined && (
              <div className="bg-yellow-50 p-4 rounded-lg text-center">
                <div className="text-2xl font-bold text-yellow-600">{filteredResults.summary.mae.toFixed(2)}</div>
                <div className="text-sm text-gray-600">MAE (μg/g)</div>
              </div>
            )}
            {filteredResults?.summary?.accuracy !== undefined && (
              <div className="bg-purple-50 p-4 rounded-lg text-center">
                <div className="text-2xl font-bold text-purple-600">{(filteredResults.summary.accuracy * 100).toFixed(1)}%</div>
                <div className="text-sm text-gray-600">Accuracy</div>
              </div>
            )}
          </div>
        </div>

        {/* Time series plot */}
        {timeSeriesData && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">
              Actual vs Predicted Time Series
              {(config.selected_sites || []).length < sites.length && (config.selected_sites || []).length > 0 && (
                <span className="text-sm font-normal text-gray-600 ml-2">
                  ({(config.selected_sites || []).length} of {sites.length} sites)
                </span>
              )}
            </h3>
            <Plot
              data={timeSeriesData.data}
              layout={timeSeriesData.layout}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        )}

        {/* Scatter plot */}
        {scatterData && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Model Performance Scatter Plot</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Plot
                data={scatterData.data}
                layout={scatterData.layout}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium mb-3">Interpretation Guide</h4>
                <ul className="text-sm space-y-2 text-gray-700">
                  <li>• Points closer to the diagonal line indicate better predictions</li>
                  <li>• Scattered points suggest higher prediction uncertainty</li>
                  <li>• Color represents different monitoring sites</li>
                  <li>• R² closer to 1.0 indicates better model performance</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // Main render
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      {/* Error Display */}
      {error && (
        <div className="max-w-4xl mx-auto mb-6">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center">
              <AlertTriangle className="w-5 h-5 text-red-600 mr-2" />
              <span className="text-red-800">{error}</span>
            </div>
          </div>
        </div>
      )}

      {/* Render current step */}
      {currentStep === 'config' && renderConfigStep()}
      {currentStep === 'realtime' && renderRealtimeStep()}
      {currentStep === 'retrospective' && renderRetrospectiveStep()}
      {currentStep === 'results' && renderResults()}
    </div>
  )
}

export default Dashboard