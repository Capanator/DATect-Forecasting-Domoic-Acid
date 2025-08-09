import React, { useState, useEffect, useRef } from 'react'
import { MapPin, BarChart3, Activity } from 'lucide-react'
import Select from 'react-select'
import Plot from 'react-plotly.js'
import api from '../services/api'
import { plotConfig, plotConfigSquare, getPlotFilename, scaleLayoutForExport } from '../utils/plotConfig'

const Historical = () => {
  const [sites, setSites] = useState([])
  const [selectedSite, setSelectedSite] = useState(null)
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(false)
  const [visualizationType, setVisualizationType] = useState('correlation')
  const [siteScope, setSiteScope] = useState('single') // 'single' or 'all'
  const [visualizationData, setVisualizationData] = useState(null)
  const [loadingVisualization, setLoadingVisualization] = useState(false)

  const visualizationOptions = [
    { value: 'correlation', label: 'Correlation Heatmap', icon: BarChart3 },
    { value: 'sensitivity', label: 'Sensitivity Analysis', icon: BarChart3 },
    { value: 'comparison', label: 'DA vs Pseudo-nitzschia', icon: Activity },
    { value: 'waterfall', label: 'Waterfall Plot', icon: BarChart3 },
    { value: 'spectral', label: 'Spectral Analysis', icon: Activity },
  ]

  const siteScopeOptions = [
    { value: 'single', label: 'Single Site' },
    { value: 'all', label: 'All Sites' }
  ]

  useEffect(() => {
    loadSites()
  }, [])

  const loadSites = async () => {
    try {
      const response = await api.get('/api/sites')
      const sitesList = response.data.sites
      setSites(sitesList)
      
      if (sitesList.length > 0) {
        setSelectedSite({ value: sitesList[0], label: sitesList[0] })
      }
    } catch (err) {
      console.error('Failed to load sites:', err)
    }
  }

  const loadHistoricalData = async () => {
    if (!selectedSite && siteScope === 'single') return

    setLoading(true)
    try {
      const params = new URLSearchParams({
        limit: '10000'
      })

      if (siteScope === 'single' && selectedSite) {
        const response = await api.get(`/api/historical/${selectedSite.value}?${params}`)
        setData(response.data.data)
      } else {
        // Load data for all sites
        const response = await api.get(`/api/historical/all?${params}`)
        setData(response.data.data)
      }
    } catch (err) {
      console.error('Failed to load historical data:', err)
    } finally {
      setLoading(false)
    }
  }

  const loadVisualizationData = async () => {
    setLoadingVisualization(true)
    try {
      const params = new URLSearchParams()

      let endpoint = ''
      if (visualizationType === 'correlation') {
        endpoint = siteScope === 'single' && selectedSite 
          ? `/api/visualizations/correlation/${selectedSite.value}` 
          : '/api/visualizations/correlation/all'
      } else if (visualizationType === 'sensitivity') {
        endpoint = '/api/visualizations/sensitivity'
      } else if (visualizationType === 'comparison') {
        // DA vs Pseudo-nitzschia only supports single site
        endpoint = selectedSite
          ? `/api/visualizations/comparison/${selectedSite.value}`
          : null
      } else if (visualizationType === 'waterfall') {
        endpoint = '/api/visualizations/waterfall'
      } else if (visualizationType === 'spectral') {
        endpoint = siteScope === 'single' && selectedSite
          ? `/api/visualizations/spectral/${selectedSite.value}`
          : '/api/visualizations/spectral/all'
      }

      if (endpoint) {
        const response = await api.get(`${endpoint}?${params}`)
        setVisualizationData(response.data)
      }
    } catch (err) {
      console.error('Failed to load visualization data:', err)
      setVisualizationData(null)
    } finally {
      setLoadingVisualization(false)
    }
  }

  useEffect(() => {
    if (siteScope === 'single' && selectedSite) {
      loadHistoricalData()
    } else if (siteScope === 'all') {
      loadHistoricalData()
    }
  }, [selectedSite, siteScope])

  useEffect(() => {
    // Force single site for comparison
    if (forceSingleSite && siteScope === 'all') {
      setSiteScope('single')
    }
    loadVisualizationData()
  }, [visualizationType, selectedSite, siteScope])

  const siteOptions = sites.map(site => ({ value: site, label: site }))

  const createTimeSeries = () => {
    if (!data || data.length === 0) return null

    if (siteScope === 'single') {
      // Filter out null/undefined DA values
      const validData = data.filter(d => d.da !== null && d.da !== undefined)
      if (validData.length === 0) return null
      
      return {
        data: [{
          x: validData.map(d => d.date),
          y: validData.map(d => d.da),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'DA Concentration',
          line: { color: '#2563eb' },
          marker: { size: 4 }
        }],
        layout: {
          title: `Historical DA Concentrations - ${selectedSite?.label}`,
          xaxis: { title: 'Date' },
          yaxis: { title: 'DA Concentration (μg/g)' },
          height: 500
        }
      }
    } else {
      // Group data by site for all sites view
      const siteData = {}
      data.forEach(d => {
        if (d.da !== null && d.da !== undefined) {
          if (!siteData[d.site]) {
            siteData[d.site] = { dates: [], values: [] }
          }
          siteData[d.site].dates.push(d.date)
          siteData[d.site].values.push(d.da)
        }
      })

      if (Object.keys(siteData).length === 0) return null

      const traces = Object.keys(siteData).map(site => ({
        x: siteData[site].dates,
        y: siteData[site].values,
        type: 'scatter',
        mode: 'lines',
        name: site,
        line: { width: 2 }
      }))

      return {
        data: traces,
        layout: {
          title: 'Historical DA Concentrations - All Sites',
          xaxis: { title: 'Date' },
          yaxis: { title: 'DA Concentration (μg/g)' },
          height: 500
        }
      }
    }
  }

  const renderVisualization = () => {
    if (visualizationData) {
      // Handle both single plot and multiple plots
      if (visualizationData.plot) {
        // Single plot visualization
        // Center heatmaps which tend to be square
        const isHeatmap = visualizationType === 'correlation'
        const config = isHeatmap ? {
          ...plotConfigSquare,
          toImageButtonOptions: {
            ...plotConfigSquare.toImageButtonOptions,
            filename: getPlotFilename(`${visualizationType}_${siteScope === 'all' ? 'all-sites' : selectedSite?.value || 'plot'}`)
          }
        } : {
          ...plotConfig,
          toImageButtonOptions: {
            ...plotConfig.toImageButtonOptions,
            filename: getPlotFilename(`${visualizationType}_${siteScope === 'all' ? 'all-sites' : selectedSite?.value || 'plot'}`)
          }
        }
        
        return (
          <div className={isHeatmap ? "flex justify-center" : ""}>
            <Plot
              data={visualizationData.plot.data}
              layout={visualizationData.plot.layout}
              config={config}
              className={isHeatmap ? "" : "w-full"}
              style={isHeatmap ? { maxWidth: '800px' } : {}}
            />
          </div>
        )
      } else if (visualizationData.plots && Array.isArray(visualizationData.plots)) {
        // Multiple plots visualization
        return (
          <div className="space-y-4">
            {visualizationData.plots.map((plot, index) => {
              const plotConfigWithFilename = {
                ...plotConfig,
                toImageButtonOptions: {
                  ...plotConfig.toImageButtonOptions,
                  filename: getPlotFilename(`${visualizationType}_plot${index + 1}`)
                }
              }
              return (
                <div key={index} className="flex justify-center">
                  <Plot
                    data={plot.data}
                    layout={plot.layout}
                    config={plotConfigWithFilename}
                    className="w-full"
                    style={{ maxWidth: '1000px' }}
                  />
                </div>
              )
            })}
          </div>
        )
      }
    }
    return <p className="text-center text-gray-500">Loading visualization...</p>
  }

  // Check if current visualization supports site scope selection
  const supportsSiteScope = ['correlation', 'spectral'].includes(visualizationType)
  // DA vs Pseudo-nitzschia only supports single site
  const forceSingleSite = visualizationType === 'comparison'
  // Waterfall plot is all-sites only; hide site controls
  const hideSiteControls = visualizationType === 'waterfall'

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Historical Data Analysis
        </h1>
        <p className="text-gray-600">
          Explore historical domoic acid measurements and advanced visualizations
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Analysis Parameters</h2>
        
        {/* Visualization Type Selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Visualization Type
          </label>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
            {visualizationOptions.map(option => (
              <button
                key={option.value}
                onClick={() => setVisualizationType(option.value)}
                className={`px-4 py-2 rounded-lg border transition-colors ${
                  visualizationType === option.value
                    ? 'bg-blue-500 text-white border-blue-500'
                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                }`}
              >
                <div className="flex items-center justify-center space-x-2">
                  <option.icon className="w-4 h-4" />
                  <span className="text-sm">{option.label}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Site Scope Selector */}
          {supportsSiteScope && !hideSiteControls && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <BarChart3 className="w-4 h-4 inline mr-1" />
                Site Scope
              </label>
              <Select
                value={siteScopeOptions.find(opt => opt.value === siteScope)}
                onChange={(option) => setSiteScope(option.value)}
                options={siteScopeOptions}
                className="text-sm"
              />
            </div>
          )}

          {/* Site Selector - show for single site scope or when forced */}
          {(siteScope === 'single' || forceSingleSite) && !hideSiteControls && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <MapPin className="w-4 h-4 inline mr-1" />
                Monitoring Site
              </label>
              <Select
                value={selectedSite}
                onChange={setSelectedSite}
                options={siteOptions}
                className="text-sm"
                placeholder="Select site..."
              />
            </div>
          )}
        </div>
      </div>

      {/* Visualization */}
      <div className="bg-white rounded-lg shadow-md p-6">
        {(loading || loadingVisualization) ? (
          <div className="text-center py-8">
            <p className="text-gray-600">Loading visualization...</p>
          </div>
        ) : (
          renderVisualization()
        )}
      </div>
    </div>
  )
}

export default Historical