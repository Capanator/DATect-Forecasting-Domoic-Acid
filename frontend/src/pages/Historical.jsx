import React, { useState, useEffect } from 'react'
import { Calendar, MapPin, Download } from 'lucide-react'
import Select from 'react-select'
import Plot from 'react-plotly.js'
import DatePicker from 'react-datepicker'
import { format, subMonths } from 'date-fns'
import api from '../services/api'

const Historical = () => {
  const [sites, setSites] = useState([])
  const [selectedSite, setSelectedSite] = useState(null)
  const [startDate, setStartDate] = useState(null)
  const [endDate, setEndDate] = useState(null)
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(false)

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
        
        // Set default date range (last 6 months)
        const endDate = new Date(response.data.date_range.max)
        const startDate = subMonths(endDate, 6)
        setStartDate(startDate)
        setEndDate(endDate)
      }
    } catch (err) {
      console.error('Failed to load sites:', err)
    }
  }

  const loadHistoricalData = async () => {
    if (!selectedSite) return

    setLoading(true)
    try {
      const params = new URLSearchParams({
        limit: '1000'
      })
      
      if (startDate) {
        params.append('start_date', format(startDate, 'yyyy-MM-dd'))
      }
      if (endDate) {
        params.append('end_date', format(endDate, 'yyyy-MM-dd'))
      }

      const response = await api.get(`/api/historical/${selectedSite.value}?${params}`)
      setData(response.data.data)
    } catch (err) {
      console.error('Failed to load historical data:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (selectedSite) {
      loadHistoricalData()
    }
  }, [selectedSite, startDate, endDate])

  const siteOptions = sites.map(site => ({ value: site, label: site }))

  const createTimeSeries = () => {
    if (!data || data.length === 0) return null

    return {
      data: [{
        x: data.map(d => d.date),
        y: data.map(d => d.da),
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
        height: 400
      }
    }
  }

  const createCategoryDistribution = () => {
    if (!data || data.length === 0) return null

    const categoryCounts = data.reduce((acc, d) => {
      if (d['da-category'] !== undefined) {
        acc[d['da-category']] = (acc[d['da-category']] || 0) + 1
      }
      return acc
    }, {})

    const categoryLabels = ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']
    const values = Object.keys(categoryCounts).map(key => categoryCounts[key] || 0)

    return {
      data: [{
        type: 'pie',
        labels: categoryLabels,
        values: values,
        hole: 0.4,
        marker: {
          colors: ['#10b981', '#f59e0b', '#ef4444', '#dc2626']
        }
      }],
      layout: {
        title: `Risk Category Distribution - ${selectedSite?.label}`,
        height: 400
      }
    }
  }

  const plotData = createTimeSeries()
  const categoryData = createCategoryDistribution()

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Historical Data Analysis
        </h1>
        <p className="text-gray-600">
          Explore historical domoic acid measurements and trends
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Analysis Parameters</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
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

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              Start Date
            </label>
            <DatePicker
              selected={startDate}
              onChange={setStartDate}
              className="w-full p-2 border border-gray-300 rounded-md"
              dateFormat="yyyy-MM-dd"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              End Date
            </label>
            <DatePicker
              selected={endDate}
              onChange={setEndDate}
              className="w-full p-2 border border-gray-300 rounded-md"
              dateFormat="yyyy-MM-dd"
            />
          </div>
        </div>
      </div>

      {/* Data Summary */}
      {data.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Data Summary</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-gray-600">Total Samples</h3>
              <p className="text-2xl font-bold text-blue-600">{data.length}</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-gray-600">Average DA</h3>
              <p className="text-2xl font-bold text-green-600">
                {(data.reduce((sum, d) => sum + d.da, 0) / data.length).toFixed(2)} μg/g
              </p>
            </div>
            <div className="bg-yellow-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-gray-600">Max DA</h3>
              <p className="text-2xl font-bold text-yellow-600">
                {Math.max(...data.map(d => d.da)).toFixed(2)} μg/g
              </p>
            </div>
            <div className="bg-red-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-gray-600">Date Range</h3>
              <p className="text-sm font-bold text-red-600">
                {data.length > 0 ? `${data[0].date} to ${data[data.length - 1].date}` : 'N/A'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Time Series Plot */}
      {plotData && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <Plot
            data={plotData.data}
            layout={plotData.layout}
            config={{ responsive: true }}
            className="w-full"
          />
        </div>
      )}

      {/* Category Distribution */}
      {categoryData && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <Plot
            data={categoryData.data}
            layout={categoryData.layout}
            config={{ responsive: true }}
            className="w-full"
          />
        </div>
      )}

      {loading && (
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <p className="text-gray-600">Loading data...</p>
        </div>
      )}
    </div>
  )
}

export default Historical