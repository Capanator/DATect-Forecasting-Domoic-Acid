/**
 * High-quality plot configuration for scientific publications
 * Ensures all plots can be exported at publication-ready resolution
 */

export const plotConfig = {
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
  toImageButtonOptions: {
    format: 'png',
    filename: 'datect_plot',
    height: 2400,  // 8 inches at 300 DPI
    width: 3200,   // 10.67 inches at 300 DPI
    scale: 4       // 4x scaling for ultra-high resolution
  }
}

export const plotConfigSmall = {
  ...plotConfig,
  toImageButtonOptions: {
    ...plotConfig.toImageButtonOptions,
    height: 1800,  // 6 inches at 300 DPI
    width: 2400,   // 8 inches at 300 DPI
  }
}

export const plotConfigSquare = {
  ...plotConfig,
  toImageButtonOptions: {
    ...plotConfig.toImageButtonOptions,
    height: 2400,  // 8 inches at 300 DPI
    width: 2400,   // 8 inches at 300 DPI (square for heatmaps)
  }
}

/**
 * Get filename with timestamp for unique downloads
 */
export const getPlotFilename = (prefix = 'datect') => {
  const date = new Date().toISOString().split('T')[0]
  return `${prefix}_${date}`
}