/**
 * High-quality plot configuration for scientific publications
 * Ensures all plots can be exported at publication-ready resolution
 */

// Font scaling function to adjust layout for high-res export
export const scaleLayoutForExport = (layout, scaleFactor = 3) => {
  const scaledLayout = JSON.parse(JSON.stringify(layout)) // Deep clone
  
  // Scale all font sizes
  if (scaledLayout.font) {
    scaledLayout.font.size = (scaledLayout.font.size || 12) * scaleFactor
  }
  if (scaledLayout.title) {
    if (typeof scaledLayout.title === 'string') {
      scaledLayout.title = { text: scaledLayout.title }
    }
    scaledLayout.title.font = scaledLayout.title.font || {}
    scaledLayout.title.font.size = (scaledLayout.title.font.size || 16) * scaleFactor
  }
  
  // Scale axis fonts
  ['xaxis', 'yaxis', 'xaxis2', 'yaxis2'].forEach(axis => {
    if (scaledLayout[axis]) {
      if (scaledLayout[axis].title) {
        if (typeof scaledLayout[axis].title === 'string') {
          scaledLayout[axis].title = { text: scaledLayout[axis].title }
        }
        scaledLayout[axis].title.font = scaledLayout[axis].title.font || {}
        scaledLayout[axis].title.font.size = (scaledLayout[axis].title.font.size || 14) * scaleFactor
      }
      if (scaledLayout[axis].tickfont) {
        scaledLayout[axis].tickfont.size = (scaledLayout[axis].tickfont.size || 11) * scaleFactor
      } else {
        scaledLayout[axis].tickfont = { size: 11 * scaleFactor }
      }
    }
  })
  
  // Scale legend
  if (scaledLayout.legend) {
    scaledLayout.legend.font = scaledLayout.legend.font || {}
    scaledLayout.legend.font.size = (scaledLayout.legend.font.size || 12) * scaleFactor
  }
  
  // Scale annotations
  if (scaledLayout.annotations) {
    scaledLayout.annotations.forEach(ann => {
      if (ann.font) {
        ann.font.size = (ann.font.size || 12) * scaleFactor
      }
    })
  }
  
  return scaledLayout
}

export const plotConfig = {
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
  toImageButtonOptions: {
    format: 'png',
    filename: 'datect_plot',
    scale: 10  // Scale up 10x for higher resolution while maintaining aspect ratio
  }
}

export const plotConfigSmall = {
  ...plotConfig,
  toImageButtonOptions: {
    ...plotConfig.toImageButtonOptions,
    scale: 10  // Scale up 10x for higher resolution
  }
}

export const plotConfigSquare = {
  ...plotConfig,
  toImageButtonOptions: {
    ...plotConfig.toImageButtonOptions,
    scale: 10  // Scale up 10x for higher resolution
  }
}

/**
 * Get filename with timestamp for unique downloads
 */
export const getPlotFilename = (prefix = 'datect') => {
  const date = new Date().toISOString().split('T')[0]
  return `${prefix}_${date}`
}