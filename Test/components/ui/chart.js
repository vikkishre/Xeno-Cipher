export const Chart = (element, config) => {
    if (typeof element === "string") {
      element = document.getElementById(element)
    }
  
    if (!element) {
      console.error("Chart: Invalid element provided.")
      return null
    }
  
    try {
      return new ChartJs(element, config)
    } catch (error) {
      console.error("Chart: Failed to create chart.", error)
      return null
    }
  }
  
  class ChartJs {
    constructor(element, config) {
      this.element = element
      this.config = config
      this.chart = new window.Chart(this.element, this.config)
    }
  
    destroy() {
      if (this.chart) {
        this.chart.destroy()
      }
    }
  }
  
  export const ChartContainer = ({ children }) => {
    return <div className="chart-container">{children}</div>
  }
  
  export const ChartTooltip = ({ children }) => {
    return <div className="tooltip">{children}</div>
  }
  
  export const ChartTooltipContent = ({ children }) => {
    return <span className="tooltiptext">{children}</span>
  }
  
  export const ChartLegend = ({ children }) => {
    return <div className="chart-legend">{children}</div>
  }
  
  export const ChartLegendContent = ({ children }) => {
    return <div className="chart-legend-content">{children}</div>
  }
  
  export const ChartStyle = () => {
    return null
  }
  