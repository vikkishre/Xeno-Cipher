export class Chart {
  constructor(ctx, config) {
    this.ctx = ctx
    this.config = config
    this.chart = new ChartImplementation(ctx, config)
  }

  update() {
    this.chart.update()
  }

  destroy() {
    this.chart.destroy()
  }
}

// Dummy implementation to avoid breaking the code
class ChartImplementation {
  constructor(ctx, config) {
    this.ctx = ctx
    this.config = config
  }

  update() {
    //console.log('Chart updated');
  }

  destroy() {
    //console.log('Chart destroyed');
  }
}

export const ChartContainer = () => {}
export const ChartTooltip = () => {}
export const ChartTooltipContent = () => {}
export const ChartLegend = () => {}
export const ChartLegendContent = () => {}
export const ChartStyle = () => {}
