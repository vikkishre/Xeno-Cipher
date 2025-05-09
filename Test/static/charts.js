// Tab switching functionality
document.addEventListener("DOMContentLoaded", () => {
  const tabButtons = document.querySelectorAll(".tab-button")
  const tabContents = document.querySelectorAll(".tab-content")

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      // Remove active class from all buttons and contents
      tabButtons.forEach((btn) => btn.classList.remove("active"))
      tabContents.forEach((content) => content.classList.remove("active"))

      // Add active class to clicked button and corresponding content
      button.classList.add("active")
      const tabId = button.getAttribute("data-tab")
      document.getElementById(tabId).classList.add("active")

      // Initialize or update charts for the active tab
      if (tabId === "overview") {
        initSecurityEquivalenceChart()
      } else if (tabId === "security") {
        initQuantumSecurityChart()
      } else if (tabId === "performance") {
        initSpeedComparisonChart()
        initMemoryUsageChart()
        initKeyExchangeChart()
      } else if (tabId === "attacks") {
        initAttackResistanceChart()
        initBruteForceChart()
        initQuantumAttackChart()
      } else if (tabId === "features") {
        initComplexityChart()
      }
    })
  })

  // Initialize charts for the default active tab
  initSecurityEquivalenceChart()

  // Initialize all charts with a small delay to ensure DOM is ready
  setTimeout(() => {
    initQuantumSecurityChart()
    initSpeedComparisonChart()
    initMemoryUsageChart()
    initKeyExchangeChart()
    initAttackResistanceChart()
    initBruteForceChart()
    initQuantumAttackChart()
    initComplexityChart()
  }, 500)
})

// Chart initialization functions
function initSecurityEquivalenceChart() {
  const ctx = document.getElementById("securityEquivalenceChart")
  if (!ctx) return

  // Destroy existing chart if it exists
  if (ctx.chart) {
    ctx.chart.destroy()
  }

  // Create gradient for bars
  const chartCtx = ctx.getContext("2d")
  const blueGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  blueGradient.addColorStop(0, "rgba(59, 130, 246, 0.8)")
  blueGradient.addColorStop(1, "rgba(59, 130, 246, 0.2)")

  const greenGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  greenGradient.addColorStop(0, "rgba(16, 185, 129, 0.8)")
  greenGradient.addColorStop(1, "rgba(16, 185, 129, 0.2)")

  const redGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  redGradient.addColorStop(0, "rgba(239, 68, 68, 0.8)")
  redGradient.addColorStop(1, "rgba(239, 68, 68, 0.2)")

  const yellowGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  yellowGradient.addColorStop(0, "rgba(245, 158, 11, 0.8)")
  yellowGradient.addColorStop(1, "rgba(245, 158, 11, 0.2)")

  ctx.chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["XenoCipher", "AES", "RSA", "ECC"],
      datasets: [
        {
          label: "Key Size (bits)",
          data: [256, 256, 2048, 256],
          backgroundColor: [blueGradient, greenGradient, redGradient, yellowGradient],
          borderColor: ["rgb(59, 130, 246)", "rgb(16, 185, 129)", "rgb(239, 68, 68)", "rgb(245, 158, 11)"],
          borderWidth: 1,
        },
        {
          label: "Effective Security (bits)",
          data: [256, 256, 112, 128],
          backgroundColor: [
            "rgba(59, 130, 246, 0.5)",
            "rgba(16, 185, 129, 0.5)",
            "rgba(239, 68, 68, 0.5)",
            "rgba(245, 158, 11, 0.5)",
          ],
          borderColor: ["rgb(59, 130, 246)", "rgb(16, 185, 129)", "rgb(239, 68, 68)", "rgb(245, 158, 11)"],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Bits",
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Key Size vs. Effective Security",
          font: {
            size: 14,
          },
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${context.raw} bits`,
          },
        },
      },
    },
  })
}

function initQuantumSecurityChart() {
  const ctx = document.getElementById("quantumSecurityChart")
  if (!ctx) return

  // Destroy existing chart if it exists
  if (ctx.chart) {
    ctx.chart.destroy()
  }

  // Create gradient for bars
  const chartCtx = ctx.getContext("2d")
  const blueGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  blueGradient.addColorStop(0, "rgba(59, 130, 246, 0.8)")
  blueGradient.addColorStop(1, "rgba(59, 130, 246, 0.2)")

  const greenGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  greenGradient.addColorStop(0, "rgba(16, 185, 129, 0.8)")
  greenGradient.addColorStop(1, "rgba(16, 185, 129, 0.2)")

  const redGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  redGradient.addColorStop(0, "rgba(239, 68, 68, 0.8)")
  redGradient.addColorStop(1, "rgba(239, 68, 68, 0.2)")

  const yellowGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  yellowGradient.addColorStop(0, "rgba(245, 158, 11, 0.8)")
  yellowGradient.addColorStop(1, "rgba(245, 158, 11, 0.2)")

  ctx.chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["XenoCipher", "AES", "RSA", "ECC"],
      datasets: [
        {
          label: "Classical Security (bits)",
          data: [256, 256, 112, 128],
          backgroundColor: [blueGradient, greenGradient, redGradient, yellowGradient],
          borderColor: ["rgb(59, 130, 246)", "rgb(16, 185, 129)", "rgb(239, 68, 68)", "rgb(245, 158, 11)"],
          borderWidth: 1,
        },
        {
          label: "Quantum Security (bits)",
          data: [128, 128, 0, 0],
          backgroundColor: [
            "rgba(59, 130, 246, 0.5)",
            "rgba(16, 185, 129, 0.5)",
            "rgba(239, 68, 68, 0.5)",
            "rgba(245, 158, 11, 0.5)",
          ],
          borderColor: ["rgb(59, 130, 246)", "rgb(16, 185, 129)", "rgb(239, 68, 68)", "rgb(245, 158, 11)"],
          borderWidth: 1,
          borderDash: [5, 5],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Security Level (bits)",
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Quantum vs. Classical Security",
          font: {
            size: 14,
          },
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${context.raw} bits`,
          },
        },
      },
    },
  })
}

function initSpeedComparisonChart() {
  const ctx = document.getElementById("speedComparisonChart")
  if (!ctx) return

  // Destroy existing chart if it exists
  if (ctx.chart) {
    ctx.chart.destroy()
  }

  ctx.chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["XenoCipher (Normal)", "XenoCipher (ZTM)", "AES-256", "RSA-2048", "ECC-256"],
      datasets: [
        {
          label: "Encryption Speed (MB/s)",
          data: [85, 65, 125, 0.4, 3.2],
          backgroundColor: "rgba(59, 130, 246, 0.7)",
          borderColor: "rgb(59, 130, 246)",
          borderWidth: 1,
        },
        {
          label: "Decryption Speed (MB/s)",
          data: [90, 70, 130, 8.5, 3.5],
          backgroundColor: "rgba(16, 185, 129, 0.7)",
          borderColor: "rgb(16, 185, 129)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Speed (MB/s)",
          },
          type: "logarithmic",
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Encryption/Decryption Speed (logarithmic scale)",
          font: {
            size: 14,
          },
        },
      },
    },
  })
}

function initMemoryUsageChart() {
  const ctx = document.getElementById("memoryUsageChart")
  if (!ctx) return

  // Destroy existing chart if it exists
  if (ctx.chart) {
    ctx.chart.destroy()
  }

  ctx.chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["XenoCipher (Normal)", "XenoCipher (ZTM)", "AES-256", "RSA-2048", "ECC-256"],
      datasets: [
        {
          label: "Memory Usage (KB)",
          data: [24, 32, 4, 256, 32],
          backgroundColor: "rgba(245, 158, 11, 0.7)",
          borderColor: "rgb(245, 158, 11)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Memory (KB)",
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Memory Usage Comparison",
          font: {
            size: 14,
          },
        },
      },
    },
  })
}

function initKeyExchangeChart() {
  const ctx = document.getElementById("keyExchangeChart")
  if (!ctx) return

  // Destroy existing chart if it exists
  if (ctx.chart) {
    ctx.chart.destroy()
  }

  ctx.chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["XenoCipher (NTRU)", "RSA-2048", "ECC-256"],
      datasets: [
        {
          label: "Key Generation Time (ms)",
          data: [15, 360, 25],
          backgroundColor: "rgba(59, 130, 246, 0.7)",
          borderColor: "rgb(59, 130, 246)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Time (ms)",
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Key Exchange Performance",
          font: {
            size: 14,
          },
        },
      },
    },
  })
}

function initAttackResistanceChart() {
  const ctx = document.getElementById("attackResistanceChart")
  if (!ctx) return

  // Destroy existing chart if it exists
  if (ctx.chart) {
    ctx.chart.destroy()
  }

  // Create gradients for datasets
  const chartCtx = ctx.getContext("2d")
  const blueGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  blueGradient.addColorStop(0, "rgba(59, 130, 246, 0.2)")
  blueGradient.addColorStop(1, "rgba(59, 130, 246, 0.05)")

  const greenGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  greenGradient.addColorStop(0, "rgba(16, 185, 129, 0.2)")
  greenGradient.addColorStop(1, "rgba(16, 185, 129, 0.05)")

  const redGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  redGradient.addColorStop(0, "rgba(239, 68, 68, 0.2)")
  redGradient.addColorStop(1, "rgba(239, 68, 68, 0.05)")

  const yellowGradient = chartCtx.createLinearGradient(0, 0, 0, 400)
  yellowGradient.addColorStop(0, "rgba(245, 158, 11, 0.2)")
  yellowGradient.addColorStop(1, "rgba(245, 158, 11, 0.05)")

  ctx.chart = new Chart(ctx, {
    type: "radar",
    data: {
      labels: ["Brute Force", "Quantum", "Side-Channel", "Chosen Plaintext", "MITM"],
      datasets: [
        {
          label: "XenoCipher",
          data: [9.8, 9.2, 7.5, 9.0, 8.5],
          backgroundColor: blueGradient,
          borderColor: "rgb(59, 130, 246)",
          pointBackgroundColor: "rgb(59, 130, 246)",
          borderWidth: 2,
        },
        {
          label: "AES",
          data: [9.9, 5.0, 6.5, 9.5, 8.0],
          backgroundColor: greenGradient,
          borderColor: "rgb(16, 185, 129)",
          pointBackgroundColor: "rgb(16, 185, 129)",
          borderWidth: 2,
        },
        {
          label: "RSA",
          data: [9.5, 1.0, 4.2, 9.0, 7.0],
          backgroundColor: redGradient,
          borderColor: "rgb(239, 68, 68)",
          pointBackgroundColor: "rgb(239, 68, 68)",
          borderWidth: 2,
        },
        {
          label: "ECC",
          data: [9.7, 1.0, 5.8, 9.2, 7.5],
          backgroundColor: yellowGradient,
          borderColor: "rgb(245, 158, 11)",
          pointBackgroundColor: "rgb(245, 158, 11)",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true,
          max: 10,
          ticks: {
            stepSize: 2,
          },
          pointLabels: {
            font: {
              size: 12,
              weight: "bold",
            },
          },
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
          angleLines: {
            color: "rgba(0, 0, 0, 0.1)",
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Attack Resistance Comparison (Higher is Better)",
          font: {
            size: 16,
            weight: "bold",
          },
        },
        legend: {
          position: "bottom",
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${context.raw}/10`,
          },
        },
      },
    },
  })
}

function initBruteForceChart() {
  const ctx = document.getElementById("bruteForceChart")
  if (!ctx) return

  // Destroy existing chart if it exists
  if (ctx.chart) {
    ctx.chart.destroy()
  }

  ctx.chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["XenoCipher", "AES", "RSA", "ECC"],
      datasets: [
        {
          label: "Years to Break (log scale)",
          data: [63, 63, 33, 38],
          backgroundColor: [
            "rgba(59, 130, 246, 0.7)",
            "rgba(16, 185, 129, 0.7)",
            "rgba(239, 68, 68, 0.7)",
            "rgba(245, 158, 11, 0.7)",
          ],
          borderColor: ["rgb(59, 130, 246)", "rgb(16, 185, 129)", "rgb(239, 68, 68)", "rgb(245, 158, 11)"],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          title: {
            display: true,
            text: "log₁₀(Years)",
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Brute Force Attack Resistance",
          font: {
            size: 14,
          },
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: 10^${context.raw} years`,
          },
        },
      },
    },
  })
}

function initQuantumAttackChart() {
  const ctx = document.getElementById("quantumAttackChart")
  if (!ctx) return

  // Destroy existing chart if it exists
  if (ctx.chart) {
    ctx.chart.destroy()
  }

  ctx.chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["XenoCipher", "AES", "RSA", "ECC"],
      datasets: [
        {
          label: "Quantum Security (bits)",
          data: [128, 128, 0, 0],
          backgroundColor: [
            "rgba(59, 130, 246, 0.7)",
            "rgba(16, 185, 129, 0.7)",
            "rgba(239, 68, 68, 0.7)",
            "rgba(245, 158, 11, 0.7)",
          ],
          borderColor: ["rgb(59, 130, 246)", "rgb(16, 185, 129)", "rgb(239, 68, 68)", "rgb(245, 158, 11)"],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Security Level (bits)",
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Quantum Attack Resistance",
          font: {
            size: 14,
          },
        },
      },
    },
  })
}

function initComplexityChart() {
  const ctx = document.getElementById("complexityChart")
  if (!ctx) return

  // Destroy existing chart if it exists
  if (ctx.chart) {
    ctx.chart.destroy()
  }

  ctx.chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["XenoCipher", "AES", "RSA", "ECC"],
      datasets: [
        {
          label: "Implementation Complexity (lower is easier)",
          data: [7, 3, 6, 5],
          backgroundColor: [
            "rgba(59, 130, 246, 0.7)",
            "rgba(16, 185, 129, 0.7)",
            "rgba(239, 68, 68, 0.7)",
            "rgba(245, 158, 11, 0.7)",
          ],
          borderColor: ["rgb(59, 130, 246)", "rgb(16, 185, 129)", "rgb(239, 68, 68)", "rgb(245, 158, 11)"],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 10,
          title: {
            display: true,
            text: "Complexity Score (0-10)",
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Implementation Complexity",
          font: {
            size: 14,
          },
        },
      },
    },
  })
}
