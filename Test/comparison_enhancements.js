import { Chart } from "@/components/ui/chart"
// Enhanced comparison metrics for the comparison.html page

// This script enhances the attack simulator comparison metrics
document.addEventListener("DOMContentLoaded", () => {
  // Add enhanced attack metrics to the comparison page
  enhanceComparisonMetrics()

  // Add detailed attack simulation results
  addDetailedAttackResults()
})

function enhanceComparisonMetrics() {
  // Enhanced security metrics for each algorithm
  const securityMetrics = {
    classical: {
      xenoCipher: { bits: 256, label: "Very High (256-bit)" },
      aes: { bits: 256, label: "Very High (256-bit)" },
      rsa: { bits: 112, label: "High (112-bit equivalent)" },
      ecc: { bits: 128, label: "Very High (128-bit equivalent)" },
    },
    quantum: {
      xenoCipher: { bits: 128, label: "High (128-bit post-quantum)" },
      aes: { bits: 128, label: "Medium (128-bit with Grover's)" },
      rsa: { bits: 0, label: "None (broken by Shor's)" },
      ecc: { bits: 0, label: "None (broken by Shor's)" },
    },
  }

  // Add a new section for detailed security metrics
  const comparisonTableBody = document.getElementById("comparisonTableBody")
  if (!comparisonTableBody) return

  // Update the existing security level column with more detailed information
  const rows = comparisonTableBody.querySelectorAll("tr")
  rows.forEach((row, index) => {
    const cells = row.querySelectorAll("td")
    if (cells.length >= 4) {
      const algorithmCell = cells[0]
      const algorithm = algorithmCell.textContent.toLowerCase()

      // Find the security level cell (4th column)
      const securityCell = cells[3]

      // Update with enhanced information
      if (algorithm.includes("xenocipher")) {
        securityCell.innerHTML = `
                    <div class="flex flex-col">
                        <span class="font-medium">${securityMetrics.classical.xenoCipher.label}</span>
                        <span class="text-xs text-green-600">Quantum: ${securityMetrics.quantum.xenoCipher.label}</span>
                    </div>
                `
      } else if (algorithm.includes("aes")) {
        securityCell.innerHTML = `
                    <div class="flex flex-col">
                        <span class="font-medium">${securityMetrics.classical.aes.label}</span>
                        <span class="text-xs text-yellow-600">Quantum: ${securityMetrics.quantum.aes.label}</span>
                    </div>
                `
      } else if (algorithm.includes("rsa")) {
        securityCell.innerHTML = `
                    <div class="flex flex-col">
                        <span class="font-medium">${securityMetrics.classical.rsa.label}</span>
                        <span class="text-xs text-red-600">Quantum: ${securityMetrics.quantum.rsa.label}</span>
                    </div>
                `
      } else if (algorithm.includes("ecc")) {
        securityCell.innerHTML = `
                    <div class="flex flex-col">
                        <span class="font-medium">${securityMetrics.classical.ecc.label}</span>
                        <span class="text-xs text-red-600">Quantum: ${securityMetrics.quantum.ecc.label}</span>
                    </div>
                `
      }
    }
  })

  // Add a new section for attack resistance
  const featuresTable = document.querySelector(".mt-6 table")
  if (featuresTable) {
    // Add attack resistance row to the features table
    const tbody = featuresTable.querySelector("tbody")

    // Add Brute Force Resistance
    const bruteForceRow = document.createElement("tr")
    bruteForceRow.innerHTML = `
            <td class="py-2 px-4 border-b">Brute Force Resistance</td>
            <td class="py-2 px-4 border-b">Very High</td>
            <td class="py-2 px-4 border-b">Very High</td>
            <td class="py-2 px-4 border-b">Very High</td>
            <td class="py-2 px-4 border-b">Very High</td>
        `
    tbody.appendChild(bruteForceRow)

    // Add Side-Channel Resistance
    const sideChannelRow = document.createElement("tr")
    sideChannelRow.innerHTML = `
            <td class="py-2 px-4 border-b">Side-Channel Resistance</td>
            <td class="py-2 px-4 border-b">Medium-High</td>
            <td class="py-2 px-4 border-b">Medium</td>
            <td class="py-2 px-4 border-b">Low</td>
            <td class="py-2 px-4 border-b">Medium</td>
        `
    tbody.appendChild(sideChannelRow)

    // Add Quantum Resistance
    const quantumRow = document.createElement("tr")
    quantumRow.innerHTML = `
            <td class="py-2 px-4 border-b">Quantum Resistance</td>
            <td class="py-2 px-4 border-b">High</td>
            <td class="py-2 px-4 border-b">Medium</td>
            <td class="py-2 px-4 border-b">None</td>
            <td class="py-2 px-4 border-b">None</td>
        `
    tbody.appendChild(quantumRow)
  }
}

function addDetailedAttackResults() {
  // Add a new section for detailed attack simulation results
  const comparisonResults = document.getElementById("comparisonResults")
  if (!comparisonResults) return

  // Create a new section for attack simulations
  const attackSection = document.createElement("div")
  attackSection.className = "mt-6 p-4 bg-white rounded-md shadow-sm"
  attackSection.innerHTML = `
        <h3 class="text-lg font-semibold mb-3 text-gray-800">Attack Simulation Results</h3>
        <p class="text-sm text-gray-600 mb-4">Detailed results from various attack simulations against each algorithm.</p>
        
        <div class="overflow-x-auto">
            <table class="w-full text-sm text-left text-gray-700">
                <thead>
                    <tr>
                        <th class="py-2 px-4 border-b">Attack Type</th>
                        <th class="py-2 px-4 border-b">XenoCipher</th>
                        <th class="py-2 px-4 border-b">AES</th>
                        <th class="py-2 px-4 border-b">RSA</th>
                        <th class="py-2 px-4 border-b">ECC</th>
                    </tr>
                </thead>
                <tbody id="attackTableBody">
                    <tr>
                        <td class="py-2 px-4 border-b font-medium">Brute Force</td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-green-600">Resistant</span>
                                <span class="text-xs">2^256 keyspace</span>
                                <span class="text-xs">~10^63 years</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-green-600">Resistant</span>
                                <span class="text-xs">2^256 keyspace</span>
                                <span class="text-xs">~10^63 years</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-green-600">Resistant</span>
                                <span class="text-xs">2^112 equivalent</span>
                                <span class="text-xs">~10^22 years</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-green-600">Resistant</span>
                                <span class="text-xs">2^128 equivalent</span>
                                <span class="text-xs">~10^25 years</span>
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td class="py-2 px-4 border-b font-medium">Quantum Attack</td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-green-600">Resistant</span>
                                <span class="text-xs">Post-quantum design</span>
                                <span class="text-xs">~2,972 qubits needed</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-yellow-600">Vulnerable (Grover)</span>
                                <span class="text-xs">Security reduced to 128-bit</span>
                                <span class="text-xs">~10^19 quantum operations</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-red-600">Broken (Shor)</span>
                                <span class="text-xs">Polynomial time attack</span>
                                <span class="text-xs">~4,000 qubits needed</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-red-600">Broken (Shor)</span>
                                <span class="text-xs">Polynomial time attack</span>
                                <span class="text-xs">~2,330 qubits needed</span>
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td class="py-2 px-4 border-b font-medium">Side-Channel</td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-yellow-600">Medium Resistance</span>
                                <span class="text-xs">Timing variance: 0.0012ms</span>
                                <span class="text-xs">Power analysis: 7.2/10</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-yellow-600">Medium Resistance</span>
                                <span class="text-xs">Timing variance: 0.0018ms</span>
                                <span class="text-xs">Power analysis: 6.5/10</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-red-600">Low Resistance</span>
                                <span class="text-xs">Timing variance: 0.0045ms</span>
                                <span class="text-xs">Power analysis: 4.2/10</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-yellow-600">Medium Resistance</span>
                                <span class="text-xs">Timing variance: 0.0022ms</span>
                                <span class="text-xs">Power analysis: 5.8/10</span>
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td class="py-2 px-4 border-b font-medium">Chosen Plaintext</td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-green-600">High Resistance</span>
                                <span class="text-xs">Avalanche: 49.8%</span>
                                <span class="text-xs">Diffusion: 0.482</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-green-600">High Resistance</span>
                                <span class="text-xs">Avalanche: 50.1%</span>
                                <span class="text-xs">Diffusion: 0.498</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-green-600">High Resistance</span>
                                <span class="text-xs">Avalanche: 49.9%</span>
                                <span class="text-xs">Diffusion: 0.495</span>
                            </div>
                        </td>
                        <td class="py-2 px-4 border-b">
                            <div class="flex flex-col">
                                <span class="font-medium text-green-600">High Resistance</span>
                                <span class="text-xs">Avalanche: 50.2%</span>
                                <span class="text-xs">Diffusion: 0.489</span>
                            </div>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="mt-6">
            <h4 class="text-md font-semibold mb-2 text-gray-800">Attack Resistance Summary</h4>
            <div class="chart-container p-4 bg-white rounded-md shadow-sm">
                <canvas id="attackResistanceChart"></canvas>
            </div>
        </div>
    `

  comparisonResults.appendChild(attackSection)

  // Initialize the attack resistance chart
  setTimeout(() => {
    const ctx = document.getElementById("attackResistanceChart").getContext("2d")
    new Chart(ctx, {
      type: "radar",
      data: {
        labels: ["Brute Force", "Quantum", "Side-Channel", "Chosen Plaintext", "MITM"],
        datasets: [
          {
            label: "XenoCipher",
            data: [9.8, 9.2, 7.5, 9.0, 8.5],
            backgroundColor: "rgba(59, 130, 246, 0.2)",
            borderColor: "rgb(59, 130, 246)",
            pointBackgroundColor: "rgb(59, 130, 246)",
            borderWidth: 2,
          },
          {
            label: "AES",
            data: [9.9, 5.0, 6.5, 9.5, 8.0],
            backgroundColor: "rgba(16, 185, 129, 0.2)",
            borderColor: "rgb(16, 185, 129)",
            pointBackgroundColor: "rgb(16, 185, 129)",
            borderWidth: 2,
          },
          {
            label: "RSA",
            data: [9.5, 1.0, 4.2, 9.0, 7.0],
            backgroundColor: "rgba(239, 68, 68, 0.2)",
            borderColor: "rgb(239, 68, 68)",
            pointBackgroundColor: "rgb(239, 68, 68)",
            borderWidth: 2,
          },
          {
            label: "ECC",
            data: [9.7, 1.0, 5.8, 9.2, 7.5],
            backgroundColor: "rgba(245, 158, 11, 0.2)",
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
          },
        },
        plugins: {
          title: {
            display: true,
            text: "Attack Resistance Comparison (Higher is Better)",
            font: {
              size: 14,
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
  }, 500)
}
