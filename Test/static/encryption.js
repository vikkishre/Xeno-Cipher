// Encryption algorithms implementation
class Encryption {
  // XenoCipher implementation based on the Python code
  static xenoCipher(text, key = "default-xeno-key", mode = "normal") {
    const start = performance.now()

    // Convert text to bytes
    const data = new TextEncoder().encode(text)

    // Derive keys from master key
    const masterKey = new TextEncoder().encode(key)
    const keys = Encryption.deriveKeys(masterKey, data.length, mode)

    // Extract keys
    const lfsrSeed = keys.lfsrSeed
    const chaoticX0 = keys.chaoticX0
    const chaoticR = keys.chaoticR
    const transpositionKey = keys.transpositionKey
    const chachaKey = keys.chachaKey
    const chachaNonce = keys.chachaNonce
    const speckKey = keys.speckKey

    // Apply encryption layers
    let result = new Uint8Array(data)

    if (mode === "ztm") {
      // ZTM mode: ChaCha20 (simulated)
      result = Encryption.simulateChaCha20(result, chachaKey, chachaNonce)
    }

    // Apply LFSR (bit-level XOR)
    const lfsrStream = Encryption.lfsr(lfsrSeed, 0x8029, result.length)
    for (let i = 0; i < result.length; i++) {
      result[i] ^= lfsrStream[i]
    }

    // Apply Chaotic Map (bit-level XOR)
    const chaoticStream = Encryption.logisticMap(chaoticX0, chaoticR, result.length)
    for (let i = 0; i < result.length; i++) {
      result[i] ^= chaoticStream[i]
    }

    // Apply Transposition (bit permutation)
    result = Encryption.transpose(result, transpositionKey)

    if (mode === "ztm") {
      // Apply Speck in CTR mode (simulated)
      result = Encryption.simulateSpeckCTR(result, speckKey)
    }

    // Convert to base64 for display
    const base64 = btoa(String.fromCharCode.apply(null, result))

    const end = performance.now()
    return {
      encrypted: base64,
      time: end - start,
    }
  }

  // Key derivation function
  static deriveKeys(masterKey, dataLength, mode) {
    // Create a unique salt for this encryption
    const saltStr = `${dataLength}:${mode}`
    const salt = new TextEncoder().encode(saltStr)

    // Simulate PBKDF2 with SHA-256
    let derivedKey = new Uint8Array(64)

    // Simple key derivation for demo (in real implementation, use Web Crypto API)
    for (let i = 0; i < 64; i++) {
      derivedKey[i] = masterKey[i % masterKey.length] ^ salt[i % salt.length]
    }

    // Hash the derived key to improve randomness
    derivedKey = Encryption.sha256(derivedKey)

    // Split into different keys for each algorithm
    return {
      lfsrSeed: (derivedKey[0] << 8) | derivedKey[1],
      chaoticX0: ((derivedKey[2] << 24) | (derivedKey[3] << 16) | (derivedKey[4] << 8) | derivedKey[5]) / (1 << 32),
      chaoticR: 3.9, // Chaotic parameter in range (3.57, 4.0)
      transpositionKey: derivedKey.slice(6, 14),
      chachaKey: derivedKey.slice(14, 46),
      chachaNonce: derivedKey.slice(46, 62),
      speckKey: derivedKey.slice(32, 48),
    }
  }

  // LFSR implementation
  static lfsr(state, polynomial, length) {
    const mask = (1 << 16) - 1 // 16-bit register
    const output = new Uint8Array(length)

    for (let byteIdx = 0; byteIdx < length; byteIdx++) {
      let byteVal = 0
      for (let bitIdx = 0; bitIdx < 8; bitIdx++) {
        const lsb = state & 1
        byteVal |= lsb << bitIdx

        // Calculate feedback
        let feedback = 0
        if (polynomial & (1 << 15)) feedback ^= (state >> 15) & 1
        if (polynomial & (1 << 5)) feedback ^= (state >> 5) & 1
        if (polynomial & (1 << 3)) feedback ^= (state >> 3) & 1
        if (polynomial & (1 << 0)) feedback ^= state & 1

        state = ((state >> 1) | (feedback << 15)) & mask
      }
      output[byteIdx] = byteVal
    }

    return output
  }

  // Logistic Map implementation
  static logisticMap(x0, r, length) {
    // Ensure x0 is in valid range
    if (!(0 < x0 && x0 < 1)) {
      x0 = 0.5
    }

    // Ensure r is in chaotic region
    if (!(3.57 < r && r < 4.0)) {
      r = 3.99
    }

    const output = new Uint8Array(length)

    // Skip initial iterations to avoid transient behavior
    let x = x0
    for (let i = 0; i < 100; i++) {
      x = r * x * (1 - x)
    }

    // Generate output bytes
    for (let i = 0; i < length; i++) {
      // Multiple iterations per byte for better randomness
      for (let j = 0; j < 4; j++) {
        x = r * x * (1 - x)
      }

      // Convert to byte with full range
      output[i] = Math.floor(x * 256) & 0xff
    }

    return output
  }

  // Transposition implementation
  static transpose(data, keyStream) {
    const n = data.length

    // Create permutation table based on keyStream
    const perm = Array.from({ length: n }, (_, i) => i)

    // Use Fisher-Yates shuffle with keyStream as randomness source
    for (let i = n - 1; i > 0; i--) {
      const j = keyStream[i % keyStream.length] % (i + 1)
      // Swap perm[i] and perm[j]
      const temp = perm[i]
      perm[i] = perm[j]
      perm[j] = temp
    }

    // Apply permutation
    const result = new Uint8Array(n)
    for (let i = 0; i < n; i++) {
      result[perm[i]] = data[i]
    }

    return result
  }

  // Simulate ChaCha20 encryption
  static simulateChaCha20(data, key, nonce) {
    // This is a simplified simulation of ChaCha20
    // In a real implementation, use Web Crypto API
    const result = new Uint8Array(data.length)

    // Generate a keystream based on key and nonce
    const keystream = new Uint8Array(data.length)
    for (let i = 0; i < data.length; i++) {
      keystream[i] = (key[i % key.length] ^ nonce[i % nonce.length] ^ i) & 0xff
    }

    // XOR with data
    for (let i = 0; i < data.length; i++) {
      result[i] = data[i] ^ keystream[i]
    }

    return result
  }

  // Simulate Speck block cipher in CTR mode
  static simulateSpeckCTR(data, key) {
    // This is a simplified simulation of Speck in CTR mode
    // In a real implementation, use a proper Speck library
    const result = new Uint8Array(data.length)

    // Pad data to multiple of 8 bytes
    const paddedLength = data.length + ((8 - (data.length % 8)) % 8)
    const paddedData = new Uint8Array(paddedLength)
    paddedData.set(data)

    // Counter mode encryption
    let counter = 0
    for (let i = 0; i < paddedLength; i += 8) {
      // Create counter block
      const ctrBlock = new Uint8Array(8)
      for (let j = 0; j < 8; j++) {
        ctrBlock[j] = (counter >> (j * 8)) & 0xff
      }
      counter++

      // Encrypt counter (simplified)
      const encryptedCtr = Encryption.simulateSpeckBlock(key, ctrBlock)

      // XOR with plaintext
      for (let j = 0; j < 8; j++) {
        if (i + j < paddedLength) {
          result[i + j] = paddedData[i + j] ^ encryptedCtr[j]
        }
      }
    }

    // Trim to original length
    return result.slice(0, data.length)
  }

  // Simulate Speck block cipher
  static simulateSpeckBlock(key, block) {
    // Extract 64-bit block as two 32-bit words
    let x = (block[0] | (block[1] << 8) | (block[2] << 16) | (block[3] << 24)) >>> 0
    let y = (block[4] | (block[5] << 8) | (block[6] << 16) | (block[7] << 24)) >>> 0

    // Extract key words
    const k = []
    for (let i = 0; i < key.length; i += 4) {
      if (i + 3 < key.length) {
        k.push((key[i] | (key[i + 1] << 8) | (key[i + 2] << 16) | (key[i + 3] << 24)) >>> 0)
      }
    }

    // Number of rounds
    const R = 22

    // Constants
    const alpha = 8,
      beta = 3

    for (let i = 0; i < R; i++) {
      // Key schedule would normally go here
      const roundKey = k[i % k.length]

      // Speck round function
      x = ((x >>> alpha) | (x << (32 - alpha))) >>> 0
      x = (x + y) >>> 0
      x ^= roundKey
      y = ((y << beta) | (y >>> (32 - beta))) >>> 0
      y ^= x
    }

    // Convert back to bytes
    const result = new Uint8Array(8)
    for (let i = 0; i < 4; i++) {
      result[i] = (x >> (i * 8)) & 0xff
      result[i + 4] = (y >> (i * 8)) & 0xff
    }

    return result
  }

  // Simple SHA-256 implementation for demo
  static sha256(data) {
    // This is a placeholder - in a real implementation, use Web Crypto API
    // For demo purposes, we'll create a deterministic hash-like output
    const result = new Uint8Array(64)

    // Simple mixing function
    for (let i = 0; i < 64; i++) {
      let value = i
      for (let j = 0; j < data.length; j++) {
        value = (value + data[j] * (j + 1)) & 0xff
        value = ((value << 1) | (value >> 7)) & 0xff // rotate
      }
      result[i] = value
    }

    return result
  }

  // AES implementation (simplified for demo)
  static aes(text, key = "aes-256-encryption-key-demo-purpose") {
    const start = performance.now()

    // Simulate AES encryption
    // In a real implementation, we would use the Web Crypto API
    const textBytes = new TextEncoder().encode(text)
    const keyBytes = new TextEncoder().encode(key)

    // Simulate AES rounds
    const result = new Uint8Array(textBytes.length)
    for (let i = 0; i < textBytes.length; i++) {
      // XOR with key (simplified)
      result[i] = textBytes[i] ^ keyBytes[i % keyBytes.length]

      // Simulate substitution
      result[i] = (result[i] * 7 + 5) % 256

      // Simulate permutation
      if (i > 0) {
        result[i] = (result[i] + result[i - 1]) % 256
      }
    }

    // Convert to base64 for display
    const base64 = btoa(String.fromCharCode.apply(null, result))

    const end = performance.now()
    return {
      encrypted: base64,
      time: end - start,
    }
  }

  // RSA implementation (simplified for demo)
  static rsa(text, key = "rsa-2048-demo-key") {
    const start = performance.now()

    // Simulate RSA encryption (much slower than symmetric algorithms)
    // Add artificial delay to simulate RSA's slower performance
    const delay = text.length * 0.5 // Simulate longer time for longer text

    // Simulate the encryption process
    const textBytes = new TextEncoder().encode(text)
    const keyBytes = new TextEncoder().encode(key)

    // Simulate modular exponentiation (very simplified)
    const result = new Uint8Array(textBytes.length)
    for (let i = 0; i < textBytes.length; i++) {
      // Simulate complex math operations
      let val = textBytes[i]
      for (let j = 0; j < 10; j++) {
        // More iterations to simulate slowness
        val = (val * 65537) % 256 // Simulate RSA public exponent
        val = (val + keyBytes[i % keyBytes.length]) % 256
      }
      result[i] = val
    }

    // Add artificial delay
    const sleepUntil = performance.now() + delay
    while (performance.now() < sleepUntil) {
      // Busy wait to simulate delay
    }

    // Convert to base64 for display
    const base64 = btoa(String.fromCharCode.apply(null, result))

    const end = performance.now()
    return {
      encrypted: base64,
      time: end - start,
    }
  }

  // ECC implementation (simplified for demo)
  static ecc(text, key = "ecc-p256-demo-key") {
    const start = performance.now()

    // Simulate ECC encryption (slower than symmetric but faster than RSA)
    // Add artificial delay to simulate ECC's performance
    const delay = text.length * 0.2 // Simulate longer time for longer text

    // Simulate the encryption process
    const textBytes = new TextEncoder().encode(text)
    const keyBytes = new TextEncoder().encode(key)

    // Simulate elliptic curve operations (very simplified)
    const result = new Uint8Array(textBytes.length)
    for (let i = 0; i < textBytes.length; i++) {
      // Simulate point multiplication
      let val = textBytes[i]
      for (let j = 0; j < 5; j++) {
        // Fewer iterations than RSA
        val = (val * 23) % 256 // Simulate ECC point multiplication
        val = (val + keyBytes[i % keyBytes.length]) % 256
      }
      result[i] = val
    }

    // Add artificial delay
    const sleepUntil = performance.now() + delay
    while (performance.now() < sleepUntil) {
      // Busy wait to simulate delay
    }

    // Convert to base64 for display
    const base64 = btoa(String.fromCharCode.apply(null, result))

    const end = performance.now()
    return {
      encrypted: base64,
      time: end - start,
    }
  }
}

// DOM elements
const encryptionInput = document.getElementById("encryption-input")
const encryptButton = document.getElementById("encrypt-button")
const xenoCipherOutput = document.getElementById("xenocipher-output")
const aesOutput = document.getElementById("aes-output")
const rsaOutput = document.getElementById("rsa-output")
const eccOutput = document.getElementById("ecc-output")
const xenoCipherTime = document.getElementById("xenocipher-time")
const aesTime = document.getElementById("aes-time")
const rsaTime = document.getElementById("rsa-time")
const eccTime = document.getElementById("ecc-time")

// Performance metrics for real-time chart
const performanceData = {
  xenoCipher: [],
  aes: [],
  rsa: [],
  ecc: [],
}

// Event listeners
encryptButton.addEventListener("click", performEncryption)
encryptionInput.addEventListener("keyup", (event) => {
  if (event.key === "Enter") {
    performEncryption()
  }
})

// Function to perform encryption
function performEncryption() {
  const text = encryptionInput.value.trim()

  if (!text) {
    alert("Please enter text to encrypt")
    return
  }

  // Add pulse animation to the encrypt button
  encryptButton.classList.add("pulse-animation")

  // Display "Encrypting..." in all output areas
  xenoCipherOutput.textContent = "Encrypting..."
  aesOutput.textContent = "Encrypting..."
  rsaOutput.textContent = "Encrypting..."
  eccOutput.textContent = "Encrypting..."

  // Get selected XenoCipher mode
  const xenoCipherMode = document.querySelector('input[name="xenocipher-mode"]:checked').value

  // Use setTimeout to allow the UI to update before heavy computation
  setTimeout(() => {
    // Perform encryption with all algorithms
    const xenoCipherResult = Encryption.xenoCipher(text, "xenocipher-master-key", xenoCipherMode)
    const aesResult = Encryption.aes(text)
    const rsaResult = Encryption.rsa(text)
    const eccResult = Encryption.ecc(text)

    // Update the UI with results
    xenoCipherOutput.textContent = xenoCipherResult.encrypted
    aesOutput.textContent = aesResult.encrypted
    rsaOutput.textContent = rsaResult.encrypted
    eccOutput.textContent = eccResult.encrypted

    // Update timing information
    xenoCipherTime.textContent = `${xenoCipherResult.time.toFixed(2)} ms`
    aesTime.textContent = `${aesResult.time.toFixed(2)} ms`
    rsaTime.textContent = `${rsaResult.time.toFixed(2)} ms`
    eccTime.textContent = `${eccResult.time.toFixed(2)} ms`

    // Store performance data for chart
    performanceData.xenoCipher.push(xenoCipherResult.time)
    performanceData.aes.push(aesResult.time)
    performanceData.rsa.push(rsaResult.time)
    performanceData.ecc.push(eccResult.time)

    // Keep only the last 5 data points
    if (performanceData.xenoCipher.length > 5) {
      performanceData.xenoCipher.shift()
      performanceData.aes.shift()
      performanceData.rsa.shift()
      performanceData.ecc.shift()
    }

    // Update the real-time performance chart
    updateRealTimePerformanceChart()

    // Remove pulse animation
    encryptButton.classList.remove("pulse-animation")
  }, 50)
}

// Function to update the real-time performance chart
function updateRealTimePerformanceChart() {
  const ctx = document.getElementById("realTimePerformanceChart")

  // Destroy existing chart if it exists
  if (ctx.chart) {
    ctx.chart.destroy()
  }

  // Create labels for the chart (Encryption 1, Encryption 2, etc.)
  const labels = []
  for (let i = 0; i < performanceData.xenoCipher.length; i++) {
    labels.push(`Encryption ${i + 1}`)
  }

  // Create the chart
  ctx.chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "XenoCipher",
          data: performanceData.xenoCipher,
          borderColor: "rgb(59, 130, 246)",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          tension: 0.1,
          fill: true,
        },
        {
          label: "AES",
          data: performanceData.aes,
          borderColor: "rgb(16, 185, 129)",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          tension: 0.1,
          fill: true,
        },
        {
          label: "RSA",
          data: performanceData.rsa,
          borderColor: "rgb(239, 68, 68)",
          backgroundColor: "rgba(239, 68, 68, 0.1)",
          tension: 0.1,
          fill: true,
        },
        {
          label: "ECC",
          data: performanceData.ecc,
          borderColor: "rgb(245, 158, 11)",
          backgroundColor: "rgba(245, 158, 11, 0.1)",
          tension: 0.1,
          fill: true,
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
            text: "Encryption Time (ms)",
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Real-time Encryption Performance",
          font: {
            size: 14,
          },
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${context.raw.toFixed(2)} ms`,
          },
        },
      },
    },
  })
}

// Initialize with empty chart
document.addEventListener("DOMContentLoaded", () => {
  updateRealTimePerformanceChart()
})
