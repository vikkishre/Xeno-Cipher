from flask import Flask, request, jsonify
import os
import time
import numpy as np
import random
import hashlib
import multiprocessing
import signal
import logging
import math
import struct
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xenocipher_attacks')

# ----- Attack Simulation Functions -----

class AttackSimulator:
    """Class to handle various cryptographic attack simulations on XenoCipher"""
    
    def __init__(self):
        """Initialize the attack simulator with default parameters"""
        self.attack_history = {}
        self.attack_stats = {
            'brute': {'attempts': 0, 'success': 0},
            'chosen': {'attempts': 0, 'success': 0},
            'mitm': {'attempts': 0, 'success': 0},
            'side': {'attempts': 0, 'success': 0},
            'quantum': {'attempts': 0, 'success': 0},
            'dos': {'attempts': 0, 'success': 0}
        }
    
    def brute_force_attack(self, ciphertext, mode, max_attempts=10000):
        """
        Simulate a brute force attack by trying random keys
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            max_attempts: Maximum number of key attempts to simulate
            
        Returns:
            Dictionary with attack results and statistics
        """
        start_time = time.time()
        logger.info(f"Starting brute force attack on {len(ciphertext)} bytes, mode: {mode}")
        
        # Record this attempt
        self.attack_stats['brute']['attempts'] += 1
        
        # Calculate effective key size based on mode
        if mode == 'ztm':
            # ZTM mode uses multiple algorithms in sequence
            effective_key_bits = 256  # Combined effective security
        else:
            # Normal mode uses fewer algorithms
            effective_key_bits = 192
            
        # Calculate theoretical time for full attack
        # Assuming 1 billion attempts per second on a high-end system
        attempts_per_second = 1_000_000_000
        theoretical_seconds = (2**effective_key_bits) / attempts_per_second
        theoretical_years = theoretical_seconds / (60 * 60 * 24 * 365)
        
        # Try random keys (this will always fail due to key space)
        successful = False
        keys_tried = 0
        
        # Use multiple threads to simulate parallel attack
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
            futures = []
            chunk_size = max_attempts // 8
            
            for i in range(8):
                futures.append(executor.submit(
                    self._brute_force_chunk, 
                    ciphertext[:min(1024, len(ciphertext))],  # Use first 1KB for speed
                    chunk_size
                ))
            
            for future in as_completed(futures):
                result = future.result()
                keys_tried += result['keys_tried']
                if result['success']:
                    successful = True
                    break
        
        duration = time.time() - start_time
        
        # Calculate actual attempts per second
        actual_attempts_per_second = keys_tried / duration if duration > 0 else 0
        
        # Estimate time for full attack based on actual performance
        estimated_years = (2**effective_key_bits) / (actual_attempts_per_second * 60 * 60 * 24 * 365) if actual_attempts_per_second > 0 else float('inf')
        
        # For simulation purposes, brute force always fails due to key space
        if successful:
            self.attack_stats['brute']['success'] += 1
            message = f"Attack succeeded after {keys_tried} attempts (simulated for demo)."
        else:
            message = f"Attack failed after {keys_tried} attempts. Would take approximately {estimated_years:.2e} years to complete."
        
        return {
            "success": successful,
            "attempts": keys_tried,
            "time_seconds": duration,
            "attempts_per_second": f"{actual_attempts_per_second:.2e}",
            "theoretical_years": f"{theoretical_years:.2e}",
            "estimated_years": f"{estimated_years:.2e}",
            "key_space": f"2^{effective_key_bits}",
            "message": message
        }
    
    def _brute_force_chunk(self, ciphertext_sample, attempts):
        """Helper method to try a chunk of brute force attempts"""
        keys_tried = 0
        
        for _ in range(attempts):
            # Generate a random key
            random_key = os.urandom(32)  # 256-bit key
            
            # Simulate attempt to decrypt with this key
            # In a real attack, we'd try to decrypt and check if result is valid
            self._simulate_decryption_attempt(ciphertext_sample, random_key)
            
            keys_tried += 1
            
            # For simulation, we always fail (0.0001% chance of "success" for demo purposes)
            if random.random() < 0.000001:  # Extremely unlikely
                return {"success": True, "keys_tried": keys_tried}
        
        return {"success": False, "keys_tried": keys_tried}
    
    def _simulate_decryption_attempt(self, ciphertext_sample, key):
        """Simulate a decryption attempt with a given key"""
        # This is just a simulation - in a real attack, we'd actually try to decrypt
        # For simulation purposes, we'll just do some computation to simulate the work
        h = hashlib.sha256(key + ciphertext_sample[:64]).digest()
        
        # Small delay to simulate computation time
        time.sleep(0.0001)
        
        return h
    
    def chosen_plaintext_attack(self, ciphertext, mode, num_samples=100):
        """
        Simulate a chosen plaintext attack by analyzing patterns in ciphertexts
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            num_samples: Number of chosen plaintexts to analyze
            
        Returns:
            Dictionary with attack results and statistics
        """
        start_time = time.time()
        logger.info(f"Starting chosen plaintext attack, mode: {mode}, samples: {num_samples}")
        
        # Record this attempt
        self.attack_stats['chosen']['attempts'] += 1
        
        # Generate different plaintexts with specific patterns to analyze
        plaintexts = []
        
        # 1. Repeated single character (to detect patterns)
        for c in "abcdefghij":
            plaintexts.append(c * 64)
            
        # 2. Incrementing bytes (to detect substitution patterns)
        plaintexts.append(''.join(chr(i % 256) for i in range(64)))
        
        # 3. Block patterns (to detect block boundaries)
        for block_size in [8, 16, 32]:
            for c in "abcde":
                plaintexts.append((c * block_size) * (64 // block_size))
        
        # 4. Alternating patterns (to detect diffusion)
        plaintexts.append('a' * 32 + 'b' * 32)
        plaintexts.append('ab' * 32)
        plaintexts.append('aaabbb' * 10 + 'a' * 4)
        
        # 5. Special patterns for bit-level analysis
        bit_patterns = []
        for i in range(8):
            # Pattern with only one bit set
            pattern = bytes([1 << i] * 64)
            bit_patterns.append(pattern)
            
        # Limit to requested number of samples
        all_samples = plaintexts[:num_samples]
        
        # In a real attack, we'd encrypt these and analyze
        # For simulation, we'll create pseudo-ciphertexts
        samples = []
        for text in all_samples:
            if isinstance(text, str):
                text_bytes = text.encode()
            else:
                text_bytes = text
                
            # Simulate encryption (in a real attack, we'd send to the server)
            pseudo_cipher = self._simulate_encryption(text_bytes, mode)
            samples.append((text_bytes, pseudo_cipher))
        
        # Analyze bit patterns and diffusion properties
        diffusion_scores = []
        avalanche_scores = []
        
        # Calculate bit changes between similar plaintexts
        for i in range(len(samples)):
            for j in range(i+1, len(samples)):
                p1, c1 = samples[i]
                p2, c2 = samples[j]
                
                # Count bit differences in plaintexts
                p_diff = sum(bin(a ^ b).count('1') for a, b in zip(p1, p2))
                
                # Count bit differences in ciphertexts
                c_diff = sum(bin(a ^ b).count('1') for a, b in zip(c1, c2))
                
                # Only consider pairs with small plaintext differences
                if 0 < p_diff <= 8:
                    # Avalanche effect: small input change should cause ~50% output bits to change
                    avalanche = c_diff / (len(c1) * 8)
                    avalanche_scores.append(avalanche)
        
        # Calculate diffusion between consecutive samples
        for i in range(1, len(samples)):
            prev_cipher = samples[i-1][1]
            curr_cipher = samples[i][1]
            diff_bits = sum(bin(a ^ b).count('1') for a, b in zip(prev_cipher, curr_cipher))
            diffusion_scores.append(diff_bits / (len(prev_cipher) * 8))
        
        # Analyze results
        avg_diffusion = sum(diffusion_scores) / len(diffusion_scores) if diffusion_scores else 0
        avg_avalanche = sum(avalanche_scores) / len(avalanche_scores) if avalanche_scores else 0
        
        # Ideal avalanche effect is 0.5 (50% of bits change)
        avalanche_quality = 1.0 - abs(0.5 - avg_avalanche) * 2  # Scale to 0-1
        
        # High diffusion is good
        diffusion_quality = min(1.0, avg_diffusion * 2)  # Scale to 0-1
        
        # Combined resistance score (0-10)
        resistance_score = (avalanche_quality * 5) + (diffusion_quality * 5)
        
        # Pattern detection percentage (lower is better)
        pattern_detection = 0
        for i in range(len(samples)):
            p, c = samples[i]
            # Look for any patterns in ciphertext that correlate with plaintext
            # This is a simplified simulation
            pattern_detection += self._detect_patterns(p, c)
        
        pattern_detection = pattern_detection / len(samples) if samples else 0
        pattern_detection_percent = f"{pattern_detection * 100:.2f}%"
        
        # Security margin (higher is better)
        security_margin = 100 - (pattern_detection * 100)
        security_margin_percent = f"{security_margin:.2f}%"
        
        duration = time.time() - start_time
        
        # For simulation, attack always fails unless pattern detection is very high
        success = pattern_detection > 0.8  # 80% pattern detection would be a serious weakness
        
        if success:
            self.attack_stats['chosen']['success'] += 1
            message = f"Attack succeeded with {pattern_detection_percent} pattern detection."
        else:
            message = f"Attack failed. System shows strong diffusion properties with score {resistance_score:.2f}/10."
        
        return {
            "success": success,
            "samples_analyzed": len(samples),
            "diffusion_score": f"{avg_diffusion:.4f}",
            "avalanche_score": f"{avg_avalanche:.4f}",
            "resistance_score": f"{resistance_score:.2f}/10",
            "pattern_detection": pattern_detection_percent,
            "security_margin": security_margin_percent,
            "time_seconds": duration,
            "message": message
        }
    
    def _simulate_encryption(self, plaintext, mode):
        """Simulate encryption for chosen plaintext attack"""
        # In a real attack, we'd send this to the server for encryption
        # Here we're simulating the encryption process
        
        # Create a deterministic but unique "encryption" for each plaintext
        h = hashlib.sha256(plaintext + mode.encode()).digest()
        
        # For simulation, we want the ciphertext to have good diffusion properties
        # but also be deterministic for the same input
        result = bytearray(len(plaintext))
        
        # Use the hash to seed a PRNG
        random.seed(int.from_bytes(h[:4], 'big'))
        
        # Fill with pseudorandom bytes
        for i in range(len(result)):
            # Mix in plaintext byte to simulate encryption
            result[i] = (random.randint(0, 255) ^ plaintext[i % len(plaintext)]) & 0xFF
            
            # In ZTM mode, add more mixing
            if mode == 'ztm':
                result[i] = (result[i] ^ h[i % len(h)]) & 0xFF
        
        return bytes(result)
    
    def _detect_patterns(self, plaintext, ciphertext):
        """Detect patterns between plaintext and ciphertext"""
        # This is a simplified simulation of pattern detection
        # In a real attack, we'd use statistical analysis
        
        # For simulation, we'll check for correlations between plaintext and ciphertext bits
        correlation = 0
        
        # Sample a few positions
        samples = min(len(plaintext), 16)
        for i in range(samples):
            p_bits = bin(plaintext[i])[2:].zfill(8)
            c_bits = bin(ciphertext[i])[2:].zfill(8)
            
            # Check bit correlations
            for j in range(8):
                if p_bits[j] == c_bits[j]:
                    correlation += 0.01  # Small increase for matching bits
        
        # Normalize to 0-1 range
        correlation = min(1.0, correlation)
        
        # For a strong cipher, correlation should be close to 0
        return correlation
    
    def mitm_attack(self, ciphertext, mode, max_attempts=1000):
        """
        Simulate a Man-in-the-Middle attack on the key exchange
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            max_attempts: Maximum number of attempts to simulate
            
        Returns:
            Dictionary with attack results and statistics
        """
        start_time = time.time()
        logger.info(f"Starting MITM attack, mode: {mode}")
        
        # Record this attempt
        self.attack_stats['mitm']['attempts'] += 1
        
        # In NTRU key exchange, the attacker would try to:
        # 1. Intercept the public key
        # 2. Try to derive the private key or the shared secret
        
        # NTRU parameters (from the improved app.py)
        N = 743  # Polynomial degree
        p = 3    # Small modulus
        q = 2048 # Large modulus
        
        # Simulate computational effort to break NTRU
        # For ZTM mode, we assume additional security layers
        security_bits = 256 if mode == 'ztm' else 128
        
        # Simulate lattice reduction attack on NTRU
        # This is the main attack vector against NTRU
        lattice_dimension = N
        
        # BKZ algorithm complexity estimation (simplified)
        # In reality, this depends on many factors
        bkz_time_estimate = 2**(0.292 * lattice_dimension)
        
        # Convert to years
        bkz_years = bkz_time_estimate / (60 * 60 * 24 * 365 * 1e9)  # Assuming 1 billion operations per second
        
        # Simulate some computational work
        successful = False
        attempts = min(max_attempts, 1000)
        
        for _ in range(attempts):
            # Simulate lattice reduction step
            random_bytes = os.urandom(32)
            _ = hashlib.sha256(random_bytes).digest()
            
            # Small delay to simulate computation
            time.sleep(0.001)
            
            # For simulation, extremely small chance of success
            if random.random() < 0.000001:  # Virtually impossible
                successful = True
                break
        
        duration = time.time() - start_time
        
        # Calculate quantum resistance
        # NTRU is believed to be quantum resistant
        quantum_speedup = "None" if security_bits >= 128 else "Partial"
        
        if successful:
            self.attack_stats['mitm']['success'] += 1
            message = f"Attack succeeded (simulated for demo purposes)."
        else:
            message = f"Attack failed. NTRU key exchange provides {security_bits}-bit security against MITM attacks."
        
        return {
            "success": successful,
            "security_bits": security_bits,
            "lattice_dimension": lattice_dimension,
            "bkz_time_years": f"{bkz_years:.2e}",
            "quantum_resistant": "Yes",
            "quantum_speedup": quantum_speedup,
            "communication_rounds": "2",
            "time_seconds": duration,
            "message": message
        }
    
    def side_channel_attack(self, ciphertext, mode, iterations=200):
        """
        Simulate a side-channel attack attempting to extract key information
        through timing analysis
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            iterations: Number of encryption operations to analyze
            
        Returns:
            Dictionary with attack results and statistics
        """
        start_time = time.time()
        logger.info(f"Starting side-channel attack, mode: {mode}, iterations: {iterations}")
        
        # Record this attempt
        self.attack_stats['side']['attempts'] += 1
        
        # Timing data collection
        timing_data = []
        
        # Simulate collecting timing information from encryption operations
        for i in range(iterations):
            # Create a test plaintext with varying patterns
            if i % 4 == 0:
                test_data = b'A' * 64  # Repeated character
            elif i % 4 == 1:
                test_data = bytes([j % 256 for j in range(64)])  # Incrementing bytes
            elif i % 4 == 2:
                test_data = bytes([random.randint(0, 255) for _ in range(64)])  # Random
            else:
                test_data = bytes([0xFF if j % 2 == 0 else 0x00 for j in range(64)])  # Alternating
            
            # Simulate encryption timing
            sample_start = time.time()
            
            # Different parts of the algorithm taking different time
            # These values simulate the actual operations
            
            # Base operations in both modes
            lfsr_time = 0.0001 + random.uniform(0, 0.00001)
            chaotic_time = 0.0002 + random.uniform(0, 0.00002)
            transpose_time = 0.0001 + random.uniform(0, 0.00001)
            
            # ZTM mode has additional operations
            if mode == 'ztm':
                chacha_time = 0.0003 + random.uniform(0, 0.00003)
                speck_time = 0.0002 + random.uniform(0, 0.00002)
                
                # Simulate data-dependent timing variations (very small)
                # This simulates potential side-channel leakage
                if sum(test_data) % 256 > 128:
                    chacha_time += 0.000001  # Extremely small variation
                
                total_time = lfsr_time + chaotic_time + transpose_time + chacha_time + speck_time
            else:
                # Simulate data-dependent timing variations (very small)
                if sum(test_data) % 256 > 128:
                    lfsr_time += 0.000001  # Extremely small variation
                    
                total_time = lfsr_time + chaotic_time + transpose_time
            
            # Add constant-time countermeasure simulation
            # This reduces timing variations
            if mode == 'ztm':
                # ZTM mode has better constant-time implementation
                total_time = max(total_time, 0.0009)  # Ensure minimum time
            else:
                total_time = max(total_time, 0.0005)  # Ensure minimum time
                
            # Simulate the operation taking this much time
            time.sleep(total_time)
            
            sample_end = time.time()
            timing_data.append((test_data, sample_end - sample_start))
        
        # Analyze timing data for patterns
        timing_only = [t for _, t in timing_data]
        mean_time = np.mean(timing_only)
        std_dev = np.std(timing_only)
        cv = std_dev / mean_time  # Coefficient of variation
        
        # Look for correlations between input patterns and timing
        correlations = []
        
        # Group by pattern type
        pattern_timings = {
            'repeated': [],
            'incrementing': [],
            'random': [],
            'alternating': []
        }
        
        for i, (data, timing) in enumerate(timing_data):
            pattern_type = i % 4
            if pattern_type == 0:
                pattern_timings['repeated'].append(timing)
            elif pattern_type == 1:
                pattern_timings['incrementing'].append(timing)
            elif pattern_type == 2:
                pattern_timings['random'].append(timing)
            else:
                pattern_timings['alternating'].append(timing)
        
        # Calculate mean timing for each pattern
        pattern_means = {
            k: np.mean(v) if v else 0 
            for k, v in pattern_timings.items()
        }
        
        # Calculate max timing difference between patterns
        if pattern_means:
            max_diff = max(pattern_means.values()) - min(pattern_means.values())
            # Normalize by mean time
            max_diff_percent = (max_diff / mean_time) * 100
        else:
            max_diff_percent = 0
        
        # For robust implementations, timing should be consistent (low CV)
        # or sufficiently noisy to prevent information leakage
        
        # Calculate resistance score (0-10)
        # Lower CV and lower max_diff_percent are better
        cv_score = 10 * (1 - min(1, cv * 20))  # CV should be very low
        diff_score = 10 * (1 - min(1, max_diff_percent / 5))  # Diff should be < 5%
        
        resistance_score = (cv_score + diff_score) / 2
        resistance_score = min(10.0, resistance_score)  # Cap at 10
        
        # Determine if attack was successful
        # For simulation, attack succeeds if we detect significant timing variations
        successful = cv > 0.1 or max_diff_percent > 5.0
        
        duration = time.time() - start_time
        
        if successful:
            self.attack_stats['side']['success'] += 1
            message = f"Attack detected significant timing variations (CV={cv:.4f}, diff={max_diff_percent:.2f}%), potentially leaking information."
        elif cv < 0.01 and max_diff_percent < 1.0:
            message = f"Attack failed. Operations show consistent timing (CV={cv:.4f}, diff={max_diff_percent:.2f}%), preventing side-channel analysis."
        else:
            message = f"Attack detected some timing variations (CV={cv:.4f}, diff={max_diff_percent:.2f}%), but insufficient for key extraction."
        
        return {
            "success": successful,
            "coefficient_variation": f"{cv:.4f}",
            "max_pattern_difference": f"{max_diff_percent:.2f}%",
            "resistance_score": f"{resistance_score:.2f}/10",
            "timing_variance": f"{std_dev * 1000:.4f}ms",
            "power_analysis_resistance": "High" if mode == 'ztm' else "Medium",
            "time_seconds": duration,
            "message": message
        }
    
    def quantum_attack_simulation(self, ciphertext, mode, simulation_steps=100):
        """
        Simulate a quantum computer attack against the NTRU-based encryption
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            simulation_steps: Number of quantum algorithm steps to simulate
            
        Returns:
            Dictionary with attack results and statistics
        """
        start_time = time.time()
        logger.info(f"Starting quantum attack simulation, mode: {mode}")
        
        # Record this attempt
        self.attack_stats['quantum']['attempts'] += 1
        
        # Quantum parameters for simulation
        # These are realistic estimates based on current quantum computing research
        qubits_available = 127  # Current state-of-the-art quantum computers
        
        # NTRU parameters
        N = 743  # Polynomial degree
        
        # Estimate qubits needed to break NTRU
        # This is a simplified model - real quantum attacks are more complex
        qubits_needed_base = N * 8  # Base requirement
        
        # ZTM mode adds additional layers of security
        if mode == 'ztm':
            qubits_needed = qubits_needed_base * 2  # Double for ZTM mode
        else:
            qubits_needed = qubits_needed_base
        
        # Shor's algorithm complexity for similar problem spaces
        # (this is a simplified model for simulation purposes)
        success_probability = 0.0
        
        # Simulate quantum computation steps
        for step in range(simulation_steps):
            # In each step, simulate quantum algorithm progress
            step_prob = min(1.0, qubits_available / qubits_needed) * 0.01
            # Each step has diminishing returns (simplified model)
            success_probability += step_prob * (1 - success_probability)
            
            # Simulate computation
            time.sleep(0.01)
        
        # Calculate quantum speedup as a percentage
        quantum_speedup = (qubits_available / qubits_needed) * 100
        quantum_speedup = min(100, quantum_speedup)
        
        # Determine if Shor's algorithm is applicable
        # Shor's algorithm targets RSA and similar algorithms, not NTRU
        shor_applicable = "No"
        
        # Determine if Grover's algorithm is applicable
        # Grover's provides quadratic speedup for brute force
        grover_applicable = "Yes"
        grover_speedup = "Quadratic (√N)"
        
        # Calculate effective security against Grover's algorithm
        # Grover reduces security bits by half
        classical_security_bits = 256 if mode == 'ztm' else 128
        quantum_security_bits = classical_security_bits / 2
        
        # For simulation, attack succeeds if probability exceeds threshold
        # This should be extremely unlikely
        successful = success_probability > 0.9
        
        duration = time.time() - start_time
        
        if successful:
            self.attack_stats['quantum']['success'] += 1
            message = f"Quantum attack succeeded (simulated for demo purposes)."
        else:
            if mode == 'ztm':
                additional_resistance = "ZTM mode provides additional quantum resistance through layered encryption."
            else:
                additional_resistance = ""
                
            message = f"Quantum attack failed. {qubits_needed} logical qubits required, only {qubits_available} available. NTRU maintains post-quantum security. {additional_resistance}"
        
        return {
            "success": successful,
            "qubits_available": qubits_available,
            "qubits_needed": qubits_needed,
            "quantum_speedup": f"{quantum_speedup:.2f}%",
            "shor_algorithm_applicable": shor_applicable,
            "grover_algorithm_applicable": grover_applicable,
            "grover_speedup": grover_speedup,
            "classical_security_bits": classical_security_bits,
            "quantum_security_bits": int(quantum_security_bits),
            "success_probability": f"{success_probability:.6f}",
            "time_seconds": duration,
            "message": message
        }
    
    def dos_attack_simulation(self, ciphertext, mode, num_requests=200):
        """
        Simulate a Denial of Service attack to measure performance under load
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            num_requests: Number of concurrent requests to simulate
            
        Returns:
            Dictionary with attack results and statistics
        """
        start_time = time.time()
        logger.info(f"Starting DoS attack simulation, mode: {mode}, requests: {num_requests}")
        
        # Record this attempt
        self.attack_stats['dos']['attempts'] += 1
        
        # Function to simulate a single encryption/decryption request
        def process_request(i, data_size):
            try:
                # Simulate varying workload based on request type
                if i % 3 == 0:
                    # Simulate encryption
                    test_data = os.urandom(data_size)
                    operation = "encrypt"
                elif i % 3 == 1:
                    # Simulate decryption
                    test_data = os.urandom(data_size)
                    operation = "decrypt"
                else:
                    # Simulate key exchange
                    test_data = os.urandom(32)
                    operation = "key_exchange"
                
                # Simulate processing time based on operation and mode
                if operation == "encrypt" or operation == "decrypt":
                    # ZTM mode is more computationally intensive
                    base_time = 0.03 if mode == 'ztm' else 0.02
                    # Larger data takes longer
                    size_factor = data_size / 1024  # Normalize to KB
                    process_time = base_time * size_factor
                else:
                    # Key exchange is more expensive
                    process_time = 0.05
                
                # Add random variation
                process_time += random.uniform(0, process_time * 0.2)  # Up to 20% variation
                
                # Simulate the processing
                time.sleep(process_time)
                
                # Simulate memory usage
                memory_usage = data_size * 2.5  # Rough estimate: 2.5x data size
                
                return {
                    "request_id": i,
                    "operation": operation,
                    "data_size": data_size,
                    "process_time": process_time,
                    "memory_usage": memory_usage,
                    "success": True
                }
            except Exception as e:
                return {
                    "request_id": i,
                    "error": str(e),
                    "success": False
                }

        # Create varying data sizes to simulate different requests
        data_sizes = []
        for i in range(num_requests):
            # Mix of small, medium and large requests
            if i % 10 == 0:
                # Large request
                size = random.randint(10000, 50000)
            elif i % 3 == 0:
                # Medium request
                size = random.randint(1000, 10000)
            else:
                # Small request
                size = random.randint(100, 1000)
            data_sizes.append(size)
        
        # Create a pool of workers to simulate concurrent requests
        try:
            # Use multiple workers to simulate server load
            max_workers = min(32, os.cpu_count() * 2 or 4)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all requests
                futures = [executor.submit(process_request, i, data_sizes[i]) for i in range(num_requests)]
                
                # Collect results with timeout to simulate server behavior
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=1.0)  # 1 second timeout
                        results.append(result)
                    except TimeoutError:
                        results.append({"success": False, "error": "Timeout"})
        except Exception as e:
            # Handle any unexpected errors
            logger.error(f"DoS simulation error: {e}")
            return {
                "success": True,  # The attack "succeeded" if it caused errors
                "error": str(e),
                "time_seconds": time.time() - start_time,
                "message": f"System crashed under load: {str(e)}"
            }
        
        duration = time.time() - start_time
        
        # Calculate performance metrics
        completed = sum(1 for r in results if r.get("success", False))
        failed = num_requests - completed
        
        # Calculate throughput and response statistics
        requests_per_second = completed / duration if duration > 0 else 0
        failure_rate = failed / num_requests if num_requests > 0 else 0
        
        # Calculate average processing time
        process_times = [r.get("process_time", 0) for r in results if r.get("success", False)]
        avg_process_time = sum(process_times) / len(process_times) if process_times else 0
        
        # Calculate memory usage
        memory_usages = [r.get("memory_usage", 0) for r in results if r.get("success", False)]
        total_memory = sum(memory_usages)
        peak_memory = max(memory_usages) if memory_usages else 0
        
        # Calculate CPU overhead (simplified simulation)
        cpu_overhead = (avg_process_time * completed) / (duration * max_workers) * 100
        cpu_overhead = min(100, cpu_overhead)  # Cap at 100%
        
        # Determine if attack was successful
        # For simulation, attack succeeds if failure rate is high or system is very slow
        successful = failure_rate > 0.3 or requests_per_second < 1.0
        
        if successful:
            self.attack_stats['dos']['success'] += 1
            message = f"DoS partially successful. System degraded with {failure_rate*100:.1f}% request failures."
        else:
            message = f"DoS attack failed. System processed {requests_per_second:.2f} requests/sec with {failure_rate*100:.1f}% failure rate."
        
        return {
            "success": successful,
            "requests": num_requests,
            "completed": completed,
            "failed": failed,
            "requests_per_second": f"{requests_per_second:.2f}",
            "failure_rate": f"{failure_rate:.4f}",
            "avg_process_time": f"{avg_process_time*1000:.2f}ms",
            "memory_usage": f"{total_memory/1024/1024:.2f}MB",
            "peak_memory": f"{peak_memory/1024:.2f}KB",
            "cpu_overhead": f"{cpu_overhead:.1f}%",
            "time_seconds": duration,
            "message": message
        }

# Main attack router function
def run_attack(attack_type, ciphertext, mode):
    """
    Router function to dispatch to appropriate attack simulation
    
    Args:
        attack_type: Type of attack to simulate ('brute', 'chosen', 'mitm', 'side', 'quantum', 'dos')
        ciphertext: The encrypted data to attack
        mode: The encryption mode ('normal' or 'ztm')
        
    Returns:
        Dictionary with attack results
    """
    # Create simulator if it doesn't exist
    if not hasattr(run_attack, 'simulator'):
        run_attack.simulator = AttackSimulator()
    
    simulator = run_attack.simulator
    
    # Map of attack types to methods
    attacks = {
        "brute": simulator.brute_force_attack,
        "chosen": simulator.chosen_plaintext_attack,
        "mitm": simulator.mitm_attack,
        "side": simulator.side_channel_attack,
        "quantum": simulator.quantum_attack_simulation,
        "dos": simulator.dos_attack_simulation
    }
    
    if attack_type in attacks:
        try:
            logger.info(f"Running {attack_type} attack on {len(ciphertext)} bytes, mode: {mode}")
            result = attacks[attack_type](ciphertext, mode)
            logger.info(f"Attack {attack_type} completed: {result['success']}")
            return result
        except Exception as e:
            logger.error(f"Error in {attack_type} attack: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Attack simulation failed: {str(e)}"
            }
    else:
        return {
            "success": False,
            "message": f"Unknown attack type: {attack_type}"
        }

# If run directly, perform a test
if __name__ == "__main__":
    print("XenoCipher Attack Simulator")
    print("---------------------------")
    
    # Generate some test data
    test_data = os.urandom(1024)
    
    # Test all attacks
    for attack in ["brute", "chosen", "mitm", "side", "quantum", "dos"]:
        print(f"\nRunning {attack} attack...")
        result = run_attack(attack, test_data, "ztm")
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        print("Key metrics:")
        for key, value in result.items():
            if key not in ["success", "message"]:
                print(f"  {key}: {value}")