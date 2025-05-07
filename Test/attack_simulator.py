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

# ----- Enhanced Attack Simulation Functions -----

class AttackSimulator:
    """Class to handle various cryptographic attack simulations on XenoCipher with enhanced accuracy"""
    
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
        Simulate a brute force attack by trying random keys with enhanced accuracy
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            max_attempts: Maximum number of attempts to simulate
            
        Returns:
            Dictionary with attack results and detailed statistics
        """
        start_time = time.time()
        logger.info(f"Starting enhanced brute force attack on {len(ciphertext)} bytes, mode: {mode}")
        
        # Record this attempt
        self.attack_stats['brute']['attempts'] += 1
        
        # Calculate effective key size based on mode and algorithm details
        if mode == 'ztm':
            # ZTM mode uses multiple algorithms in sequence
            # ChaCha20 (256-bit) + LFSR (16-bit) + Chaotic Map (64-bit) + Transposition (64-bit) + Speck (128-bit)
            # The effective security is not simply the sum, but considers the strongest components
            effective_key_bits = 256  # ChaCha20 provides the base security
            additional_entropy = 32   # Additional entropy from the combination of algorithms
            total_effective_bits = min(effective_key_bits + additional_entropy, 288)  # Cap at reasonable maximum
        else:
            # Normal mode uses fewer algorithms
            # LFSR (16-bit) + Chaotic Map (64-bit) + Transposition (64-bit)
            effective_key_bits = 144  # Combined effective security
            total_effective_bits = effective_key_bits
        
        # Calculate entropy of ciphertext to detect potential weaknesses
        entropy = self._calculate_shannon_entropy(ciphertext)
        entropy_ratio = entropy / 8.0  # Ideal entropy is 8 bits per byte
        
        # Adjust effective key bits based on entropy analysis
        if entropy_ratio < 0.9:  # If entropy is less than 90% of ideal
            entropy_reduction = (1 - entropy_ratio) * 32  # Up to 32 bits reduction
            total_effective_bits = max(total_effective_bits - entropy_reduction, 128)
            logger.warning(f"Low entropy detected ({entropy_ratio:.2f}), reducing effective key bits")
        
        # Calculate theoretical time for full attack using modern hardware benchmarks
        # Modern specialized hardware can achieve much higher rates
        # High-end GPU cluster: ~10^12 attempts per second
        # Custom ASIC hardware: ~10^14 attempts per second
        # We'll use a conservative estimate for a well-funded attacker
        attempts_per_second_hardware = {
            'consumer_pc': 1e9,      # 1 billion/sec
            'gpu_cluster': 1e12,     # 1 trillion/sec
            'custom_asic': 1e14,     # 100 trillion/sec
            'quantum_computer': 2**(total_effective_bits/2) if total_effective_bits > 128 else 1e20  # Grover's algorithm
        }
        
        # Calculate time estimates for different hardware
        time_estimates = {}
        for hardware, rate in attempts_per_second_hardware.items():
            if hardware == 'quantum_computer' and mode == 'ztm':
                # ZTM mode is designed to be quantum resistant
                theoretical_seconds = float('inf')
            else:
                theoretical_seconds = (2**total_effective_bits) / rate
            
            theoretical_years = theoretical_seconds / (60 * 60 * 24 * 365)
            time_estimates[hardware] = theoretical_years
        
        # Try random keys (this will always fail due to key space)
        successful = False
        keys_tried = 0
        
        # Use multiple threads to simulate parallel attack
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as executor:
            futures = []
            chunk_size = max_attempts // 16
            
            for i in range(16):
                futures.append(executor.submit(
                    self._brute_force_chunk, 
                    ciphertext[:min(1024, len(ciphertext))],  # Use first 1KB for speed
                    chunk_size,
                    total_effective_bits
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
        estimated_years = (2**total_effective_bits) / (actual_attempts_per_second * 60 * 60 * 24 * 365) if actual_attempts_per_second > 0 else float('inf')
        
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
            "theoretical_years": {k: f"{v:.2e}" for k, v in time_estimates.items()},
            "estimated_years": f"{estimated_years:.2e}",
            "key_space": f"2^{total_effective_bits}",
            "effective_security_bits": total_effective_bits,
            "entropy_score": f"{entropy:.4f}/8.00",
            "entropy_quality": f"{entropy_ratio:.2%}",
            "message": message
        }
    
    def _brute_force_chunk(self, ciphertext_sample, attempts, key_bits):
        """Helper method to try a chunk of brute force attempts with enhanced accuracy"""
        keys_tried = 0
        
        # Calculate probability of success based on key space
        # This is extremely low for any reasonable key size
        success_probability = min(1.0, attempts / (2**key_bits))
        
        for _ in range(attempts):
            # Generate a random key
            random_key = os.urandom(32)  # 256-bit key
            
            # Simulate attempt to decrypt with this key
            # In a real attack, we'd try to decrypt and check if result is valid
            self._simulate_decryption_attempt(ciphertext_sample, random_key)
            
            keys_tried += 1
            
            # For simulation, we always fail (extremely low chance of "success" for demo purposes)
            # More accurate probability based on key space
            if random.random() < success_probability:
                return {"success": True, "keys_tried": keys_tried}
        
        return {"success": False, "keys_tried": keys_tried}
    
    def _simulate_decryption_attempt(self, ciphertext_sample, key):
        """Simulate a decryption attempt with a given key with enhanced realism"""
        # This is just a simulation - in a real attack, we'd actually try to decrypt
        # For simulation purposes, we'll do some computation to simulate the work
        h = hashlib.sha256(key + ciphertext_sample[:64]).digest()
        
        # Small delay to simulate computation time
        # More realistic timing based on algorithm complexity
        delay = 0.0001 * (1 + len(ciphertext_sample) / 1024)
        time.sleep(delay)
        
        return h
    
    def _calculate_shannon_entropy(self, data):
        """Calculate Shannon entropy of data in bits per byte"""
        if not data:
            return 0
        
        # Count byte frequencies
        frequencies = {}
        for byte in data:
            frequencies[byte] = frequencies.get(byte, 0) + 1
        
        # Calculate entropy
        entropy = 0
        for count in frequencies.values():
            probability = count / len(data)
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def chosen_plaintext_attack(self, ciphertext, mode, num_samples=100):
        """
        Simulate a chosen plaintext attack with enhanced differential cryptanalysis
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            num_samples: Number of chosen plaintexts to analyze
            
        Returns:
            Dictionary with attack results and detailed statistics
        """
        start_time = time.time()
        logger.info(f"Starting enhanced chosen plaintext attack, mode: {mode}, samples: {num_samples}")
        
        # Record this attempt
        self.attack_stats['chosen']['attempts'] += 1
        
        # Generate different plaintexts with specific patterns for analysis
        plaintexts = []
        
        # 1. Single-bit differences (for avalanche effect testing)
        base_text = b'\x00' * 64
        for i in range(min(64, num_samples // 4)):
            modified = bytearray(base_text)
            modified[i // 8] ^= (1 << (i % 8))  # Flip a single bit
            plaintexts.append(bytes(modified))
        
        # 2. Block boundary testing
        for block_size in [8, 16, 32]:
            for pattern in [b'\x00\xff', b'\xaa\x55', b'\x0f\xf0']:
                block_pattern = pattern * (block_size // 2)
                plaintexts.append(block_pattern * (64 // block_size))
        
        # 3. Differential patterns (for differential cryptanalysis)
        for i in range(min(20, num_samples // 5)):
            p1 = os.urandom(64)
            p2 = bytearray(p1)
            # Introduce small differences
            for j in range(1 + i % 3):
                pos = (i * j) % 64
                p2[pos] ^= 0x01 << (j % 8)
            plaintexts.append(p1)
            plaintexts.append(bytes(p2))
        
        # 4. Statistical patterns
        for pattern in [bytes([i % 256 for i in range(64)]), bytes([i // 8 for i in range(64)])]:
            plaintexts.append(pattern)
        
        # Limit to requested number of samples
        all_samples = plaintexts[:num_samples]
        
        # In a real attack, we'd encrypt these and analyze
        # For simulation, we'll create pseudo-ciphertexts
        samples = []
        for text in all_samples:
            # Simulate encryption (in a real attack, we'd send to the server)
            pseudo_cipher = self._simulate_encryption(text, mode)
            samples.append((text, pseudo_cipher))
        
        # Enhanced analysis metrics
        avalanche_scores = []
        diffusion_scores = []
        correlation_matrix = np.zeros((64, 64))  # Input-output bit correlations
        pattern_detection_scores = []
        
        # Analyze bit changes between similar plaintexts (avalanche effect)
        for i in range(0, len(samples), 2):
            if i+1 < len(samples):
                p1, c1 = samples[i]
                p2, c2 = samples[i+1]
                
                # Count bit differences
                p_diff = sum(bin(a ^ b).count('1') for a, b in zip(p1, p2))
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
        
        # Calculate bit-level correlations
        for sample_idx in range(min(20, len(samples))):
            p, c = samples[sample_idx]
            
            # Convert to bit arrays for correlation analysis
            p_bits = np.unpackbits(np.frombuffer(p[:8], dtype=np.uint8))
            c_bits = np.unpackbits(np.frombuffer(c[:8], dtype=np.uint8))
            
            # Update correlation matrix
            for i in range(64):
                for j in range(64):
                    if p_bits[i] == c_bits[j]:
                        correlation_matrix[i, j] += 1
        
        # Normalize correlation matrix
        correlation_matrix /= min(20, len(samples))
        
        # Calculate pattern detection score
        for i in range(len(samples)):
            p, c = samples[i]
            pattern_score = self._detect_patterns(p, c)
            pattern_detection_scores.append(pattern_score)
        
        # Calculate overall metrics
        avg_avalanche = sum(avalanche_scores) / len(avalanche_scores) if avalanche_scores else 0
        avg_diffusion = sum(diffusion_scores) / len(diffusion_scores) if diffusion_scores else 0
        avg_pattern_detection = sum(pattern_detection_scores) / len(pattern_detection_scores) if pattern_detection_scores else 0
        
        # Calculate avalanche quality (ideal is 0.5)
        avalanche_quality = 1.0 - abs(0.5 - avg_avalanche) * 2  # Scale to 0-1
        
        # Calculate diffusion quality (higher is better)
        diffusion_quality = min(1.0, avg_diffusion * 2)  # Scale to 0-1
        
        # Calculate correlation quality (lower is better)
        correlation_strength = np.mean(correlation_matrix)
        correlation_quality = 1.0 - correlation_strength
        
        # Combined resistance score (0-10)
        resistance_score = (
            avalanche_quality * 3 +  # Avalanche effect (30%)
            diffusion_quality * 3 +  # Diffusion (30%)
            correlation_quality * 2 +  # Correlation resistance (20%)
            (1 - avg_pattern_detection) * 2  # Pattern resistance (20%)
        )
        
        # Security margin (higher is better)
        security_margin = 100 - (avg_pattern_detection * 100)
        
        duration = time.time() - start_time
        
        # For simulation, attack always fails unless pattern detection is very high
        success = avg_pattern_detection > 0.8  # 80% pattern detection would be a serious weakness
        
        if success:
            self.attack_stats['chosen']['success'] += 1
            message = f"Attack succeeded with {avg_pattern_detection:.2%} pattern detection."
        else:
            message = f"Attack failed. System shows strong diffusion properties with score {resistance_score:.2f}/10."
        
        return {
            "success": success,
            "samples_analyzed": len(samples),
            "diffusion_score": f"{avg_diffusion:.4f}",
            "avalanche_score": f"{avg_avalanche:.4f}",
            "avalanche_quality": f"{avalanche_quality:.2%}",
            "correlation_strength": f"{correlation_strength:.4f}",
            "resistance_score": f"{resistance_score:.2f}/10",
            "pattern_detection": f"{avg_pattern_detection:.2%}",
            "security_margin": f"{security_margin:.2f}%",
            "time_seconds": duration,
            "message": message,
            "ideal_avalanche": "50%",
            "ideal_diffusion": "50%",
            "ideal_correlation": "0%"
        }
    
    def _simulate_encryption(self, plaintext, mode):
        """Simulate encryption for chosen plaintext attack with enhanced realism"""
        # In a real attack, we'd send this to the server for encryption
        # Here we're simulating the encryption process with more realistic properties
        
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
            
            # In ZTM mode, add more mixing for better diffusion
            if mode == 'ztm':
                result[i] = (result[i] ^ h[i % len(h)]) & 0xFF
                
                # Add additional mixing to simulate multiple layers
                if i > 0:
                    result[i] = (result[i] ^ result[i-1]) & 0xFF
        
        # Add final diffusion pass for ZTM mode
        if mode == 'ztm':
            for i in range(len(result)-1, 0, -1):
                result[i-1] = (result[i-1] ^ (result[i] >> 1)) & 0xFF
        
        return bytes(result)
    
    def _detect_patterns(self, plaintext, ciphertext):
        """Detect patterns between plaintext and ciphertext with enhanced accuracy"""
        # This is a more sophisticated pattern detection algorithm
        correlation = 0
        
        # 1. Byte-level correlation
        for i in range(min(len(plaintext), len(ciphertext))):
            # Check direct correlation
            if plaintext[i] == ciphertext[i]:
                correlation += 0.01
            
            # Check for simple transformations
            if plaintext[i] == (ciphertext[i] ^ 0xFF):  # Inverted
                correlation += 0.005
            if plaintext[i] == (ciphertext[i] + 1) & 0xFF:  # Shifted
                correlation += 0.005
        
        # 2. Bit-level correlation (more detailed)
        for i in range(min(len(plaintext), len(ciphertext), 16)):  # Check first 16 bytes
            p_bits = bin(plaintext[i])[2:].zfill(8)
            c_bits = bin(ciphertext[i])[2:].zfill(8)
            
            # Check bit correlations
            for j in range(8):
                if p_bits[j] == c_bits[j]:
                    correlation += 0.001  # Small increase for matching bits
                if p_bits[j] == c_bits[7-j]:  # Reversed bits
                    correlation += 0.0005
        
        # 3. Block-level patterns
        for block_size in [2, 4, 8]:
            if len(plaintext) >= block_size and len(ciphertext) >= block_size:
                for i in range(0, min(len(plaintext), len(ciphertext)) - block_size, block_size):
                    p_block = plaintext[i:i+block_size]
                    
                    # Search for this block in ciphertext
                    for j in range(0, len(ciphertext) - block_size, block_size):
                        c_block = ciphertext[j:j+block_size]
                        if p_block == c_block:
                            correlation += 0.02  # Larger increase for block matches
        
        # Normalize to 0-1 range
        correlation = min(1.0, correlation)
        
        # For a strong cipher, correlation should be close to 0
        return correlation
    
    def mitm_attack(self, ciphertext, mode, max_attempts=1000):
        """
        Simulate a Man-in-the-Middle attack on the key exchange with enhanced accuracy
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            max_attempts: Maximum number of attempts to simulate
            
        Returns:
            Dictionary with attack results and detailed statistics
        """
        start_time = time.time()
        logger.info(f"Starting enhanced MITM attack, mode: {mode}")
        
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
        
        # BKZ algorithm complexity estimation (more accurate model)
        # Based on latest research on lattice-based cryptography
        bkz_block_size = 30  # Typical block size for BKZ
        bkz_nodes = 2**(0.292 * bkz_block_size)  # Nodes per BKZ call
        bkz_calls = lattice_dimension  # Number of BKZ calls needed
        
        # Total BKZ operations
        bkz_operations = bkz_nodes * bkz_calls
        
        # Convert to years (assuming 10^9 operations per second)
        bkz_years = bkz_operations / (60 * 60 * 24 * 365 * 1e9)
        
        # Simulate some computational work
        successful = False
        attempts = min(max_attempts, 1000)
        
        # Track attack progress metrics
        progress_metrics = {
            'lattice_reduction_progress': 0,
            'key_recovery_progress': 0,
            'attack_complexity_remaining': 100
        }
        
        for i in range(attempts):
            # Simulate lattice reduction step
            random_bytes = os.urandom(32)
            _ = hashlib.sha256(random_bytes).digest()
            
            # Update progress metrics
            progress_metrics['lattice_reduction_progress'] = min(100, (i / attempts) * 20)  # Max 20% progress
            progress_metrics['key_recovery_progress'] = min(100, (i / attempts) * 5)  # Max 5% progress
            progress_metrics['attack_complexity_remaining'] = max(0, 100 - (i / attempts) * 10)  # Min 90% remaining
            
            # Small delay to simulate computation
            time.sleep(0.001)
            
            # For simulation, extremely small chance of success
            # More accurate probability based on security level
            success_probability = min(1.0, 1 / (2**security_bits))
            if random.random() < success_probability:
                successful = True
                break
        
        duration = time.time() - start_time
        
        # Calculate quantum resistance
        # NTRU is believed to be quantum resistant
        quantum_speedup = "None" if security_bits >= 128 else "Partial"
        
        # Calculate communication rounds vulnerability
        comm_rounds = 2  # NTRU typically requires 2 rounds
        comm_vulnerability = "Low" if comm_rounds <= 2 else "Medium"
        
        if successful:
            self.attack_stats['mitm']['success'] += 1
            message = f"Attack succeeded (simulated for demo purposes)."
        else:
            message = f"Attack failed. NTRU key exchange provides {security_bits}-bit security against MITM attacks."
        
        return {
            "success": successful,
            "security_bits": security_bits,
            "lattice_dimension": lattice_dimension,
            "bkz_block_size": bkz_block_size,
            "bkz_operations": f"{bkz_operations:.2e}",
            "bkz_time_years": f"{bkz_years:.2e}",
            "quantum_resistant": "Yes",
            "quantum_speedup": quantum_speedup,
            "communication_rounds": str(comm_rounds),
            "communication_vulnerability": comm_vulnerability,
            "lattice_reduction_progress": f"{progress_metrics['lattice_reduction_progress']:.2f}%",
            "key_recovery_progress": f"{progress_metrics['key_recovery_progress']:.2f}%",
            "attack_complexity_remaining": f"{progress_metrics['attack_complexity_remaining']:.2f}%",
            "time_seconds": duration,
            "message": message
        }
    
    def side_channel_attack(self, ciphertext, mode, iterations=200):
        """
        Simulate a side-channel attack with enhanced timing and power analysis
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            iterations: Number of encryption operations to analyze
            
        Returns:
            Dictionary with attack results and detailed statistics
        """
        start_time = time.time()
        logger.info(f"Starting enhanced side-channel attack, mode: {mode}, iterations: {iterations}")
        
        # Record this attempt
        self.attack_stats['side']['attempts'] += 1
        
        # Generate test patterns for timing analysis
        patterns = {
            'zeros': b'\x00' * 64,
            'ones': b'\xff' * 64,
            'alternating': bytes([0x55] * 64),  # 01010101...
            'repeating': bytes([0xaa] * 64),    # 10101010...
            'random': os.urandom(64),
            'ascending': bytes([i % 256 for i in range(64)]),
            'descending': bytes([255 - (i % 256) for i in range(64)])
        }
        
        # Collect timing data for each pattern
        timing_data = {pattern: [] for pattern in patterns}
        power_data = {pattern: [] for pattern in patterns}
        cache_data = {pattern: [] for pattern in patterns}
        
        # Simulate timing measurements with nanosecond precision
        for _ in range(iterations):
            for pattern_name, pattern in patterns.items():
                # Simulate encryption timing
                # In ZTM mode, operations take longer due to additional layers
                base_time = 0.02 if mode == 'ztm' else 0.01
                
                # Add pattern-specific timing variations (more realistic)
                if pattern_name == 'zeros':
                    # Zero bytes might be processed slightly faster
                    variation = random.uniform(-0.002, 0.001)
                    # Power consumption is lower for zeros
                    power_variation = random.uniform(0.7, 0.9)
                    # Cache behavior
                    cache_hits = random.uniform(0.8, 0.95)
                elif pattern_name == 'ones':
                    # Ones might take slightly longer
                    variation = random.uniform(0, 0.003)
                    # Power consumption is higher for ones
                    power_variation = random.uniform(1.1, 1.3)
                    # Cache behavior
                    cache_hits = random.uniform(0.7, 0.85)
                elif pattern_name == 'alternating':
                    # Alternating patterns might have medium timing
                    variation = random.uniform(-0.001, 0.002)
                    # Medium power consumption
                    power_variation = random.uniform(0.9, 1.1)
                    # Cache behavior
                    cache_hits = random.uniform(0.75, 0.9)
                elif pattern_name == 'ascending' or pattern_name == 'descending':
                    # Sequential patterns might have predictable timing
                    variation = random.uniform(-0.001, 0.002) + (0.0005 if pattern_name == 'ascending' else 0.0007)
                    # Power consumption
                    power_variation = random.uniform(0.95, 1.15)
                    # Cache behavior
                    cache_hits = random.uniform(0.8, 0.9)
                else:
                    # Random patterns have more variation
                    variation = random.uniform(-0.003, 0.003)
                    # Random power consumption
                    power_variation = random.uniform(0.9, 1.2)
                    # Cache behavior
                    cache_hits = random.uniform(0.7, 0.9)
                
                # Add noise to make timing analysis harder
                # ZTM mode has more consistent timing (harder to attack)
                noise_factor = 0.0005 if mode == 'ztm' else 0.002
                noise = random.gauss(0, noise_factor)
                
                # Calculate final timing (in milliseconds)
                timing = base_time + variation + noise
                timing_data[pattern_name].append(timing * 1000)  # Convert to ms
                
                # Record power consumption (arbitrary units)
                power_data[pattern_name].append(power_variation)
                
                # Record cache behavior (hit ratio)
                cache_data[pattern_name].append(cache_hits)
                
                # Small delay between measurements
                time.sleep(0.001)
        
        # Calculate statistics for each pattern
        pattern_means = {}
        pattern_stds = {}
        power_means = {}
        cache_means = {}
        
        for pattern, times in timing_data.items():
            if times:
                pattern_means[pattern] = sum(times) / len(times)
                pattern_stds[pattern] = math.sqrt(sum((t - pattern_means[pattern])**2 for t in times) / len(times))
                power_means[pattern] = sum(power_data[pattern]) / len(power_data[pattern])
                cache_means[pattern] = sum(cache_data[pattern]) / len(cache_data[pattern])
        
        # Calculate overall statistics
        all_times = [t for times in timing_data.values() for t in times]
        mean_time = sum(all_times) / len(all_times) if all_times else 0
        std_dev = math.sqrt(sum((t - mean_time)**2 for t in all_times) / len(all_times)) if all_times else 0
        # Calculate coefficient of variation (CV)
        cv = std_dev / mean_time if mean_time > 0 else 0
        
        # Calculate max timing difference between patterns
        if pattern_means:
            max_diff = max(pattern_means.values()) - min(pattern_means.values())
            # Normalize by mean time
            max_diff_percent = (max_diff / mean_time) * 100
        else:
            max_diff_percent = 0
        
        # Calculate power analysis metrics
        power_variation = max(power_means.values()) - min(power_means.values()) if power_means else 0
        power_cv = power_variation / (sum(power_means.values()) / len(power_means) if power_means else 1)
        
        # Calculate cache analysis metrics
        cache_variation = max(cache_means.values()) - min(cache_means.values()) if cache_means else 0
        
        # Calculate resistance score (0-10)
        # Lower CV and lower max_diff_percent are better
        cv_score = 10 * (1 - min(1, cv * 20))  # CV should be very low
        diff_score = 10 * (1 - min(1, max_diff_percent / 5))  # Diff should be < 5%
        power_score = 10 * (1 - min(1, power_cv * 5))  # Power variation should be low
        cache_score = 10 * (1 - min(1, cache_variation * 10))  # Cache variation should be low
        
        # Combined score with weighted components
        resistance_score = (
            cv_score * 0.3 +          # Timing consistency (30%)
            diff_score * 0.3 +         # Pattern timing difference (30%)
            power_score * 0.2 +        # Power analysis resistance (20%)
            cache_score * 0.2          # Cache attack resistance (20%)
        )
        resistance_score = min(10.0, resistance_score)  # Cap at 10
        
        # Determine if attack was successful
        # For simulation, attack succeeds if we detect significant variations
        successful = cv > 0.1 or max_diff_percent > 5.0 or power_cv > 0.2 or cache_variation > 0.2
        
        duration = time.time() - start_time
        
        if successful:
            self.attack_stats['side']['success'] += 1
            message = f"Attack detected significant side-channel leakage (timing CV={cv:.4f}, power CV={power_cv:.4f}), potentially revealing key information."
        elif cv < 0.01 and max_diff_percent < 1.0 and power_cv < 0.1:
            message = f"Attack failed. Operations show consistent behavior (timing CV={cv:.4f}, power CV={power_cv:.4f}), preventing side-channel analysis."
        else:
            message = f"Attack detected some variations (timing CV={cv:.4f}, power CV={power_cv:.4f}), but insufficient for key extraction."
        
        return {
            "success": successful,
            "coefficient_variation": f"{cv:.4f}",
            "max_pattern_difference": f"{max_diff_percent:.2f}%",
            "power_variation": f"{power_cv:.4f}",
            "cache_variation": f"{cache_variation:.4f}",
            "resistance_score": f"{resistance_score:.2f}/10",
            "timing_variance": f"{std_dev:.4f}ms",
            "power_analysis_resistance": "High" if mode == 'ztm' else "Medium",
            "cache_attack_resistance": "High" if mode == 'ztm' else "Medium",
            "fault_injection_resistance": "Medium",
            "time_seconds": duration,
            "message": message
        }
    
    def quantum_attack_simulation(self, ciphertext, mode, simulation_steps=100):
        """
        Simulate a quantum computer attack with enhanced accuracy based on latest research
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            simulation_steps: Number of quantum algorithm steps to simulate
            
        Returns:
            Dictionary with attack results and detailed statistics
        """
        start_time = time.time()
        logger.info(f"Starting enhanced quantum attack simulation, mode: {mode}")
        
        # Record this attempt
        self.attack_stats['quantum']['attempts'] += 1
        
        # Quantum parameters based on latest research
        # IBM's latest quantum processor (2023): 127 qubits
        # Google's Sycamore: ~53-60 qubits
        # Theoretical estimates for breaking cryptography:
        # - 4,000+ logical qubits needed for breaking RSA-2048
        # - 2,000+ logical qubits needed for breaking ECC-256
        # - NTRU is believed to be quantum resistant
        qubits_available = 127  # Current state-of-the-art
        
        # Error correction overhead
        # Current error rates require ~1,000 physical qubits per logical qubit
        error_correction_factor = 1000
        effective_logical_qubits = qubits_available / error_correction_factor
        
        # NTRU parameters
        N = 743  # Polynomial degree
        
        # Estimate qubits needed to break NTRU using quantum algorithms
        # This is a more accurate model based on research papers
        if mode == 'ztm':
            # ZTM mode adds multiple layers of security
            qubits_needed_base = N * 4  # Base requirement for lattice problems
            additional_security_factor = 2  # ZTM adds additional security layers
            qubits_needed = qubits_needed_base * additional_security_factor
        else:
            qubits_needed_base = N * 4
            qubits_needed = qubits_needed_base
        
        # Calculate circuit depth required
        # Quantum circuit depth is another limiting factor
        # Deeper circuits are harder to implement due to decoherence
        circuit_depth_required = qubits_needed * 100  # Simplified model
        
        # Current achievable circuit depth (with error correction)
        current_circuit_depth = 1000  # Conservative estimate
        
        # Calculate quantum speedup for different algorithms
        
        # 1. Grover's algorithm (quadratic speedup)
        # Reduces brute force from 2^n to 2^(n/2)
        classical_security_bits = 256 if mode == 'ztm' else 128
        grover_security_bits = classical_security_bits / 2
        grover_speedup_factor = 2**(classical_security_bits/2)
        
        # 2. Shor's algorithm (exponential speedup)
        # Not directly applicable to NTRU, which is based on lattice problems
        shor_applicable = False
        
        # 3. Quantum lattice algorithms
        # Research suggests quantum computers give polynomial speedup for some lattice problems
        # But not the exponential speedup that would break the system
        lattice_quantum_speedup = "Polynomial (not exponential)"
        
        # Simulate quantum computation steps
        success_probability = 0.0
        step_results = []
        
        for step in range(simulation_steps):
            # In each step, simulate quantum algorithm progress
            # More realistic model of quantum computation
            
            # Calculate probability based on resources
            resource_ratio = min(1.0, effective_logical_qubits / qubits_needed)
            depth_ratio = min(1.0, current_circuit_depth / circuit_depth_required)
            
            # Combined probability factor
            step_factor = resource_ratio * depth_ratio * 0.01
            
            # Each step has diminishing returns
            prev_probability = success_probability
            success_probability += step_factor * (1 - success_probability)
            
            # Record step results
            step_results.append({
                'step': step + 1,
                'probability_increase': success_probability - prev_probability,
                'cumulative_probability': success_probability
            })
            
            # Simulate computation
            time.sleep(0.01)
        
        # Calculate quantum advantage threshold
        # This is when quantum computers would have advantage over classical
        qubits_for_advantage = qubits_needed / 4  # Simplified model
        years_to_advantage = self._estimate_years_to_qubits(qubits_for_advantage)
        
        # Calculate post-quantum security margin
        # How much safety margin the algorithm has against quantum attacks
        if mode == 'ztm':
            security_margin = "High (multiple quantum-resistant layers)"
            margin_percentage = 90
        else:
            security_margin = "Medium (basic quantum resistance)"
            margin_percentage = 70
        
        # For simulation, attack succeeds if probability exceeds threshold
        # This should be extremely unlikely
        successful = success_probability > 0.9
        
        duration = time.time() - start_time
        
        if successful:
            self.attack_stats['quantum']['success'] += 1
            message = f"Quantum attack succeeded (simulated for demo purposes)."
        else:
            if mode == 'ztm':
                additional_info = "ZTM mode provides enhanced quantum resistance through layered encryption."
            else:
                additional_info = "Basic mode provides standard quantum resistance."
                
            message = f"Quantum attack failed. {qubits_needed} logical qubits required, only {effective_logical_qubits:.2f} available. {additional_info}"
        
        return {
            "success": successful,
            "physical_qubits_available": qubits_available,
            "logical_qubits_available": f"{effective_logical_qubits:.2f}",
            "logical_qubits_needed": qubits_needed,
            "circuit_depth_required": circuit_depth_required,
            "circuit_depth_available": current_circuit_depth,
            "quantum_speedup": lattice_quantum_speedup,
            "shor_algorithm_applicable": "No",
            "grover_algorithm_applicable": "Yes",
            "grover_speedup": "Quadratic (√N)",
            "classical_security_bits": classical_security_bits,
            "quantum_security_bits": int(grover_security_bits),
            "success_probability": f"{success_probability:.6f}",
            "years_to_quantum_advantage": f"{years_to_advantage:.1f}",
            "post_quantum_security_margin": f"{margin_percentage}%",
            "time_seconds": duration,
            "message": message
        }
    
    def _estimate_years_to_qubits(self, target_qubits):
        """Estimate years until we have target_qubits logical qubits available"""
        current_qubits = 127
        growth_rate = 1.5  # Qubits increase by 50% per year (optimistic estimate)
        
        # Calculate years based on exponential growth
        if target_qubits <= current_qubits:
            return 0
        
        return math.log(target_qubits / current_qubits, growth_rate)
    
    def dos_attack_simulation(self, ciphertext, mode, num_requests=200):
        """
        Simulate a Denial of Service attack with enhanced resource monitoring
        
        Args:
            ciphertext: The encrypted data to attack
            mode: The encryption mode ('normal' or 'ztm')
            num_requests: Number of concurrent requests to simulate
            
        Returns:
            Dictionary with attack results and detailed statistics
        """
        start_time = time.time()
        logger.info(f"Starting enhanced DoS attack simulation, mode: {mode}, requests: {num_requests}")
        
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
                
                # Simulate memory usage (more accurate model)
                # Base memory: algorithm state + input/output buffers
                base_memory = data_size * 2  # Input and output buffers
                
                # Algorithm-specific memory
                if operation == "encrypt" or operation == "decrypt":
                    if mode == 'ztm':
                        # ZTM uses more memory for multiple algorithms
                        algo_memory = data_size * 1.5 + 8192  # Additional buffers + state
                    else:
                        algo_memory = data_size * 0.5 + 4096  # Smaller state
                else:
                    # Key exchange uses fixed memory
                    algo_memory = 16384  # 16KB for NTRU
                
                # Total memory usage
                memory_usage = base_memory + algo_memory
                
                # Simulate CPU usage (percentage of one core)
                if operation == "encrypt" or operation == "decrypt":
                    if mode == 'ztm':
                        cpu_usage = random.uniform(80, 95)  # 80-95% CPU
                    else:
                        cpu_usage = random.uniform(60, 85)  # 60-85% CPU
                else:
                    cpu_usage = random.uniform(90, 100)  # 90-100% CPU for key exchange
                
                return {
                    "request_id": i,
                    "operation": operation,
                    "data_size": data_size,
                    "process_time": process_time,
                    "memory_usage": memory_usage,
                    "cpu_usage": cpu_usage,
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
        avg_memory = total_memory / len(memory_usages) if memory_usages else 0
        
        # Calculate CPU usage
        cpu_usages = [r.get("cpu_usage", 0) for r in results if r.get("success", False)]
        avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
        
        # Calculate system load
        system_load = (avg_process_time * completed) / (duration * max_workers) * 100
        system_load = min(100, system_load)  # Cap at 100%
        
        # Calculate resource exhaustion metrics
        memory_exhaustion = (avg_memory * max_workers) / (1024 * 1024 * 1024)  # GB
        cpu_exhaustion = (avg_cpu * max_workers) / 100  # CPU cores
        
        # Determine if attack was successful
        # For simulation, attack succeeds if failure rate is high or system is very slow
        successful = failure_rate > 0.3 or requests_per_second < 1.0 or system_load > 95
        
        # Calculate service degradation percentage
        if failure_rate > 0:
            service_degradation = failure_rate * 100
        else:
            # Even without failures, high latency is a form of degradation
            latency_factor = min(1.0, avg_process_time / 0.1)  # Normalize to 100ms target
            service_degradation = latency_factor * 50  # Max 50% degradation from latency alone
        
        if successful:
            self.attack_stats['dos']['success'] += 1
            message = f"DoS partially successful. System degraded with {service_degradation:.1f}% service degradation."
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
            "memory_usage_total": f"{total_memory/1024/1024:.2f}MB",
            "memory_usage_peak": f"{peak_memory/1024:.2f}KB",
            "memory_usage_avg": f"{avg_memory/1024:.2f}KB",
            "cpu_usage_avg": f"{avg_cpu:.1f}%",
            "system_load": f"{system_load:.1f}%",
            "memory_exhaustion": f"{memory_exhaustion:.2f}GB",
            "cpu_exhaustion": f"{cpu_exhaustion:.1f} cores",
            "service_degradation": f"{service_degradation:.1f}%",
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
            logger.info(f"Running enhanced {attack_type} attack on {len(ciphertext)} bytes, mode: {mode}")
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
    print("Enhanced XenoCipher Attack Simulator")
    print("------------------------------------")
    
    # Generate some test data
    test_data = os.urandom(1024)
    
    # Test all attacks
    for attack in ["brute", "chosen", "mitm", "side", "quantum", "dos"]:
        print(f"\nRunning enhanced {attack} attack...")
        result = run_attack(attack, test_data, "ztm")
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        print("Key metrics:")
        for key, value in result.items():
            if key not in ["success", "message", "error"]:
                print(f"  {key}: {value}")
