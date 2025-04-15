import numpy as np
import unittest

# LFSR Class for pseudo-random key stream generation
class LFSR:
    def __init__(self, taps, initial_state):
        """Initialize LFSR with tap positions and initial state."""
        self.taps = taps  # List of tap positions (0-based indices)
        self.state = initial_state  # Initial state as an integer

    def next_bit(self):
        """Generate the next bit in the sequence and update the state."""
        output_bit = self.state & 1  # Get the least significant bit
    
        # Calculate feedback bit using XOR of tapped bits
        feedback = 0
        for tap in self.taps:
            feedback ^= (self.state >> tap) & 1
    
        # Shift right and add feedback as the most significant bit
        self.state = (self.state >> 1) 
        if feedback:
            # Set the MSB based on the maximum tap position
            self.state |= (1 << max(self.taps))
    
        return output_bit
    def generate_key_stream(self, length):
        """Generate a key stream of specified length."""
        return [self.next_bit() for _ in range(length)]

# Chaotic Map Functions for encryption
def logistic_map(x, r):
    """Compute the next value in the Logistic Map sequence."""
    return r * x * (1 - x)

def chaotic_encrypt(data, r, x0, iterations):
    """Encrypt data using a chaotic map."""
    x = x0  # Initial value
    for _ in range(iterations):
        x = logistic_map(x, r)
    scaled_x = int(x * 255)  # Scale to byte range
    return bytes([b ^ scaled_x for b in data])

# Transposition Cipher Functions
def generate_permutation_matrix(size, lfsr):
    """Generate a permutation matrix using LFSR."""
    indices = []
    bit_length = int(np.log2(size)) + 1
    for _ in range(size):
        index = 0
        for _ in range(bit_length):
            index = (index << 1) | lfsr.next_bit()
        indices.append(index % size)
    return indices

def transpose(data, permutation):
    """Apply transposition to data using the permutation matrix."""
    return bytes([data[i] for i in permutation])

# Integration Function
def encrypt(plaintext, lfsr, r, x0, iterations):
    """Encrypt plaintext by integrating LFSR, chaotic map, and transposition."""
    # Generate key stream and convert to bytes
    key_stream = lfsr.generate_key_stream(len(plaintext) * 8)
    key_bytes = [int(''.join(map(str, key_stream[i:i+8])), 2)
                 for i in range(0, len(key_stream), 8)]
    
    # LFSR-based stream cipher
    lfsr_encrypted = bytes([p ^ k for p, k in zip(plaintext, key_bytes)])
    
    # Chaotic map encryption
    chaos_encrypted = chaotic_encrypt(lfsr_encrypted, r, x0, iterations)
    
    # Transposition cipher
    perm = generate_permutation_matrix(len(chaos_encrypted), lfsr)
    ciphertext = transpose(chaos_encrypted, perm)
    
    return ciphertext

# Test Cases
class TestXenoCipher(unittest.TestCase):
    def test_lfsr(self):
        """Test LFSR key stream generation."""
        lfsr = LFSR([0, 2], 0b101)  # Taps at positions 0 and 2, initial state 101
        key_stream = lfsr.generate_key_stream(5)
        expected = [1, 0, 1, 0, 0]  # Expected output for this configuration
        self.assertEqual(key_stream, expected)

    def test_chaotic_encrypt(self):
        """Test chaotic map encryption."""
        data = b"test"
        encrypted = chaotic_encrypt(data, r=3.8, x0=0.5, iterations=10)
        self.assertEqual(len(encrypted), len(data))  # Length should remain unchanged

    def test_transpose(self):
        """Test transposition cipher."""
        lfsr = LFSR([0, 2], 0b101)
        data = b"abcd"
        perm = generate_permutation_matrix(len(data), lfsr)
        transposed = transpose(data, perm)
        self.assertEqual(len(transposed), len(data))  # Length should remain unchanged

    def test_encrypt(self):
        """Test integrated encryption."""
        plaintext = b"test"
        lfsr = LFSR([0, 2], 0b101)
        ciphertext = encrypt(plaintext, lfsr, r=3.8, x0=0.5, iterations=10)
        self.assertIsInstance(ciphertext, bytes)  # Output should be bytes

if __name__ == "__main__":
    unittest.main()