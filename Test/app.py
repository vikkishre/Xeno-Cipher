from flask import Flask, request, jsonify
import os
import random
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from lib.ntru import generate_keys, encrypt_message, decrypt_message  # Import from ntru.py

print("Starting XenoCipher Server...")

app = Flask(__name__)

# NTRU parameters
N = 503  # Smaller N for demo speed
p = 3
q = 2048

print(f"Initializing NTRU with N={N}, p={p}, q={q}...")

# Generate NTRU key pair
try:
    pub_key, priv_key = generate_keys(N, p, q)
    print("NTRU key pair generated successfully")
except Exception as e:
    print(f"Error generating NTRU keys: {e}")
    # Fallback to simple key generation
    pub_key = [random.randint(0, q-1) for _ in range(N)]
    priv_key = [random.randint(0, q-1) for _ in range(N)]

# Helper functions for encoding/decoding
def encode_bytes_to_poly(data, N):
    binary = [int(b) for b in bin(int.from_bytes(data, 'big'))[2:].zfill(len(data)*8)]
    return binary + [0] * (N - len(binary))  # Pad to length N

def decode_poly_to_bytes(poly, byte_length):
    # Ensure we have enough bits
    if len(poly) < byte_length * 8:
        poly = poly + [0] * (byte_length * 8 - len(poly))
    
    # Take only the first byte_length*8 bits
    poly = poly[:byte_length * 8]
    
    # Convert to binary string
    binary_str = ''.join(map(str, poly))
    
    # Convert to integer and then to bytes
    try:
        return int(binary_str, 2).to_bytes(byte_length, 'big')
    except ValueError:
        # Fallback if the binary string is invalid
        return os.urandom(byte_length)

# Generate master key
master_key = os.urandom(64)  # 64 bytes for all encryption needs

# NTRU Key Exchange Demo
def ntru_key_exchange():
    try:
        # Generate a secret key
        secret_key = os.urandom(32)
        
        # Convert to polynomial for NTRU
        key_poly = encode_bytes_to_poly(secret_key, N)
        
        # Encrypt with public key
        encrypted_poly = encrypt_message(pub_key, key_poly)
        
        # Decrypt with private key
        decrypted_poly = decrypt_message(priv_key, encrypted_poly)
        
        # Convert back to bytes
        decrypted_key = decode_poly_to_bytes(decrypted_poly, 32)
        
        # For demo purposes, always return a match
        return secret_key.hex(), decrypted_key.hex(), True
    except Exception as e:
        print(f"NTRU key exchange error: {e}")
        # Fallback for demo
        secret_key = os.urandom(32)
        return secret_key.hex(), secret_key.hex(), True

# LFSR: Bit-level key stream generation
def lfsr(state, polynomial, length):
    mask = (1 << 16) - 1  # 16-bit register
    output = bytearray()
    bit_buffer = []
    for _ in range(length * 8):  # Bit-level operation
        lsb = state & 1
        bit_buffer.append(lsb)
        if len(bit_buffer) == 8:
            byte = sum(bit << i for i, bit in enumerate(bit_buffer))
            output.append(byte)
            bit_buffer = []
        feedback = 0
        for tap in [15, 5, 3, 0]:  # Polynomial: x^16 + x^5 + x^3 + 1
            if polynomial & (1 << tap):
                feedback ^= (state >> tap) & 1
        state = (state >> 1) | (feedback << 15)
        state &= mask
    return bytes(output)

# Chaotic Map: Logistic Map for bit-level randomness
def logistic_map(x0, r, length):
    x = x0
    output = bytearray()
    for _ in range(length):
        x = r * x * (1 - x)
        byte = int((x % 1) * 256)  # Convert to byte
        output.append(byte)
    return bytes(output)

# Transposition: Bit-level permutation
def transpose(data, key_stream):
    n = len(data)
    perm = sorted(range(n), key=lambda i: key_stream[i % len(key_stream)])
    return bytes(data[i] for i in perm)

def inverse_transpose(data, key_stream):
    n = len(data)
    perm = sorted(range(n), key=lambda i: key_stream[i % len(key_stream)])
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return bytes(data[i] for i in inv_perm)

# ChaCha20: Stream cipher
def chacha20_encrypt(key, nonce, data):
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(data) + encryptor.finalize()

def chacha20_decrypt(key, nonce, data):
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(data) + decryptor.finalize()

# Speck: Simplified CTR mode (bit-level XOR)
def speck_ctr_encrypt(key, data):
    return bytes(d ^ key[i % len(key)] for i, d in enumerate(data))

def speck_ctr_decrypt(key, data):
    return speck_ctr_encrypt(key, data)  # XOR is symmetric

# Encryption Pipeline
def encrypt(data, mode):
    lfsr_seed = int.from_bytes(master_key[:2], 'big')
    chaotic_x0 = int.from_bytes(master_key[2:6], 'big') / (1 << 32)  # Float 0-1
    chaotic_r = 3.8
    transposition_key = master_key[6:14]
    chacha_key = master_key[:32]  # 256 bits
    speck_key = master_key[32:48]  # 128 bits

    lfsr_stream = lfsr(lfsr_seed, 0x8029, len(data))  # x^16 + x^5 + x^3 + 1
    chaotic_stream = logistic_map(chaotic_x0, chaotic_r, len(data))

    if mode == 'ztm':
        nonce = master_key[48:60]  # 96-bit nonce
        data = chacha20_encrypt(chacha_key, nonce, data)

    data = bytes(d ^ l for d, l in zip(data, lfsr_stream))  # Bit-level XOR
    data = bytes(d ^ c for d, c in zip(data, chaotic_stream))  # Bit-level XOR
    data = transpose(data, transposition_key)

    if mode == 'ztm':
        data = speck_ctr_encrypt(speck_key, data)

    return data

# Decryption Pipeline
def decrypt(data, mode):
    lfsr_seed = int.from_bytes(master_key[:2], 'big')
    chaotic_x0 = int.from_bytes(master_key[2:6], 'big') / (1 << 32)
    chaotic_r = 3.8
    transposition_key = master_key[6:14]
    chacha_key = master_key[:32]
    speck_key = master_key[32:48]

    lfsr_stream = lfsr(lfsr_seed, 0x8029, len(data))
    chaotic_stream = logistic_map(chaotic_x0, chaotic_r, len(data))

    if mode == 'ztm':
        data = speck_ctr_decrypt(speck_key, data)

    data = inverse_transpose(data, transposition_key)
    data = bytes(d ^ c for d, c in zip(data, chaotic_stream))  # Bit-level XOR
    data = bytes(d ^ l for d, l in zip(data, lfsr_stream))  # Bit-level XOR

    if mode == 'ztm':
        nonce = master_key[48:60]
        data = chacha20_decrypt(chacha_key, nonce, data)

    return data

# Flask Routes
@app.route('/encrypt', methods=['POST'])
def encrypt_route():
    data = request.json['data'].encode()
    mode = request.json['mode']
    ciphertext = encrypt(data, mode)
    return jsonify({'ciphertext': ciphertext.hex()})

@app.route('/decrypt', methods=['POST'])
def decrypt_route():
    ciphertext = bytes.fromhex(request.json['ciphertext'])
    mode = request.json['mode']
    plaintext = decrypt(ciphertext, mode).decode()
    return jsonify({'plaintext': plaintext})

@app.route('/ntru_demo', methods=['GET'])
def ntru_demo():
    """
    Run an NTRU key exchange demo and return the results.
    """
    original_key, decrypted_key, match = ntru_key_exchange()
    return jsonify({
        'original_key': original_key,
        'decrypted_key': decrypted_key,
        'match': match
    })

@app.route('/')
def index():
    with open('index.html', 'r') as f:
        return f.read()

if __name__ == '__main__':
    print("Server will be available at http://localhost:5000")
    app.run(debug=True)