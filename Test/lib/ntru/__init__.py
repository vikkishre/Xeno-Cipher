import numpy as np
import random
import os
from sympy import Poly, ZZ
from sympy.abc import x

def generate_keys(N, p, q):
    # Simple implementation to generate NTRU keys
    # Creates a public key and private key pair
    f_coeffs = [random.choice([-1, 0, 1]) for _ in range(N)]
    g_coeffs = [random.choice([-1, 0, 1]) for _ in range(N)]
    
    # Convert to polynomials
    f_poly = Poly(f_coeffs[::-1], x).set_domain(ZZ)
    g_poly = Poly(g_coeffs[::-1], x).set_domain(ZZ)
    
    # Public key is simplified for demo purposes
    h_coeffs = [(f * g) % q for f, g in zip(f_coeffs, g_coeffs)]
    
    # For simplicity, we're just returning the coefficient arrays
    return h_coeffs, f_coeffs

def encrypt_message(pub_key, message):
    # Simple encryption using the public key
    # For demo purposes, just XOR with the key
    return [m ^ pub_key[i % len(pub_key)] for i, m in enumerate(message)]

def decrypt_message(priv_key, ciphertext):
    # Simple decryption using the private key
    # For demo purposes, just XOR with the key
    return [c ^ priv_key[i % len(priv_key)] for i, c in enumerate(ciphertext)]
