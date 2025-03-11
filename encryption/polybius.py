# polybius.py - Implements the Polybius Cipher

# Define the Polybius square mapping
polybius_square = {
    'A': '11', 'B': '12', 'C': '13', 'D': '14', 'E': '15',
    'F': '21', 'G': '22', 'H': '23', 'I': '24', 'J': '24', 'K': '25',
    'L': '31', 'M': '32', 'N': '33', 'O': '34', 'P': '35',
    'Q': '41', 'R': '42', 'S': '43', 'T': '44', 'U': '45',
    'V': '51', 'W': '52', 'X': '53', 'Y': '54', 'Z': '55'
}

# Reverse mapping for decryption
reverse_polybius_square = {v: k for k, v in polybius_square.items()}

def polybius_encrypt(plaintext):
    """Encrypts plaintext using the Polybius Cipher with consistent I/J mapping."""
    plaintext = plaintext.upper().replace("J", "I")  # Ensure 'J' is treated as 'I'
    encrypted_text = " ".join(polybius_square[char] for char in plaintext if char in polybius_square)
    return encrypted_text


def polybius_decrypt(ciphertext):
    """Decrypts Polybius Cipher with default '24' to 'I' (no J/I replacement)."""
    numbers = ciphertext.split()
    decrypted_text = "".join(reverse_polybius_square[num] if num != "24" else "I" for num in numbers)
    return decrypted_text




# Example usage
if __name__ == "__main__":
    plaintext = "HELLO"
    
    encrypted_text = polybius_encrypt(plaintext)
    print("Encrypted:", encrypted_text)
    
    decrypted_text = polybius_decrypt(encrypted_text)
    print("Decrypted:", decrypted_text)
