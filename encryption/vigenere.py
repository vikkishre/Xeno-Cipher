# vigenere.py - Implements the Vigenère Cipher

def generateKey(plaintext, key):
    """Generates a key that matches the length of the plaintext."""
    key = list(key)
    if len(plaintext) == len(key):
        return "".join(key)
    else:
        for i in range(len(plaintext) - len(key)):
            key.append(key[i % len(key)])
    return "".join(key)

def vigenere_encrypt(plaintext, key):
    """Encrypts the plaintext using Vigenère Cipher."""
    key = generateKey(plaintext, key)
    ciphertext = []
    
    for i in range(len(plaintext)):
        p = ord(plaintext[i]) - ord('A')  # Convert letter to number
        k = ord(key[i]) - ord('A')        # Convert key letter to number
        c = (p + k) % 26                  # Shift the letter
        ciphertext.append(chr(c + ord('A')))  # Convert back to letter
    
    return "".join(ciphertext)

def vigenere_decrypt(ciphertext, key):
    """Decrypts the ciphertext using Vigenère Cipher."""
    key = generateKey(ciphertext, key)
    plaintext = []
    
    for i in range(len(ciphertext)):
        c = ord(ciphertext[i]) - ord('A')  # Convert letter to number
        k = ord(key[i]) - ord('A')        # Convert key letter to number
        p = (c - k + 26) % 26             # Reverse the shift
        plaintext.append(chr(p + ord('A')))  # Convert back to letter
    
    return "".join(plaintext)

# Example usage
if __name__ == "__main__":
    plaintext = "HELLO"
    key = "HAIL"
    
    encrypted_text = vigenere_encrypt(plaintext, key)
    print("Encrypted:", encrypted_text)
    
    decrypted_text = vigenere_decrypt(encrypted_text, key)
    print("Decrypted:", decrypted_text)
