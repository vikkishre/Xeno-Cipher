# hybrid_cipher.py - Implements the Fixed Hybrid Cipher (Vigenère + Polybius)
from vigenere import vigenere_encrypt, vigenere_decrypt
from polybius import polybius_encrypt, polybius_decrypt

def hybrid_encrypt(plaintext, key):
    vigenere_output = vigenere_encrypt(plaintext, key)
    print("DEBUG: Vigenère Output:", vigenere_output)  # Debugging step
    vigenere_output = vigenere_output.replace("J", "I")  # Ensure Polybius compatibility
    polybius_output = polybius_encrypt(vigenere_output)
    return polybius_output


def hybrid_decrypt(ciphertext, key):
    """Decrypts ciphertext by reversing Polybius Cipher first, then Vigenère Cipher."""
    polybius_reversed = polybius_decrypt(ciphertext)
    vigenere_reversed = vigenere_decrypt(polybius_reversed, key)
    return vigenere_reversed

# Example usage
if __name__ == "__main__":
    plaintext = "HELLO"
    key = "KEY"

    encrypted_text = hybrid_encrypt(plaintext, key)
    print("Hybrid Encrypted:", encrypted_text)

    decrypted_text = hybrid_decrypt(encrypted_text, key)
    print("Hybrid Decrypted:", decrypted_text)
