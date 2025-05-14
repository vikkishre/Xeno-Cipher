# XenoCipher

XenoCipher is an advanced encryption application that implements several cryptographic algorithms including NTRU, LFSR, ChaCha20, and more.

## Quick Start

### Windows

1. Make sure you have Python 3.7+ installed
2. Clone or download this repository
3. Run the deployment script:
   ```
   deploy.bat
   ```
4. Choose your deployment method when prompted

### Manual Installation

1. Navigate to the Test directory:
   ```
   cd Test
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and go to: http://localhost:5000

## Production Deployment

For production deployment, we recommend using Waitress (on Windows) or Gunicorn (on Linux/macOS).

### Windows Production Deployment

1. Install Waitress:
   ```
   pip install waitress
   ```

2. Run with Waitress:
   ```
   cd Test
   python serve.py
   ```

3. Access the application at http://localhost:8000

### Docker Deployment

If you have Docker installed:

1. Build the Docker image:
   ```
   docker build -t xenocipher .
   ```

2. Run the container:
   ```
   docker run -p 5000:5000 xenocipher
   ```

3. Access the application at http://localhost:5000

### Vercel Deployment

For cloud deployment using Vercel:

1. Install Vercel CLI:
   ```
   npm install -g vercel
   ```

2. Deploy the application:
   ```
   vercel login
   vercel
   ```

3. For production deployment:
   ```
   vercel --prod
   ```

4. Access your application at the URL provided by Vercel

## Features

- Hybrid encryption combining multiple algorithms
- NTRU post-quantum key exchange
- LFSR and Chaotic Map for stream cipher
- Transposition for bit-level permutation
- ChaCha20 stream cipher
- Speck lightweight block cipher
- Web interface for encryption/decryption
- Attack simulation

## Security Considerations

This application is for educational and demonstration purposes. In a production environment, consider:

1. Implementing proper key management
2. Using TLS/SSL for secure communication
3. Regular security audits
4. Following industry standard security practices

## License

[MIT License](LICENSE)

## More Information

For more detailed deployment options, see the [deployment_guide.md](deployment_guide.md). 