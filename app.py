from flask import Flask, jsonify, render_template_string
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>XenoCipher</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
            }
            .card {
                border: 1px solid #ddd;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 4px;
            }
            .success {
                color: green;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>XenoCipher</h1>
            <div class="card">
                <h2>Status: <span class="success">Running</span></h2>
                <p>This is a simplified version of XenoCipher running on Railway.</p>
                <p>The full application functionality will be integrated soon.</p>
            </div>
            <div class="card">
                <h3>API Health Check</h3>
                <p>You can check the API health at <a href="/health">/health</a></p>
            </div>
        </div>
    </body>
    </html>
    """)

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "service": "XenoCipher API"
    })

@app.route('/info')
def info():
    return jsonify({
        "application": "XenoCipher",
        "deployment": "Railway",
        "status": "Simplified version active"
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port) 