# Alternative Deployment Methods

If you continue to face issues with Railway, here are two alternative deployment options that might work better with your project structure.

## Option 1: Render

Render has excellent support for Flask applications and handles directory structures well.

### Steps for Render Deployment:

1. **Sign up for Render**
   - Go to [render.com](https://render.com/)
   - Create an account or sign in

2. **Create a render.yaml file**
   ```yaml
   services:
     - type: web
       name: xenocipher
       env: python
       plan: free
       buildCommand: pip install -r Test/requirements.txt && pip install gunicorn
       startCommand: cd Test && gunicorn app:app
       envVars:
         - key: PYTHON_VERSION
           value: 3.9.16
   ```

3. **Push to GitHub**
   - Push your code to a GitHub repository

4. **Deploy on Render**
   - In Render dashboard, click "New" → "Blueprint"
   - Connect to your GitHub repository
   - Render will detect the render.yaml file and set up your service

## Option 2: PythonAnywhere

PythonAnywhere is specifically designed for Python web applications and has excellent support for custom directory structures.

### Steps for PythonAnywhere Deployment:

1. **Sign up for PythonAnywhere**
   - Go to [pythonanywhere.com](https://www.pythonanywhere.com/)
   - Create a free account

2. **Upload Your Code**
   - Use the Files tab to upload a ZIP of your project
   - Or clone from GitHub: `git clone https://github.com/yourusername/XenoCipher.git`

3. **Set Up a Web App**
   - Go to the Web tab
   - Click "Add a new web app"
   - Choose "Flask" and Python 3.9
   - Set the path to your Flask app: `/home/yourusername/XenoCipher/Test/app.py`

4. **Configure WSGI File**
   - Edit the WSGI configuration file
   - Update the path to point to your app in the Test directory
   ```python
   import sys
   path = '/home/yourusername/XenoCipher'
   if path not in sys.path:
       sys.path.append(path)
   
   from Test.app import app as application
   ```

5. **Install Requirements**
   - Go to the Consoles tab
   - Start a new Bash console
   - Run: `pip install -r Test/requirements.txt`

6. **Reload Your Web App**
   - Go back to the Web tab
   - Click the Reload button for your web app

## Option 3: Fly.io

Fly.io also has good support for Flask applications with custom directory structures.

### Steps for Fly.io Deployment:

1. **Install Fly CLI**
   - Install from [fly.io/docs/hands-on/install-flyctl/](https://fly.io/docs/hands-on/install-flyctl/)

2. **Create a Dockerfile**
   ```Dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY . .
   
   RUN pip install -r Test/requirements.txt gunicorn
   
   CMD cd Test && gunicorn app:app
   ```

3. **Create a fly.toml file**
   ```toml
   app = "xenocipher"
   
   [build]
   
   [http_service]
     internal_port = 8080
     force_https = true
   
   [[services.ports]]
     handlers = ["http"]
     port = "80"
   
   [[services.ports]]
     handlers = ["tls", "http"]
     port = "443"
   ```

4. **Deploy to Fly.io**
   ```bash
   fly auth login
   fly launch
   ```

Choose the option that best fits your needs. All three of these platforms generally have better support for custom directory structures compared to Railway. 