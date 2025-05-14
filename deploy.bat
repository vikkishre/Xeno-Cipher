@echo off
echo XenoCipher Deployment Script
echo ===========================

echo Installing requirements...
cd Test
pip install -r requirements.txt
cd ..

echo.
echo Choose deployment method:
echo 1. Run with Flask (development server)
echo 2. Run with Waitress (production server)
echo 3. Deploy with Docker (requires Docker)
echo 4. Vercel deployment instructions
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Starting Flask development server...
    cd Test
    python app.py
) else if "%choice%"=="2" (
    echo Installing Waitress if not already installed...
    pip install waitress
    echo Starting production server with Waitress...
    cd Test
    python serve.py
) else if "%choice%"=="3" (
    echo Building and running Docker container...
    docker build -t xenocipher . 2>nul
    if %errorlevel% neq 0 (
        echo Docker not found or error building image.
        echo Please make sure Docker is installed and running.
        goto end
    )
    echo Running Docker container...
    docker run -p 5000:5000 xenocipher
    echo Docker container is running at http://localhost:5000
) else if "%choice%"=="4" (
    echo.
    echo For Vercel deployment, follow these steps manually:
    echo.
    echo 1. Install Node.js from https://nodejs.org/
    echo 2. Install Vercel CLI: npm install -g vercel
    echo 3. Run: vercel login
    echo 4. Run: vercel
    echo 5. For production: vercel --prod
    echo.
    echo The project has already been configured with vercel.json and api/index.py.
) else (
    echo Invalid choice. Please run the script again and select a valid option.
)

:end
pause 
 