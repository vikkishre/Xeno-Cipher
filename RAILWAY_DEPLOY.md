 # Railway Deployment Guide for XenoCipher

This guide provides detailed instructions for deploying XenoCipher on Railway.

## Fixing "cd command not found" Error

If you encounter the error `The executable 'cd' could not be found`, try these solutions:

### Solution 1: Use gunicorn with --chdir flag

The current configuration in Procfile and railway.toml uses this approach:
```
web: gunicorn --chdir Test app:app
```

This tells gunicorn to change to the Test directory before running the app.

### Solution 2: Use main.py

If Solution 1 doesn't work, uncomment the alternative startCommand in railway.toml:
```
startCommand = "python main.py"
```

This uses the main.py file we've created, which imports from the Test directory.

### Solution 3: Use run.sh with proper permissions

If you're still having issues, try:

1. Make sure run.sh is executable:
   ```
   git update-index --chmod=+x run.sh
   git commit -m "Make run.sh executable"
   git push
   ```

2. Update railway.toml to use:
   ```
   startCommand = "./run.sh"
   ```

## Troubleshooting the Deployment

### Check Logs

Railway provides detailed logs for each deployment:
1. Go to your Railway dashboard
2. Click on your XenoCipher project
3. Select the "Deployments" tab
4. Click on the latest deployment
5. View logs to see what's happening

### Try Different Python Versions

If you're experiencing compatibility issues, you can try a different Python version:

Add to railway.toml:
```
[variables]
PYTHON_VERSION = "3.9"
```

### Ensure All Dependencies are Listed

Make sure all required packages are listed in Test/requirements.txt.

## Deployment Files

Your project should include these files for Railway deployment:

1. `Procfile` - Tells Railway how to run your application
2. `railway.toml` or `railway.json` - Configuration for Railway
3. `main.py` - Alternative entry point 
4. `run.sh` - Shell script as another alternative

## Alternative Platforms

If you continue to face issues with Railway, consider these alternatives:

1. **Render** - Similar to Railway with excellent Flask support
2. **PythonAnywhere** - Specifically designed for Python web apps
3. **Fly.io** - Good for Flask applications with a simple setup

## Getting Help

If you continue to face issues, you can:

1. Check Railway's documentation: https://docs.railway.app/
2. Join Railway's Discord community for support
3. Open an issue on Railway's GitHub repository

Remember to commit and push all configuration files to your repository before deploying to Railway.