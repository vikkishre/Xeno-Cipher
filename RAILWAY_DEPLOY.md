# XenoCipher Railway Deployment Guide

## The Problem: Directory Structure

The original error `Error: can't chdir to 'Test'` indicates that Railway can't find the Test directory. This is because Railway has a specific way of handling file uploads and directory structures.

## Solution: Simplified App Structure

To solve this issue, I've created a simplified version of the app at the root level:

1. `app.py` - A simplified version of your Flask application
2. `requirements.txt` - Basic requirements with just Flask and Gunicorn
3. `Procfile` - Simple configuration using the root-level app.py
4. `railway.json` - Updated configuration for Railway

## How This Works

1. This simplified app serves as a "proxy" or "placeholder"
2. It confirms that Railway deployment works correctly
3. Once this is working, you can gradually migrate functionality from your Test directory

## Next Steps After Successful Deployment

1. Once the basic app is working, you can:
   - Gradually move your actual functionality to app.py
   - Or modify app.py to properly import your Test directory modules
   - Add more dependencies to requirements.txt as needed

2. To move toward your full application:
   ```python
   # In app.py after successful basic deployment
   import sys
   import os
   
   # Add paths
   sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Test'))
   
   # Import your components
   from Test.your_module import your_function
   ```

## Troubleshooting

If you continue to encounter issues:

1. Check Railway logs for specific error details
2. Verify that your requirements.txt has all necessary dependencies
3. Consider using Railway's environment variables for configuration
4. Review Railway's documentation for Python apps: https://docs.railway.app/guides/python

## Alternative Approach

If you prefer to maintain your existing structure:

1. Ensure your Test directory is properly uploaded to Railway
2. Try using a different service like Render or PythonAnywhere that has better support for custom directory structures