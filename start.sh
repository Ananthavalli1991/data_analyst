#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -o errexit

# Start the Gunicorn web server with your Flask application.
# The format is 'module_name:app_instance_name'.
# In your case, it's 'app:app' because your Flask app instance is named 'app'
# and the file is named 'app.py'.
echo "Starting Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT "app:app"
