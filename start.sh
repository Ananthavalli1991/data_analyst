#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -o errexit

# A helpful function for logging messages
log() {
  echo "[start.sh] $1"
}

# 1. Log the starting process
log "Starting the Python web service..."

# 2. Check if a virtual environment is available, and activate it
#    Render automatically creates a virtual environment, so this is good practice
if [ -d "venv" ]; then
  log "Activating virtual environment..."
  source venv/bin/activate
fi

# 3. Ensure all Python dependencies are installed
#    This is the build command, but running it here again is a failsafe
log "Installing dependencies..."
pip install -r requirements.txt

# 4. Run your application using the waitress server
log "Starting the Waitress server on port $PORT..."
python app.py
