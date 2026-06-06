#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")/.."

# Activate the virtual environment
source venv/bin/activate

# Start the Daphne server
daphne -b 0.0.0.0 -p 8000 ppe.asgi:application