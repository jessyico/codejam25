#!/bin/bash

# Motion Tracking Server Startup Script

echo "🚀 Starting Motion Tracking Integration"
echo ""

# Check if in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  No virtual environment detected"
    echo "Looking for .venv..."
    
    if [ -d ".venv" ]; then
        echo "✅ Found .venv - activating..."
        source .venv/bin/activate
    else
        echo "❌ No .venv found. Please create one first:"
        echo "   python3 -m venv .venv"
        echo "   source .venv/bin/activate"
        exit 1
    fi
fi

echo ""
echo "📦 Installing backend dependencies..."
pip install -q -r requirements.txt

echo ""
echo "🎥 Starting Flask motion tracking server..."
echo "   Server will run on http://localhost:5000"
echo ""
echo "📝 Quick start:"
echo "   1. Keep this terminal running (Flask backend)"
echo "   2. Open another terminal and run: cd codejam_frontend && npm start"
echo "   3. Open http://localhost:3000 in your browser"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""

cd backend
python app.py
