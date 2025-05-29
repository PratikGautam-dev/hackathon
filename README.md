# KrishiAI - Smart Farming Assistant

An AI-powered farming assistant that helps farmers make data-driven decisions about crop selection and management.

## Features
- Crop recommendation based on location and environmental factors
- Weather forecast integration
- Market price analysis
- Detailed crop analysis with visualizations
- Interactive map interface

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create .env file with required API keys:
```
OPENWEATHER_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

3. Start the application:
- Double click `start_servers.bat` 
OR
- Run these commands in separate terminals:
```bash
# Terminal 1 - Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
streamlit run streamlit_app.py
```

4. Access the application:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000/docs

## Environment Variables

The following environment variables are required:

- `OPENWEATHER_API_KEY`: Get from [OpenWeather](https://openweathermap.org/api)

## Development Setup

For development, you might want to install additional packages:

```bash
pip install -r requirements-dev.txt  # For development dependencies
```
