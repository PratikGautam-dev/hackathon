# KrishiAI - Smart Farming Assistant

An AI-powered farming assistant that helps farmers make data-driven decisions about crop selection and management.

## Features
- Crop recommendation based on location and environmental factors
- Weather forecast integration
- Market price analysis
- Detailed crop analysis with visualizations
- Interactive map interface

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/krishiai.git
cd krishiai
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
Create a `.env` file in the project root:
```
OPENWEATHER_API_KEY=your_api_key_here
```

5. Start the application:
```bash
# Terminal 1: Start backend
cd backend
uvicorn main:app --reload

# Terminal 2: Start frontend
cd ..
streamlit run project-env/streamlit_app.py
```

## Environment Variables

The following environment variables are required:

- `OPENWEATHER_API_KEY`: Get from [OpenWeather](https://openweathermap.org/api)

## Development Setup

For development, you might want to install additional packages:

```bash
pip install -r requirements-dev.txt  # For development dependencies
```
