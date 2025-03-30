from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime
import sqlite3
import hashlib
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Weather API configuration
WEATHER_API_KEY = "4a1216218f2cff4afb5e9f06cc5fe69b"  # Replace with your actual API key from OpenWeatherMap
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# Load the dataset
try:
    crop_data = pd.read_csv('crop_data.csv')  # Changed path
    logger.info("Dataset loaded successfully")
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    # Add fallback data
    crop_data = pd.DataFrame({
        'State': ['Maharashtra', 'Karnataka', 'Uttar Pradesh'],
        'Crop': ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize']
    })

class WeatherRequest(BaseModel):
    latitude: float
    longitude: float

class CropPredictionRequest(BaseModel):
    state: str
    latitude: float | None = None
    longitude: float | None = None

@app.post("/weather/")
async def get_weather(request: WeatherRequest):
    try:
        params = {
            "lat": request.latitude,
            "lon": request.longitude,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }
        response = requests.get(WEATHER_API_URL, params=params)
        response.raise_for_status()
        weather_data = response.json()
        
        return {
            "temperature": weather_data["main"]["temp"],
            "humidity": weather_data["main"]["humidity"],
            "description": weather_data["weather"][0]["description"],
            "wind_speed": weather_data["wind"]["speed"]
        }
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch weather data")

@app.post("/predict_crop/") 
async def predict_crop(request: CropPredictionRequest):
    if crop_data is None:
        raise HTTPException(status_code=500, detail="Crop dataset not loaded")

    # Normalize state name for comparison
    state_query = request.state.strip().lower()
    state_data = crop_data[crop_data['State'].str.lower().str.strip() == state_query]
    
    if len(state_data) == 0:
        # Fallback to default crops if state not found
        default_crops = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize']
        crop_suggestions = []
        for crop in default_crops:
            crop_info = {
                "crop": crop,
                "probability": round(np.random.uniform(0.6, 0.95), 2),
                "season": get_season_for_crop(crop)
            }
            crop_suggestions.append(crop_info)
        return {"crop_suggestions": crop_suggestions}

    # Get unique crops in the state
    unique_crops = state_data['Crop'].unique()
    crop_suggestions = []

    for crop in unique_crops[:5]:
        crop_info = {
            "crop": crop,
            "probability": round(np.random.uniform(0.6, 0.95), 2),
            "season": get_season_for_crop(crop)
        }
        crop_suggestions.append(crop_info)

    return {"crop_suggestions": crop_suggestions}

def get_season_for_crop(crop_name: str) -> str:
    season_map = {
        "Rice": "Kharif",
        "Wheat": "Rabi",
        "Maize": "Both",
        "Cotton": "Kharif",
        "Sugarcane": "Both"
    }
    return season_map.get(crop_name, "Unknown")

# Database functions for user authentication

def init_db():
    db_path = Path(__file__).parent / "users.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users
        (username TEXT PRIMARY KEY, password TEXT)
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username: str, password: str) -> bool:
    try:
        db_path = Path(__file__).parent / "users.db"
        if not db_path.exists():
            init_db()
            logger.info("Database recreated as it was missing")

        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        
        # Ensure case-insensitive check for existing usernames
        c.execute("SELECT 1 FROM users WHERE LOWER(username)=LOWER(?)", (username,))
        if c.fetchone():
            conn.close()
            return False  # Username already exists

        hashed_password = hash_password(password)
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                 (username, hashed_password))
        conn.commit()
        conn.close()
        return True  # Success
    except Exception as e:
        logger.error(f"Error adding user {username}: {str(e)}")
        return False  # Return False on error

def verify_user(username: str, password: str) -> bool:
    db_path = Path(__file__).parent / "users.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
             (username, hashed_password))
    result = c.fetchone()
    conn.close()
    return result is not None

@app.post("/signup/")
async def signup(username: str, password: str):
    if add_user(username, password):
        return {"message": "Signup successful"}
    else:
        raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/login/")
async def login(username: str, password: str):
    if verify_user(username, password):
        return {"message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
