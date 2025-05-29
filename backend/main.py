from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, validator
from sqlalchemy import create_engine, Column, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import bcrypt
import uuid
import requests
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import aiohttp
from aiohttp import ClientTimeout
from crop_advisor import get_crop_schedule

# First, drop the existing database file
if os.path.exists("krishiai.db"):
    os.remove("krishiai.db")

# Database setup
DATABASE_URL = "sqlite:///./krishiai.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    password = Column(String, nullable=False)

# Farmer Data model - Updated schema
class Farmer(Base):
    __tablename__ = "farmers"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    state = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

# Create all tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI()

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    message: str
    token: str

class FarmerCreate(BaseModel):
    name: str
    state: str
    latitude: float
    longitude: float

    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v

    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

class CropPredictionRequest(BaseModel):
    state: str
    nitrogen: float = 0.0
    phosphorus: float = 0.0
    potassium: float = 0.0
    temperature: float = 25.0
    humidity: float = 50.0
    ph_value: float = 7.0
    rainfall: float = 100.0
    latitude: float
    longitude: float

    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v

    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

    @validator('state')
    def validate_state(cls, v):
        if not v.strip():
            raise ValueError('State cannot be empty')
        return v.strip()

    @validator('nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph_value', 'rainfall')
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError('Value cannot be negative')
        return v

class CropPredictionResponse(BaseModel):
    crop_suggestions: list

class DetailedAnalysisRequest(BaseModel):
    crop_name: str
    state: str
    latitude: float
    longitude: float

class WeatherForecast(BaseModel):
    date: str
    temperature: float
    humidity: float
    rainfall: float

class MarketTrend(BaseModel):
    date: str
    price: float
    volume: float

class DetailedAnalysisResponse(BaseModel):
    weather_forecast: list[WeatherForecast]
    market_trends: list[MarketTrend]
    weather_graph: dict
    price_graph: dict

class CropScheduleRequest(BaseModel):
    crop_name: str
    state: str
    season: str

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Signup route
@app.post("/signup/", response_model=LoginResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    if not user.username or not user.password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed_password = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    new_user = User(username=user.username, password=hashed_password)
    db.add(new_user)
    db.commit()
    return {"message": "User registered successfully", "token": "dummy_token"}

# Login route
@app.post("/login/", response_model=LoginResponse)
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not bcrypt.checkpw(user.password.encode("utf-8"), db_user.password.encode("utf-8")):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"message": "Login successful", "token": "dummy_token"}

# Save farmer data route
@app.post("/submit_farm_data/")
async def submit_farm_data(farmer: FarmerCreate, db: Session = Depends(get_db)):
    try:
        print(f"Received farmer data: {farmer.dict()}")  # Debug print
        
        farmer_id = str(uuid.uuid4())
        new_farmer = Farmer(
            id=farmer_id,
            name=farmer.name,
            state=farmer.state,
            latitude=farmer.latitude,
            longitude=farmer.longitude
        )
        db.add(new_farmer)
        db.commit()
        
        return {"message": "Farmer data saved successfully", "id": farmer_id}
    except Exception as e:
        print(f"Error saving farmer data: {str(e)}")  # Debug print
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save farmer data: {str(e)}")

# Fetch state-wise crop prices
def fetch_crop_prices(state):
    try:
        response = requests.get(f"https://api.realexample.com/crop-prices?state={state}", timeout=5, verify=False)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        print("Error fetching crop prices:", e)
    return {}

# Load and prepare dataset
def prepare_model():
    try:
        # Read dataset
        df = pd.read_csv('crop_data.csv')
        print("Loading dataset...")
        
        if df.empty:
            raise ValueError("Dataset is empty")
            
        # Convert column names to match expected format
        df.columns = df.columns.str.strip()
        
        # Create dummy data if missing columns
        if 'Season' not in df.columns:
            df['Season'] = 'All Season'
        if 'Production' not in df.columns:
            df['Production'] = 1000
            
        return None, None, None, df  # We'll use direct DataFrame operations instead of ML model
    except Exception as e:
        print(f"Error in prepare_model: {str(e)}")
        return None, None, None, None

model, state_encoder, crop_encoder, crop_data = prepare_model()

def get_current_season():
    # Define seasons based on months
    current_month = datetime.now().month
    if 3 <= current_month <= 5:
        return "Summer"
    elif 6 <= current_month <= 9:
        return "Monsoon"
    elif 10 <= current_month <= 11:
        return "Post-Monsoon"
    else:
        return "Winter"

def get_season_for_crop(crop_name):
    # Define season rules for different crops
    crop_seasons = {
        'Rice': 'Kharif',
        'Maize': 'Kharif',
        'ChickPea': 'Rabi',
        'KidneyBeans': 'Kharif',
        'PigeonPeas': 'Kharif',
        'MothBeans': 'Kharif',
        'MungBean': 'Kharif',
        'Blackgram': 'Kharif',
        'Lentil': 'Rabi',
        'Pomegranate': 'Perennial',
        'Banana': 'Perennial',
        'Mango': 'Perennial',
        'Grapes': 'Perennial',
        'Watermelon': 'Summer',
        'Muskmelon': 'Summer',
        'Apple': 'Perennial',
        'Orange': 'Perennial',
        'Papaya': 'Perennial',
        'Coconut': 'Perennial',
        'Cotton': 'Kharif',
        'Jute': 'Kharif',
        'Wheat': 'Rabi',
        'Sugarcane': 'Perennial',
        'Groundnut': 'Kharif',
        'Soybean': 'Kharif',
        'Tomato': 'All Season',
        'Potato': 'Rabi',
        'Onion': 'Rabi',
        'Garlic': 'Rabi',
        'Turmeric': 'Kharif'
    }
    return crop_seasons.get(crop_name, 'Unknown')

# Load environment variables
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AGRICULTURE_PRICE_API = "your-api-key"  # Get from data.gov.in or similar source

async def get_weather_data(lat: float, lon: float):
    try:
        if not OPENWEATHER_API_KEY:
            # Return dummy data if no API key
            return {
                "temperature": 25.0,
                "humidity": 65.0,
                "rainfall": 50.0
            }
            
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"],
                        "rainfall": data.get("rain", {}).get("1h", 0) * 24  # Convert to daily
                    }
                else:
                    print(f"Weather API error: {await response.text()}")
                    return None
    except Exception as e:
        print(f"Weather API error: {str(e)}")
        return None

async def get_weather_forecast(lat: float, lon: float, days: int = 180):  # Changed to 180 days
    try:
        if not OPENWEATHER_API_KEY:
            # Return dummy forecast for 6 months
            forecast = []
            base_temp = 25.0
            base_humidity = 65.0
            base_rainfall = 50.0
            for i in range(days):
                date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                # More realistic seasonal variations
                month = (datetime.now() + timedelta(days=i)).month
                # Seasonal temperature adjustments
                seasonal_temp = base_temp + 5 * np.sin(2 * np.pi * (i/365))  # Yearly cycle
                temp = seasonal_temp + np.random.normal(0, 2)
                
                # Seasonal humidity and rainfall
                if 6 <= month <= 9:  # Monsoon
                    humidity = base_humidity + 20 + np.random.normal(0, 5)
                    rainfall = max(0, base_rainfall * 2 + np.random.normal(0, 30))
                elif 10 <= month <= 11:  # Post-monsoon
                    humidity = base_humidity + 10 + np.random.normal(0, 5)
                    rainfall = max(0, base_rainfall + np.random.normal(0, 20))
                elif month <= 2:  # Winter
                    humidity = base_humidity - 10 + np.random.normal(0, 5)
                    rainfall = max(0, base_rainfall * 0.3 + np.random.normal(0, 10))
                else:  # Summer
                    humidity = base_humidity - 20 + np.random.normal(0, 5)
                    rainfall = max(0, base_rainfall * 0.1 + np.random.normal(0, 5))
                
                forecast.append(WeatherForecast(
                    date=date,
                    temperature=round(temp, 1),
                    humidity=round(humidity, 1),
                    rainfall=round(rainfall, 1)
                ))
            return forecast

        # For the actual API, we'll continue with 7-day forecast as OpenWeather free tier limitation
        timeout = ClientTimeout(total=10)
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    forecast = []
                    for item in data['list'][:days*8:8]:  # Get daily data
                        temp = item['main']['temp']
                        humidity = item['main']['humidity']
                        rainfall = item.get('rain', {}).get('3h', 0) * 8
                        
                        forecast.append(WeatherForecast(
                            date=datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d'),
                            temperature=round(temp, 1),
                            humidity=round(humidity, 1),
                            rainfall=round(rainfall, 1)
                        ))
                    return forecast
                else:
                    print(f"Weather forecast API error: {await response.text()}")
                    return None
    except Exception as e:
        print(f"Weather forecast API error: {str(e)}")
        return None

async def get_crop_price(crop_name: str):
    try:
        crop_prices = {
            "Rice": {"min": 18.5, "max": 22.0, "trend": "stable"},
            "Wheat": {"min": 20.0, "max": 24.0, "trend": "rising"},
            "Maize": {"min": 15.0, "max": 18.0, "trend": "falling"},
            "Cotton": {"min": 45.0, "max": 55.0, "trend": "stable"},
            "Sugarcane": {"min": 3.0, "max": 4.0, "trend": "stable"},
            "Potato": {"min": 12.0, "max": 15.0, "trend": "rising"},
            "Onion": {"min": 15.0, "max": 25.0, "trend": "volatile"},
            "Tomato": {"min": 20.0, "max": 35.0, "trend": "volatile"},
            "Groundnut": {"min": 40.0, "max": 50.0, "trend": "stable"},
            "Soybean": {"min": 35.0, "max": 42.0, "trend": "rising"},
            "ChickPea": {"min": 45.0, "max": 55.0, "trend": "stable"},
            "Turmeric": {"min": 65.0, "max": 85.0, "trend": "rising"},
            "Garlic": {"min": 40.0, "max": 60.0, "trend": "stable"}
        }
        return crop_prices.get(crop_name, {"min": 0, "max": 0, "trend": "unknown"})
    except Exception as e:
        print(f"Price API error: {str(e)}")
        return None

async def get_market_history(crop_name: str, days: int = 30):
    try:
        # Simulate historical data - replace with actual API call
        base_price = get_crop_price(crop_name)['min']
        history = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            # Simulate some price variation
            price = base_price * (1 + np.random.normal(0, 0.1))
            volume = np.random.randint(1000, 5000)
            history.append(MarketTrend(
                date=date,
                price=round(price, 2),
                volume=volume
            ))
        return history
    except Exception as e:
        print(f"Market history error: {str(e)}")
        return None

@app.post("/predict_crop/", response_model=CropPredictionResponse)
async def predict_crop(data: CropPredictionRequest):
    try:
        if crop_data is None:
            raise HTTPException(status_code=500, detail="Dataset not loaded")
        
        # Get weather data for location
        weather = await get_weather_data(data.latitude, data.longitude)
        if weather:
            current_temp = weather["temperature"]
            current_humidity = weather["humidity"]
            current_rainfall = weather["rainfall"]
        else:
            # Use default values from request if weather API fails
            current_temp = data.temperature
            current_humidity = data.humidity
            current_rainfall = data.rainfall

        # Get state-specific data
        state_crops = crop_data[crop_data['State'].str.lower() == data.state.lower()]
        if state_crops.empty:
            raise HTTPException(status_code=400, detail=f"No data available for state: {data.state}")
        
        # Calculate crop suitability
        crop_scores = []
        for crop in state_crops['Crop'].unique():
            crop_data_subset = state_crops[state_crops['Crop'] == crop]
            
            # Get current market price
            price_data = await get_crop_price(crop)
            
            params = crop_data_subset.iloc[0]
            similarity_score = 100 - float(
                abs(params['Temperature'] - current_temp) * 2 +  # Give more weight to temperature
                abs(params['Humidity'] - current_humidity) +
                abs(params['Rainfall'] - current_rainfall) * 1.5  # Give more weight to rainfall
            ) / 4.5  # Adjusted denominator for weighted scores
            
            if similarity_score > 50:  # Only include viable crops
                crop_scores.append({
                    "crop": str(crop),
                    "probability": f"{similarity_score:.1f}%",
                    "price": {
                        "min": price_data["min"],
                        "max": price_data["max"],
                        "trend": price_data["trend"]
                    },
                    "season": get_season_for_crop(crop),
                    "current_weather": {
                        "temperature": current_temp,
                        "humidity": current_humidity,
                        "rainfall": current_rainfall
                    },
                    "details": {
                        "soil_parameters": {
                            "N": float(params['Nitrogen']),
                            "P": float(params['Phosphorus']),
                            "K": float(params['Potassium']),
                            "pH": float(params['pH_Value'])
                        }
                    }
                })
        
        # Sort by probability
        crop_scores.sort(key=lambda x: float(x["probability"].strip('%')), reverse=True)
        return {"crop_suggestions": crop_scores[:5]}
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detailed_analysis/", response_model=DetailedAnalysisResponse)
async def get_detailed_analysis(data: DetailedAnalysisRequest):
    try:
        # Get weather forecast with error handling
        forecast = await get_weather_forecast(data.latitude, data.longitude)
        if not forecast:
            forecast = []  # Use empty list instead of failing
            print("Using empty forecast due to API failure")

        # Get market trends with error handling
        market_history = await get_market_history(data.crop_name)
        if not market_history:
            market_history = []
            print("Using empty market history due to API failure")

        # Create weather forecast graph with improved styling
        weather_dates = [f.date for f in forecast]
        temps = [f.temperature for f in forecast]
        humidity = [f.humidity for f in forecast]
        rainfall = [f.rainfall for f in forecast]

        weather_fig = go.Figure()
        
        # Temperature line with monthly average trendline
        monthly_temps = []
        monthly_dates = []
        for i in range(0, len(temps), 30):
            monthly_temps.append(np.mean(temps[i:i+30]))
            monthly_dates.append(weather_dates[i])

        weather_fig.add_trace(go.Scatter(
            x=weather_dates,
            y=temps,
            name="Daily Temperature",
            line=dict(color="#FF9900", width=1),
            hovertemplate="Temp: %{y}°C<br>Date: %{x}"
        ))
        
        weather_fig.add_trace(go.Scatter(
            x=monthly_dates,
            y=monthly_temps,
            name="Monthly Average",
            line=dict(color="#FF0000", width=2, dash='dash'),
            hovertemplate="Avg Temp: %{y:.1f}°C<br>Month: %{x}"
        ))
        
        # Humidity line
        weather_fig.add_trace(go.Scatter(
            x=weather_dates,
            y=humidity,
            name="Humidity",
            line=dict(color="#00CC96", width=2),
            hovertemplate="Humidity: %{y}%<br>Date: %{x}"
        ))
        
        # Rainfall bars
        weather_fig.add_trace(go.Bar(
            x=weather_dates,
            y=rainfall,
            name="Rainfall",
            marker_color="#636EFA",
            hovertemplate="Rainfall: %{y}mm<br>Date: %{x}"
        ))
        
        # Update layout for 6-month view
        weather_fig.update_layout(
            title="6-Month Weather Forecast",
            xaxis_title="Date",
            yaxis_title="Temperature (°C) / Humidity (%) / Rainfall (mm)",
            hovermode='x unified',
            showlegend=True,
            template="plotly_white",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(step="all", label="6m")
                    ])
                )
            )
        )

        # Create market trends graph with improved styling
        market_dates = [m.date for m in market_history]
        prices = [m.price for m in market_history]
        volumes = [m.volume for m in market_history]

        price_fig = go.Figure()
        
        # Price line
        price_fig.add_trace(go.Scatter(
            x=market_dates,
            y=prices,
            name="Price",
            line=dict(color="#FF9900", width=2),
            hovertemplate="Price: ₹%{y:.2f}/kg<br>Date: %{x}"
        ))
        
        # Volume bars
        price_fig.add_trace(go.Bar(
            x=market_dates,
            y=volumes,
            name="Volume",
            marker_color="#636EFA",
            yaxis="y2",
            hovertemplate="Volume: %{y:,} kg<br>Date: %{x}"
        ))
        
        price_fig.update_layout(
            title=f"{data.crop_name} Price Trends",
            xaxis_title="Date",
            yaxis_title="Price (₹/kg)",
            yaxis2=dict(
                title="Volume (kg)",
                overlaying="y",
                side="right"
            ),
            hovermode='x unified',
            showlegend=True,
            template="plotly_white"
        )

        return DetailedAnalysisResponse(
            weather_forecast=forecast,
            market_trends=market_history,
            weather_graph=weather_fig.to_dict(),
            price_graph=price_fig.to_dict()
        )

    except Exception as e:
        print(f"Detailed analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate detailed analysis: {str(e)}"
        )

@app.post("/get_crop_schedule/")
async def get_schedule(data: CropScheduleRequest):
    schedule = get_crop_schedule(data.crop_name, data.state, data.season)
    if schedule:
        return {"schedule": schedule}
    raise HTTPException(status_code=500, detail="Failed to generate crop schedule")

def get_season_score(params, current_season):
    # Define season compatibility for different temperature and rainfall ranges
    if current_season == "Summer":
        if 25 <= params['Temperature'] <= 35 and params['Rainfall'] <= 100:
            return 1.0
        return 0.5
    elif current_season == "Monsoon":
        if 20 <= params['Temperature'] <= 30 and params['Rainfall'] >= 100:
            return 1.0
        return 0.5
    elif current_season == "Winter":
        if 15 <= params['Temperature'] <= 25 and params['Rainfall'] <= 50:
            return 1.0
        return 0.5
    else:  # Post-Monsoon
        if 20 <= params['Temperature'] <= 30 and 50 <= params['Rainfall'] <= 100:
            return 1.0
        return 0.5
