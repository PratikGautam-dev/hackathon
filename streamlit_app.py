import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables (for API keys)
load_dotenv()

# ----------------------
# App Configuration
# ----------------------
st.set_page_config(page_title="KrishiAI", layout="wide")

# ----------------------
# Sidebar
# ----------------------
with st.sidebar:
    st.title("🌾 KrishiAI Settings")
    farmer_name = st.text_input("Farmer Name")
    selected_crop = st.selectbox("Crop Type", ["Rice", "Wheat", "Cotton"])
    selected_language = st.radio("Language", ["English", "Hindi", "Tamil"])

# ----------------------
# Main Interface
# ----------------------
st.title("🌱 KrishiAI - Smart Farming Assistant")

# 1. Map Section
st.header("Field Map")
col1, col2 = st.columns([2, 1])

with col1:
    # Example coordinates for Punjab
    df = pd.DataFrame({
        'lat': [30.7333, 30.7350],
        'lon': [76.7794, 76.7810]
    })
    st.map(df, zoom=13, use_container_width=True)

with col2:
    st.metric("Current NDVI", "0.62", delta="-0.12 (Last 7 days)")
    st.progress(0.65)
    st.write("*Recommendation*: Irrigate northwest section tomorrow")

# 2. AI Recommendations
st.header("AI Recommendations")
tab1, tab2, tab3 = st.tabs(["🚰 Irrigation", "🌱 Fertilizer", "🐛 Pest Control"])

with tab1:
    st.subheader("Water Management")
    st.write("""
    - *Next Irrigation*: 2023-10-05 6:00 AM
    - *Duration*: 45 minutes
    - *Priority Zones*: 
        - Northwest (Dry Soil)
    """)

with tab2:
    st.subheader("Nutrient Management")
    st.image("https://via.placeholder.com/400x200?text=Soil+Analysis+Chart", width=400)

with tab3:
    st.subheader("Pest Alerts")
    st.error("High risk of Cotton Bollworm detected! Spray neem oil immediately.")

# 3. WhatsApp Integration
if st.button("📱 Send Alert to Farmer"):
    # Add Twilio integration here
    st.success("Alert sent via WhatsApp!")
