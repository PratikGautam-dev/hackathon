from flask import Blueprint, jsonify
from database import get_random_prices
from weather import get_weather

api = Blueprint('api', __name__)

@api.route('/prices', methods=['GET'])
def get_prices():
    prices = get_random_prices()
    return jsonify(prices)

@api.route('/weather/<float:lat>/<float:lon>', methods=['GET'])
def get_weather_data(lat, lon):
    weather = get_weather(lat, lon)
    return jsonify(weather)
