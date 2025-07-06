"""
This file uses the trained model to make solar radiation predictions
based on current weather conditions.
"""

import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import config

class SolarPredictor:
    """
    A class to make solar radiation predictions
    """
    
    def __init__(self):
        """Initialize the predictor by loading the trained model"""
        try:
            self.model = joblib.load('models/solar_prediction_model.pkl')
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Error: Model not found. Please run model_trainer.py first.")
            self.model = None
    
    def predict_from_weather_data(self, weather_data):
        """
        Make a prediction based on weather data
        
        Args:
            weather_data: Dictionary with weather information
        
        Returns:
            Predicted solar radiation value
        """
        if self.model is None:
            return None
        
        # Prepare the data for prediction
        current_time = datetime.now()
        
        # Create feature array in the same order as training
        features = [
            weather_data.get('temperature', 20),
            weather_data.get('humidity', 50),
            weather_data.get('pressure', 1013),
            weather_data.get('wind_speed', 3),
            weather_data.get('cloud_cover', 30),
            weather_data.get('visibility', 15),
            current_time.hour,
            current_time.timetuple().tm_yday
        ]
        
        # Convert to format expected by model
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features_array)[0]
        
        # Solar radiation can't be negative
        prediction = max(0, prediction)
        
        return round(prediction, 1)
    
    def get_weather_from_api(self, latitude=None, longitude=None):
        """
        Get current weather data from OpenWeatherMap API
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
        
        Returns:
            Dictionary with weather data
        """
        if latitude is None:
            latitude = config.DEFAULT_LATITUDE
        if longitude is None:
            longitude = config.DEFAULT_LONGITUDE
        
        if config.WEATHER_API_KEY == "32136073cec9811a5b96bf05fadd3bce":
            print("Warning: Please set your API key in config.py")
            return None
        
        try:
            url = f"{config.WEATHER_BASE_URL}/weather"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': config.WEATHER_API_KEY,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                # Extract relevant weather information
                weather_data = {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data.get('wind', {}).get('speed', 0) * 3.6,  # Convert m/s to km/h
                    'cloud_cover': data.get('clouds', {}).get('all', 0),
                    'visibility': data.get('visibility', 10000) / 1000,  # Convert m to km
                    'location': data['name']
                }
                return weather_data
            else:
                print(f"Error getting weather data: {data.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"Error connecting to weather API: {e}")
            return None
    
    def predict_current_solar_radiation(self, latitude=None, longitude=None):
        """
        Get current weather and predict solar radiation
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
        
        Returns:
            Prediction result with weather data
        """
        print("Getting current weather data...")
        
        # Get current weather
        weather_data = self.get_weather_from_api(latitude, longitude)
        
        if weather_data is None:
            print("Using sample weather data for demonstration...")
            # Use sample data if API is not available
            weather_data = {
                'temperature': 25.0,
                'humidity': 60.0,
                'pressure': 1013.0,
                'wind_speed': 10.0,
                'cloud_cover': 20.0,
                'visibility': 15.0,
                'location': 'Sample Location'
            }
        
        # Make prediction
        prediction = self.predict_from_weather_data(weather_data)
        
        if prediction is not None:
            # Create result
            result = {
                'predicted_solar_radiation': prediction,
                'weather_conditions': weather_data,
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
        
        return None

def manual_prediction():
    """
    Allow user to input weather conditions manually
    """
    print("\n=== Manual Weather Input ===")
    print("Enter weather conditions (press Enter for default values):")
    
    try:
        temp = input("Temperature (°C) [default: 25]: ")
        temperature = float(temp) if temp else 25.0
        
        hum = input("Humidity (%) [default: 60]: ")
        humidity = float(hum) if hum else 60.0
        
        pres = input("Pressure (hPa) [default: 1013]: ")
        pressure = float(pres) if pres else 1013.0
        
        wind = input("Wind Speed (km/h) [default: 10]: ")
        wind_speed = float(wind) if wind else 10.0
        
        cloud = input("Cloud Cover (%) [default: 30]: ")
        cloud_cover = float(cloud) if cloud else 30.0
        
        vis = input("Visibility (km) [default: 15]: ")
        visibility = float(vis) if vis else 15.0
        
        weather_data = {
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'cloud_cover': cloud_cover,
            'visibility': visibility
        }
        
        return weather_data
        
    except ValueError:
        print("Invalid input. Using default values.")
        return {
            'temperature': 25.0,
            'humidity': 60.0,
            'pressure': 1013.0,
            'wind_speed': 10.0,
            'cloud_cover': 30.0,
            'visibility': 15.0
        }

def main():
    """
    Main function to run predictions
    """
    print("=== Solar Radiation Predictor ===\n")
    
    # Create predictor
    predictor = SolarPredictor()
    
    if predictor.model is None:
        return
    
    while True:
        print("\nChoose an option:")
        print("1. Predict using current weather (requires API key)")
        print("2. Enter weather conditions manually")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            result = predictor.predict_current_solar_radiation()
            if result:
                print(f"\n=== Prediction Results ===")
                print(f"Location: {result['weather_conditions'].get('location', 'Unknown')}")
                print(f"Prediction Time: {result['prediction_time']}")
                print(f"Predicted Solar Radiation: {result['predicted_solar_radiation']} W/m²")
                print(f"\nWeather Conditions:")
                for key, value in result['weather_conditions'].items():
                    if key != 'location':
                        print(f"  {key.replace('_', ' ').title()}: {value}")
        
        elif choice == '2':
            weather_data = manual_prediction()
            prediction = predictor.predict_from_weather_data(weather_data)
            
            if prediction is not None:
                print(f"\n=== Prediction Results ===")
                print(f"Predicted Solar Radiation: {prediction} W/m²")
                print(f"\nWeather Conditions Used:")
                for key, value in weather_data.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        
        elif choice == '3':
            print("Thank you for using the Solar Radiation Predictor!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()