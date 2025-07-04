"""
Interactive Map-Based Solar Radiation Prediction Web App
========================================================

This web application allows users to:
1. Click anywhere on an interactive world map
2. Automatically fetch real-time weather data for that location
3. Get solar radiation predictions using the trained AI model
4. View results with location information and weather details

Features:
- Interactive Leaflet map with click functionality
- Real-time weather API integration
- Automatic coordinate detection
- Beautiful responsive design
- Error handling and fallback options
- Location search functionality
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import requests
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    import config
except ImportError:
    print("Error: Could not import config.py")
    sys.exit(1)

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load(config.MODEL_PATH)
    print("Model loaded successfully for web app!")
    model_loaded = True
except FileNotFoundError:
    print("Warning: Model not found. Please run model_trainer.py first.")
    print("The app will still work with sample data for demonstration.")
    model = None
    model_loaded = False

class WeatherAPI:
    """Handle weather API requests"""
    
    def __init__(self):
        self.api_key = config.WEATHER_API_KEY
        self.base_url = config.WEATHER_BASE_URL
    
    def get_weather_data(self, latitude, longitude):
        """Get current weather data for given coordinates"""
        if self.api_key == "PUT_YOUR_API_KEY_HERE":
            return self._get_mock_weather_data(latitude, longitude)
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_weather_data(data)
            else:
                print(f"Weather API Error: {response.status_code}")
                return self._get_mock_weather_data(latitude, longitude)
                
        except Exception as e:
            print(f"Weather API Exception: {e}")
            return self._get_mock_weather_data(latitude, longitude)
    
    def _parse_weather_data(self, api_data):
        """Parse OpenWeatherMap API response"""
        try:
            weather_data = {
                'temperature': api_data['main']['temp'],
                'humidity': api_data['main']['humidity'],
                'pressure': api_data['main']['pressure'],
                'wind_speed': api_data.get('wind', {}).get('speed', 0) * 3.6,  # m/s to km/h
                'cloud_cover': api_data.get('clouds', {}).get('all', 0),
                'visibility': api_data.get('visibility', 10000) / 1000,  # m to km
                'location': f"{api_data['name']}, {api_data['sys']['country']}",
                'weather_description': api_data['weather'][0]['description'].title(),
                'latitude': api_data['coord']['lat'],
                'longitude': api_data['coord']['lon'],
                'data_source': 'OpenWeatherMap API'
            }
            return weather_data
        except KeyError as e:
            print(f"Error parsing weather data: {e}")
            return None
    
    def _get_mock_weather_data(self, latitude, longitude):
        """Generate realistic mock weather data when API is not available"""
        # Generate realistic weather based on location and season
        current_time = datetime.now()
        day_of_year = current_time.timetuple().tm_yday
        hour = current_time.hour
        
        # Base temperature varies by latitude (colder near poles)
        base_temp = 25 - abs(latitude) * 0.5
        
        # Seasonal variation
        seasonal_temp = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily variation
        daily_temp = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        temperature = base_temp + seasonal_temp + daily_temp + np.random.normal(0, 2)
        
        # Generate other weather parameters
        humidity = max(30, min(90, 60 + np.random.normal(0, 15)))
        pressure = 1013 + np.random.normal(0, 10)
        wind_speed = max(0, 5 + np.random.normal(0, 3))
        cloud_cover = max(0, min(100, np.random.normal(40, 25)))
        visibility = max(5, min(30, 20 - (cloud_cover/100) * 10 + np.random.normal(0, 3)))
        
        # Generate location name based on coordinates
        location_name = self._generate_location_name(latitude, longitude)
        
        return {
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'pressure': round(pressure, 1),
            'wind_speed': round(wind_speed, 1),
            'cloud_cover': round(cloud_cover, 1),
            'visibility': round(visibility, 1),
            'location': location_name,
            'weather_description': self._get_weather_description(cloud_cover),
            'latitude': latitude,
            'longitude': longitude,
            'data_source': 'Mock Data (Demo Mode)'
        }
    
    def _generate_location_name(self, lat, lon):
        """Generate a location name based on coordinates"""
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        return f"Location {abs(lat):.1f}{lat_dir}, {abs(lon):.1f}{lon_dir}"
    
    def _get_weather_description(self, cloud_cover):
        """Get weather description based on cloud cover"""
        if cloud_cover < 10:
            return "Clear Sky"
        elif cloud_cover < 30:
            return "Few Clouds"
        elif cloud_cover < 60:
            return "Partly Cloudy"
        elif cloud_cover < 85:
            return "Mostly Cloudy"
        else:
            return "Overcast"

class SolarPredictor:
    """Handle solar radiation predictions"""
    
    def __init__(self, model):
        self.model = model
    
    def predict_solar_radiation(self, weather_data):
        """Predict solar radiation based on weather data"""
        if self.model is None:
            return self._get_mock_prediction(weather_data)
        
        try:
            # Prepare features for prediction
            current_time = datetime.now()
            
            features = [
                weather_data['temperature'],
                weather_data['humidity'],
                weather_data['pressure'],
                weather_data['wind_speed'],
                weather_data['cloud_cover'],
                weather_data['visibility'],
                current_time.hour,
                current_time.timetuple().tm_yday
            ]
            
            # Make prediction
            features_array = np.array(features).reshape(1, -1)
            prediction = self.model.predict(features_array)[0]
            prediction = max(0, round(prediction, 1))
            
            # Calculate confidence and quality metrics
            quality_score = self._calculate_prediction_quality(weather_data, current_time.hour)
            confidence = self._calculate_confidence(weather_data)
            
            return {
                'solar_radiation': prediction,
                'quality_score': quality_score,
                'confidence': confidence,
                'prediction_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_status': 'AI Model Prediction'
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._get_mock_prediction(weather_data)
    
    def _calculate_prediction_quality(self, weather_data, hour):
        """Calculate prediction quality description"""
        cloud_cover = weather_data['cloud_cover']
        
        if hour < 6 or hour > 18:
            return "Night time - minimal solar radiation expected"
        elif cloud_cover > 80:
            return "Very cloudy - low solar radiation expected"
        elif cloud_cover > 50:
            return "Partly cloudy - moderate solar radiation"
        elif cloud_cover < 20:
            return "Clear sky - high solar radiation expected"
        else:
            return "Normal conditions - moderate solar radiation"
    
    def _calculate_confidence(self, weather_data):
        """Calculate prediction confidence percentage"""
        # Base confidence
        confidence = 85
        
        # Reduce confidence for extreme conditions
        if weather_data['cloud_cover'] > 90:
            confidence -= 10
        if weather_data['humidity'] > 85:
            confidence -= 5
        if weather_data['visibility'] < 5:
            confidence -= 10
        
        return max(50, min(95, confidence))
    
    def _get_mock_prediction(self, weather_data):
        """Generate mock prediction when model is not available"""
        # Simple prediction based on weather conditions
        base_radiation = 300
        
        # Reduce for clouds
        cloud_reduction = (100 - weather_data['cloud_cover']) / 100
        
        # Reduce for humidity
        humidity_reduction = (100 - weather_data['humidity']) / 100 * 0.2 + 0.8
        
        # Time of day effect
        hour = datetime.now().hour
        if hour < 6 or hour > 18:
            time_effect = 0
        else:
            time_effect = np.sin(np.pi * (hour - 6) / 12)
        
        prediction = base_radiation * cloud_reduction * humidity_reduction * time_effect
        
        return {
            'solar_radiation': round(max(0, prediction), 1),
            'quality_score': self._calculate_prediction_quality(weather_data, hour),
            'confidence': 75,
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_status': 'Demo Mode (Train model for AI predictions)'
        }

# Initialize components
weather_api = WeatherAPI()
solar_predictor = SolarPredictor(model)

@app.route('/')
def home():
    """Main page with interactive map"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Solar Prediction Map</title>
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            color: #2c5530;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .container {
            display: flex;
            height: calc(100vh - 120px);
            margin: 20px;
            gap: 20px;
        }
        
        .map-container {
            flex: 2;
            background: white;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        #map {
            height: 100%;
            width: 100%;
        }
        
        .results-panel {
            flex: 1;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            overflow-y: auto;
        }
        
        .instruction {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4CAF50;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 10px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #4CAF50;
        }
        
        .result-card h3 {
            color: #2c5530;
            margin-bottom: 15px;
        }
        
        .solar-prediction {
            background: linear-gradient(135deg, #FF6B35, #F7931E);
            color: white;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .solar-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .weather-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }
        
        .weather-item {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        
        .weather-item .label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .weather-item .value {
            font-weight: bold;
            color: #2c5530;
        }
        
        .coordinates {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-family: monospace;
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #f44336;
        }
        
        .search-container {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 1000;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        
        .search-container input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            width: 200px;
        }
        
        .search-container button {
            padding: 8px 12px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-left: 5px;
        }
        
        .search-container button:hover {
            background: #45a049;
        }
        
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
                height: auto;
            }
            
            .map-container {
                height: 400px;
            }
            
            .header h1 {
                font-size: 1.8em;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Interactive Solar Prediction Map</h1>
        <p>Click anywhere on the map to get real-time weather data and solar radiation predictions</p>
    </div>
    
    <div class="container">
        <div class="map-container">
            <div class="search-container">
                <input type="text" id="searchInput" placeholder="Search location..." />
                <button onclick="searchLocation()">Search</button>
            </div>
            <div id="map"></div>
        </div>
        
        <div class="results-panel">
            <div class="instruction">
                <h3>How to Use</h3>
                <p>Click any location on the map to get:</p>
                <ul style="text-align: left; margin-top: 10px;">
                    <li>Real-time weather data</li>
                    <li>Solar radiation prediction</li>
                    <li>Location coordinates</li>
                    <li>Prediction confidence</li>
                </ul>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Getting weather data and making prediction...</p>
            </div>
            
            <div id="results"></div>
        </div>
    </div>

    <!-- Leaflet JavaScript -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <script>
        // Initialize map
        var map = L.map('map').setView([40.7128, -74.0060], 3);
        
        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        
        // Store current marker
        var currentMarker = null;
        
        // Handle map clicks
        map.on('click', function(e) {
            var lat = e.latlng.lat;
            var lng = e.latlng.lng;
            
            // Remove previous marker
            if (currentMarker) {
                map.removeLayer(currentMarker);
            }
            
            // Add new marker
            currentMarker = L.marker([lat, lng]).addTo(map);
            
            // Get prediction
            getPrediction(lat, lng);
        });
        
        function getPrediction(lat, lng) {
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            // Make API request
            fetch('/predict_location', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    latitude: lat,
                    longitude: lng
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                displayResults(data);
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('results').innerHTML = 
                    '<div class="error"><h3>Error</h3><p>Failed to get prediction: ' + error.message + '</p></div>';
            });
        }
        
        function displayResults(data) {
            if (data.error) {
                document.getElementById('results').innerHTML = 
                    '<div class="error"><h3>Error</h3><p>' + data.error + '</p></div>';
                return;
            }
            
            const weather = data.weather_data;
            const prediction = data.prediction;
            
            document.getElementById('results').innerHTML = `
                <div class="solar-prediction">
                    <h3>Solar Radiation Prediction</h3>
                    <div class="solar-value">${prediction.solar_radiation} W/m2</div>
                    <p>${prediction.quality_score}</p>
                    <p>Confidence: ${prediction.confidence}%</p>
                </div>
                
                <div class="result-card">
                    <h3>Location Information</h3>
                    <p><strong>Location:</strong> ${weather.location}</p>
                    <p><strong>Description:</strong> ${weather.weather_description}</p>
                    <div class="coordinates">
                        <strong>Coordinates:</strong><br>
                        Latitude: ${weather.latitude.toFixed(4)} degrees<br>
                        Longitude: ${weather.longitude.toFixed(4)} degrees
                    </div>
                </div>
                
                <div class="result-card">
                    <h3>Current Weather</h3>
                    <div class="weather-grid">
                        <div class="weather-item">
                            <div class="label">Temperature</div>
                            <div class="value">${weather.temperature} C</div>
                        </div>
                        <div class="weather-item">
                            <div class="label">Humidity</div>
                            <div class="value">${weather.humidity}%</div>
                        </div>
                        <div class="weather-item">
                            <div class="label">Pressure</div>
                            <div class="value">${weather.pressure} hPa</div>
                        </div>
                        <div class="weather-item">
                            <div class="label">Wind Speed</div>
                            <div class="value">${weather.wind_speed} km/h</div>
                        </div>
                        <div class="weather-item">
                            <div class="label">Cloud Cover</div>
                            <div class="value">${weather.cloud_cover}%</div>
                        </div>
                        <div class="weather-item">
                            <div class="label">Visibility</div>
                            <div class="value">${weather.visibility} km</div>
                        </div>
                    </div>
                </div>
                
                <div class="result-card">
                    <h3>Prediction Details</h3>
                    <p><strong>Model Status:</strong> ${prediction.model_status}</p>
                    <p><strong>Data Source:</strong> ${weather.data_source}</p>
                    <p><strong>Prediction Time:</strong> ${prediction.prediction_time}</p>
                </div>
            `;
        }
        
        function searchLocation() {
            const query = document.getElementById('searchInput').value;
            if (!query) return;
            
            fetch('https://nominatim.openstreetmap.org/search?format=json&q=' + encodeURIComponent(query))
                .then(response => response.json())
                .then(data => {
                    if (data && data.length > 0) {
                        const lat = parseFloat(data[0].lat);
                        const lng = parseFloat(data[0].lon);
                        
                        map.setView([lat, lng], 10);
                        
                        if (currentMarker) {
                            map.removeLayer(currentMarker);
                        }
                        
                        currentMarker = L.marker([lat, lng]).addTo(map);
                        getPrediction(lat, lng);
                    } else {
                        alert('Location not found. Please try a different search term.');
                    }
                })
                .catch(error => {
                    alert('Error searching location: ' + error.message);
                });
        }
        
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchLocation();
            }
        });
        
        setTimeout(() => {
            L.marker([40.7128, -74.0060]).addTo(map)
                .bindPopup('<b>New York City</b><br>Click to get solar prediction!')
                .openPopup();
        }, 1000);
    </script>
</body>
</html>
    '''

@app.route('/predict_location', methods=['POST'])
def predict_location():
    """Handle location-based prediction requests"""
    try:
        data = request.get_json()
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        
        # Get weather data for the location
        weather_data = weather_api.get_weather_data(latitude, longitude)
        
        if weather_data is None:
            return jsonify({'error': 'Failed to get weather data for this location'})
        
        # Make solar radiation prediction
        prediction = solar_predictor.predict_solar_radiation(weather_data)
        
        return jsonify({
            'weather_data': weather_data,
            'prediction': prediction,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'api_configured': config.WEATHER_API_KEY != "PUT_YOUR_API_KEY_HERE",
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("STARTING INTERACTIVE SOLAR PREDICTION MAP")
    print("=" * 50)
    print(f"Model Status: {'Loaded' if model_loaded else 'Demo Mode'}")
    print(f"API Status: {'Configured' if config.WEATHER_API_KEY != 'PUT_YOUR_API_KEY_HERE' else 'Demo Mode'}")
    print("Interactive Features:")
    print("   • Click anywhere on map for predictions")
    print("   • Search locations by name")
    print("   • Real-time weather data integration")
    print("   • Beautiful responsive design")
    print("\nOpen your browser and go to: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)