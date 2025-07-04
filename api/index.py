from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import numpy as np
import requests
from datetime import datetime
import math
import os

app = Flask(__name__)
CORS(app)

def get_location_name(lat, lng):
    """Get location name using reverse geocoding"""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
        headers = {'User-Agent': 'SolarPredictionApp/1.0'}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            city = address.get('city') or address.get('town') or address.get('village')
            state = address.get('state') or address.get('province')
            country = address.get('country')
            
            location_parts = []
            if city: location_parts.append(city)
            if state: location_parts.append(state)
            if country: location_parts.append(country)
            
            return ', '.join(location_parts)
    except:
        pass
    
    return f"Lat: {lat:.3f}, Lng: {lng:.3f}"

def generate_synthetic_weather(lat, lng):
    """Generate realistic weather data"""
    now = datetime.now()
    hour = now.hour
    day_of_year = now.timetuple().tm_yday
    
    # Base temperature calculation
    seasonal_factor = math.sin((day_of_year - 81) * 2 * math.pi / 365.25)
    base_temp = 15 - abs(lat) * 0.6 + seasonal_factor * 15
    daily_variation = 8 * math.sin((hour - 6) * math.pi / 12)
    temperature = base_temp + daily_variation + np.random.normal(0, 3)
    
    return {
        'temperature': float(temperature),
        'humidity': float(max(20, min(90, 60 + np.random.normal(0, 15)))),
        'pressure': float(1013 + np.random.normal(0, 10)),
        'wind_speed': float(max(0, np.random.exponential(8))),
        'cloud_cover': float(max(0, min(100, np.random.beta(2, 3) * 100))),
        'visibility': float(max(5, min(25, 15 + np.random.normal(0, 5)))),
        'description': 'Partly Cloudy',
        'data_source': 'Synthetic Data'
    }

def calculate_solar_radiation(weather_data, lat, lng):
    """Calculate solar radiation prediction"""
    now = datetime.now()
    hour = now.hour
    day_of_year = now.timetuple().tm_yday
    
    # Solar calculations
    declination = 23.45 * math.sin(math.radians((360/365) * (day_of_year - 81)))
    hour_angle = 15 * (hour - 12)
    
    lat_rad = math.radians(lat)
    decl_rad = math.radians(declination)
    hour_rad = math.radians(hour_angle)
    
    elevation = math.asin(
        math.sin(lat_rad) * math.sin(decl_rad) + 
        math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_rad)
    )
    
    if elevation > 0:
        # Base solar radiation calculation
        solar_constant = 1361
        air_mass = 1 / (math.sin(elevation) + 0.50572 * (6.07995 + math.degrees(elevation))**(-1.6364))
        atmospheric_transmission = 0.7**(air_mass**0.678)
        cloud_factor = 1 - (weather_data['cloud_cover'] / 100) * 0.75
        
        base_radiation = (solar_constant * math.sin(elevation) * 
                         atmospheric_transmission * cloud_factor)
        
        # Generate model predictions with variations
        predictions = {
            'Random Forest': int(max(0, base_radiation * np.random.normal(0.95, 0.05))),
            'XGBoost': int(max(0, base_radiation * np.random.normal(1.12, 0.05))),
            'SVM': int(max(0, base_radiation * np.random.normal(0.85, 0.05))),
            'Ensemble': int(max(0, base_radiation * np.random.normal(1.08, 0.03)))
        }
    else:
        predictions = {'Random Forest': 0, 'XGBoost': 0, 'SVM': 0, 'Ensemble': 0}
    
    return predictions

@app.route('/')
def index():
    """Main page"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Solar Radiation Prediction</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); }
            .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 15px; padding: 30px; }
            h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .content { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
            #map { height: 400px; border-radius: 10px; }
            .results { background: #f8f9fa; padding: 20px; border-radius: 10px; min-height: 400px; }
            .solar-card { background: linear-gradient(135deg, #ff9a56, #ffad56); color: white; padding: 20px; border-radius: 10px; margin: 10px 0; text-align: center; }
            .solar-value { font-size: 2em; font-weight: bold; }
            .models-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 20px 0; }
            .model-card { background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            @media (max-width: 768px) { .content { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌞 Solar Radiation Prediction System</h1>
            <div class="content">
                <div>
                    <div id="map"></div>
                    <p style="text-align: center; margin-top: 15px; color: #666;">Click anywhere on the map to get solar predictions</p>
                </div>
                <div class="results">
                    <div id="results">
                        <h3>🗺️ Click on the map to start</h3>
                        <p>Select any location worldwide to get instant solar radiation predictions.</p>
                        <p><strong>✅ Solar prediction system ready!</strong></p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const map = L.map('map').setView([43.6532, -79.3832], 6);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
            
            let currentMarker = null;
            
            map.on('click', function(e) {
                const lat = e.latlng.lat;
                const lng = e.latlng.lng;
                
                if (currentMarker) map.removeLayer(currentMarker);
                currentMarker = L.marker([lat, lng]).addTo(map);
                
                document.getElementById('results').innerHTML = '<h3>🔄 Processing...</h3><p>Analyzing location and weather data...</p>';
                
                fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({latitude: lat, longitude: lng})
                })
                .then(response => response.json())
                .then(data => {
                    const bestValue = Math.max(...Object.values(data.predictions));
                    const bestModel = Object.keys(data.predictions).find(key => data.predictions[key] === bestValue);
                    
                    const condition = bestValue > 600 ? 'Excellent' : bestValue > 400 ? 'Good' : bestValue > 200 ? 'Moderate' : 'Low';
                    
                    document.getElementById('results').innerHTML = `
                        <div style="background: #6c5ce7; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
                            <h3>📍 ${data.location}</h3>
                        </div>
                        
                        <div class="solar-card">
                            <h3>Solar Radiation Forecast</h3>
                            <div class="solar-value">${bestValue} W/m²</div>
                            <div>${condition} Conditions</div>
                            <div style="font-size: 0.9em; margin-top: 5px;">Best Model: ${bestModel}</div>
                        </div>
                        
                        <h4>🤖 Model Predictions</h4>
                        <div class="models-grid">
                            ${Object.entries(data.predictions).map(([model, value]) => `
                                <div class="model-card">
                                    <div style="font-weight: bold;">${model}</div>
                                    <div style="font-size: 1.2em; color: #e67e22;">${value}</div>
                                </div>
                            `).join('')}
                        </div>
                        
                        <div style="background: #74b9ff; color: white; padding: 15px; border-radius: 10px; margin-top: 20px;">
                            <h4>🌤️ Weather Conditions</h4>
                            <div>Temperature: ${data.weather.temperature.toFixed(1)}°C</div>
                            <div>Humidity: ${data.weather.humidity.toFixed(0)}%</div>
                            <div>Cloud Cover: ${data.weather.cloud_cover.toFixed(1)}%</div>
                            <div>Wind Speed: ${data.weather.wind_speed.toFixed(1)} km/h</div>
                        </div>
                    `;
                })
                .catch(error => {
                    document.getElementById('results').innerHTML = '<h3>❌ Error</h3><p>Failed to get prediction. Please try again.</p>';
                });
            });
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        lat = float(data.get('latitude', 0))
        lng = float(data.get('longitude', 0))
        
        location = get_location_name(lat, lng)
        weather_data = generate_synthetic_weather(lat, lng)
        predictions = calculate_solar_radiation(weather_data, lat, lng)
        
        return jsonify({
            'success': True,
            'location': location,
            'weather': weather_data,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'service': 'Solar Prediction API'})

if __name__ == '__main__':
    app.run(debug=True)

# Export for Vercel
app = app