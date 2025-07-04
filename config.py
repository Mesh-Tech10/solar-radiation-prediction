"""
Configuration file for Solar Radiation Prediction Project
"""

# =============================================================================
# API SETTINGS - CHANGE THIS WITH YOUR API KEY
# =============================================================================
WEATHER_API_KEY = "32136073cec9811a5b96bf05fadd3bce"  # Replace with your OpenWeatherMap API key
WEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5"

# =============================================================================
# PROJECT PATHS
# =============================================================================
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# File paths
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "weather_solar_data.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "solar_prediction_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# =============================================================================
# MODEL SETTINGS
# =============================================================================
# Features used for prediction (don't change unless you know what you're doing)
WEATHER_FEATURES = [
    'temperature',      # Temperature in Celsius
    'humidity',         # Relative humidity (%)
    'pressure',         # Atmospheric pressure (hPa)
    'wind_speed',       # Wind speed (km/h)
    'cloud_cover',      # Cloud coverage (%)
    'visibility',       # Visibility (km)
    'hour',            # Hour of day (0-23)
    'day_of_year'      # Day of year (1-365)
]

# Target variable
TARGET_VARIABLE = 'solar_radiation'  # What we're trying to predict

# Model parameters
RANDOM_STATE = 42  # For reproducible results
TEST_SIZE = 0.2   # 20% of data for testing

# =============================================================================
# DEFAULT LOCATION 
# =============================================================================
DEFAULT_LATITUDE = 43.5853   
DEFAULT_LONGITUDE = 79.6450
DEFAULT_LOCATION_NAME = "Mississauga, ON"

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
FIGURE_SIZE = (12, 8)
DPI = 300
SAVE_FIGURES = True

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================
def validate_config():
    """Check if configuration is valid"""
    issues = []
    
    if WEATHER_API_KEY == "PUT_YOUR_API_KEY_HERE":
        issues.append("⚠️  Please set your OpenWeatherMap API key in config.py")
    
    if not os.path.exists(PROJECT_ROOT):
        issues.append("❌ Project root directory not found")
    
    return issues

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory exists: {directory}")

def print_config_status():
    """Print current configuration status"""
    print("=" * 50)
    print("🔧 SOLAR PREDICTION PROJECT CONFIGURATION")
    print("=" * 50)
    
    print(f"📍 Project Root: {PROJECT_ROOT}")
    print(f"📊 Data Directory: {DATA_DIR}")
    print(f"🤖 Models Directory: {MODELS_DIR}")
    print(f"📈 Outputs Directory: {OUTPUTS_DIR}")
    print(f"🌍 Default Location: {DEFAULT_LOCATION_NAME}")
    
    # Check for issues
    issues = validate_config()
    if issues:
        print("\n❌ CONFIGURATION ISSUES:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ Configuration looks good!")
    
    print("=" * 50)

if __name__ == "__main__":
    # Run this to check your configuration
    create_directories()
    print_config_status()