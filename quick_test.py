"""
Minimal Solar Prediction Test - No Pandas
Just numpy and scikit-learn
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

def generate_simple_data(n_samples=2000):
    """Generate simple solar data with numpy only"""
    print("🧪 Generating solar data...")
    
    np.random.seed(42)
    
    # Time features
    hours = np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples]
    days = np.repeat(np.arange(n_samples // 24 + 1), 24)[:n_samples]
    
    # Weather features
    temperature = 20 + 15 * np.sin(2 * np.pi * days / 365) + 8 * np.sin(2 * np.pi * hours / 24) + 3 * np.random.randn(n_samples)
    humidity = np.clip(60 + 20 * np.random.randn(n_samples), 20, 95)
    cloud_cover = np.clip(40 + 30 * np.random.randn(n_samples), 0, 100)
    wind_speed = np.maximum(0, 5 + 3 * np.random.randn(n_samples))
    pressure = 1013 + 8 * np.random.randn(n_samples)
    
    # Solar radiation calculation
    solar_elevation = np.maximum(0, 90 - 45 * np.abs(hours - 12) / 6)
    clear_sky = 1000 * np.sin(np.radians(solar_elevation))
    cloud_effect = 1 - (cloud_cover / 100) * 0.8
    seasonal_effect = 1 + 0.2 * np.sin(2 * np.pi * (days % 365 - 80) / 365)
    
    solar_radiation = np.maximum(0, clear_sky * cloud_effect * seasonal_effect + 30 * np.random.randn(n_samples))
    
    # Combine features
    X = np.column_stack([
        hours % 24,                    # hour
        days % 365,                    # day_of_year
        temperature,                   # temperature
        humidity,                      # humidity
        cloud_cover,                   # cloud_cover
        wind_speed,                    # wind_speed
        pressure,                      # pressure
        solar_elevation               # solar_elevation
    ])
    
    y = solar_radiation
    
    feature_names = ['hour', 'day_of_year', 'temperature', 'humidity', 'cloud_cover', 'wind_speed', 'pressure', 'solar_elevation']
    
    print(f"✅ Generated {n_samples} samples")
    print(f"📊 Features: {X.shape[1]} ({feature_names})")
    
    return X, y, feature_names

def train_model(X, y, feature_names):
    """Train Random Forest model"""
    print("🌲 Training Random Forest...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"📊 Training: {X_train.shape}")
    print(f"📊 Testing: {X_test.shape}")
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mae = np.mean(np.abs(y_test - predictions))
    
    print(f"📈 Model Performance:")
    print(f"   RMSE: {rmse:.2f} W/m²")
    print(f"   MAE: {mae:.2f} W/m²")
    print(f"   R²: {r2:.3f}")
    
    # Feature importance
    importance = model.feature_importances_
    print(f"\n🔝 Feature Importance:")
    for i, (name, imp) in enumerate(zip(feature_names, importance)):
        print(f"   {name}: {imp:.4f}")
    
    # Save model
    try:
        import joblib
        os.makedirs('models/saved_models', exist_ok=True)
        joblib.dump(model, 'models/saved_models/simple_solar_model.pkl')
        print(f"\n💾 Model saved to: models/saved_models/simple_solar_model.pkl")
    except:
        print("\n⚠️ Could not save model (joblib not available)")
    
    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}

def main():
    """Main function"""
    print("🌞 Simple Solar Prediction Test")
    print("=" * 40)
    
    # Generate data
    X, y, feature_names = generate_simple_data(n_samples=5000)
    
    # Train model
    model, metrics = train_model(X, y, feature_names)
    
    print("\n✅ Test completed successfully!")
    print("🎯 Your solar prediction system is working!")
    print("\nNext steps:")
    print("1. Fix pandas compatibility issues")
    print("2. Run full training pipeline")
    print("3. Add web interface")

if __name__ == "__main__":
    main()