"""
Ultra Simple Solar Test - Just Numpy
"""

import numpy as np
import os

def simple_solar_prediction():
    print("🌞 Ultra Simple Solar Prediction Test")
    print("=" * 40)
    
    # Generate sample data
    print("🧪 Generating sample data...")
    np.random.seed(42)
    n_samples = 1000
    
    # Simple features
    hours = np.random.randint(0, 24, n_samples)
    temperatures = 20 + 15 * np.random.randn(n_samples)
    cloud_cover = np.clip(np.random.normal(40, 20, n_samples), 0, 100)
    
    # Simple solar radiation model
    base_solar = np.where(
        (hours >= 6) & (hours <= 18),  # Daylight hours
        800 * np.sin(np.pi * (hours - 6) / 12),  # Sun curve
        0  # Night
    )
    
    # Reduce for clouds
    cloud_reduction = 1 - (cloud_cover / 100) * 0.8
    
    # Temperature effect
    temp_effect = 1 + (temperatures - 20) * 0.01
    
    # Final solar radiation
    solar_radiation = base_solar * cloud_reduction * temp_effect
    solar_radiation = np.maximum(0, solar_radiation + np.random.normal(0, 20, n_samples))
    
    print(f"✅ Generated {n_samples} samples")
    print(f"📊 Solar radiation range: {solar_radiation.min():.1f} - {solar_radiation.max():.1f} W/m²")
    
    # Simple analysis
    day_mask = (hours >= 6) & (hours <= 18)
    day_solar = solar_radiation[day_mask]
    night_solar = solar_radiation[~day_mask]
    
    print(f"📈 Daytime average: {day_solar.mean():.1f} W/m²")
    print(f"📈 Nighttime average: {night_solar.mean():.1f} W/m²")
    
    # Test prediction
    test_scenarios = [
        {"hour": 12, "temp": 25, "clouds": 0, "name": "Perfect noon"},
        {"hour": 12, "temp": 25, "clouds": 80, "name": "Cloudy noon"},
        {"hour": 0, "temp": 15, "clouds": 0, "name": "Midnight"},
    ]
    
    print("\n🔮 Prediction Tests:")
    for scenario in test_scenarios:
        h, t, c = scenario["hour"], scenario["temp"], scenario["clouds"]
        
        if 6 <= h <= 18:
            base = 800 * np.sin(np.pi * (h - 6) / 12)
            cloud_adj = base * (1 - c / 100 * 0.8)
            temp_adj = cloud_adj * (1 + (t - 20) * 0.01)
            prediction = max(0, temp_adj)
        else:
            prediction = 0
        
        print(f"   {scenario['name']}: {prediction:.1f} W/m²")
    
    print("\n✅ Basic solar prediction working!")
    return True

def main():
    simple_solar_prediction()
    print("\n🚀 Next: Install scikit-learn for advanced models")

if __name__ == "__main__":
    main()