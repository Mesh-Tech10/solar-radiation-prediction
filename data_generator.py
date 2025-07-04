"""
This file creates sample weather and solar data for training our model.
In a real project, you'd get this data from weather stations and solar panels.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(days=365):
    """
    Generate sample weather and solar radiation data
    
    Args:
        days: Number of days of data to generate
    
    Returns:
        DataFrame with weather and solar data
    """
    print("Generating sample weather and solar data...")
    
    # Create date range
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    data = []
    
    for date in dates:
        # Generate 24 hours of data for each day
        for hour in range(24):
            current_time = date.replace(hour=hour)
            
            # Simulate realistic weather patterns
            day_of_year = date.timetuple().tm_yday
            
            # Temperature varies by season and time of day
            base_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)
            hourly_temp_variation = 5 * np.sin(2 * np.pi * hour / 24)
            temperature = base_temp + hourly_temp_variation + np.random.normal(0, 2)
            
            # Humidity (higher at night, varies by season)
            base_humidity = 60 + 20 * np.sin(2 * np.pi * (day_of_year + 90) / 365)
            humidity = base_humidity + 10 * np.sin(2 * np.pi * (hour + 12) / 24) + np.random.normal(0, 5)
            humidity = max(20, min(95, humidity))  # Keep within realistic range
            
            # Pressure (relatively stable with small variations)
            pressure = 1013 + np.random.normal(0, 10)
            
            # Wind speed (generally higher during day)
            wind_speed = 3 + 2 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1)
            wind_speed = max(0, wind_speed)
            
            # Cloud cover (affects solar radiation significantly)
            cloud_cover = max(0, min(100, np.random.normal(40, 25)))
            
            # Visibility (lower when cloudy)
            visibility = 20 - (cloud_cover / 100) * 10 + np.random.normal(0, 2)
            visibility = max(1, visibility)
            
            # Solar radiation calculation
            # Higher when sun is up, lower when cloudy
            solar_elevation = max(0, np.sin(2 * np.pi * (hour - 6) / 12))  # Simplified sun position
            clear_sky_radiation = 1000 * solar_elevation  # Max solar radiation
            
            # Reduce radiation based on cloud cover
            cloud_reduction = 1 - (cloud_cover / 100) * 0.8
            solar_radiation = clear_sky_radiation * cloud_reduction
            
            # Add some randomness
            solar_radiation += np.random.normal(0, 20)
            solar_radiation = max(0, solar_radiation)
            
            # If it's night time, solar radiation should be 0
            if hour < 6 or hour > 18:
                solar_radiation = 0
            
            data.append({
                'datetime': current_time,
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'pressure': round(pressure, 1),
                'wind_speed': round(wind_speed, 1),
                'cloud_cover': round(cloud_cover, 1),
                'visibility': round(visibility, 1),
                'hour': hour,
                'day_of_year': day_of_year,
                'solar_radiation': round(solar_radiation, 1)  # This is what we want to predict
            })
    
    df = pd.DataFrame(data)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save the data
    df.to_csv('data/weather_solar_data.csv', index=False)
    print(f"Generated {len(df)} rows of sample data")
    print("Data saved to 'data/weather_solar_data.csv'")
    
    return df

if __name__ == "__main__":
    # Generate sample data when this file is run
    generate_sample_data(365)