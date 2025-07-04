"""
Solar Radiation Data Generator
=============================

This file creates realistic weather and solar radiation data for training our AI model.
In a real project, this data would come from weather stations and solar panel measurements.

What this does:
- Generates 365 days of hourly weather data (8,760 data points)
- Creates realistic patterns (hot summers, cold winters, sunny days, cloudy days)
- Calculates corresponding solar radiation based on weather conditions
- Saves everything to a CSV file for training

Author: Beginner-Friendly Solar Prediction Project
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the project root to Python path so we can import config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    import config
except ImportError:
    print("❌ Error: Could not import config.py")
    print("Make sure config.py exists in your project root folder")
    sys.exit(1)

class WeatherSolarGenerator:
    """
    Generates realistic weather and solar radiation data
    """
    
    def __init__(self, days=365, start_date="2023-01-01"):
        """
        Initialize the generator
        
        Args:
            days: Number of days to generate data for
            start_date: Starting date for data generation
        """
        self.days = days
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.data = []
        
        print(f"🌤️  Initializing Weather & Solar Data Generator")
        print(f"📅 Generating {days} days of data starting from {start_date}")
        print(f"📊 Total data points: {days * 24:,} (hourly data)")
    
    def calculate_solar_position(self, day_of_year, hour):
        """
        Calculate sun position (simplified model)
        
        Args:
            day_of_year: Day of the year (1-365)
            hour: Hour of the day (0-23)
        
        Returns:
            Solar elevation angle (0 = horizon, 90 = directly overhead)
        """
        # Solar declination (sun's position throughout the year)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle (sun's position throughout the day)
        hour_angle = 15 * (hour - 12)  # 15 degrees per hour
        
        # Assume latitude of 40 degrees (New York City area)
        latitude = config.DEFAULT_LATITUDE
        
        # Calculate solar elevation
        elevation = np.arcsin(
            np.sin(np.radians(declination)) * np.sin(np.radians(latitude)) +
            np.cos(np.radians(declination)) * np.cos(np.radians(latitude)) * 
            np.cos(np.radians(hour_angle))
        )
        
        return max(0, np.degrees(elevation))  # Can't be negative
    
    def generate_temperature(self, day_of_year, hour, base_temp=15):
        """
        Generate realistic temperature patterns
        
        Args:
            day_of_year: Day of the year (1-365)
            hour: Hour of the day (0-23)
            base_temp: Base temperature in Celsius
        
        Returns:
            Temperature in Celsius
        """
        # Seasonal variation (warmer in summer, colder in winter)
        seasonal_variation = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily variation (warmer during day, cooler at night)
        daily_variation = 8 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Random variation
        random_variation = np.random.normal(0, 3)
        
        temperature = base_temp + seasonal_variation + daily_variation + random_variation
        
        return round(temperature, 1)
    
    def generate_humidity(self, temperature, day_of_year, hour):
        """
        Generate realistic humidity patterns
        
        Args:
            temperature: Current temperature
            day_of_year: Day of the year
            hour: Hour of the day
        
        Returns:
            Relative humidity percentage
        """
        # Base humidity varies by season
        base_humidity = 60 + 15 * np.sin(2 * np.pi * (day_of_year + 90) / 365)
        
        # Higher humidity at night
        daily_variation = -10 * np.sin(2 * np.pi * (hour - 12) / 24)
        
        # Lower humidity when it's hotter
        temp_effect = -0.5 * (temperature - 20)
        
        # Random variation
        random_variation = np.random.normal(0, 8)
        
        humidity = base_humidity + daily_variation + temp_effect + random_variation
        
        # Keep within realistic bounds
        return round(max(20, min(95, humidity)), 1)
    
    def generate_pressure(self, day_of_year):
        """
        Generate atmospheric pressure
        
        Args:
            day_of_year: Day of the year
        
        Returns:
            Atmospheric pressure in hPa
        """
        # Base pressure with seasonal variation
        base_pressure = 1013
        seasonal_variation = 5 * np.sin(2 * np.pi * day_of_year / 365)
        random_variation = np.random.normal(0, 8)
        
        pressure = base_pressure + seasonal_variation + random_variation
        
        return round(pressure, 1)
    
    def generate_wind_speed(self, hour, season_factor):
        """
        Generate wind speed
        
        Args:
            hour: Hour of the day
            season_factor: Seasonal influence
        
        Returns:
            Wind speed in km/h
        """
        # Generally windier during the day
        daily_pattern = 2 + 3 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Seasonal variation (windier in winter)
        seasonal_effect = season_factor * 2
        
        # Random variation
        random_variation = np.random.normal(0, 2)
        
        wind_speed = daily_pattern + seasonal_effect + random_variation
        
        return round(max(0, wind_speed), 1)
    
    def generate_cloud_cover(self, humidity, day_of_year):
        """
        Generate cloud cover percentage
        
        Args:
            humidity: Current humidity
            day_of_year: Day of the year
        
        Returns:
            Cloud cover percentage (0-100)
        """
        # More clouds when humidity is high
        humidity_effect = (humidity - 50) * 0.8
        
        # Seasonal variation (more clouds in winter)
        seasonal_effect = 20 * np.sin(2 * np.pi * (day_of_year + 180) / 365)
        
        # Random variation
        random_variation = np.random.normal(0, 25)
        
        cloud_cover = 40 + humidity_effect + seasonal_effect + random_variation
        
        return round(max(0, min(100, cloud_cover)), 1)
    
    def generate_visibility(self, cloud_cover, humidity):
        """
        Generate visibility
        
        Args:
            cloud_cover: Current cloud cover percentage
            humidity: Current humidity
        
        Returns:
            Visibility in kilometers
        """
        # Base visibility
        base_visibility = 20
        
        # Reduced visibility with clouds and high humidity
        cloud_effect = -(cloud_cover / 100) * 8
        humidity_effect = -(humidity - 60) * 0.1
        
        # Random variation
        random_variation = np.random.normal(0, 3)
        
        visibility = base_visibility + cloud_effect + humidity_effect + random_variation
        
        return round(max(1, min(30, visibility)), 1)
    
    def calculate_solar_radiation(self, solar_elevation, cloud_cover, humidity, visibility):
        """
        Calculate solar radiation based on conditions
        
        Args:
            solar_elevation: Sun's elevation angle
            cloud_cover: Cloud cover percentage
            humidity: Relative humidity
            visibility: Visibility in km
        
        Returns:
            Solar radiation in W/m²
        """
        if solar_elevation <= 0:
            return 0.0  # No sun = no solar radiation
        
        # Maximum possible solar radiation based on sun position
        max_radiation = 1200 * np.sin(np.radians(solar_elevation))
        
        # Reduction due to clouds (clouds block sunlight)
        cloud_reduction = 1 - (cloud_cover / 100) * 0.85
        
        # Reduction due to humidity (water vapor absorbs radiation)
        humidity_reduction = 1 - (humidity / 100) * 0.15
        
        # Reduction due to low visibility (atmospheric particles)
        visibility_reduction = min(1.0, visibility / 20)
        
        # Calculate final radiation
        solar_radiation = max_radiation * cloud_reduction * humidity_reduction * visibility_reduction
        
        # Add small random variation
        solar_radiation += np.random.normal(0, 10)
        
        return round(max(0, solar_radiation), 1)
    
    def generate_single_day(self, date):
        """
        Generate 24 hours of data for a single day
        
        Args:
            date: Date object for the day
        
        Returns:
            List of hourly data dictionaries
        """
        day_of_year = date.timetuple().tm_yday
        day_data = []
        
        # Generate seasonal factor for the day
        season_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        for hour in range(24):
            current_datetime = date.replace(hour=hour)
            
            # Generate weather conditions
            temperature = self.generate_temperature(day_of_year, hour)
            humidity = self.generate_humidity(temperature, day_of_year, hour)
            pressure = self.generate_pressure(day_of_year)
            wind_speed = self.generate_wind_speed(hour, season_factor)
            cloud_cover = self.generate_cloud_cover(humidity, day_of_year)
            visibility = self.generate_visibility(cloud_cover, humidity)
            
            # Calculate solar position and radiation
            solar_elevation = self.calculate_solar_position(day_of_year, hour)
            solar_radiation = self.calculate_solar_radiation(
                solar_elevation, cloud_cover, humidity, visibility
            )
            
            # Create data point
            data_point = {
                'datetime': current_datetime,
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'cloud_cover': cloud_cover,
                'visibility': visibility,
                'hour': hour,
                'day_of_year': day_of_year,
                'solar_elevation': round(solar_elevation, 1),
                'solar_radiation': solar_radiation
            }
            
            day_data.append(data_point)
        
        return day_data
    
    def generate_all_data(self):
        """
        Generate data for all specified days
        """
        print("\n🔄 Generating weather and solar data...")
        
        for day_num in range(self.days):
            current_date = self.start_date + timedelta(days=day_num)
            day_data = self.generate_single_day(current_date)
            self.data.extend(day_data)
            
            # Progress indicator
            if (day_num + 1) % 50 == 0 or day_num == 0:
                progress = ((day_num + 1) / self.days) * 100
                print(f"📈 Progress: {progress:.1f}% ({day_num + 1}/{self.days} days)")
        
        print(f"✅ Generated {len(self.data):,} data points")
    
    def create_dataframe(self):
        """
        Convert generated data to pandas DataFrame
        
        Returns:
            pandas DataFrame with all generated data
        """
        print("\n📊 Creating DataFrame...")
        df = pd.DataFrame(self.data)
        
        # Add some derived features
        df['month'] = df['datetime'].dt.month
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Add time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                  include_lowest=True)
        
        print(f"📋 DataFrame created with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def save_data(self, df):
        """
        Save data to CSV file
        
        Args:
            df: pandas DataFrame to save
        """
        print("\n💾 Saving data...")
        
        # Create data directory if it doesn't exist
        config.create_directories()
        
        # Save main dataset
        df.to_csv(config.TRAINING_DATA_PATH, index=False)
        print(f"✅ Main dataset saved to: {config.TRAINING_DATA_PATH}")
        
        # Save a smaller test dataset (last 7 days)
        test_data = df.tail(24 * 7)  # Last 7 days
        test_path = os.path.join(config.DATA_DIR, "test_data.csv")
        test_data.to_csv(test_path, index=False)
        print(f"✅ Test dataset saved to: {test_path}")
    
    def create_summary_stats(self, df):
        """
        Create and display summary statistics
        
        Args:
            df: pandas DataFrame with generated data
        """
        print("\n📈 DATA SUMMARY STATISTICS")
        print("=" * 50)
        
        # Basic statistics
        numeric_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 
                          'cloud_cover', 'visibility', 'solar_radiation']
        
        summary = df[numeric_columns].describe()
        print(summary.round(1))
        
        # Additional insights
        print(f"\n🌞 SOLAR RADIATION INSIGHTS:")
        print(f"   Maximum solar radiation: {df['solar_radiation'].max():.1f} W/m²")
        print(f"   Average daylight radiation: {df[df['solar_radiation'] > 0]['solar_radiation'].mean():.1f} W/m²")
        print(f"   Hours with sunlight: {len(df[df['solar_radiation'] > 0]):,} ({len(df[df['solar_radiation'] > 0])/len(df)*100:.1f}%)")
        
        print(f"\n🌡️  TEMPERATURE INSIGHTS:")
        print(f"   Hottest temperature: {df['temperature'].max():.1f}°C")
        print(f"   Coldest temperature: {df['temperature'].min():.1f}°C")
        print(f"   Average temperature: {df['temperature'].mean():.1f}°C")
        
        print(f"\n☁️  WEATHER INSIGHTS:")
        print(f"   Clearest day (lowest cloud cover): {df['cloud_cover'].min():.1f}%")
        print(f"   Cloudiest day: {df['cloud_cover'].max():.1f}%")
        print(f"   Average cloud cover: {df['cloud_cover'].mean():.1f}%")
    
    def create_visualization(self, df):
        """
        Create visualizations of the generated data
        
        Args:
            df: pandas DataFrame with generated data
        """
        print("\n📊 Creating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Generated Weather and Solar Data Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Solar radiation over time (first 30 days)
        sample_data = df.head(24 * 30)  # First 30 days
        axes[0, 0].plot(sample_data['datetime'], sample_data['solar_radiation'], 
                       color='orange', alpha=0.7, linewidth=0.5)
        axes[0, 0].set_title('Solar Radiation - First 30 Days')
        axes[0, 0].set_ylabel('Solar Radiation (W/m²)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Temperature vs Solar Radiation
        sample_for_scatter = df.sample(n=1000)  # Random sample for clarity
        scatter = axes[0, 1].scatter(sample_for_scatter['temperature'], 
                                   sample_for_scatter['solar_radiation'],
                                   c=sample_for_scatter['cloud_cover'], 
                                   cmap='Blues_r', alpha=0.6)
        axes[0, 1].set_title('Temperature vs Solar Radiation')
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Solar Radiation (W/m²)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Cloud Cover (%)')
        
        # Plot 3: Monthly average solar radiation
        monthly_solar = df.groupby('month')['solar_radiation'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[1, 0].bar(months, monthly_solar.values, color='gold', alpha=0.8)
        axes[1, 0].set_title('Average Solar Radiation by Month')
        axes[1, 0].set_ylabel('Average Solar Radiation (W/m²)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Hourly average solar radiation
        hourly_solar = df.groupby('hour')['solar_radiation'].mean()
        axes[1, 1].plot(hourly_solar.index, hourly_solar.values, 
                       marker='o', color='red', linewidth=2, markersize=4)
        axes[1, 1].set_title('Average Solar Radiation by Hour of Day')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Average Solar Radiation (W/m²)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xticks(range(0, 24, 3))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(config.OUTPUTS_DIR, "generated_data_overview.png")
        plt.savefig(plot_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✅ Visualization saved to: {plot_path}")
        plt.show()

def main():
    """
    Main function to generate all weather and solar data
    """
    print("🌞 SOLAR RADIATION DATA GENERATOR")
    print("=" * 50)
    print("This tool creates realistic weather and solar radiation data")
    print("for training your machine learning model.\n")
    
    # Create generator
    generator = WeatherSolarGenerator(days=365, start_date="2023-01-01")
    
    # Generate all data
    generator.generate_all_data()
    
    # Create DataFrame
    df = generator.create_dataframe()
    
    # Save data
    generator.save_data(df)
    
    # Create summary statistics
    generator.create_summary_stats(df)
    
    # Create visualizations
    generator.create_visualization(df)
    
    print("\n🎉 DATA GENERATION COMPLETE!")
    print("=" * 50)
    print("✅ Your training data is ready!")
    print(f"📁 Main dataset: {config.TRAINING_DATA_PATH}")
    print(f"📁 Test dataset: {os.path.join(config.DATA_DIR, 'test_data.csv')}")
    print(f"📊 Visualization: {os.path.join(config.OUTPUTS_DIR, 'generated_data_overview.png')}")
    print("\n🚀 Next step: Run the model trainer to create your AI model!")
    print("   Command: python src/model_trainer.py")

if __name__ == "__main__":
    main()