"""
Advanced Feature Engineering for Solar Radiation Prediction
Exactly as described in the project document

python
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pvlib
from astral import LocationInfo
from astral.sun import sun
import math

class SolarFeatureEngineering:
    """
    Advanced feature engineering for solar radiation prediction
    Implements all features mentioned in the project document
    """
    
    def __init__(self):
        self.location_cache = {}
    
    def engineer_solar_features(self, df):
        """
        Create solar-specific features exactly as in the document
        
        Args:
            df: DataFrame with weather data and coordinates
            
        Returns:
            DataFrame with engineered features
        """
        print("🔧 Engineering solar-specific features...")
        
        # Ensure datetime column
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df.index)
        
        df = df.copy()
        
        # Solar position calculations
        df = self._calculate_solar_position(df)
        
        # Atmospheric features
        df = self._calculate_atmospheric_features(df)
        
        # Temporal features
        df = self._create_temporal_features(df)
        
        # Meteorological derivatives
        df = self._create_meteorological_derivatives(df)
        
        # Interaction features
        df = self._create_interaction_features(df)
        
        print(f"✅ Feature engineering complete. Added {len(df.columns)} total features.")
        return df
    
    def _calculate_solar_position(self, df):
        """Calculate solar zenith and azimuth angles"""
        print("   📐 Calculating solar position...")
        
        # Default coordinates if not provided
        if 'latitude' not in df.columns:
            df['latitude'] = 40.7128  # NYC default
        if 'longitude' not in df.columns:
            df['longitude'] = -74.0060
        
        solar_positions = []
        
        for idx, row in df.iterrows():
            try:
                # Calculate solar position using pvlib
                times = pd.DatetimeIndex([row['datetime']])
                solar_pos = pvlib.solarposition.get_solarposition(
                    times, row['latitude'], row['longitude']
                )
                
                zenith = solar_pos['zenith'].iloc[0]
                azimuth = solar_pos['azimuth'].iloc[0]
                elevation = solar_pos['elevation'].iloc[0]
                
                solar_positions.append({
                    'solar_zenith': max(0, zenith),
                    'solar_azimuth': azimuth,
                    'solar_elevation': max(0, elevation)
                })
                
            except Exception as e:
                # Fallback calculation
                solar_positions.append({
                    'solar_zenith': self._simple_solar_zenith(
                        row['datetime'], row['latitude']
                    ),
                    'solar_azimuth': 180,  # Default south
                    'solar_elevation': 45   # Default elevation
                })
        
        # Add to dataframe
        solar_df = pd.DataFrame(solar_positions)
        for col in solar_df.columns:
            df[col] = solar_df[col].values
        
        return df
    
    def _simple_solar_zenith(self, dt, latitude):
        """Simple solar zenith calculation fallback"""
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour + dt.minute / 60.0
        
        # Solar declination
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle
        hour_angle = 15 * (hour - 12)
        
        # Solar zenith angle
        zenith = np.arccos(
            np.sin(np.radians(declination)) * np.sin(np.radians(latitude)) +
            np.cos(np.radians(declination)) * np.cos(np.radians(latitude)) * 
            np.cos(np.radians(hour_angle))
        )
        
        return max(0, min(90, np.degrees(zenith)))
    
    def _calculate_atmospheric_features(self, df):
        """Calculate atmospheric clearness features"""
        print("   🌤️ Calculating atmospheric features...")
        
        # Clear sky GHI calculation
        df['clear_sky_ghi'] = self._calculate_clear_sky_ghi(df)
        
        # Clear sky index (if we have actual GHI measurements)
        if 'solar_radiation' in df.columns:
            df['clear_sky_index'] = df['solar_radiation'] / (df['clear_sky_ghi'] + 1e-6)
            df['clear_sky_index'] = df['clear_sky_index'].clip(0, 1.5)
        else:
            df['clear_sky_index'] = 0.7  # Default clear sky index
        
        # Atmospheric turbidity (simplified)
        if 'visibility' in df.columns:
            df['atmospheric_turbidity'] = 2.0 + (20 - df['visibility'].clip(1, 20)) / 10
        else:
            df['atmospheric_turbidity'] = 2.5  # Default turbidity
        
        # Air mass calculation
        df['air_mass'] = 1 / (np.cos(np.radians(df['solar_zenith'])) + 1e-6)
        df['air_mass'] = df['air_mass'].clip(1, 40)
        
        return df
    
    def _calculate_clear_sky_ghi(self, df):
        """Calculate clear sky Global Horizontal Irradiance"""
        # Simplified clear sky model
        solar_constant = 1361  # W/m²
        
        # Calculate extraterrestrial radiation
        day_of_year = df['datetime'].dt.dayofyear
        eccentricity_correction = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
        
        # Solar elevation factor
        elevation_rad = np.radians(90 - df['solar_zenith'])
        elevation_factor = np.sin(elevation_rad).clip(0, 1)
        
        # Atmospheric attenuation (simplified)
        if 'atmospheric_turbidity' in df.columns:
            atm_transmission = 0.7 ** (df['atmospheric_turbidity'] ** 0.678)
        else:
            atm_transmission = 0.75  # Default transmission
        
        clear_sky_ghi = (solar_constant * eccentricity_correction * 
                        elevation_factor * atm_transmission)
        
        return clear_sky_ghi.clip(0, 1400)
    
    def _create_temporal_features(self, df):
        """Create temporal features as specified in document"""
        print("   ⏰ Creating temporal features...")
        
        dt = df['datetime']
        
        # Basic temporal features
        df['hour'] = dt.dt.hour
        df['day_of_year'] = dt.dt.dayofyear
        df['month'] = dt.dt.month
        df['day_of_week'] = dt.dt.dayofweek
        df['week_of_year'] = dt.dt.isocalendar().week
        
        # Season mapping
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        df['season'] = df['month'].map(season_map)
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time of day categories
        def categorize_time(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        
        df['time_of_day'] = df['hour'].apply(categorize_time)
        
        # Daylight indicators
        df['is_daylight'] = (df['solar_elevation'] > 0).astype(int)
        df['daylight_hours'] = df.groupby(df['datetime'].dt.date)['is_daylight'].transform('sum')
        
        return df
    
    def _create_meteorological_derivatives(self, df):
        """Create meteorological derivative features"""
        print("   🌡️ Creating meteorological derivatives...")
        
        # Temperature features
        if 'temperature' in df.columns:
            # Daily temperature statistics
            daily_temp = df.groupby(df['datetime'].dt.date)['temperature']
            df['daily_temp_mean'] = daily_temp.transform('mean')
            df['daily_temp_max'] = daily_temp.transform('max')
            df['daily_temp_min'] = daily_temp.transform('min')
            df['daily_temp_range'] = df['daily_temp_max'] - df['daily_temp_min']
            df['temp_anomaly'] = df['temperature'] - df['daily_temp_mean']
        
        # Humidity features
        if 'humidity' in df.columns:
            df['humidity_comfort'] = ((df['humidity'] >= 40) & (df['humidity'] <= 60)).astype(int)
            
            # Dew point calculation (if temperature available)
            if 'temperature' in df.columns:
                df['dew_point'] = self._calculate_dew_point(df['temperature'], df['humidity'])
                df['relative_humidity_deficit'] = df['temperature'] - df['dew_point']
        
        # Pressure features
        if 'pressure' in df.columns:
            # Pressure tendency (change over time)
            df['pressure_tendency'] = df['pressure'].diff().fillna(0)
            df['pressure_anomaly'] = df['pressure'] - df['pressure'].rolling(24, min_periods=1).mean()
        
        # Wind features
        if 'wind_speed' in df.columns:
            # Wind categories
            def wind_category(speed):
                if speed < 2:
                    return 'Calm'
                elif speed < 6:
                    return 'Light'
                elif speed < 12:
                    return 'Moderate'
                else:
                    return 'Strong'
            
            df['wind_category'] = df['wind_speed'].apply(wind_category)
            
            # Wind direction features (if available)
            if 'wind_direction' in df.columns:
                df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction']))
                df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction']))
        
        # Cloud features
        if 'cloud_cover' in df.columns:
            df['sky_condition'] = pd.cut(df['cloud_cover'], 
                                       bins=[0, 10, 25, 75, 90, 100],
                                       labels=['Clear', 'Few', 'Scattered', 'Broken', 'Overcast'])
            df['cloud_opacity'] = df['cloud_cover'] / 100.0
        
        return df
    
    def _calculate_dew_point(self, temperature, humidity):
        """Calculate dew point temperature"""
        # Magnus formula approximation
        a = 17.27
        b = 237.7
        
        alpha = ((a * temperature) / (b + temperature)) + np.log(humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        
        return dew_point
    
    def _create_interaction_features(self, df):
        """Create interaction features as mentioned in document"""
        print("   🔗 Creating interaction features...")
        
        # Temperature-humidity interactions
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            df['heat_index'] = self._calculate_heat_index(df['temperature'], df['humidity'])
        
        # Pressure-altitude interactions (if altitude available)
        if 'pressure' in df.columns:
            # Assume sea level if altitude not provided
            altitude = df.get('altitude', 0)
            df['pressure_altitude'] = df['pressure'] * (1 + altitude / 8400)
        
        # Solar-atmospheric interactions
        df['solar_clearness'] = df['solar_elevation'] * (1 - df.get('cloud_opacity', 0.3))
        
        if 'visibility' in df.columns:
            df['atmospheric_clarity'] = df['visibility'] * (1 - df.get('cloud_opacity', 0.3))
        
        # Wind-temperature interactions
        if 'wind_speed' in df.columns and 'temperature' in df.columns:
            df['wind_chill_factor'] = df['wind_speed'] * (35 - df['temperature']) / 35
            df['wind_chill_factor'] = df['wind_chill_factor'].clip(-10, 10)
        
        return df
    
    def _calculate_heat_index(self, temperature, humidity):
        """Calculate heat index (feels like temperature)"""
        # Simplified heat index calculation
        T = temperature * 9/5 + 32  # Convert to Fahrenheit
        R = humidity
        
        heat_index = (0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (R * 0.094)))
        
        # Convert back to Celsius
        heat_index = (heat_index - 32) * 5/9
        
        return heat_index

def engineer_solar_features(df):
    """
    Main function to engineer solar features
    Matches the function signature from the document
    """
    engineer = SolarFeatureEngineering()
    return engineer.engineer_solar_features(df)

if __name__ == "__main__":
    # Test feature engineering
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-01-07', freq='H')
    sample_data = pd.DataFrame({
        'datetime': dates,
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 24),
        'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 24 + np.pi),
        'pressure': 1013 + 5 * np.random.randn(len(dates)),
        'wind_speed': 5 + 3 * np.random.randn(len(dates)).clip(0, None),
        'wind_direction': 180 + 90 * np.random.randn(len(dates)),
        'cloud_cover': 30 + 40 * np.random.rand(len(dates)),
        'visibility': 15 + 5 * np.random.randn(len(dates)).clip(5, None),
        'latitude': 40.7128,
        'longitude': -74.0060
    })
    
    print("🧪 Testing feature engineering...")
    engineered_data = engineer_solar_features(sample_data)
    print(f"✅ Original features: {len(sample_data.columns)}")
    print(f"✅ Engineered features: {len(engineered_data.columns)}")
    print("\n📊 New features created:")
    new_features = set(engineered_data.columns) - set(sample_data.columns)
    for feature in sorted(new_features):
        print(f"   • {feature}")