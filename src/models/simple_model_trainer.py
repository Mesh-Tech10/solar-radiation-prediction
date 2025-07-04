"""
Simplified Model Trainer - No External Dependencies
==================================================

Basic version that works with just scikit-learn and pandas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

class SimpleSolarTrainer:
    """Simplified solar prediction trainer"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def generate_sample_data(self, n_samples=2000):
        """Generate sample solar radiation data"""
        print("🧪 Generating sample data...")
        
        np.random.seed(self.random_state)
        
        # Create time-based features
        hours = np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples]
        days = np.repeat(np.arange(n_samples // 24 + 1), 24)[:n_samples]
        
        # Generate realistic weather patterns
        data = pd.DataFrame({
            'hour': hours,
            'day_of_year': (days % 365) + 1,
            'temperature': 20 + 15 * np.sin(2 * np.pi * days / 365) + 
                          8 * np.sin(2 * np.pi * hours / 24) + 
                          3 * np.random.randn(n_samples),
            'humidity': np.clip(60 + 20 * np.random.randn(n_samples), 20, 95),
            'pressure': 1013 + 10 * np.random.randn(n_samples),
            'wind_speed': np.maximum(0, 5 + 3 * np.random.randn(n_samples)),
            'cloud_cover': np.clip(40 + 30 * np.random.randn(n_samples), 0, 100),
            'visibility': np.maximum(1, 15 + 5 * np.random.randn(n_samples))
        })
        
        # Calculate solar radiation (target)
        solar_elevation = np.maximum(0, 90 - 45 * np.abs(hours - 12) / 6)
        clear_sky = 1000 * np.sin(np.radians(solar_elevation))
        cloud_effect = 1 - (data['cloud_cover'] / 100) * 0.8
        seasonal_effect = 1 + 0.2 * np.sin(2 * np.pi * (data['day_of_year'] - 80) / 365)
        
        data['solar_radiation'] = np.maximum(0, 
            clear_sky * cloud_effect * seasonal_effect + 
            30 * np.random.randn(n_samples))
        
        print(f"✅ Generated {n_samples} samples")
        return data
    
    def train_model(self, data):
        """Train Random Forest model"""
        print("🌲 Training Random Forest model...")
        
        # Prepare features and target
        feature_cols = ['hour', 'day_of_year', 'temperature', 'humidity', 
                       'pressure', 'wind_speed', 'cloud_cover', 'visibility']
        
        X = data[feature_cols]
        y = data['solar_radiation']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        print(f"📊 Training set: {X_train.shape}")
        print(f"📊 Test set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"📈 Model Performance:")
        print(f"   RMSE: {rmse:.2f} W/m²")
        print(f"   MAE: {mae:.2f} W/m²")
        print(f"   R²: {r2:.3f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Feature Importance:")
        for _, row in importance.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_importance': importance
        }
    
    def save_model(self, filepath='models/saved_models/simple_solar_model.pkl'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def main():
    """Main function"""
    print("🌞 Simple Solar Prediction Model Trainer")
    print("=" * 50)
    
    # Initialize trainer
    trainer = SimpleSolarTrainer()
    
    # Generate sample data
    data = trainer.generate_sample_data(n_samples=5000)
    
    # Train model
    results = trainer.train_model(data)
    
    # Save model
    trainer.save_model()
    
    print("\n✅ Training completed successfully!")
    print("🎯 Model ready for predictions!")

if __name__ == "__main__":
    main()