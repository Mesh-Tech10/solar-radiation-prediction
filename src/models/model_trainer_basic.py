"""
Basic Model Trainer - Without TensorFlow Dependency
Works with Random Forest and Ensemble models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

class BasicSolarTrainer:
    """Basic solar prediction trainer without TensorFlow"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.performance_metrics = {}
        
        print("🌞 Basic Solar Prediction Trainer Initialized")
        np.random.seed(random_state)
    
    def generate_sample_data(self, n_samples=5000):
        """Generate comprehensive sample data"""
        print("🧪 Generating sample solar radiation data...")
        
        # Create time index
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        data = pd.DataFrame(index=dates)
        data['datetime'] = dates
        
        # Time-based features
        data['hour'] = data.index.hour
        data['day_of_year'] = data.index.dayofyear
        data['month'] = data.index.month
        
        # Weather features with realistic patterns
        np.random.seed(self.random_state)
        
        # Temperature with daily and seasonal cycles
        data['temperature'] = (15 + 
                              15 * np.sin(2 * np.pi * data['day_of_year'] / 365) +
                              8 * np.sin(2 * np.pi * data['hour'] / 24) +
                              3 * np.random.randn(n_samples))
        
        # Humidity (inverse relationship with temperature)
        data['humidity'] = np.clip(
            70 - 0.5 * data['temperature'] + 
            10 * np.sin(2 * np.pi * (data['hour'] + 12) / 24) +
            8 * np.random.randn(n_samples), 20, 95
        )
        
        # Other weather variables
        data['pressure'] = 1013 + 8 * np.random.randn(n_samples)
        data['wind_speed'] = np.maximum(0, 5 + 3 * np.random.randn(n_samples))
        data['cloud_cover'] = np.clip(40 + 30 * np.random.randn(n_samples), 0, 100)
        data['visibility'] = np.maximum(1, 20 - 0.1 * data['cloud_cover'] + 3 * np.random.randn(n_samples))
        
        # Solar position (simplified)
        data['solar_zenith'] = np.maximum(0, 90 - 45 * np.abs(data['hour'] - 12) / 6)
        data['solar_elevation'] = 90 - data['solar_zenith']
        
        # Clear sky index
        data['clear_sky_index'] = 0.7 + 0.3 * (1 - data['cloud_cover'] / 100)
        
        # Interaction features
        data['temp_humidity_interaction'] = data['temperature'] * data['humidity']
        data['solar_clearness'] = data['solar_elevation'] * (1 - data['cloud_cover'] / 100)
        
        # Solar radiation (target) - realistic model
        clear_sky_radiation = 1000 * np.maximum(0, np.sin(np.radians(data['solar_elevation'])))
        cloud_attenuation = 1 - (data['cloud_cover'] / 100) * 0.8
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (data['day_of_year'] - 80) / 365)
        
        data['solar_radiation'] = np.maximum(0, 
            clear_sky_radiation * cloud_attenuation * seasonal_factor +
            30 * np.random.randn(n_samples))
        
        print(f"✅ Generated {n_samples} samples with realistic patterns")
        print(f"📊 Features: {len(data.columns)} columns")
        
        return data
    
    def prepare_data(self, data, target_col='solar_radiation'):
        """Prepare features and target"""
        print("🔧 Preparing data...")
        
        # Select numeric features (excluding target and datetime)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        X = data[feature_cols]
        y = data[target_col]
        
        print(f"📊 Features: {len(feature_cols)}")
        print(f"📊 Target: {target_col}")
        print(f"📊 Samples: {len(X)}")
        
        return X, y, feature_cols
    
    def split_data(self, X, y, train_size=0.7, val_size=0.15):
        """Split data maintaining temporal order"""
        print("✂️ Splitting data...")
        
        n_samples = len(X)
        n_train = int(train_size * n_samples)
        n_val = int(val_size * n_samples)
        
        X_train = X.iloc[:n_train]
        y_train = y.iloc[:n_train]
        X_val = X.iloc[n_train:n_train+n_val]
        y_val = y.iloc[n_train:n_train+n_val]
        X_test = X.iloc[n_train+n_val:]
        y_test = y.iloc[n_train+n_val:]
        
        print(f"📊 Training: {X_train.shape}")
        print(f"📊 Validation: {X_val.shape}")
        print(f"📊 Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest...")
        
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        self.models['Random Forest'] = rf_model
        
        print("✅ Random Forest training completed")
        return rf_model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model if available"""
        if not XGBOOST_AVAILABLE:
            print("⚠️ XGBoost not available, skipping...")
            return None
        
        print("\n🚀 Training XGBoost...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        print("✅ XGBoost training completed")
        return xgb_model
    
    def train_svm(self, X_train, y_train):
        """Train SVM model"""
        print("\n⚙️ Training SVM...")
        
        # Scale data for SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        svm_model = SVR(kernel='rbf', C=100, gamma=0.1)
        svm_model.fit(X_train_scaled, y_train)
        
        # Store both model and scaler
        self.models['SVM'] = {'model': svm_model, 'scaler': scaler}
        
        print("✅ SVM training completed")
        return svm_model
    
    def train_ensemble(self, X_train, y_train):
        """Train ensemble model"""
        print("\n🎭 Training Ensemble...")
        
        # Prepare models for ensemble
        ensemble_models = []
        
        if 'Random Forest' in self.models:
            ensemble_models.append(('rf', self.models['Random Forest']))
        
        if 'XGBoost' in self.models:
            ensemble_models.append(('xgb', self.models['XGBoost']))
        
        if len(ensemble_models) >= 2:
            voting_ensemble = VotingRegressor(ensemble_models)
            voting_ensemble.fit(X_train, y_train)
            self.models['Ensemble'] = voting_ensemble
            
            print("✅ Ensemble training completed")
            return voting_ensemble
        else:
            print("⚠️ Need at least 2 models for ensemble, skipping...")
            return None
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate a single model"""
        try:
            if model_name == 'SVM':
                # SVM needs scaled data
                X_test_scaled = model['scaler'].transform(X_test)
                predictions = model['model'].predict(X_test_scaled)
            else:
                predictions = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-6))) * 100
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
            
            print(f"📈 {model_name}:")
            print(f"   RMSE: {rmse:.2f} W/m²")
            print(f"   MAE: {mae:.2f} W/m²")
            print(f"   R²: {r2:.3f}")
            print(f"   MAPE: {mape:.1f}%")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Evaluation failed for {model_name}: {e}")
            return {'error': str(e)}
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n📊 Evaluating All Models...")
        print("=" * 50)
        
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model_name, model, X_test, y_test)
            self.performance_metrics[model_name] = metrics
        
        return self.performance_metrics
    
    def compare_models(self):
        """Compare model performance"""
        print("\n🏆 Model Comparison...")
        print("=" * 50)
        
        if not self.performance_metrics:
            print("❌ No performance metrics available")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.performance_metrics.items():
            if 'error' not in metrics:
                comparison_data.append({
                    'Model': model_name,
                    'RMSE (W/m²)': metrics['rmse'],
                    'MAE (W/m²)': metrics['mae'],
                    'R² Score': metrics['r2'],
                    'MAPE (%)': metrics['mape']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('RMSE (W/m²)')
            
            print("\n📋 Performance Comparison:")
            print(comparison_df.to_string(index=False, float_format='%.2f'))
            
            # Best model
            best_model = comparison_df.iloc[0]['Model']
            best_rmse = comparison_df.iloc[0]['RMSE (W/m²)']
            print(f"\n🏆 Best Model: {best_model} (RMSE: {best_rmse:.2f} W/m²)")
            
            return comparison_df
    
    def save_models(self):
        """Save all trained models"""
        print("\n💾 Saving models...")
        
        # Create models directory
        os.makedirs('models/saved_models', exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                filepath = f'models/saved_models/{model_name.lower().replace(" ", "_")}_model.pkl'
                joblib.dump(model, filepath)
                print(f"✅ {model_name} saved to {filepath}")
            except Exception as e:
                print(f"❌ Failed to save {model_name}: {e}")
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("🌞 Starting Solar Prediction Training Pipeline")
        print("=" * 60)
        
        # Generate data
        data = self.generate_sample_data(n_samples=5000)
        
        # Prepare data
        X, y, feature_cols = self.prepare_data(data)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Train models
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_svm(X_train, y_train)
        self.train_ensemble(X_train, y_train)
        
        # Evaluate models
        self.evaluate_all_models(X_test, y_test)
        
        # Compare models
        comparison = self.compare_models()
        
        # Save models
        self.save_models()
        
        print("\n🎉 Training Pipeline Complete!")
        print("=" * 60)
        print(f"🎯 Models trained: {len(self.models)}")
        print("📊 Check models/saved_models/ for saved models")
        
        return self.models, self.performance_metrics

def main():
    """Main function"""
    trainer = BasicSolarTrainer(random_state=42)
    models, metrics = trainer.run_full_pipeline()

if __name__ == "__main__":
    main()