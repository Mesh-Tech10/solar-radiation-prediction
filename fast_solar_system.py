"""
Fast Solar Prediction System - No Hyperparameter Tuning
Quick training with good default parameters
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os

class FastSolarSystem:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        
        print("⚡ Fast Solar Prediction System")
        print("Quick training with optimized parameters")
        print("=" * 50)
    
    def generate_data(self, n_samples=2000):  # Smaller dataset for speed
        """Generate solar data quickly"""
        print(f"🧪 Generating {n_samples} samples...")
        
        np.random.seed(self.random_state)
        
        # Create time features
        hours = np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples]
        days = np.repeat(np.arange(n_samples // 24 + 1), 24)[:n_samples]
        
        # Create DataFrame
        data = pd.DataFrame({
            'hour': hours % 24,
            'day_of_year': (days % 365) + 1,
            'temperature': 15 + 15 * np.sin(2 * np.pi * (days % 365) / 365) + 8 * np.sin(2 * np.pi * hours / 24) + 3 * np.random.randn(n_samples),
            'humidity': np.clip(60 + 20 * np.random.randn(n_samples), 20, 95),
            'pressure': 1013 + 8 * np.random.randn(n_samples),
            'wind_speed': np.maximum(0, 5 + 3 * np.random.randn(n_samples)),
            'cloud_cover': np.clip(40 + 30 * np.random.randn(n_samples), 0, 100),
            'visibility': np.maximum(1, 20 - 0.1 * np.random.randn(n_samples)),
            'uv_index': np.maximum(0, 6 + 2 * np.random.randn(n_samples))
        })
        
        # Solar radiation calculation
        solar_elevation = np.maximum(0, 90 - 45 * np.abs(hours - 12) / 6)
        clear_sky = 1000 * np.sin(np.radians(solar_elevation))
        cloud_effect = 1 - (data['cloud_cover'] / 100) * 0.8
        seasonal_effect = 1 + 0.2 * np.sin(2 * np.pi * (data['day_of_year'] - 80) / 365)
        
        data['solar_radiation'] = np.maximum(0, 
            clear_sky * cloud_effect * seasonal_effect + 30 * np.random.randn(n_samples))
        
        print(f"✅ Generated data with {len(data.columns)} features")
        return data
    
    def train_all_models_fast(self, data):
        """Train all models with good default parameters - NO hyperparameter tuning"""
        print("\n🚀 Fast Model Training (No Hyperparameter Tuning)")
        print("-" * 50)
        
        # Prepare data
        feature_cols = ['hour', 'day_of_year', 'temperature', 'humidity', 'pressure', 
                       'wind_speed', 'cloud_cover', 'visibility', 'uv_index']
        
        X = data[feature_cols]
        y = data['solar_radiation']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"📊 Training: {X_train.shape}")
        print(f"📊 Testing: {X_test.shape}")
        
        results = {}
        
        # 1. Random Forest (good default parameters)
        print("\n🌲 Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,  # Fast but effective
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_test)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        print(f"📈 Random Forest: RMSE={rf_rmse:.1f}, MAE={rf_mae:.1f}, R²={rf_r2:.3f}")
        
        self.models['Random Forest'] = rf_model
        results['Random Forest'] = {'rmse': rf_rmse, 'mae': rf_mae, 'r2': rf_r2}
        
        # 2. XGBoost
        print("\n🚀 Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,  # Fast training
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        
        xgb_pred = xgb_model.predict(X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        
        print(f"📈 XGBoost: RMSE={xgb_rmse:.1f}, MAE={xgb_mae:.1f}, R²={xgb_r2:.3f}")
        
        self.models['XGBoost'] = xgb_model
        results['XGBoost'] = {'rmse': xgb_rmse, 'mae': xgb_mae, 'r2': xgb_r2}
        
        # 3. SVM (smaller dataset for speed)
        print("\n⚙️ Training SVM...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        svm_model = SVR(kernel='rbf', C=100, gamma='scale')  # Use 'scale' for speed
        svm_model.fit(X_train_scaled, y_train)
        
        svm_pred = svm_model.predict(X_test_scaled)
        svm_rmse = np.sqrt(mean_squared_error(y_test, svm_pred))
        svm_mae = mean_absolute_error(y_test, svm_pred)
        svm_r2 = r2_score(y_test, svm_pred)
        
        print(f"📈 SVM: RMSE={svm_rmse:.1f}, MAE={svm_mae:.1f}, R²={svm_r2:.3f}")
        
        self.models['SVM'] = svm_model
        self.scalers['SVM'] = scaler
        results['SVM'] = {'rmse': svm_rmse, 'mae': svm_mae, 'r2': svm_r2}
        
        # 4. Ensemble
        print("\n🎭 Training Ensemble...")
        ensemble = VotingRegressor([
            ('rf', rf_model),
            ('xgb', xgb_model)
        ])
        ensemble.fit(X_train, y_train)
        
        ens_pred = ensemble.predict(X_test)
        ens_rmse = np.sqrt(mean_squared_error(y_test, ens_pred))
        ens_mae = mean_absolute_error(y_test, ens_pred)
        ens_r2 = r2_score(y_test, ens_pred)
        
        print(f"📈 Ensemble: RMSE={ens_rmse:.1f}, MAE={ens_mae:.1f}, R²={ens_r2:.3f}")
        
        self.models['Ensemble'] = ensemble
        results['Ensemble'] = {'rmse': ens_rmse, 'mae': ens_mae, 'r2': ens_r2}
        
        return results
    
    def show_comparison(self, results):
        """Show performance comparison"""
        print("\n🏆 Model Performance Comparison")
        print("=" * 50)
        print(f"{'Model':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8}")
        print("-" * 50)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<15} {metrics['rmse']:<8.1f} {metrics['mae']:<8.1f} {metrics['r2']:<8.3f}")
        
        # Find best model
        best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
        print("-" * 50)
        print(f"🥇 Best Model: {best_model}")
    
    def save_models(self):
        """Save all models"""
        print("\n💾 Saving models...")
        os.makedirs('models/saved_models', exist_ok=True)
        
        for name, model in self.models.items():
            try:
                if name == 'SVM':
                    joblib.dump({'model': model, 'scaler': self.scalers['SVM']}, 
                               f'models/saved_models/{name.lower()}_model.pkl')
                else:
                    joblib.dump(model, f'models/saved_models/{name.lower().replace(" ", "_")}_model.pkl')
                print(f"✅ {name} saved")
            except Exception as e:
                print(f"❌ {name} failed: {e}")
    
    def run_fast_pipeline(self):
        """Run complete fast pipeline"""
        print("⚡ Starting Fast Solar Prediction Pipeline")
        
        # Generate data (smaller for speed)
        data = self.generate_data(n_samples=2000)
        
        # Train models (no hyperparameter tuning)
        results = self.train_all_models_fast(data)
        
        # Show results
        self.show_comparison(results)
        
        # Save models
        self.save_models()
        
        print("\n🎉 Fast training completed in under 2 minutes!")
        print("🚀 Models are ready for predictions!")
        
        return self.models, results

def main():
    system = FastSolarSystem()
    models, results = system.run_fast_pipeline()
    return system

if __name__ == "__main__":
    system = main()