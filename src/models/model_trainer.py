"""
Unified Model Training Pipeline for Solar Radiation Prediction
==============================================================

Comprehensive training system that:
- Trains all model types (Random Forest, LSTM, XGBoost, SVM, Ensemble)
- Handles data preprocessing and feature engineering
- Performs model evaluation and comparison
- Saves all trained models
- Generates comprehensive reports

Author: Meshwa Patel
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

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

try:
    import config
    from src.data_processing.feature_engineering import engineer_solar_features
    from models.traditional.random_forest_model import RandomForestSolarPredictor
    from models.deep_learning.lstm_model import LSTMSolarPredictor
    from models.ensemble.ensemble_model import EnsembleSolarPredictor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required modules are available")
    sys.exit(1)

class SolarPredictionTrainer:
    """
    Unified training pipeline for all solar prediction models
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.performance_metrics = {}
        self.training_history = {}
        
        # Set random seeds
        np.random.seed(random_state)
        
        print("🌞 Solar Prediction Trainer Initialized")
        print(f"🎲 Random state: {random_state}")
    
    def load_and_prepare_data(self, data_path=None, target_col='solar_radiation'):
        """
        Load and prepare data for training
        
        Args:
            data_path: Path to data file
            target_col: Target column name
        
        Returns:
            Prepared features and target
        """
        print("📊 Loading and preparing data...")
        
        if data_path is None:
            # Use config path or generate sample data
            data_path = getattr(config, 'TRAINING_DATA_PATH', None)
            
        if data_path and os.path.exists(data_path):
            print(f"📂 Loading data from: {data_path}")
            data = pd.read_csv(data_path)
        else:
            print("🧪 Generating sample data for demonstration...")
            data = self._generate_comprehensive_sample_data()
        
        print(f"📋 Data shape: {data.shape}")
        print(f"📋 Columns: {list(data.columns)}")
        
        # Feature engineering
        print("🔧 Applying feature engineering...")
        if 'datetime' not in data.columns and data.index.name != 'datetime':
            # Create datetime index if not present
            data['datetime'] = pd.date_range(
                start='2023-01-01', 
                periods=len(data), 
                freq='H'
            )
        
        # Apply advanced feature engineering
        try:
            data_engineered = engineer_solar_features(data)
            print(f"✅ Feature engineering completed. New shape: {data_engineered.shape}")
        except Exception as e:
            print(f"⚠️ Feature engineering failed: {e}")
            print("Using original data...")
            data_engineered = data
        
        # Prepare features and target
        if target_col not in data_engineered.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Select numeric features (excluding target and datetime)
        numeric_cols = data_engineered.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        X = data_engineered[feature_cols]
        y = data_engineered[target_col]
        
        print(f"📊 Features: {len(feature_cols)}")
        print(f"📊 Target: {target_col}")
        print(f"📊 Samples: {len(X)}")
        
        return X, y, data_engineered
    
    def split_data(self, X, y, train_size=0.7, val_size=0.15, test_size=0.15):
        """
        Split data maintaining temporal order for time series
        
        Args:
            X: Features
            y: Target
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion
        
        Returns:
            Split datasets
        """
        print("✂️ Splitting data...")
        
        n_samples = len(X)
        n_train = int(train_size * n_samples)
        n_val = int(val_size * n_samples)
        
        # Temporal split (important for time series)
        X_train = X.iloc[:n_train]
        y_train = y.iloc[:n_train]
        
        X_val = X.iloc[n_train:n_train+n_val]
        y_val = y.iloc[n_train:n_train+n_val]
        
        X_test = X.iloc[n_train+n_val:]
        y_test = y.iloc[n_train+n_val:]
        
        print(f"📊 Training set: {X_train.shape}")
        print(f"📊 Validation set: {X_val.shape}")
        print(f"📊 Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest Model...")
        print("-" * 40)
        
        rf_model = RandomForestSolarPredictor(random_state=self.random_state)
        rf_model.train(X_train, y_train, X_val, y_val, tune_hyperparameters=True)
        
        self.models['random_forest'] = rf_model
        print("✅ Random Forest training completed")
        
        return rf_model
    
    def train_lstm(self, data_engineered, feature_cols, target_col='solar_radiation'):
        """Train LSTM model"""
        print("\n🧠 Training LSTM Model...")
        print("-" * 40)
        
        lstm_model = LSTMSolarPredictor(
            sequence_length=24,
            n_features=len(feature_cols),
            random_state=self.random_state
        )
        
        # Prepare sequences for LSTM
        X_seq, y_seq, _ = lstm_model.prepare_sequences(
            data_engineered, target_col, feature_cols
        )
        
        # Split sequences temporally
        n_samples = len(X_seq)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        X_train_seq = X_seq[:n_train]
        y_train_seq = y_seq[:n_train]
        X_val_seq = X_seq[n_train:n_train+n_val]
        y_val_seq = y_seq[n_train:n_train+n_val]
        
        # Train LSTM
        lstm_model.train(
            X_train_seq, y_train_seq, 
            X_val_seq, y_val_seq,
            epochs=100, batch_size=32
        )
        
        self.models['lstm'] = lstm_model
        print("✅ LSTM training completed")
        
        return lstm_model
    
    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Train Ensemble model"""
        print("\n🎭 Training Ensemble Model...")
        print("-" * 40)
        
        ensemble_model = EnsembleSolarPredictor(random_state=self.random_state)
        ensemble_model.train(X_train, y_train, X_val, y_val)
        
        self.models['ensemble'] = ensemble_model
        print("✅ Ensemble training completed")
        
        return ensemble_model
    
    def train_all_models(self, data_path=None, target_col='solar_radiation'):
        """
        Train all available models
        
        Args:
            data_path: Path to training data
            target_col: Target column name
        
        Returns:
            Dictionary of trained models
        """
        print("🚀 Starting Comprehensive Model Training...")
        print("=" * 60)
        
        # Load and prepare data
        X, y, data_engineered = self.load_and_prepare_data(data_path, target_col)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Store test data for final evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Train Random Forest
        try:
            self.train_random_forest(X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"❌ Random Forest training failed: {e}")
        
        # Train LSTM
        try:
            feature_cols = X.columns.tolist()
            self.train_lstm(data_engineered, feature_cols, target_col)
        except Exception as e:
            print(f"❌ LSTM training failed: {e}")
        
        # Train Ensemble
        try:
            self.train_ensemble(X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"❌ Ensemble training failed: {e}")
        
        print("\n✅ All model training completed!")
        print(f"🎯 Models trained: {list(self.models.keys())}")
        
        return self.models
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("\n📊 Evaluating All Models...")
        print("=" * 60)
        
        if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
            print("❌ No test data available. Run train_all_models first.")
            return {}
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n📈 Evaluating {name.title()}...")
            
            try:
                if name == 'lstm':
                    # LSTM needs sequence data
                    X_seq, y_seq, _ = model.prepare_sequences(
                        pd.concat([
                            pd.DataFrame(self.X_test), 
                            pd.DataFrame({'solar_radiation': self.y_test})
                        ], axis=1),
                        'solar_radiation', self.X_test.columns
                    )
                    metrics = model.evaluate(X_seq, y_seq)
                else:
                    # Traditional models
                    metrics = model.evaluate(self.X_test, self.y_test)
                
                results[name] = metrics
                
            except Exception as e:
                print(f"❌ Evaluation failed for {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.performance_metrics = results
        return results
    
    def compare_models(self):
        """Create comprehensive model comparison"""
        print("\n🏆 Model Comparison Summary...")
        print("=" * 60)
        
        if not self.performance_metrics:
            print("❌ No performance metrics available. Run evaluate_all_models first.")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.performance_metrics.items():
            if 'error' not in metrics:
                comparison_data.append({
                    'Model': model_name.title(),
                    'RMSE (W/m²)': metrics.get('rmse', 0),
                    'MAE (W/m²)': metrics.get('mae', 0),
                    'R² Score': metrics.get('r2', 0),
                    'MAPE (%)': metrics.get('mape', 0)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('RMSE (W/m²)')
            
            print("\n📋 Performance Comparison:")
            print(comparison_df.to_string(index=False, float_format='%.2f'))
            
            # Highlight best model
            best_model = comparison_df.iloc[0]['Model']
            best_rmse = comparison_df.iloc[0]['RMSE (W/m²)']
            print(f"\n🏆 Best Model: {best_model} (RMSE: {best_rmse:.2f} W/m²)")
            
            # Create visualization
            self._plot_model_comparison(comparison_df)
            
            return comparison_df
        else:
            print("❌ No valid performance metrics for comparison")
    
    def _plot_model_comparison(self, comparison_df):
        """Plot model comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RMSE comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['RMSE (W/m²)'], 
                       color='skyblue', alpha=0.8)
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE (W/m²)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['MAE (W/m²)'], 
                       color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('MAE Comparison')
        axes[0, 1].set_ylabel('MAE (W/m²)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['R² Score'], 
                       color='orange', alpha=0.8)
        axes[1, 0].set_title('R² Score Comparison')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 1].bar(comparison_df['Model'], comparison_df['MAPE (%)'], 
                       color='coral', alpha=0.8)
        axes[1, 1].set_title('MAPE Comparison')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(
            getattr(config, 'OUTPUTS_DIR', 'outputs'),
            'model_comparison.png'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {save_path}")
        
        plt.show()
    
    def save_all_models(self):
        """Save all trained models"""
        print("\n💾 Saving All Models...")
        print("-" * 30)
        
        for name, model in self.models.items():
            try:
                model.save_model()
                print(f"✅ {name.title()} model saved")
            except Exception as e:
                print(f"❌ Failed to save {name}: {e}")
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        print("\n📝 Generating Training Report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'performance_metrics': self.performance_metrics,
            'best_model': None,
            'training_summary': {}
        }
        
        # Determine best model
        if self.performance_metrics:
            best_model = min(
                self.performance_metrics.keys(),
                key=lambda x: self.performance_metrics[x].get('rmse', float('inf'))
            )
            report['best_model'] = best_model
        
        # Save report
        report_path = os.path.join(
            getattr(config, 'OUTPUTS_DIR', 'outputs'),
            'training_report.json'
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📋 Training report saved to: {report_path}")
        return report
    
    def _generate_comprehensive_sample_data(self, n_samples=5000):
        """Generate comprehensive sample data for testing"""
        print("🧪 Generating comprehensive sample data...")
        
        # Create time index
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        data = pd.DataFrame(index=dates)
        data['datetime'] = dates
        
        # Generate realistic weather patterns
        np.random.seed(self.random_state)
        
        # Base weather features
        data['hour'] = data.index.hour
        data['day_of_year'] = data.index.dayofyear
        data['month'] = data.index.month
        
        # Temperature with realistic patterns
        data['temperature'] = (15 + 
                              15 * np.sin(2 * np.pi * data['day_of_year'] / 365) +
                              8 * np.sin(2 * np.pi * data['hour'] / 24) +
                              3 * np.random.randn(n_samples))
        
        # Other weather variables
        data['humidity'] = np.clip(60 - 0.5 * data['temperature'] + 
                                  10 * np.sin(2 * np.pi * (data['hour'] + 12) / 24) +
                                  8 * np.random.randn(n_samples), 20, 95)
        
        data['pressure'] = 1013 + 8 * np.random.randn(n_samples)
        data['wind_speed'] = np.maximum(0, 5 + 3 * np.random.randn(n_samples))
        data['cloud_cover'] = np.clip(40 + 30 * np.random.randn(n_samples), 0, 100)
        data['visibility'] = np.maximum(1, 20 - 0.1 * data['cloud_cover'] + 
                                       3 * np.random.randn(n_samples))
        
        # Solar radiation (complex realistic model)
        solar_elevation = np.maximum(0, 90 - 45 * np.abs(data['hour'] - 12) / 6)
        clear_sky_radiation = 1000 * np.sin(np.radians(solar_elevation))
        cloud_attenuation = 1 - (data['cloud_cover'] / 100) * 0.8
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (data['day_of_year'] - 80) / 365)
        
        data['solar_radiation'] = np.maximum(0, 
            clear_sky_radiation * cloud_attenuation * seasonal_factor +
            30 * np.random.randn(n_samples))
        
        # Add coordinates
        data['latitude'] = 40.7128
        data['longitude'] = -74.0060
        
        print(f"✅ Generated {n_samples} samples with realistic patterns")
        return data

def main():
    """Main training pipeline"""
    print("🌞 Solar Radiation Prediction - Unified Training Pipeline")
    print("=" * 70)
    
    # Initialize trainer
    trainer = SolarPredictionTrainer(random_state=42)
    
    # Train all models
    models = trainer.train_all_models()
    
    # Evaluate models
    results = trainer.evaluate_all_models()
    
    # Compare models
    comparison = trainer.compare_models()
    
    # Save all models
    trainer.save_all_models()
    
    # Generate report
    report = trainer.generate_training_report()
    
    print("\n🎉 Training Pipeline Complete!")
    print("=" * 70)
    print(f"🎯 Models trained: {len(models)}")
    print(f"📊 Best model: {report.get('best_model', 'N/A')}")
    print("📝 Check outputs folder for detailed results and visualizations")

if __name__ == "__main__":
    main()