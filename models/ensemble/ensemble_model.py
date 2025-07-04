"""
Ensemble Model for Solar Radiation Prediction
=============================================

Exact implementation as described in the project document:
- VotingRegressor with Random Forest, XGBoost, and SVM
- Stacking ensemble methods
- Weighted ensemble combinations

Author: Meshwa Patel
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# XGBoost import
try:
    import xgboost as xgb
    print("✅ XGBoost available")
except ImportError:
    print("❌ XGBoost not installed. Install with: pip install xgboost")
    xgb = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import config
except ImportError:
    print("Warning: Could not import config. Using default settings.")
    class Config:
        ENSEMBLE_MODEL_PATH = "models/saved_models/ensemble_model.pkl"
        RANDOM_STATE = 42
    config = Config()

class EnsembleSolarPredictor:
    """
    Ensemble model for solar radiation prediction
    Implements multiple ensemble strategies from the project document
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Individual models
        self.rf_model = None
        self.xgb_model = None
        self.svm_model = None
        
        # Ensemble models
        self.voting_ensemble = None
        self.stacking_ensemble = None
        self.weighted_ensemble = None
        
        # Model weights (learned from validation)
        self.model_weights = None
        self.feature_importance_ = None
        self.cv_scores_ = {}
        
    def _create_base_models(self):
        """Create base models as specified in the document"""
        print("🧱 Creating base models...")
        
        # Random Forest (exact parameters from document)
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # XGBoost (if available)
        if xgb is not None:
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            print("⚠️ XGBoost not available, using Gradient Boosting instead")
            from sklearn.ensemble import GradientBoostingRegressor
            self.xgb_model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                random_state=self.random_state
            )
        
        # SVM (exact parameters from document)
        self.svm_model = SVR(
            kernel='rbf',
            C=100,
            gamma=0.1
        )
        
        print("✅ Base models created")
        
    def _create_voting_ensemble(self):
        """Create voting ensemble as specified in document"""
        print("🗳️ Creating voting ensemble...")
        
        # Create ensemble of multiple models (exact from document)
        self.voting_ensemble = VotingRegressor([
            ('rf', self.rf_model),
            ('xgb', self.xgb_model),
            ('svr', self.svm_model)
        ])
        
        print("✅ Voting ensemble created")
        
    def _create_stacking_ensemble(self, X_train, y_train):
        """Create stacking ensemble with meta-learner"""
        print("🥞 Creating stacking ensemble...")
        
        from sklearn.model_selection import cross_val_predict
        
        # Generate meta-features using cross-validation
        meta_features = []
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Get predictions from each base model
        rf_pred = cross_val_predict(self.rf_model, X_train, y_train, cv=tscv)
        xgb_pred = cross_val_predict(self.xgb_model, X_train, y_train, cv=tscv)
        svm_pred = cross_val_predict(self.svm_model, X_train, y_train, cv=tscv)
        
        # Combine meta-features
        meta_X = np.column_stack([rf_pred, xgb_pred, svm_pred])
        
        # Train meta-learner (Ridge regression for stability)
        self.stacking_ensemble = Ridge(alpha=1.0, random_state=self.random_state)
        self.stacking_ensemble.fit(meta_X, y_train)
        
        print("✅ Stacking ensemble created")
        
    def _learn_model_weights(self, X_val, y_val):
        """Learn optimal weights for weighted ensemble"""
        print("⚖️ Learning model weights...")
        
        # Get predictions from individual models
        rf_pred = self.rf_model.predict(X_val)
        xgb_pred = self.xgb_model.predict(X_val)
        svm_pred = self.svm_model.predict(X_val)
        
        # Calculate individual model errors
        rf_mse = mean_squared_error(y_val, rf_pred)
        xgb_mse = mean_squared_error(y_val, xgb_pred)
        svm_mse = mean_squared_error(y_val, svm_pred)
        
        # Calculate weights (inverse of MSE, normalized)
        mse_values = np.array([rf_mse, xgb_mse, svm_mse])
        weights = 1 / (mse_values + 1e-6)  # Add small epsilon to avoid division by zero
        weights = weights / weights.sum()  # Normalize
        
        self.model_weights = {
            'rf': weights[0],
            'xgb': weights[1],
            'svm': weights[2]
        }
        
        print(f"📊 Learned weights:")
        print(f"   Random Forest: {self.model_weights['rf']:.3f}")
        print(f"   XGBoost: {self.model_weights['xgb']:.3f}")
        print(f"   SVM: {self.model_weights['svm']:.3f}")
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all ensemble models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        print("🎭 Training Ensemble Models...")
        print("=" * 50)
        print(f"📊 Training data shape: {X_train.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Create base models
        self._create_base_models()
        
        # Train individual models
        print("\n🔄 Training individual models...")
        
        print("   Training Random Forest...")
        self.rf_model.fit(X_train_scaled, y_train)
        
        print("   Training XGBoost...")
        self.xgb_model.fit(X_train_scaled, y_train)
        
        print("   Training SVM...")
        self.svm_model.fit(X_train_scaled, y_train)
        
        # Create and train ensemble models
        self._create_voting_ensemble()
        
        print("\n🗳️ Training voting ensemble...")
        self.voting_ensemble.fit(X_train_scaled, y_train)
        
        # Create stacking ensemble
        self._create_stacking_ensemble(X_train_scaled, y_train)
        
        # Learn weights for weighted ensemble (if validation data available)
        if X_val is not None and y_val is not None:
            # Train individual models on validation set for weight learning
            self._learn_model_weights(X_val_scaled, y_val)
        else:
            # Use equal weights if no validation data
            self.model_weights = {'rf': 1/3, 'xgb': 1/3, 'svm': 1/3}
        
        # Perform cross-validation
        self._cross_validation(X_train_scaled, y_train)
        
        # Calculate feature importance
        self._calculate_feature_importance(X_train.columns)
        
        print("\n✅ Ensemble training completed!")
        return self
        
    def _cross_validation(self, X_train, y_train):
        """Perform cross-validation for all models"""
        print("\n📊 Performing cross-validation...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        models = {
            'Random Forest': self.rf_model,
            'XGBoost': self.xgb_model,
            'SVM': self.svm_model,
            'Voting Ensemble': self.voting_ensemble
        }
        
        for name, model in models.items():
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=tscv, scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores)
            self.cv_scores_[name] = cv_rmse
            
            print(f"   {name}: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f} RMSE")
    
    def _calculate_feature_importance(self, feature_names):
        """Calculate ensemble feature importance"""
        # Get feature importance from tree-based models
        rf_importance = self.rf_model.feature_importances_
        
        if hasattr(self.xgb_model, 'feature_importances_'):
            xgb_importance = self.xgb_model.feature_importances_
        else:
            # For non-tree models, use zero importance
            xgb_importance = np.zeros_like(rf_importance)
        
        # SVM doesn't have feature importance, so we'll exclude it
        # Weighted average of tree-based models
        weights = np.array([self.model_weights['rf'], self.model_weights['xgb']])
        weights = weights / weights.sum()  # Normalize
        
        ensemble_importance = (weights[0] * rf_importance + 
                             weights[1] * xgb_importance)
        
        self.feature_importance_ = pd.DataFrame({
            'feature': feature_names,
            'importance': ensemble_importance
        }).sort_values('importance', ascending=False)
    
    def predict(self, X, method='voting'):
        """
        Make predictions using specified ensemble method
        
        Args:
            X: Features
            method: Ensemble method ('voting', 'stacking', 'weighted', 'best')
        
        Returns:
            Predictions
        """
        if self.voting_ensemble is None:
            raise ValueError("Models not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if method == 'voting':
            return self.voting_ensemble.predict(X_scaled)
        
        elif method == 'stacking':
            # Get base model predictions
            rf_pred = self.rf_model.predict(X_scaled)
            xgb_pred = self.xgb_model.predict(X_scaled)
            svm_pred = self.svm_model.predict(X_scaled)
            
            # Create meta-features
            meta_X = np.column_stack([rf_pred, xgb_pred, svm_pred])
            
            # Get stacking prediction
            return self.stacking_ensemble.predict(meta_X)
        
        elif method == 'weighted':
            # Get individual predictions
            rf_pred = self.rf_model.predict(X_scaled)
            xgb_pred = self.xgb_model.predict(X_scaled)
            svm_pred = self.svm_model.predict(X_scaled)
            
            # Weighted combination
            weighted_pred = (self.model_weights['rf'] * rf_pred +
                           self.model_weights['xgb'] * xgb_pred +
                           self.model_weights['svm'] * svm_pred)
            
            return weighted_pred
        
        elif method == 'best':
            # Use the method with best cross-validation score
            best_method = min(self.cv_scores_, key=lambda x: self.cv_scores_[x].mean())
            if 'Voting' in best_method:
                return self.predict(X, 'voting')
            elif 'Random Forest' in best_method:
                return self.rf_model.predict(X_scaled)
            elif 'XGBoost' in best_method:
                return self.xgb_model.predict(X_scaled)
            else:
                return self.svm_model.predict(X_scaled)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict_all_methods(self, X):
        """Get predictions from all ensemble methods"""
        predictions = {}
        
        predictions['Individual_RF'] = self.rf_model.predict(self.scaler.transform(X))
        predictions['Individual_XGB'] = self.xgb_model.predict(self.scaler.transform(X))
        predictions['Individual_SVM'] = self.svm_model.predict(self.scaler.transform(X))
        predictions['Voting'] = self.predict(X, 'voting')
        predictions['Stacking'] = self.predict(X, 'stacking')
        predictions['Weighted'] = self.predict(X, 'weighted')
        
        return predictions
    
    def evaluate(self, X_test, y_test, methods=['voting', 'stacking', 'weighted']):
        """
        Comprehensive evaluation of all ensemble methods
        
        Args:
            X_test: Test features
            y_test: Test targets
            methods: List of methods to evaluate
        
        Returns:
            Dictionary with evaluation metrics for each method
        """
        print("📊 Evaluating Ensemble Models...")
        print("=" * 50)
        
        results = {}
        
        # Evaluate each method
        for method in methods:
            predictions = self.predict(X_test, method)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-6))) * 100
            
            results[method] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
            
            print(f"📈 {method.title()} Ensemble:")
            print(f"   RMSE: {rmse:.2f} W/m²")
            print(f"   MAE: {mae:.2f} W/m²")
            print(f"   R²: {r2:.3f}")
            print(f"   MAPE: {mape:.1f}%")
            print()
        
        # Find best method
        best_method = min(results.keys(), key=lambda x: results[x]['rmse'])
        print(f"🏆 Best method: {best_method.title()} (RMSE: {results[best_method]['rmse']:.2f})")
        
        return results
    
    def plot_model_comparison(self, X_test, y_test, save_path=None):
        """Plot comparison of all models"""
        predictions = self.predict_all_methods(X_test)
        
        n_models = len(predictions)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, (name, pred) in enumerate(predictions.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(y_test, pred, alpha=0.6, s=20)
            ax.plot([0, max(y_test)], [0, max(y_test)], 'r--', lw=2)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            
            ax.set_xlabel('Actual Solar Radiation (W/m²)')
            ax.set_ylabel('Predicted Solar Radiation (W/m²)')
            ax.set_title(f'{name}\nRMSE: {rmse:.1f}, R²: {r2:.3f}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(predictions), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, top_n=15, save_path=None):
        """Plot ensemble feature importance"""
        if self.feature_importance_ is None:
            raise ValueError("Feature importance not calculated.")
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance_.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - Ensemble Model')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print top features
        print(f"\n🔝 Top {top_n} Most Important Features:")
        for idx, row in top_features.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
    
    def save_model(self, filepath=None):
        """Save all ensemble models"""
        if filepath is None:
            filepath = getattr(config, 'ENSEMBLE_MODEL_PATH', 'models/saved_models/ensemble_model.pkl')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'svm_model': self.svm_model,
            'voting_ensemble': self.voting_ensemble,
            'stacking_ensemble': self.stacking_ensemble,
            'scaler': self.scaler,
            'model_weights': self.model_weights,
            'feature_importance': self.feature_importance_,
            'cv_scores': self.cv_scores_
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Ensemble models saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath=None):
        """Load saved ensemble models"""
        if filepath is None:
            filepath = getattr(config, 'ENSEMBLE_MODEL_PATH', 'models/saved_models/ensemble_model.pkl')
        
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls()
        instance.rf_model = model_data['rf_model']
        instance.xgb_model = model_data['xgb_model']
        instance.svm_model = model_data['svm_model']
        instance.voting_ensemble = model_data['voting_ensemble']
        instance.stacking_ensemble = model_data['stacking_ensemble']
        instance.scaler = model_data['scaler']
        instance.model_weights = model_data.get('model_weights')
        instance.feature_importance_ = model_data.get('feature_importance')
        instance.cv_scores_ = model_data.get('cv_scores', {})
        
        print(f"📂 Ensemble models loaded from: {filepath}")
        return instance

def train_ensemble_model(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
    """
    Main function to train and evaluate ensemble model
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        X_test, y_test: Test data (optional)
    
    Returns:
        Trained EnsembleSolarPredictor instance
    """
    print("🎭 Starting Ensemble Model Training Pipeline...")
    print("=" * 60)
    
    # Initialize ensemble model
    ensemble_model = EnsembleSolarPredictor(random_state=getattr(config, 'RANDOM_STATE', 42))
    
    # Train ensemble
    ensemble_model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        results = ensemble_model.evaluate(X_test, y_test)
        
        # Create visualizations
        ensemble_model.plot_model_comparison(X_test, y_test)
        ensemble_model.plot_feature_importance(top_n=15)
    
    # Save ensemble models
    ensemble_model.save_model()
    
    print("=" * 60)
    print("✅ Ensemble Model Training Complete!")
    
    return ensemble_model

if __name__ == "__main__":
    # Test with sample data
    print("🧪 Testing Ensemble Model with sample data...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 2000
    n_features = 15
    
    # Create sample features with names
    feature_names = [
        'temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover',
        'visibility', 'solar_zenith', 'solar_azimuth', 'clear_sky_index',
        'hour', 'day_of_year', 'month', 'temp_humidity_interaction',
        'pressure_altitude', 'atmospheric_turbidity'
    ]
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    
    # Create sample target with realistic relationships
    y = (100 + 
         50 * X['temperature'] + 
         -30 * X['cloud_cover'] + 
         40 * np.sin(X['solar_zenith']) +
         20 * X['clear_sky_index'] +
         15 * np.cos(X['hour'] * 2 * np.pi / 24) +
         10 * np.random.randn(n_samples))
    y = np.maximum(y, 0)  # Solar radiation can't be negative
    
    # Split data
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Train ensemble model
    model = train_ensemble_model(X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("\n🎉 Sample test completed successfully!")
    print("📝 Use this script with your actual solar radiation data.")
    print("💡 The ensemble combines Random Forest, XGBoost, and SVM models.")
    print("🏆 Expected performance improvements over individual models.")