"""
Random Forest Model for Solar Radiation Prediction
==================================================

Exact implementation as described in the project document with:
- Hyperparameter tuning using GridSearchCV
- Feature importance analysis
- Cross-validation
- Model persistence

Author: Meshwa Patel
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import config
except ImportError:
    print("Warning: Could not import config. Using default settings.")
    class Config:
        MODEL_PATH = "models/saved_models/random_forest_model.pkl"
        RANDOM_STATE = 42
    config = Config()

class RandomForestSolarPredictor:
    """
    Random Forest model for solar radiation prediction
    Implements exact specifications from the project document
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance_ = None
        self.best_params_ = None
        self.cv_scores_ = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None, tune_hyperparameters=True):
        """
        Train Random Forest model with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        print("🌲 Training Random Forest Model...")
        print(f"📊 Training data shape: {X_train.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if tune_hyperparameters:
            print("🔧 Performing hyperparameter tuning...")
            self.model = self._hyperparameter_tuning(X_train_scaled, y_train)
        else:
            # Use default parameters from document
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
        
        # Perform cross-validation
        self._cross_validation(X_train_scaled, y_train)
        
        # Calculate feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Validation performance
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_predictions = self.model.predict(X_val_scaled)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            val_mae = mean_absolute_error(y_val, val_predictions)
            val_r2 = r2_score(y_val, val_predictions)
            
            print(f"📈 Validation Performance:")
            print(f"   RMSE: {val_rmse:.2f} W/m²")
            print(f"   MAE: {val_mae:.2f} W/m²")
            print(f"   R²: {val_r2:.3f}")
        
        print("✅ Random Forest training completed!")
        return self
    
    def _hyperparameter_tuning(self, X_train, y_train):
        """
        Hyperparameter tuning using GridSearchCV
        Exact parameters from the project document
        """
        # Hyperparameter grid from document
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Create base model
        rf_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Grid search
        grid_search = GridSearchCV(
            rf_model, 
            param_grid, 
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("🔍 Running grid search (this may take a while)...")
        grid_search.fit(X_train, y_train)
        
        self.best_params_ = grid_search.best_params_
        print(f"🎯 Best parameters: {self.best_params_}")
        print(f"🎯 Best CV score: {-grid_search.best_score_:.2f}")
        
        return grid_search.best_estimator_
    
    def _cross_validation(self, X_train, y_train):
        """Perform cross-validation analysis"""
        print("📊 Performing cross-validation...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Calculate CV scores
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=tscv, scoring='neg_mean_squared_error'
        )
        
        cv_rmse_scores = np.sqrt(-cv_scores)
        self.cv_scores_ = cv_rmse_scores
        
        print(f"📈 Cross-validation RMSE: {cv_rmse_scores.mean():.2f} ± {cv_rmse_scores.std():.2f} W/m²")
        
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_with_uncertainty(self, X, n_estimators_subset=None):
        """
        Predict with uncertainty estimation using tree predictions
        
        Args:
            X: Features
            n_estimators_subset: Number of trees to use for uncertainty estimation
        
        Returns:
            predictions, lower_bound, upper_bound
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from individual trees
        tree_predictions = np.array([
            tree.predict(X_scaled) for tree in self.model.estimators_
        ])
        
        if n_estimators_subset:
            tree_predictions = tree_predictions[:n_estimators_subset]
        
        # Calculate statistics
        mean_pred = np.mean(tree_predictions, axis=0)
        std_pred = np.std(tree_predictions, axis=0)
        
        # 95% confidence interval
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        return mean_pred, lower_bound, upper_bound
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("📊 Evaluating Random Forest model...")
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-6))) * 100
        
        # Additional metrics
        max_error = np.max(np.abs(y_test - predictions))
        median_abs_error = np.median(np.abs(y_test - predictions))
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'max_error': max_error,
            'median_abs_error': median_abs_error,
            'n_samples': len(y_test)
        }
        
        print(f"📈 Random Forest Performance:")
        print(f"   RMSE: {rmse:.2f} W/m²")
        print(f"   MAE: {mae:.2f} W/m²")
        print(f"   R²: {r2:.3f}")
        print(f"   MAPE: {mape:.1f}%")
        
        return metrics
    
    def plot_feature_importance(self, top_n=15, save_path=None):
        """Plot feature importance"""
        if self.feature_importance_ is None:
            raise ValueError("Model not trained or feature importance not calculated.")
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance_.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - Random Forest')
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
    
    def plot_predictions(self, X_test, y_test, save_path=None):
        """Plot prediction vs actual values"""
        predictions = self.predict(X_test)
        
        plt.figure(figsize=(12, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, predictions, alpha=0.6, color='blue')
        plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', lw=2)
        plt.xlabel('Actual Solar Radiation (W/m²)')
        plt.ylabel('Predicted Solar Radiation (W/m²)')
        plt.title('Random Forest: Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(1, 2, 2)
        residuals = y_test - predictions
        plt.scatter(predictions, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Solar Radiation (W/m²)')
        plt.ylabel('Residuals (W/m²)')
        plt.title('Random Forest: Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = getattr(config, 'MODEL_PATH', 'models/saved_models/random_forest_model.pkl')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance_,
            'best_params': self.best_params_,
            'cv_scores': self.cv_scores_
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath=None):
        """Load a saved model"""
        if filepath is None:
            filepath = getattr(config, 'MODEL_PATH', 'models/saved_models/random_forest_model.pkl')
        
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls()
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_importance_ = model_data.get('feature_importance')
        instance.best_params_ = model_data.get('best_params')
        instance.cv_scores_ = model_data.get('cv_scores')
        
        print(f"📂 Model loaded from: {filepath}")
        return instance

def train_random_forest_model(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
    """
    Main function to train and evaluate Random Forest model
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        X_test, y_test: Test data (optional)
    
    Returns:
        Trained RandomForestSolarPredictor instance
    """
    print("🌲 Starting Random Forest Model Training Pipeline...")
    print("=" * 60)
    
    # Initialize model
    rf_model = RandomForestSolarPredictor(random_state=getattr(config, 'RANDOM_STATE', 42))
    
    # Train model
    rf_model.train(X_train, y_train, X_val, y_val, tune_hyperparameters=True)
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        metrics = rf_model.evaluate(X_test, y_test)
        
        # Create visualizations
        rf_model.plot_feature_importance(top_n=15)
        rf_model.plot_predictions(X_test, y_test)
    
    # Save model
    rf_model.save_model()
    
    print("=" * 60)
    print("✅ Random Forest Model Training Complete!")
    
    return rf_model

if __name__ == "__main__":
    # Test with sample data
    print("🧪 Testing Random Forest Model with sample data...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create sample features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create sample target (solar radiation)
    y = (100 + 50 * X['feature_0'] + 30 * X['feature_1'] + 
         20 * np.sin(X['feature_2']) + 10 * np.random.randn(n_samples))
    y = np.maximum(y, 0)  # Solar radiation can't be negative
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Train model
    model = train_random_forest_model(X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("\n🎉 Sample test completed successfully!")
    print("📝 Use this script with your actual solar radiation data.")