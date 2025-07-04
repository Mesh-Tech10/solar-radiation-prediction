"""
LSTM Deep Learning Model for Solar Radiation Prediction
=======================================================

Exact implementation as described in the project document:
- LSTM with Dropout layers
- 24-hour sequence length
- 8 features input
- Sequential architecture with 3 LSTM layers

Author: Meshwa Patel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("❌ TensorFlow not installed. Install with: pip install tensorflow")
    exit(1)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import config
except ImportError:
    print("Warning: Could not import config. Using default settings.")
    class Config:
        LSTM_MODEL_PATH = "models/saved_models/lstm_model.h5"
        RANDOM_STATE = 42
    config = Config()

class LSTMSolarPredictor:
    """
    LSTM model for solar radiation prediction
    Implements exact specifications from the project document
    """
    
    def __init__(self, sequence_length=24, n_features=8, random_state=42):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.random_state = random_state
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
    def create_lstm_model(self, sequence_length=None, n_features=None):
        """
        Create LSTM model with exact architecture from document:
        - 3 LSTM layers with 50 units each
        - Dropout layers with 0.2 rate
        - Dense output layer
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        if n_features is None:
            n_features = self.n_features
            
        print(f"🧠 Creating LSTM model...")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Number of features: {n_features}")
        
        model = Sequential()
        
        # First LSTM layer with return_sequences=True
        model.add(LSTM(50, return_sequences=True, 
                      input_shape=(sequence_length, n_features)))
        model.add(Dropout(0.2))
        
        # Second LSTM layer with return_sequences=True
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        
        # Third LSTM layer (final, no return_sequences)
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        
        # Dense output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("✅ LSTM model created successfully!")
        print("\n📋 Model Architecture:")
        model.summary()
        
        return model
    
    def prepare_sequences(self, data, target_col='solar_radiation', feature_cols=None):
        """
        Prepare time series sequences for LSTM training
        
        Args:
            data: DataFrame with time series data
            target_col: Name of target column
            feature_cols: List of feature column names
            
        Returns:
            X_sequences, y_sequences
        """
        print(f"🔄 Preparing sequences...")
        print(f"   Sequence length: {self.sequence_length}")
        
        if feature_cols is None:
            # Use all numeric columns except target
            feature_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                           if col != target_col]
        
        print(f"   Features: {feature_cols}")
        
        # Select features and target
        features = data[feature_cols].values
        target = data[target_col].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.sequence_length, len(features_scaled)):
            # Get sequence of features
            X.append(features_scaled[i-self.sequence_length:i])
            # Get corresponding target
            y.append(target[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Sequences prepared:")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        
        return X, y, feature_cols
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """
        Train LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        print("🚀 Training LSTM model...")
        print(f"📊 Training data shape: {X_train.shape}")
        
        # Create model if not exists
        if self.model is None:
            self.model = self.create_lstm_model(
                sequence_length=X_train.shape[1],
                n_features=X_train.shape[2]
            )
        
        # Prepare callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        model_path = getattr(config, 'LSTM_MODEL_PATH', 'models/saved_models/lstm_model.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            print(f"📊 Validation data shape: {X_val.shape}")
        
        # Train model
        print(f"🎯 Training for {epochs} epochs with batch size {batch_size}")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("✅ LSTM training completed!")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_sequences(self, data, feature_cols, steps_ahead=1):
        """
        Predict multiple steps ahead
        
        Args:
            data: Input data
            feature_cols: Feature columns
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare last sequence
        features = data[feature_cols].values
        features_scaled = self.scaler.transform(features)
        
        # Get last sequence
        last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps_ahead):
            # Predict next value
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence (this is simplified - in practice you'd need to update all features)
            # For now, we just shift the sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            # You would update the last timestep with new feature values here
        
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("📊 Evaluating LSTM model...")
        
        # Make predictions
        predictions = self.predict(X_test).flatten()
        
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
        
        print(f"📈 LSTM Performance:")
        print(f"   RMSE: {rmse:.2f} W/m²")
        print(f"   MAE: {mae:.2f} W/m²")
        print(f"   R²: {r2:.3f}")
        print(f"   MAPE: {mape:.1f}%")
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MAE plot
        plt.subplot(1, 3, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        plt.subplot(1, 3, 3)
        if 'lr' in self.history.history:
            plt.plot(self.history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
        else:
            # Plot epoch vs final validation loss
            epochs = range(1, len(self.history.history['loss']) + 1)
            plt.plot(epochs, self.history.history['loss'])
            plt.title('Training Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions(self, X_test, y_test, save_path=None, sample_size=200):
        """Plot prediction vs actual values"""
        predictions = self.predict(X_test).flatten()
        
        plt.figure(figsize=(15, 5))
        
        # Scatter plot
        plt.subplot(1, 3, 1)
        idx = np.random.choice(len(y_test), min(sample_size, len(y_test)), replace=False)
        plt.scatter(y_test[idx], predictions[idx], alpha=0.6, color='blue')
        plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', lw=2)
        plt.xlabel('Actual Solar Radiation (W/m²)')
        plt.ylabel('Predicted Solar Radiation (W/m²)')
        plt.title('LSTM: Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Time series plot
        plt.subplot(1, 3, 2)
        time_window = min(500, len(y_test))
        time_idx = range(time_window)
        plt.plot(time_idx, y_test[:time_window], label='Actual', alpha=0.8)
        plt.plot(time_idx, predictions[:time_window], label='Predicted', alpha=0.8)
        plt.xlabel('Time Steps')
        plt.ylabel('Solar Radiation (W/m²)')
        plt.title('LSTM: Time Series Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(1, 3, 3)
        residuals = y_test - predictions
        plt.scatter(predictions[idx], residuals[idx], alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Solar Radiation (W/m²)')
        plt.ylabel('Residuals (W/m²)')
        plt.title('LSTM: Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Cannot save.")
        
        if filepath is None:
            filepath = getattr(config, 'LSTM_MODEL_PATH', 'models/saved_models/lstm_model.h5')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        
        # Save scaler and metadata
        import joblib
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        metadata = {
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'random_state': self.random_state
        }
        joblib.dump(metadata, scaler_path)
        
        print(f"💾 LSTM model saved to: {filepath}")
        print(f"💾 Scaler saved to: {scaler_path}")
    
    @classmethod
    def load_model(cls, filepath=None):
        """Load a saved model"""
        if filepath is None:
            filepath = getattr(config, 'LSTM_MODEL_PATH', 'models/saved_models/lstm_model.h5')
        
        # Load model
        model = tf.keras.models.load_model(filepath)
        
        # Load scaler and metadata
        import joblib
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        metadata = joblib.load(scaler_path)
        
        # Create instance
        instance = cls(
            sequence_length=metadata['sequence_length'],
            n_features=metadata['n_features'],
            random_state=metadata['random_state']
        )
        instance.model = model
        instance.scaler = metadata['scaler']
        
        print(f"📂 LSTM model loaded from: {filepath}")
        return instance

def create_sample_time_series_data(n_samples=2000, n_features=8, sequence_length=24):
    """Create sample time series data for testing"""
    print("🧪 Creating sample time series data...")
    
    # Generate time index
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # Generate features with realistic patterns
    data = pd.DataFrame(index=dates)
    
    # Time-based features
    data['hour'] = data.index.hour
    data['day_of_year'] = data.index.dayofyear
    
    # Weather-like features with temporal patterns
    np.random.seed(42)
    
    # Temperature with daily and seasonal cycles
    data['temperature'] = (20 + 
                          15 * np.sin(2 * np.pi * data['day_of_year'] / 365) +  # Seasonal
                          8 * np.sin(2 * np.pi * data['hour'] / 24) +  # Daily
                          3 * np.random.randn(n_samples))  # Random noise
    
    # Humidity (inverse relationship with temperature)
    data['humidity'] = (70 - 0.5 * data['temperature'] + 
                       10 * np.sin(2 * np.pi * (data['hour'] + 12) / 24) +
                       5 * np.random.randn(n_samples)).clip(20, 95)
    
    # Pressure with small variations
    data['pressure'] = 1013 + 10 * np.random.randn(n_samples)
    
    # Wind speed with some temporal correlation
    wind_base = 5 + 3 * np.sin(2 * np.pi * data['hour'] / 24)
    data['wind_speed'] = np.maximum(0, wind_base + 2 * np.random.randn(n_samples))
    
    # Cloud cover with persistence
    cloud_noise = np.random.randn(n_samples)
    data['cloud_cover'] = np.maximum(0, np.minimum(100, 
        40 + 20 * np.cumsum(0.1 * cloud_noise) / np.sqrt(np.arange(1, n_samples + 1))))
    
    # Visibility (inversely related to cloud cover)
    data['visibility'] = np.maximum(1, 20 - 0.1 * data['cloud_cover'] + 2 * np.random.randn(n_samples))
    
    # Solar zenith angle (simplified)
    data['solar_zenith'] = np.maximum(0, 90 - 45 * np.sin(2 * np.pi * (data['hour'] - 6) / 12))
    
    # UV index
    data['uv_index'] = np.maximum(0, 8 * np.sin(2 * np.pi * (data['hour'] - 6) / 12) * 
                                 (1 - data['cloud_cover'] / 100) + np.random.randn(n_samples))
    
    # Solar radiation (target) - complex relationship with multiple factors
    solar_base = (800 * np.maximum(0, np.sin(2 * np.pi * (data['hour'] - 6) / 12)) *  # Time of day
                 (1 - data['cloud_cover'] / 100) *  # Cloud effect
                 (1 + 0.1 * np.sin(2 * np.pi * data['day_of_year'] / 365)))  # Seasonal effect
    
    data['solar_radiation'] = np.maximum(0, solar_base + 50 * np.random.randn(n_samples))
    
    # Select the specified number of features
    feature_cols = data.columns.tolist()
    if 'solar_radiation' in feature_cols:
        feature_cols.remove('solar_radiation')
    
    if len(feature_cols) > n_features:
        feature_cols = feature_cols[:n_features]
    
    print(f"✅ Created time series data:")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {feature_cols}")
    print(f"   Target: solar_radiation")
    
    return data, feature_cols

def train_lstm_model(data, feature_cols, target_col='solar_radiation', 
                    sequence_length=24, test_size=0.2, val_size=0.2):
    """
    Main function to train and evaluate LSTM model
    
    Args:
        data: DataFrame with time series data
        feature_cols: List of feature column names
        target_col: Target column name
        sequence_length: LSTM sequence length
        test_size: Test set proportion
        val_size: Validation set proportion
    
    Returns:
        Trained LSTMSolarPredictor instance
    """
    print("🧠 Starting LSTM Model Training Pipeline...")
    print("=" * 60)
    
    # Initialize model
    lstm_model = LSTMSolarPredictor(
        sequence_length=sequence_length,
        n_features=len(feature_cols),
        random_state=getattr(config, 'RANDOM_STATE', 42)
    )
    
    # Prepare sequences
    X, y, feature_cols = lstm_model.prepare_sequences(data, target_col, feature_cols)
    
    # Split data (maintaining time order)
    n_samples = len(X)
    n_test = int(test_size * n_samples)
    n_val = int(val_size * n_samples)
    n_train = n_samples - n_test - n_val
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    print(f"📊 Data splits:")
    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Testing: {X_test.shape}")
    
    # Train model
    lstm_model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Evaluate model
    metrics = lstm_model.evaluate(X_test, y_test)
    
    # Create visualizations
    lstm_model.plot_training_history()
    lstm_model.plot_predictions(X_test, y_test)
    
    # Save model
    lstm_model.save_model()
    
    print("=" * 60)
    print("✅ LSTM Model Training Complete!")
    
    return lstm_model

if __name__ == "__main__":
    # Test with sample data
    print("🧪 Testing LSTM Model with sample data...")
    
    # Create sample time series data
    data, feature_cols = create_sample_time_series_data(
        n_samples=2000, 
        n_features=8, 
        sequence_length=24
    )
    
    # Train model
    model = train_lstm_model(
        data=data,
        feature_cols=feature_cols,
        target_col='solar_radiation',
        sequence_length=24
    )
    
    print("\n🎉 Sample test completed successfully!")
    print("📝 Use this script with your actual solar radiation time series data.")
    print("💡 Make sure your data has consistent time intervals (hourly recommended).")