"""
This file trains a machine learning model to predict solar radiation based on weather conditions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """
    Load the weather data and prepare it for machine learning
    """
    print("Loading weather data...")
    
    # Load the data
    df = pd.read_csv('data/weather_solar_data.csv')
    
    # Convert datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"Loaded {len(df)} rows of data")
    
    # Features (input variables) - these are the weather conditions
    feature_columns = [
        'temperature', 'humidity', 'pressure', 'wind_speed', 
        'cloud_cover', 'visibility', 'hour', 'day_of_year'
    ]
    
    X = df[feature_columns]  # Input features
    y = df['solar_radiation']  # Target variable (what we want to predict)
    
    print("Features (inputs):", feature_columns)
    print("Target (output): solar_radiation")
    
    return X, y, df

def train_model(X, y):
    """
    Train a Random Forest model to predict solar radiation
    
    Args:
        X: Input features (weather data)
        y: Target values (solar radiation)
    
    Returns:
        Trained model
    """
    print("\nTraining the prediction model...")
    
    # Split data into training and testing sets
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data: {len(X_train)} samples")
    print(f"Testing data: {len(X_test)} samples")
    
    # Create and train the model
    # Random Forest is good for beginners - it's accurate and robust
    model = RandomForestRegressor(
        n_estimators=100,  # Number of trees in the forest
        random_state=42,   # For reproducible results
        max_depth=20       # Maximum depth of trees
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Test the model
    print("\nTesting the model...")
    y_pred = model.predict(X_test)
    
    # Calculate accuracy metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  RMSE (Root Mean Square Error): {rmse:.2f} W/m²")
    print(f"  MAE (Mean Absolute Error): {mae:.2f} W/m²")
    print(f"  R² Score: {r2:.3f} (closer to 1.0 is better)")
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nMost Important Features:")
    for _, row in feature_importance.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the trained model
    joblib.dump(model, 'models/solar_prediction_model.pkl')
    print("\nModel saved to 'models/solar_prediction_model.pkl'")
    
    return model, X_test, y_test, y_pred

def plot_results(y_test, y_pred):
    """
    Create visualizations to show how well the model performs
    """
    print("\nCreating visualization...")
    
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', lw=2)
    plt.xlabel('Actual Solar Radiation (W/m²)')
    plt.ylabel('Predicted Solar Radiation (W/m²)')
    plt.title('Actual vs Predicted Solar Radiation')
    plt.grid(True)
    
    # Plot 2: Prediction errors
    plt.subplot(1, 2, 2)
    errors = y_test - y_pred
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel('Prediction Error (W/m²)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to 'model_performance.png'")
    plt.show()

def main():
    """
    Main function that runs the entire training process
    """
    print("=== Solar Radiation Prediction Model Training ===\n")
    
    # Step 1: Load and prepare data
    X, y, df = load_and_prepare_data()
    
    # Step 2: Train the model
    model, X_test, y_test, y_pred = train_model(X, y)
    
    # Step 3: Visualize results
    plot_results(y_test, y_pred)
    
    print("\n=== Training Complete ===")
    print("You can now use the trained model to make predictions!")

if __name__ == "__main__":
    main()