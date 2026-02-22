import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json

# Configuration
BACKEND_URL = "http://localhost:5000"
MODEL_PATH = "water_demand_model.pkl"
SCALER_PATH = "scaler.pkl"

def fetch_training_data():
    """Fetch water demand data from backend API"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/water-demand")
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_features(df):
    """Prepare features for model training"""
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Select features
    features = ['day_of_year', 'month', 'day_of_week', 'population', 'avg_temp_c', 'rainfall_mm']
    X = df[features].fillna(0)
    y = df['demand_liters']
    
    return X, y, features

def train_model(X, y):
    """Train Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Testing R² Score: {test_score:.4f}")
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    
    return model, scaler

def generate_predictions(model, scaler, df, features):
    """Generate predictions for future dates"""
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    X = df[features].fillna(0)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    
    return predictions

def send_predictions_to_db(df, predictions):
    """Send predictions back to backend for storage"""
    # Prepare payload
    payload = []
    for i, pred in enumerate(predictions):
        payload.append({
            'date': str(df.iloc[i]['date'].date()) if hasattr(df.iloc[i]['date'], 'date') else str(df.iloc[i]['date']),
            'zone_id': df.iloc[i].get('zone_id', None),
            'predicted_demand': float(pred)
        })

    if not payload:
        print('No predictions to send')
        return

    try:
        url = f"{BACKEND_URL}/api/predictions"
        resp = requests.post(url, json={ 'predictions': payload }, timeout=30)
        resp.raise_for_status()
        print(f"Sent {len(payload)} predictions to backend (status {resp.status_code})")
    except Exception as e:
        print('Failed to send predictions to backend:', e)

if __name__ == "__main__":
    print("=== Water Demand Forecasting Model ===\n")
    
    # Step 1: Fetch data
    print("Step 1: Fetching data from backend...")
    df = fetch_training_data()
    if df is None or df.empty:
        print("No data available. Ensure backend is running and database is seeded.")
        exit(1)
    
    print(f"Fetched {len(df)} records\n")
    
    # Step 2: Prepare features
    print("Step 2: Preparing features...")
    X, y, features = prepare_features(df)
    print(f"Features: {features}\n")
    
    # Step 3: Train model
    print("Step 3: Training model...")
    model, scaler = train_model(X, y)
    print()
    
    # Step 4: Generate predictions
    print("Step 4: Generating predictions...")
    predictions = generate_predictions(model, scaler, df, features)
    df['predicted_demand'] = predictions
    
    # Step 5: Send to database
    print("Step 5: Sending predictions to backend database...")
    send_predictions_to_db(df, predictions)
    
    print("\n=== Training Complete ===")
