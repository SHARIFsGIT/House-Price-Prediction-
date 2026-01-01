"""
Utility functions for House Price Prediction App
Shared functions used across Streamlit and Gradio interfaces
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os
from typing import Dict, Tuple, Any, Optional
import config


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model_and_scaler(
    model_path: str = None,
    scaler_path: str = None
) -> Tuple[Any, Any]:
    """
    Load the trained model and scaler from disk.

    Args:
        model_path: Path to the model pickle file
        scaler_path: Path to the scaler pickle file

    Returns:
        Tuple of (model, scaler)

    Raises:
        FileNotFoundError: If model or scaler files don't exist
        Exception: For other errors during loading
    """
    model_path = model_path or config.MODEL_PATH
    scaler_path = scaler_path or config.SCALER_PATH

    # Check if files exist before attempting to load
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Please run train_model.py first to train and save the model."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler file not found at: {scaler_path}\n"
            f"Please run train_model.py first to train and save the scaler."
        )

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading model files: {e}")

    return model, scaler


def load_metrics(metrics_path: str = None) -> Optional[Dict]:
    """
    Load model metrics from disk.

    Args:
        metrics_path: Path to the metrics pickle file

    Returns:
        Dictionary of model metrics or None if file doesn't exist
    """
    metrics_path = metrics_path or config.METRICS_PATH

    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            return pickle.load(f)
    return None


# =============================================================================
# INPUT PROCESSING
# =============================================================================
def encode_input(
    area: float,
    bedrooms: int,
    bathrooms: int,
    stories: int,
    parking: int,
    mainroad: str,
    guestroom: str,
    basement: str,
    hotwaterheating: str,
    airconditioning: str,
    prefarea: str,
    furnishingstatus: str
) -> np.ndarray:
    """
    Encode user inputs into the format expected by the model.

    Args:
        All input parameters from the UI

    Returns:
        Numpy array with encoded features
    """
    # Encode binary features
    mainroad_encoded = 1.0 if mainroad == "Yes" else 0.0
    guestroom_encoded = 1.0 if guestroom == "Yes" else 0.0
    basement_encoded = 1.0 if basement == "Yes" else 0.0
    hotwaterheating_encoded = 1.0 if hotwaterheating == "Yes" else 0.0
    airconditioning_encoded = 1.0 if airconditioning == "Yes" else 0.0
    prefarea_encoded = 1.0 if prefarea == "Yes" else 0.0

    # Encode furnishing status
    furnishingstatus_encoded = config.FURNISHING_MAP.get(furnishingstatus, 1.0)

    # Create input array in the correct order
    input_data = np.array([[
        area, bedrooms, bathrooms, stories,
        mainroad_encoded, guestroom_encoded, basement_encoded,
        hotwaterheating_encoded, airconditioning_encoded,
        parking, prefarea_encoded, furnishingstatus_encoded
    ]])

    return input_data


def validate_inputs(
    area: float,
    bedrooms: int,
    bathrooms: int,
    stories: int,
    parking: int
) -> Tuple[bool, Optional[str]]:
    """
    Validate user inputs against allowed ranges.

    Args:
        Input values to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if area < config.AREA_MIN or area > config.AREA_MAX:
        return False, f"Area must be between {config.AREA_MIN} and {config.AREA_MAX} sq ft"

    if bedrooms not in config.BEDROOM_OPTIONS:
        return False, f"Bedrooms must be between {min(config.BEDROOM_OPTIONS)} and {max(config.BEDROOM_OPTIONS)}"

    if bathrooms not in config.BATHROOM_OPTIONS:
        return False, f"Bathrooms must be between {min(config.BATHROOM_OPTIONS)} and {max(config.BATHROOM_OPTIONS)}"

    if stories not in config.STORIES_OPTIONS:
        return False, f"Stories must be between {min(config.STORIES_OPTIONS)} and {max(config.STORIES_OPTIONS)}"

    if parking not in config.PARKING_OPTIONS:
        return False, f"Parking must be between {min(config.PARKING_OPTIONS)} and {max(config.PARKING_OPTIONS)}"

    return True, None


# =============================================================================
# PREDICTION
# =============================================================================
def predict_price(
    model: Any,
    scaler: Any,
    input_data: np.ndarray
) -> float:
    """
    Make a price prediction using the model.

    Args:
        model: Trained model
        scaler: Fitted scaler
        input_data: Encoded input features

    Returns:
        Predicted price
    """
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]


# =============================================================================
# PRICE FORMATTING AND ANALYSIS
# =============================================================================
def format_price(price: float, currency_symbol: str = None) -> str:
    """
    Format price with currency symbol and thousands separator.

    Args:
        price: The price value
        currency_symbol: Currency symbol to use

    Returns:
        Formatted price string
    """
    currency = currency_symbol or config.CURRENCY_SYMBOL
    return f"{currency}{config.PRICE_FORMAT.format(price)}"


def calculate_price_per_sqft(price: float, area: float) -> float:
    """
    Calculate price per square foot.

    Args:
        price: Total price
        area: Area in square feet

    Returns:
        Price per square foot
    """
    if area > 0:
        return price / area
    return 0.0


def get_price_category(price: float) -> str:
    """
    Categorize the predicted price.

    Args:
        price: Predicted price

    Returns:
        Price category string
    """
    if price < 3000000:
        return "Budget-Friendly"
    elif price < 5000000:
        return "Mid-Range"
    elif price < 8000000:
        return "Premium"
    else:
        return "Luxury"


def get_price_insight(predicted_price: float, area: float) -> Dict[str, Any]:
    """
    Generate insights about the predicted price.

    Args:
        predicted_price: The predicted price
        area: The area in square feet

    Returns:
        Dictionary with various insights
    """
    price_per_sqft = calculate_price_per_sqft(predicted_price, area)
    category = get_price_category(predicted_price)

    # Compare with average (using actual average area from training data)
    avg_price_per_sqft = config.AVG_PRICE / config.AVG_AREA
    comparison = "Above Average" if price_per_sqft > avg_price_per_sqft else "Below Average"

    # Calculate affordability score (0-100)
    score = min(100, max(0, 100 - (predicted_price - config.MIN_PRICE) /
                         (config.MAX_PRICE - config.MIN_PRICE) * 100))

    return {
        "price_per_sqft": price_per_sqft,
        "category": category,
        "comparison": comparison,
        "affordability_score": round(score, 1)
    }


# =============================================================================
# HISTORY MANAGEMENT
# =============================================================================
def save_prediction(
    area: float,
    bedrooms: int,
    bathrooms: int,
    stories: int,
    parking: int,
    mainroad: str,
    guestroom: str,
    basement: str,
    hotwaterheating: str,
    airconditioning: str,
    prefarea: str,
    furnishingstatus: str,
    predicted_price: float,
    history_path: str = None
) -> None:
    """
    Save prediction to history CSV file.

    Args:
        All input parameters and predicted price
        history_path: Path to history CSV file
    """
    history_path = history_path or config.HISTORY_PATH

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
        "predicted_price": predicted_price,
        "price_per_sqft": calculate_price_per_sqft(predicted_price, area)
    }

    df = pd.DataFrame([record])

    if os.path.exists(history_path):
        existing_df = pd.read_csv(history_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    # Keep only last N records
    df = df.tail(config.HISTORY_LIMIT)
    df.to_csv(history_path, index=False)


def load_prediction_history(
    history_path: str = None,
    limit: int = None
) -> pd.DataFrame:
    """
    Load prediction history from CSV file.

    Args:
        history_path: Path to history CSV file
        limit: Maximum number of records to return

    Returns:
        DataFrame with prediction history
    """
    history_path = history_path or config.HISTORY_PATH
    limit = limit or config.HISTORY_LIMIT

    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        return df.tail(limit)
    return pd.DataFrame()


def export_to_csv(data: pd.DataFrame, filename: str = "prediction_export.csv") -> str:
    """
    Export DataFrame to CSV file.

    Args:
        data: DataFrame to export
        filename: Name of the output file

    Returns:
        Path to the exported file
    """
    data.to_csv(filename, index=False)
    return filename


# =============================================================================
# FEATURE ANALYSIS
# =============================================================================
def get_feature_importance() -> pd.DataFrame:
    """
    Get feature importance as a DataFrame.

    Returns:
        DataFrame with feature names and importance scores
    """
    importance_data = []
    for feature, score in config.FEATURE_IMPORTANCE.items():
        importance_data.append({
            "Feature": config.FEATURE_LABELS.get(feature, feature),
            "Importance": abs(score),
            "Correlation": "Positive" if score > 0 else "Negative"
        })

    df = pd.DataFrame(importance_data)
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# MODEL SUMMARY
# =============================================================================
def get_model_summary(metrics: Dict) -> str:
    """
    Generate a human-readable model summary.

    Args:
        metrics: Dictionary of model metrics

    Returns:
        Formatted summary string
    """
    if metrics is None:
        return "Model metrics not available."

    summary = f"""
### Model Performance Metrics

| Metric | Value |
|--------|-------|
| R² Score | {metrics.get('r2', 0):.4f} |
| RMSE | {metrics.get('rmse', 0):,.2f} |
| MAE | {metrics.get('mae', 0):,.2f} |
| MSE | {metrics.get('mse', 0):,.2f} |

### Interpretation

- **R² Score**: {metrics.get('r2', 0) * 100:.1f}% of the price variation is explained by the model
- **RMSE**: Average prediction error is approximately {metrics.get('rmse', 0):,.0f}
- **MAE**: Mean absolute error is {metrics.get('mae', 0):,.0f}
"""
    return summary


def create_comparison_chart_data(predicted_price: float) -> pd.DataFrame:
    """
    Create data for price comparison chart.

    Args:
        predicted_price: The predicted price

    Returns:
        DataFrame with comparison data
    """
    data = {
        "Category": ["Predicted", "Minimum", "Average", "Maximum"],
        "Price": [
            predicted_price,
            config.MIN_PRICE,
            config.AVG_PRICE,
            config.MAX_PRICE
        ]
    }
    return pd.DataFrame(data)
