"""
Configuration file for House Price Prediction App
Contains all app settings, model paths, and default values
"""

# =============================================================================
# APPLICATION INFORMATION
# =============================================================================
APP_TITLE = "House Price Prediction"
APP_SUBTITLE = "Machine Learning Based Real Estate Valuation"
APP_VERSION = "1.0.0"
MODEL_TYPE = "Linear Regression"

# =============================================================================
# FILE PATHS
# =============================================================================
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "model_metrics.pkl"
HISTORY_PATH = "prediction_history.csv"

# =============================================================================
# DEFAULT INPUT VALUES
# =============================================================================
DEFAULT_AREA = 5000
DEFAULT_BEDROOMS = 3
DEFAULT_BATHROOMS = 2
DEFAULT_STORIES = 2
DEFAULT_PARKING = 1

# =============================================================================
# INPUT VALIDATION RANGES
# =============================================================================
AREA_MIN = 1650
AREA_MAX = 16200
AREA_STEP = 50

# =============================================================================
# FEATURE OPTIONS
# =============================================================================
BEDROOM_OPTIONS = list(range(1, 7))    # 1-6 bedrooms
BATHROOM_OPTIONS = list(range(1, 5))   # 1-4 bathrooms
STORIES_OPTIONS = list(range(1, 5))    # 1-4 stories
PARKING_OPTIONS = list(range(0, 4))    # 0-3 parking spaces

YES_NO_OPTIONS = ["Yes", "No"]
FURNISHING_OPTIONS = ["furnished", "semi-furnished", "unfurnished"]

# =============================================================================
# FEATURE LABELS (for UI display)
# =============================================================================
FEATURE_LABELS = {
    "area": "Area (sq ft)",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "stories": "Stories",
    "mainroad": "Main Road",
    "guestroom": "Guest Room",
    "basement": "Basement",
    "hotwaterheating": "Hot Water Heating",
    "airconditioning": "Air Conditioning",
    "parking": "Parking Spaces",
    "prefarea": "Preferred Area",
    "furnishingstatus": "Furnishing Status"
}

# =============================================================================
# FEATURE DESCRIPTIONS (help text)
# =============================================================================
FEATURE_DESCRIPTIONS = {
    "area": "Total built-up area of the house in square feet",
    "bedrooms": "Number of bedrooms in the house",
    "bathrooms": "Number of bathrooms in the house",
    "stories": "Number of floors/stories in the building",
    "mainroad": "Whether the house is located on a main road",
    "guestroom": "Availability of a separate guest room",
    "basement": "Whether the house has a basement",
    "hotwaterheating": "Hot water heating system availability",
    "airconditioning": "Central air conditioning availability",
    "parking": "Number of covered parking spaces",
    "prefarea": "Located in a preferred/residential area",
    "furnishingstatus": "Furnishing condition of the house"
}

# =============================================================================
# ENCODING MAPS
# =============================================================================
# Note: Order is unfurnished (lowest/0) < semi-furnished (1) < furnished (highest/2)
# This matches the ordinal encoding used in train_model.py and the notebook
FURNISHING_MAP = {
    "unfurnished": 0.0,
    "semi-furnished": 1.0,
    "furnished": 2.0
}

FURNISHING_REVERSE_MAP = {
    0.0: "Unfurnished",
    1.0: "Semi-Furnished",
    2.0: "Furnished"
}

# =============================================================================
# PRICE FORMATTING
# =============================================================================
CURRENCY_SYMBOL = "INR "
PRICE_FORMAT = "{:,.2f}"
PRICE_PER_SQFT_FORMAT = "{:,.2f}"

# =============================================================================
# PRICE REFERENCE VALUES (from training data)
# =============================================================================
MIN_PRICE = 1750000
MAX_PRICE = 13300000
AVG_PRICE = 4766729
AVG_AREA = 5150  # Average area in sq ft from training data

# =============================================================================
# FEATURE IMPORTANCE (correlation coefficients)
# =============================================================================
FEATURE_IMPORTANCE = {
    "area": 0.54,
    "bathrooms": 0.45,
    "airconditioning": 0.44,
    "stories": 0.42,
    "bedrooms": 0.36,
    "prefarea": 0.33,
    "furnishingstatus": -0.31,
    "parking": 0.29,
    "mainroad": 0.29,
    "guestroom": 0.26,
    "basement": 0.19,
    "hotwaterheating": 0.10
}

# =============================================================================
# PRICING TIPS AND INSIGHTS
# =============================================================================
PRICING_TIPS = [
    "Houses on main roads tend to have higher prices",
    "Air conditioning adds significant value to the property",
    "Preferred areas command premium prices",
    "Furnished houses are priced higher than unfurnished ones",
    "Additional bathrooms increase property value",
    "Parking spaces add convenience and value"
]

# =============================================================================
# UI SETTINGS
# =============================================================================
SHOW_METRICS = True
SHOW_HISTORY = True
HISTORY_LIMIT = 100
ENABLE_EXPORT = True

# =============================================================================
# CHART SETTINGS
# =============================================================================
CHART_HEIGHT = 400
CHART_WIDTH = None

# =============================================================================
# MODEL DESCRIPTION
# =============================================================================
MODEL_DESCRIPTION = """
This application uses Linear Regression to predict house prices based on various features.

**Model Details:**
- Training samples: 545 houses
- Input features: 12 parameters
- Algorithm: Linear Regression with MinMaxScaler normalization
- Train-test split: 80/20 with random state 42

**Performance Metrics:**
- R² Score: Proportion of variance in price explained by the model
- RMSE: Root Mean Squared Error - standard deviation of prediction errors
- MAE: Mean Absolute Error - average magnitude of prediction errors

**Feature Encoding:**
- Binary features (Yes/No): Encoded as 1/0
- Furnishing status: Ordinal encoding (Furnished: 0, Semi-furnished: 1, Unfurnished: 2)
"""

# =============================================================================
# FOOTER TEXT
# =============================================================================
FOOTER_TEXT = (
    f"House Price Prediction v{APP_VERSION} | "
    f"Model: {MODEL_TYPE} | "
    f"Training Data: 545 samples | "
    f"Developed by Shariful Islam" 
)