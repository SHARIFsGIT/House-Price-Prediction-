# House Price Prediction

A machine learning application for predicting house prices using Linear Regression. The project provides both Streamlit and Gradio web interfaces for flexible interaction.

## Features

- Multiple Interface Options: Streamlit or Gradio UI
- Real-time Predictions: Instant house price estimates
- Comprehensive Inputs: 12 parameters including area, bedrooms, bathrooms, amenities, and location
- Price Visualization: Visual representation of prediction results
- Prediction History: Track and view previous predictions
- Model Metrics: Performance statistics (R², RMSE, MAE)
- Data Insights: Feature correlations and price distributions
- Export Results: Save predictions to CSV format

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.623 |
| RMSE | 1,379,764 |
| MAE | 1,057,675 |

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd House Predict
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:

Streamlit Version:
```bash
streamlit run app.py
```

Gradio Version:
```bash
python gradio.py
```

## Usage

### Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| Area | Number | Total area in square feet (1650 - 16200) |
| Bedrooms | Select | Number of bedrooms (1-6) |
| Bathrooms | Select | Number of bathrooms (1-4) |
| Stories | Select | Number of stories (1-4) |
| Parking | Select | Parking spaces available (0-3) |
| Main Road | Select | Located on main road (Yes/No) |
| Guest Room | Select | Has guest room (Yes/No) |
| Basement | Select | Has basement (Yes/No) |
| Hot Water | Select | Has hot water heating (Yes/No) |
| Air Conditioning | Select | Has air conditioning (Yes/No) |
| Preferred Area | Select | Located in preferred area (Yes/No) |
| Furnishing | Select | Furnishing status (Furnished/Semi-furnished/Unfurnished) |

### Example Prediction

Input:
- Area: 5000 sq ft
- Bedrooms: 3
- Bathrooms: 2
- Stories: 2
- Parking: 1
- Main Road: Yes
- Guest Room: No
- Basement: No
- Hot Water: No
- Air Conditioning: Yes
- Preferred Area: Yes
- Furnishing: Semi-furnished

Predicted Price: ~5,500,000 INR

## Project Structure

```
house-predict/
├── app.py                      # Streamlit application
├── gradio.py                   # Gradio application
├── train_model.py              # Model training script
├── utils.py                    # Shared utility functions
├── config.py                   # Configuration settings
├── model.pkl                   # Trained model
├── scaler.pkl                  # Feature scaler
├── model_metrics.pkl           # Model performance metrics
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── linear_regression.ipynb     # Training notebook
```

## Technology Stack

- Machine Learning: scikit-learn
- Framework: Streamlit / Gradio
- Data Processing: pandas, numpy
- Model: Linear Regression with MinMaxScaler

## Model Details

### Feature Encoding
- Binary features encoded as 0/1
- Ordinal encoding for furnishing status:
  - Furnished: 0
  - Semi-furnished: 1
  - Unfurnished: 2

### Data Preprocessing
- MinMaxScaler for feature normalization
- 80/20 train-test split
- Random state: 42

### Features Used
1. Area (continuous)
2. Bedrooms (categorical)
3. Bathrooms (categorical)
4. Stories (categorical)
5. Main Road (binary)
6. Guest Room (binary)
7. Basement (binary)
8. Hot Water Heating (binary)
9. Air Conditioning (binary)
10. Parking (categorical)
11. Preferred Area (binary)
12. Furnishing Status (ordinal)

## License

This project is open source and available under the MIT License.

## Contact

For questions or suggestions, please open an issue on GitHub.
