"""
Model Training Script for House Price Prediction
Trains a Linear Regression model and saves the model, scaler, and metrics
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from datetime import datetime

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class HousePriceModelTrainer:
    """Class to handle the complete model training pipeline"""

    def __init__(self, data_path: str = None, random_state: int = 42, test_size: float = 0.2):
        """
        Initialize the model trainer.

        Args:
            data_path: Path to the housing CSV file
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use for testing
        """
        self.data_path = data_path
        self.random_state = random_state
        self.test_size = test_size
        self.df = None
        self.model = None
        self.scaler = None
        self.metrics = {}
        self.feature_names = []

        # Feature columns
        self.numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        self.binary_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                                'airconditioning', 'prefarea']
        self.ordinal_features = ['furnishingstatus']

        # Furnishing status order (unfurnished < semi-furnished < furnished)
        # With OrdinalEncoder, first item in categories gets encoded as 0
        self.furnishing_order = ['unfurnished', 'semi-furnished', 'furnished']

    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load the housing dataset from CSV file.

        Args:
            data_path: Path to the CSV file

        Returns:
            DataFrame with the loaded data
        """
        path = data_path or self.data_path or 'Housing.csv'

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at: {path}")

        self.df = pd.read_csv(path)
        print(f"[OK] Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def explore_data(self):
        """Print data exploration information"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)

        print("\n[Dataset Info]")
        print(self.df.info())

        print("\n[Summary Statistics]")
        print(self.df.describe())

        print("\n[Missing Values]")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
        else:
            print(missing[missing > 0])

        print("\n[Price Statistics]")
        print(f"  Min Price: {self.df['price'].min():,.0f}")
        print(f"  Max Price: {self.df['price'].max():,.0f}")
        print(f"  Mean Price: {self.df['price'].mean():,.0f}")
        print(f"  Median Price: {self.df['price'].median():,.0f}")

    def preprocess_data(self) -> tuple:
        """
        Preprocess the data: encode categorical variables and split features/target.

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)

        # Create a copy to avoid modifying original
        df_processed = self.df.copy()

        # Encode binary features using OrdinalEncoder
        encoder = OrdinalEncoder()
        for feature in self.binary_features:
            df_processed[feature] = encoder.fit_transform(df_processed[[feature]])
            print(f"  [OK] Encoded {feature}")

        # Encode furnishing status (ordinal)
        oe = OrdinalEncoder(categories=[self.furnishing_order])
        df_processed['furnishingstatus'] = oe.fit_transform(df_processed[['furnishingstatus']])
        print(f"  [OK] Encoded furnishingstatus")

        # Define features and target
        X = df_processed.drop(['price'], axis=1)
        y = df_processed['price']

        self.feature_names = list(X.columns)

        print(f"\n  Total features: {len(self.feature_names)}")
        print(f"  Feature names: {self.feature_names}")

        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Split data into train and test sets.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        print("\n" + "="*50)
        print("DATA SPLIT")
        print("="*50)
        print(f"  Training set: {len(X_train)} samples ({(1-self.test_size)*100:.0f}%)")
        print(f"  Test set: {len(X_test)} samples ({self.test_size*100:.0f}%)")

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> tuple:
        """
        Scale features using MinMaxScaler.

        Args:
            X_train: Training feature matrix
            X_test: Test feature matrix

        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        self.scaler = MinMaxScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("\n" + "="*50)
        print("FEATURE SCALING")
        print("="*50)
        print("  [OK] Applied MinMaxScaler to all features")
        print(f"  Training range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
        print(f"  Test range: [{X_test_scaled.min():.2f}, {X_test_scaled.max():.2f}]")

        return X_train_scaled, X_test_scaled

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the Linear Regression model.

        Args:
            X_train: Scaled training features
            y_train: Training target values
        """
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        print("  [OK] Linear Regression model trained successfully")

        # Print model coefficients
        print(f"\n  Model intercept: {self.model.intercept_:,.2f}")
        print(f"  Number of coefficients: {len(self.model.coef_)}")

        # Feature importance (coefficients)
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)

        print("\n  Top 5 Most Influential Features:")
        for idx, row in importance.head(5).iterrows():
            print(f"    {row['Feature']}: {row['Coefficient']:,.2f}")

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on test data.

        Args:
            X_test: Scaled test features
            y_test: Test target values

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        self.metrics = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        print(f"  R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
        print(f"  Mean Squared Error: {mse:,.2f}")
        print(f"  Root Mean Squared Error: {rmse:,.2f}")
        print(f"  Mean Absolute Error: {mae:,.2f}")
        print(f"\n  Average prediction error: ~{rmse:,.0f} ({(rmse/y_test.mean()*100):.1f}% of mean price)")

        return self.metrics

    def save_model(self, model_path: str = 'model.pkl',
                   scaler_path: str = 'scaler.pkl',
                   metrics_path: str = 'model_metrics.pkl'):
        """
        Save the trained model, scaler, and metrics to disk.

        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler
            metrics_path: Path to save the metrics
        """
        print("\n" + "="*50)
        print("SAVING MODEL")
        print("="*50)

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"  [OK] Model saved to: {model_path}")

        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  [OK] Scaler saved to: {scaler_path}")

        # Save metrics
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        print(f"  [OK] Metrics saved to: {metrics_path}")

    def create_visualizations(self, output_dir: str = '.'):
        """
        Create and save visualization plots.

        Args:
            output_dir: Directory to save the plots
        """
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)

        # Price distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['price'], bins=50, kde=True, color='#FF6B6B')
        plt.title('House Price Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Price (INR)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'price_distribution.png'), dpi=300, bbox_inches='tight')
        print("  [OK] Saved: price_distribution.png")
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        df_encoded = self.df.copy()
        encoder = OrdinalEncoder()
        for col in df_encoded.select_dtypes(include='object').columns:
            df_encoded[col] = encoder.fit_transform(df_encoded[[col]])

        correlation = df_encoded.corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='RdYlGn',
                    mask=mask, square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        print("  [OK] Saved: correlation_heatmap.png")
        plt.close()

        # Area vs Price scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['area'], self.df['price'], alpha=0.6, c='#4ECDC4', edgecolors='black', linewidth=0.5)
        plt.title('Area vs Price', fontsize=16, fontweight='bold')
        plt.xlabel('Area (sq ft)', fontsize=12)
        plt.ylabel('Price (INR)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'area_vs_price.png'), dpi=300, bbox_inches='tight')
        print("  [OK] Saved: area_vs_price.png")
        plt.close()

    def run_full_pipeline(self, data_path: str = None, create_plots: bool = True) -> dict:
        """
        Run the complete training pipeline.

        Args:
            data_path: Path to the dataset
            create_plots: Whether to create visualization plots

        Returns:
            Dictionary of training metrics
        """
        print("\n" + "="*60)
        print("HOUSE PRICE MODEL - TRAINING PIPELINE")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        # Load data
        self.load_data(data_path)

        # Explore data
        self.explore_data()

        # Preprocess
        X, y = self.preprocess_data()

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        # Train model
        self.train_model(X_train_scaled, y_train)

        # Evaluate
        self.evaluate_model(X_test_scaled, y_test)

        # Create visualizations
        if create_plots:
            self.create_visualizations()

        # Save model
        self.save_model()

        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        return self.metrics


def main():
    """Main function to run the training"""
    # Initialize trainer with correct data path
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'Datasets/Housing.csv'

    trainer = HousePriceModelTrainer(
        data_path=data_path,
        random_state=42,
        test_size=0.2
    )

    # Run full pipeline
    metrics = trainer.run_full_pipeline(create_plots=True)

    print("\n[OK] Model is ready to use!")
    print("\nTo run the prediction app:")
    print("  Streamlit: streamlit run app.py")
    print("  Gradio:   python gradio.py")


if __name__ == "__main__":
    main()
