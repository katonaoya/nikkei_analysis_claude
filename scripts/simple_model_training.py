#!/usr/bin/env python
"""
Simplified model training script for stock prediction
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime
import joblib

# ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleModelTrainer:
    """Simplified model trainer for stock prediction"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
    
    def load_features(self, filename: str) -> pd.DataFrame:
        """Load processed features"""
        file_path = self.processed_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Features file not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded features: {df.shape}")
        return df
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        time_split: bool = True
    ):
        """Prepare training and test data"""
        
        # Select feature columns (exclude metadata and target columns)
        exclude_cols = {
            'Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume',
            'date', 'code', 'open', 'high', 'low', 'close', 'volume',
            'UpperLimit', 'LowerLimit', 'turnover_value', 'adjustment_factor',
            'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume',
            target_column, 'Next_Day_Return', 'Return_Direction', 'Binary_Direction'
        }
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with missing target values
        clean_df = df[df[target_column].notna()].copy()
        
        # Fill missing feature values
        X = clean_df[feature_cols].fillna(0)
        y = clean_df[target_column]
        
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        
        if time_split:
            # Time-based split (last 20% of data as test)
            split_point = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_regression_models(self, X_train, X_test, y_train, y_test):
        """Train regression models"""
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
        }
        
        results = {}
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            if name == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'predictions': y_pred
            }
            
            logger.info(f"{name} - MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        return results
    
    def train_classification_models(self, X_train, X_test, y_train, y_test):
        """Train classification models"""
        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
        }
        
        results = {}
        
        # Scale features for models that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.3f}")
        
        return results
    
    def save_model(self, model_name: str, model_data: dict, target_column: str) -> Path:
        """Save trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{target_column}_{timestamp}.joblib"
        file_path = self.models_dir / filename
        
        # Package model with metadata
        model_package = {
            'model': model_data['model'],
            'scaler': self.scaler,
            'target_column': target_column,
            'timestamp': timestamp,
            'performance': {k: v for k, v in model_data.items() if k not in ['model', 'predictions', 'probabilities']}
        }
        
        joblib.dump(model_package, file_path)
        logger.info(f"Saved {model_name} to {file_path}")
        
        return file_path
    
    def create_prediction_report(self, results: dict, y_test, target_type: str):
        """Create prediction performance report"""
        
        print("\n" + "="*60)
        print(f"🎯 MODEL PERFORMANCE REPORT ({target_type.upper()})")
        print("="*60)
        
        if target_type == 'regression':
            # Regression metrics
            for model_name, data in results.items():
                print(f"\n📊 {model_name.upper()}:")
                print(f"   MSE:  {data['mse']:.6f}")
                print(f"   MAE:  {data['mae']:.6f}")
                print(f"   RMSE: {data['rmse']:.6f}")
                
                # Show some predictions vs actual
                y_pred = data['predictions']
                comparison = pd.DataFrame({
                    'Actual': y_test.iloc[:10],
                    'Predicted': y_pred[:10],
                    'Error': y_test.iloc[:10] - y_pred[:10]
                })
                print(f"\n   📋 Sample Predictions:")
                print(comparison.to_string(index=False, float_format='%.4f'))
        
        else:
            # Classification metrics
            for model_name, data in results.items():
                print(f"\n📊 {model_name.upper()}:")
                print(f"   Accuracy: {data['accuracy']:.3f}")
                
                # Classification report
                y_pred = data['predictions']
                print(f"\n   📋 Classification Report:")
                print(classification_report(y_test, y_pred))


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Simple model training for stock prediction")
    parser.add_argument(
        "--features-file",
        type=str,
        required=True,
        help="Features file to use for training"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Next_Day_Return",
        help="Target column for prediction"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['regression', 'classification'],
        default='regression',
        help="Type of models to train"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing"
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save trained models to disk"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = SimpleModelTrainer()
        
        # Load features
        print("📊 Loading features...")
        df = trainer.load_features(args.features_file)
        
        # For classification, use binary target
        target_column = args.target
        if args.model_type == 'classification' and target_column == "Next_Day_Return":
            target_column = "Binary_Direction"
        
        # Prepare data
        print("🔧 Preparing training data...")
        X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(
            df, target_column, args.test_size
        )
        
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Features: {len(feature_cols)}")
        
        # Train models
        print(f"\n🤖 Training {args.model_type} models...")
        
        if args.model_type == 'regression':
            results = trainer.train_regression_models(X_train, X_test, y_train, y_test)
        else:
            results = trainer.train_classification_models(X_train, X_test, y_train, y_test)
        
        # Show results
        trainer.create_prediction_report(results, y_test, args.model_type)
        
        # Save models if requested
        if args.save_models:
            print(f"\n💾 Saving models...")
            saved_paths = []
            for model_name, model_data in results.items():
                path = trainer.save_model(model_name, model_data, target_column)
                saved_paths.append(path)
            
            print(f"\n📁 Models saved:")
            for path in saved_paths:
                print(f"   - {path.name}")
        
        print(f"\n✅ Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())