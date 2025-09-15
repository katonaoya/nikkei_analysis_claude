#!/usr/bin/env python
"""
Simple prediction script using trained models
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import logging
import joblib
from datetime import datetime, date

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplePredictor:
    """Simple predictor using trained models"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.models_dir = self.data_dir / "models"
        self.processed_dir = self.data_dir / "processed"
        self.predictions_dir = self.data_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_filename: str):
        """Load trained model"""
        model_path = self.models_dir / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_package = joblib.load(model_path)
        logger.info(f"Loaded model: {model_filename}")
        
        return model_package
    
    def load_latest_data(self, features_file: str = None) -> pd.DataFrame:
        """Load latest feature data for prediction"""
        
        if features_file:
            # Use specific file
            file_path = self.processed_dir / features_file
        else:
            # Find latest features file
            feature_files = list(self.processed_dir.glob("features_*.parquet"))
            if not feature_files:
                raise FileNotFoundError("No feature files found")
            file_path = max(feature_files, key=lambda f: f.stat().st_mtime)
        
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded data: {df.shape} from {file_path.name}")
        
        return df
    
    def prepare_prediction_data(self, df: pd.DataFrame, latest_only: bool = True):
        """Prepare data for prediction"""
        
        # Get latest date for each stock if requested
        if latest_only:
            latest_date = df['Date'].max()
            df_pred = df[df['Date'] == latest_date].copy()
            logger.info(f"Using latest date: {latest_date}")
        else:
            df_pred = df.copy()
        
        # Select feature columns (same as training)
        exclude_cols = {
            'Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume',
            'date', 'code', 'open', 'high', 'low', 'close', 'volume',
            'UpperLimit', 'LowerLimit', 'turnover_value', 'adjustment_factor',
            'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume',
            'Next_Day_Return', 'Return_Direction', 'Binary_Direction'
        }
        
        feature_cols = [col for col in df_pred.columns if col not in exclude_cols]
        
        # Prepare features
        X = df_pred[feature_cols].fillna(0)
        
        # Keep metadata for results
        metadata = df_pred[['Date', 'Code', 'Close']].copy()
        
        return X, metadata, feature_cols
    
    def make_predictions(self, model_package: dict, X: pd.DataFrame):
        """Make predictions using loaded model"""
        
        model = model_package['model']
        scaler = model_package.get('scaler')
        target_column = model_package['target_column']
        
        # Scale features if needed
        model_name = str(type(model).__name__).lower()
        if scaler and 'linear' in model_name or 'logistic' in model_name:
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            
            # Get probabilities for classification
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
            else:
                probabilities = None
        else:
            predictions = model.predict(X)
            
            # Get probabilities for classification
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
            else:
                probabilities = None
        
        return predictions, probabilities, target_column
    
    def format_results(
        self, 
        predictions, 
        probabilities, 
        metadata: pd.DataFrame, 
        target_column: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """Format prediction results"""
        
        results = metadata.copy()
        results['prediction'] = predictions
        
        # Add probabilities for classification
        if probabilities is not None:
            if probabilities.shape[1] == 2:  # Binary classification
                results['prob_down'] = probabilities[:, 0]
                results['prob_up'] = probabilities[:, 1]
                results['confidence'] = np.max(probabilities, axis=1)
            else:
                results['confidence'] = np.max(probabilities, axis=1)
        
        # Sort by prediction value
        if target_column == 'Next_Day_Return':
            # For regression, show highest predicted returns
            results = results.sort_values('prediction', ascending=False)
            results['rank'] = range(1, len(results) + 1)
        elif target_column == 'Binary_Direction':
            # For classification, show by probability
            if 'prob_up' in results.columns:
                results = results.sort_values('prob_up', ascending=False)
                results['rank'] = range(1, len(results) + 1)
        
        return results
    
    def save_predictions(self, results: pd.DataFrame, model_name: str) -> Path:
        """Save predictions to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{model_name}_{timestamp}.csv"
        output_path = self.predictions_dir / filename
        
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")
        
        return output_path
    
    def create_prediction_report(self, results: pd.DataFrame, target_column: str, top_n: int = 10):
        """Create prediction report"""
        
        print("\n" + "="*60)
        print("📈 STOCK PREDICTION RESULTS")
        print("="*60)
        print(f"📊 Prediction date: {results['Date'].iloc[0]}")
        print(f"🎯 Target: {target_column}")
        print(f"📈 Total stocks: {len(results)}")
        
        if target_column == 'Next_Day_Return':
            print(f"\n🏆 TOP {top_n} PREDICTED PERFORMERS:")
            print("-" * 50)
            top_results = results.head(top_n)
            
            for i, (_, row) in enumerate(top_results.iterrows(), 1):
                pred_return = row['prediction'] * 100  # Convert to percentage
                print(f"{i:2d}. {row['Code']} - Predicted: {pred_return:+.2f}% (Current: ¥{row['Close']:.0f})")
            
            print(f"\n📉 BOTTOM {top_n} PREDICTED PERFORMERS:")
            print("-" * 50)
            bottom_results = results.tail(top_n)
            
            for i, (_, row) in enumerate(bottom_results.iterrows(), 1):
                pred_return = row['prediction'] * 100
                print(f"{i:2d}. {row['Code']} - Predicted: {pred_return:+.2f}% (Current: ¥{row['Close']:.0f})")
        
        elif target_column == 'Binary_Direction':
            print(f"\n🔥 TOP {top_n} BULLISH PREDICTIONS:")
            print("-" * 50)
            top_results = results.head(top_n)
            
            for i, (_, row) in enumerate(top_results.iterrows(), 1):
                prob_up = row.get('prob_up', 0) * 100
                confidence = row.get('confidence', 0) * 100
                print(f"{i:2d}. {row['Code']} - Up Prob: {prob_up:.1f}% (Conf: {confidence:.1f}%, Current: ¥{row['Close']:.0f})")
        
        print(f"\n📊 PREDICTION STATISTICS:")
        print("-" * 30)
        
        if target_column == 'Next_Day_Return':
            positive_preds = (results['prediction'] > 0).sum()
            negative_preds = (results['prediction'] < 0).sum()
            avg_pred = results['prediction'].mean() * 100
            
            print(f"Positive predictions: {positive_preds} ({positive_preds/len(results)*100:.1f}%)")
            print(f"Negative predictions: {negative_preds} ({negative_preds/len(results)*100:.1f}%)")
            print(f"Average predicted return: {avg_pred:+.2f}%")
            print(f"Prediction range: {results['prediction'].min()*100:.2f}% to {results['prediction'].max()*100:.2f}%")
        
        elif target_column == 'Binary_Direction' and 'prob_up' in results.columns:
            up_preds = (results['prediction'] == 1).sum()
            down_preds = (results['prediction'] == 0).sum()
            avg_confidence = results['confidence'].mean() * 100
            
            print(f"Up predictions: {up_preds} ({up_preds/len(results)*100:.1f}%)")
            print(f"Down predictions: {down_preds} ({down_preds/len(results)*100:.1f}%)")
            print(f"Average confidence: {avg_confidence:.1f}%")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Make stock predictions using trained models")
    parser.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="Trained model file to use"
    )
    parser.add_argument(
        "--features-file",
        type=str,
        help="Features file to use (latest if not specified)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top predictions to show"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to CSV file"
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="Make predictions for all dates (not just latest)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = SimplePredictor()
        
        # Load model
        print("🤖 Loading trained model...")
        model_package = predictor.load_model(args.model_file)
        model_name = args.model_file.split('_')[0]
        
        # Load data
        print("📊 Loading prediction data...")
        df = predictor.load_latest_data(args.features_file)
        
        # Prepare data
        X, metadata, feature_cols = predictor.prepare_prediction_data(
            df, latest_only=not args.all_dates
        )
        
        print(f"   Making predictions for {len(X)} samples")
        
        # Make predictions
        print("🔮 Making predictions...")
        predictions, probabilities, target_column = predictor.make_predictions(
            model_package, X
        )
        
        # Format results
        results = predictor.format_results(
            predictions, probabilities, metadata, target_column, args.top_n
        )
        
        # Show report
        predictor.create_prediction_report(results, target_column, args.top_n)
        
        # Save predictions if requested
        if args.save_predictions:
            print(f"\n💾 Saving predictions...")
            output_path = predictor.save_predictions(results, model_name)
            print(f"   Saved to: {output_path.name}")
        
        print(f"\n✅ Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())