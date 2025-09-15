"""
Daily prediction system for stock analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime, date, timedelta
import warnings

from src.models.ensemble_model import EnsembleModel
from src.features.feature_pipeline import FeaturePipeline
from src.data.stock_data_fetcher import StockDataFetcher
from src.data.data_preprocessor import DataPreprocessor
from utils.logger import get_logger
from utils.config import get_config
from utils.calendar_utils import is_business_day, next_business_day


class DailyPredictor:
    """Daily stock prediction system"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize daily predictor"""
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(f'inference.{key}', value)
        
        self.logger = get_logger("daily_predictor")
        
        # Components
        self.ensemble_model = None
        self.feature_pipeline = FeaturePipeline()
        self.data_fetcher = StockDataFetcher()
        self.data_preprocessor = DataPreprocessor()
        
        # Prediction parameters
        self.lookback_days = self.config.get('inference.lookback_days', 252)  # 1 year
        self.target_symbols = self.config.get('inference.target_symbols', [])  # Nikkei 225 codes
        
        # Model paths
        self.model_dir = self.config.get_data_dir('models')
        self.results_dir = self.config.get_data_dir('predictions')
        
        # Prediction metadata
        self.prediction_date = None
        self.features_used = None
        self.model_info = None
        
    def load_model(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load ensemble model for predictions
        
        Args:
            model_path: Path to ensemble model file
        """
        if model_path is None:
            model_path = self.model_dir / "ensemble_model.joblib"
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.ensemble_model = EnsembleModel()
        self.ensemble_model.load_ensemble(model_path)
        
        self.model_info = self.ensemble_model.get_ensemble_info()
        
        self.logger.info("Ensemble model loaded successfully",
                        models=self.model_info['models'],
                        calibrated=self.model_info['is_fitted'])
    
    def prepare_data(self, prediction_date: Optional[Union[str, date, datetime]] = None) -> pd.DataFrame:
        """
        Prepare data for daily prediction
        
        Args:
            prediction_date: Date for prediction (defaults to latest business day)
            
        Returns:
            Prepared feature data
        """
        # Set prediction date
        if prediction_date is None:
            prediction_date = datetime.now().date()
        elif isinstance(prediction_date, str):
            prediction_date = pd.to_datetime(prediction_date).date()
        elif isinstance(prediction_date, datetime):
            prediction_date = prediction_date.date()
        
        # Ensure it's a business day
        if not is_business_day(prediction_date):
            # Get previous business day
            prev_bday = prediction_date - timedelta(days=1)
            while not is_business_day(prev_bday):
                prev_bday -= timedelta(days=1)
            prediction_date = prev_bday
            self.logger.info(f"Adjusted to previous business day: {prediction_date}")
        
        self.prediction_date = prediction_date
        
        # Calculate date range for features
        end_date = prediction_date
        start_date = end_date - timedelta(days=int(self.lookback_days * 1.5))  # Add buffer
        
        self.logger.info("Preparing prediction data",
                        prediction_date=prediction_date,
                        start_date=start_date,
                        end_date=end_date)
        
        # Get target symbols
        if not self.target_symbols:
            # If no specific symbols, get Nikkei 225 constituents
            try:
                nikkei_symbols = self._get_nikkei225_symbols()
                self.target_symbols = nikkei_symbols[:50]  # Limit for testing
                self.logger.info(f"Using Nikkei 225 symbols: {len(self.target_symbols)} stocks")
            except Exception as e:
                self.logger.warning(f"Could not get Nikkei 225 symbols: {e}")
                # Use some major Japanese stocks as fallback
                self.target_symbols = ['7203', '6758', '8306', '4063', '9984']
                
        # Fetch stock data
        all_data = []
        successful_fetches = 0
        
        for symbol in self.target_symbols:
            try:
                stock_data = self.data_fetcher.get_stock_prices(
                    code=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not stock_data.empty:
                    all_data.append(stock_data)
                    successful_fetches += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No stock data could be fetched")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        self.logger.info("Stock data fetched",
                        symbols_requested=len(self.target_symbols),
                        symbols_successful=successful_fetches,
                        total_records=len(combined_data))
        
        # Preprocess data
        processed_data = self.data_preprocessor.preprocess_data(combined_data)
        
        # Create features
        feature_data = self.feature_pipeline.create_basic_features(processed_data)
        
        # Filter to prediction date
        prediction_data = feature_data[
            pd.to_datetime(feature_data['Date']).dt.date == prediction_date
        ].copy()
        
        if prediction_data.empty:
            raise ValueError(f"No data available for prediction date: {prediction_date}")
        
        self.features_used = [col for col in prediction_data.columns 
                             if col not in ['Date', 'Code', 'target', 'next_day_return', 'next_day_high_return']]
        
        self.logger.info("Prediction data prepared",
                        stocks=prediction_data['Code'].nunique(),
                        features=len(self.features_used),
                        records=len(prediction_data))
        
        return prediction_data
    
    def predict(self, data: Optional[pd.DataFrame] = None, 
                prediction_date: Optional[Union[str, date, datetime]] = None) -> pd.DataFrame:
        """
        Make daily predictions
        
        Args:
            data: Prepared feature data (if None, will prepare automatically)
            prediction_date: Date for prediction
            
        Returns:
            Predictions DataFrame
        """
        if self.ensemble_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare data if not provided
        if data is None:
            data = self.prepare_data(prediction_date)
        
        self.logger.info("Starting daily predictions",
                        prediction_date=self.prediction_date,
                        stocks=len(data))
        
        # Prepare features for prediction
        feature_cols = [col for col in data.columns if col in self.features_used]
        X = data[feature_cols].copy()
        
        # Handle missing features
        missing_features = set(self.features_used) - set(feature_cols)
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                X[feature] = 0.0
        
        # Reorder columns to match training
        X = X[self.features_used]
        
        # Make predictions
        try:
            probabilities = self.ensemble_model.predict_proba(X, use_calibration=True)
            individual_preds = self.ensemble_model.get_individual_predictions(X, use_calibration=True)
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': self.prediction_date,
            'Code': data['Code'].values,
            'prediction_probability': probabilities[:, 1],
            'prediction_binary': (probabilities[:, 1] >= 0.5).astype(int)
        })
        
        # Add individual model predictions
        for model_name, preds in individual_preds.items():
            results[f'{model_name}_probability'] = preds
        
        # Add current price information
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in price_cols:
            if col in data.columns:
                results[f'current_{col.lower()}'] = data[col].values
        
        # Sort by prediction probability
        results = results.sort_values('prediction_probability', ascending=False).reset_index(drop=True)
        
        self.logger.info("Daily predictions completed",
                        total_predictions=len(results),
                        positive_predictions=(probabilities[:, 1] >= 0.5).sum(),
                        mean_probability=probabilities[:, 1].mean())
        
        return results
    
    def get_shap_explanations(self, data: pd.DataFrame, top_k_stocks: int = 5) -> Dict[str, Any]:
        """
        Get SHAP explanations for top predictions
        
        Args:
            data: Feature data
            top_k_stocks: Number of top stocks to explain
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        if self.ensemble_model is None:
            raise ValueError("Model not loaded")
        
        # Get predictions first to identify top stocks
        predictions = self.predict(data)
        top_stocks = predictions.head(top_k_stocks)
        
        # Get feature data for top stocks
        top_indices = top_stocks.index.tolist()
        feature_cols = [col for col in data.columns if col in self.features_used]
        X_top = data.iloc[top_indices][feature_cols].copy()
        
        # Handle missing features
        missing_features = set(self.features_used) - set(feature_cols)
        for feature in missing_features:
            X_top[feature] = 0.0
        
        X_top = X_top[self.features_used]
        
        explanations = {}
        
        # Get SHAP values from individual models
        for model_name, model in self.ensemble_model.models.items():
            try:
                if hasattr(model, 'get_shap_values'):
                    shap_values = model.get_shap_values(X_top)
                    
                    # Create explanation for each stock
                    for i, (_, stock_row) in enumerate(top_stocks.iterrows()):
                        stock_code = stock_row['Code']
                        
                        if stock_code not in explanations:
                            explanations[stock_code] = {
                                'prediction_probability': stock_row['prediction_probability'],
                                'models': {}
                            }
                        
                        # Get top contributing features
                        stock_shap = shap_values[i]
                        feature_contributions = pd.Series(stock_shap, index=self.features_used)
                        top_features = feature_contributions.abs().nlargest(10)
                        
                        explanations[stock_code]['models'][model_name] = {
                            'top_features': top_features.to_dict(),
                            'total_contribution': stock_shap.sum()
                        }
                        
            except Exception as e:
                self.logger.warning(f"Could not get SHAP values for {model_name}: {e}")
        
        self.logger.info("SHAP explanations generated",
                        explained_stocks=len(explanations))
        
        return explanations
    
    def _get_nikkei225_symbols(self) -> List[str]:
        """Get Nikkei 225 constituent symbols"""
        # This would typically fetch from an API or database
        # For now, return a subset of major Nikkei 225 stocks
        major_stocks = [
            '7203', '6758', '8306', '4063', '9984', '6861', '6954',
            '9432', '8316', '4755', '7974', '2914', '6367', '4568',
            '9020', '2802', '4502', '7267', '8058', '6501'
        ]
        
        self.logger.info(f"Using major stock subset: {len(major_stocks)} symbols")
        return major_stocks
    
    def save_predictions(self, predictions: pd.DataFrame, 
                        explanations: Optional[Dict] = None) -> Path:
        """
        Save predictions to file
        
        Args:
            predictions: Predictions DataFrame
            explanations: SHAP explanations dictionary
            
        Returns:
            Path to saved file
        """
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        date_str = self.prediction_date.strftime('%Y%m%d')
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"predictions_{date_str}_{timestamp}.csv"
        file_path = self.results_dir / filename
        
        # Save main predictions
        predictions.to_csv(file_path, index=False)
        
        # Save explanations if available
        if explanations:
            exp_filename = f"explanations_{date_str}_{timestamp}.json"
            exp_path = self.results_dir / exp_filename
            
            import json
            with open(exp_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_explanations = {}
                for stock, data in explanations.items():
                    json_explanations[stock] = {
                        'prediction_probability': float(data['prediction_probability']),
                        'models': {}
                    }
                    
                    for model_name, model_data in data['models'].items():
                        json_explanations[stock]['models'][model_name] = {
                            'top_features': {k: float(v) for k, v in model_data['top_features'].items()},
                            'total_contribution': float(model_data['total_contribution'])
                        }
                
                json.dump(json_explanations, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Explanations saved", file_path=exp_path)
        
        self.logger.info("Predictions saved", 
                        file_path=file_path,
                        records=len(predictions))
        
        return file_path
    
    def get_prediction_summary(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for predictions
        
        Args:
            predictions: Predictions DataFrame
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'prediction_date': self.prediction_date,
            'total_stocks': len(predictions),
            'positive_predictions': (predictions['prediction_probability'] >= 0.5).sum(),
            'mean_probability': predictions['prediction_probability'].mean(),
            'std_probability': predictions['prediction_probability'].std(),
            'max_probability': predictions['prediction_probability'].max(),
            'min_probability': predictions['prediction_probability'].min(),
            'top_10_stocks': predictions.head(10)[['Code', 'prediction_probability']].to_dict('records'),
            'model_info': self.model_info
        }
        
        # Probability distribution
        summary['probability_ranges'] = {
            'very_high (>0.8)': (predictions['prediction_probability'] > 0.8).sum(),
            'high (0.6-0.8)': ((predictions['prediction_probability'] > 0.6) & 
                              (predictions['prediction_probability'] <= 0.8)).sum(),
            'medium (0.4-0.6)': ((predictions['prediction_probability'] > 0.4) & 
                                (predictions['prediction_probability'] <= 0.6)).sum(),
            'low (<0.4)': (predictions['prediction_probability'] <= 0.4).sum()
        }
        
        return summary