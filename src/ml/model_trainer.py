"""
Machine learning model training for stock prediction
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple
from pathlib import Path
import joblib
import logging
from datetime import datetime
import warnings

# ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb


class ModelTrainer:
    """Train machine learning models for stock prediction"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize model trainer
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.regression_models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        }
        
        self.classification_models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        }
        
        self.scaler = StandardScaler()
    
    def load_features(self, filename: str) -> pd.DataFrame:
        """Load processed features"""
        file_path = self.processed_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Features file not found: {file_path}")
        
        if filename.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        self.logger.info(f"Loaded features: {df.shape}")
        return df
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        group_column: str = 'Code'
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and targets
            target_column: Name of target column
            feature_columns: List of feature columns (auto-detect if None)
            group_column: Column for grouping (e.g., stock codes)
            
        Returns:
            Tuple of (features_df, targets_series, feature_names)
        """
        # Auto-detect feature columns if not specified
        if feature_columns is None:
            # Exclude non-feature columns
            exclude_columns = {
                'Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                target_column, 'returns', 'true_range'
            }
            feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Filter to only include rows with target values
        clean_df = df[df[target_column].notna()].copy()
        
        # Remove rows with too many missing features
        feature_df = clean_df[feature_columns]
        missing_threshold = 0.3  # Allow up to 30% missing features
        missing_ratio = feature_df.isnull().sum(axis=1) / len(feature_columns)
        valid_rows = missing_ratio <= missing_threshold
        
        clean_df = clean_df[valid_rows].copy()
        
        if len(clean_df) == 0:
            raise ValueError("No valid training data after cleaning")
        
        # Forward fill missing values within each group
        if group_column in clean_df.columns:
            clean_df[feature_columns] = clean_df.groupby(group_column)[feature_columns].fillna(method='ffill')
        
        # Final cleanup
        clean_df = clean_df.dropna(subset=[target_column])
        features = clean_df[feature_columns].fillna(0)  # Fill remaining NaNs with 0
        targets = clean_df[target_column]
        
        self.logger.info(f"Training data prepared: {features.shape}, Target: {len(targets)}")
        return features, targets, feature_columns
    
    def create_time_series_splits(
        self,
        df: pd.DataFrame,
        date_column: str = 'Date',
        n_splits: int = 5,
        test_size_months: int = 1
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Create time-series aware train/test splits
        
        Args:
            df: DataFrame with date information
            date_column: Name of date column
            n_splits: Number of splits
            test_size_months: Size of test set in months
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        df_with_date = df.copy()
        df_with_date[date_column] = pd.to_datetime(df_with_date[date_column])
        df_with_date = df_with_date.sort_values(date_column)
        
        # Create time-based splits
        splits = []
        dates = df_with_date[date_column].unique()
        dates = sorted(dates)
        
        # Calculate split sizes
        total_days = len(dates)
        test_days = max(1, total_days // (n_splits + 1))
        
        for i in range(n_splits):
            # Test set: take the last test_days from available data
            test_end_idx = total_days - (n_splits - i - 1) * test_days
            test_start_idx = max(0, test_end_idx - test_days)
            
            test_start_date = dates[test_start_idx]
            test_end_date = dates[min(test_end_idx, len(dates) - 1)]
            
            # Train set: everything before test set
            train_mask = df_with_date[date_column] < test_start_date
            test_mask = (df_with_date[date_column] >= test_start_date) & (df_with_date[date_column] <= test_end_date)
            
            train_indices = df_with_date[train_mask].index
            test_indices = df_with_date[test_mask].index
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        self.logger.info(f"Created {len(splits)} time-series splits")
        return splits
    
    def train_regression_models(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        models_to_train: Optional[List[str]] = None,
        use_cross_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Train regression models
        
        Args:
            features: Feature matrix
            targets: Target values
            models_to_train: List of model names to train
            use_cross_validation: Whether to use cross-validation
            
        Returns:
            Dictionary with trained models and results
        """
        if models_to_train is None:
            models_to_train = list(self.regression_models.keys())
        
        results = {
            'models': {},
            'scores': {},
            'feature_importance': {}
        }
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
        
        for model_name in models_to_train:
            if model_name not in self.regression_models:
                self.logger.warning(f"Unknown model: {model_name}")
                continue
            
            self.logger.info(f"Training {model_name}...")
            
            model = self.regression_models[model_name]
            
            if use_cross_validation:
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                cv_scores = []
                
                for train_idx, test_idx in tscv.split(features_scaled_df):
                    X_train, X_test = features_scaled_df.iloc[train_idx], features_scaled_df.iloc[test_idx]
                    y_train, y_test = targets.iloc[train_idx], targets.iloc[test_idx]
                    
                    model_copy = type(model)(**model.get_params())
                    model_copy.fit(X_train, y_train)
                    y_pred = model_copy.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    cv_scores.append(mse)
                
                results['scores'][model_name] = {
                    'cv_mse_mean': np.mean(cv_scores),
                    'cv_mse_std': np.std(cv_scores)
                }
            
            # Train final model on all data
            model.fit(features_scaled_df, targets)
            results['models'][model_name] = model
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': features.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results['feature_importance'][model_name] = importance_df
            
            self.logger.info(f"Completed training {model_name}")
        
        return results
    
    def train_classification_models(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        models_to_train: Optional[List[str]] = None,
        use_cross_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Train classification models
        
        Args:
            features: Feature matrix
            targets: Target labels
            models_to_train: List of model names to train
            use_cross_validation: Whether to use cross-validation
            
        Returns:
            Dictionary with trained models and results
        """
        if models_to_train is None:
            models_to_train = list(self.classification_models.keys())
        
        results = {
            'models': {},
            'scores': {},
            'feature_importance': {}
        }
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
        
        # Encode labels if necessary
        if targets.dtype == 'object':
            le = LabelEncoder()
            targets_encoded = le.fit_transform(targets)
            results['label_encoder'] = le
        else:
            targets_encoded = targets
        
        for model_name in models_to_train:
            if model_name not in self.classification_models:
                self.logger.warning(f"Unknown model: {model_name}")
                continue
            
            self.logger.info(f"Training {model_name}...")
            
            model = self.classification_models[model_name]
            
            if use_cross_validation:
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                cv_scores = []
                
                for train_idx, test_idx in tscv.split(features_scaled_df):
                    X_train, X_test = features_scaled_df.iloc[train_idx], features_scaled_df.iloc[test_idx]
                    y_train, y_test = targets_encoded[train_idx], targets_encoded[test_idx]
                    
                    model_copy = type(model)(**model.get_params())
                    model_copy.fit(X_train, y_train)
                    y_pred = model_copy.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    cv_scores.append(accuracy)
                
                results['scores'][model_name] = {
                    'cv_accuracy_mean': np.mean(cv_scores),
                    'cv_accuracy_std': np.std(cv_scores)
                }
            
            # Train final model on all data
            model.fit(features_scaled_df, targets_encoded)
            results['models'][model_name] = model
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': features.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results['feature_importance'][model_name] = importance_df
            
            self.logger.info(f"Completed training {model_name}")
        
        return results
    
    def save_models(
        self, 
        results: Dict[str, Any], 
        model_type: str,
        suffix: str = ""
    ) -> Dict[str, Path]:
        """
        Save trained models to disk
        
        Args:
            results: Results dictionary from training
            model_type: 'regression' or 'classification'
            suffix: Optional suffix for filenames
            
        Returns:
            Dictionary mapping model names to saved file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = {}
        
        for model_name, model in results.get('models', {}).items():
            filename = f"{model_type}_{model_name}_{timestamp}{suffix}.joblib"
            file_path = self.models_dir / filename
            
            # Save model along with scaler and other metadata
            model_package = {
                'model': model,
                'scaler': self.scaler,
                'feature_importance': results.get('feature_importance', {}).get(model_name),
                'scores': results.get('scores', {}).get(model_name),
                'model_type': model_type,
                'timestamp': timestamp
            }
            
            if 'label_encoder' in results:
                model_package['label_encoder'] = results['label_encoder']
            
            joblib.dump(model_package, file_path)
            saved_paths[model_name] = file_path
            
            self.logger.info(f"Saved {model_name} to {file_path}")
        
        return saved_paths
    
    def run_training_pipeline(
        self,
        features_file: str,
        target_column: str,
        model_type: str = "regression",
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete model training pipeline
        
        Args:
            features_file: Name of features file
            target_column: Name of target column
            model_type: 'regression' or 'classification'
            models_to_train: List of specific models to train
            
        Returns:
            Training results and saved model paths
        """
        self.logger.info(f"Starting {model_type} training pipeline...")
        
        # Load data
        df = self.load_features(features_file)
        
        # Prepare training data
        features, targets, feature_names = self.prepare_training_data(df, target_column)
        
        # Train models
        if model_type.lower() == "regression":
            results = self.train_regression_models(features, targets, models_to_train)
        elif model_type.lower() == "classification":
            results = self.train_classification_models(features, targets, models_to_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Save models
        saved_paths = self.save_models(results, model_type)
        
        # Create summary
        training_summary = {
            'model_type': model_type,
            'target_column': target_column,
            'feature_count': len(feature_names),
            'training_samples': len(features),
            'models_trained': list(results['models'].keys()),
            'saved_paths': saved_paths,
            'scores': results.get('scores', {}),
            'feature_names': feature_names
        }
        
        self.logger.info("Training pipeline completed successfully!")
        
        return training_summary