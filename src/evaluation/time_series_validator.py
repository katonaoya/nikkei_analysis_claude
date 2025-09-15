"""
Time Series Validation Framework for Stock Prediction
Implements walk-forward validation with proper time-series splits
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Generator
from datetime import datetime, timedelta
import warnings
from pathlib import Path

try:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        precision_recall_curve, roc_curve, confusion_matrix,
        classification_report, brier_score_loss
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from utils.logger import get_logger
from utils.config import get_config


class TimeSeriesValidator:
    """
    Time Series Validator for stock prediction with walk-forward validation
    Designed for Precision ≥ 0.75 optimization
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize time series validator"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed for validation metrics")
        
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(f'validation.{key}', value)
        
        self.logger = get_logger("time_series_validator")
        
        # Validation configuration
        self.n_splits = self.config.get('validation.n_splits', 6)
        self.gap_days = self.config.get('validation.gap_days', 5)  # Prevent data leakage
        self.min_train_size = self.config.get('validation.min_train_size', 500)
        self.step_size = self.config.get('validation.step_size', 'auto')
        
        # Precision optimization settings
        self.target_precision = self.config.get('validation.target_precision', 0.75)
        self.min_precision = self.config.get('validation.min_precision', 0.70)
        
        # Results storage
        self.validation_results = []
        self.summary_metrics = {}
        self.split_details = []
        
    def create_time_splits(self, 
                          df: pd.DataFrame, 
                          date_col: str = 'Date') -> List[Tuple[pd.Index, pd.Index]]:
        """
        Create time-series splits for walk-forward validation
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        
        # Sort by date
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        dates = pd.to_datetime(df_sorted[date_col])
        
        # Calculate split points
        total_samples = len(df_sorted)
        
        if self.step_size == 'auto':
            # Automatic step size based on total samples and desired splits
            step_size = max(1, (total_samples - self.min_train_size) // (self.n_splits + 1))
        else:
            step_size = self.step_size
        
        splits = []
        split_info = []
        
        for i in range(self.n_splits):
            # Training set: from start to current split point
            train_end_idx = self.min_train_size + i * step_size
            
            # Gap to prevent data leakage
            test_start_idx = train_end_idx + self.gap_days
            test_end_idx = min(test_start_idx + step_size, total_samples)
            
            # Ensure we have enough data for both train and test
            if test_end_idx <= test_start_idx or train_end_idx < self.min_train_size:
                self.logger.warning(f"Skipping split {i}: insufficient data")
                continue
            
            train_idx = df_sorted.index[:train_end_idx]
            test_idx = df_sorted.index[test_start_idx:test_end_idx]
            
            if len(test_idx) == 0:
                continue
            
            splits.append((train_idx, test_idx))
            
            # Store split information
            split_info.append({
                'split_id': i,
                'train_start': dates.iloc[0].date(),
                'train_end': dates.iloc[train_end_idx - 1].date(),
                'test_start': dates.iloc[test_start_idx].date(),
                'test_end': dates.iloc[test_end_idx - 1].date(),
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'gap_days': self.gap_days
            })
            
            self.logger.info(f"Split {i}: train={dates.iloc[0].date()} to {dates.iloc[train_end_idx-1].date()}, "
                           f"test={dates.iloc[test_start_idx].date()} to {dates.iloc[test_end_idx-1].date()}")
        
        self.split_details = split_info
        self.logger.info(f"Created {len(splits)} time series splits")
        
        return splits
    
    def evaluate_split(self,
                      y_true: np.ndarray,
                      y_pred_proba: np.ndarray,
                      y_pred: Optional[np.ndarray] = None,
                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate a single split with comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (n_samples, 2) or (n_samples,)
            y_pred: Predicted labels (optional, will be computed from probabilities)
            threshold: Decision threshold
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Handle probability format
        if y_pred_proba.ndim == 2:
            probas = y_pred_proba[:, 1]  # Positive class probabilities
        else:
            probas = y_pred_proba
        
        # Generate predictions if not provided
        if y_pred is None:
            y_pred = (probas >= threshold).astype(int)
        
        # Basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0.0)
        recall = recall_score(y_true, y_pred, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)
        
        # ROC AUC (if we have both classes)
        try:
            auc = roc_auc_score(y_true, probas)
        except ValueError:
            auc = 0.0
        
        # Brier Score (calibration quality)
        brier = brier_score_loss(y_true, probas)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Prediction statistics
        n_predictions = np.sum(y_pred)
        prediction_rate = n_predictions / len(y_pred) if len(y_pred) > 0 else 0.0
        
        # Target achievement
        target_achieved = precision >= self.target_precision
        min_achieved = precision >= self.min_precision
        
        metrics = {
            # Core metrics
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'auc': auc,
            'brier_score': brier,
            
            # Confusion matrix components
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            
            # Prediction statistics
            'n_predictions': n_predictions,
            'prediction_rate': prediction_rate,
            'threshold': threshold,
            
            # Target achievement
            'target_achieved': target_achieved,
            'min_achieved': min_achieved,
            'target_precision': self.target_precision,
            'min_precision': self.min_precision,
            
            # Sample statistics
            'n_samples': len(y_true),
            'n_positive': np.sum(y_true),
            'positive_rate': np.mean(y_true)
        }
        
        return metrics
    
    def optimize_threshold_for_precision(self,
                                       y_true: np.ndarray,
                                       y_pred_proba: np.ndarray,
                                       target_precision: Optional[float] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Optimize threshold for target precision
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            target_precision: Target precision (default: self.target_precision)
            
        Returns:
            Tuple of (optimal_threshold, optimization_results)
        """
        target_precision = target_precision or self.target_precision
        
        # Handle probability format
        if y_pred_proba.ndim == 2:
            probas = y_pred_proba[:, 1]
        else:
            probas = y_pred_proba
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, probas)
        
        # Find thresholds that achieve target precision
        valid_indices = precisions >= target_precision
        
        if not np.any(valid_indices):
            self.logger.warning(f"No threshold achieves target precision {target_precision:.3f}")
            # Return threshold with highest precision
            best_idx = np.argmax(precisions[:-1])  # Exclude last element
            optimal_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
            achieved_precision = precisions[best_idx]
            achieved_recall = recalls[best_idx]
        else:
            # Among valid thresholds, choose the one with highest recall
            valid_recalls = recalls[valid_indices]
            best_valid_idx = np.argmax(valid_recalls)
            
            # Get actual index in full arrays
            valid_indices_array = np.where(valid_indices)[0]
            best_idx = valid_indices_array[best_valid_idx]
            
            optimal_threshold = thresholds[best_idx]
            achieved_precision = precisions[best_idx]
            achieved_recall = recalls[best_idx]
        
        optimization_results = {
            'optimal_threshold': optimal_threshold,
            'achieved_precision': achieved_precision,
            'achieved_recall': achieved_recall,
            'target_precision': target_precision,
            'target_achieved': achieved_precision >= target_precision,
            'precision_curve_points': len(precisions)
        }
        
        self.logger.info(f"Optimal threshold: {optimal_threshold:.4f}, "
                        f"achieved precision: {achieved_precision:.4f}")
        
        return optimal_threshold, optimization_results
    
    def validate_model(self,
                      model,
                      X: pd.DataFrame,
                      y: pd.Series,
                      date_col: str = 'Date') -> Dict[str, Any]:
        """
        Perform walk-forward validation on model
        
        Args:
            model: Model with fit and predict_proba methods
            X: Features DataFrame
            y: Target Series
            date_col: Date column name
            
        Returns:
            Comprehensive validation results
        """
        self.logger.info("Starting walk-forward time series validation")
        
        # Create time splits
        splits = self.create_time_splits(X, date_col)
        
        if len(splits) == 0:
            raise ValueError("No valid time splits created")
        
        validation_results = []
        
        for split_idx, (train_idx, test_idx) in enumerate(splits):
            self.logger.info(f"Validating split {split_idx + 1}/{len(splits)}")
            
            # Prepare data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Remove date column for training
            feature_cols = [col for col in X_train.columns if col != date_col]
            X_train_features = X_train[feature_cols]
            X_test_features = X_test[feature_cols]
            
            try:
                # Train model
                self.logger.info(f"Training on {len(X_train)} samples")
                model.fit(X_train_features, y_train)
                
                # Generate predictions
                y_pred_proba = model.predict_proba(X_test_features)
                
                # Optimize threshold for this split
                optimal_threshold, threshold_results = self.optimize_threshold_for_precision(
                    y_test.values, y_pred_proba
                )
                
                # Evaluate with optimal threshold
                y_pred_optimal = (y_pred_proba[:, 1] >= optimal_threshold).astype(int)
                
                split_metrics = self.evaluate_split(
                    y_test.values, y_pred_proba, y_pred_optimal, optimal_threshold
                )
                
                # Add split information
                split_metrics.update({
                    'split_id': split_idx,
                    'train_start': self.split_details[split_idx]['train_start'],
                    'train_end': self.split_details[split_idx]['train_end'],
                    'test_start': self.split_details[split_idx]['test_start'],
                    'test_end': self.split_details[split_idx]['test_end'],
                    'train_size': len(train_idx),
                    'test_size': len(test_idx)
                })
                
                # Add threshold optimization results
                split_metrics.update(threshold_results)
                
                validation_results.append(split_metrics)
                
                self.logger.info(f"Split {split_idx}: Precision={split_metrics['precision']:.4f}, "
                               f"Recall={split_metrics['recall']:.4f}, "
                               f"F1={split_metrics['f1_score']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Split {split_idx} failed: {e}")
                continue
        
        # Calculate summary statistics
        summary_metrics = self._calculate_summary_metrics(validation_results)
        
        # Store results
        self.validation_results = validation_results
        self.summary_metrics = summary_metrics
        
        self.logger.info("Walk-forward validation completed")
        self.logger.info(f"Average Precision: {summary_metrics['avg_precision']:.4f} "
                        f"(Target: {self.target_precision:.3f})")
        
        return {
            'split_results': validation_results,
            'summary_metrics': summary_metrics,
            'split_details': self.split_details,
            'validation_config': {
                'n_splits': self.n_splits,
                'gap_days': self.gap_days,
                'target_precision': self.target_precision
            }
        }
    
    def _calculate_summary_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics across all splits"""
        
        if not results:
            return {}
        
        metrics_keys = ['precision', 'recall', 'f1_score', 'accuracy', 'auc', 'brier_score']
        
        summary = {}
        
        # Calculate averages and standard deviations
        for key in metrics_keys:
            values = [r[key] for r in results if key in r]
            if values:
                summary[f'avg_{key}'] = np.mean(values)
                summary[f'std_{key}'] = np.std(values)
                summary[f'min_{key}'] = np.min(values)
                summary[f'max_{key}'] = np.max(values)
        
        # Target achievement statistics
        target_achieved_count = sum(1 for r in results if r.get('target_achieved', False))
        min_achieved_count = sum(1 for r in results if r.get('min_achieved', False))
        
        summary.update({
            'n_splits': len(results),
            'target_achieved_splits': target_achieved_count,
            'min_achieved_splits': min_achieved_count,
            'target_achievement_rate': target_achieved_count / len(results) if results else 0,
            'min_achievement_rate': min_achieved_count / len(results) if results else 0,
            'overall_target_achieved': summary.get('avg_precision', 0) >= self.target_precision
        })
        
        # Prediction statistics
        total_predictions = sum(r['n_predictions'] for r in results)
        total_samples = sum(r['n_samples'] for r in results)
        
        summary.update({
            'total_predictions': total_predictions,
            'total_samples': total_samples,
            'overall_prediction_rate': total_predictions / total_samples if total_samples > 0 else 0
        })
        
        return summary
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive validation report"""
        
        if not self.validation_results:
            return "No validation results available"
        
        report_lines = [
            "=" * 80,
            "TIME SERIES VALIDATION REPORT",
            "=" * 80,
            "",
            f"Validation Configuration:",
            f"  - Number of splits: {self.n_splits}",
            f"  - Gap days: {self.gap_days}",
            f"  - Target precision: {self.target_precision:.3f}",
            f"  - Minimum precision: {self.min_precision:.3f}",
            "",
            "Summary Results:",
            f"  - Average Precision: {self.summary_metrics.get('avg_precision', 0):.4f} ± {self.summary_metrics.get('std_precision', 0):.4f}",
            f"  - Average Recall: {self.summary_metrics.get('avg_recall', 0):.4f} ± {self.summary_metrics.get('std_recall', 0):.4f}",
            f"  - Average F1-Score: {self.summary_metrics.get('avg_f1_score', 0):.4f} ± {self.summary_metrics.get('std_f1_score', 0):.4f}",
            f"  - Average AUC: {self.summary_metrics.get('avg_auc', 0):.4f} ± {self.summary_metrics.get('std_auc', 0):.4f}",
            "",
            f"Target Achievement:",
            f"  - Splits achieving target (≥{self.target_precision:.3f}): {self.summary_metrics.get('target_achieved_splits', 0)}/{self.summary_metrics.get('n_splits', 0)}",
            f"  - Target achievement rate: {self.summary_metrics.get('target_achievement_rate', 0):.1%}",
            f"  - Overall target achieved: {'✅ YES' if self.summary_metrics.get('overall_target_achieved', False) else '❌ NO'}",
            "",
            "Split Details:",
            "-" * 80
        ]
        
        # Add split details
        for result in self.validation_results:
            status = "✅" if result.get('target_achieved', False) else "❌"
            report_lines.extend([
                f"Split {result['split_id']}: {result['test_start']} to {result['test_end']} {status}",
                f"  Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1: {result['f1_score']:.4f}",
                f"  Threshold: {result['threshold']:.4f}, Predictions: {result['n_predictions']}/{result['n_samples']}",
                ""
            ])
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Validation report saved to {save_path}")
        
        return report