"""
Evaluation metrics calculator for stock prediction models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from pathlib import Path

try:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        roc_auc_score, average_precision_score, log_loss, brier_score_loss,
        confusion_matrix, classification_report, roc_curve, precision_recall_curve
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from utils.logger import get_logger
from utils.config import get_config


class MetricsCalculator:
    """Comprehensive evaluation metrics calculator for binary classification"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize metrics calculator"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for metrics calculation")
        
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(f'evaluation.{key}', value)
        
        self.logger = get_logger("metrics_calculator")
        
        # Evaluation parameters
        self.primary_threshold = self.config.get('evaluation.primary_threshold', 0.5)
        self.threshold_range = self.config.get('evaluation.threshold_range', (0.1, 0.9))
        self.n_threshold_steps = self.config.get('evaluation.n_threshold_steps', 50)
        
        # Results storage
        self.results_dir = self.config.get_data_dir('evaluation')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_basic_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred_proba: Union[pd.Series, np.ndarray],
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of basic metrics
        """
        if threshold is None:
            threshold = self.primary_threshold
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            self.logger.warning("Only one class present in y_true")
            return self._get_empty_metrics()
        
        metrics = {}
        
        # Basic classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Probability-based metrics
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['roc_auc'] = np.nan
            
        try:
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            metrics['average_precision'] = np.nan
        
        try:
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except ValueError:
            metrics['log_loss'] = np.nan
            
        try:
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        except ValueError:
            metrics['brier_score'] = np.nan
        
        # Additional metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Financial metrics
        positive_rate = y_pred.mean()
        metrics['prediction_rate'] = positive_rate
        metrics['base_rate'] = y_true.mean()
        
        if positive_rate > 0:
            metrics['precision_lift'] = metrics['precision'] / metrics['base_rate']
        else:
            metrics['precision_lift'] = np.nan
        
        metrics['threshold'] = threshold
        
        self.logger.debug("Basic metrics calculated", 
                         precision=f"{metrics['precision']:.3f}",
                         recall=f"{metrics['recall']:.3f}",
                         f1=f"{metrics['f1_score']:.3f}")
        
        return metrics
    
    def calculate_threshold_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred_proba: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """
        Calculate metrics across different thresholds
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            DataFrame with metrics for each threshold
        """
        thresholds = np.linspace(self.threshold_range[0], self.threshold_range[1], self.n_threshold_steps)
        
        results = []
        
        for threshold in thresholds:
            metrics = self.calculate_basic_metrics(y_true, y_pred_proba, threshold)
            results.append(metrics)
        
        df = pd.DataFrame(results)
        
        self.logger.info(f"Threshold analysis completed for {len(thresholds)} thresholds")
        
        return df
    
    def find_optimal_threshold(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred_proba: Union[pd.Series, np.ndarray],
        metric: str = 'f1_score',
        min_precision: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Find optimal threshold based on specified metric
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1_score', 'precision', 'recall', etc.)
            min_precision: Minimum precision constraint
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        threshold_metrics = self.calculate_threshold_metrics(y_true, y_pred_proba)
        
        # Apply precision constraint if specified
        if min_precision is not None:
            valid_thresholds = threshold_metrics[threshold_metrics['precision'] >= min_precision]
            if valid_thresholds.empty:
                self.logger.warning(f"No thresholds meet minimum precision {min_precision}")
                valid_thresholds = threshold_metrics
        else:
            valid_thresholds = threshold_metrics
        
        # Find optimal threshold
        if metric not in valid_thresholds.columns:
            raise ValueError(f"Metric '{metric}' not available. Available metrics: {list(valid_thresholds.columns)}")
        
        optimal_idx = valid_thresholds[metric].idxmax()
        optimal_row = valid_thresholds.loc[optimal_idx]
        
        result = {
            'optimal_threshold': optimal_row['threshold'],
            'optimal_metrics': optimal_row.to_dict(),
            'optimization_metric': metric,
            'min_precision_constraint': min_precision
        }
        
        self.logger.info(f"Optimal threshold found: {optimal_row['threshold']:.3f} "
                        f"({metric}={optimal_row[metric]:.3f})")
        
        return result
    
    def calculate_calibration_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred_proba: Union[pd.Series, np.ndarray],
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate probability calibration metrics
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics and curves
        """
        try:
            # Calibration curve
            fraction_positives, mean_predicted = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
            
            # Brier score components
            brier_score = brier_score_loss(y_true, y_pred_proba)
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            # Maximum Calibration Error (MCE)
            bin_errors = []
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                if in_bin.sum() > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    bin_errors.append(abs(avg_confidence_in_bin - accuracy_in_bin))
            
            mce = max(bin_errors) if bin_errors else 0
            
            calibration_metrics = {
                'brier_score': brier_score,
                'expected_calibration_error': ece,
                'maximum_calibration_error': mce,
                'calibration_curve': {
                    'fraction_positives': fraction_positives,
                    'mean_predicted': mean_predicted
                }
            }
            
            self.logger.info("Calibration metrics calculated",
                           brier_score=f"{brier_score:.4f}",
                           ece=f"{ece:.4f}")
            
            return calibration_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate calibration metrics: {e}")
            return {'error': str(e)}
    
    def generate_classification_report(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred_proba: Union[pd.Series, np.ndarray],
        threshold: Optional[float] = None,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate detailed classification report
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            target_names: Names for the classes
            
        Returns:
            Formatted classification report string
        """
        if threshold is None:
            threshold = self.primary_threshold
            
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if target_names is None:
            target_names = ['Negative', 'Positive']
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        
        return report
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary for edge cases"""
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'roc_auc': np.nan,
            'average_precision': np.nan,
            'log_loss': np.nan,
            'brier_score': np.nan,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'specificity': 0.0,
            'negative_predictive_value': 0.0,
            'prediction_rate': 0.0,
            'base_rate': 0.0,
            'precision_lift': np.nan,
            'threshold': self.primary_threshold
        }
    
    def plot_metrics_curves(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred_proba: Union[pd.Series, np.ndarray],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> Optional[str]:
        """
        Plot ROC, PR, and Calibration curves
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Path to saved plot (if saved)
        """
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Matplotlib not available for plotting")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        axes[0, 1].plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {avg_precision:.3f})')
        axes[0, 1].axhline(y=y_true.mean(), color='k', linestyle='--', linewidth=1, label='Random')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calibration Curve
        calibration_metrics = self.calculate_calibration_metrics(y_true, y_pred_proba)
        if 'calibration_curve' in calibration_metrics:
            frac_pos = calibration_metrics['calibration_curve']['fraction_positives']
            mean_pred = calibration_metrics['calibration_curve']['mean_predicted']
            
            axes[1, 0].plot(mean_pred, frac_pos, marker='o', linewidth=2, label='Model')
            axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
            axes[1, 0].set_xlabel('Mean Predicted Probability')
            axes[1, 0].set_ylabel('Fraction of Positives')
            axes[1, 0].set_title('Calibration Curve')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Threshold Analysis
        threshold_metrics = self.calculate_threshold_metrics(y_true, y_pred_proba)
        
        axes[1, 1].plot(threshold_metrics['threshold'], threshold_metrics['precision'], 
                       label='Precision', linewidth=2)
        axes[1, 1].plot(threshold_metrics['threshold'], threshold_metrics['recall'], 
                       label='Recall', linewidth=2)
        axes[1, 1].plot(threshold_metrics['threshold'], threshold_metrics['f1_score'], 
                       label='F1 Score', linewidth=2)
        axes[1, 1].axvline(x=self.primary_threshold, color='k', linestyle='--', 
                          linewidth=1, label=f'Default ({self.primary_threshold})')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Metrics vs Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.results_dir / f"metrics_curves_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Metrics curves saved to {save_path}")
        
        return str(save_path)
    
    def export_metrics_report(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred_proba: Union[pd.Series, np.ndarray],
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> str:
        """
        Export comprehensive metrics report
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the report
            
        Returns:
            Path to saved report
        """
        if save_path is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.results_dir / f"metrics_report_{model_name}_{timestamp}.txt"
        
        # Calculate metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred_proba)
        optimal_threshold = self.find_optimal_threshold(y_true, y_pred_proba, 'f1_score')
        calibration_metrics = self.calculate_calibration_metrics(y_true, y_pred_proba)
        classification_rep = self.generate_classification_report(y_true, y_pred_proba)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"EVALUATION METRICS REPORT - {model_name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic metrics
            f.write("BASIC METRICS\n")
            f.write("-" * 15 + "\n")
            f.write(f"Precision: {basic_metrics['precision']:.4f}\n")
            f.write(f"Recall: {basic_metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {basic_metrics['f1_score']:.4f}\n")
            f.write(f"Accuracy: {basic_metrics['accuracy']:.4f}\n")
            f.write(f"ROC AUC: {basic_metrics['roc_auc']:.4f}\n")
            f.write(f"Average Precision: {basic_metrics['average_precision']:.4f}\n")
            f.write(f"Log Loss: {basic_metrics['log_loss']:.4f}\n")
            f.write(f"Brier Score: {basic_metrics['brier_score']:.4f}\n\n")
            
            # Optimal threshold
            f.write("OPTIMAL THRESHOLD ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Optimal Threshold (F1): {optimal_threshold['optimal_threshold']:.4f}\n")
            opt_metrics = optimal_threshold['optimal_metrics']
            f.write(f"Precision at Optimal: {opt_metrics['precision']:.4f}\n")
            f.write(f"Recall at Optimal: {opt_metrics['recall']:.4f}\n")
            f.write(f"F1 Score at Optimal: {opt_metrics['f1_score']:.4f}\n\n")
            
            # Calibration
            f.write("CALIBRATION METRICS\n")
            f.write("-" * 20 + "\n")
            if 'error' not in calibration_metrics:
                f.write(f"Brier Score: {calibration_metrics['brier_score']:.4f}\n")
                f.write(f"Expected Calibration Error: {calibration_metrics['expected_calibration_error']:.4f}\n")
                f.write(f"Maximum Calibration Error: {calibration_metrics['maximum_calibration_error']:.4f}\n\n")
            
            # Classification report
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("-" * 35 + "\n")
            f.write(classification_rep)
            f.write("\n")
            
            # Confusion Matrix
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 16 + "\n")
            f.write(f"True Positives: {basic_metrics['true_positives']}\n")
            f.write(f"True Negatives: {basic_metrics['true_negatives']}\n")
            f.write(f"False Positives: {basic_metrics['false_positives']}\n")
            f.write(f"False Negatives: {basic_metrics['false_negatives']}\n\n")
            
            # Additional insights
            f.write("ADDITIONAL INSIGHTS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Base Rate (% Positive): {basic_metrics['base_rate']:.2%}\n")
            f.write(f"Prediction Rate: {basic_metrics['prediction_rate']:.2%}\n")
            f.write(f"Precision Lift: {basic_metrics['precision_lift']:.2f}x\n")
        
        self.logger.info(f"Metrics report exported to {save_path}")
        
        return str(save_path)