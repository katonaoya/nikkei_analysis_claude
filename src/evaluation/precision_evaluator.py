"""
Precision-focused evaluation system for stock prediction models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from utils.logger import get_logger
from utils.config import get_config


@dataclass
class PrecisionMetrics:
    """Precision-focused metrics data class"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    auc_roc: float
    auc_pr: float
    specificity: float
    npv: float  # Negative Predictive Value
    
    # Trading specific metrics
    selected_stocks_count: int
    profitable_stocks_count: int
    avg_return: float
    total_return: float
    hit_rate: float
    
    # Time-based metrics
    evaluation_date: str
    n_samples: int


@dataclass
class MarketEnvironmentMetrics:
    """Market environment specific metrics"""
    environment_type: str  # 'bull', 'bear', 'sideways', 'volatile'
    market_return: float
    market_volatility: float
    precision: float
    recall: float
    selected_count: int
    profitable_count: int
    avg_return: float
    hit_rate: float


class PrecisionEvaluator:
    """Precision-focused evaluation system optimized for Precision ≥ 0.75"""
    
    def __init__(self, target_precision: float = 0.75, max_daily_selections: int = 3):
        """
        Initialize precision evaluator
        
        Args:
            target_precision: Target precision threshold (default: 0.75)
            max_daily_selections: Maximum stocks to select per day
        """
        self.target_precision = target_precision
        self.max_daily_selections = max_daily_selections
        
        self.config = get_config()
        self.logger = get_logger("precision_evaluator")
        
        # Market environment thresholds
        self.bull_market_threshold = 0.15  # >15% annual return
        self.bear_market_threshold = -0.10  # <-10% annual return
        self.high_volatility_threshold = 0.25  # >25% volatility
        
    def calculate_precision_metrics(
        self, 
        y_true: Union[pd.Series, np.ndarray], 
        y_pred_proba: Union[pd.Series, np.ndarray],
        y_pred_binary: Optional[Union[pd.Series, np.ndarray]] = None,
        returns: Optional[Union[pd.Series, np.ndarray]] = None,
        date_index: Optional[pd.DatetimeIndex] = None,
        threshold: Optional[float] = None
    ) -> PrecisionMetrics:
        """
        Calculate comprehensive precision-focused metrics
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            y_pred_binary: Binary predictions (if None, will use threshold)
            returns: Actual returns for selected stocks
            date_index: Date index for evaluation period
            threshold: Probability threshold for binary classification
            
        Returns:
            PrecisionMetrics dataclass with all metrics
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, accuracy_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        # Determine binary predictions
        if y_pred_binary is None:
            if threshold is None:
                threshold = self._optimize_threshold_for_precision(y_true, y_pred_proba)
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred_binary = np.array(y_pred_binary)
        
        # Basic classification metrics
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred_binary)
        
        # ROC and PR AUC
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
            auc_pr = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            auc_roc = 0.5
            auc_pr = 0.0
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Trading-specific metrics
        selected_count = int(y_pred_binary.sum())
        profitable_count = int((y_pred_binary & y_true).sum())
        hit_rate = profitable_count / selected_count if selected_count > 0 else 0
        
        # Return metrics
        if returns is not None:
            returns = np.array(returns)
            selected_returns = returns[y_pred_binary.astype(bool)]
            avg_return = float(selected_returns.mean()) if len(selected_returns) > 0 else 0.0
            total_return = float(selected_returns.sum()) if len(selected_returns) > 0 else 0.0
        else:
            avg_return = hit_rate * 0.01  # Assume 1% return for profitable stocks
            total_return = avg_return * selected_count
        
        # Evaluation date
        eval_date = date_index.max().strftime('%Y-%m-%d') if date_index is not None else datetime.now().strftime('%Y-%m-%d')
        
        return PrecisionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            specificity=specificity,
            npv=npv,
            selected_stocks_count=selected_count,
            profitable_stocks_count=profitable_count,
            avg_return=avg_return,
            total_return=total_return,
            hit_rate=hit_rate,
            evaluation_date=eval_date,
            n_samples=len(y_true)
        )
    
    def _optimize_threshold_for_precision(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        min_precision: Optional[float] = None
    ) -> float:
        """
        Optimize threshold to achieve target precision
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            min_precision: Minimum precision requirement
            
        Returns:
            Optimal threshold
        """
        from sklearn.metrics import precision_recall_curve
        
        if min_precision is None:
            min_precision = self.target_precision
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Find thresholds that meet minimum precision
        valid_indices = precisions >= min_precision
        
        if not valid_indices.any():
            # If no threshold meets target precision, use highest precision
            best_idx = np.argmax(precisions)
            self.logger.warning(f"No threshold achieves precision ≥ {min_precision:.3f}. "
                               f"Using threshold with precision {precisions[best_idx]:.3f}")
            return thresholds[best_idx] if best_idx < len(thresholds) else 0.9
        
        # Among valid thresholds, choose the one with highest recall
        valid_precisions = precisions[valid_indices]
        valid_recalls = recalls[valid_indices]
        valid_thresholds = thresholds[valid_indices[:-1]]  # thresholds is 1 element shorter
        
        if len(valid_thresholds) == 0:
            return 0.9
        
        # Select threshold with highest recall among valid ones
        best_recall_idx = np.argmax(valid_recalls[:-1])  # Exclude last element to match thresholds
        optimal_threshold = valid_thresholds[best_recall_idx]
        
        self.logger.info(f"Optimal threshold: {optimal_threshold:.3f} "
                        f"(Precision: {valid_precisions[best_recall_idx]:.3f}, "
                        f"Recall: {valid_recalls[best_recall_idx]:.3f})")
        
        return float(optimal_threshold)
    
    def evaluate_by_market_environment(
        self,
        df: pd.DataFrame,
        y_true_col: str = 'target',
        y_pred_proba_col: str = 'pred_proba',
        returns_col: str = 'next_day_return',
        market_return_col: str = 'market_return'
    ) -> Dict[str, MarketEnvironmentMetrics]:
        """
        Evaluate performance by market environment
        
        Args:
            df: DataFrame with predictions and market data
            y_true_col: Column name for true labels
            y_pred_proba_col: Column name for predicted probabilities
            returns_col: Column name for actual returns
            market_return_col: Column name for market returns
            
        Returns:
            Dictionary of market environment metrics
        """
        # Classify market environments
        df = df.copy()
        df['market_environment'] = self._classify_market_environment(df[market_return_col])
        
        environment_metrics = {}
        
        for env_type in df['market_environment'].unique():
            env_data = df[df['market_environment'] == env_type]
            
            if len(env_data) == 0:
                continue
            
            # Calculate metrics for this environment
            metrics = self.calculate_precision_metrics(
                env_data[y_true_col],
                env_data[y_pred_proba_col],
                returns=env_data[returns_col] if returns_col in env_data.columns else None
            )
            
            # Market environment specific metrics
            market_return = env_data[market_return_col].mean()
            market_volatility = env_data[market_return_col].std()
            
            environment_metrics[env_type] = MarketEnvironmentMetrics(
                environment_type=env_type,
                market_return=market_return,
                market_volatility=market_volatility,
                precision=metrics.precision,
                recall=metrics.recall,
                selected_count=metrics.selected_stocks_count,
                profitable_count=metrics.profitable_stocks_count,
                avg_return=metrics.avg_return,
                hit_rate=metrics.hit_rate
            )
        
        return environment_metrics
    
    def _classify_market_environment(self, market_returns: pd.Series) -> pd.Series:
        """
        Classify market environment based on returns and volatility
        
        Args:
            market_returns: Market return time series
            
        Returns:
            Series with market environment classifications
        """
        # Calculate rolling metrics (30-day window)
        rolling_return = market_returns.rolling(30).mean() * 252  # Annualized
        rolling_volatility = market_returns.rolling(30).std() * np.sqrt(252)  # Annualized
        
        # Initialize with 'sideways'
        environment = pd.Series('sideways', index=market_returns.index)
        
        # Classify based on returns and volatility
        bull_mask = rolling_return > self.bull_market_threshold
        bear_mask = rolling_return < self.bear_market_threshold
        volatile_mask = rolling_volatility > self.high_volatility_threshold
        
        environment.loc[bull_mask & ~volatile_mask] = 'bull'
        environment.loc[bear_mask & ~volatile_mask] = 'bear'
        environment.loc[volatile_mask] = 'volatile'
        
        return environment
    
    def generate_precision_report(
        self,
        metrics_history: List[PrecisionMetrics],
        environment_metrics: Optional[Dict[str, MarketEnvironmentMetrics]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive precision evaluation report
        
        Args:
            metrics_history: List of PrecisionMetrics from different time periods
            environment_metrics: Market environment specific metrics
            save_path: Path to save the report
            
        Returns:
            Comprehensive report dictionary
        """
        if not metrics_history:
            return {}
        
        # Overall statistics
        precisions = [m.precision for m in metrics_history]
        recalls = [m.recall for m in metrics_history]
        hit_rates = [m.hit_rate for m in metrics_history]
        
        # Target achievement analysis
        target_achieved_periods = sum(1 for p in precisions if p >= self.target_precision)
        target_achievement_rate = target_achieved_periods / len(precisions)
        
        report = {
            'summary': {
                'evaluation_periods': len(metrics_history),
                'target_precision': self.target_precision,
                'target_achievement_rate': target_achievement_rate,
                'avg_precision': np.mean(precisions),
                'min_precision': np.min(precisions),
                'max_precision': np.max(precisions),
                'precision_std': np.std(precisions),
                'avg_recall': np.mean(recalls),
                'avg_hit_rate': np.mean(hit_rates)
            },
            'detailed_metrics': {
                'precision_history': precisions,
                'recall_history': recalls,
                'hit_rate_history': hit_rates,
                'evaluation_dates': [m.evaluation_date for m in metrics_history]
            },
            'market_environment_analysis': {}
        }
        
        # Add market environment analysis
        if environment_metrics:
            for env_type, env_metrics in environment_metrics.items():
                report['market_environment_analysis'][env_type] = asdict(env_metrics)
        
        # Trading performance summary
        total_selections = sum(m.selected_stocks_count for m in metrics_history)
        total_profitable = sum(m.profitable_stocks_count for m in metrics_history)
        total_return = sum(m.total_return for m in metrics_history)
        
        report['trading_performance'] = {
            'total_stock_selections': total_selections,
            'total_profitable_selections': total_profitable,
            'overall_hit_rate': total_profitable / total_selections if total_selections > 0 else 0,
            'total_return': total_return,
            'avg_daily_selections': total_selections / len(metrics_history),
            'avg_daily_profitable': total_profitable / len(metrics_history)
        }
        
        # Log summary
        self.logger.info(f"Precision Report Summary:")
        self.logger.info(f"  Target Precision ≥ {self.target_precision}: {target_achievement_rate:.1%} of periods")
        self.logger.info(f"  Average Precision: {np.mean(precisions):.3f}")
        self.logger.info(f"  Overall Hit Rate: {report['trading_performance']['overall_hit_rate']:.1%}")
        
        # Save report if requested
        if save_path:
            self._save_report_to_file(report, save_path)
        
        return report
    
    def _save_report_to_file(self, report: Dict[str, Any], file_path: str) -> None:
        """Save report to JSON file"""
        import json
        from pathlib import Path
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        report_serializable = recursive_convert(report)
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Precision report saved to {file_path}")
    
    def plot_precision_analysis(
        self,
        metrics_history: List[PrecisionMetrics],
        environment_metrics: Optional[Dict[str, MarketEnvironmentMetrics]] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Plot comprehensive precision analysis
        
        Args:
            metrics_history: List of PrecisionMetrics
            environment_metrics: Market environment metrics
            figsize: Figure size
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            axes = axes.flatten()
            
            if not metrics_history:
                return
            
            # Extract data
            dates = [datetime.strptime(m.evaluation_date, '%Y-%m-%d') for m in metrics_history]
            precisions = [m.precision for m in metrics_history]
            recalls = [m.recall for m in metrics_history]
            hit_rates = [m.hit_rate for m in metrics_history]
            selections = [m.selected_stocks_count for m in metrics_history]
            
            # 1. Precision over time
            axes[0].plot(dates, precisions, 'b-o', alpha=0.7)
            axes[0].axhline(y=self.target_precision, color='red', linestyle='--', label=f'Target ({self.target_precision})')
            axes[0].set_title('Precision Over Time')
            axes[0].set_ylabel('Precision')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. Precision vs Recall
            axes[1].scatter(recalls, precisions, alpha=0.6)
            axes[1].axhline(y=self.target_precision, color='red', linestyle='--', alpha=0.7)
            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            axes[1].set_title('Precision vs Recall')
            axes[1].grid(True, alpha=0.3)
            
            # 3. Hit rate over time
            axes[2].plot(dates, hit_rates, 'g-o', alpha=0.7)
            axes[2].set_title('Hit Rate Over Time')
            axes[2].set_ylabel('Hit Rate')
            axes[2].grid(True, alpha=0.3)
            
            # 4. Stock selections per period
            axes[3].bar(range(len(selections)), selections, alpha=0.7)
            axes[3].set_title('Stock Selections Per Period')
            axes[3].set_ylabel('Number of Stocks')
            axes[3].grid(True, alpha=0.3)
            
            # 5. Precision distribution
            axes[4].hist(precisions, bins=15, alpha=0.7, edgecolor='black')
            axes[4].axvline(x=self.target_precision, color='red', linestyle='--', label=f'Target ({self.target_precision})')
            axes[4].set_title('Precision Distribution')
            axes[4].set_xlabel('Precision')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
            
            # 6. Market environment performance
            if environment_metrics:
                env_names = list(environment_metrics.keys())
                env_precisions = [environment_metrics[env].precision for env in env_names]
                
                bars = axes[5].bar(env_names, env_precisions, alpha=0.7)
                axes[5].axhline(y=self.target_precision, color='red', linestyle='--', alpha=0.7)
                axes[5].set_title('Precision by Market Environment')
                axes[5].set_ylabel('Precision')
                axes[5].grid(True, alpha=0.3)
                
                # Color bars based on target achievement
                for i, bar in enumerate(bars):
                    if env_precisions[i] >= self.target_precision:
                        bar.set_color('green')
                    else:
                        bar.set_color('orange')
            else:
                axes[5].text(0.5, 0.5, 'No environment\ndata available', 
                           ha='center', va='center', transform=axes[5].transAxes)
                axes[5].set_title('Market Environment Analysis')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")