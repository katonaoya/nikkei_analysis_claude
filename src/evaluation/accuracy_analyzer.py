"""
Prediction accuracy analysis system for detailed model performance evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, date, timedelta
from pathlib import Path
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from .metrics_calculator import MetricsCalculator
from utils.logger import get_logger
from utils.config import get_config


class AccuracyAnalyzer:
    """Comprehensive accuracy analysis system for prediction models"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize accuracy analyzer"""
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(f'analysis.{key}', value)
        
        self.logger = get_logger("accuracy_analyzer")
        self.metrics_calculator = MetricsCalculator()
        
        # Analysis parameters
        self.confidence_bins = self.config.get('analysis.confidence_bins', 10)
        self.time_window = self.config.get('analysis.time_window', 30)  # days
        
        # Results storage
        self.results_dir = self.config.get_data_dir('analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_temporal_accuracy(
        self,
        predictions: pd.DataFrame,
        actual_returns: pd.DataFrame,
        window_size: str = 'M'  # 'D', 'W', 'M', 'Q'
    ) -> Dict[str, Any]:
        """
        Analyze prediction accuracy over time periods
        
        Args:
            predictions: DataFrame with Date, Code, prediction_probability
            actual_returns: DataFrame with Date, Code, actual_return, target_achieved
            window_size: Time window for grouping ('D', 'W', 'M', 'Q')
            
        Returns:
            Temporal accuracy analysis results
        """
        self.logger.info("Starting temporal accuracy analysis",
                        predictions=len(predictions),
                        window_size=window_size)
        
        # Merge predictions with actual outcomes
        merged_data = self._merge_predictions_actuals(predictions, actual_returns)
        
        if merged_data.empty:
            return {'error': 'No matching prediction-actual pairs found'}
        
        # Group by time periods
        merged_data['period'] = merged_data['Date'].dt.to_period(window_size)
        
        temporal_results = {}
        period_metrics = []
        
        for period, period_data in merged_data.groupby('period'):
            if len(period_data) < 10:  # Skip periods with too few samples
                continue
            
            # Calculate metrics for this period
            metrics = self.metrics_calculator.calculate_basic_metrics(
                period_data['target_achieved'],
                period_data['prediction_probability']
            )
            
            metrics['period'] = str(period)
            metrics['sample_size'] = len(period_data)
            metrics['date_range'] = {
                'start': period_data['Date'].min(),
                'end': period_data['Date'].max()
            }
            
            period_metrics.append(metrics)
        
        if not period_metrics:
            return {'error': 'Insufficient data for temporal analysis'}
        
        temporal_df = pd.DataFrame(period_metrics)
        
        # Calculate trends and stability
        temporal_results = {
            'period_metrics': temporal_df,
            'trends': self._calculate_temporal_trends(temporal_df),
            'stability': self._calculate_temporal_stability(temporal_df),
            'summary_stats': self._get_temporal_summary_stats(temporal_df)
        }
        
        self.logger.info("Temporal accuracy analysis completed",
                        periods_analyzed=len(temporal_df))
        
        return temporal_results
    
    def analyze_stock_level_accuracy(
        self,
        predictions: pd.DataFrame,
        actual_returns: pd.DataFrame,
        stock_metadata: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Analyze prediction accuracy by individual stocks and sectors
        
        Args:
            predictions: DataFrame with Date, Code, prediction_probability
            actual_returns: DataFrame with Date, Code, actual_return, target_achieved
            stock_metadata: DataFrame with Code, sector, market_cap, etc.
            
        Returns:
            Stock-level accuracy analysis results
        """
        self.logger.info("Starting stock-level accuracy analysis")
        
        # Merge predictions with actual outcomes
        merged_data = self._merge_predictions_actuals(predictions, actual_returns)
        
        if merged_data.empty:
            return {'error': 'No matching prediction-actual pairs found'}
        
        # Add stock metadata if available
        if stock_metadata is not None:
            merged_data = merged_data.merge(stock_metadata, on='Code', how='left')
        
        stock_results = {}
        
        # Individual stock analysis
        stock_metrics = []
        for code, stock_data in merged_data.groupby('Code'):
            if len(stock_data) < 5:  # Skip stocks with too few predictions
                continue
            
            metrics = self.metrics_calculator.calculate_basic_metrics(
                stock_data['target_achieved'],
                stock_data['prediction_probability']
            )
            
            metrics['code'] = code
            metrics['predictions_count'] = len(stock_data)
            metrics['avg_probability'] = stock_data['prediction_probability'].mean()
            metrics['avg_actual_return'] = stock_data['actual_return'].mean()
            
            # Add metadata if available
            if stock_metadata is not None and 'sector' in stock_data.columns:
                metrics['sector'] = stock_data['sector'].iloc[0]
            
            stock_metrics.append(metrics)
        
        stock_df = pd.DataFrame(stock_metrics)
        stock_results['individual_stocks'] = stock_df
        
        # Sector analysis (if metadata available)
        if stock_metadata is not None and 'sector' in merged_data.columns:
            sector_results = self._analyze_sector_performance(merged_data)
            stock_results['sector_analysis'] = sector_results
        
        # Top/Bottom performers
        if not stock_df.empty:
            stock_results['top_performers'] = self._identify_top_bottom_stocks(stock_df, top=True)
            stock_results['bottom_performers'] = self._identify_top_bottom_stocks(stock_df, top=False)
        
        self.logger.info("Stock-level accuracy analysis completed",
                        stocks_analyzed=len(stock_df))
        
        return stock_results
    
    def analyze_market_condition_accuracy(
        self,
        predictions: pd.DataFrame,
        actual_returns: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Analyze prediction accuracy under different market conditions
        
        Args:
            predictions: DataFrame with Date, Code, prediction_probability
            actual_returns: DataFrame with Date, Code, actual_return, target_achieved
            market_data: DataFrame with Date, market_return, volatility, etc.
            
        Returns:
            Market condition accuracy analysis results
        """
        self.logger.info("Starting market condition accuracy analysis")
        
        # Merge predictions with actual outcomes
        merged_data = self._merge_predictions_actuals(predictions, actual_returns)
        
        if merged_data.empty:
            return {'error': 'No matching prediction-actual pairs found'}
        
        market_results = {}
        
        # Add market data if available
        if market_data is not None:
            daily_market = market_data.groupby('Date').first().reset_index()
            merged_data = merged_data.merge(daily_market, on='Date', how='left')
        
        # Volatility-based analysis
        volatility_analysis = self._analyze_by_volatility(merged_data)
        market_results['volatility_analysis'] = volatility_analysis
        
        # Market trend analysis
        if market_data is not None and 'market_return' in merged_data.columns:
            trend_analysis = self._analyze_by_market_trend(merged_data)
            market_results['trend_analysis'] = trend_analysis
        
        # Day of week analysis
        dow_analysis = self._analyze_by_day_of_week(merged_data)
        market_results['day_of_week_analysis'] = dow_analysis
        
        self.logger.info("Market condition accuracy analysis completed")
        
        return market_results
    
    def analyze_confidence_calibration(
        self,
        predictions: pd.DataFrame,
        actual_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze confidence calibration and reliability of prediction probabilities
        
        Args:
            predictions: DataFrame with Date, Code, prediction_probability
            actual_returns: DataFrame with Date, Code, actual_return, target_achieved
            
        Returns:
            Confidence calibration analysis results
        """
        self.logger.info("Starting confidence calibration analysis")
        
        # Merge predictions with actual outcomes
        merged_data = self._merge_predictions_actuals(predictions, actual_returns)
        
        if merged_data.empty:
            return {'error': 'No matching prediction-actual pairs found'}
        
        # Bin predictions by confidence level
        merged_data['confidence_bin'] = pd.cut(
            merged_data['prediction_probability'],
            bins=self.confidence_bins,
            labels=False
        )
        
        calibration_results = {}
        bin_analysis = []
        
        for bin_idx, bin_data in merged_data.groupby('confidence_bin'):
            if len(bin_data) < 5:
                continue
            
            prob_range = merged_data[merged_data['confidence_bin'] == bin_idx]['prediction_probability']
            actual_rate = bin_data['target_achieved'].mean()
            avg_predicted_prob = prob_range.mean()
            
            bin_info = {
                'confidence_bin': bin_idx,
                'prob_range_min': prob_range.min(),
                'prob_range_max': prob_range.max(),
                'avg_predicted_prob': avg_predicted_prob,
                'actual_success_rate': actual_rate,
                'sample_size': len(bin_data),
                'calibration_error': abs(avg_predicted_prob - actual_rate)
            }
            
            bin_analysis.append(bin_info)
        
        calibration_df = pd.DataFrame(bin_analysis)
        
        # Overall calibration metrics
        if not calibration_df.empty:
            calibration_results['bin_analysis'] = calibration_df
            calibration_results['overall_calibration_error'] = calibration_df['calibration_error'].mean()
            calibration_results['max_calibration_error'] = calibration_df['calibration_error'].max()
            
            # Reliability metrics
            calibration_results['reliability_metrics'] = self._calculate_reliability_metrics(merged_data)
        
        self.logger.info("Confidence calibration analysis completed")
        
        return calibration_results
    
    def _merge_predictions_actuals(
        self,
        predictions: pd.DataFrame,
        actual_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge predictions with actual outcomes"""
        
        # Ensure datetime columns
        predictions = predictions.copy()
        actual_returns = actual_returns.copy()
        
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        actual_returns['Date'] = pd.to_datetime(actual_returns['Date'])
        
        # Merge on Date and Code
        merged = predictions.merge(
            actual_returns,
            on=['Date', 'Code'],
            how='inner'
        )
        
        # Standardize column names for consistency
        # Check for various return column names and standardize to 'actual_return'
        return_columns = ['actual_return', 'return_1d', 'high_return_1d', 'return', 'daily_return']
        for col in return_columns:
            if col in merged.columns and 'actual_return' not in merged.columns:
                merged['actual_return'] = merged[col]
                break
        
        return merged
    
    def _calculate_temporal_trends(self, temporal_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trends in temporal metrics"""
        trends = {}
        
        key_metrics = ['precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in key_metrics:
            if metric in temporal_df.columns:
                # Linear trend (correlation with time)
                time_index = range(len(temporal_df))
                correlation = np.corrcoef(time_index, temporal_df[metric])[0, 1]
                
                # Slope (simple linear regression)
                if len(temporal_df) > 1:
                    slope = np.polyfit(time_index, temporal_df[metric], 1)[0]
                else:
                    slope = 0
                
                trends[metric] = {
                    'correlation_with_time': correlation,
                    'slope': slope,
                    'trend_direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
                }
        
        return trends
    
    def _calculate_temporal_stability(self, temporal_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate stability metrics for temporal performance"""
        stability = {}
        
        key_metrics = ['precision', 'recall', 'f1_score']
        
        for metric in key_metrics:
            if metric in temporal_df.columns:
                stability[metric] = {
                    'mean': temporal_df[metric].mean(),
                    'std': temporal_df[metric].std(),
                    'cv': temporal_df[metric].std() / temporal_df[metric].mean() if temporal_df[metric].mean() > 0 else np.inf,
                    'range': temporal_df[metric].max() - temporal_df[metric].min()
                }
        
        return stability
    
    def _get_temporal_summary_stats(self, temporal_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for temporal analysis"""
        return {
            'periods_analyzed': len(temporal_df),
            'avg_sample_size': temporal_df['sample_size'].mean(),
            'total_samples': temporal_df['sample_size'].sum(),
            'date_range': {
                'start': temporal_df['period'].min(),
                'end': temporal_df['period'].max()
            }
        }
    
    def _analyze_sector_performance(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by sector"""
        sector_metrics = []
        
        for sector, sector_data in merged_data.groupby('sector'):
            if len(sector_data) < 10:
                continue
            
            metrics = self.metrics_calculator.calculate_basic_metrics(
                sector_data['target_achieved'],
                sector_data['prediction_probability']
            )
            
            metrics['sector'] = sector
            metrics['sample_size'] = len(sector_data)
            metrics['avg_probability'] = sector_data['prediction_probability'].mean()
            
            sector_metrics.append(metrics)
        
        return pd.DataFrame(sector_metrics)
    
    def _identify_top_bottom_stocks(self, stock_df: pd.DataFrame, top: bool = True, n: int = 5) -> pd.DataFrame:
        """Identify top/bottom performing stocks by F1 score"""
        if stock_df.empty:
            return pd.DataFrame()
        
        sorted_stocks = stock_df.sort_values('f1_score', ascending=not top)
        return sorted_stocks.head(n)
    
    def _analyze_by_volatility(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by volatility levels"""
        
        # Calculate rolling volatility if not available
        if 'volatility' not in merged_data.columns:
            # Simple proxy: rolling standard deviation of returns
            merged_data['volatility'] = merged_data.groupby('Code')['actual_return'].rolling(
                window=20, min_periods=5
            ).std().reset_index(drop=True)
        
        # Categorize volatility
        vol_quantiles = merged_data['volatility'].quantile([0.33, 0.67])
        merged_data['volatility_regime'] = 'medium'
        merged_data.loc[merged_data['volatility'] <= vol_quantiles.iloc[0], 'volatility_regime'] = 'low'
        merged_data.loc[merged_data['volatility'] >= vol_quantiles.iloc[1], 'volatility_regime'] = 'high'
        
        vol_results = []
        for regime, regime_data in merged_data.groupby('volatility_regime'):
            if len(regime_data) < 10:
                continue
            
            metrics = self.metrics_calculator.calculate_basic_metrics(
                regime_data['target_achieved'],
                regime_data['prediction_probability']
            )
            
            metrics['volatility_regime'] = regime
            metrics['sample_size'] = len(regime_data)
            metrics['avg_volatility'] = regime_data['volatility'].mean()
            
            vol_results.append(metrics)
        
        return pd.DataFrame(vol_results)
    
    def _analyze_by_market_trend(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by market trend"""
        
        # Categorize market conditions
        merged_data['market_regime'] = 'neutral'
        merged_data.loc[merged_data['market_return'] > 0.01, 'market_regime'] = 'bull'
        merged_data.loc[merged_data['market_return'] < -0.01, 'market_regime'] = 'bear'
        
        trend_results = []
        for regime, regime_data in merged_data.groupby('market_regime'):
            if len(regime_data) < 10:
                continue
            
            metrics = self.metrics_calculator.calculate_basic_metrics(
                regime_data['target_achieved'],
                regime_data['prediction_probability']
            )
            
            metrics['market_regime'] = regime
            metrics['sample_size'] = len(regime_data)
            metrics['avg_market_return'] = regime_data['market_return'].mean()
            
            trend_results.append(metrics)
        
        return pd.DataFrame(trend_results)
    
    def _analyze_by_day_of_week(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by day of week"""
        merged_data['day_of_week'] = merged_data['Date'].dt.day_name()
        
        dow_results = []
        for dow, dow_data in merged_data.groupby('day_of_week'):
            if len(dow_data) < 5:
                continue
            
            metrics = self.metrics_calculator.calculate_basic_metrics(
                dow_data['target_achieved'],
                dow_data['prediction_probability']
            )
            
            metrics['day_of_week'] = dow
            metrics['sample_size'] = len(dow_data)
            
            dow_results.append(metrics)
        
        return pd.DataFrame(dow_results)
    
    def _calculate_reliability_metrics(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate additional reliability metrics"""
        
        # Confidence vs accuracy correlation
        conf_acc_corr = np.corrcoef(
            merged_data['prediction_probability'],
            merged_data['target_achieved']
        )[0, 1]
        
        # High confidence subset performance
        high_conf_threshold = merged_data['prediction_probability'].quantile(0.8)
        high_conf_data = merged_data[merged_data['prediction_probability'] >= high_conf_threshold]
        
        high_conf_accuracy = high_conf_data['target_achieved'].mean() if len(high_conf_data) > 0 else 0
        
        return {
            'confidence_accuracy_correlation': conf_acc_corr,
            'high_confidence_threshold': high_conf_threshold,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_sample_size': len(high_conf_data)
        }
    
    def generate_comprehensive_report(
        self,
        predictions: pd.DataFrame,
        actual_returns: pd.DataFrame,
        stock_metadata: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive accuracy analysis report"""
        
        if save_path is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.results_dir / f"accuracy_analysis_report_{timestamp}.txt"
        
        self.logger.info("Generating comprehensive accuracy analysis report")
        
        # Run all analyses
        temporal_analysis = self.analyze_temporal_accuracy(predictions, actual_returns)
        stock_analysis = self.analyze_stock_level_accuracy(predictions, actual_returns, stock_metadata)
        market_analysis = self.analyze_market_condition_accuracy(predictions, actual_returns, market_data)
        calibration_analysis = self.analyze_confidence_calibration(predictions, actual_returns)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE ACCURACY ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Temporal Analysis
            if 'error' not in temporal_analysis:
                f.write("TEMPORAL ACCURACY ANALYSIS\n")
                f.write("-" * 30 + "\n")
                summary = temporal_analysis['summary_stats']
                f.write(f"Periods Analyzed: {summary['periods_analyzed']}\n")
                f.write(f"Total Samples: {summary['total_samples']}\n")
                f.write(f"Average Sample Size: {summary['avg_sample_size']:.1f}\n\n")
                
                stability = temporal_analysis['stability']
                f.write("Metric Stability:\n")
                for metric, stats in stability.items():
                    f.write(f"  {metric}: Mean={stats['mean']:.3f}, CV={stats['cv']:.3f}\n")
                f.write("\n")
            
            # Stock-Level Analysis
            if 'error' not in stock_analysis:
                f.write("STOCK-LEVEL ACCURACY ANALYSIS\n")
                f.write("-" * 32 + "\n")
                
                if 'top_performers' in stock_analysis:
                    f.write("Top 5 Performing Stocks (by F1 Score):\n")
                    for _, stock in stock_analysis['top_performers'].iterrows():
                        f.write(f"  {stock['code']}: F1={stock['f1_score']:.3f}, Precision={stock['precision']:.3f}\n")
                    f.write("\n")
            
            # Market Condition Analysis
            if 'error' not in market_analysis:
                f.write("MARKET CONDITION ANALYSIS\n")
                f.write("-" * 27 + "\n")
                
                if 'volatility_analysis' in market_analysis:
                    vol_df = market_analysis['volatility_analysis']
                    f.write("Performance by Volatility Regime:\n")
                    for _, row in vol_df.iterrows():
                        f.write(f"  {row['volatility_regime']}: Precision={row['precision']:.3f}, F1={row['f1_score']:.3f}\n")
                    f.write("\n")
            
            # Calibration Analysis
            if 'error' not in calibration_analysis:
                f.write("CONFIDENCE CALIBRATION ANALYSIS\n")
                f.write("-" * 35 + "\n")
                f.write(f"Overall Calibration Error: {calibration_analysis['overall_calibration_error']:.4f}\n")
                f.write(f"Maximum Calibration Error: {calibration_analysis['max_calibration_error']:.4f}\n\n")
                
                if 'reliability_metrics' in calibration_analysis:
                    rel_metrics = calibration_analysis['reliability_metrics']
                    f.write(f"Confidence-Accuracy Correlation: {rel_metrics['confidence_accuracy_correlation']:.3f}\n")
                    f.write(f"High Confidence Accuracy: {rel_metrics['high_confidence_accuracy']:.1%}\n")
        
        self.logger.info(f"Comprehensive accuracy analysis report saved to {save_path}")
        
        return str(save_path)
    
    def plot_accuracy_analysis(
        self,
        predictions: pd.DataFrame,
        actual_returns: pd.DataFrame,
        stock_metadata: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12)
    ) -> Optional[str]:
        """Plot comprehensive accuracy analysis visualizations"""
        
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Matplotlib not available for plotting")
            return None
        
        self.logger.info("Generating accuracy analysis visualizations")
        
        # Run analyses
        temporal_analysis = self.analyze_temporal_accuracy(predictions, actual_returns, window_size='M')
        stock_analysis = self.analyze_stock_level_accuracy(predictions, actual_returns, stock_metadata)
        market_analysis = self.analyze_market_condition_accuracy(predictions, actual_returns, market_data)
        calibration_analysis = self.analyze_confidence_calibration(predictions, actual_returns)
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. Temporal Accuracy Trends
        if 'error' not in temporal_analysis and 'monthly_metrics' in temporal_analysis:
            monthly_df = temporal_analysis['monthly_metrics']
            if not monthly_df.empty:
                axes[0, 0].plot(monthly_df.index, monthly_df['precision'], 'o-', label='Precision', linewidth=2)
                axes[0, 0].plot(monthly_df.index, monthly_df['recall'], 's-', label='Recall', linewidth=2)
                axes[0, 0].plot(monthly_df.index, monthly_df['f1_score'], '^-', label='F1 Score', linewidth=2)
                axes[0, 0].set_title('Temporal Accuracy Trends')
                axes[0, 0].set_ylabel('Score')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Sample Size Distribution Over Time
        if 'error' not in temporal_analysis and 'monthly_metrics' in temporal_analysis:
            monthly_df = temporal_analysis['monthly_metrics']
            if not monthly_df.empty:
                axes[0, 1].bar(range(len(monthly_df)), monthly_df['sample_size'], alpha=0.7)
                axes[0, 1].set_title('Monthly Sample Size Distribution')
                axes[0, 1].set_ylabel('Sample Size')
                axes[0, 1].set_xticks(range(0, len(monthly_df), max(1, len(monthly_df)//6)))
                axes[0, 1].set_xticklabels([str(monthly_df.index[i]) for i in range(0, len(monthly_df), max(1, len(monthly_df)//6))], rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Stock-Level Performance Distribution
        if 'error' not in stock_analysis and 'stock_performance' in stock_analysis:
            stock_perf = stock_analysis['stock_performance']
            if not stock_perf.empty:
                axes[1, 0].hist(stock_perf['precision'], bins=20, alpha=0.7, label='Precision', edgecolor='black')
                axes[1, 0].hist(stock_perf['f1_score'], bins=20, alpha=0.7, label='F1 Score', edgecolor='black')
                axes[1, 0].set_title('Stock-Level Performance Distribution')
                axes[1, 0].set_xlabel('Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Market Volatility vs Performance
        if 'error' not in market_analysis and 'volatility_analysis' in market_analysis:
            vol_df = market_analysis['volatility_analysis']
            if not vol_df.empty:
                x_pos = range(len(vol_df))
                width = 0.35
                axes[1, 1].bar([x - width/2 for x in x_pos], vol_df['precision'], width, label='Precision', alpha=0.8)
                axes[1, 1].bar([x + width/2 for x in x_pos], vol_df['f1_score'], width, label='F1 Score', alpha=0.8)
                axes[1, 1].set_title('Performance by Market Volatility')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels(vol_df['volatility_regime'], rotation=45)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Confidence Calibration
        if 'error' not in calibration_analysis and 'calibration_data' in calibration_analysis:
            cal_df = calibration_analysis['calibration_data']
            if not cal_df.empty:
                # Plot calibration curve
                axes[2, 0].plot(cal_df['confidence_bin_center'], cal_df['actual_accuracy'], 'o-', linewidth=2, markersize=8, label='Model')
                axes[2, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.7, label='Perfect Calibration')
                axes[2, 0].set_title('Confidence Calibration Curve')
                axes[2, 0].set_xlabel('Mean Predicted Confidence')
                axes[2, 0].set_ylabel('Actual Accuracy')
                axes[2, 0].legend()
                axes[2, 0].grid(True, alpha=0.3)
                axes[2, 0].set_xlim(0, 1)
                axes[2, 0].set_ylim(0, 1)
        
        # 6. Key Statistics Summary
        axes[2, 1].axis('off')
        
        # Compile key statistics
        stats_text = "KEY ACCURACY STATISTICS\n" + "="*25 + "\n\n"
        
        if 'error' not in temporal_analysis and 'stability' in temporal_analysis:
            stability = temporal_analysis['stability']
            stats_text += f"Temporal Stability:\n"
            stats_text += f"  Precision CV: {stability['precision']['cv']:.3f}\n"
            stats_text += f"  F1 Score CV: {stability['f1_score']['cv']:.3f}\n\n"
        
        if 'error' not in stock_analysis and 'summary_stats' in stock_analysis:
            summary = stock_analysis['summary_stats']
            stats_text += f"Stock Coverage:\n"
            stats_text += f"  Stocks Analyzed: {summary['stocks_analyzed']}\n"
            stats_text += f"  Avg Predictions/Stock: {summary['avg_predictions_per_stock']:.1f}\n\n"
        
        if 'error' not in calibration_analysis:
            stats_text += f"Calibration Quality:\n"
            stats_text += f"  Overall Error: {calibration_analysis['overall_calibration_error']:.4f}\n"
            stats_text += f"  Max Error: {calibration_analysis['max_calibration_error']:.4f}\n"
        
        axes[2, 1].text(0.05, 0.95, stats_text, transform=axes[2, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.results_dir / f"accuracy_analysis_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Accuracy analysis plots saved to {save_path}")
        
        return str(save_path)