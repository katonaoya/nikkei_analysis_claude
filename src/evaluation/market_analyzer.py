"""
Market environment analysis for stock prediction performance evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

from utils.logger import get_logger
from utils.config import get_config


@dataclass
class MarketPeriod:
    """Market period classification data"""
    start_date: str
    end_date: str
    period_type: str
    market_return: float
    volatility: float
    duration_days: int


@dataclass
class SectorAnalysis:
    """Sector-specific analysis results"""
    sector: str
    precision: float
    recall: float
    hit_rate: float
    avg_return: float
    stock_count: int
    selected_count: int


class MarketAnalyzer:
    """Advanced market environment analysis for precision optimization"""
    
    def __init__(self):
        """Initialize market analyzer"""
        self.config = get_config()
        self.logger = get_logger("market_analyzer")
        
        # Market environment classification thresholds
        self.bull_threshold = 0.15      # >15% annual return
        self.bear_threshold = -0.10     # <-10% annual return
        self.high_vol_threshold = 0.25  # >25% annual volatility
        self.sideways_vol_threshold = 0.15  # <15% annual volatility
        
        # Analysis windows
        self.short_window = 30   # 30 days
        self.medium_window = 90  # 90 days
        self.long_window = 252   # 1 year
    
    def classify_market_regime(
        self, 
        market_data: pd.DataFrame,
        date_col: str = 'date',
        price_col: str = 'close_price',
        volume_col: Optional[str] = 'volume'
    ) -> pd.DataFrame:
        """
        Classify market regimes based on returns and volatility
        
        Args:
            market_data: Market data with date, price, and optionally volume
            date_col: Date column name
            price_col: Price column name
            volume_col: Volume column name (optional)
            
        Returns:
            DataFrame with market regime classifications
        """
        df = market_data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Calculate returns
        df['daily_return'] = df[price_col].pct_change()
        
        # Calculate rolling metrics
        for window in [self.short_window, self.medium_window, self.long_window]:
            # Annualized return and volatility
            df[f'return_{window}d'] = df['daily_return'].rolling(window).mean() * 252
            df[f'volatility_{window}d'] = df['daily_return'].rolling(window).std() * np.sqrt(252)
            
            # Trend indicators
            df[f'sma_{window}'] = df[price_col].rolling(window).mean()
            df[f'price_vs_sma_{window}'] = (df[price_col] / df[f'sma_{window}'] - 1) * 100
        
        # Volume analysis if available
        if volume_col and volume_col in df.columns:
            df['avg_volume_30d'] = df[volume_col].rolling(30).mean()
            df['volume_ratio'] = df[volume_col] / df['avg_volume_30d']
            df['volume_spike'] = df['volume_ratio'] > 1.5
        
        # Classify market regimes (using medium window as primary)
        conditions = [
            (df[f'return_{self.medium_window}d'] > self.bull_threshold) & 
            (df[f'volatility_{self.medium_window}d'] <= self.high_vol_threshold),
            
            (df[f'return_{self.medium_window}d'] < self.bear_threshold) & 
            (df[f'volatility_{self.medium_window}d'] <= self.high_vol_threshold),
            
            df[f'volatility_{self.medium_window}d'] > self.high_vol_threshold,
            
            (df[f'volatility_{self.medium_window}d'] <= self.sideways_vol_threshold) &
            (df[f'return_{self.medium_window}d'].abs() <= 0.05)  # Low return, low vol
        ]
        
        choices = ['bull', 'bear', 'volatile', 'sideways']
        df['market_regime'] = np.select(conditions, choices, default='transitional')
        
        # Add regime strength score
        df['regime_strength'] = self._calculate_regime_strength(df)
        
        self.logger.info(f"Market regime classification completed for {len(df)} periods")
        
        return df
    
    def _calculate_regime_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime strength score (0-1)"""
        window = self.medium_window
        
        # Normalize metrics to 0-1 range
        abs_return = df[f'return_{window}d'].abs()
        norm_return = abs_return / (abs_return.max() + 1e-8)
        
        volatility = df[f'volatility_{window}d']
        norm_vol = volatility / (volatility.max() + 1e-8)
        
        # Strength based on regime type
        strength = pd.Series(0.5, index=df.index)  # Default moderate strength
        
        # Bull/Bear: higher return = higher strength
        bull_bear_mask = df['market_regime'].isin(['bull', 'bear'])
        strength.loc[bull_bear_mask] = norm_return.loc[bull_bear_mask]
        
        # Volatile: higher volatility = higher strength
        volatile_mask = df['market_regime'] == 'volatile'
        strength.loc[volatile_mask] = norm_vol.loc[volatile_mask]
        
        # Sideways: lower volatility = higher strength
        sideways_mask = df['market_regime'] == 'sideways'
        strength.loc[sideways_mask] = 1 - norm_vol.loc[sideways_mask]
        
        return strength.clip(0, 1)
    
    def analyze_performance_by_regime(
        self,
        prediction_data: pd.DataFrame,
        market_regimes: pd.DataFrame,
        date_col: str = 'date',
        target_col: str = 'target',
        pred_proba_col: str = 'pred_proba',
        return_col: str = 'return',
        sector_col: Optional[str] = 'sector'
    ) -> Dict[str, Any]:
        """
        Analyze prediction performance by market regime
        
        Args:
            prediction_data: DataFrame with predictions and outcomes
            market_regimes: DataFrame with market regime classifications
            date_col: Date column name
            target_col: Target column name
            pred_proba_col: Prediction probability column name
            return_col: Return column name
            sector_col: Sector column name (optional)
            
        Returns:
            Dictionary with performance analysis by regime
        """
        # Merge prediction data with market regimes
        pred_df = prediction_data.copy()
        market_df = market_regimes.copy()
        
        pred_df[date_col] = pd.to_datetime(pred_df[date_col])
        market_df[date_col] = pd.to_datetime(market_df[date_col])
        
        merged_df = pd.merge(pred_df, market_df[[date_col, 'market_regime', 'regime_strength']], 
                            on=date_col, how='left')
        
        # Remove rows without regime classification
        merged_df = merged_df.dropna(subset=['market_regime'])
        
        results = {
            'regime_performance': {},
            'regime_periods': {},
            'overall_stats': {},
            'sector_analysis': {}
        }
        
        # Analyze performance by regime
        for regime in merged_df['market_regime'].unique():
            regime_data = merged_df[merged_df['market_regime'] == regime]
            
            if len(regime_data) == 0:
                continue
            
            # Calculate metrics
            regime_metrics = self._calculate_regime_metrics(
                regime_data, target_col, pred_proba_col, return_col
            )
            
            results['regime_performance'][regime] = regime_metrics
        
        # Identify regime periods
        results['regime_periods'] = self._identify_regime_periods(market_regimes, date_col)
        
        # Overall statistics
        results['overall_stats'] = {
            'total_periods': len(merged_df),
            'regime_distribution': merged_df['market_regime'].value_counts().to_dict(),
            'avg_regime_strength': merged_df.groupby('market_regime')['regime_strength'].mean().to_dict()
        }
        
        # Sector analysis if available
        if sector_col and sector_col in merged_df.columns:
            results['sector_analysis'] = self._analyze_by_sector(
                merged_df, sector_col, target_col, pred_proba_col, return_col
            )
        
        self.logger.info(f"Market regime analysis completed for {len(merged_df)} predictions")
        
        return results
    
    def _calculate_regime_metrics(
        self,
        regime_data: pd.DataFrame,
        target_col: str,
        pred_proba_col: str,
        return_col: str,
        threshold: float = 0.75
    ) -> Dict[str, float]:
        """Calculate performance metrics for a specific regime"""
        from sklearn.metrics import precision_score, recall_score, roc_auc_score
        
        y_true = regime_data[target_col].values
        y_pred_proba = regime_data[pred_proba_col].values
        
        # Binary predictions using threshold
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        # Basic metrics
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            auc = 0.5
        
        # Trading metrics
        selected_count = y_pred_binary.sum()
        profitable_count = (y_pred_binary & y_true).sum()
        hit_rate = profitable_count / selected_count if selected_count > 0 else 0
        
        # Return metrics
        if return_col in regime_data.columns:
            selected_returns = regime_data.loc[y_pred_binary.astype(bool), return_col]
            avg_return = selected_returns.mean() if len(selected_returns) > 0 else 0
            total_return = selected_returns.sum() if len(selected_returns) > 0 else 0
        else:
            avg_return = hit_rate * 0.01  # Assume 1% for profitable
            total_return = avg_return * selected_count
        
        return {
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'hit_rate': hit_rate,
            'selected_count': int(selected_count),
            'profitable_count': int(profitable_count),
            'avg_return': avg_return,
            'total_return': total_return,
            'sample_count': len(regime_data)
        }
    
    def _identify_regime_periods(
        self, 
        market_data: pd.DataFrame, 
        date_col: str
    ) -> List[MarketPeriod]:
        """Identify continuous periods of each market regime"""
        df = market_data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        periods = []
        current_regime = None
        start_date = None
        
        for idx, row in df.iterrows():
            regime = row['market_regime']
            date = row[date_col]
            
            if regime != current_regime:
                # End previous period
                if current_regime is not None and start_date is not None:
                    end_date = df.loc[idx - 1, date_col]
                    period_data = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
                    
                    if len(period_data) > 0:
                        market_return = period_data['daily_return'].mean() * 252 if 'daily_return' in period_data.columns else 0
                        volatility = period_data['daily_return'].std() * np.sqrt(252) if 'daily_return' in period_data.columns else 0
                        
                        periods.append(MarketPeriod(
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d'),
                            period_type=current_regime,
                            market_return=market_return,
                            volatility=volatility,
                            duration_days=len(period_data)
                        ))
                
                # Start new period
                current_regime = regime
                start_date = date
        
        # Handle final period
        if current_regime is not None and start_date is not None:
            end_date = df[date_col].iloc[-1]
            period_data = df[df[date_col] >= start_date]
            
            if len(period_data) > 0:
                market_return = period_data['daily_return'].mean() * 252 if 'daily_return' in period_data.columns else 0
                volatility = period_data['daily_return'].std() * np.sqrt(252) if 'daily_return' in period_data.columns else 0
                
                periods.append(MarketPeriod(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    period_type=current_regime,
                    market_return=market_return,
                    volatility=volatility,
                    duration_days=len(period_data)
                ))
        
        return periods
    
    def _analyze_by_sector(
        self,
        data: pd.DataFrame,
        sector_col: str,
        target_col: str,
        pred_proba_col: str,
        return_col: str
    ) -> Dict[str, SectorAnalysis]:
        """Analyze performance by sector"""
        from sklearn.metrics import precision_score, recall_score
        
        sector_results = {}
        
        for sector in data[sector_col].unique():
            if pd.isna(sector):
                continue
            
            sector_data = data[data[sector_col] == sector]
            
            if len(sector_data) == 0:
                continue
            
            y_true = sector_data[target_col].values
            y_pred_proba = sector_data[pred_proba_col].values
            y_pred_binary = (y_pred_proba >= 0.75).astype(int)
            
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            
            selected_count = y_pred_binary.sum()
            profitable_count = (y_pred_binary & y_true).sum()
            hit_rate = profitable_count / selected_count if selected_count > 0 else 0
            
            if return_col in sector_data.columns:
                selected_returns = sector_data.loc[y_pred_binary.astype(bool), return_col]
                avg_return = selected_returns.mean() if len(selected_returns) > 0 else 0
            else:
                avg_return = hit_rate * 0.01
            
            sector_results[sector] = SectorAnalysis(
                sector=sector,
                precision=precision,
                recall=recall,
                hit_rate=hit_rate,
                avg_return=avg_return,
                stock_count=len(sector_data),
                selected_count=int(selected_count)
            )
        
        return sector_results
    
    def generate_market_report(
        self,
        analysis_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive market analysis report
        
        Args:
            analysis_results: Results from analyze_performance_by_regime
            save_path: Path to save report
            
        Returns:
            Formatted report dictionary
        """
        report = {
            'market_regime_summary': {},
            'best_performing_regimes': [],
            'worst_performing_regimes': [],
            'regime_transitions': {},
            'sector_insights': {},
            'recommendations': []
        }
        
        # Regime performance summary
        regime_perf = analysis_results.get('regime_performance', {})
        
        for regime, metrics in regime_perf.items():
            report['market_regime_summary'][regime] = {
                'precision': f"{metrics['precision']:.3f}",
                'hit_rate': f"{metrics['hit_rate']:.1%}",
                'avg_return': f"{metrics['avg_return']:.2%}",
                'sample_count': metrics['sample_count'],
                'meets_target': metrics['precision'] >= 0.75
            }
        
        # Identify best and worst performing regimes
        if regime_perf:
            sorted_regimes = sorted(regime_perf.items(), 
                                  key=lambda x: x[1]['precision'], reverse=True)
            
            report['best_performing_regimes'] = [
                {
                    'regime': regime,
                    'precision': f"{metrics['precision']:.3f}",
                    'hit_rate': f"{metrics['hit_rate']:.1%}"
                }
                for regime, metrics in sorted_regimes[:2]
            ]
            
            report['worst_performing_regimes'] = [
                {
                    'regime': regime,
                    'precision': f"{metrics['precision']:.3f}",
                    'hit_rate': f"{metrics['hit_rate']:.1%}"
                }
                for regime, metrics in sorted_regimes[-2:]
            ]
        
        # Sector insights
        sector_analysis = analysis_results.get('sector_analysis', {})
        if sector_analysis:
            best_sectors = sorted(sector_analysis.items(), 
                                key=lambda x: x[1].precision, reverse=True)[:3]
            
            report['sector_insights'] = {
                'top_sectors': [
                    {
                        'sector': sector_analysis[sector].sector,
                        'precision': f"{sector_analysis[sector].precision:.3f}",
                        'hit_rate': f"{sector_analysis[sector].hit_rate:.1%}"
                    }
                    for sector, _ in best_sectors
                ]
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(analysis_results)
        
        # Log summary
        self.logger.info("Market Analysis Report Generated:")
        for regime, summary in report['market_regime_summary'].items():
            meets_target = "✓" if summary['meets_target'] else "✗"
            self.logger.info(f"  {regime}: Precision {summary['precision']} {meets_target}")
        
        # Save report
        if save_path:
            self._save_market_report(report, save_path)
        
        return report
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        regime_perf = analysis_results.get('regime_performance', {})
        
        # Check which regimes meet target precision
        high_precision_regimes = [
            regime for regime, metrics in regime_perf.items()
            if metrics['precision'] >= 0.75
        ]
        
        low_precision_regimes = [
            regime for regime, metrics in regime_perf.items()
            if metrics['precision'] < 0.50
        ]
        
        if high_precision_regimes:
            recommendations.append(
                f"Focus trading during {', '.join(high_precision_regimes)} market conditions "
                f"where precision consistently meets target (≥0.75)"
            )
        
        if low_precision_regimes:
            recommendations.append(
                f"Reduce or avoid trading during {', '.join(low_precision_regimes)} market conditions "
                f"where precision is consistently low"
            )
        
        # Check for regime-specific patterns
        if 'volatile' in regime_perf and regime_perf['volatile']['precision'] < 0.50:
            recommendations.append(
                "Consider implementing volatility filters or adjusting position sizing during volatile periods"
            )
        
        # Overall recommendations
        avg_precision = np.mean([m['precision'] for m in regime_perf.values()])
        if avg_precision < 0.75:
            recommendations.append(
                "Overall precision is below target. Consider model retraining or feature engineering improvements"
            )
        
        return recommendations
    
    def _save_market_report(self, report: Dict[str, Any], file_path: str) -> None:
        """Save market report to JSON file"""
        import json
        from pathlib import Path
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Market analysis report saved to {file_path}")