"""
Stock extraction system with threshold and filtering rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, date
import warnings

from utils.logger import get_logger
from utils.config import get_config


class StockExtractor:
    """Stock extraction system with configurable rules"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize stock extractor"""
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(f'extraction.{key}', value)
        
        self.logger = get_logger("stock_extractor")
        
        # Extraction parameters from requirements
        self.base_threshold = self.config.get('extraction.base_threshold', 0.85)
        self.max_selections = self.config.get('extraction.max_selections', 3)
        self.min_selections = self.config.get('extraction.min_selections', 0)
        
        # Market condition adjustments
        self.high_volatility_adjustment = self.config.get('extraction.high_volatility_adjustment', -0.05)
        self.low_volatility_bonus = self.config.get('extraction.low_volatility_bonus', 0.02)
        
        # Volatility thresholds
        self.high_volatility_threshold = self.config.get('extraction.high_volatility_threshold', 0.03)
        self.low_volatility_threshold = self.config.get('extraction.low_volatility_threshold', 0.015)
        
        # Additional filters
        self.min_volume_threshold = self.config.get('extraction.min_volume_threshold', 100000)
        self.max_gap_threshold = self.config.get('extraction.max_gap_threshold', 0.05)
        self.exclude_penny_stocks = self.config.get('extraction.exclude_penny_stocks', True)
        self.min_price = self.config.get('extraction.min_price', 100)  # JPY
        
        # Market environment factors
        self.market_sentiment_weight = self.config.get('extraction.market_sentiment_weight', 0.1)
        
        # Extraction metadata
        self.extraction_info = {}
        
    def extract_stocks(
        self,
        predictions: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        custom_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Extract stocks based on threshold and filtering rules
        
        Args:
            predictions: Predictions DataFrame with probabilities
            market_data: Market environment data (VIX, etc.)
            custom_threshold: Override base threshold
            
        Returns:
            Extracted stocks DataFrame
        """
        self.logger.info("Starting stock extraction",
                        total_stocks=len(predictions),
                        base_threshold=self.base_threshold)
        
        # Calculate dynamic threshold
        effective_threshold = self._calculate_dynamic_threshold(predictions, market_data, custom_threshold)
        
        # Apply base filtering
        filtered_predictions = self._apply_base_filters(predictions)
        
        # Apply threshold filtering
        candidate_stocks = self._apply_threshold_filter(filtered_predictions, effective_threshold)
        
        # Apply ranking and selection
        selected_stocks = self._apply_ranking_selection(candidate_stocks)
        
        # Add extraction metadata
        selected_stocks = self._add_extraction_metadata(selected_stocks, effective_threshold)
        
        # Store extraction info
        self.extraction_info = {
            'extraction_date': datetime.now(),
            'total_stocks_analyzed': len(predictions),
            'stocks_after_filters': len(filtered_predictions),
            'candidate_stocks': len(candidate_stocks),
            'final_selections': len(selected_stocks),
            'effective_threshold': effective_threshold,
            'base_threshold': self.base_threshold,
            'threshold_adjustment': effective_threshold - self.base_threshold
        }
        
        self.logger.info("Stock extraction completed",
                        extracted_stocks=len(selected_stocks),
                        effective_threshold=f"{effective_threshold:.3f}",
                        threshold_adjustment=f"{effective_threshold - self.base_threshold:+.3f}")
        
        return selected_stocks
    
    def _calculate_dynamic_threshold(
        self,
        predictions: pd.DataFrame,
        market_data: Optional[pd.DataFrame],
        custom_threshold: Optional[float]
    ) -> float:
        """Calculate dynamic threshold based on market conditions"""
        
        if custom_threshold is not None:
            self.logger.info(f"Using custom threshold: {custom_threshold:.3f}")
            return custom_threshold
        
        threshold = self.base_threshold
        adjustments = []
        
        # Market volatility adjustment
        if market_data is not None:
            market_volatility = self._estimate_market_volatility(market_data)
            
            if market_volatility > self.high_volatility_threshold:
                # Lower threshold in high volatility (more selections)
                vol_adjustment = self.high_volatility_adjustment
                threshold += vol_adjustment
                adjustments.append(f"High volatility: {vol_adjustment:+.3f}")
                
            elif market_volatility < self.low_volatility_threshold:
                # Slight bonus in low volatility (higher threshold, fewer selections)
                vol_adjustment = self.low_volatility_bonus
                threshold += vol_adjustment
                adjustments.append(f"Low volatility: {vol_adjustment:+.3f}")
        
        # Prediction distribution adjustment
        pred_adjustment = self._calculate_prediction_distribution_adjustment(predictions)
        if abs(pred_adjustment) > 0.001:
            threshold += pred_adjustment
            adjustments.append(f"Prediction distribution: {pred_adjustment:+.3f}")
        
        # Market sentiment adjustment (if available)
        if market_data is not None:
            sentiment_adjustment = self._calculate_sentiment_adjustment(market_data)
            if abs(sentiment_adjustment) > 0.001:
                threshold += sentiment_adjustment
                adjustments.append(f"Market sentiment: {sentiment_adjustment:+.3f}")
        
        # Ensure threshold stays within reasonable bounds
        threshold = np.clip(threshold, 0.7, 0.95)
        
        if adjustments:
            self.logger.info(f"Threshold adjustments applied: {', '.join(adjustments)}")
        
        return threshold
    
    def _estimate_market_volatility(self, market_data: pd.DataFrame) -> float:
        """Estimate current market volatility from market data"""
        # Use VIX if available, otherwise estimate from market movements
        if 'VIX' in market_data.columns:
            latest_vix = market_data['VIX'].iloc[-1] if not market_data.empty else 20.0
            # Convert VIX to daily volatility estimate
            return latest_vix / 100 / np.sqrt(252)  # Annualized to daily
        
        # Fallback: use market index volatility
        if 'market_return' in market_data.columns:
            recent_returns = market_data['market_return'].tail(20)
            return recent_returns.std() if len(recent_returns) > 1 else 0.02
        
        # Default moderate volatility
        return 0.02
    
    def _calculate_prediction_distribution_adjustment(self, predictions: pd.DataFrame) -> float:
        """Adjust threshold based on prediction distribution"""
        probs = predictions['prediction_probability']
        
        # If very few high-confidence predictions, lower threshold slightly
        high_conf_count = (probs >= self.base_threshold).sum()
        total_count = len(probs)
        
        if total_count > 0:
            high_conf_ratio = high_conf_count / total_count
            
            if high_conf_ratio < 0.02:  # Less than 2% above threshold
                return -0.02  # Lower threshold slightly
            elif high_conf_ratio > 0.15:  # More than 15% above threshold  
                return +0.03  # Raise threshold to be more selective
        
        return 0.0
    
    def _calculate_sentiment_adjustment(self, market_data: pd.DataFrame) -> float:
        """Calculate sentiment-based threshold adjustment"""
        # This is a placeholder for more sophisticated sentiment analysis
        # Could incorporate indicators like USD/JPY trend, market breadth, etc.
        
        sentiment_adjustment = 0.0
        
        # Example: if USD/JPY is strengthening rapidly, might favor export stocks
        if 'USDJPY' in market_data.columns and len(market_data) > 5:
            recent_usdjpy_change = (market_data['USDJPY'].iloc[-1] / 
                                   market_data['USDJPY'].iloc[-5] - 1)
            
            if recent_usdjpy_change > 0.02:  # 2% strengthening in recent period
                sentiment_adjustment = -0.01  # Slightly lower threshold for export stocks
        
        return sentiment_adjustment * self.market_sentiment_weight
    
    def _apply_base_filters(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply basic filtering rules"""
        filtered = predictions.copy()
        initial_count = len(filtered)
        
        # Volume filter
        if 'current_volume' in filtered.columns:
            volume_mask = filtered['current_volume'] >= self.min_volume_threshold
            filtered = filtered[volume_mask]
            self.logger.info(f"Volume filter: {len(filtered)}/{initial_count} stocks remaining")
        
        # Price filter (exclude penny stocks)
        if self.exclude_penny_stocks and 'current_close' in filtered.columns:
            price_mask = filtered['current_close'] >= self.min_price
            filtered = filtered[price_mask]
            self.logger.info(f"Price filter: {len(filtered)}/{initial_count} stocks remaining")
        
        # Gap filter (exclude stocks with large overnight gaps)
        if 'current_open' in filtered.columns and 'current_close' in filtered.columns:
            # Estimate gap from open vs previous close (approximation)
            gap_mask = True  # Placeholder - would need previous day's close
            
            # For now, exclude stocks with very large intraday moves as proxy
            if 'current_high' in filtered.columns and 'current_low' in filtered.columns:
                intraday_range = (filtered['current_high'] - filtered['current_low']) / filtered['current_close']
                large_move_mask = intraday_range <= self.max_gap_threshold
                filtered = filtered[large_move_mask]
                self.logger.info(f"Large move filter: {len(filtered)}/{initial_count} stocks remaining")
        
        return filtered
    
    def _apply_threshold_filter(self, predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Apply probability threshold filter"""
        candidates = predictions[predictions['prediction_probability'] >= threshold].copy()
        
        self.logger.info(f"Threshold filter ({threshold:.3f}): {len(candidates)} candidates")
        
        return candidates
    
    def _apply_ranking_selection(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Apply final ranking and selection logic"""
        if len(candidates) == 0:
            self.logger.info("No candidates passed threshold - returning empty selection")
            return candidates
        
        # Sort by prediction probability (descending)
        ranked = candidates.sort_values('prediction_probability', ascending=False).copy()
        
        # Apply max selection limit
        if len(ranked) > self.max_selections:
            selected = ranked.head(self.max_selections).copy()
            self.logger.info(f"Selected top {self.max_selections} stocks from {len(ranked)} candidates")
        else:
            selected = ranked.copy()
            self.logger.info(f"Selected all {len(selected)} candidates (below max limit)")
        
        # Check minimum selection requirement
        if len(selected) < self.min_selections:
            self.logger.warning(f"Only {len(selected)} selections (minimum: {self.min_selections})")
            # Could implement logic to lower threshold and retry if needed
        
        return selected
    
    def _add_extraction_metadata(self, selected_stocks: pd.DataFrame, 
                                effective_threshold: float) -> pd.DataFrame:
        """Add extraction metadata to selected stocks"""
        if selected_stocks.empty:
            return selected_stocks
        
        enriched = selected_stocks.copy()
        
        # Add selection ranking
        enriched['selection_rank'] = range(1, len(enriched) + 1)
        
        # Add threshold information
        enriched['extraction_threshold'] = effective_threshold
        enriched['threshold_margin'] = enriched['prediction_probability'] - effective_threshold
        
        # Add selection confidence categories
        def categorize_confidence(prob):
            if prob >= 0.9:
                return 'very_high'
            elif prob >= 0.8:
                return 'high'  
            elif prob >= 0.7:
                return 'medium'
            else:
                return 'low'
        
        enriched['confidence_category'] = enriched['prediction_probability'].apply(categorize_confidence)
        
        # Add extraction timestamp
        enriched['extraction_timestamp'] = datetime.now()
        
        return enriched
    
    def get_extraction_statistics(self, original_predictions: pd.DataFrame,
                                 selected_stocks: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed extraction statistics"""
        stats = {
            'extraction_summary': self.extraction_info,
            'selection_statistics': {
                'total_analyzed': len(original_predictions),
                'total_selected': len(selected_stocks),
                'selection_rate': len(selected_stocks) / len(original_predictions) if len(original_predictions) > 0 else 0
            },
            'probability_distribution': {
                'mean_selected': selected_stocks['prediction_probability'].mean() if not selected_stocks.empty else 0,
                'min_selected': selected_stocks['prediction_probability'].min() if not selected_stocks.empty else 0,
                'max_selected': selected_stocks['prediction_probability'].max() if not selected_stocks.empty else 0,
                'std_selected': selected_stocks['prediction_probability'].std() if not selected_stocks.empty else 0
            }
        }
        
        # Confidence category breakdown
        if not selected_stocks.empty and 'confidence_category' in selected_stocks.columns:
            category_counts = selected_stocks['confidence_category'].value_counts().to_dict()
            stats['confidence_breakdown'] = category_counts
        
        # Threshold analysis
        prob_above_base = (original_predictions['prediction_probability'] >= self.base_threshold).sum()
        prob_above_effective = (original_predictions['prediction_probability'] >= 
                               self.extraction_info.get('effective_threshold', self.base_threshold)).sum()
        
        stats['threshold_analysis'] = {
            'stocks_above_base_threshold': prob_above_base,
            'stocks_above_effective_threshold': prob_above_effective,
            'threshold_selectivity_ratio': prob_above_effective / prob_above_base if prob_above_base > 0 else 0
        }
        
        return stats
    
    def export_selection_report(self, selected_stocks: pd.DataFrame,
                               stats: Dict[str, Any],
                               file_path: Optional[str] = None) -> str:
        """Export detailed selection report"""
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"stock_selection_report_{timestamp}.txt"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("DAILY STOCK SELECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            f.write("SELECTION SUMMARY\n")
            f.write("-" * 20 + "\n")
            extraction_info = stats['extraction_summary']
            f.write(f"Analysis Date: {extraction_info['extraction_date']}\n")
            f.write(f"Stocks Analyzed: {extraction_info['total_stocks_analyzed']}\n")
            f.write(f"Final Selections: {extraction_info['final_selections']}\n")
            f.write(f"Effective Threshold: {extraction_info['effective_threshold']:.3f}\n")
            f.write(f"Threshold Adjustment: {extraction_info['threshold_adjustment']:+.3f}\n\n")
            
            # Selected stocks
            if not selected_stocks.empty:
                f.write("SELECTED STOCKS\n")
                f.write("-" * 15 + "\n")
                
                for _, stock in selected_stocks.iterrows():
                    f.write(f"Rank {stock['selection_rank']}: {stock['Code']}\n")
                    f.write(f"  Probability: {stock['prediction_probability']:.3f}\n")
                    f.write(f"  Confidence: {stock.get('confidence_category', 'N/A')}\n")
                    if 'current_close' in stock:
                        f.write(f"  Price: ¥{stock['current_close']:.0f}\n")
                    if 'current_volume' in stock:
                        f.write(f"  Volume: {stock['current_volume']:,.0f}\n")
                    f.write("\n")
            else:
                f.write("NO STOCKS SELECTED\n")
                f.write("No stocks met the selection criteria.\n\n")
            
            # Statistics
            f.write("EXTRACTION STATISTICS\n")
            f.write("-" * 20 + "\n")
            sel_stats = stats['selection_statistics']
            f.write(f"Selection Rate: {sel_stats['selection_rate']:.2%}\n")
            
            prob_stats = stats['probability_distribution']
            if prob_stats['mean_selected'] > 0:
                f.write(f"Mean Probability: {prob_stats['mean_selected']:.3f}\n")
                f.write(f"Probability Range: {prob_stats['min_selected']:.3f} - {prob_stats['max_selected']:.3f}\n")
            
            if 'confidence_breakdown' in stats:
                f.write("\nConfidence Categories:\n")
                for category, count in stats['confidence_breakdown'].items():
                    f.write(f"  {category}: {count}\n")
        
        self.logger.info(f"Selection report exported to {file_path}")
        
        return file_path