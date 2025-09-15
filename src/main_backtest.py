"""
Main backtest execution script for comprehensive model evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Model imports
from models.ensemble_model import EnsembleModel
from features.feature_engineer import FeatureEngineer
from data.data_loader import DataLoader

# Evaluation imports
from evaluation.time_series_validator import TimeSeriesValidator
from evaluation.precision_evaluator import PrecisionEvaluator
from evaluation.market_analyzer import MarketAnalyzer
from evaluation.trading_simulator import TradingSimulator

# Utility imports
from utils.logger import get_logger
from utils.config import get_config


class MainBacktest:
    """Main backtest orchestrator for comprehensive evaluation"""
    
    def __init__(self, config_path: str = None):
        """Initialize main backtest system"""
        self.config = get_config()
        self.logger = get_logger("main_backtest")
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.ensemble_model = EnsembleModel()
        
        # Initialize evaluation components
        self.ts_validator = TimeSeriesValidator()
        self.precision_evaluator = PrecisionEvaluator(target_precision=0.75)
        self.market_analyzer = MarketAnalyzer()
        self.trading_simulator = TradingSimulator()
        
        # Paths
        self.results_dir = Path("results/backtest_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("MainBacktest initialized successfully")
    
    def run_comprehensive_backtest(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        initial_train_years: int = 2,
        validation_splits: int = 10,
        save_results: bool = True
    ) -> dict:
        """
        Run comprehensive backtest with time series validation
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest  
            initial_train_years: Years of initial training data
            validation_splits: Number of time series validation splits
            save_results: Whether to save detailed results
            
        Returns:
            Dictionary with comprehensive backtest results
        """
        self.logger.info(f"Starting comprehensive backtest: {start_date} to {end_date}")
        
        try:
            # Step 1: Load and prepare data
            self.logger.info("Step 1: Loading historical data...")
            data_df = self._load_historical_data(start_date, end_date)
            
            if data_df is None or len(data_df) == 0:
                raise ValueError("No data loaded")
            
            self.logger.info(f"Loaded {len(data_df):,} records with {len(data_df.columns)} features")
            
            # Step 2: Feature engineering
            self.logger.info("Step 2: Engineering features...")
            features_df = self._engineer_features(data_df)
            
            if features_df is None or len(features_df) == 0:
                raise ValueError("Feature engineering failed")
            
            self.logger.info(f"Generated {len(features_df.columns)} features")
            
            # Step 3: Prepare market data for regime analysis
            self.logger.info("Step 3: Preparing market environment data...")
            market_data = self._prepare_market_data(data_df)
            
            # Step 4: Time series validation
            self.logger.info("Step 4: Running time series validation...")
            validation_results = self._run_time_series_validation(
                features_df, initial_train_years, validation_splits
            )
            
            # Step 5: Market regime analysis
            self.logger.info("Step 5: Analyzing market regimes...")
            market_analysis = self._analyze_market_regimes(
                validation_results['predictions'], market_data
            )
            
            # Step 6: Trading simulation
            self.logger.info("Step 6: Running trading simulation...")
            trading_results = self._run_trading_simulation(
                validation_results['predictions'], data_df
            )
            
            # Step 7: Comprehensive evaluation
            self.logger.info("Step 7: Generating comprehensive evaluation...")
            evaluation_results = self._generate_comprehensive_evaluation(
                validation_results, market_analysis, trading_results
            )
            
            # Step 8: Save results
            if save_results:
                self.logger.info("Step 8: Saving results...")
                self._save_backtest_results(evaluation_results)
            
            # Log final summary
            self._log_backtest_summary(evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            raise
    
    def _load_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical stock data"""
        try:
            # Try to load from processed data first
            processed_data_path = Path("data/processed/nikkei225_processed.pkl")
            
            if processed_data_path.exists():
                self.logger.info("Loading from processed data file...")
                df = pd.read_pickle(processed_data_path)
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter by date range
                mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                df = df[mask].reset_index(drop=True)
                
                return df
            
            else:
                # Load raw data and process
                self.logger.info("Loading raw data files...")
                df = self.data_loader.load_nikkei225_data()
                
                if df is not None and len(df) > 0:
                    # Filter by date range
                    df['date'] = pd.to_datetime(df['date'])
                    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                    df = df[mask].reset_index(drop=True)
                    
                    return df
                else:
                    self.logger.warning("No data found in data loader")
                    return self._create_synthetic_data(start_date, end_date)
        
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            # Create synthetic data for demonstration
            return self._create_synthetic_data(start_date, end_date)
    
    def _create_synthetic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create synthetic data for demonstration purposes"""
        self.logger.warning("Creating synthetic data for demonstration")
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Nikkei 225 symbols (sample)
        symbols = [
            '7203', '9984', '6758', '9432', '8306', '8035', '6367', '7974', '9983', '4063',
            '6501', '7267', '6902', '8001', '2914', '4519', '4543', '6954', '6502', '8309'
        ]
        
        # Create synthetic data
        data = []
        base_price = 1000
        
        for date in dates:
            for symbol in symbols:
                # Random walk with drift
                price_change = np.random.normal(0.001, 0.02)  # Small positive drift, 2% volatility
                base_price *= (1 + price_change)
                
                # Generate basic data
                volume = np.random.randint(100000, 1000000)
                open_price = base_price * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, base_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, base_price) * (1 - abs(np.random.normal(0, 0.01)))
                
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'open_price': open_price,
                    'high_price': high_price,
                    'low_price': low_price,
                    'close_price': base_price,
                    'volume': volume,
                    'sector': np.random.choice(['Technology', 'Finance', 'Manufacturing', 'Retail']),
                    'market_cap': np.random.uniform(100e9, 10e12)  # 100B to 10T yen
                })
        
        df = pd.DataFrame(data)
        self.logger.info(f"Created synthetic dataset with {len(df):,} records")
        
        return df
    
    def _engineer_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for model training"""
        try:
            # Calculate next day return target
            data_df = data_df.sort_values(['symbol', 'date']).reset_index(drop=True)
            data_df['next_day_return'] = data_df.groupby('symbol')['close_price'].pct_change().shift(-1)
            data_df['target'] = (data_df['next_day_return'] >= 0.01).astype(int)
            
            # Basic features
            features_df = data_df.copy()
            
            # Technical indicators (simplified)
            for symbol in features_df['symbol'].unique():
                symbol_data = features_df[features_df['symbol'] == symbol].sort_values('date')
                
                if len(symbol_data) > 20:
                    # Moving averages
                    features_df.loc[features_df['symbol'] == symbol, 'sma_5'] = \
                        symbol_data['close_price'].rolling(5).mean()
                    features_df.loc[features_df['symbol'] == symbol, 'sma_20'] = \
                        symbol_data['close_price'].rolling(20).mean()
                    
                    # RSI (simplified)
                    price_changes = symbol_data['close_price'].pct_change()
                    gains = price_changes.where(price_changes > 0, 0)
                    losses = -price_changes.where(price_changes < 0, 0)
                    avg_gain = gains.rolling(14).mean()
                    avg_loss = losses.rolling(14).mean()
                    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                    features_df.loc[features_df['symbol'] == symbol, 'rsi'] = rsi
                    
                    # Volatility
                    features_df.loc[features_df['symbol'] == symbol, 'volatility_20'] = \
                        price_changes.rolling(20).std()
            
            # Market features
            market_data = features_df.groupby('date').agg({
                'close_price': ['mean', 'std'],
                'volume': 'sum',
                'next_day_return': 'mean'
            }).reset_index()
            
            market_data.columns = ['date', 'market_price_avg', 'market_price_std', 
                                 'total_volume', 'market_return']
            
            features_df = features_df.merge(market_data, on='date', how='left')
            
            # Remove rows with missing targets or insufficient data
            features_df = features_df.dropna(subset=['target', 'next_day_return'])
            features_df = features_df.fillna(0)  # Fill remaining NaN with 0
            
            self.logger.info(f"Feature engineering completed: {len(features_df)} samples, {len(features_df.columns)} features")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def _prepare_market_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare market-level data for regime analysis"""
        # Aggregate daily market data
        market_df = data_df.groupby('date').agg({
            'close_price': 'mean',
            'volume': 'sum'
        }).reset_index()
        
        market_df.columns = ['date', 'market_close', 'total_volume']
        market_df = market_df.sort_values('date').reset_index(drop=True)
        
        return market_df
    
    def _run_time_series_validation(
        self, 
        features_df: pd.DataFrame, 
        initial_train_years: int,
        validation_splits: int
    ) -> dict:
        """Run time series cross-validation"""
        # Prepare feature columns (exclude non-feature columns)
        exclude_cols = ['date', 'symbol', 'target', 'next_day_return', 'sector']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Prepare data for time series validation
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Add date as index for time series validation
        X_with_date = X.copy()
        X_with_date['date'] = features_df['date']
        
        # Run time series validation
        validation_results = self.ts_validator.validate_model(
            model=self.ensemble_model,
            X=X_with_date,
            y=y,
            date_col='date'
        )
        
        return validation_results
    
    def _analyze_market_regimes(self, predictions_df: pd.DataFrame, market_data: pd.DataFrame) -> dict:
        """Analyze performance by market regime"""
        # Classify market regimes
        market_regimes = self.market_analyzer.classify_market_regime(
            market_data, date_col='date', price_col='market_close'
        )
        
        # Analyze performance by regime
        market_analysis = self.market_analyzer.analyze_performance_by_regime(
            predictions_df, market_regimes,
            date_col='date', target_col='target', pred_proba_col='pred_proba'
        )
        
        return market_analysis
    
    def _run_trading_simulation(self, predictions_df: pd.DataFrame, price_data: pd.DataFrame) -> dict:
        """Run realistic trading simulation"""
        # Prepare price data
        price_df = price_data[['date', 'symbol', 'close_price', 'sector']].copy()
        
        # Run simulation
        simulation_results = self.trading_simulator.run_simulation(
            predictions_df=predictions_df,
            price_data_df=price_df,
            min_confidence=0.75
        )
        
        # Generate trading report
        trading_report = self.trading_simulator.generate_trading_report(simulation_results)
        
        return {
            'simulation_results': simulation_results,
            'trading_report': trading_report
        }
    
    def _generate_comprehensive_evaluation(
        self, 
        validation_results: dict, 
        market_analysis: dict,
        trading_results: dict
    ) -> dict:
        """Generate comprehensive evaluation combining all analyses"""
        
        # Extract precision metrics from validation
        predictions_df = validation_results['predictions']
        
        # Calculate precision metrics by period
        metrics_history = []
        unique_dates = sorted(predictions_df['date'].unique())
        
        # Group by validation periods (monthly)
        for i in range(0, len(unique_dates), 30):
            period_dates = unique_dates[i:i+30]
            period_data = predictions_df[predictions_df['date'].isin(period_dates)]
            
            if len(period_data) > 0:
                metrics = self.precision_evaluator.calculate_precision_metrics(
                    y_true=period_data['target'],
                    y_pred_proba=period_data['pred_proba'],
                    returns=period_data.get('next_day_return'),
                    date_index=pd.to_datetime(period_data['date'])
                )
                metrics_history.append(metrics)
        
        # Generate precision report
        precision_report = self.precision_evaluator.generate_precision_report(
            metrics_history=metrics_history,
            environment_metrics=market_analysis.get('regime_performance')
        )
        
        # Generate market report
        market_report = self.market_analyzer.generate_market_report(market_analysis)
        
        # Combine all results
        comprehensive_results = {
            'validation_results': validation_results,
            'precision_analysis': precision_report,
            'market_analysis': market_report,
            'trading_results': trading_results,
            'overall_summary': self._create_overall_summary(
                validation_results, precision_report, trading_results
            ),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return comprehensive_results
    
    def _create_overall_summary(
        self, 
        validation_results: dict, 
        precision_report: dict, 
        trading_results: dict
    ) -> dict:
        """Create overall summary of backtest performance"""
        
        # Extract key metrics
        avg_precision = precision_report.get('summary', {}).get('avg_precision', 0)
        target_achievement = precision_report.get('summary', {}).get('target_achievement_rate', 0)
        
        trading_metrics = trading_results.get('simulation_results', {}).get('metrics')
        trading_return = trading_metrics.total_return_pct if trading_metrics else 0
        win_rate = trading_metrics.win_rate if trading_metrics else 0
        
        # Overall assessment
        precision_target_met = avg_precision >= 0.75
        positive_returns = trading_return > 0
        good_win_rate = win_rate > 0.60
        
        overall_success = precision_target_met and positive_returns and good_win_rate
        
        summary = {
            'overall_success': overall_success,
            'key_metrics': {
                'avg_precision': f"{avg_precision:.3f}",
                'target_achievement_rate': f"{target_achievement:.1%}",
                'trading_return': f"{trading_return:.2%}",
                'win_rate': f"{win_rate:.1%}"
            },
            'success_criteria': {
                'precision_target_met': precision_target_met,
                'positive_returns': positive_returns,
                'good_win_rate': good_win_rate
            },
            'recommendations': self._generate_overall_recommendations(
                precision_target_met, positive_returns, good_win_rate, avg_precision
            )
        }
        
        return summary
    
    def _generate_overall_recommendations(
        self, 
        precision_ok: bool, 
        returns_ok: bool, 
        win_rate_ok: bool,
        avg_precision: float
    ) -> list:
        """Generate overall recommendations based on results"""
        recommendations = []
        
        if not precision_ok:
            if avg_precision < 0.50:
                recommendations.append("Precision is significantly below target. Consider major model architecture changes.")
            else:
                recommendations.append("Precision is close to target. Fine-tune hyperparameters and feature selection.")
        
        if not returns_ok:
            recommendations.append("Trading returns are negative. Review position sizing and risk management.")
        
        if not win_rate_ok:
            recommendations.append("Win rate is low. Consider adjusting prediction thresholds or holding periods.")
        
        if precision_ok and returns_ok and win_rate_ok:
            recommendations.append("System performance meets targets. Consider scaling up capital allocation.")
            recommendations.append("Monitor performance closely during different market regimes.")
        
        return recommendations
    
    def _save_backtest_results(self, results: dict):
        """Save comprehensive backtest results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results
        results_file = self.results_dir / f"backtest_results_{timestamp}.json"
        
        # Convert complex objects to serializable format
        serializable_results = self._make_serializable(results)
        
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Save predictions separately
        if 'validation_results' in results and 'predictions' in results['validation_results']:
            predictions_file = self.results_dir / f"predictions_{timestamp}.csv"
            results['validation_results']['predictions'].to_csv(predictions_file, index=False)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def _log_backtest_summary(self, results: dict):
        """Log comprehensive backtest summary"""
        self.logger.info("=" * 80)
        self.logger.info("COMPREHENSIVE BACKTEST RESULTS")
        self.logger.info("=" * 80)
        
        summary = results.get('overall_summary', {})
        
        # Overall success
        success = summary.get('overall_success', False)
        self.logger.info(f"Overall Success: {'✓ YES' if success else '✗ NO'}")
        
        # Key metrics
        metrics = summary.get('key_metrics', {})
        self.logger.info(f"Average Precision: {metrics.get('avg_precision', 'N/A')}")
        self.logger.info(f"Target Achievement: {metrics.get('target_achievement_rate', 'N/A')}")
        self.logger.info(f"Trading Return: {metrics.get('trading_return', 'N/A')}")
        self.logger.info(f"Win Rate: {metrics.get('win_rate', 'N/A')}")
        
        # Recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            self.logger.info("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                self.logger.info(f"  {i}. {rec}")
        
        self.logger.info("=" * 80)


def main():
    """Main execution function"""
    # Initialize backtest system
    backtest = MainBacktest()
    
    # Run comprehensive backtest
    results = backtest.run_comprehensive_backtest(
        start_date="2020-01-01",
        end_date="2024-12-31",
        initial_train_years=2,
        validation_splits=8,
        save_results=True
    )
    
    return results


if __name__ == "__main__":
    main()