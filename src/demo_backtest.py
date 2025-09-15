"""
Simplified demo backtest for system validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Evaluation imports
from evaluation.precision_evaluator import PrecisionEvaluator
from evaluation.market_analyzer import MarketAnalyzer
from evaluation.trading_simulator import TradingSimulator

# Utility imports
from utils.logger import get_logger
from utils.config import get_config


class DemoBacktest:
    """Simplified backtest for demonstration"""
    
    def __init__(self):
        """Initialize demo backtest"""
        self.config = get_config()
        self.logger = get_logger("demo_backtest")
        
        # Initialize evaluation components
        self.precision_evaluator = PrecisionEvaluator(target_precision=0.75)
        self.market_analyzer = MarketAnalyzer()
        self.trading_simulator = TradingSimulator()
        
        # Results directory
        self.results_dir = Path("results/demo_backtest")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Demo backtest initialized")
    
    def generate_demo_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate realistic demo data for testing"""
        
        # Generate date range (business days only)
        start_date = datetime(2020, 1, 1)
        dates = pd.date_range(start=start_date, periods=n_samples//20, freq='B')
        
        # Nikkei 225 symbols (sample)
        symbols = [
            '7203', '9984', '6758', '9432', '8306', '8035', '6367', '7974', '9983', '4063',
            '6501', '7267', '6902', '8001', '2914', '4519', '4543', '6954', '6502', '8309'
        ]
        
        sectors = ['Technology', 'Finance', 'Manufacturing', 'Retail', 'Healthcare']
        
        data = []
        base_prices = {symbol: np.random.uniform(1000, 5000) for symbol in symbols}
        
        for date in dates:
            market_return = np.random.normal(0.0005, 0.015)  # Market trend
            
            for symbol in symbols:
                # Individual stock factors
                stock_alpha = np.random.normal(0, 0.01)
                sector_effect = np.random.normal(0, 0.005)
                
                # Price simulation
                base_prices[symbol] *= (1 + market_return + stock_alpha + sector_effect)
                
                # Calculate next day return (target variable)
                next_return = np.random.normal(market_return + stock_alpha, 0.02)
                target = 1 if next_return >= 0.01 else 0  # +1% threshold
                
                # Generate features (simplified but realistic)
                features = {
                    'date': date,
                    'symbol': symbol,
                    'sector': np.random.choice(sectors),
                    'close_price': base_prices[symbol],
                    'volume': np.random.randint(100000, 10000000),
                    
                    # Technical features
                    'rsi': np.clip(np.random.normal(50, 15), 0, 100),
                    'macd': np.random.normal(0, 0.02),
                    'bollinger_position': np.random.uniform(0, 1),
                    'volume_ratio': np.random.lognormal(0, 0.3),
                    
                    # Market features  
                    'market_beta': np.clip(np.random.normal(1.0, 0.3), 0.3, 2.0),
                    'relative_strength': np.random.normal(0, 0.1),
                    'sector_momentum': np.random.normal(0, 0.05),
                    
                    # Fundamental features
                    'pe_ratio': np.random.lognormal(3, 0.5),
                    'market_cap_rank': np.random.uniform(0, 1),
                    
                    # Target and return
                    'next_day_return': next_return,
                    'target': target
                }
                
                data.append(features)
        
        df = pd.DataFrame(data)
        
        # Generate prediction probabilities (correlated with target)
        # High precision model should have higher probabilities for actual targets
        base_prob = 0.3  # Base probability
        signal_strength = 0.4  # How much target influences probability
        noise = 0.2  # Random noise
        
        df['pred_proba'] = (
            base_prob +
            signal_strength * df['target'] +
            noise * np.random.uniform(-1, 1, len(df)) +
            0.1 * df['rsi'] / 100 +  # RSI influence
            0.1 * df['relative_strength'] +  # Relative strength influence
            0.05 * np.random.randn(len(df))  # Additional noise
        )
        
        # Ensure probabilities are in valid range
        df['pred_proba'] = np.clip(df['pred_proba'], 0.0, 1.0)
        
        self.logger.info(f"Generated demo data: {len(df)} samples")
        self.logger.info(f"Target distribution: {df['target'].mean():.1%} positive")
        self.logger.info(f"High confidence predictions (>0.75): {(df['pred_proba'] > 0.75).mean():.1%}")
        
        return df
    
    def run_demo_validation(self, demo_data: pd.DataFrame) -> dict:
        """Run comprehensive validation on demo data"""
        
        # Step 1: Precision Analysis
        self.logger.info("Step 1: Running precision analysis...")
        
        # Split data into time-based chunks for analysis
        demo_data = demo_data.sort_values('date')
        unique_dates = sorted(demo_data['date'].unique())
        
        # Analyze by month
        metrics_history = []
        monthly_chunks = len(unique_dates) // 30 if len(unique_dates) > 30 else 1
        
        for i in range(0, len(unique_dates), max(30, len(unique_dates)//monthly_chunks)):
            period_dates = unique_dates[i:i+30]
            period_data = demo_data[demo_data['date'].isin(period_dates)]
            
            if len(period_data) > 0:
                metrics = self.precision_evaluator.calculate_precision_metrics(
                    y_true=period_data['target'],
                    y_pred_proba=period_data['pred_proba'],
                    returns=period_data['next_day_return'],
                    date_index=pd.to_datetime(period_data['date'])
                )
                metrics_history.append(metrics)
        
        # Generate precision report
        precision_report = self.precision_evaluator.generate_precision_report(metrics_history)
        
        # Step 2: Market Environment Analysis
        self.logger.info("Step 2: Running market environment analysis...")
        
        # Prepare market data
        market_data = demo_data.groupby('date').agg({
            'close_price': 'mean',
            'volume': 'sum'
        }).reset_index()
        market_data.columns = ['date', 'market_close', 'total_volume']
        
        # Classify market regimes
        market_regimes = self.market_analyzer.classify_market_regime(
            market_data, date_col='date', price_col='market_close'
        )
        
        # Analyze performance by regime
        market_analysis = self.market_analyzer.analyze_performance_by_regime(
            demo_data, market_regimes,
            date_col='date', target_col='target', pred_proba_col='pred_proba',
            return_col='next_day_return', sector_col='sector'
        )
        
        # Generate market report
        market_report = self.market_analyzer.generate_market_report(market_analysis)
        
        # Step 3: Trading Simulation
        self.logger.info("Step 3: Running trading simulation...")
        
        # Prepare data for trading simulation
        price_data = demo_data[['date', 'symbol', 'close_price', 'sector']].copy()
        predictions_data = demo_data[['date', 'symbol', 'pred_proba', 'target', 'next_day_return']].copy()
        
        # Run simulation
        simulation_results = self.trading_simulator.run_simulation(
            predictions_df=predictions_data,
            price_data_df=price_data,
            min_confidence=0.75
        )
        
        # Generate trading report
        trading_report = self.trading_simulator.generate_trading_report(simulation_results)
        
        # Step 4: Comprehensive Evaluation
        self.logger.info("Step 4: Generating comprehensive evaluation...")
        
        results = {
            'demo_data_summary': {
                'total_samples': len(demo_data),
                'unique_dates': len(unique_dates),
                'unique_symbols': demo_data['symbol'].nunique(),
                'target_rate': demo_data['target'].mean(),
                'high_confidence_rate': (demo_data['pred_proba'] > 0.75).mean()
            },
            'precision_analysis': precision_report,
            'market_analysis': market_report,
            'trading_simulation': {
                'simulation_results': simulation_results,
                'trading_report': trading_report
            },
            'overall_assessment': self._assess_performance(precision_report, trading_report),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return results
    
    def _assess_performance(self, precision_report: dict, trading_report: dict) -> dict:
        """Assess overall system performance"""
        
        # Extract key metrics
        avg_precision = precision_report.get('summary', {}).get('avg_precision', 0)
        target_achievement = precision_report.get('summary', {}).get('target_achievement_rate', 0)
        
        trading_metrics = trading_report.get('performance_summary', {})
        trading_return = trading_metrics.get('total_return_percentage', '0%')
        trading_return_num = float(trading_return.replace('%', '')) / 100 if '%' in str(trading_return) else 0
        
        win_rate_str = trading_report.get('trading_statistics', {}).get('win_rate', '0%')
        win_rate = float(win_rate_str.replace('%', '')) / 100 if '%' in str(win_rate_str) else 0
        
        # Performance criteria
        precision_target_met = avg_precision >= 0.75
        positive_returns = trading_return_num > 0
        good_win_rate = win_rate > 0.60
        
        overall_success = precision_target_met and positive_returns and good_win_rate
        
        assessment = {
            'overall_success': overall_success,
            'key_metrics': {
                'avg_precision': f"{avg_precision:.3f}",
                'target_achievement_rate': f"{target_achievement:.1%}",
                'trading_return': f"{trading_return_num:.2%}",
                'win_rate': f"{win_rate:.1%}"
            },
            'success_criteria': {
                'precision_target_met': precision_target_met,
                'positive_returns': positive_returns,
                'good_win_rate': good_win_rate
            },
            'recommendations': self._generate_recommendations(precision_target_met, positive_returns, good_win_rate)
        }
        
        return assessment
    
    def _generate_recommendations(self, precision_ok: bool, returns_ok: bool, win_rate_ok: bool) -> list:
        """Generate recommendations based on performance"""
        recommendations = []
        
        if precision_ok and returns_ok and win_rate_ok:
            recommendations.extend([
                "✅ All performance targets achieved! System ready for real data validation.",
                "Consider running with actual historical data for final validation.",
                "Monitor precision stability across different market conditions."
            ])
        else:
            if not precision_ok:
                recommendations.append("⚠️ Precision below target (0.75). Review model calibration and feature selection.")
            
            if not returns_ok:
                recommendations.append("⚠️ Trading returns negative. Review position sizing and selection criteria.")
            
            if not win_rate_ok:
                recommendations.append("⚠️ Win rate low. Consider adjusting prediction thresholds.")
        
        return recommendations
    
    def save_results(self, results: dict, filename: str = None) -> str:
        """Save results to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"demo_backtest_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
        return str(filepath)
    
    def print_summary(self, results: dict):
        """Print comprehensive summary"""
        self.logger.info("=" * 80)
        self.logger.info("DEMO BACKTEST RESULTS")
        self.logger.info("=" * 80)
        
        assessment = results.get('overall_assessment', {})
        
        # Overall success
        success = assessment.get('overall_success', False)
        self.logger.info(f"Overall Success: {'✅ YES' if success else '❌ NO'}")
        
        # Key metrics
        metrics = assessment.get('key_metrics', {})
        self.logger.info(f"Average Precision: {metrics.get('avg_precision', 'N/A')}")
        self.logger.info(f"Target Achievement: {metrics.get('target_achievement_rate', 'N/A')}")
        self.logger.info(f"Trading Return: {metrics.get('trading_return', 'N/A')}")
        self.logger.info(f"Win Rate: {metrics.get('win_rate', 'N/A')}")
        
        # Recommendations
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            self.logger.info("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                self.logger.info(f"  {i}. {rec}")
        
        self.logger.info("=" * 80)


def main():
    """Run demo backtest"""
    demo = DemoBacktest()
    
    # Generate demo data
    demo_data = demo.generate_demo_data(n_samples=10000)
    
    # Run validation
    results = demo.run_demo_validation(demo_data)
    
    # Save results
    demo.save_results(results)
    
    # Print summary
    demo.print_summary(results)
    
    return results


if __name__ == "__main__":
    main()