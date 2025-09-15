"""
Comprehensive backtesting system for stock prediction models
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

from utils.logger import get_logger
from utils.config import get_config
from utils.calendar_utils import is_business_day, next_business_day


class Backtester:
    """Comprehensive backtesting system with transaction costs and risk metrics"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize backtester"""
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(f'backtest.{key}', value)
        
        self.logger = get_logger("backtester")
        
        # Transaction cost parameters
        self.transaction_cost_rate = self.config.get('backtest.transaction_cost_rate', 0.0005)  # 0.05%
        self.slippage_rate = self.config.get('backtest.slippage_rate', 0.001)  # 0.10%
        self.total_cost_rate = self.transaction_cost_rate + self.slippage_rate  # 0.15% total
        
        # Position parameters
        self.max_position_size = self.config.get('backtest.max_position_size', 1.0)  # Full portfolio per position
        self.max_positions = self.config.get('backtest.max_positions', 3)  # Max 3 positions per day
        self.holding_period = self.config.get('backtest.holding_period', 1)  # Days to hold
        
        # Risk parameters
        self.stop_loss = self.config.get('backtest.stop_loss', -0.05)  # -5% stop loss
        self.take_profit = self.config.get('backtest.take_profit', None)  # No take profit by default
        
        # Results storage
        self.results_dir = self.config.get_data_dir('backtest')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_walkforward_backtest(
        self,
        predictions: pd.DataFrame,
        price_data: pd.DataFrame,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest on prediction data
        
        Args:
            predictions: DataFrame with Date, Code, prediction_probability columns
            price_data: DataFrame with Date, Code, Open, High, Low, Close columns
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Comprehensive backtest results
        """
        self.logger.info("Starting walk-forward backtest",
                        predictions=len(predictions),
                        price_records=len(price_data))
        
        # Prepare data
        predictions = self._prepare_prediction_data(predictions, start_date, end_date)
        price_data = self._prepare_price_data(price_data, start_date, end_date)
        
        if predictions.empty or price_data.empty:
            raise ValueError("No data available for backtesting period")
        
        # Run simulation
        trades = self._simulate_trades(predictions, price_data)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(trades, price_data)
        
        # Generate detailed results
        results = {
            'trades': trades,
            'performance': performance,
            'parameters': self._get_backtest_parameters(),
            'period': {
                'start_date': predictions['Date'].min(),
                'end_date': predictions['Date'].max(),
                'trading_days': predictions['Date'].nunique()
            }
        }
        
        self.logger.info("Backtest completed",
                        total_trades=len(trades),
                        winning_trades=len(trades[trades['return'] > 0]) if not trades.empty else 0,
                        total_return=f"{performance.get('total_return', 0):.2%}")
        
        return results
    
    def _prepare_prediction_data(
        self,
        predictions: pd.DataFrame,
        start_date: Optional[Union[str, date]],
        end_date: Optional[Union[str, date]]
    ) -> pd.DataFrame:
        """Prepare and filter prediction data"""
        df = predictions.copy()
        
        # Ensure Date column is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Apply date filters
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df['Date'] >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df['Date'] <= end_date]
        
        # Sort by date and probability
        df = df.sort_values(['Date', 'prediction_probability'], ascending=[True, False])
        
        return df
    
    def _prepare_price_data(
        self,
        price_data: pd.DataFrame,
        start_date: Optional[Union[str, date]],
        end_date: Optional[Union[str, date]]
    ) -> pd.DataFrame:
        """Prepare and filter price data"""
        df = price_data.copy()
        
        # Ensure Date column is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Apply date filters with buffer for holding period
        if start_date:
            start_date = pd.to_datetime(start_date) - timedelta(days=5)  # Buffer
            df = df[df['Date'] >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date) + timedelta(days=10)  # Buffer
            df = df[df['Date'] <= end_date]
        
        # Sort by code and date
        df = df.sort_values(['Code', 'Date'])
        
        return df
    
    def _simulate_trades(
        self,
        predictions: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Simulate actual trading based on predictions"""
        trades = []
        
        # Group predictions by date
        for trade_date, day_predictions in predictions.groupby('Date'):
            
            # Select top predictions up to max_positions
            selected_predictions = day_predictions.head(self.max_positions)
            
            for _, pred_row in selected_predictions.iterrows():
                trade = self._execute_trade(pred_row, price_data, trade_date)
                if trade:
                    trades.append(trade)
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        self.logger.info(f"Simulated {len(trades)} trades")
        
        return trades_df
    
    def _execute_trade(
        self,
        prediction: pd.Series,
        price_data: pd.DataFrame,
        trade_date: pd.Timestamp
    ) -> Optional[Dict]:
        """Execute a single trade based on prediction"""
        
        code = prediction['Code']
        probability = prediction['prediction_probability']
        
        # Get entry price (next day open)
        entry_date = self._get_next_business_day(trade_date)
        entry_price_data = price_data[
            (price_data['Code'] == code) & 
            (price_data['Date'] == entry_date)
        ]
        
        if entry_price_data.empty:
            return None
        
        entry_price = entry_price_data.iloc[0]['Open']
        entry_price_adj = entry_price * (1 + self.total_cost_rate)  # Add transaction costs
        
        # Calculate position size
        position_size = self.max_position_size / self.max_positions
        
        # Get exit price (holding period or stop loss/take profit)
        exit_info = self._get_exit_price(
            code, entry_date, entry_price, price_data
        )
        
        if not exit_info:
            return None
        
        exit_date, exit_price, exit_reason = exit_info
        exit_price_adj = exit_price * (1 - self.total_cost_rate)  # Subtract transaction costs
        
        # Calculate returns
        gross_return = (exit_price - entry_price) / entry_price
        net_return = (exit_price_adj - entry_price_adj) / entry_price_adj
        
        trade = {
            'prediction_date': trade_date,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'code': code,
            'prediction_probability': probability,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_price_adj': entry_price_adj,
            'exit_price_adj': exit_price_adj,
            'position_size': position_size,
            'gross_return': gross_return,
            'net_return': net_return,
            'return': net_return,  # Use net return as primary
            'holding_days': (exit_date - entry_date).days,
            'exit_reason': exit_reason
        }
        
        return trade
    
    def _get_next_business_day(self, current_date: pd.Timestamp) -> pd.Timestamp:
        """Get next business day from given date"""
        next_date = current_date + timedelta(days=1)
        
        # Simple approximation - skip weekends
        while next_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_date += timedelta(days=1)
        
        return next_date
    
    def _get_exit_price(
        self,
        code: str,
        entry_date: pd.Timestamp,
        entry_price: float,
        price_data: pd.DataFrame
    ) -> Optional[Tuple[pd.Timestamp, float, str]]:
        """Determine exit price and date based on holding rules"""
        
        stock_data = price_data[price_data['Code'] == code].copy()
        stock_data = stock_data[stock_data['Date'] > entry_date].head(10)  # Max 10 days lookahead
        
        if stock_data.empty:
            return None
        
        for _, day_data in stock_data.iterrows():
            current_date = day_data['Date']
            high_price = day_data['High']
            low_price = day_data['Low']
            close_price = day_data['Close']
            
            # Check stop loss
            if self.stop_loss and low_price <= entry_price * (1 + self.stop_loss):
                exit_price = entry_price * (1 + self.stop_loss)
                return current_date, exit_price, 'stop_loss'
            
            # Check take profit
            if self.take_profit and high_price >= entry_price * (1 + self.take_profit):
                exit_price = entry_price * (1 + self.take_profit)
                return current_date, exit_price, 'take_profit'
            
            # Check holding period
            holding_days = (current_date - entry_date).days
            if holding_days >= self.holding_period:
                return current_date, close_price, 'holding_period'
        
        # If no exit condition met, use last available price
        if not stock_data.empty:
            last_row = stock_data.iloc[-1]
            return last_row['Date'], last_row['Close'], 'data_end'
        
        return None
    
    def _calculate_performance_metrics(
        self,
        trades: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if trades.empty:
            return self._get_empty_performance_metrics()
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = len(trades[trades['return'] > 0])
        losing_trades = len(trades[trades['return'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Return statistics
        total_return = trades['return'].sum()  # Assuming equal position sizes
        mean_return = trades['return'].mean()
        std_return = trades['return'].std()
        
        # Win/Loss statistics
        avg_win = trades[trades['return'] > 0]['return'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['return'] < 0]['return'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        max_return = trades['return'].max()
        max_loss = trades['return'].min()
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + trades['return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Profit factor
        gross_profit = trades[trades['return'] > 0]['return'].sum()
        gross_loss = abs(trades[trades['return'] < 0]['return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Average holding period
        avg_holding_days = trades['holding_days'].mean()
        
        # Exit reason analysis
        exit_reasons = trades['exit_reason'].value_counts().to_dict()
        
        performance = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'mean_return': mean_return,
            'std_return': std_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_return': max_return,
            'max_loss': max_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding_days,
            'exit_reasons': exit_reasons
        }
        
        # Risk-adjusted metrics
        performance['calmar_ratio'] = total_return / abs(max_drawdown) if max_drawdown < 0 else 0
        performance['expectancy'] = mean_return
        performance['risk_reward_ratio'] = abs(avg_win / avg_loss) if avg_loss < 0 else 0
        
        return performance
    
    def _get_empty_performance_metrics(self) -> Dict[str, Any]:
        """Return empty performance metrics for cases with no trades"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'mean_return': 0.0,
            'std_return': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_return': 0.0,
            'max_loss': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0,
            'avg_holding_days': 0.0,
            'exit_reasons': {},
            'calmar_ratio': 0.0,
            'expectancy': 0.0,
            'risk_reward_ratio': 0.0
        }
    
    def _get_backtest_parameters(self) -> Dict[str, Any]:
        """Get current backtest parameters"""
        return {
            'transaction_cost_rate': self.transaction_cost_rate,
            'slippage_rate': self.slippage_rate,
            'total_cost_rate': self.total_cost_rate,
            'max_position_size': self.max_position_size,
            'max_positions': self.max_positions,
            'holding_period': self.holding_period,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }
    
    def plot_backtest_results(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 12)
    ) -> Optional[str]:
        """Plot comprehensive backtest results"""
        
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Matplotlib not available for plotting")
            return None
        
        trades = results['trades']
        performance = results['performance']
        
        if trades.empty:
            self.logger.warning("No trades to plot")
            return None
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. Cumulative Returns
        cumulative_returns = (1 + trades['return']).cumprod()
        axes[0, 0].plot(trades['entry_date'], cumulative_returns, linewidth=2)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Return Distribution
        axes[0, 1].hist(trades['return'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(trades['return'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {trades["return"].mean():.3f}')
        axes[0, 1].set_title('Return Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Drawdown
        cumulative_returns = (1 + trades['return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        axes[1, 0].fill_between(trades['entry_date'], drawdown, 0, alpha=0.3, color='red')
        axes[1, 0].plot(trades['entry_date'], drawdown, color='red', linewidth=1)
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_ylabel('Drawdown')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Monthly Returns
        trades_monthly = trades.copy()
        trades_monthly['year_month'] = trades_monthly['entry_date'].dt.to_period('M')
        monthly_returns = trades_monthly.groupby('year_month')['return'].sum()
        
        colors = ['green' if x > 0 else 'red' for x in monthly_returns.values]
        axes[1, 1].bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Monthly Returns')
        axes[1, 1].set_ylabel('Monthly Return')
        axes[1, 1].set_xticks(range(0, len(monthly_returns), max(1, len(monthly_returns)//6)))
        axes[1, 1].set_xticklabels([str(monthly_returns.index[i]) for i in range(0, len(monthly_returns), max(1, len(monthly_returns)//6))], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Win/Loss Analysis
        win_loss_data = [performance['winning_trades'], performance['losing_trades']]
        labels = ['Winning Trades', 'Losing Trades']
        colors = ['green', 'red']
        
        axes[2, 0].pie(win_loss_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[2, 0].set_title(f'Win Rate: {performance["win_rate"]:.1%}')
        
        # 6. Key Metrics Summary
        axes[2, 1].axis('off')
        metrics_text = f"""
        Total Trades: {performance['total_trades']}
        Win Rate: {performance['win_rate']:.1%}
        Total Return: {performance['total_return']:.1%}
        Avg Return: {performance['mean_return']:.2%}
        Sharpe Ratio: {performance['sharpe_ratio']:.2f}
        Max Drawdown: {performance['max_drawdown']:.1%}
        Profit Factor: {performance['profit_factor']:.2f}
        Avg Holding: {performance['avg_holding_days']:.1f} days
        """
        axes[2, 1].text(0.1, 0.9, metrics_text, transform=axes[2, 1].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        axes[2, 1].set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.results_dir / f"backtest_results_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Backtest results plot saved to {save_path}")
        
        return str(save_path)
    
    def export_backtest_report(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Export detailed backtest report"""
        
        if save_path is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.results_dir / f"backtest_report_{timestamp}.txt"
        
        trades = results['trades']
        performance = results['performance']
        parameters = results['parameters']
        period = results['period']
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE BACKTEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Period information
            f.write("BACKTEST PERIOD\n")
            f.write("-" * 15 + "\n")
            f.write(f"Start Date: {period['start_date']}\n")
            f.write(f"End Date: {period['end_date']}\n")
            f.write(f"Trading Days: {period['trading_days']}\n\n")
            
            # Parameters
            f.write("BACKTEST PARAMETERS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Transaction Cost: {parameters['transaction_cost_rate']:.2%}\n")
            f.write(f"Slippage: {parameters['slippage_rate']:.2%}\n")
            f.write(f"Total Cost: {parameters['total_cost_rate']:.2%}\n")
            f.write(f"Max Positions: {parameters['max_positions']}\n")
            f.write(f"Holding Period: {parameters['holding_period']} days\n")
            f.write(f"Stop Loss: {parameters['stop_loss']:.1%}\n\n")
            
            # Performance Summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Trades: {performance['total_trades']}\n")
            f.write(f"Winning Trades: {performance['winning_trades']}\n")
            f.write(f"Losing Trades: {performance['losing_trades']}\n")
            f.write(f"Win Rate: {performance['win_rate']:.1%}\n\n")
            
            f.write(f"Total Return: {performance['total_return']:.2%}\n")
            f.write(f"Average Return: {performance['mean_return']:.3%}\n")
            f.write(f"Return Volatility: {performance['std_return']:.3%}\n")
            f.write(f"Average Win: {performance['avg_win']:.3%}\n")
            f.write(f"Average Loss: {performance['avg_loss']:.3%}\n\n")
            
            # Risk Metrics
            f.write("RISK METRICS\n")
            f.write("-" * 12 + "\n")
            f.write(f"Maximum Return: {performance['max_return']:.2%}\n")
            f.write(f"Maximum Loss: {performance['max_loss']:.2%}\n")
            f.write(f"Maximum Drawdown: {performance['max_drawdown']:.2%}\n")
            f.write(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}\n")
            f.write(f"Calmar Ratio: {performance['calmar_ratio']:.3f}\n")
            f.write(f"Profit Factor: {performance['profit_factor']:.3f}\n\n")
            
            # Trade Analysis
            f.write("TRADE ANALYSIS\n")
            f.write("-" * 14 + "\n")
            f.write(f"Average Holding Period: {performance['avg_holding_days']:.1f} days\n")
            f.write(f"Risk/Reward Ratio: {performance['risk_reward_ratio']:.2f}\n")
            f.write(f"Expectancy: {performance['expectancy']:.3%}\n\n")
            
            # Exit Reasons
            f.write("EXIT REASONS\n")
            f.write("-" * 12 + "\n")
            for reason, count in performance['exit_reasons'].items():
                f.write(f"{reason}: {count} ({count/performance['total_trades']:.1%})\n")
            f.write("\n")
            
            # Top/Bottom Trades
            if not trades.empty:
                f.write("TOP 5 TRADES\n")
                f.write("-" * 12 + "\n")
                top_trades = trades.nlargest(5, 'return')
                for _, trade in top_trades.iterrows():
                    f.write(f"{trade['code']} ({trade['entry_date'].date()}): {trade['return']:.2%}\n")
                f.write("\n")
                
                f.write("BOTTOM 5 TRADES\n")
                f.write("-" * 15 + "\n")
                bottom_trades = trades.nsmallest(5, 'return')
                for _, trade in bottom_trades.iterrows():
                    f.write(f"{trade['code']} ({trade['entry_date'].date()}): {trade['return']:.2%}\n")
        
        self.logger.info(f"Backtest report exported to {save_path}")
        
        return str(save_path)