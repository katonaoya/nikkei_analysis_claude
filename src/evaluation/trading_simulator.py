"""
Trading simulation system for backtesting stock prediction models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import warnings

from utils.logger import get_logger
from utils.config import get_config


@dataclass
class Trade:
    """Individual trade record"""
    date: str
    symbol: str
    action: str  # 'buy' or 'sell'
    quantity: int
    price: float
    total_value: float
    commission: float
    prediction_confidence: float
    sector: Optional[str] = None


@dataclass
class Position:
    """Current position information"""
    symbol: str
    quantity: int
    avg_buy_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    days_held: int
    sector: Optional[str] = None


@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    total_return: float
    total_return_pct: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_commission: float
    days_simulated: int


class TradingSimulator:
    """Advanced trading simulation with realistic constraints and costs"""
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        max_positions: int = 10,
        max_position_size: float = 0.10,
        commission_rate: float = 0.001,
        holding_period: int = 5,
        risk_free_rate: float = 0.01
    ):
        """
        Initialize trading simulator
        
        Args:
            initial_capital: Starting capital in yen
            max_positions: Maximum number of concurrent positions
            max_position_size: Maximum position size as fraction of portfolio
            commission_rate: Commission rate (0.1% = 0.001)
            holding_period: Default holding period in days
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.commission_rate = commission_rate
        self.holding_period = holding_period
        self.risk_free_rate = risk_free_rate
        
        self.config = get_config()
        self.logger = get_logger("trading_simulator")
        
        # Trading state
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        self.daily_returns: List[float] = []
    
    def run_simulation(
        self,
        predictions_df: pd.DataFrame,
        price_data_df: pd.DataFrame,
        date_col: str = 'date',
        symbol_col: str = 'symbol',
        pred_proba_col: str = 'pred_proba',
        target_col: str = 'target',
        price_col: str = 'close_price',
        sector_col: Optional[str] = 'sector',
        min_confidence: float = 0.75
    ) -> Dict[str, Any]:
        """
        Run complete trading simulation
        
        Args:
            predictions_df: DataFrame with predictions
            price_data_df: DataFrame with price data
            date_col: Date column name
            symbol_col: Symbol column name  
            pred_proba_col: Prediction probability column
            target_col: Target column (actual outcome)
            price_col: Price column name
            sector_col: Sector column name
            min_confidence: Minimum confidence threshold for trades
            
        Returns:
            Comprehensive simulation results
        """
        self.logger.info(f"Starting trading simulation with {self.initial_capital:,} yen initial capital")
        
        # Prepare data
        pred_df = predictions_df.copy()
        price_df = price_data_df.copy()
        
        pred_df[date_col] = pd.to_datetime(pred_df[date_col])
        price_df[date_col] = pd.to_datetime(price_df[date_col])
        
        # Merge predictions with price data
        merged_df = pd.merge(
            pred_df, 
            price_df[[date_col, symbol_col, price_col]], 
            on=[date_col, symbol_col], 
            how='inner'
        )
        
        # Sort by date
        merged_df = merged_df.sort_values([date_col, pred_proba_col], ascending=[True, False])
        
        # Reset trading state
        self._reset_simulation_state()
        
        # Simulate trading day by day
        unique_dates = sorted(merged_df[date_col].unique())
        
        for date in unique_dates:
            daily_data = merged_df[merged_df[date_col] == date]
            
            # Update positions with current prices
            self._update_positions(daily_data, date, symbol_col, price_col)
            
            # Process sell orders (close positions after holding period)
            self._process_sell_orders(daily_data, date, symbol_col, price_col, target_col, sector_col)
            
            # Process buy orders (new positions)
            self._process_buy_orders(
                daily_data, date, symbol_col, pred_proba_col, 
                price_col, min_confidence, sector_col
            )
            
            # Record portfolio state
            self._record_portfolio_state(date)
        
        # Calculate final metrics
        simulation_results = self._calculate_simulation_results(unique_dates)
        
        self.logger.info(f"Simulation completed: {simulation_results['metrics'].total_return_pct:.1%} total return, "
                        f"{simulation_results['metrics'].win_rate:.1%} win rate")
        
        return simulation_results
    
    def _reset_simulation_state(self):
        """Reset simulation state to initial conditions"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
    
    def _update_positions(
        self, 
        daily_data: pd.DataFrame, 
        date: pd.Timestamp, 
        symbol_col: str, 
        price_col: str
    ):
        """Update current positions with latest prices"""
        for symbol, position in self.positions.items():
            # Find current price
            symbol_data = daily_data[daily_data[symbol_col] == symbol]
            
            if len(symbol_data) > 0:
                current_price = symbol_data[price_col].iloc[0]
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = position.market_value - (position.quantity * position.avg_buy_price)
                position.days_held += 1
    
    def _process_sell_orders(
        self,
        daily_data: pd.DataFrame,
        date: pd.Timestamp,
        symbol_col: str,
        price_col: str,
        target_col: str,
        sector_col: Optional[str]
    ):
        """Process sell orders for positions that have reached holding period"""
        symbols_to_sell = []
        
        for symbol, position in self.positions.items():
            # Sell if holding period reached
            if position.days_held >= self.holding_period:
                symbols_to_sell.append(symbol)
        
        for symbol in symbols_to_sell:
            position = self.positions[symbol]
            symbol_data = daily_data[daily_data[symbol_col] == symbol]
            
            if len(symbol_data) > 0:
                sell_price = symbol_data[price_col].iloc[0]
                
                # Calculate trade details
                total_value = position.quantity * sell_price
                commission = total_value * self.commission_rate
                net_proceeds = total_value - commission
                
                # Record trade
                trade = Trade(
                    date=date.strftime('%Y-%m-%d'),
                    symbol=symbol,
                    action='sell',
                    quantity=position.quantity,
                    price=sell_price,
                    total_value=total_value,
                    commission=commission,
                    prediction_confidence=0.0,  # Not applicable for sells
                    sector=getattr(position, 'sector', None)
                )
                self.trades.append(trade)
                
                # Update capital
                self.current_capital += net_proceeds
                
                # Remove position
                del self.positions[symbol]
                
                self.logger.debug(f"Sold {symbol}: {position.quantity} shares at {sell_price:.2f}")
    
    def _process_buy_orders(
        self,
        daily_data: pd.DataFrame,
        date: pd.Timestamp,
        symbol_col: str,
        pred_proba_col: str,
        price_col: str,
        min_confidence: float,
        sector_col: Optional[str]
    ):
        """Process buy orders for new positions"""
        # Filter high-confidence predictions
        buy_candidates = daily_data[daily_data[pred_proba_col] >= min_confidence]
        
        if len(buy_candidates) == 0:
            return
        
        # Sort by prediction confidence
        buy_candidates = buy_candidates.sort_values(pred_proba_col, ascending=False)
        
        # Calculate available slots and capital
        available_slots = self.max_positions - len(self.positions)
        if available_slots <= 0:
            return
        
        total_portfolio_value = self._calculate_total_portfolio_value()
        available_capital = min(self.current_capital * 0.95, total_portfolio_value * 0.90)
        
        # Process buy orders
        for idx, row in buy_candidates.head(available_slots).iterrows():
            symbol = row[symbol_col]
            price = row[price_col]
            confidence = row[pred_proba_col]
            
            # Skip if already holding this symbol
            if symbol in self.positions:
                continue
            
            # Calculate position size
            max_position_value = total_portfolio_value * self.max_position_size
            affordable_shares = int(available_capital / price)
            max_shares = int(max_position_value / price)
            
            quantity = min(affordable_shares, max_shares)
            
            if quantity < 100:  # Minimum lot size
                continue
            
            # Calculate trade details
            total_value = quantity * price
            commission = total_value * self.commission_rate
            total_cost = total_value + commission
            
            if total_cost > available_capital:
                continue
            
            # Execute trade
            sector = row.get(sector_col, None) if sector_col else None
            
            trade = Trade(
                date=date.strftime('%Y-%m-%d'),
                symbol=symbol,
                action='buy',
                quantity=quantity,
                price=price,
                total_value=total_value,
                commission=commission,
                prediction_confidence=confidence,
                sector=sector
            )
            self.trades.append(trade)
            
            # Create position
            position = Position(
                symbol=symbol,
                quantity=quantity,
                avg_buy_price=price,
                current_price=price,
                market_value=total_value,
                unrealized_pnl=0.0,
                days_held=0,
                sector=sector
            )
            self.positions[symbol] = position
            
            # Update capital
            self.current_capital -= total_cost
            available_capital -= total_cost
            
            self.logger.debug(f"Bought {symbol}: {quantity} shares at {price:.2f} (confidence: {confidence:.3f})")
            
            if available_capital < 100000:  # Stop if remaining capital too low
                break
    
    def _calculate_total_portfolio_value(self) -> float:
        """Calculate total portfolio value including cash and positions"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.current_capital + positions_value
    
    def _record_portfolio_state(self, date: pd.Timestamp):
        """Record current portfolio state"""
        total_value = self._calculate_total_portfolio_value()
        positions_value = sum(pos.market_value for pos in self.positions.values())
        
        state = {
            'date': date.strftime('%Y-%m-%d'),
            'total_value': total_value,
            'cash': self.current_capital,
            'positions_value': positions_value,
            'positions_count': len(self.positions),
            'daily_return': (total_value / self.initial_capital - 1) if self.initial_capital > 0 else 0
        }
        
        # Calculate daily return
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1]['total_value']
            daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0
            self.daily_returns.append(daily_return)
        
        self.portfolio_history.append(state)
    
    def _calculate_simulation_results(self, trading_dates: List) -> Dict[str, Any]:
        """Calculate comprehensive simulation results"""
        if not self.portfolio_history:
            return {'error': 'No portfolio history available'}
        
        # Basic return calculations
        final_value = self.portfolio_history[-1]['total_value']
        total_return = final_value - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # Annualized return
        days_simulated = len(self.portfolio_history)
        years = days_simulated / 252
        annualized_return = ((final_value / self.initial_capital) ** (1/years) - 1) if years > 0 else 0
        
        # Maximum drawdown
        portfolio_values = [state['total_value'] for state in self.portfolio_history]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Sharpe ratio
        if len(self.daily_returns) > 1:
            excess_returns = np.array(self.daily_returns) - (self.risk_free_rate / 252)
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Trade analysis
        winning_trades = 0
        losing_trades = 0
        total_commission = 0
        wins = []
        losses = []
        
        # Group trades by symbol to calculate P&L
        trade_groups = {}
        for trade in self.trades:
            if trade.symbol not in trade_groups:
                trade_groups[trade.symbol] = []
            trade_groups[trade.symbol].append(trade)
            total_commission += trade.commission
        
        for symbol, trades in trade_groups.items():
            buy_trades = [t for t in trades if t.action == 'buy']
            sell_trades = [t for t in trades if t.action == 'sell']
            
            for sell_trade in sell_trades:
                # Match with most recent buy
                if buy_trades:
                    buy_trade = buy_trades[-1]  # LIFO matching
                    pnl = (sell_trade.price - buy_trade.price) * sell_trade.quantity - sell_trade.commission - buy_trade.commission
                    
                    if pnl > 0:
                        winning_trades += 1
                        wins.append(pnl)
                    else:
                        losing_trades += 1
                        losses.append(abs(pnl))
        
        # Trading metrics
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else float('inf')
        
        metrics = TradingMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_commission=total_commission,
            days_simulated=days_simulated
        )
        
        return {
            'metrics': metrics,
            'portfolio_history': self.portfolio_history,
            'trades': [asdict(trade) for trade in self.trades],
            'final_positions': {k: asdict(v) for k, v in self.positions.items()},
            'summary': {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return_yen': total_return,
                'total_return_pct': f"{total_return_pct:.1%}",
                'max_drawdown_pct': f"{max_drawdown:.1%}",
                'sharpe_ratio': f"{sharpe_ratio:.2f}",
                'win_rate': f"{win_rate:.1%}",
                'total_trades': total_trades
            }
        }
    
    def generate_trading_report(
        self,
        simulation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trading report
        
        Args:
            simulation_results: Results from run_simulation
            save_path: Path to save report
            
        Returns:
            Formatted trading report
        """
        if 'error' in simulation_results:
            return simulation_results
        
        metrics = simulation_results['metrics']
        
        report = {
            'performance_summary': {
                'total_return': f"{metrics.total_return:,.0f} yen",
                'total_return_percentage': f"{metrics.total_return_pct:.2%}",
                'annualized_return': f"{metrics.annualized_return:.2%}",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'max_drawdown': f"{metrics.max_drawdown:.2%}",
                'days_simulated': metrics.days_simulated
            },
            'trading_statistics': {
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'win_rate': f"{metrics.win_rate:.1%}",
                'avg_win': f"{metrics.avg_win:,.0f} yen",
                'avg_loss': f"{metrics.avg_loss:,.0f} yen",
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'total_commission': f"{metrics.total_commission:,.0f} yen"
            },
            'risk_metrics': {
                'max_drawdown': f"{metrics.max_drawdown:.2%}",
                'sharpe_ratio': metrics.sharpe_ratio,
                'volatility': np.std(simulation_results.get('daily_returns', [0])) * np.sqrt(252) if 'daily_returns' in simulation_results else 0
            }
        }
        
        # Add trade analysis
        trades = simulation_results.get('trades', [])
        if trades:
            # Sector analysis
            sector_trades = {}
            for trade in trades:
                sector = trade.get('sector', 'Unknown')
                if sector not in sector_trades:
                    sector_trades[sector] = {'buy': 0, 'sell': 0}
                sector_trades[sector][trade['action']] += 1
            
            report['sector_analysis'] = sector_trades
            
            # Monthly breakdown
            monthly_trades = {}
            for trade in trades:
                month = trade['date'][:7]  # YYYY-MM
                if month not in monthly_trades:
                    monthly_trades[month] = 0
                monthly_trades[month] += 1
            
            report['monthly_activity'] = monthly_trades
        
        # Performance evaluation
        report['evaluation'] = {
            'meets_return_target': metrics.total_return_pct > 0.15,  # >15% annual return
            'acceptable_drawdown': metrics.max_drawdown > -0.20,    # <20% drawdown
            'good_sharpe_ratio': metrics.sharpe_ratio > 1.0,
            'sufficient_win_rate': metrics.win_rate > 0.60,         # >60% win rate
        }
        
        # Save report
        if save_path:
            self._save_trading_report(report, save_path)
        
        self.logger.info("Trading Report Generated:")
        self.logger.info(f"  Total Return: {report['performance_summary']['total_return_percentage']}")
        self.logger.info(f"  Win Rate: {report['trading_statistics']['win_rate']}")
        self.logger.info(f"  Max Drawdown: {report['performance_summary']['max_drawdown']}")
        
        return report
    
    def _save_trading_report(self, report: Dict[str, Any], file_path: str):
        """Save trading report to JSON file"""
        import json
        from pathlib import Path
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            return obj
        
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
        
        self.logger.info(f"Trading report saved to {file_path}")