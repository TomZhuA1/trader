# trader/stages/backtest.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from ..utils import (
    calculate_returns_metrics,
    calculate_drawdown,
    calculate_hit_rate,
    calculate_avg_win_loss,
    calculate_calmar_ratio
)

logger = logging.getLogger(__name__)

class Backtester:
    """Backtest the complete trading strategy."""
    
    def __init__(self, config: dict):
        self.config = config
        self.commission_bps = config['costs']['commission_bps']
        self.slippage_bps = config['costs']['slippage_bps']
        self.rebalance_freq = config['rebalance']['frequency']
        
    def run(
        self,
        price_data: Dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> Dict:
        """Run backtest on historical data."""
        # Initialize portfolio
        from .portfolio import Portfolio
        portfolio = Portfolio(self.config['portfolio'])
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        
        # Track performance
        portfolio_values = []
        trades = []
        
        for date in rebalance_dates:
            # Get signals for this date
            date_signals = signals[signals.get('date', pd.Timestamp.now()) <= date].tail(100)
            
            # Get current prices
            current_prices = {}
            for symbol, df in price_data.items():
                if date in df.index:
                    current_prices[symbol] = df.loc[date, 'close']
            
            # Rebalance portfolio
            portfolio_state = portfolio.rebalance(date_signals, current_prices)
            
            # Record portfolio value
            portfolio_values.append({
                'date': date,
                'value': portfolio_state['portfolio_value'],
                'cash': portfolio_state['cash'],
                'n_positions': len(portfolio_state['positions'])
            })
            
            # Record trades
            # (simplified - would track actual trades)
        
        # Calculate returns
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        
        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_df)
        
        return {
            'metrics': metrics,
            'portfolio_values': portfolio_df,
            'trades': trades
        }
    
    def _get_rebalance_dates(self, start: str, end: str) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency."""
        dates = pd.date_range(start=start, end=end, freq='B')  # Business days
        
        if self.rebalance_freq == 'daily':
            return dates.tolist()
        elif self.rebalance_freq == 'weekly':
            return dates[dates.weekday == 4].tolist()  # Fridays
        elif self.rebalance_freq == 'monthly':
            return dates[dates.is_month_end].tolist()
        else:
            return dates.tolist()
    
    def _calculate_metrics(self, portfolio_df: pd.DataFrame) -> Dict:
        """Calculate backtest metrics."""
        returns = portfolio_df['returns'].dropna()
        
        # Basic metrics
        metrics = calculate_returns_metrics(returns)
        
        # Drawdown metrics
        dd_series, max_dd, max_duration = calculate_drawdown(returns)
        metrics['max_drawdown'] = max_dd
        metrics['max_dd_duration'] = max_duration
        
        # Trade metrics
        metrics['hit_rate'] = calculate_hit_rate(returns)
        avg_win, avg_loss, win_loss = calculate_avg_win_loss(returns)
        metrics['avg_win'] = avg_win
        metrics['avg_loss'] = avg_loss
        metrics['win_loss_ratio'] = win_loss
        
        # Risk metrics
        metrics['calmar'] = calculate_calmar_ratio(returns)
        
        # Calculate turnover
        position_changes = portfolio_df['n_positions'].diff().abs().sum()
        metrics['turnover'] = position_changes / len(portfolio_df)
        
        # Apply costs
        total_trades = position_changes
        cost_drag = (self.commission_bps + self.slippage_bps) * total_trades / 10000
        metrics['cost_adjusted_return'] = metrics['total_return'] - cost_drag
        
        return metrics